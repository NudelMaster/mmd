"""
ControlNet adapter for MMD TemporalUnet.

Design choices (see TEMPORAL_UNET_ARCHITECTURE.md for full rationale):
- Approach B (Global Conditioning): SDF information enters via FiLM, not cross-attention
- Design 2 (FiLM via time embedding): cond_emb = t_emb + sdf_emb
- No attention blocks (faithful copy of base model, which has self_attention=False)
- 3 down-block residuals (matching decoder level count; h[0] is never consumed)
- Scale parameter removed (SDF already encodes scaled geometry)

Architecture:
    SDF [B,1,64,64] -> MapSDFEncoder -> sdf_emb [B,32]
                                              |
                              cond_emb = t_emb + sdf_emb  [B,32]
                                              |
                              ControlNet encoder (4 levels, no attention)
                                              |
                              3 ZeroConv residuals + 1 mid residual
                                              |
                              Inject into frozen base model decoder

Target configuration (EnvConveyor2D):
    conditioning_type = None
    cond_dim = time_emb_dim = 32
    dim_mults = (1, 2, 4, 8)
    dims = [4, 32, 64, 128, 256]
    4 encoder levels, 3 decoder levels
    self_attention = False
"""
import copy
import torch
import torch.nn as nn
import einops

from mmd.models.layers.layers import (
    ResidualTemporalBlock, TimeEncoder, Downsample1d, group_norm_n_groups
)
from mmd.models.helpers.map_encoder import MapSDFEncoder


class ZeroConv1d(nn.Module):
    """
    Zero-initialized 1D convolution.

    Ensures the ControlNet has no effect at training start — residuals are
    initially zero, so the model begins from the base model's behavior.
    This is a key component of the original ControlNet design.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class MMDControlNet(nn.Module):
    """
    ControlNet adapter for MMD's TemporalUnet.

    Creates a trainable copy of the TemporalUnet's encoder + mid blocks,
    conditioned on SDF grids via FiLM (Design 2, Approach B).

    The ControlNet:
    1. Encodes SDF grid -> global vector [B, 32] via MapSDFEncoder
    2. Adds SDF embedding to time embedding: cond_emb = t_emb + sdf_emb
    3. Runs trajectory through trainable encoder copy with cond_emb
    4. Produces residuals via zero-initialized convolutions
    5. Residuals are injected into the frozen base model's decoder

    Only 3 down-block residuals are produced (for encoder levels 1, 2, 3),
    matching the 3 decoder levels. Encoder level 0's skip (h[0]) is never
    consumed by the decoder (asymmetric UNet design), so no residual is needed.

    Args:
        base_model: A TemporalUnet instance to copy architecture from
        cond_dim: Conditioning dimension (must match base model's time_emb_dim)
        sdf_encoder_hidden_dim: Hidden dimension for MapSDFEncoder CNN
    """

    def __init__(
        self,
        base_model,
        cond_dim=32,
        sdf_encoder_hidden_dim=256,
    ):
        super().__init__()

        # Extract architecture info from base model
        self.state_dim = base_model.state_dim
        self.cond_dim = cond_dim
        self.dims = self._extract_dims(base_model)
        # dims = [4, 32, 64, 128, 256] for dim_mults=(1,2,4,8)

        # Number of encoder levels and decoder levels
        self.n_encoder_levels = len(base_model.downs)
        self.n_decoder_levels = len(base_model.ups)
        # For dim_mults=(1,2,4,8): 4 encoder, 3 decoder

        # --- SDF Encoder (trainable, Design 2) ---
        self.map_encoder = MapSDFEncoder(
            cond_dim=cond_dim,
            hidden_dim=sdf_encoder_hidden_dim,
        )

        # --- Time encoder (trainable copy, same architecture as base) ---
        self.control_time_mlp = TimeEncoder(32, cond_dim)

        # --- Control encoder blocks (trainable copies) ---
        self.control_downs = self._create_control_encoder(base_model)

        # --- Control mid blocks (trainable copies) ---
        mid_dim = self.dims[-1]
        self.control_mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, cond_dim, n_support_points=0
        )
        self.control_mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, cond_dim, n_support_points=0
        )

        # --- Zero convolutions ---
        # 3 down-block residuals (for encoder levels 1, 2, 3 — skipping level 0)
        self.zero_convs_down = nn.ModuleList([
            ZeroConv1d(self.dims[i + 1])
            for i in range(1, self.n_encoder_levels)
            # i=1 -> dims[2]=64, i=2 -> dims[3]=128, i=3 -> dims[4]=256
        ])

        # 1 mid-block residual
        self.zero_conv_mid = ZeroConv1d(mid_dim)

        # Log architecture
        print(f"[MMDControlNet] Encoder levels: {self.n_encoder_levels}, "
              f"Decoder levels: {self.n_decoder_levels}")
        print(f"[MMDControlNet] Channel dims: {self.dims}")
        print(f"[MMDControlNet] Down residuals: {self.n_decoder_levels} "
              f"(levels {list(range(1, self.n_encoder_levels))})")
        print(f"[MMDControlNet] SDF encoder: CNN -> pool -> [{cond_dim}] "
              f"(hidden={sdf_encoder_hidden_dim})")

    def _extract_dims(self, base_model):
        """Extract channel dimensions from base model's encoder blocks."""
        dims = [self.state_dim]
        for down_block in base_model.downs:
            resnet1 = down_block[0]
            out_channels = resnet1.blocks[1].block[0].out_channels
            dims.append(out_channels)
        return dims

    def _create_control_encoder(self, base_model):
        """
        Create trainable encoder blocks mirroring the base model.

        No self-attention or cross-attention blocks — faithful copy of the
        base model which has self_attention=False and conditioning_type=None.
        """
        control_downs = nn.ModuleList()

        for idx in range(self.n_encoder_levels):
            dim_in = self.dims[idx]
            dim_out = self.dims[idx + 1]
            is_last = idx >= self.n_encoder_levels - 1

            control_block = nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, self.cond_dim, n_support_points=0),
                ResidualTemporalBlock(dim_out, dim_out, self.cond_dim, n_support_points=0),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ])
            control_downs.append(control_block)

        return control_downs

    def forward(self, x, time, sdf_emb):
        """
        Run ControlNet encoder to produce residuals for injection.

        Args:
            x: Noisy trajectory [B, H, state_dim] (same input as base model)
            time: Diffusion timestep [B]
            sdf_emb: Pre-computed SDF embedding [B, cond_dim] from MapSDFEncoder

        Returns:
            down_residuals: List of 3 tensors for decoder skip injection
            mid_residual: Single tensor for mid-block injection
        """
        # Time embedding + SDF embedding (FiLM conditioning)
        t_emb = self.control_time_mlp(time)   # [B, cond_dim]
        cond_emb = t_emb + sdf_emb            # [B, cond_dim]

        # Prepare trajectory: [B, H, C] -> [B, C, H]
        x_ctrl = einops.rearrange(x, 'b h c -> b c h')

        # Run through control encoder
        down_residuals = []
        for idx, block in enumerate(self.control_downs):
            resnet1, resnet2, downsample = block

            x_ctrl = resnet1(x_ctrl, cond_emb)
            x_ctrl = resnet2(x_ctrl, cond_emb)

            # Produce residual for levels 1, 2, 3 (skip level 0)
            if idx > 0:
                residual = self.zero_convs_down[idx - 1](x_ctrl)
                down_residuals.append(residual)

            x_ctrl = downsample(x_ctrl)

        # Run through control mid blocks
        x_ctrl = self.control_mid_block1(x_ctrl, cond_emb)
        x_ctrl = self.control_mid_block2(x_ctrl, cond_emb)
        mid_residual = self.zero_conv_mid(x_ctrl)

        return down_residuals, mid_residual

    def get_trainable_parameters(self):
        """Return all trainable parameters (everything in the ControlNet)."""
        return list(self.parameters())


class ControlledDiffusionModel(nn.Module):
    """
    Wrapper combining a frozen GaussianDiffusionModel with a trainable ControlNet.

    Handles the full pipeline:
    1. Encode SDF grid via ControlNet's MapSDFEncoder
    2. Run ControlNet encoder to produce residuals
    3. Inject residuals into frozen base model's forward pass
    4. Compute diffusion training loss

    Args:
        diffusion_model: GaussianDiffusionModel with pretrained TemporalUnet
        controlnet: MMDControlNet adapter
    """

    def __init__(self, diffusion_model, controlnet):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.controlnet = controlnet

        # Freeze the entire diffusion model (UNet + diffusion params)
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
        self.diffusion_model.eval()

    def forward(self, x, time, sdf_grid):
        """
        Forward pass: ControlNet encoder + frozen base model with residual injection.

        Args:
            x: Noisy trajectory [B, H, state_dim]
            time: Diffusion timestep [B]
            sdf_grid: SDF grid [B, 1, 64, 64]

        Returns:
            Predicted noise (or denoised trajectory) [B, H, state_dim]
        """
        # Encode SDF to global conditioning vector
        sdf_emb = self.controlnet.map_encoder(sdf_grid)  # [B, cond_dim]

        # Get residuals from ControlNet
        down_residuals, mid_residual = self.controlnet(x, time, sdf_emb)

        # Run frozen base model with injected residuals
        output = self.diffusion_model.model(
            x, time, context=None,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
        )

        return output

    def p_losses(self, x_start, t, hard_conds, sdf_grid):
        """
        Compute diffusion training loss with ControlNet conditioning.

        Args:
            x_start: Clean trajectory [B, H, state_dim]
            t: Diffusion timestep [B]
            hard_conds: Hard conditioning dict (start/goal constraints)
            sdf_grid: SDF grid [B, 1, 64, 64]

        Returns:
            loss: Scalar MSE loss
        """
        from mmd.models.diffusion_models.sample_functions import apply_hard_conditioning

        noise = torch.randn_like(x_start)

        # Add noise to clean trajectory (q_sample from frozen diffusion model)
        x_noisy = self.diffusion_model.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_hard_conditioning(x_noisy, hard_conds)

        # Predict noise using ControlNet-augmented model
        x_recon = self.forward(x_noisy, t, sdf_grid)
        x_recon = apply_hard_conditioning(x_recon, hard_conds)

        assert noise.shape == x_recon.shape

        if self.diffusion_model.predict_epsilon:
            loss = nn.functional.mse_loss(x_recon, noise)
        else:
            loss = nn.functional.mse_loss(x_recon, x_start)

        return loss

    def loss(self, x, hard_conds, sdf_grid):
        """
        Compute loss with randomly sampled timesteps.

        Args:
            x: Clean trajectory [B, H, state_dim]
            hard_conds: Hard conditioning dict
            sdf_grid: SDF grid [B, 1, 64, 64]

        Returns:
            loss: Scalar MSE loss
        """
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.diffusion_model.n_diffusion_steps,
            (batch_size,), device=x.device
        ).long()
        return self.p_losses(x, t, hard_conds, sdf_grid)

    def get_trainable_parameters(self):
        """Return only ControlNet's trainable parameters."""
        return self.controlnet.get_trainable_parameters()

    def train(self, mode=True):
        """Override to keep base diffusion model always frozen/eval."""
        super().train(mode)
        self.diffusion_model.eval()
        self.controlnet.train(mode)
        return self
