"""
MapSDFEncoder: Encodes SDF grids into global conditioning vectors for ControlNet.

Design 2 (FiLM conditioning):
    SDF [B, 1, 64, 64]  -->  CNN  -->  [B, 256, 8, 8]
                                            |
                                    global avg pool
                                            |
                                       [B, 256]
                                            |
                                    projection MLP
                                            |
                                       [B, cond_dim]  (default: 32)

The output is added to the time embedding: cond_emb = t_emb + sdf_emb
and passed to all ResNet blocks via FiLM modulation.

This design:
- Avoids cross-domain mapping (2D SDF -> 1D trajectory)
- Uses the same conditioning mechanism the base model already uses for time
- Scale parameter is NOT used (SDF already encodes scaled geometry)

See TEMPORAL_UNET_ARCHITECTURE.md for full design rationale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_sdf(sdf_tensor, target_size=64):
    """
    Resize SDF grid from native resolution to target size.

    Native SDF resolution is 400x400 (cell_size=0.005, environment 2x2 units).
    We resize to 64x64 for neural network input.

    Args:
        sdf_tensor: SDF grid, one of:
            - [H, W]           (single grid)
            - [1, H, W]        (single grid with channel dim)
            - [B, 1, H, W]     (batched)
            - [B, H, W]        (batched, no channel dim)
        target_size: Target spatial resolution (default: 64)

    Returns:
        Resized SDF [B, 1, target_size, target_size] or [1, 1, target_size, target_size]
    """
    # Normalize to [B, 1, H, W]
    if sdf_tensor.dim() == 2:
        sdf_tensor = sdf_tensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif sdf_tensor.dim() == 3:
        if sdf_tensor.shape[0] == 1:
            # [1, H, W] -> [1, 1, H, W]
            sdf_tensor = sdf_tensor.unsqueeze(0)
        else:
            # [B, H, W] -> [B, 1, H, W]
            sdf_tensor = sdf_tensor.unsqueeze(1)
    # else: already [B, 1, H, W]

    if sdf_tensor.shape[-1] == target_size and sdf_tensor.shape[-2] == target_size:
        return sdf_tensor

    return F.interpolate(
        sdf_tensor.float(),
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False,
    )


class MapSDFEncoder(nn.Module):
    """
    Encodes SDF grid into a global conditioning vector via CNN + pooling.

    Design 2 (FiLM): SDF -> CNN -> global avg pool -> projection -> [B, cond_dim]
    The output is added to the time embedding for FiLM conditioning of ResNet blocks.

    Architecture:
        CNN (4 layers):
            [B, 1, 64, 64] -> [B, 32, 32, 32] -> [B, 64, 16, 16]
                            -> [B, hidden_dim, 8, 8] -> [B, hidden_dim, 8, 8]
        Global average pool:
            [B, hidden_dim, 8, 8] -> [B, hidden_dim]
        Projection MLP:
            [B, hidden_dim] -> [B, cond_dim]

    Args:
        cond_dim: Output dimension, must match time_emb_dim of the base model (default: 32)
        hidden_dim: Internal CNN channel width (default: 256)
    """

    def __init__(self, cond_dim=32, hidden_dim=256):
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # CNN: [B, 1, 64, 64] -> [B, hidden_dim, 8, 8]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),       # 64 -> 32
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),      # 32 -> 16
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),           # 1x1 refinement
        )

        # Projection MLP: [B, hidden_dim] -> [B, cond_dim]
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, cond_dim),
        )

    def forward(self, sdf_grid):
        """
        Encode SDF grid into a global conditioning vector.

        Args:
            sdf_grid: [B, 1, 64, 64] - Resized SDF grid (already encodes scale)

        Returns:
            sdf_emb: [B, cond_dim] - Global conditioning vector to add to time embedding
        """
        features = self.cnn(sdf_grid)               # [B, hidden_dim, 8, 8]
        global_vec = features.mean(dim=[2, 3])       # [B, hidden_dim]
        sdf_emb = self.projection(global_vec)        # [B, cond_dim]
        return sdf_emb
