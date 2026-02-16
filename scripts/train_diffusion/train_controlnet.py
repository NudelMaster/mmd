"""
Training script for ControlNet adapter on MMD trajectory diffusion model.

This script:
1. Loads a pretrained TemporalUnet/GaussianDiffusionModel (frozen)
2. Creates an MMDControlNet adapter with MapSDFEncoder
3. Trains only the ControlNet parameters (SDF encoder + control encoder + zero convs)
4. Saves checkpoints of the ControlNet

Design choices:
- Approach B (Global): SDF -> FiLM conditioning via time embedding
- Design 2: MapSDFEncoder outputs [B, 32] added to t_emb
- No attention blocks (faithful copy of base model)
- 3 down-block residuals (matching decoder levels)
- Scale parameter removed (SDF already encodes geometry)

Usage:
    python train_controlnet.py

Note:
    Requires:
    - Path to pretrained model (pretrained_model_dir)
    - Dataset with SDF grids (ControlNetTrajectoryDataset — TODO)
    - Pre-computed SDF cache for the target environment
"""
import os
import copy
import torch
import wandb
import numpy as np
from math import ceil
from tqdm.autonotebook import tqdm
from collections import defaultdict

from experiment_launcher import single_experiment_yaml, run_experiment
from mmd import trainer
from mmd.models import UNET_DIM_MULTS, TemporalUnet
from mmd.models.diffusion_models.controlnet import MMDControlNet, ControlledDiffusionModel
from mmd.models.diffusion_models.diffusion_model_base import GaussianDiffusionModel
from mmd.trainer import get_dataset, get_model
from mmd.trainer.trainer import get_num_epochs, save_model_to_disk, EMA
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, dict_to_device, to_numpy
from torch_robotics.torch_utils.torch_timer import TimerCUDA

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def controlnet_loss_fn(model, batch_dict, dataset):
    """
    Loss function for ControlNet training.

    Args:
        model: ControlledDiffusionModel instance
        batch_dict: Dictionary from dataloader containing:
            - '{field_key_traj}_normalized': trajectory [B, H, state_dim]
            - 'hard_conds': dict of hard conditioning
            - 'sdf_grid': SDF grid [B, 1, 64, 64]
        dataset: Dataset object (for field_key_traj)

    Returns:
        losses: Dictionary of loss values
        info: Dictionary of additional info
    """
    # Get trajectory (use normalized key, matching base training code)
    traj_key = f'{dataset.field_key_traj}_normalized'
    traj = batch_dict[traj_key]
    hard_conds = batch_dict.get('hard_conds', {})

    # Get SDF grid from batch
    sdf_grid = batch_dict['sdf_grid']  # [B, 1, 64, 64]

    # Compute diffusion loss with ControlNet conditioning
    loss = model.loss(traj, hard_conds, sdf_grid)

    losses = {'controlnet_loss': loss}
    info = {}

    return losses, info


def save_controlnet_to_disk(controlnet, epoch, total_steps, checkpoints_dir):
    """Save only ControlNet parameters to disk."""
    # Save ControlNet state dict (current)
    torch.save(
        controlnet.state_dict(),
        os.path.join(checkpoints_dir, 'controlnet_current_state_dict.pth')
    )
    # Save ControlNet state dict (versioned)
    torch.save(
        controlnet.state_dict(),
        os.path.join(checkpoints_dir, f'controlnet_epoch_{epoch:04d}_iter_{total_steps:06d}_state_dict.pth')
    )
    # Save full ControlNet model object
    torch.save(
        controlnet,
        os.path.join(checkpoints_dir, 'controlnet_current.pth')
    )


def train_controlnet(
    # ControlNet model
    controlled_model,

    # Data
    train_dataloader,
    train_subset,
    val_dataloader=None,
    val_subset=None,

    # Training params
    epochs=100,
    lr=1e-4,

    # Logging
    steps_til_summary=100,
    steps_til_checkpoint=5000,
    model_dir='logs',

    # Options
    clip_grad=True,
    clip_grad_max_norm=1.0,
    use_ema=True,
    ema_decay=0.995,
    step_start_ema=1000,
    update_ema_every=10,
    use_amp=False,

    # Misc
    debug=False,
    tensor_args=None,
):
    """
    Training loop for ControlNet.

    Similar to the main trainer but specifically handles ControlNet training
    where the base model is frozen and SDF grids provide conditioning.
    """
    if tensor_args is None:
        tensor_args = {'device': 'cuda', 'dtype': torch.float32}

    print(f'\n------- CONTROLNET TRAINING STARTED -------\n')

    # EMA model for ControlNet
    ema_controlnet = None
    if use_ema:
        ema = EMA(beta=ema_decay)
        ema_controlnet = copy.deepcopy(controlled_model.controlnet)

    # Optimizer - only optimize ControlNet parameters
    trainable_params = controlled_model.get_trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in controlled_model.diffusion_model.parameters())
    print(f"Trainable parameters (ControlNet): {n_trainable:,}")
    print(f"Frozen parameters (base model):    {n_frozen:,}")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Directories
    os.makedirs(model_dir, exist_ok=True)
    summaries_dir = os.path.join(model_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Training loop
    train_steps_current = 0
    train_losses_l = []
    validation_losses_l = []

    # Save initial model
    save_controlnet_to_disk(controlled_model.controlnet, 0, 0, checkpoints_dir)

    total_steps = len(train_dataloader) * epochs
    with tqdm(total=total_steps, mininterval=1 if debug else 60) as pbar:
        for epoch in range(epochs):
            controlled_model.train()

            for step, train_batch_dict in enumerate(train_dataloader):
                # Move batch to device
                train_batch_dict = dict_to_device(train_batch_dict, tensor_args['device'])

                # Compute loss
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    train_losses, train_losses_info = controlnet_loss_fn(
                        controlled_model,
                        train_batch_dict,
                        train_subset.dataset,
                    )

                train_loss_batch = sum(loss.mean() for loss in train_losses.values())

                # Logging
                if train_steps_current % steps_til_summary == 0:
                    train_losses_log = {k: to_numpy(v.mean()).item() for k, v in train_losses.items()}
                    print(f"\n-----------------------------------------")
                    print(f"Step: {train_steps_current}, Epoch: {epoch}")
                    print(f"Training loss: {train_loss_batch:.6f}")
                    print(f"Losses: {train_losses_log}")

                    train_losses_l.append((train_steps_current, train_losses_log))

                    # Validation
                    if val_dataloader is not None:
                        controlled_model.eval()
                        val_losses_total = []
                        with torch.no_grad():
                            for val_step, val_batch_dict in enumerate(val_dataloader):
                                val_batch_dict = dict_to_device(val_batch_dict, tensor_args['device'])
                                val_losses, _ = controlnet_loss_fn(
                                    controlled_model,
                                    val_batch_dict,
                                    val_subset.dataset,
                                )
                                val_losses_total.append(to_numpy(val_losses['controlnet_loss'].mean()))
                                if val_step >= 10:  # Limit validation steps
                                    break

                        val_loss_mean = np.mean(val_losses_total)
                        print(f"Validation loss: {val_loss_mean:.6f}")
                        validation_losses_l.append((train_steps_current, {'val_loss': val_loss_mean}))
                        controlled_model.train()

                    wandb.log({'train_loss': train_loss_batch.item()}, step=train_steps_current)

                # Optimization
                optimizer.zero_grad()
                scaler.scale(train_loss_batch).backward()

                if clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad_max_norm)

                scaler.step(optimizer)
                scaler.update()

                # EMA update
                if ema_controlnet is not None:
                    if train_steps_current % update_ema_every == 0:
                        if train_steps_current < step_start_ema:
                            ema_controlnet.load_state_dict(controlled_model.controlnet.state_dict())
                        else:
                            ema.update_model_average(ema_controlnet, controlled_model.controlnet)

                # Checkpointing
                if steps_til_checkpoint is not None and train_steps_current % steps_til_checkpoint == 0:
                    save_controlnet_to_disk(
                        controlled_model.controlnet, epoch, train_steps_current, checkpoints_dir
                    )
                    if ema_controlnet is not None:
                        torch.save(
                            ema_controlnet.state_dict(),
                            os.path.join(checkpoints_dir, 'ema_controlnet_current_state_dict.pth')
                        )

                    # Save losses
                    np.save(os.path.join(checkpoints_dir, 'train_losses.npy'), train_losses_l)
                    np.save(os.path.join(checkpoints_dir, 'val_losses.npy'), validation_losses_l)

                pbar.update(1)
                train_steps_current += 1

    # Final save
    save_controlnet_to_disk(controlled_model.controlnet, epochs, train_steps_current, checkpoints_dir)
    if ema_controlnet is not None:
        torch.save(
            ema_controlnet.state_dict(),
            os.path.join(checkpoints_dir, 'ema_controlnet_final_state_dict.pth')
        )

    print(f'\n------- CONTROLNET TRAINING FINISHED -------')

    return controlled_model


@single_experiment_yaml
def experiment(
    ########################################################################
    # Pretrained Model
    pretrained_model_dir: str = None,  # Path to pretrained model directory
    use_ema_model: bool = True,  # Whether to use EMA model

    ########################################################################
    # Dataset
    dataset_subdir: str = 'EnvConveyor2D-RobotPlanarDisk',
    include_velocity: bool = True,

    ########################################################################
    # ControlNet
    sdf_encoder_hidden_dim: int = 256,  # MapSDFEncoder CNN hidden dimension

    ########################################################################
    # Diffusion Model (must match pretrained)
    diffusion_model_class: str = 'GaussianDiffusionModel',
    variance_schedule: str = 'exponential',
    n_diffusion_steps: int = 25,
    predict_epsilon: bool = True,

    # Unet (must match pretrained)
    unet_input_dim: int = 32,
    unet_dim_mults_option: int = 1,  # 1 for EnvConveyor2D: (1,2,4,8)

    ########################################################################
    # Training parameters
    batch_size: int = 32,
    lr: float = 1e-4,
    num_train_steps: int = 100000,

    use_ema: bool = True,
    use_amp: bool = False,

    # Logging
    steps_til_summary: int = 100,
    steps_til_ckpt: int = 10000,

    ########################################################################
    device: str = 'cuda',
    debug: bool = True,

    ########################################################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs_controlnet',

    ########################################################################
    # WandB
    wandb_mode: str = 'disabled',
    wandb_entity: str = 'scoreplan',
    wandb_project: str = 'mmd_controlnet',
    **kwargs
):
    """
    Main experiment function for training ControlNet.

    Target: EnvConveyor2D-RobotPlanarDisk
    Design: Approach B (Global FiLM) + Design 2 (SDF via time embedding)
    """
    fix_random_seed(seed)

    device = get_torch_device(device=device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Initialize wandb
    wandb.init(
        mode=wandb_mode,
        entity=wandb_entity,
        project=wandb_project,
        config={
            'pretrained_model_dir': pretrained_model_dir,
            'sdf_encoder_hidden_dim': sdf_encoder_hidden_dim,
            'lr': lr,
            'batch_size': batch_size,
            'unet_dim_mults_option': unet_dim_mults_option,
        }
    )

    ########################################################################
    # Load Dataset
    ########################################################################
    # TODO: Replace with ControlNetTrajectoryDataset that provides sdf_grid
    # For now, use base TrajectoryDataset (sdf_grid will need to be added)
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=include_velocity,
        dataset_subdir=dataset_subdir,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args
    )

    dataset = train_subset.dataset

    ########################################################################
    # Load Pretrained Model
    ########################################################################
    if pretrained_model_dir is None:
        raise ValueError("Must provide pretrained_model_dir for ControlNet training")

    # Build model architecture (must match pretrained)
    diffusion_configs = dict(
        variance_schedule=variance_schedule,
        n_diffusion_steps=n_diffusion_steps,
        predict_epsilon=predict_epsilon,
    )

    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],
    )

    diffusion_model = get_model(
        model_class=diffusion_model_class,
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )

    # Load pretrained weights
    model_filename = 'ema_model_current_state_dict.pth' if use_ema_model else 'model_current_state_dict.pth'
    model_path = os.path.join(pretrained_model_dir, 'checkpoints', model_filename)
    print(f"Loading pretrained model from: {model_path}")

    diffusion_model.load_state_dict(
        torch.load(model_path, map_location=tensor_args['device'])
    )
    diffusion_model.eval()

    ########################################################################
    # Create ControlNet
    ########################################################################
    print("Creating ControlNet adapter...")

    controlnet = MMDControlNet(
        base_model=diffusion_model.model,  # The TemporalUnet inside GaussianDiffusionModel
        cond_dim=32,  # Verified: matches all pretrained models
        sdf_encoder_hidden_dim=sdf_encoder_hidden_dim,
    )
    controlnet = controlnet.to(device)

    # Wrap in ControlledDiffusionModel
    controlled_model = ControlledDiffusionModel(
        diffusion_model=diffusion_model,
        controlnet=controlnet,
    )
    controlled_model = controlled_model.to(device)

    ########################################################################
    # Train ControlNet
    ########################################################################
    train_controlnet(
        controlled_model=controlled_model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        epochs=get_num_epochs(num_train_steps, batch_size, len(dataset)),
        lr=lr,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        model_dir=results_dir,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        tensor_args=tensor_args,
    )

    wandb.finish()


if __name__ == '__main__':
    run_experiment(experiment)
