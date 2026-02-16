"""
Generate multi-scale trajectory data for ControlNet training using the pretrained MMD diffusion model.

This script generates trajectory datasets at multiple discrete scales (e.g. 1.0, 1.1, ..., 1.5).
It uses the MPD (Motion Planning with Diffusion) planner with a PRETRAINED MMD model
(trained on 1.0x scale) to generate trajectories in scaled environments.
This ensures the data reflects the diffusion model's capabilities (and failures) which
ControlNet will learn to correct.

Usage:
    python scripts/generate_data/generate_multiscale_data.py \
        --env_id EnvConveyor2D \
        --robot_id RobotPlanarDisk \
        --model_id EnvConveyor2D-RobotPlanarDisk \
        --scales 1.0 1.1 1.2 1.3 1.4 1.5 \
        --num_trajs 500

Output structure:
    data_trajectories/{env_id}-multiscale/
        scale_1.00/
            0/
                trajs-free.pt
                metadata.yaml
                ...
"""
import argparse
import os
import time
import yaml
import torch
import numpy as np
import shutil

from mmd.planners.single_agent.mpd import MPD
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

def generate_trajectories_for_scale(
    env_id,
    scale,
    robot_id,
    model_id,
    num_trajectories,
    results_dir,
    device,
    tensor_args,
    **kwargs
):
    """
    Generate trajectories for a specific environment scale using MPD.
    """
    print(f"Generating {num_trajectories} trajectories for scale {scale:.2f} using MMD model {model_id}...")
    
    # Initialize MPD Planner
    # We pass env_scale to resize the environment while keeping the model the same
    planner = MPD(
        model_id=model_id,
        planner_alg='mmd',
        start_state_pos=None, # Will be sampled per trajectory
        goal_state_pos=None,
        use_guide_on_extra_objects_only=False,
        start_guide_steps_fraction=0.2, # Start guidance after 20% of diffusion steps
        n_guide_steps=1,
        n_diffusion_steps_without_noise=5,
        weight_grad_cost_collision=1.0, # Standard guidance weights
        weight_grad_cost_smoothness=1.0,
        weight_grad_cost_constraints=1.0,
        weight_grad_cost_soft_constraints=1.0,
        factor_num_interpolated_points_for_collision=1.5,
        trajectory_duration=5.0,
        device=device,
        debug=False,
        seed=kwargs.get('seed', 0),
        results_dir=os.path.join(results_dir, f'temp_mpd_scale_{scale}'), # Temp dir for MPD internals
        trained_models_dir='data_trained_models',
        n_samples=1, # Generate 1 trajectory per start/goal pair
        n_local_inference_noising_steps=0,
        n_local_inference_denoising_steps=0,
        env_scale=scale, # CRITICAL: Create environment at this scale
        horizon=64 # Default horizon
    )
    
    trajs_free_all = []
    
    # Generation Loop
    # We generate trajectories one by one to ensure diverse start/goal pairs
    
    attempts = 0
    max_attempts = num_trajectories * 5 # Allow some failures
    
    while len(trajs_free_all) < num_trajectories and attempts < max_attempts:
        attempts += 1
        
        # 1. Sample Start/Goal using the planner's task/env
        start_state_pos, goal_state_pos = None, None
        n_tries = 100
        for _ in range(n_tries):
            q_free = planner.task.random_coll_free_q(n_samples=2)
            s_pos, g_pos = q_free[0], q_free[1]
            
            # Check validity (distance, etc)
            if torch.linalg.norm(s_pos - g_pos) > 0.5: # threshold_start_goal_pos
                start_state_pos, goal_state_pos = s_pos, g_pos
                break
        
        if start_state_pos is None:
            continue
            
        # 2. Run Inference
        try:
            # Update internal state for new start/goal
            planner.start_state_pos = start_state_pos
            planner.goal_state_pos = goal_state_pos
            
            # Re-create hard conditions for the new start/goal
            planner.hard_conds = planner.dataset.get_hard_conditions(
                torch.vstack((start_state_pos, goal_state_pos)), 
                normalize=True
            )
            
            # Run inference
            result = planner(start_state_pos, goal_state_pos)
            
            # Check result
            if result.trajs_final_free is not None:
                trajs_free_all.append(result.trajs_final_free)
                print(f"  [{len(trajs_free_all)}/{num_trajectories}] Generated (Scale {scale:.2f})")
            
        except Exception as e:
            print(f"  Generation failed: {e}")
            continue

    # Cleanup temp dir
    try:
        shutil.rmtree(os.path.join(results_dir, f'temp_mpd_scale_{scale}'))
    except:
        pass

    # -------------------------------- Save ---------------------------------
    if len(trajs_free_all) > 0:
        trajs_free_tensor = torch.cat(trajs_free_all)[:num_trajectories]
        
        # Save directory structure matching TrajectoryDataset expectations:
        # base/scale_X.XX/0/trajs-free.pt
        save_dir = os.path.join(results_dir, f"scale_{scale:.2f}", "0")
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(trajs_free_tensor, os.path.join(save_dir, 'trajs-free.pt'))
        
        # Save metadata
        metadata = {
            'env_id': env_id,
            'robot_id': robot_id,
            'model_id': model_id,
            'num_trajectories': len(trajs_free_tensor),
            'scale': scale,
            'n_support_points': 64,
            'duration': 5.0
        }
        with open(os.path.join(save_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f)
            
        # Save minimal args.yaml for compatibility
        args_data = {
            'obstacle_cutoff_margin': 0.05,
            'threshold_start_goal_pos': 0.5,
            'env_scale': scale
        }
        with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
            yaml.dump(args_data, f)
            
        print(f"Saved {len(trajs_free_tensor)} trajectories to {save_dir}")
    else:
        print(f"Failed to generate any trajectories for scale {scale}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-scale trajectories using pretrained MMD.")
    parser.add_argument('--env_id', type=str, default='EnvConveyor2D')
    parser.add_argument('--robot_id', type=str, default='RobotPlanarDisk')
    parser.add_argument('--model_id', type=str, default='EnvConveyor2D-RobotPlanarDisk', help='Pretrained MMD model directory name')
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    parser.add_argument('--num_trajs', type=int, default=100, help='Number of trajectories per scale')
    parser.add_argument('--output_dir', type=str, default='data_trajectories')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    fix_random_seed(args.seed)
    device = get_torch_device(device=args.device)
    tensor_args = {'device': device, 'dtype': torch.float32}
    
    # Construct output directory
    # Format: {env_id}-{robot_id}-multiscale
    base_output_dir = os.path.join(args.output_dir, f"{args.env_id}-{args.robot_id}-multiscale")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create top-level '0' directory with default args/metadata for TrajectoryDataset compatibility
    top_level_0 = os.path.join(base_output_dir, '0')
    os.makedirs(top_level_0, exist_ok=True)
    
    # Default metadata
    with open(os.path.join(top_level_0, 'metadata.yaml'), 'w') as f:
        yaml.dump({
            'env_id': args.env_id,
            'robot_id': args.robot_id,
            'is_multiscale': True
        }, f)
        
    # Default args
    with open(os.path.join(top_level_0, 'args.yaml'), 'w') as f:
        yaml.dump({
            'threshold_start_goal_pos': 0.5,
            'obstacle_cutoff_margin': 0.05,
            'env_scale': 1.0
        }, f)
    
    print(f"Generating data for {args.env_id} using model {args.model_id}")
    print(f"Scales: {args.scales}")
    print(f"Output directory: {base_output_dir}")
    
    for scale in args.scales:
        generate_trajectories_for_scale(
            env_id=args.env_id,
            scale=scale,
            robot_id=args.robot_id,
            model_id=args.model_id,
            num_trajectories=args.num_trajs,
            results_dir=base_output_dir,
            device=args.device,
            tensor_args=tensor_args,
            seed=args.seed
        )
        
    print("\nGeneration complete.")

if __name__ == "__main__":
    main()
