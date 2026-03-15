"""
MIT License

Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.
import argparse
import hashlib
import json

# Project includes.
from mmd.common.experiments.experiment_utils import *
from inference_multi_agent import run_multi_agent_trial


def run_multi_agent_experiment(experiment_config: MultiAgentPlanningExperimentConfig):
    # Run the multi-agent planning experiment.
    startt = time.time()
    # Create the experiment config with microsecond precision + PID to guarantee uniqueness for parallel runs.
    import os
    experiment_config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + f"-pid{os.getpid()}"
    # Get single trial configs from the experiment config.
    single_trial_configs = experiment_config.get_single_trial_configs_from_experiment_config()
    # So let's run sequentially.
    for single_trial_config in single_trial_configs:
        print(single_trial_config)
        try:
            run_multi_agent_trial(single_trial_config)

            # Aggregate and save data on every step. This is not needed (can be done once at the end).
            combine_and_save_results_for_experiment(experiment_config)
        except Exception as e:
            print("Error in run_multi_agent_experiment: ", e)
            # Save to a file.
            with open(f"error_{experiment_config.time_str}.txt", "a") as f:
                f.write(str(e))
                f.write("This is for single_trial_config: ")
                f.write(str(single_trial_config))
                f.write("\n")
            break

    # Print the runtime.
    print("Runtime: ", time.time() - startt)
    print("Run: OK.")


if __name__ == "__main__":
    # Parse command-line arguments for hyperparameters
    parser = argparse.ArgumentParser(description='Run multi-agent planning experiment with configurable hyperparameters')
    parser.add_argument('--weight_grad_cost_collision', type=float, default=2e-2,
                        help='Weight for collision cost gradient (default: 2e-2)')
    parser.add_argument('--weight_grad_cost_smoothness', type=float, default=8e-2,
                        help='Weight for smoothness cost gradient (default: 8e-2)')
    parser.add_argument('--weight_grad_cost_constraints', type=float, default=2e-1,
                        help='Weight for constraints cost gradient (default: 2e-1)')
    parser.add_argument('--weight_grad_cost_soft_constraints', type=float, default=2e-2,
                        help='Weight for soft constraints cost gradient (default: 2e-2)')
    parser.add_argument('--start_guide_steps_fraction', type=float, default=0.5,
                        help='Fraction of inference steps that are guided (default: 0.5)')
    parser.add_argument('--env_scale', type=float, default=1,
                        help='Environment scale factor (default: 1)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of trajectory samples (default: use MMDParams default)')
    parser.add_argument('--trajectory_duration', type=float, default=None,
                        help='Trajectory duration (default: use MMDParams default)')
    parser.add_argument('--n_guide_steps', type=int, default=None,
                        help='Number of guide steps per diffusion step (default: use MMDParams default)')
    parser.add_argument('--single_agent_planner_class', type=str, default='MPDEnsemble',
                        choices=['MPD', 'MPDEnsemble'],
                        help='Single-agent planner backend (default: MPDEnsemble)')
    parser.add_argument('--controlnet_checkpoint_path', type=str, default=None,
                        help='Optional ControlNet checkpoint path for MPD inference')
    parser.add_argument('--sdf_cache_dir', type=str, default=None,
                        help='Optional directory containing cached sdf_scale_XX.pt files')
    parser.add_argument('--control_scale', type=float, default=1.0,
                        help='Uniform ControlNet residual scale at inference (default: 1.0)')
    
    args = parser.parse_args()

    # Instance names. These dictate the maps and start/goals.
    # Create an experiment config.
    experiment_config = MultiAgentPlanningExperimentConfig()
    # Set the experiment config.
    experiment_config.num_agents_l = [3]

    # Single tile.
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskCircle"
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskBoundary"
    # experiment_config.instance_name = "EnvConveyor2DRobotPlanarDiskBoundary"
    # experiment_config.instance_name = "EnvHighways2DRobotPlanarDiskSmallCircle"
    # experiment_config.instance_name = "EnvDropRegion2DRobotPlanarDiskBoundary"
    experiment_config.instance_name = "EnvConveyor2DRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskRandom"
    #experiment_config.instance_name = "EnvHighways2DRobotPlanarDiskRandom"

    # Multiple tiles.
    # experiment_config.instance_name = "EnvTestTwoByTwoRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvTestThreeByThreeRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvTestFourByFourRobotPlanarDiskRandom"

    experiment_config.stagger_start_time_dt = 0
    experiment_config.multi_agent_planner_class_l = ["XECBS"]  # , "ECBS", "PP", "XCBS", "CBS"]
    experiment_config.single_agent_planner_class = args.single_agent_planner_class
    experiment_config.runtime_limit = 60 * 3
    experiment_config.num_trials_per_combination = 1
    experiment_config.render_animation = True
    experiment_config.env_scale = args.env_scale  # Use command-line argument
    experiment_config.controlnet_checkpoint_path = args.controlnet_checkpoint_path
    experiment_config.sdf_cache_dir = args.sdf_cache_dir
    experiment_config.control_scale = args.control_scale

    if experiment_config.controlnet_checkpoint_path is not None and experiment_config.single_agent_planner_class != 'MPD':
        raise ValueError('ControlNet inference is currently supported only with --single_agent_planner_class MPD.')
    
    # Build hyperparameter overrides from command-line arguments
    experiment_config.hyperparam_overrides = {
        'weight_grad_cost_collision': args.weight_grad_cost_collision,
        'weight_grad_cost_smoothness': args.weight_grad_cost_smoothness,
        'weight_grad_cost_constraints': args.weight_grad_cost_constraints,
        'weight_grad_cost_soft_constraints': args.weight_grad_cost_soft_constraints,
        'start_guide_steps_fraction': args.start_guide_steps_fraction,
    }

    if args.n_samples is not None:
        experiment_config.hyperparam_overrides['n_samples'] = args.n_samples
    if args.trajectory_duration is not None:
        experiment_config.hyperparam_overrides['trajectory_duration'] = args.trajectory_duration
    if args.n_guide_steps is not None:
        experiment_config.hyperparam_overrides['n_guide_steps'] = args.n_guide_steps

    # Create a unique identifier for this hyperparameter configuration
    # This prevents race conditions when running concurrent experiments
    hyperparam_str = json.dumps(experiment_config.hyperparam_overrides, sort_keys=True)
    hyperparam_hash = hashlib.md5(hyperparam_str.encode()).hexdigest()[:8]
    
    print("\n" + "="*80)
    print(f"EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Hyperparameter configuration hash: {hyperparam_hash}")
    print(f"Hyperparameters: {experiment_config.hyperparam_overrides}")
    print(f"Instance: {experiment_config.instance_name}")
    print(f"Num agents: {experiment_config.num_agents_l}")
    print(f"Single-agent planner: {experiment_config.single_agent_planner_class}")
    print(f"Env scale: {experiment_config.env_scale}")
    print(f"ControlNet checkpoint: {experiment_config.controlnet_checkpoint_path}")
    print(f"SDF cache dir: {experiment_config.sdf_cache_dir}")
    print(f"Control scale: {experiment_config.control_scale}")
    print("="*80 + "\n")
    
    # Run the experiment.
    run_multi_agent_experiment(experiment_config)
    
    print("\n" + "="*80)
    print(f"EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Results saved with time_str: {experiment_config.time_str}")
    print(f"Hyperparameters used: {experiment_config.hyperparam_overrides}")
    print("="*80 + "\n")

    # Example for only combining results without running the experiment.
    # experiment_config.time_str = "2024-09-08-13-55-05"
    # # Get all the results.
    # aggregated_results = combine_and_save_results_for_experiment(experiment_config)
    print(f"Finished running experiment with hyperparameters: {experiment_config.hyperparam_overrides}")
