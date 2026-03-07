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

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from mmd.common.experiments import MultiAgentPlanningExperimentConfig, get_results_dir_from_experiment_config
from mmd.config.mmd_params import MMDParams as params
from torch_robotics.torch_utils.seed import fix_random_seed

from launch_multi_agent_experiment import run_multi_agent_experiment


BASE = Path(__file__).resolve().parents[2]
DEFAULT_CONTROLNET_CHECKPOINT = BASE / 'logs_controlnet_full' / '0' / 'checkpoints' / 'ema_controlnet_final_state_dict.pth'
DEFAULT_SDF_CACHE_DIR = BASE / 'data_trained_models' / 'EnvConveyor2D-RobotPlanarDisk' / 'controlnet' / 'sdf_cache'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the full multi-agent MAPF pipeline on EnvConveyor2D with the base MPD '
            'planner and the ControlNet-augmented MPD planner.'
        )
    )
    parser.add_argument(
        '--instance_name',
        type=str,
        default='EnvConveyor2DRobotPlanarDiskRandom',
        help='Planning problem config to evaluate (default: EnvConveyor2DRobotPlanarDiskRandom)',
    )
    parser.add_argument(
        '--scales',
        nargs='+',
        type=float,
        default=[1.0, 1.2],
        help='Environment scales to evaluate (default: 1.0 1.2)',
    )
    parser.add_argument(
        '--num_agents',
        nargs='+',
        type=int,
        default=[3],
        help='Numbers of agents to evaluate (default: 3)',
    )
    parser.add_argument(
        '--num_trials_per_combination',
        type=int,
        default=3,
        help='Trials per scale / mode / agent-count combination (default: 3)',
    )
    parser.add_argument(
        '--multi_agent_planner_classes',
        nargs='+',
        type=str,
        default=['XECBS'],
        help='High-level multi-agent planners to evaluate (default: XECBS)',
    )
    parser.add_argument(
        '--runtime_limit',
        type=int,
        default=60 * 3,
        help='Per-trial runtime limit in seconds (default: 180)',
    )
    parser.add_argument(
        '--stagger_start_time_dt',
        type=int,
        default=0,
        help='Start-time stagger between agents (default: 0)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=params.seed,
        help='Seed used for planning-problem generation and MPD sampling (default: MMDParams.seed)',
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['base', 'controlnet'],
        default=['base', 'controlnet'],
        help='Planner variants to evaluate (default: base controlnet)',
    )
    parser.add_argument(
        '--controlnet_checkpoint_path',
        type=str,
        default=str(DEFAULT_CONTROLNET_CHECKPOINT),
        help='ControlNet checkpoint path (default: EMA checkpoint from logs_controlnet_full)',
    )
    parser.add_argument(
        '--sdf_cache_dir',
        type=str,
        default=str(DEFAULT_SDF_CACHE_DIR),
        help='Directory containing sdf_scale_XX.pt cache files',
    )
    parser.add_argument(
        '--control_scale',
        type=float,
        default=1.0,
        help='Uniform ControlNet residual scale at inference (default: 1.0)',
    )
    parser.add_argument(
        '--no_render_animation',
        action='store_true',
        help='Disable GIF rendering and only save the static image per successful trial',
    )

    parser.add_argument('--weight_grad_cost_collision', type=float, default=None)
    parser.add_argument('--weight_grad_cost_smoothness', type=float, default=None)
    parser.add_argument('--weight_grad_cost_constraints', type=float, default=None)
    parser.add_argument('--weight_grad_cost_soft_constraints', type=float, default=None)
    parser.add_argument('--start_guide_steps_fraction', type=float, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--trajectory_duration', type=float, default=None)
    parser.add_argument('--n_guide_steps', type=int, default=None)

    return parser.parse_args()


def build_hyperparam_overrides(args):
    overrides = {}
    override_names = [
        'weight_grad_cost_collision',
        'weight_grad_cost_smoothness',
        'weight_grad_cost_constraints',
        'weight_grad_cost_soft_constraints',
        'start_guide_steps_fraction',
        'n_samples',
        'trajectory_duration',
        'n_guide_steps',
    ]

    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value

    if args.seed != params.seed:
        overrides['seed'] = args.seed

    return overrides if overrides else None


def build_experiment_config(args, env_scale, mode, hyperparam_overrides):
    experiment_config = MultiAgentPlanningExperimentConfig()
    experiment_config.num_agents_l = list(args.num_agents)
    experiment_config.instance_name = args.instance_name
    experiment_config.stagger_start_time_dt = args.stagger_start_time_dt
    experiment_config.multi_agent_planner_class_l = list(args.multi_agent_planner_classes)
    experiment_config.single_agent_planner_class = 'MPD'
    experiment_config.runtime_limit = args.runtime_limit
    experiment_config.num_trials_per_combination = args.num_trials_per_combination
    experiment_config.render_animation = not args.no_render_animation
    experiment_config.env_scale = env_scale
    experiment_config.hyperparam_overrides = hyperparam_overrides

    if mode == 'controlnet':
        experiment_config.controlnet_checkpoint_path = args.controlnet_checkpoint_path
        experiment_config.sdf_cache_dir = args.sdf_cache_dir
        experiment_config.control_scale = args.control_scale

    return experiment_config


def validate_args(args):
    if 'controlnet' not in args.modes:
        return

    controlnet_checkpoint_path = Path(args.controlnet_checkpoint_path)
    sdf_cache_dir = Path(args.sdf_cache_dir)

    if not controlnet_checkpoint_path.exists():
        raise FileNotFoundError(f'ControlNet checkpoint not found: {controlnet_checkpoint_path}')
    if not sdf_cache_dir.exists():
        raise FileNotFoundError(f'SDF cache directory not found: {sdf_cache_dir}')


def print_run_header(evaluation_id, mode, env_scale, args, hyperparam_overrides):
    print('\n' + '=' * 80)
    print('CONTROLNET EVALUATION RUN')
    print('=' * 80)
    print(f'Evaluation id: {evaluation_id}')
    print(f'Mode: {mode}')
    print(f'Instance: {args.instance_name}')
    print(f'Env scale: {env_scale}')
    print(f'Num agents: {list(args.num_agents)}')
    print(f'Multi-agent planners: {list(args.multi_agent_planner_classes)}')
    print(f'Trials per combination: {args.num_trials_per_combination}')
    print(f'Render animation: {not args.no_render_animation}')
    print(f'Seed: {args.seed}')
    if mode == 'controlnet':
        print(f'ControlNet checkpoint: {args.controlnet_checkpoint_path}')
        print(f'SDF cache dir: {args.sdf_cache_dir}')
        print(f'Control scale: {args.control_scale}')
    print(f'Hyperparameter overrides: {hyperparam_overrides}')
    print('=' * 80 + '\n')


def print_results_summary(results_dir, mode, env_scale):
    csv_path = results_dir / 'aggregated_results_all_agents.csv'
    print(f'Results dir: {results_dir}')

    if not csv_path.exists():
        print(f'Aggregated summary not found: {csv_path}')
        return

    df = pd.read_csv(csv_path)
    summary_columns = [
        'method',
        'num_agents',
        'success_rate',
        'avg_planning_time',
        'avg_ct_expansions',
        'avg_num_collisions_in_solution',
        'env_scale',
    ]
    summary_columns = [column for column in summary_columns if column in df.columns]

    print(f'Aggregated summary for mode={mode}, env_scale={env_scale}:')
    print(df[summary_columns].to_string(index=False))


def main():
    args = parse_args()
    validate_args(args)
    hyperparam_overrides = build_hyperparam_overrides(args)
    evaluation_id = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_records = []

    print('\n' + '=' * 80)
    print('CONTROLNET MULTI-AGENT EVALUATION')
    print('=' * 80)
    print('This launcher reuses the existing MAPF experiment pipeline and compares base MPD')
    print('against ControlNet-augmented MPD under identical random seeds per scale.')
    print(f'Evaluation id: {evaluation_id}')
    print('=' * 80 + '\n')

    for env_scale in args.scales:
        for mode in args.modes:
            fix_random_seed(args.seed)
            print_run_header(evaluation_id, mode, env_scale, args, hyperparam_overrides)

            experiment_config = build_experiment_config(args, env_scale, mode, hyperparam_overrides)
            run_multi_agent_experiment(experiment_config)

            results_dir = Path(get_results_dir_from_experiment_config(experiment_config))
            run_records.append({
                'mode': mode,
                'env_scale': env_scale,
                'time_str': experiment_config.time_str,
                'results_dir': results_dir,
            })
            print_results_summary(results_dir, mode, env_scale)

    print('\n' + '=' * 80)
    print('EVALUATION RUNS COMPLETE')
    print('=' * 80)
    for run_record in run_records:
        print(
            f"mode={run_record['mode']}, env_scale={run_record['env_scale']}, "
            f"time_str={run_record['time_str']}, results_dir={run_record['results_dir']}"
        )
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
