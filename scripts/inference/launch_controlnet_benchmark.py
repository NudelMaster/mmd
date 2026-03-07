"""
Generate benchmark shell scripts for the ControlNet Conveyor evaluation.

This launcher does not run the benchmark itself. It writes GPU-specific shell
scripts that call `launch_controlnet_evaluation.py` with the fixed
best-performing baseline hyperparameters, so the user can inspect and launch the
full experiment manually.
"""

import argparse
import shlex
from datetime import datetime
from pathlib import Path

from mmd.config.mmd_params import MMDParams as params


BASE = Path(__file__).resolve().parents[2]
INFERENCE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = INFERENCE_DIR / 'gpu_scripts'
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / 'controlnet_benchmark_manifest.txt'
DEFAULT_EMA_CHECKPOINT = (
    BASE / 'logs_controlnet_full' / '0' / 'checkpoints' / 'ema_controlnet_final_state_dict.pth'
)
DEFAULT_BESTVAL_CHECKPOINT = (
    BASE / 'logs_controlnet_full' / '0' / 'checkpoints' / 'controlnet_epoch_0333_iter_030000_state_dict.pth'
)
DEFAULT_SDF_CACHE_DIR = (
    BASE / 'data_trained_models' / 'EnvConveyor2D-RobotPlanarDisk' / 'controlnet' / 'sdf_cache'
)

DEFAULT_SCALES = [1.0, 1.1, 1.2, 1.3, 1.4]
DEFAULT_NUM_AGENTS = [6, 9, 12, 15]
DEFAULT_MULTI_AGENT_PLANNERS = ['XECBS']
WINNING_HYPERPARAMS = {
    'weight_grad_cost_collision': 0.05,
    'weight_grad_cost_smoothness': 0.08,
    'weight_grad_cost_constraints': 0.2,
    'weight_grad_cost_soft_constraints': 0.01,
    'start_guide_steps_fraction': 0.35,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate GPU shell scripts for the base-vs-ControlNet benchmark on '
            'EnvConveyor2D using the best hyperparameter configuration found by '
            'grid search.'
        )
    )
    parser.add_argument(
        '--instance_name',
        type=str,
        default='EnvConveyor2DRobotPlanarDiskRandom',
        help='Planning problem config to benchmark',
    )
    parser.add_argument(
        '--scales',
        nargs='+',
        type=float,
        default=DEFAULT_SCALES,
        help='Environment scales for the benchmark sweep',
    )
    parser.add_argument(
        '--num_agents',
        nargs='+',
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help='Agent counts for the benchmark sweep',
    )
    parser.add_argument(
        '--num_trials_per_combination',
        type=int,
        default=10,
        help='Trials per (scale, num_agents, mode) combination',
    )
    parser.add_argument(
        '--runtime_limit',
        type=int,
        default=60 * 3,
        help='Per-trial runtime limit in seconds',
    )
    parser.add_argument(
        '--stagger_start_time_dt',
        type=int,
        default=0,
        help='Start-time stagger between agents',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=params.seed,
        help='Seed passed through to the evaluation launcher',
    )
    parser.add_argument(
        '--multi_agent_planner_classes',
        nargs='+',
        type=str,
        default=DEFAULT_MULTI_AGENT_PLANNERS,
        help='High-level multi-agent planners to benchmark',
    )
    parser.add_argument(
        '--control_scale',
        type=float,
        default=1.0,
        help='Uniform ControlNet residual scale used in ControlNet runs',
    )
    parser.add_argument(
        '--ema_checkpoint_path',
        type=str,
        default=str(DEFAULT_EMA_CHECKPOINT),
        help='EMA ControlNet checkpoint path',
    )
    parser.add_argument(
        '--bestval_checkpoint_path',
        type=str,
        default=str(DEFAULT_BESTVAL_CHECKPOINT),
        help='Best-validation-region ControlNet checkpoint path',
    )
    parser.add_argument(
        '--sdf_cache_dir',
        type=str,
        default=str(DEFAULT_SDF_CACHE_DIR),
        help='Directory containing cached sdf_scale_XX.pt files for ControlNet runs',
    )
    parser.add_argument(
        '--base_gpu',
        type=int,
        default=0,
        help='GPU id assigned to the base benchmark script',
    )
    parser.add_argument(
        '--controlnet_gpu',
        type=int,
        default=1,
        help='GPU id assigned to the ControlNet benchmark script',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory where generated shell scripts and manifest are written',
    )
    parser.add_argument(
        '--render_animation',
        action='store_true',
        help='Render GIFs during the benchmark sweep (default: disabled)',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print the generated commands without writing any files',
    )
    return parser.parse_args()


def validate_args(args):
    ema_checkpoint = Path(args.ema_checkpoint_path)
    bestval_checkpoint = Path(args.bestval_checkpoint_path)
    sdf_cache_dir = Path(args.sdf_cache_dir)

    if not ema_checkpoint.exists():
        raise FileNotFoundError(f'EMA checkpoint not found: {ema_checkpoint}')
    if not bestval_checkpoint.exists():
        raise FileNotFoundError(f'Best-val checkpoint not found: {bestval_checkpoint}')
    if not sdf_cache_dir.exists():
        raise FileNotFoundError(f'SDF cache directory not found: {sdf_cache_dir}')

    if args.num_trials_per_combination <= 0:
        raise ValueError('num_trials_per_combination must be positive.')
    if args.runtime_limit <= 0:
        raise ValueError('runtime_limit must be positive.')


def build_base_command(args):
    command = [
        'python',
        'launch_controlnet_evaluation.py',
        '--instance_name', args.instance_name,
        '--modes', 'base',
        '--scales', *[str(scale) for scale in args.scales],
        '--num_agents', *[str(num_agents) for num_agents in args.num_agents],
        '--num_trials_per_combination', str(args.num_trials_per_combination),
        '--multi_agent_planner_classes', *args.multi_agent_planner_classes,
        '--runtime_limit', str(args.runtime_limit),
        '--stagger_start_time_dt', str(args.stagger_start_time_dt),
        '--seed', str(args.seed),
    ]
    command.extend(flatten_hyperparam_args())
    if not args.render_animation:
        command.append('--no_render_animation')
    return command


def build_controlnet_command(args, checkpoint_path):
    command = [
        'python',
        'launch_controlnet_evaluation.py',
        '--instance_name', args.instance_name,
        '--modes', 'controlnet',
        '--scales', *[str(scale) for scale in args.scales],
        '--num_agents', *[str(num_agents) for num_agents in args.num_agents],
        '--num_trials_per_combination', str(args.num_trials_per_combination),
        '--multi_agent_planner_classes', *args.multi_agent_planner_classes,
        '--runtime_limit', str(args.runtime_limit),
        '--stagger_start_time_dt', str(args.stagger_start_time_dt),
        '--seed', str(args.seed),
        '--control_scale', str(args.control_scale),
        '--controlnet_checkpoint_path', str(checkpoint_path),
        '--sdf_cache_dir', str(args.sdf_cache_dir),
    ]
    command.extend(flatten_hyperparam_args())
    if not args.render_animation:
        command.append('--no_render_animation')
    return command


def flatten_hyperparam_args():
    arg_list = []
    for name, value in WINNING_HYPERPARAMS.items():
        arg_list.extend([f'--{name}', str(value)])
    return arg_list


def shell_join(command):
    return ' '.join(shlex.quote(token) for token in command)


def build_script_body(gpu_id, labeled_commands):
    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        f'export CUDA_VISIBLE_DEVICES={gpu_id}',
        f'cd {shlex.quote(str(INFERENCE_DIR))}',
        '',
    ]

    total = len(labeled_commands)
    for index, (label, command) in enumerate(labeled_commands, 1):
        lines.extend([
            "echo ''",
            "echo '============================================================'",
            f"echo 'GPU {gpu_id} - Benchmark command {index}/{total}'",
            f"echo 'Label: {label}'",
            f"echo 'Command: {shell_join(command)}'",
            "echo '============================================================'",
            "echo ''",
            shell_join(command),
            '',
        ])

    return '\n'.join(lines) + '\n'


def build_manifest(args, script_paths, labeled_commands):
    total_base_trials = len(args.scales) * len(args.num_agents) * args.num_trials_per_combination
    total_controlnet_trials = total_base_trials * 2
    total_trials = total_base_trials + total_controlnet_trials

    lines = [
        'CONTROLNET BENCHMARK MANIFEST',
        '=' * 80,
        f'Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Instance: {args.instance_name}',
        f'Scales: {args.scales}',
        f'Num agents: {args.num_agents}',
        f'Trials per combination: {args.num_trials_per_combination}',
        f'Runtime limit: {args.runtime_limit}',
        f'Seed: {args.seed}',
        f'Render animation: {args.render_animation}',
        f'Control scale: {args.control_scale}',
        '',
        'Winning hyperparameters:',
    ]

    for name, value in WINNING_HYPERPARAMS.items():
        lines.append(f'- {name}: {value}')

    lines.extend([
        '',
        f'Base trials: {total_base_trials}',
        f'ControlNet trials: {total_controlnet_trials}',
        f'Total trials: {total_trials}',
        '',
        'Checkpoints:',
        f'- EMA: {args.ema_checkpoint_path}',
        f'- Best-val: {args.bestval_checkpoint_path}',
        f'- SDF cache: {args.sdf_cache_dir}',
        '',
        'Generated scripts:',
    ])

    for script_path in script_paths:
        lines.append(f'- {script_path}')

    lines.extend([
        '',
        'Commands:',
    ])

    for label, command in labeled_commands:
        lines.append(f'- {label}: {shell_join(command)}')

    lines.extend([
        '',
        'Launch manually with:',
        f'- bash {script_paths[0]}',
        f'- bash {script_paths[1]}',
    ])

    return '\n'.join(lines) + '\n'


def print_summary(args, script_paths, labeled_commands):
    print('\n' + '=' * 80)
    print('CONTROLNET BENCHMARK SCRIPT GENERATION')
    print('=' * 80)
    print(f'Instance: {args.instance_name}')
    print(f'Scales: {args.scales}')
    print(f'Num agents: {args.num_agents}')
    print(f'Trials per combination: {args.num_trials_per_combination}')
    print(f'Runtime limit: {args.runtime_limit}')
    print(f'Seed: {args.seed}')
    print(f'Render animation: {args.render_animation}')
    print(f'Control scale: {args.control_scale}')
    print('Winning hyperparameters:')
    for name, value in WINNING_HYPERPARAMS.items():
        print(f'  {name}={value}')
    print('Generated commands:')
    for label, command in labeled_commands:
        print(f'  {label}: {shell_join(command)}')
    if script_paths:
        print('Generated scripts:')
        for script_path in script_paths:
            print(f'  {script_path}')
    print('=' * 80 + '\n')


def main():
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    base_script_path = output_dir / 'controlnet_benchmark_gpu0.sh'
    controlnet_script_path = output_dir / 'controlnet_benchmark_gpu1.sh'
    manifest_path = output_dir / DEFAULT_MANIFEST_PATH.name

    base_command = build_base_command(args)
    controlnet_ema_command = build_controlnet_command(args, args.ema_checkpoint_path)
    controlnet_bestval_command = build_controlnet_command(args, args.bestval_checkpoint_path)

    labeled_commands = [
        ('base', base_command),
        ('controlnet_ema', controlnet_ema_command),
        ('controlnet_bestval', controlnet_bestval_command),
    ]

    print_summary(args, [base_script_path, controlnet_script_path], labeled_commands)

    if args.dry_run:
        print('Dry run only: no files were written.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    base_script_path.write_text(
        build_script_body(args.base_gpu, [('base', base_command)]),
        encoding='ascii',
    )
    controlnet_script_path.write_text(
        build_script_body(
            args.controlnet_gpu,
            [
                ('controlnet_ema', controlnet_ema_command),
                ('controlnet_bestval', controlnet_bestval_command),
            ],
        ),
        encoding='ascii',
    )
    manifest_path.write_text(
        build_manifest(args, [base_script_path, controlnet_script_path], labeled_commands),
        encoding='ascii',
    )

    base_script_path.chmod(0o755)
    controlnet_script_path.chmod(0o755)

    print(f'Wrote {base_script_path}')
    print(f'Wrote {controlnet_script_path}')
    print(f'Wrote {manifest_path}')
    print('Launch manually when ready:')
    print(f'  bash {base_script_path}')
    print(f'  bash {controlnet_script_path}')


if __name__ == '__main__':
    main()
