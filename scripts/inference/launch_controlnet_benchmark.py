"""
Generate benchmark shell scripts for the ControlNet Conveyor evaluation.

This launcher does not run the benchmark itself. It writes GPU-specific shell
scripts plus a tmux launcher that call `launch_controlnet_evaluation.py`, so
the user can inspect and launch long runs manually.
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
DEFAULT_TMUX_SCRIPT_PATH = DEFAULT_OUTPUT_DIR / 'launch_controlnet_benchmark_tmux.sh'
DEFAULT_EMA_CHECKPOINT = (
    BASE / 'logs_controlnet_full' / '0' / 'checkpoints' / 'ema_controlnet_final_state_dict.pth'
)
DEFAULT_BESTVAL_CHECKPOINT = (
    BASE / 'logs_controlnet_full' / '0' / 'checkpoints' / 'controlnet_epoch_0333_iter_030000_state_dict.pth'
)
DEFAULT_SDF_CACHE_DIR = (
    BASE / 'data_trained_models' / 'EnvConveyor2D-RobotPlanarDisk' / 'controlnet' / 'sdf_cache'
)

DEFAULT_SCALES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DEFAULT_NUM_AGENTS = [6, 9, 12, 15]
DEFAULT_MULTI_AGENT_PLANNERS = ['XECBS']
WINNING_HYPERPARAMS = {
    'weight_grad_cost_collision': 0.05,
    'weight_grad_cost_smoothness': 0.08,
    'weight_grad_cost_constraints': 0.2,
    'weight_grad_cost_soft_constraints': 0.01,
    'start_guide_steps_fraction': 0.35,
}
PAPER_DEFAULT_HYPERPARAMS = {
    'weight_grad_cost_collision': params.weight_grad_cost_collision,
    'weight_grad_cost_smoothness': params.weight_grad_cost_smoothness,
    'weight_grad_cost_constraints': params.weight_grad_cost_constraints,
    'weight_grad_cost_soft_constraints': params.weight_grad_cost_soft_constraints,
    'start_guide_steps_fraction': params.start_guide_steps_fraction,
}
DEFAULT_TMUX_SESSION_PREFIX = 'controlnet_benchmark'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate GPU shell scripts for the EnvConveyor2D ControlNet '
            'benchmark, including base-vs-ControlNet and ControlNet-only modes.'
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
        '--num_workers',
        type=int,
        default=0,
        help=(
            'Number of worker scripts to generate for sharding the (scale, mode) '
            'matrix. Use 0 to keep the legacy base/controlnet 2-script layout.'
        ),
    )
    parser.add_argument(
        '--worker_gpu_ids',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Optional GPU ids for worker mode. Length must match num_workers. '
            'Defaults to base/controlnet GPUs for 2 workers, otherwise 0..num_workers-1.'
        ),
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
        '--skip_base',
        action='store_true',
        help='Generate ControlNet-only scripts (skip base mode command generation)',
    )
    parser.add_argument(
        '--controlnet_use_paper_defaults',
        action='store_true',
        help=(
            'Do not pass hyperparameter override flags for ControlNet commands; '
            'this uses paper defaults from mmd_params.py.'
        ),
    )
    parser.add_argument(
        '--controlnet_ema_run_label',
        type=str,
        default=None,
        help='Optional run label for EMA ControlNet command',
    )
    parser.add_argument(
        '--controlnet_bestval_run_label',
        type=str,
        default=None,
        help='Optional run label for best-val ControlNet command',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory where generated shell scripts and manifest are written',
    )
    parser.add_argument(
        '--tmux_session_prefix',
        type=str,
        default=DEFAULT_TMUX_SESSION_PREFIX,
        help='Prefix used for generated tmux session names',
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
    if args.num_workers < 0:
        raise ValueError('num_workers must be non-negative.')

    if args.worker_gpu_ids is not None:
        if args.num_workers <= 0:
            raise ValueError('worker_gpu_ids requires num_workers > 0.')
        if len(args.worker_gpu_ids) != args.num_workers:
            raise ValueError('worker_gpu_ids length must match num_workers.')
        if len(set(args.worker_gpu_ids)) != len(args.worker_gpu_ids):
            raise ValueError('worker_gpu_ids must be unique so each worker gets one GPU.')


def get_controlnet_run_labels(args):
    if args.controlnet_use_paper_defaults:
        default_ema_label = 'controlnet_ema_v2'
        default_bestval_label = 'controlnet_bestval_v2'
    else:
        default_ema_label = 'controlnet_ema'
        default_bestval_label = 'controlnet_bestval'

    ema_label = args.controlnet_ema_run_label or default_ema_label
    bestval_label = args.controlnet_bestval_run_label or default_bestval_label
    return ema_label, bestval_label


def build_base_command(args, run_label, scales_override=None):
    scales = args.scales if scales_override is None else scales_override
    command = [
        'python',
        'launch_controlnet_evaluation.py',
        '--instance_name', args.instance_name,
        '--run_label', run_label,
        '--modes', 'base',
        '--scales', *[str(scale) for scale in scales],
        '--num_agents', *[str(num_agents) for num_agents in args.num_agents],
        '--num_trials_per_combination', str(args.num_trials_per_combination),
        '--multi_agent_planner_classes', *args.multi_agent_planner_classes,
        '--runtime_limit', str(args.runtime_limit),
        '--stagger_start_time_dt', str(args.stagger_start_time_dt),
        '--seed', str(args.seed),
    ]
    command.extend(flatten_hyperparam_args(WINNING_HYPERPARAMS))
    if not args.render_animation:
        command.append('--no_render_animation')
    return command


def build_controlnet_command(args, checkpoint_path, run_label, hyperparam_overrides=None, scales_override=None):
    scales = args.scales if scales_override is None else scales_override
    command = [
        'python',
        'launch_controlnet_evaluation.py',
        '--instance_name', args.instance_name,
        '--run_label', run_label,
        '--modes', 'controlnet',
        '--scales', *[str(scale) for scale in scales],
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
    if hyperparam_overrides is not None:
        command.extend(flatten_hyperparam_args(hyperparam_overrides))
    if not args.render_animation:
        command.append('--no_render_animation')
    return command


def flatten_hyperparam_args(hyperparams):
    arg_list = []
    for name, value in hyperparams.items():
        arg_list.extend([f'--{name}', str(value)])
    return arg_list


def shell_join(command):
    return ' '.join(shlex.quote(token) for token in command)


def resolve_worker_gpu_ids(args):
    if args.worker_gpu_ids is not None:
        return list(args.worker_gpu_ids)

    if args.num_workers == 1:
        return [args.base_gpu]
    if args.num_workers == 2:
        return [args.base_gpu, args.controlnet_gpu]
    return list(range(args.num_workers))


def build_worker_command_label(run_label, scale):
    return f'{run_label} (env_scale={scale})'


def distribute_commands_across_workers(labeled_commands, num_workers):
    worker_command_groups = [[] for _ in range(num_workers)]
    for index, labeled_command in enumerate(labeled_commands):
        worker_command_groups[index % num_workers].append(labeled_command)
    return worker_command_groups


def build_worker_mode_commands(
        args,
        ema_run_label,
        bestval_run_label,
        controlnet_hyperparam_overrides,
):
    labeled_commands = []

    for scale in args.scales:
        if not args.skip_base:
            labeled_commands.append(
                (
                    build_worker_command_label('base', scale),
                    build_base_command(args, 'base', scales_override=[scale]),
                )
            )

        labeled_commands.append(
            (
                build_worker_command_label(ema_run_label, scale),
                build_controlnet_command(
                    args,
                    args.ema_checkpoint_path,
                    ema_run_label,
                    hyperparam_overrides=controlnet_hyperparam_overrides,
                    scales_override=[scale],
                ),
            )
        )
        labeled_commands.append(
            (
                build_worker_command_label(bestval_run_label, scale),
                build_controlnet_command(
                    args,
                    args.bestval_checkpoint_path,
                    bestval_run_label,
                    hyperparam_overrides=controlnet_hyperparam_overrides,
                    scales_override=[scale],
                ),
            )
        )

    return labeled_commands


def build_script_body(gpu_id, labeled_commands):
    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        f'export CUDA_VISIBLE_DEVICES={gpu_id}',
        f'cd {shlex.quote(str(INFERENCE_DIR))}',
        '',
    ]

    total = len(labeled_commands)
    if total == 0:
        lines.extend([
            "echo ''",
            "echo 'No benchmark commands assigned to this worker.'",
            '',
        ])
        return '\n'.join(lines) + '\n'

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


def build_tmux_script(session_specs):
    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        '# ControlNet benchmark tmux launcher',
        '# Activate the desired environment before running this script so tmux inherits it.',
        '',
    ]

    for label, session_name, script_path in session_specs:
        quoted_session_name = shlex.quote(session_name)
        quoted_script_path = shlex.quote(str(script_path))
        tmux_command = (
            f'bash {quoted_script_path}; '
            "printf '\n'; "
            "echo 'Benchmark worker finished. Press Ctrl+D to close.'; "
            'exec bash'
        )
        lines.extend([
            f'if tmux has-session -t {quoted_session_name} 2>/dev/null; then',
            f"  echo 'tmux session already exists: {session_name}'",
            '  exit 1',
            'fi',
            '',
            f'tmux new-session -d -s {quoted_session_name} {shlex.quote(tmux_command)}',
            f"echo 'Started {label} benchmark in tmux session: {session_name}'",
            '',
        ])

    lines.extend([
        "echo ''",
        "echo 'All benchmark workers launched in tmux sessions.'",
        "echo 'Monitor with:'",
        "echo '  tmux ls'",
    ])

    for _, session_name, _ in session_specs:
        lines.append(f"echo '  tmux attach -t {session_name}'")

    lines.append("echo '  watch -n 1 nvidia-smi'")
    return '\n'.join(lines) + '\n'


def build_worker_tmux_script(session_name, window_specs):
    if not window_specs:
        raise ValueError('window_specs must be non-empty for worker tmux generation.')

    lines = [
        '#!/bin/bash',
        'set -euo pipefail',
        '# ControlNet benchmark tmux launcher (worker mode)',
        '# Activate the desired environment before running this script so tmux inherits it.',
        '',
    ]

    quoted_session_name = shlex.quote(session_name)
    lines.extend([
        f'if tmux has-session -t {quoted_session_name} 2>/dev/null; then',
        f"  echo 'tmux session already exists: {session_name}'",
        '  exit 1',
        'fi',
        '',
    ])

    for index, (window_name, label, script_path) in enumerate(window_specs):
        quoted_script_path = shlex.quote(str(script_path))
        quoted_window_name = shlex.quote(window_name)
        tmux_command = (
            f'bash {quoted_script_path}; '
            "printf '\n'; "
            "echo 'Benchmark worker finished. Press Ctrl+D to close.'; "
            'exec bash'
        )
        if index == 0:
            lines.extend([
                f'tmux new-session -d -s {quoted_session_name} -n {quoted_window_name} {shlex.quote(tmux_command)}',
                f"echo 'Started {label} in tmux session: {session_name} (window: {window_name})'",
                '',
            ])
        else:
            lines.extend([
                f'tmux new-window -t {quoted_session_name} -n {quoted_window_name} {shlex.quote(tmux_command)}',
                f"echo 'Started {label} in tmux session: {session_name} (window: {window_name})'",
                '',
            ])

    lines.extend([
        f'tmux select-window -t {quoted_session_name}:0',
        "echo ''",
        "echo 'All benchmark workers launched in tmux windows.'",
        "echo 'Monitor with:'",
        "echo '  tmux ls'",
        f"echo '  tmux attach -t {session_name}'",
        "echo '  watch -n 1 nvidia-smi'",
    ])
    return '\n'.join(lines) + '\n'


def build_manifest(
        args,
        script_paths,
        tmux_script_path,
        labeled_commands,
        controlnet_hyperparam_overrides,
        tmux_info_lines,
        tmux_attach_commands,
        direct_launch_commands,
        worker_gpu_ids,
):
    trials_per_mode = len(args.scales) * len(args.num_agents) * args.num_trials_per_combination
    total_base_trials = 0 if args.skip_base else trials_per_mode
    total_controlnet_trials = trials_per_mode * 2
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
        f'Skip base mode: {args.skip_base}',
        f'Num workers: {args.num_workers}',
        f'Worker GPU ids: {worker_gpu_ids}',
        '',
        'Base hyperparameters (when base mode is included):',
    ]

    for name, value in WINNING_HYPERPARAMS.items():
        lines.append(f'- {name}: {value}')

    lines.append('')
    if controlnet_hyperparam_overrides is None:
        lines.append('ControlNet hyperparameters: paper defaults (no CLI overrides)')
        for name, value in PAPER_DEFAULT_HYPERPARAMS.items():
            lines.append(f'- {name}: {value}')
    else:
        lines.append('ControlNet hyperparameters: explicit CLI overrides')
        for name, value in controlnet_hyperparam_overrides.items():
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
    lines.append(f'- {tmux_script_path}')

    lines.extend([
        '',
        'tmux launch info:',
    ])

    for info_line in tmux_info_lines:
        lines.append(f'- {info_line}')

    lines.extend([
        '',
        'Commands:',
    ])

    for label, command in labeled_commands:
        lines.append(f'- {label}: {shell_join(command)}')

    lines.extend([
        '',
        'Launch manually with tmux (recommended):',
        f'- bash {tmux_script_path}',
    ])

    for attach_command in tmux_attach_commands:
        lines.append(f'- {attach_command}')

    lines.extend([
        '',
        'Launch manually without tmux:',
    ])

    for direct_launch_command in direct_launch_commands:
        lines.append(f'- {direct_launch_command}')

    return '\n'.join(lines) + '\n'


def print_summary(
        args,
        script_paths,
        tmux_script_path,
        labeled_commands,
        controlnet_hyperparam_overrides,
        tmux_info_lines,
        worker_gpu_ids,
):
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
    print(f'Skip base mode: {args.skip_base}')
    print(f'Num workers: {args.num_workers}')
    print(f'Worker GPU ids: {worker_gpu_ids}')
    print('Winning hyperparameters:')
    for name, value in WINNING_HYPERPARAMS.items():
        print(f'  {name}={value}')
    if controlnet_hyperparam_overrides is None:
        print('ControlNet hyperparameters: paper defaults (no CLI overrides)')
        for name, value in PAPER_DEFAULT_HYPERPARAMS.items():
            print(f'  {name}={value}')
    else:
        print('ControlNet hyperparameters: explicit CLI overrides')
        for name, value in controlnet_hyperparam_overrides.items():
            print(f'  {name}={value}')
    print('Generated commands:')
    for label, command in labeled_commands:
        print(f'  {label}: {shell_join(command)}')
    if script_paths:
        print('Generated scripts:')
        for script_path in script_paths:
            print(f'  {script_path}')
        print(f'  {tmux_script_path}')
    print('tmux launch info:')
    for info_line in tmux_info_lines:
        print(f'  {info_line}')
    print('=' * 80 + '\n')


def main():
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / DEFAULT_MANIFEST_PATH.name
    tmux_script_path = output_dir / DEFAULT_TMUX_SCRIPT_PATH.name

    controlnet_hyperparam_overrides = None
    if not args.controlnet_use_paper_defaults:
        controlnet_hyperparam_overrides = WINNING_HYPERPARAMS

    ema_run_label, bestval_run_label = get_controlnet_run_labels(args)
    script_specs = []
    labeled_commands = []
    tmux_info_lines = []
    tmux_attach_commands = []
    direct_launch_commands = []
    tmux_script_body = None
    worker_gpu_ids = []

    if args.num_workers > 0:
        labeled_commands = build_worker_mode_commands(
            args,
            ema_run_label,
            bestval_run_label,
            controlnet_hyperparam_overrides,
        )

        if args.num_workers > len(labeled_commands):
            raise ValueError(
                f'num_workers={args.num_workers} exceeds the number of (scale, mode) work items={len(labeled_commands)}.'
            )

        worker_gpu_ids = resolve_worker_gpu_ids(args)
        worker_command_groups = distribute_commands_across_workers(labeled_commands, args.num_workers)
        window_specs = []
        for worker_index, (gpu_id, worker_commands) in enumerate(zip(worker_gpu_ids, worker_command_groups)):
            script_path = output_dir / f'controlnet_benchmark_worker_{worker_index}.sh'
            script_specs.append((script_path, gpu_id, worker_commands))
            window_name = f'worker_{worker_index}_gpu_{gpu_id}'
            window_specs.append((window_name, f'worker_{worker_index}', script_path))
            command_labels = ', '.join(label for label, _ in worker_commands)
            tmux_info_lines.append(f'worker_{worker_index}: gpu={gpu_id}, window={window_name}, commands=[{command_labels}]')
            direct_launch_commands.append(f'bash {script_path}')

        tmux_session_name = f'{args.tmux_session_prefix}_workers'
        tmux_attach_commands.append(f'tmux attach -t {tmux_session_name}')
        tmux_script_body = build_worker_tmux_script(tmux_session_name, window_specs)
    else:
        worker_gpu_ids = [args.base_gpu, args.controlnet_gpu]
        base_script_path = output_dir / 'controlnet_benchmark_gpu0.sh'
        controlnet_script_path = output_dir / 'controlnet_benchmark_gpu1.sh'

        controlnet_ema_command = build_controlnet_command(
            args,
            args.ema_checkpoint_path,
            ema_run_label,
            hyperparam_overrides=controlnet_hyperparam_overrides,
        )
        controlnet_bestval_command = build_controlnet_command(
            args,
            args.bestval_checkpoint_path,
            bestval_run_label,
            hyperparam_overrides=controlnet_hyperparam_overrides,
        )

        tmux_session_specs = []
        if args.skip_base:
            script_specs = [
                (base_script_path, args.base_gpu, [(ema_run_label, controlnet_ema_command)]),
                (controlnet_script_path, args.controlnet_gpu, [(bestval_run_label, controlnet_bestval_command)]),
            ]
            labeled_commands = [
                (ema_run_label, controlnet_ema_command),
                (bestval_run_label, controlnet_bestval_command),
            ]
            tmux_session_specs = [
                ('controlnet_ema', f'{args.tmux_session_prefix}_ema_gpu{args.base_gpu}', base_script_path),
                (
                    'controlnet_bestval',
                    f'{args.tmux_session_prefix}_bestval_gpu{args.controlnet_gpu}',
                    controlnet_script_path,
                ),
            ]
        else:
            base_command = build_base_command(args, 'base')
            script_specs = [
                (base_script_path, args.base_gpu, [('base', base_command)]),
                (
                    controlnet_script_path,
                    args.controlnet_gpu,
                    [(ema_run_label, controlnet_ema_command), (bestval_run_label, controlnet_bestval_command)],
                ),
            ]
            labeled_commands = [
                ('base', base_command),
                (ema_run_label, controlnet_ema_command),
                (bestval_run_label, controlnet_bestval_command),
            ]
            tmux_session_specs = [
                ('base', f'{args.tmux_session_prefix}_base_gpu{args.base_gpu}', base_script_path),
                (
                    'controlnet',
                    f'{args.tmux_session_prefix}_controlnet_gpu{args.controlnet_gpu}',
                    controlnet_script_path,
                ),
            ]

        tmux_script_body = build_tmux_script(tmux_session_specs)
        for label, session_name, script_path in tmux_session_specs:
            tmux_info_lines.append(f'{label}: session={session_name}, script={script_path}')
            tmux_attach_commands.append(f'tmux attach -t {session_name}')
        for script_path, _, _ in script_specs:
            direct_launch_commands.append(f'bash {script_path}')

    script_paths = [script_path for script_path, _, _ in script_specs]

    print_summary(
        args,
        script_paths,
        tmux_script_path,
        labeled_commands,
        controlnet_hyperparam_overrides,
        tmux_info_lines,
        worker_gpu_ids,
    )

    if args.dry_run:
        print('Dry run only: no files were written.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for script_path, gpu_id, worker_labeled_commands in script_specs:
        script_path.write_text(
            build_script_body(gpu_id, worker_labeled_commands),
            encoding='ascii',
        )

    tmux_script_path.write_text(
        tmux_script_body,
        encoding='ascii',
    )
    manifest_path.write_text(
        build_manifest(
            args,
            script_paths,
            tmux_script_path,
            labeled_commands,
            controlnet_hyperparam_overrides,
            tmux_info_lines,
            tmux_attach_commands,
            direct_launch_commands,
            worker_gpu_ids,
        ),
        encoding='ascii',
    )

    for script_path in script_paths:
        script_path.chmod(0o755)
    tmux_script_path.chmod(0o755)

    for script_path in script_paths:
        print(f'Wrote {script_path}')
    print(f'Wrote {tmux_script_path}')
    print(f'Wrote {manifest_path}')
    print('Launch manually when ready (recommended):')
    print(f'  bash {tmux_script_path}')
    print('Attach to workers with:')
    for attach_command in tmux_attach_commands:
        print(f'  {attach_command}')
    print('Direct non-tmux launch is still available:')
    for direct_launch_command in direct_launch_commands:
        print(f'  {direct_launch_command}')


if __name__ == '__main__':
    main()
