"""
Generate multi-scale trajectory data for ControlNet training.

This script generates single-agent trajectories at multiple environment scales using
the pretrained MMD model through MPD. It intentionally mirrors the parameter wiring
used by scripts/inference/inference_multi_agent.py so generated data is aligned with
the same guidance/sampling defaults used in experiments.

Usage:
    python scripts/generate_data/generate_multiscale_data.py \
        --env_id EnvConveyor2D \
        --robot_id RobotPlanarDisk \
        --model_id EnvConveyor2D-RobotPlanarDisk \
        --scales 1.0 1.1 1.2 1.3 1.4 1.5 \
        --num_trajs_per_scale 500
"""

import argparse
import os
from pathlib import Path

import torch
import yaml

from mmd.config.mmd_params import MMDParams as params
from mmd.planners.single_agent import MPD
from torch_robotics.torch_utils.seed import fix_random_seed


BASE = Path(__file__).resolve().parents[2]
TRAINED_MODELS_DIR = str(BASE / "data_trained_models")


def _build_hyperparam_overrides(args):
    candidate_overrides = {
        "weight_grad_cost_collision": args.weight_grad_cost_collision,
        "weight_grad_cost_smoothness": args.weight_grad_cost_smoothness,
        "weight_grad_cost_constraints": args.weight_grad_cost_constraints,
        "weight_grad_cost_soft_constraints": args.weight_grad_cost_soft_constraints,
        "start_guide_steps_fraction": args.start_guide_steps_fraction,
        "n_guide_steps": args.n_guide_steps,
        "n_samples": args.n_samples,
        "trajectory_duration": args.trajectory_duration,
        "n_diffusion_steps_without_noise": args.n_diffusion_steps_without_noise,
        "n_local_inference_noising_steps": args.n_local_inference_noising_steps,
        "n_local_inference_denoising_steps": args.n_local_inference_denoising_steps,
        "horizon": args.horizon,
    }
    return {k: v for k, v in candidate_overrides.items() if v is not None}


def _sample_start_goal(planner):
    n_tries = 100
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = planner.task.random_coll_free_q(n_samples=2)
        s_pos, g_pos = q_free[0], q_free[1]
        if torch.linalg.norm(s_pos - g_pos) > planner.dataset.threshold_start_goal_pos:
            start_state_pos, goal_state_pos = s_pos, g_pos
            break
    return start_state_pos, goal_state_pos


def _build_planner_args(scale):
    return {
        "model_id": params.model_id,
        "planner_alg": "mmd",
        "start_state_pos": None,
        "goal_state_pos": None,
        "use_guide_on_extra_objects_only": params.use_guide_on_extra_objects_only,
        "n_samples": params.n_samples,
        "n_local_inference_noising_steps": params.n_local_inference_noising_steps,
        "n_local_inference_denoising_steps": params.n_local_inference_denoising_steps,
        "start_guide_steps_fraction": params.start_guide_steps_fraction,
        "n_guide_steps": params.n_guide_steps,
        "n_diffusion_steps_without_noise": params.n_diffusion_steps_without_noise,
        "weight_grad_cost_collision": params.weight_grad_cost_collision,
        "weight_grad_cost_smoothness": params.weight_grad_cost_smoothness,
        "weight_grad_cost_constraints": params.weight_grad_cost_constraints,
        "weight_grad_cost_soft_constraints": params.weight_grad_cost_soft_constraints,
        "factor_num_interpolated_points_for_collision": params.factor_num_interpolated_points_for_collision,
        "trajectory_duration": params.trajectory_duration,
        "horizon": params.horizon,
        "device": params.device,
        "debug": params.debug,
        "seed": params.seed,
        "results_dir": params.results_dir,
        "trained_models_dir": TRAINED_MODELS_DIR,
        "env_scale": scale,
    }


def _save_scale_outputs(base_output_dir, scale, env_id, robot_id, model_id, trajs_free_tensor):
    save_dir = os.path.join(base_output_dir, f"scale_{scale:.2f}", "0")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(trajs_free_tensor, os.path.join(save_dir, "trajs-free.pt"))

    metadata = {
        "env_id": env_id,
        "robot_id": robot_id,
        "model_id": model_id,
        "num_trajectories": int(trajs_free_tensor.shape[0]),
        "scale": float(scale),
        "n_support_points": int(trajs_free_tensor.shape[1]),
        "duration": float(params.trajectory_duration),
    }
    with open(os.path.join(save_dir, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f)

    args_data = {
        "obstacle_cutoff_margin": 0.02,
        "threshold_start_goal_pos": 1.0,
        "env_scale": float(scale),
    }
    with open(os.path.join(save_dir, "args.yaml"), "w") as f:
        yaml.dump(args_data, f)


def generate_trajectories_for_scale(args, scale, base_output_dir):
    print("=" * 80)
    print(f"Generating trajectories for scale={scale:.2f}")
    print("=" * 80)

    planner_args = _build_planner_args(scale)
    planner = MPD(**planner_args)

    trajs_free_all = []
    num_collected = 0
    num_calls = 0
    max_calls = max(args.num_trajs_per_scale * 3, 100)

    while num_collected < args.num_trajs_per_scale and num_calls < max_calls:
        num_calls += 1

        start_state_pos, goal_state_pos = _sample_start_goal(planner)
        if start_state_pos is None or goal_state_pos is None:
            continue

        planner.start_state_pos = start_state_pos
        planner.goal_state_pos = goal_state_pos
        planner.hard_conds = planner.dataset.get_hard_conditions(
            torch.vstack((start_state_pos, goal_state_pos)),
            normalize=True,
        )

        try:
            result = planner(start_state_pos, goal_state_pos)
        except Exception as err:
            print(f"Call failed at scale={scale:.2f}, call={num_calls}: {err}")
            continue

        if result.trajs_final_free is None or result.trajs_final_free.shape[0] == 0:
            continue

        trajs_free = result.trajs_final_free.detach().cpu()
        trajs_free_all.append(trajs_free)
        num_collected += trajs_free.shape[0]
        print(
            f"scale={scale:.2f} | call={num_calls} | +{trajs_free.shape[0]} free | "
            f"collected={num_collected}/{args.num_trajs_per_scale}"
        )

    if len(trajs_free_all) == 0:
        print(f"Failed to generate trajectories for scale={scale:.2f}")
        return

    trajs_free_tensor = torch.cat(trajs_free_all, dim=0)[:args.num_trajs_per_scale]
    _save_scale_outputs(
        base_output_dir=base_output_dir,
        scale=scale,
        env_id=args.env_id,
        robot_id=args.robot_id,
        model_id=args.model_id,
        trajs_free_tensor=trajs_free_tensor,
    )
    print(f"Saved {trajs_free_tensor.shape[0]} trajectories for scale={scale:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-scale trajectories with MMDParams-aligned inference settings."
    )
    parser.add_argument("--env_id", type=str, default="EnvConveyor2D")
    parser.add_argument("--robot_id", type=str, default="RobotPlanarDisk")
    parser.add_argument("--model_id", type=str, default="EnvConveyor2D-RobotPlanarDisk")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    parser.add_argument("--num_trajs_per_scale", "--num_trajs", dest="num_trajs_per_scale", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data_trajectories")
    parser.add_argument("--seed", type=int, default=18)

    parser.add_argument("--weight_grad_cost_collision", type=float, default=None)
    parser.add_argument("--weight_grad_cost_smoothness", type=float, default=None)
    parser.add_argument("--weight_grad_cost_constraints", type=float, default=None)
    parser.add_argument("--weight_grad_cost_soft_constraints", type=float, default=None)
    parser.add_argument("--start_guide_steps_fraction", type=float, default=None)
    parser.add_argument("--n_guide_steps", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--trajectory_duration", type=float, default=None)
    parser.add_argument("--n_diffusion_steps_without_noise", type=int, default=None)
    parser.add_argument("--n_local_inference_noising_steps", type=int, default=None)
    parser.add_argument("--n_local_inference_denoising_steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)

    args = parser.parse_args()

    params.model_id = args.model_id
    params.seed = args.seed

    fix_random_seed(params.seed)

    overrides = _build_hyperparam_overrides(args)
    if len(overrides) > 0:
        print("Applying overrides:")
        params.apply_overrides(overrides)

    if not os.path.isdir(TRAINED_MODELS_DIR):
        raise FileNotFoundError(f"Trained models directory not found: {TRAINED_MODELS_DIR}")

    model_dir = os.path.join(TRAINED_MODELS_DIR, args.model_id)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    base_output_dir = os.path.join(args.output_dir, f"{args.env_id}-{args.robot_id}-multiscale")
    os.makedirs(base_output_dir, exist_ok=True)

    top_level_0 = os.path.join(base_output_dir, "0")
    os.makedirs(top_level_0, exist_ok=True)
    with open(os.path.join(top_level_0, "metadata.yaml"), "w") as f:
        yaml.dump({"env_id": args.env_id, "robot_id": args.robot_id, "is_multiscale": True}, f)
    with open(os.path.join(top_level_0, "args.yaml"), "w") as f:
        yaml.dump({"threshold_start_goal_pos": 1.0, "obstacle_cutoff_margin": 0.02, "env_scale": 1.0}, f)

    print("=" * 80)
    print("Multi-scale generation config")
    print("=" * 80)
    print(f"model_id: {args.model_id}")
    print(f"trained_models_dir: {TRAINED_MODELS_DIR}")
    print(f"scales: {args.scales}")
    print(f"num_trajs_per_scale: {args.num_trajs_per_scale}")
    print(f"n_samples (per planner call): {params.n_samples}")
    print(f"guidance: n_guide_steps={params.n_guide_steps}, start_fraction={params.start_guide_steps_fraction}")
    print(
        "weights: "
        f"collision={params.weight_grad_cost_collision}, "
        f"smoothness={params.weight_grad_cost_smoothness}, "
        f"constraints={params.weight_grad_cost_constraints}, "
        f"soft_constraints={params.weight_grad_cost_soft_constraints}"
    )
    print(f"output_dir: {base_output_dir}")
    print("=" * 80)

    for scale in args.scales:
        generate_trajectories_for_scale(args=args, scale=scale, base_output_dir=base_output_dir)

    print("Generation complete.")


if __name__ == "__main__":
    main()
