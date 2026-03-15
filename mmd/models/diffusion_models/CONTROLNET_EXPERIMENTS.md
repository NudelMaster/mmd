# ControlNet Experiments for MMD Trajectory Diffusion

This document records the benchmark plan, benchmark results, stress tests, and follow-up conclusions for the ControlNet integration described in `TEMPORAL_UNET_ARCHITECTURE.md`.
For the adapter design, deferred alternatives, training setup, and inference wiring, see `TEMPORAL_UNET_ARCHITECTURE.md` Sections 4, 6, 7, and 13.

**Target**: EnvConveyor2D-RobotPlanarDisk

---

## 1. Benchmark Evaluation Plan (Best Hyperparameter Baseline vs ControlNet)

This section records the planned benchmark used to compare the trained ControlNet adapter against the strongest known non-ControlNet inference configuration for scaled Conveyor environments.

### Purpose

The benchmark goal is not to prove ControlNet must help. It is to measure whether the current FiLM-based adapter changes multi-agent planning quality in a meaningful way when compared against the best hyperparameter setting already found by grid search for scaled environments.

This is an intentionally conservative comparison:

- same planning problem family
- same random seeds
- same start/goal generation per scale
- same guidance hyperparameters
- same runtime limit
- only the presence/absence of ControlNet changes

### Baseline Configuration to Beat

From the completed grid search and result summaries (`scripts/inference/best_hyperparameter_configs.txt`), the strongest average configuration for scaled Conveyor experiments was:

| Parameter | Value |
|-----------|-------|
| `weight_grad_cost_collision` | `0.05` |
| `weight_grad_cost_smoothness` | `0.08` |
| `weight_grad_cost_constraints` | `0.2` |
| `weight_grad_cost_soft_constraints` | `0.01` |
| `start_guide_steps_fraction` | `0.35` |

This configuration will be used unchanged for both:

1. base MPD benchmark runs
2. ControlNet-augmented MPD benchmark runs

### Benchmark Scope

The experiment matrix is:

| Dimension | Values |
|-----------|--------|
| Planning problem | `EnvConveyor2DRobotPlanarDiskRandom` |
| Scales | `1.0, 1.1, 1.2, 1.3, 1.4` |
| Agent counts | `6, 9, 12, 15` |
| Trials per combination | `10` |
| Runtime limit | `180 s` |
| Base modes | `base`, `controlnet` |
| Control strength | `control_scale = 1.0` |

Scale `1.5` is intentionally excluded from this first benchmark sweep because prior experiments already showed near-universal failure there, making it a poor first comparison point.

### ControlNet Checkpoints to Evaluate

Two ControlNet checkpoints will be compared against the same base baseline:

1. `logs_controlnet_full/0/checkpoints/ema_controlnet_final_state_dict.pth`
2. `logs_controlnet_full/0/checkpoints/controlnet_epoch_0333_iter_030000_state_dict.pth`

Rationale:

- the EMA final checkpoint is the default inference choice
- the `iter_030000` checkpoint is closest to the best validation region and may generalize better if late-training overfitting matters

### Total Trial Count

Per planner variant:

- `5 scales x 4 agent counts x 10 trials = 200 trials`

Full benchmark total:

- base MPD: `200`
- ControlNet + EMA: `200`
- ControlNet + best-val checkpoint: `200`
- overall total: `600 trials`

### Why the Comparison Must Re-run the Base Mode

Although prior baseline data already exists from grid search, the benchmark should re-run the base planner in the same launcher used for ControlNet so that:

- `fix_random_seed(...)` is applied identically before each `(scale, mode)` pair
- base and ControlNet see the same generated start/goal sets
- differences in results are attributable to the model path, not to different sampled planning problems

This makes the benchmark a paired comparison rather than a loose comparison across historical runs.

### Expected Outcome / Hypothesis

Current expectations remain cautious.

The implemented ControlNet uses a **global FiLM-style conditioning signal** (`sdf_emb` added to the time embedding), not cross-attention or per-waypoint spatial conditioning. That means the conditioning is much less expressive than original image-domain ControlNet, where the condition and latent share strong spatial correspondence.

So the most likely outcomes, in order of expectation, are:

1. **neutral or slightly worse** than the tuned base configuration
2. **small gains at larger scales** (`1.3` to `1.4`) if the global SDF signal helps obstacle-awareness enough to reduce CBS burden
3. **clear gains across all scales**, which would be surprising but valuable if observed

Even a negative result is useful here because it would support the hypothesis that global FiLM conditioning is too weak for this cross-domain setting.

### Metrics to Compare

The benchmark will compare the standard aggregated experiment outputs already produced by the MAPF pipeline:

- `success_rate`
- `avg_planning_time`
- `avg_ct_expansions`
- `avg_num_collisions_in_solution`
- `avg_path_length_per_agent`
- `avg_mean_path_acceleration_per_agent`
- `avg_data_adherence`
- failure rates (`runtime_limit`, `no_solution`, `collision_agents`)

Primary decision metric:

- `success_rate` as a function of `env_scale` and `num_agents`

Secondary interpretation metrics:

- planning time and CT expansions (does ControlNet reduce CBS search burden?)
- path quality metrics (does it change trajectory realism/smoothness?)

### Launcher Design Decision

The benchmark launcher should **reuse existing code** rather than duplicating experiment logic.

Existing scripts reviewed:

- `scripts/inference/launch_grid_search.py`
  - reused as the pattern for generating shell scripts and batching experiment commands
- `scripts/inference/launch_controlnet_evaluation.py`
  - reused as the actual execution entry point for base-vs-ControlNet comparisons
- `scripts/inference/launch_multi_agent_experiment.py`
  - reused indirectly through `launch_controlnet_evaluation.py`

Decision:

- add a new thin script: `scripts/inference/launch_controlnet_benchmark.py`
- it should generate benchmark shell scripts rather than re-implement the experiment runner
- full benchmark launching remains manual/user-controlled

### Planned GPU Distribution

Machine state at planning time: `2 x A100 80GB` available.

Recommended split:

- **GPU 0**: base benchmark (`200` trials)
- **GPU 1**: ControlNet benchmark (`400` trials), running:
  1. EMA checkpoint
  2. best-val checkpoint

This avoids memory contention while still parallelizing the full sweep.

### Benchmark Script Behavior

The benchmark launcher should generate scripts under `scripts/inference/gpu_scripts/` and point all experiment outputs to the shared inference results root:

- `scripts/inference/results/`

Expected generated shell scripts:

- `scripts/inference/gpu_scripts/controlnet_benchmark_gpu0.sh`
- `scripts/inference/gpu_scripts/controlnet_benchmark_gpu1.sh`

The generated commands should call `launch_controlnet_evaluation.py` with:

- fixed winning hyperparameters
- `--num_agents 6 9 12 15`
- `--scales 1.0 1.1 1.2 1.3 1.4`
- `--num_trials_per_combination 10`
- `--runtime_limit 180`
- `--modes base` or `--modes controlnet`
- `--control_scale 1.0`
- the selected checkpoint path for ControlNet runs

### Planned Smoke Validation Before Full Launch

Only smoke validation should be executed automatically before handing the full launcher to the user.

Recommended smoke checks:

1. launcher dry-run / script-generation check
2. syntax check of the new launcher
3. optional tiny benchmark smoke command with reduced scope:
   - one scale
   - one agent count
   - one trial

The full benchmark itself should be started manually by the user after inspecting the generated scripts.

### Benchmark Launcher Implementation (Completed)

The benchmark launcher has now been added at:

- `scripts/inference/launch_controlnet_benchmark.py`

Implementation notes:

- it reuses `launch_controlnet_evaluation.py` as the execution entry point
- it hardcodes the winning Conveyor hyperparameters as the benchmark default
- it writes GPU-specific shell scripts instead of launching experiments automatically
- it now also writes a tmux launcher so the long benchmark can be launched the same way as the earlier hyperparameter sweeps
- it writes a manifest file summarizing the full benchmark matrix, checkpoints, and exact commands
- GIF rendering is disabled by default via `--no_render_animation` to keep the benchmark focused on metrics rather than visualization overhead

Generated artifacts:

- `scripts/inference/gpu_scripts/controlnet_benchmark_gpu0.sh`
- `scripts/inference/gpu_scripts/controlnet_benchmark_gpu1.sh`
- `scripts/inference/gpu_scripts/launch_controlnet_benchmark_tmux.sh`
- `scripts/inference/gpu_scripts/controlnet_benchmark_manifest.txt`

GPU role split encoded by the generated scripts:

- `controlnet_benchmark_gpu0.sh` -> base benchmark
- `controlnet_benchmark_gpu1.sh` -> ControlNet benchmark (EMA then best-val checkpoint)
- `launch_controlnet_benchmark_tmux.sh` -> detached tmux launcher that starts one session per GPU worker and keeps the shell open after completion for inspection

### Smoke Validation Executed for the Launcher

The following smoke checks were run after implementing the launcher:

#### 1. Python syntax check

```bash
python3 -m py_compile scripts/inference/launch_controlnet_benchmark.py
```

Result:

- passed

#### 2. Dry-run generation check

```bash
python3 scripts/inference/launch_controlnet_benchmark.py --dry_run
```

Result:

- printed the expected 3 benchmark commands
- confirmed benchmark scope: scales `[1.0, 1.1, 1.2, 1.3, 1.4]`, agents `[6, 9, 12, 15]`, `10` trials per combination
- confirmed base / ControlNet command separation
- confirmed `--no_render_animation` is passed by default

#### 3. Actual script-generation smoke run

```bash
python3 scripts/inference/launch_controlnet_benchmark.py
```

Result:

- generated both GPU shell scripts successfully
- generated the tmux launcher script successfully
- generated the benchmark manifest successfully
- did **not** launch the full benchmark itself

#### 4. Generated shell syntax check

```bash
bash -n scripts/inference/gpu_scripts/controlnet_benchmark_gpu0.sh && \
bash -n scripts/inference/gpu_scripts/controlnet_benchmark_gpu1.sh && \
bash -n scripts/inference/gpu_scripts/launch_controlnet_benchmark_tmux.sh
```

Result:

- all generated shell scripts passed syntax validation

### Manual Launch Commands

When ready to run the full benchmark manually:

```bash
bash scripts/inference/gpu_scripts/launch_controlnet_benchmark_tmux.sh
tmux attach -t controlnet_benchmark_gpu0
tmux attach -t controlnet_benchmark_gpu1
```

If tmux is not desired, the direct scripts still work:

```bash
bash scripts/inference/gpu_scripts/controlnet_benchmark_gpu0.sh
bash scripts/inference/gpu_scripts/controlnet_benchmark_gpu1.sh
```

These commands intentionally remain manual/user-triggered.

### Benchmark Launch Fix: Invalid Random Start/Goal States (Completed)

After the first tmux benchmark launch, both workers could terminate early with:

```text
ValueError: Start or goal states are invalid.
```

Root cause:

- `EnvConveyor2DRobotPlanarDiskRandom` sampled random positions with `obstacle_margin=0.08`
- the benchmark now uses single-agent `MPD`, whose task loader hardcodes `obstacle_cutoff_margin=0.05`
- with `RobotPlanarDisk` collision margin `0.055`, the effective MPD safety margin is about `0.105`, so some sampled states that were acceptable for earlier `MPDEnsemble`-style experiments were rejected by CBS startup validation under `MPD`

Fix decision after deeper review:

- the first attempted fix added retry/validation logic in `mmd/common/multi_agent_utils.py` and tighter sampling in `mmd/config/mmd_experiment_configs.py`
- that approach was intentionally **reverted** because it changed shared random problem generation behavior, which would weaken comparability with the earlier benchmark runs
- the real mismatch was not the sampler itself; it was that the benchmark now used `MPD` while the earlier grid-search experiments used `MPDEnsemble`
- `MPDEnsemble` loads its task with `obstacle_cutoff_margin=0.01`, while `MPD` used `0.05`

Final minimal fix kept in code:

- `mmd/planners/single_agent/mpd.py`
  - changed `obstacle_cutoff_margin` from `0.05` to `0.01`
- `mmd/common/multi_agent_utils.py`
  - reverted to the original implementation (no whole-set retry logic kept)
- `mmd/config/mmd_experiment_configs.py`
  - reverted to the original Conveyor random sampling path (no special validation hook kept)

Why this matches the earlier benchmark behavior:

- `RobotPlanarDisk` contributes collision margin `0.055`
- with `MPD` at `0.05`, the effective task collision margin was about `0.105`
- with `MPDEnsemble` at `0.01`, the effective task collision margin is about `0.065`
- after the one-line `MPD` change, CBS start/goal validation and obstacle guidance now use the same margin convention as `MPDEnsemble`, which is the behavior the earlier grid search actually measured

Smoke validation executed after the final minimal fix:

```bash
python3 scripts/inference/launch_controlnet_evaluation.py \
  --modes base \
  --scales 1.2 \
  --num_agents 6 \
  --num_trials_per_combination 1 \
  --runtime_limit 60 \
  --no_render_animation \
  --weight_grad_cost_collision 0.05 \
  --weight_grad_cost_smoothness 0.08 \
  --weight_grad_cost_constraints 0.2 \
  --weight_grad_cost_soft_constraints 0.01 \
  --start_guide_steps_fraction 0.35

python3 scripts/inference/launch_controlnet_evaluation.py \
  --modes controlnet \
  --scales 1.2 \
  --num_agents 6 \
  --num_trials_per_combination 1 \
  --runtime_limit 60 \
  --no_render_animation \
  --weight_grad_cost_collision 0.05 \
  --weight_grad_cost_smoothness 0.08 \
  --weight_grad_cost_constraints 0.2 \
  --weight_grad_cost_soft_constraints 0.01 \
  --start_guide_steps_fraction 0.35
```

Observed result:

- base smoke run completed successfully at `env_scale=1.2`, `num_agents=6`, `success_rate=1.0`, `avg_planning_time=10.91s`, `avg_ct_expansions=1.0`
- ControlNet smoke run completed successfully at `env_scale=1.2`, `num_agents=6`, `success_rate=1.0`, `avg_planning_time=11.31s`, `avg_ct_expansions=1.0`
- no `Start or goal states are invalid` failure occurred in either mode
- the existing benchmark launcher scripts remain valid; no regeneration was required for this fix

### Benchmark Result Folder Identification Metadata (Completed)

The timestamped results folders were hard to interpret after the benchmark because the directory name itself did not encode whether a run was:

- `base`
- `controlnet_ema`
- `controlnet_bestval`

Minimal fix implemented:

- `scripts/inference/launch_controlnet_evaluation.py`
  - added optional `--run_label`
  - writes `run_metadata.json` into each run root at:
    - `scripts/inference/results/<time_str>/instance_name___.../run_metadata.json`
  - metadata includes:
    - `run_label`
    - `mode`
    - `time_str`
    - `results_dir`
    - ControlNet checkpoint info (when applicable)
    - full CLI args / `argv`
- `scripts/inference/launch_controlnet_benchmark.py`
  - now generates benchmark commands with explicit labels:
    - `--run_label base`
    - `--run_label controlnet_ema`
    - `--run_label controlnet_bestval`
  - regenerated shell scripts and manifest now preserve that labeling end-to-end

Practical consequence:

- after the benchmark finishes, open any timestamped result folder and inspect `run_metadata.json` to identify exactly which experiment produced it
- this avoids relying on timestamp ordering or tmux scrollback

Smoke validation command for metadata:

```bash
python3 scripts/inference/launch_controlnet_evaluation.py \
  --instance_name EnvConveyor2DRobotPlanarDiskRandom \
  --run_label metadata_smoke \
  --modes base \
  --scales 1.2 \
  --num_agents 3 \
  --num_trials_per_combination 1 \
  --multi_agent_planner_classes XECBS \
  --runtime_limit 60 \
  --stagger_start_time_dt 0 \
  --seed 18 \
  --no_render_animation
```

Observed result:

- the run completed successfully
- `run_metadata.json` was written to the run root
- the metadata correctly recorded `run_label=metadata_smoke`, `mode=base`, and the generated `time_str`

Status update:

- the full benchmark was launched and completed after this setup
- results and analysis are recorded in Section 2

---

## 2. First Benchmark Results (Completed, But ControlNet Hyperparameters Were Wrong)

This section documents the completed 600-trial benchmark run from the result folders starting at `2026-03-07-21-06-50`.

### Important Experimental Caveat

All three modes used the same **grid-search-winning** hyperparameters:

- `weight_grad_cost_collision=0.05`
- `weight_grad_cost_smoothness=0.08`
- `weight_grad_cost_constraints=0.2`
- `weight_grad_cost_soft_constraints=0.01`
- `start_guide_steps_fraction=0.35`

This is valid for the base baseline, but it is **not** the intended setup for ControlNet isolation. ControlNet runs should use paper defaults (Section 3).

### Run Mapping

- `evaluation_id=2026-03-07-21-06-50`, PID `1857416` -> `run_label=base`
- `evaluation_id=2026-03-07-21-06-50`, PID `1857417` -> `run_label=controlnet_ema`
- `evaluation_id=2026-03-08-00-16-44`, PID `1881242` -> `run_label=controlnet_bestval`

Each mode includes scales `1.0..1.4`, agent counts `6/9/12/15`, and `10` trials per combination.

### Result Folder Completeness Check

The benchmark artifacts are complete and usable:

- `15/15` run folders contain `run_metadata.json`
- `15/15` run folders contain `aggregated_results_all_agents.csv`
- `15/15` run folders contain all four per-agent CSVs (`6/9/12/15`)
- Trial outputs exist for all combinations; one visualization-only artifact is missing:
  - `base`, `env_scale=1.4`, `num_agents=15`, trial `5` missing `mmd_single_trial.gif.png`
  - numeric trial outputs (`results.pkl`, `results.txt`, `config.pkl`) are present

### 15-Agent Comparison by Scale

| Mode | Scale | Success | Avg CT Expansions | Avg Planning Time (s) | Avg Path Length/Agent | Avg Acceleration/Agent |
|------|-------|---------|-------------------|------------------------|-----------------------|------------------------|
| base | 1.0 | 1.00 | 18.8 | 56.57 | 3.683 | 0.163 |
| base | 1.1 | 1.00 | 15.1 | 50.35 | 3.629 | 0.155 |
| base | 1.2 | 1.00 | 14.6 | 47.83 | 3.906 | 0.160 |
| base | 1.3 | 1.00 | 26.4 | 66.78 | 4.321 | 0.205 |
| base | 1.4 | 0.90 | 31.1 | 72.63 | 4.391 | 0.215 |
| controlnet_ema | 1.0 | 1.00 | 11.3 | 44.55 | 3.465 | 0.144 |
| controlnet_ema | 1.1 | 1.00 | 9.6 | 42.13 | 3.566 | 0.147 |
| controlnet_ema | 1.2 | 1.00 | 11.0 | 45.07 | 3.779 | 0.156 |
| controlnet_ema | 1.3 | 1.00 | 15.1 | 53.34 | 3.974 | 0.172 |
| controlnet_ema | 1.4 | 1.00 | 15.8 | 56.37 | 4.046 | 0.175 |
| controlnet_bestval | 1.0 | 1.00 | 11.2 | 44.22 | 3.518 | 0.149 |
| controlnet_bestval | 1.1 | 1.00 | 12.3 | 47.83 | 3.717 | 0.162 |
| controlnet_bestval | 1.2 | 1.00 | 10.9 | 44.78 | 3.808 | 0.157 |
| controlnet_bestval | 1.3 | 1.00 | 11.7 | 46.54 | 3.788 | 0.156 |
| controlnet_bestval | 1.4 | 1.00 | 15.5 | 53.37 | 4.033 | 0.177 |

### Cross-Scale Means (All 20 combinations per mode)

| Mode | Mean Success | Mean CT Expansions | Mean Planning Time (s) | Mean Path Length/Agent | Mean Acceleration/Agent | Mean Runtime-Limit Fail Rate |
|------|--------------|--------------------|------------------------|------------------------|-------------------------|------------------------------|
| base | 0.995 | 9.26 | 30.22 | 3.680 | 0.146 | 0.005 |
| controlnet_ema | 1.000 | 6.24 | 27.53 | 3.615 | 0.142 | 0.000 |
| controlnet_bestval | 1.000 | 6.05 | 27.26 | 3.616 | 0.142 | 0.000 |

### What This First Benchmark Shows

Even with the wrong ControlNet hyperparameter policy, the signal is consistent:

1. ControlNet modes require fewer CBS CT expansions (especially at scales `1.3` and `1.4`).
2. ControlNet modes reduce planning time at higher scales.
3. Base mode has one runtime-limit failure at `scale=1.4, agents=15`; both ControlNet modes remain at 100% success there.

Concrete high-scale examples (15-agent rows):

- At `scale=1.4`, CT expansions: base `31.1` vs EMA `15.8` vs best-val `15.5` (about 2x lower for ControlNet).
- At `scale=1.4`, planning time: base `72.63s` vs EMA `56.37s` vs best-val `53.37s`.

Interpretation caution: this benchmark still mixes two variables for ControlNet modes (adapter + non-default guidance settings), so it is not the final causal comparison.

---

## 3. Corrected ControlNet Re-Run Results (Completed)

This section documents the corrected 400-trial ControlNet-only rerun that was executed after Section 2.

### Corrected Protocol

- Modes: `controlnet_ema_v2`, `controlnet_bestval_v2`
- Scales: `1.0 1.1 1.2 1.3 1.4`
- Agents: `6 9 12 15`
- Trials per combination: `10`
- Runtime limit: `180`
- Total corrected trials: `400` (`2 x 5 x 4 x 10`)

### Run Mapping

| Run label | PID | Number of folders | Scales covered |
|-----------|-----|-------------------|----------------|
| `controlnet_ema_v2` | `1985123` | 5 | `1.0..1.4` |
| `controlnet_bestval_v2` | `1985124` | 5 | `1.0..1.4` |

### Default-Hyperparameter Verification

All 10 corrected folders contain `run_metadata.json` with:

- `hyperparam_overrides: null`
- no `--weight_grad_cost_*` override flags in `argv`
- no `--start_guide_steps_fraction` override in `argv`

Therefore, corrected runs use default guidance values from `mmd/config/mmd_params.py`:

- `weight_grad_cost_collision=0.02`
- `weight_grad_cost_smoothness=0.08`
- `weight_grad_cost_constraints=0.2`
- `weight_grad_cost_soft_constraints=0.02`
- `start_guide_steps_fraction=0.5`

### 15-Agent Comparison by Scale (Corrected Runs)

| Mode | Scale | Success | Avg CT Expansions | Avg Planning Time (s) | Avg Path Length/Agent | Avg Acceleration/Agent |
|------|-------|---------|-------------------|------------------------|-----------------------|------------------------|
| controlnet_ema_v2 | 1.0 | 1.00 | 9.3 | 43.58 | 3.353 | 0.138 |
| controlnet_ema_v2 | 1.1 | 1.00 | 9.2 | 42.89 | 3.400 | 0.136 |
| controlnet_ema_v2 | 1.2 | 1.00 | 8.6 | 42.42 | 3.601 | 0.139 |
| controlnet_ema_v2 | 1.3 | 1.00 | 10.4 | 44.53 | 3.783 | 0.152 |
| controlnet_ema_v2 | 1.4 | 1.00 | 19.5 | 58.06 | 4.042 | 0.176 |
| controlnet_bestval_v2 | 1.0 | 1.00 | 9.2 | 42.45 | 3.381 | 0.140 |
| controlnet_bestval_v2 | 1.1 | 1.00 | 9.0 | 42.20 | 3.450 | 0.140 |
| controlnet_bestval_v2 | 1.2 | 1.00 | 10.3 | 44.90 | 3.645 | 0.145 |
| controlnet_bestval_v2 | 1.3 | 1.00 | 10.3 | 44.17 | 3.834 | 0.159 |
| controlnet_bestval_v2 | 1.4 | 1.00 | 10.7 | 44.83 | 3.846 | 0.156 |

### Cross-Scale Means (All 20 combinations per mode)

| Mode | Mean Success | Mean CT Expansions | Mean Planning Time (s) | Mean Path Length/Agent | Mean Acceleration/Agent | Mean Runtime-Limit Fail Rate |
|------|--------------|--------------------|------------------------|------------------------|-------------------------|------------------------------|
| controlnet_ema_v2 | 1.000 | 5.16 | 27.85 | 3.503 | 0.133 | 0.000 |
| controlnet_bestval_v2 | 1.000 | 4.84 | 27.14 | 3.502 | 0.134 | 0.000 |

### What the Corrected Re-Run Shows

1. Both corrected modes maintain `100%` success across all scales and agent counts.
2. Both corrected modes retain the main Section 2 signal: lower CT expansions and lower planning time than the strong base reference.
3. `controlnet_bestval_v2` is more stable at the hardest setting (`scale=1.4`, `agents=15`): CT expansions `10.7` vs `19.5` for `controlnet_ema_v2`.
4. The corrected rerun removes the hyperparameter-confound for ControlNet modes and becomes the primary ControlNet-vs-ControlNet checkpoint comparison.

---

## 4. Complete Experiment Summary (All Benchmark Runs)

This section consolidates all benchmark results from:

- Section 2: first 600-trial run (`base`, `controlnet_ema`, `controlnet_bestval`) with grid-search-winning guidance overrides
- Section 3: corrected 400-trial run (`controlnet_ema_v2`, `controlnet_bestval_v2`) using defaults from `mmd/config/mmd_params.py`

### Caveat on Baseline Comparability

The corrected rerun intentionally did not include `base_v2` (base mode with default guidance). For final interpretation, we therefore use the Section 2 base run as the reference baseline and explicitly keep this caveat in mind.

This is still a conservative baseline choice, because Section 2 base uses the strongest known non-ControlNet hyperparameter setting from grid search.

### 15-Agent CT Expansions by Scale (All Variants)

| Scale | base | controlnet_ema | controlnet_bestval | controlnet_ema_v2 | controlnet_bestval_v2 |
|-------|------|----------------|--------------------|-------------------|-----------------------|
| 1.0 | 18.8 | 11.3 | 11.2 | 9.3 | 9.2 |
| 1.1 | 15.1 | 9.6 | 12.3 | 9.2 | 9.0 |
| 1.2 | 14.6 | 11.0 | 10.9 | 8.6 | 10.3 |
| 1.3 | 26.4 | 15.1 | 11.7 | 10.4 | 10.3 |
| 1.4 | 31.1 | 15.8 | 15.5 | 19.5 | 10.7 |

### 15-Agent Planning Time by Scale (All Variants)

| Scale | base | controlnet_ema | controlnet_bestval | controlnet_ema_v2 | controlnet_bestval_v2 |
|-------|------|----------------|--------------------|-------------------|-----------------------|
| 1.0 | 56.57 | 44.55 | 44.22 | 43.58 | 42.45 |
| 1.1 | 50.35 | 42.13 | 47.83 | 42.89 | 42.20 |
| 1.2 | 47.83 | 45.07 | 44.78 | 42.42 | 44.90 |
| 1.3 | 66.78 | 53.34 | 46.54 | 44.53 | 44.17 |
| 1.4 | 72.63 | 56.37 | 53.37 | 58.06 | 44.83 |

### Cross-Scale Means (All 20 combinations per variant)

| Mode | Mean Success | Mean CT Expansions | Mean Planning Time (s) | Mean Path Length/Agent | Mean Acceleration/Agent | Mean Runtime-Limit Fail Rate |
|------|--------------|--------------------|------------------------|------------------------|-------------------------|------------------------------|
| base | 0.995 | 9.26 | 30.22 | 3.680 | 0.146 | 0.005 |
| controlnet_ema | 1.000 | 6.24 | 27.53 | 3.615 | 0.142 | 0.000 |
| controlnet_bestval | 1.000 | 6.05 | 27.26 | 3.616 | 0.142 | 0.000 |
| controlnet_ema_v2 | 1.000 | 5.16 | 27.85 | 3.503 | 0.133 | 0.000 |
| controlnet_bestval_v2 | 1.000 | 4.84 | 27.14 | 3.502 | 0.134 | 0.000 |

### Key Findings

1. Every ControlNet variant outperforms the base reference on mean CT expansions and mean planning time.
2. At the hardest case (`scale=1.4`, `agents=15`), `controlnet_bestval_v2` shows the strongest result:
   - CT expansions: `10.7` vs base `31.1` (about `65.6%` lower)
   - planning time: `44.83s` vs base `72.63s` (about `38.3%` lower)
   - success rate: `1.00` vs base `0.90`
3. Corrected default-guidance runs (`*_v2`) are not weaker than the first ControlNet run; they improve mean CT expansions further.
4. The best checkpoint for robustness is `controlnet_bestval_v2` (`controlnet_epoch_0333_iter_030000_state_dict.pth`) due to its more stable high-scale behavior.

### Recommended Primary Result Set

- **Primary non-ControlNet reference**: Section 2 `base` (strongest known baseline).
- **Primary ControlNet result**: Section 3 `controlnet_bestval_v2`.
- **Main claim**: ControlNet conditioning improves high-scale multi-agent planning under both tuned and default guidance policies.

---

## 5. Conclusions and Next Steps

### Conclusions from Completed Experiments

1. The global FiLM-style ControlNet adapter is effective in this cross-domain setting (2D SDF -> 1D trajectory diffusion), despite initial caution in Section 1.
2. Improvements are largest where planning is hardest (higher scales, more agents), indicating that conditioning primarily helps reduce CBS search burden.
3. The corrected rerun confirms that ControlNet gains are not tied to the earlier hyperparameter-overridden setup.

### Implications for Architecture Decisions

Section 6 of `TEMPORAL_UNET_ARCHITECTURE.md` deferred stronger spatial conditioning approaches (especially cross-attention / Approach A) because of complexity and base-model mismatch concerns.

Given that the simpler global FiLM approach already improves outcomes, a stronger conditioning mechanism is now better motivated as a follow-up, not as a speculative detour.

### Recommended Next Actions

1. Keep `controlnet_bestval_v2` as the default inference checkpoint for Conveyor scaled experiments.
2. ~~Add a targeted `env_scale=1.5` benchmark slice to document the current failure boundary and identify where ControlNet stops helping.~~ Completed; see Sections 6 to 7.
3. Run an ablation on control strength (`control_scale`: `0.5`, `1.0`, `1.5`) to verify whether high-scale gains can be increased further without harming low-scale behavior.
4. Extend the benchmark to `env_scale=1.6` or `1.7` to locate the actual failure boundary before prototyping Approach A (cross-attention conditioning).

---

## 6. 1.5x Stress-Test Execution Plan

This section records the next benchmark extension that follows directly from Section 5.

### Scope

- Extend the default Conveyor benchmark scale sweep from `1.0-1.4` to `1.0-1.5`.
- Keep the primary comparison fixed to:
  - `base` with `WINNING_HYPERPARAMS`
  - `controlnet_bestval_v2` using paper-default guidance from `mmd/config/mmd_params.py`
- Keep `control_scale=1.0` fixed for this run so the new `scale=1.5` slice isolates environment difficulty rather than introducing a second ablation axis.

### Reporting Changes

- Surface `avg_data_adherence` directly in the terminal summary printed from `launch_controlnet_evaluation.py`.
- After each scale finishes, print a cross-agent-count summary line that averages `avg_data_adherence` across agent counts (`6`, `9`, `12`, `15`).
- Preserve the existing CSV behavior where data-adherence metrics remain `0.0` when a run has zero successful trials, so the new `scale=1.5` failure boundary can be reported cleanly.

### Execution Workflow

- Use `launch_controlnet_benchmark.py` as the orchestration entry point.
- Add `--num_workers` so the benchmark matrix can be sharded by `(env_scale, mode)` pair across multiple shell scripts.
- Assign one GPU per worker and launch the generated worker scripts in separate tmux windows.
- This keeps each worker responsible for a complete `(scale, mode)` slice while reusing the existing `launch_controlnet_evaluation.py -> run_multi_agent_experiment()` flow without changing experiment semantics.

### Experimental Intent

The `scale=1.5` slice is the first targeted stress-test beyond the currently completed benchmark range. The goal is not only to measure success rate, CT expansions, and planning time, but also to check whether successful trajectories still follow the corridor structure captured by `cost_data(tau^i)` as congestion increases.

If ControlNet gains persist at `1.5`, that strengthens the case that the current global FiLM-style conditioning remains effective near the practical boundary. If both methods degrade sharply, the result provides a concrete trigger for the next architecture iteration proposed in Section 5, especially stronger spatial conditioning such as cross-attention (see `TEMPORAL_UNET_ARCHITECTURE.md` Section 6).

---

## 7. Scale 1.5 Stress-Test Results

This section reports the executed `env_scale=1.5` benchmark from Section 6. Both runs used `--seed 18`, so start/goal configurations were matched across the base and ControlNet evaluations.

### Run Metadata

| Field | `controlnet_bestval_v2` | `base` |
|-------|-------------------------|--------|
| Run ID | `2026-03-14-17-24-27-561499-pid1929600` | `2026-03-14-17-25-33-254288-pid1930758` |
| Model / checkpoint | `controlnet_epoch_0333_iter_030000_state_dict.pth` | base diffusion model |
| Hyperparameter policy | defaults from `mmd/config/mmd_params.py` | `WINNING_HYPERPARAMS` |
| `control_scale` | `1.0` | N/A |
| Seed | `18` | `18` |
| Agent counts | `6, 9, 12, 15` | `6, 9, 12, 15` |
| Trials per agent count | `10` | `10` |
| `env_scale` | `1.5` | `1.5` |

### Side-by-Side Results

#### `controlnet_bestval_v2`

| Agents | Success Rate | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|--------------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 | 1.6 | 9.09 | 0.950 | 3.804 | 0.134 | - |
| 9 | 1.0 | 4.1 | 14.81 | 0.933 | 3.862 | 0.148 | - |
| 12 | 0.9 | 5.67 | 20.15 | 0.917 | 3.802 | 0.148 | `no_solution=0.1` |
| 15 | 0.9 | 15.0 | 32.93 | 0.926 | 3.951 | 0.169 | `no_solution=0.1` |

#### `base`

| Agents | Success Rate | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|--------------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 | 2.0 | 6.67 | 0.967 | 3.837 | 0.129 | - |
| 9 | 1.0 | 9.4 | 16.17 | 0.956 | 4.136 | 0.172 | - |
| 12 | 0.9 | 16.0 | 25.62 | 0.944 | 4.257 | 0.193 | `no_solution=0.1` |
| 15 | 0.9 | 52.0 | 67.37 | 0.985 | 4.809 | 0.251 | `runtime_limit=0.1` |

### Extended 15-Agent Tables (Scales 1.0 to 1.5)

These extend the primary comparison from Section 4 by adding the `env_scale=1.5` row.

#### 15-Agent CT Expansions by Scale

| Scale | `base` | `controlnet_bestval_v2` |
|-------|--------|-------------------------|
| 1.0 | 18.8 | 9.2 |
| 1.1 | 15.1 | 9.0 |
| 1.2 | 14.6 | 10.3 |
| 1.3 | 26.4 | 10.3 |
| 1.4 | 31.1 | 10.7 |
| 1.5 | 52.0 | 15.0 |

At `env_scale=1.5`, ControlNet reduces 15-agent CT expansions by `71.2%` (`52.0 -> 15.0`), which is the largest relative reduction observed in the current sweep.

#### 15-Agent Planning Time by Scale (seconds)

| Scale | `base` | `controlnet_bestval_v2` |
|-------|--------|-------------------------|
| 1.0 | 56.57 | 42.45 |
| 1.1 | 50.35 | 42.20 |
| 1.2 | 47.83 | 44.90 |
| 1.3 | 66.78 | 44.17 |
| 1.4 | 72.63 | 44.83 |
| 1.5 | 67.37 | 32.93 |

At `env_scale=1.5`, ControlNet reduces 15-agent planning time by `51.1%` (`67.37s -> 32.93s`).

### Per-Scale Breakdown by Agent Count (Primary Comparison Pair)

These tables complement the cross-scale 15-agent view by showing the primary comparison pair at each fixed scale across all four agent counts. Each cell is listed as `base / controlnet_bestval_v2`.

#### Scale 1.0

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 | 0.9 / 0.8 | 9.64 / 12.64 | 0.883 / 0.867 | 2.999 / 3.085 | 0.100 / 0.108 | - / - |
| 9 | 1.0 / 1.0 | 3.1 / 2.6 | 17.56 / 21.20 | 0.967 / 0.933 | 3.162 / 3.231 | 0.115 / 0.123 | - / - |
| 12 | 1.0 / 1.0 | 7.8 / 5.0 | 30.09 / 30.13 | 0.958 / 0.933 | 3.543 / 3.415 | 0.144 / 0.134 | - / - |
| 15 | 1.0 / 1.0 | 18.8 / 9.2 | 56.57 / 42.45 | 0.960 / 0.947 | 3.683 / 3.381 | 0.163 / 0.140 | - / - |

#### Scale 1.1

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 | 1.2 / 0.8 | 9.67 / 12.15 | 0.867 / 0.917 | 3.089 / 3.143 | 0.104 / 0.111 | - / - |
| 9 | 1.0 / 1.0 | 3.9 / 3.2 | 18.14 / 21.12 | 0.967 / 0.967 | 3.413 / 3.314 | 0.128 / 0.127 | - / - |
| 12 | 1.0 / 1.0 | 8.7 / 5.5 | 31.75 / 30.76 | 0.967 / 0.917 | 3.666 / 3.384 | 0.146 / 0.132 | - / - |
| 15 | 1.0 / 1.0 | 15.1 / 9.0 | 50.35 / 42.20 | 0.960 / 0.947 | 3.629 / 3.450 | 0.155 / 0.140 | - / - |

#### Scale 1.2

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 | 1.7 / 0.9 | 10.29 / 12.66 | 0.933 / 0.883 | 3.470 / 3.514 | 0.119 / 0.124 | - / - |
| 9 | 1.0 / 1.0 | 4.1 / 2.8 | 18.47 / 20.77 | 0.933 / 0.922 | 3.481 / 3.527 | 0.127 / 0.132 | - / - |
| 12 | 1.0 / 1.0 | 8.9 / 5.0 | 32.15 / 30.11 | 0.950 / 0.942 | 3.600 / 3.413 | 0.144 / 0.133 | - / - |
| 15 | 1.0 / 1.0 | 14.6 / 10.3 | 47.83 / 44.90 | 0.960 / 0.960 | 3.906 / 3.645 | 0.160 / 0.145 | - / - |

#### Scale 1.3

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 | 1.0 / 0.8 | 9.37 / 12.48 | 0.950 / 0.950 | 3.408 / 3.436 | 0.108 / 0.117 | - / - |
| 9 | 1.0 / 1.0 | 5.5 / 3.9 | 20.30 / 22.83 | 0.967 / 0.911 | 3.809 / 3.615 | 0.146 / 0.138 | - / - |
| 12 | 1.0 / 1.0 | 12.2 / 5.2 | 36.61 / 30.55 | 0.942 / 0.900 | 4.184 / 3.584 | 0.182 / 0.139 | - / - |
| 15 | 1.0 / 1.0 | 26.4 / 10.3 | 66.78 / 44.17 | 0.987 / 0.960 | 4.321 / 3.834 | 0.205 / 0.159 | - / - |

#### Scale 1.4

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 | 2.4 / 1.5 | 11.24 / 13.27 | 0.917 / 0.950 | 3.835 / 3.690 | 0.139 / 0.134 | - / - |
| 9 | 1.0 / 1.0 | 5.0 / 2.9 | 19.27 / 21.27 | 0.944 / 0.944 | 3.960 / 3.741 | 0.149 / 0.141 | - / - |
| 12 | 1.0 / 1.0 | 12.8 / 6.4 | 35.80 / 32.24 | 0.975 / 0.950 | 4.057 / 3.795 | 0.172 / 0.148 | - / - |
| 15 | 0.9 / 1.0 | 31.1 / 10.7 | 72.63 / 44.83 | 0.970 / 0.953 | 4.391 / 3.846 | 0.215 / 0.156 | runtime_limit=0.1 / - |

#### Scale 1.5

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 | 2.0 / 1.6 | 6.67 / 9.09 | 0.967 / 0.950 | 3.837 / 3.804 | 0.129 / 0.134 | - / - |
| 9 | 1.0 / 1.0 | 9.4 / 4.1 | 16.17 / 14.81 | 0.956 / 0.933 | 4.136 / 3.862 | 0.172 / 0.148 | - / - |
| 12 | 0.9 / 0.9 | 16.0 / 5.67 | 25.62 / 20.15 | 0.944 / 0.917 | 4.257 / 3.802 | 0.193 / 0.148 | no_solution=0.1 / no_solution=0.1 |
| 15 | 0.9 / 0.9 | 52.0 / 15.0 | 67.37 / 32.93 | 0.985 / 0.926 | 4.809 / 3.951 | 0.251 / 0.169 | runtime_limit=0.1 / no_solution=0.1 |

### Updated Cross-Scale Means (Primary Comparison Pair)

The `1.0-1.4` rows are copied from Section 4. The `1.0-1.5` rows add the new `env_scale=1.5` slice, giving `24` scale/agent-count combinations per mode.

| Mode | Mean Success | Mean CT Expansions | Mean Planning Time (s) | Mean Path Length/Agent | Mean Acceleration/Agent | Mean Runtime-Limit Fail Rate |
|------|--------------|--------------------|------------------------|------------------------|-------------------------|------------------------------|
| `base` (`1.0-1.4`) | 0.995 | 9.26 | 30.22 | 3.680 | 0.146 | 0.005 |
| `base` (`1.0-1.5`) | 0.988 | 11.03 | 30.01 | 3.777 | 0.153 | 0.008 |
| `controlnet_bestval_v2` (`1.0-1.4`) | 1.000 | 4.84 | 27.14 | 3.502 | 0.134 | 0.000 |
| `controlnet_bestval_v2` (`1.0-1.5`) | 0.992 | 5.13 | 25.82 | 3.561 | 0.137 | 0.000 |

The main cross-scale effect is that the CT-expansion gap widens further when `env_scale=1.5` is included: ControlNet moves from `47.7%` lower mean CT expansions than base (`4.84` vs `9.26`) to `53.5%` lower (`5.13` vs `11.03`).

### Analysis

#### 1. Scale 1.5 Is Not a Failure Boundary

The Section 6 stress test was intended to locate the current failure boundary. It does not do that. Both methods still achieve `90-100%` success across all agent counts, with only one failed trial at `12` agents and one failed trial at `15` agents for each mode.

This means the practical boundary lies beyond `env_scale=1.5`.

#### 2. ControlNet Advantage Grows with Scale

The 15-agent CT-expansion trend shows that the ControlNet benefit grows as the environment gets harder:

| Scale | `base` CT | `controlnet_bestval_v2` CT | Reduction |
|-------|-----------|----------------------------|-----------|
| 1.0 | 18.8 | 9.2 | 51.1% |
| 1.2 | 14.6 | 10.3 | 29.5% |
| 1.4 | 31.1 | 10.7 | 65.6% |
| 1.5 | 52.0 | 15.0 | 71.2% |

The base model's CT expansions grow much faster than ControlNet's as scale increases. This is the clearest sign that SDF conditioning keeps the generated trajectories closer to what CBS can resolve efficiently.

#### 3. Planning Time Flips from Overhead to Payoff

At `6` agents, the base model is faster (`6.67s` vs `9.09s`), which is consistent with the extra SDF-encoder forward pass adding fixed overhead on easy instances.

By `15` agents, that overhead is dominated by CBS savings: ControlNet is `51.1%` faster (`32.93s` vs `67.37s`). The gain appears as soon as the search problem becomes difficult enough for CT expansions to dominate runtime.

#### 4. Path Quality Improves at the Hardest Slice

At `15` agents, ControlNet also improves the successful trajectories themselves:

- path length: `3.951` vs `4.809` (`17.8%` shorter)
- acceleration: `0.169` vs `0.251` (`32.7%` lower)

So the higher-scale benefit is not limited to search effort. The trajectories are also shorter and smoother.

#### 5. Data Adherence Remains High for Both Modes

The new reporting path added in Section 6 shows that both methods remain well aligned with the conveyor corridor structure at `env_scale=1.5`:

- `controlnet_bestval_v2`: `0.917-0.950` across agent counts (mean `0.932`)
- `base`: `0.944-0.985` across agent counts (mean `0.963`)

Base is slightly higher on this metric, especially at `15` agents, but both methods remain comfortably above `0.9`. The successful trajectories therefore still follow the intended data geometry even under the added congestion.

#### 6. Failure Modes Separate at 15 Agents

At `12` agents, both modes have one `no_solution` failure. At `15` agents, the failure modes diverge:

- `base`: `runtime_limit=0.1`
- `controlnet_bestval_v2`: `no_solution=0.1`

This suggests two different bottlenecks. The base model primarily struggles because CBS search grows too large, while ControlNet primarily struggles when a small subset of generated trajectories still leads to an unresolved conflict pattern.

### Revised Next Steps

1. Keep `controlnet_bestval_v2` as the default inference checkpoint for Conveyor scaled experiments.
2. Treat `env_scale=1.5` as completed, not as the failure boundary.
3. Run the planned `control_scale` ablation (`0.5`, `1.0`, `1.5`) to check whether the remaining `no_solution` failures can be reduced without giving up the CT-expansion gains.
4. Extend the benchmark to `env_scale=1.6` or `1.7` before investing in Approach A (cross-attention conditioning).
5. Investigate the `no_solution` failure mode at high scale/high agent count as the most likely near-term improvement target.
