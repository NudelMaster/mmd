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
3. ~~Run the planned `control_scale` ablation (`0.5`, `1.0`, `1.5`) to check whether the remaining `no_solution` failures can be reduced without giving up the CT-expansion gains.~~ Completed; see Section 8.
4. Extend the benchmark to `env_scale=1.6` or `1.7` before investing in Approach A (cross-attention conditioning).
5. Investigate the `no_solution` failure mode at high scale/high agent count as the most likely near-term improvement target.

---

## 8. Control-Scale Ablation Results

This section documents the completed `control_scale` ablation proposed at the end of Section 7. The goal was to test whether changing ControlNet residual strength can remove the remaining `no_solution` failures at `env_scale=1.5`, especially for `12` and `15` agents.

### Protocol

- checkpoint fixed to `controlnet_epoch_0333_iter_030000_state_dict.pth`
- guidance fixed to the paper defaults from `mmd/config/mmd_params.py`
- scales: `1.0, 1.1, 1.2, 1.3, 1.4, 1.5`
- agent counts: `6, 9, 12, 15`
- trials per combination: `10`
- seed: `18`
- compared strengths: `control_scale = 0.5, 1.0, 1.5`

The `control_scale=1.0` reference reuses the completed `controlnet_bestval_v2` results from Sections 3 and 7. The new ablation launches added only the `0.5` and `1.5` variants.

### Run Mapping

| Run label | `control_scale` | Folders | Scales covered | Source |
|-----------|-----------------|---------|----------------|--------|
| `controlnet_bestval_v2_cs050` | `0.5` | `6` | `1.0..1.5` | new ablation run |
| `controlnet_bestval_v2` | `1.0` | `6` | `1.0..1.5` | existing reference from Sections 3 and 7 |
| `controlnet_bestval_v2_cs150` | `1.5` | `6` | `1.0..1.5` | new ablation run |

### Cross-Scale Means (All 24 combinations per control scale)

| Run label | `control_scale` | Mean Success | Mean CT Expansions | Mean Planning Time (s) | Mean Data Adherence | Mean Path Length/Agent | Mean Acceleration/Agent | Mean `no_solution` Rate |
|-----------|-----------------|--------------|--------------------|------------------------|---------------------|------------------------|-------------------------|-------------------------|
| `controlnet_bestval_v2_cs050` | `0.5` | 0.971 | 6.04 | 21.28 | 0.945 | 3.532 | 0.133 | 0.029 |
| `controlnet_bestval_v2` | `1.0` | 0.992 | 5.13 | 25.82 | 0.932 | 3.561 | 0.137 | 0.008 |
| `controlnet_bestval_v2_cs150` | `1.5` | 0.992 | 4.59 | 19.55 | 0.933 | 3.692 | 0.146 | 0.008 |

All observed ControlNet ablation failures remain `no_solution` failures. No `runtime_limit` failures appear in any of the three ControlNet-strength variants.

### Scale 1.5 Breakdown by Agent Count

Each cell is listed as `control_scale=0.5 / 1.0 / 1.5`.

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 / 1.0 / 1.0 | 1.60 / 1.60 / 1.40 | 9.12 / 9.09 / 9.04 | 0.967 / 0.950 / 0.917 | 3.712 / 3.804 / 3.997 | 0.129 / 0.134 / 0.147 | - / - / - |
| 9 | 1.0 / 1.0 / 1.0 | 4.40 / 4.10 / 3.40 | 15.36 / 14.81 / 14.51 | 0.956 / 0.933 / 0.933 | 3.834 / 3.862 / 3.974 | 0.144 / 0.148 / 0.157 | - / - / - |
| 12 | 0.9 / 0.9 / 1.0 | 9.67 / 5.67 / 6.00 | 24.31 / 20.15 / 21.57 | 0.926 / 0.917 / 0.942 | 3.999 / 3.802 / 4.031 | 0.164 / 0.148 / 0.164 | no_solution=0.1 / no_solution=0.1 / - |
| 15 | 0.8 / 0.9 / 1.0 | 22.75 / 15.00 / 14.50 | 41.52 / 32.93 / 35.16 | 0.958 / 0.926 / 0.920 | 4.183 / 3.951 / 4.093 | 0.192 / 0.169 / 0.188 | no_solution=0.2 / no_solution=0.1 / - |

### 15-Agent Comparison by Scale

Each cell is listed as `control_scale=0.5 / 1.0 / 1.5`.

| Scale | Success | CT Expansions | Planning Time (s) | Data Adherence | Fail Mode |
|-------|---------|---------------|-------------------|----------------|-----------|
| 1.0 | 1.0 / 1.0 / 0.9 | 9.10 / 9.20 / 6.89 | 33.27 / 42.45 / 30.06 | 0.953 / 0.947 / 0.956 | - / - / no_solution=0.1 |
| 1.1 | 1.0 / 1.0 / 1.0 | 9.40 / 9.00 / 7.30 | 33.03 / 42.20 / 30.65 | 0.967 / 0.947 / 0.987 | - / - / - |
| 1.2 | 1.0 / 1.0 / 1.0 | 10.60 / 10.30 / 7.70 | 34.10 / 44.90 / 31.98 | 0.953 / 0.960 / 0.967 | - / - / - |
| 1.3 | 0.9 / 1.0 / 1.0 | 11.78 / 10.30 / 8.60 | 35.98 / 44.17 / 31.33 | 0.963 / 0.960 / 0.920 | no_solution=0.1 / - / - |
| 1.4 | 0.9 / 1.0 / 1.0 | 16.78 / 10.70 / 12.50 | 37.77 / 44.83 / 33.35 | 0.948 / 0.953 / 0.933 | no_solution=0.1 / - / - |
| 1.5 | 0.8 / 0.9 / 1.0 | 22.75 / 15.00 / 14.50 | 41.52 / 32.93 / 35.16 | 0.958 / 0.926 / 0.920 | no_solution=0.2 / no_solution=0.1 / - |

### What the Ablation Shows

1. Lowering ControlNet strength to `0.5` does **not** help. It is the weakest setting overall: mean success drops to `0.971`, mean `no_solution` rate rises to `0.029`, and the motivating `env_scale=1.5` slice gets worse rather than better.
2. Raising ControlNet strength to `1.5` removes the specific `env_scale=1.5` failures that motivated the ablation. At `12` agents, success improves from `0.9` to `1.0`; at `15` agents, success improves from `0.9` to `1.0` and `no_solution` disappears completely.
3. The `1.5` setting is **not** a strict global winner. It introduces isolated `no_solution=0.1` failures at `scale=1.0, agents=15` and `scale=1.1, agents=12`, so its mean success ties `control_scale=1.0` rather than clearly exceeding it.
4. At the hardest `scale=1.5` slice, the main gain from `control_scale=1.5` is robustness, not speed. Relative to `1.0`, CT expansions improve slightly at `15` agents (`15.0 -> 14.5`), but planning time is slightly worse (`32.93s -> 35.16s`) and successful trajectories are longer and less smooth.
5. The original "loosen the bundle" hypothesis is therefore not supported. If the immediate objective is to remove the remaining `no_solution` failures at `env_scale=1.5`, increasing control strength helps more than decreasing it.

### Updated Recommendation

- keep `controlnet_bestval_v2` with `control_scale=1.0` as the safer balanced default across the full `1.0-1.5` sweep
- treat `control_scale=1.5` as the preferred stress-test variant when the main objective is success at `env_scale=1.5`
- next step: extend the same comparison to `env_scale=1.6` or `1.7` to see whether the `1.5` setting continues to help beyond the current stress-test boundary

---

## 9. Metric Definitions and Aggregation Rules

This section explains how the metrics reported throughout this document are computed in the current inference pipeline.

### Aggregation Levels

The reporting pipeline has three levels:

1. **Per-trial metrics** are computed for one MAPF attempt.
2. **Per-row aggregated metrics** in `aggregated_results_all_agents.csv` summarize one fixed `(env_scale, num_agents, planner)` slice across all trials for that slice.
3. **Cross-scale means** reported in this document are simple arithmetic means of those aggregated CSV rows across the included `(env_scale, num_agents)` combinations.

Unless stated otherwise, tables in this document use the per-row aggregated values from `aggregated_results_all_agents.csv`.

### Trial Outcome Categories

Each trial ends in exactly one of these statuses:

- `SUCCESS`
- `FAIL_RUNTIME_LIMIT`
- `FAIL_NO_SOLUTION`
- `FAIL_COLLISION_AGENTS`

The success and fail-rate metrics are computed from these discrete outcome labels.

### Metrics Normalized by Total Number of Trials

For a fixed `(env_scale, num_agents, planner)` slice with `N` total trials:

- `success_rate`
  - fraction of trials whose final status is `SUCCESS`
  - formula: `#SUCCESS / N`
- `fail_rate_runtime_limit`
  - formula: `#FAIL_RUNTIME_LIMIT / N`
- `fail_rate_no_solution`
  - formula: `#FAIL_NO_SOLUTION / N`
- `fail_rate_collision_agents`
  - formula: `#FAIL_COLLISION_AGENTS / N`

These rates therefore sum to `1.0` up to floating-point rounding.

### Metrics Normalized by Number of Successful Trials

Let `S` be the number of successful trials in the same slice.

The following metrics are accumulated only over successful trials and then divided by `S`:

- `avg_ct_expansions`
  - mean number of conflict-tree nodes expanded by the multi-agent planner
- `avg_planning_time`
  - mean wall-clock planning time in seconds
- `avg_data_adherence`
  - mean trajectory adherence score
- `avg_path_length_per_agent`
  - mean path length, after first averaging over agents within each successful trial
- `avg_mean_path_acceleration_per_agent`
  - mean path acceleration, after first averaging over agents within each successful trial

If `S = 0`, these fields remain `0.0` in the aggregated CSV. This is why fully failed slices can show `0.0` for path-quality or adherence metrics.

### `avg_num_collisions_in_solution`

`avg_num_collisions_in_solution` is a special case in the current implementation:

- per trial, `num_collisions_in_solution` starts from the planner-reported collision count in the returned solution
- if a supposedly successful solution still contains agent-agent collisions under the post-check, those collisions are added and the trial status is changed to `FAIL_COLLISION_AGENTS`
- during aggregation, the metric is accumulated only from trials whose final status is `SUCCESS`
- the final sum is then divided by the total number of trials `N`

So this field is best interpreted as a sanity-check metric. In the successful MAPF runs reported here it is usually `0.0`.

### Trial-Level Path Metrics

For one successful trial with `A` agents:

- `path_length_per_agent`
  - compute each agent's 2D path length from the returned position trajectory
  - average those `A` path lengths across agents
- `mean_path_acceleration_per_agent`
  - compute each agent's average acceleration from its position/velocity trajectory
  - average those `A` per-agent accelerations across agents

This means the aggregated CSV already stores an agent-averaged value at the trial level before averaging across successful trials.

### Trial-Level `data_adherence`

`data_adherence` is environment-defined. For the Conveyor environments used in this document:

- each trajectory segment is scored against the conveyor corridor structure
- in `EnvConveyor2D`, a segment receives `1` if it traverses a full valid corridor pattern (top corridor right-to-left or bottom corridor left-to-right) and `0` otherwise
- if an agent trajectory spans multiple skeleton tiles, its adherence is the mean of the per-tile adherence scores
- the trial-level `data_adherence` is then the mean of those per-agent adherence values across all agents

Because of that averaging, `avg_data_adherence` can take any value in `[0, 1]`, not just binary values.

### Cross-Scale Means Used in This Document

When this document reports values such as:

- "mean success"
- "mean CT expansions"
- "updated cross-scale means"

the computation is a simple arithmetic mean over the relevant aggregated CSV rows. For example:

- the `1.0-1.5` cross-scale means average the `24` rows coming from `6` scales x `4` agent counts
- the `1.0-1.4` means average `20` rows

These are means over experiment slices, not means over individual trials pooled together globally.

### Percentage Improvements Reported in the Text

Whenever the document states that one method reduces a metric by some percentage, the calculation is:

- `reduction = (reference - variant) / reference`

For example, a CT-expansion reduction from `52.0` to `15.0` is reported as:

- `(52.0 - 15.0) / 52.0 = 71.2%`

The same convention is used for planning-time reductions and similar comparisons.

### Cross-Agent `avg_data_adherence` Summary Lines

The console summary printed after each scale in `launch_controlnet_evaluation.py` reports a cross-agent `avg_data_adherence` value by:

- reading the `avg_data_adherence` column from `aggregated_results_all_agents.csv`
- filling any missing values with `0.0`
- averaging across the four agent-count rows for that scale

This summary is only a convenience printout. The primary source of truth for analysis remains the per-row aggregated CSV values used throughout this document.

---

## 10. Hybrid-Planner Multi-Scale Data Generation

This section documents the alternative data generation pipeline added in `scripts/generate_data/generate_multiscale_data_hybrid.py`.

### Motivation

The original multi-scale dataset (`EnvConveyor2D-RobotPlanarDisk-multiscale/`) was generated by `generate_multiscale_data.py`, which uses the pretrained MMD diffusion model to produce trajectories. Training ControlNet on data produced by the same model it conditions introduces a self-distillation risk: the adapter may learn to replicate the base model's existing behaviour rather than acquiring genuinely new conditioning information from the SDF signal.

The hybrid-planner script avoids this by using the same classical planner pipeline as the original non-ControlNet data generation (`generate_trajectories.py`): RRTConnect / RRTStar for initial paths, followed by GPMP2 trajectory optimisation. Because this pipeline is completely independent of the diffusion model, the resulting trajectories form a clean, non-circular training source for the adapter.

### Script Location

`scripts/generate_data/generate_multiscale_data_hybrid.py`

### Output

The script writes to:

```
data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid/
  0/metadata.yaml, args.yaml
  scale_1.00/0/trajs-free.pt, metadata.yaml, args.yaml
  scale_1.10/0/trajs-free.pt, metadata.yaml, args.yaml
  scale_1.20/0/trajs-free.pt, metadata.yaml, args.yaml
  scale_1.30/0/trajs-free.pt, metadata.yaml, args.yaml
  scale_1.40/0/trajs-free.pt, metadata.yaml, args.yaml
  scale_1.50/0/trajs-free.pt, metadata.yaml, args.yaml
```

Output tensor shape per scale: `[N, 64, 4]` (64 waypoints, 4D state: 2D pos + 2D vel). This matches the `ControlNetTrajectoryDataset` format used by the training pipeline.

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--env_id` | `EnvConveyor2D` | Environment class name |
| `--robot_id` | `RobotPlanarDisk` | Robot class name |
| `--scales` | `1.0 1.1 1.2 1.3 1.4 1.5` | Environment scales to generate |
| `--num_trajs_per_scale` | `500` | Target trajectory count per scale |
| `--output_dir` | `data_trajectories` | Root output directory |
| `--num_trajs_per_context` | `20` | Trajectories per start-goal pair |
| `--threshold_start_goal_pos` | `0.5` | Min Euclidean distance between start and goal |
| `--obstacle_cutoff_margin` | `0.05` | Task collision margin |
| `--rrt_max_time` | `300` | Max RRT planning time per start-goal pair (seconds) |
| `--gpmp_opt_iters` | `500` | GPMP2 optimisation iterations |
| `--n_support_points` | `64` | Waypoints per trajectory |
| `--duration` | `5.0` | Trajectory duration (seconds) |
| `--no_skills` | off | Disable conveyor belt skill sequences; use direct RRT start-to-goal |
| `--device` | `cuda` | Torch device |
| `--seed` | `18` | Random seed |
| `--debug` | off | Print planner timing and debug output |

### Commands

#### Full dataset — multi-GPU parallel (recommended)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_data/generate_multiscale_data_hybrid.py --scales 1.0 1.1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_data/generate_multiscale_data_hybrid.py --scales 1.2 1.3 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_data/generate_multiscale_data_hybrid.py --scales 1.4 1.5 &
wait
```

Each process writes only the scales it is given, so the three processes can safely write to the same output directory in parallel (different subdirectories per scale).

#### Single process, all scales

```bash
python scripts/generate_data/generate_multiscale_data_hybrid.py
```

#### Retrain ControlNet on the hybrid dataset

```bash
cd scripts/train_diffusion
python train_controlnet.py \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid \
    --results_dir logs_controlnet_hybrid/0
```

### Relation to Existing Scripts

- `generate_trajectories.py` — original single-scale hybrid-planner data generation; this script adapts and extends it with the `scale` parameter.
- `generate_multiscale_data.py` — diffusion-model-based multi-scale generation; produces `EnvConveyor2D-RobotPlanarDisk-multiscale/`.
- `generate_multiscale_data_hybrid.py` — hybrid-planner multi-scale generation; produces `EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid/`.

### Generation Status (COMPLETE as of 2026-03-28)

Generation is COMPLETE. All 6 scales produced 500 trajectories each. Six tmux sessions were launched (one per GPU 0–5, one per scale 1.0–1.5) with the following command pattern:

```bash
CUDA_VISIBLE_DEVICES=X PYTHONUNBUFFERED=1 python -u \
  scripts/generate_data/generate_multiscale_data_hybrid.py \
  --scales X.X \
  --num_trajs_per_scale 500 \
  --num_trajs_per_context 20 \
  --rrt_max_time 60 \
  --gpmp_opt_iters 500
```

Estimated wall time: 5–6 hours (gated by scale 1.5).

Three bugs were resolved before launch:

1. `optimize_sequentially=True` was changed to `False` for RRT* planners to enable 20-worker parallelism (~20x speedup).
2. `mp.set_start_method` called multiple times across workers — wrapped in `try/except` in `deps/motion_planning_baselines/mp_baselines/planners/multi_processing.py`.
3. `IdentityPlanner` must remain `optimize_sequentially=True` because it returns CUDA tensors that cannot round-trip through `forkserver`.

#### rrt_max_time=60s concern for larger scales

At `--rrt_max_time 60`, RRT* has 60 seconds per start-goal pair to find a path through the conveyor corridors. At scales ≥1.3 the corridors become narrower, and the skip rate (contexts that yield fewer than 3 free trajectories) may be consistently high.

**Monitor**: if scales 1.3–1.5 produce noticeably fewer than 500 trajectories, relaunch those scales with `--rrt_max_time 120`. The wall time cost is roughly proportional: doubling `rrt_max_time` roughly doubles time per context.

#### Pre-training checklist (run before launching ControlNet retraining)

Before calling `train_controlnet.py --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid`:

- [ ] Verify all 6 scale directories exist under `data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid/`
- [ ] Verify each `scale_X.XX/0/trajs-free.pt` has shape `[500, 64, 4]`
- [ ] Verify the SDF cache at `data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache/` contains `sdf_scale_X.XX.pt` for all 6 scales (1.00, 1.10, 1.20, 1.30, 1.40, 1.50)
- [ ] If any scale has fewer than the target count, decide whether to top-up with a re-run or train on the available subset

#### Generation Results (verified 2026-03-28)

**Verified shapes (all correct):**
```
scale=1.00  [500, 64, 4]  ✓
scale=1.10  [500, 64, 4]  ✓
scale=1.20  [500, 64, 4]  ✓
scale=1.30  [500, 64, 4]  ✓
scale=1.40  [500, 64, 4]  ✓
scale=1.50  [500, 64, 4]  ✓
```

**Context statistics:**
| Scale | Total contexts | Skip rate |
|-------|---------------|-----------|
| 1.0 | 83 | 33% |
| 1.1 | 78 | 37% |
| 1.2 | 71 | 45% |
| 1.3 | 73 | 51% |
| 1.4 | 109 | 51% |
| 1.5 | 110 | 63% |

---

## 11. Improved ControlNet Training on Hybrid Dataset (v2 Run)

**Status: v1 COMPLETE — v2 COMPLETE — Benchmarks COMPLETE (2026-03-28) — Results in Section 12**

### Overview

Two parallel ControlNet training runs were launched on the hybrid dataset (`EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid/`) on 2026-03-28:

- **v1 (unimproved baseline)**: runs the authors' original `train_controlnet.py` code verbatim, with no modifications. This serves as a controlled comparison point that matches the earlier `logs_controlnet_full/` run in training logic, applied to the hybrid dataset.
- **v2 (improved)**: runs a modified version of `train_controlnet.py` incorporating 5 user-initiated experimental improvements (see below). These are additions designed for the multi-scale training setting that the authors' code was not originally built for.

Both runs use the same frozen base model (`data_trained_models/EnvConveyor2D-RobotPlanarDisk/`) and the same SDF cache (`data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache/`).

### Why the Authors' Training Code Needed Adaptation

The authors' `train_controlnet.py` was designed for the simpler single-scale training setting in the original paper. Applying it directly to multi-scale training introduces several issues:

1. **Fixed dataset ordering**: the multi-scale dataset is stored per-scale (all scale 1.0 samples first, then 1.1, etc.). With no shuffle, the model sees samples in strict scale order every epoch, creating systematic presentation bias.
2. **Fixed learning rate**: a constant 1e-4 LR is appropriate when training terminates early (thousands of steps). For 100k–150k steps over 3000 samples, the loss plateaus and a fixed LR prevents fine-grained late convergence.
3. **Random validation split**: with only ~150 val samples across 6 scales, a purely random 5% split can leave one or more scales nearly absent from validation, making the val loss a poor indicator of generalisation.
4. **No best-val checkpoint tracking**: the only saved checkpoint (beyond fixed intervals) is the final one. Selecting the best checkpoint requires manually inspecting the loss curve and re-running with a manually identified step.

### The 5 Experimental Improvements in v2

| # | Improvement | What changed | Motivation |
|---|-------------|--------------|------------|
| 1 | Per-epoch shuffle | `shuffle=True` in train DataLoader | Eliminates systematic bias from the per-scale storage order of the multi-scale dataset |
| 2 | Cosine LR decay | `CosineAnnealingLR` from 1e-4 to 1e-5 over all training steps | Prevents large steps late in training after the loss has already flattened |
| 3 | Stratified val split | 25 samples per scale (exact), using `np.random.default_rng(seed)` | Guarantees all 6 scales are represented equally in validation; random split cannot guarantee this with only ~150 val samples |
| 4 | Auto best-val checkpoint | `controlnet_best_val_state_dict.pth` + `ema_controlnet_best_val_state_dict.pth` saved automatically on improvement | Removes the need to manually watch the val loss curve to select a checkpoint |
| 5 | AMP (mixed precision) | `--use_amp true` flag | RTX 6000 Ada supports BF16/FP16 natively; enables ~1.5x training speedup with no accuracy loss |

### Training Hyperparameters

| Parameter | v1 (baseline) | v2 (improved) |
|-----------|--------------|---------------|
| `num_train_steps` | 100 000 | 150 000 |
| `batch_size` | 32 | 32 |
| `lr` | 1e-4 (fixed) | 1e-4 → 1e-5 cosine |
| `use_amp` | false | true |
| `seed` | 0 | 0 |
| GPU | 0 | 1 |
| PID | 1738440 | (started shortly after v1) |
| Start time | ~19:36 2026-03-28 | ~19:36 2026-03-28 |
| Results dir | `logs_controlnet_hybrid/0/0/` | `logs_controlnet_hybrid_v2/0/0/` |

Both runs use the frozen base model at `data_trained_models/EnvConveyor2D-RobotPlanarDisk/`.

### Training Commands

**v1 (baseline — authors' original code, hybrid dataset):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_hybrid/0 \
    --num_train_steps 100000
```

**v2 (improved — with 5 experimental additions):**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale-hybrid \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_hybrid_v2/0 \
    --num_train_steps 150000 \
    --use_amp true
```

### Files Modified

| File | Change |
|------|--------|
| `mmd/trainer/train_loaders.py` | Added `shuffle=False` parameter (backward-compatible) to `get_dataset`; v2 training script passes `shuffle=True` for the train split |
| `scripts/train_diffusion/train_controlnet.py` | Stratified val split by scale; `CosineAnnealingLR` scheduler; auto best-val checkpoint saving; AMP pass-through via `--use_amp` flag; imports updated |

### Next Steps

1. **Phase 4 verification**: COMPLETE — both v1 (100k steps) and v2 (150k steps) training done; checkpoints confirmed at expected paths.
2. **Smoke tests**: COMPLETE — both variants passed (2026-03-28). v1 checkpoint + SDF cache loaded, 1 trial without exception (FAIL_NO_SOLUTION — planning outcome, not code failure). v2 SUCCESS: data_adherence=0.833, 4 CT expansions, 10.9s.
3. **Phase 3 benchmark**: COMPLETE (2026-03-28). v1 on GPUs 0/2/3 (`controlnet_hybrid_v1`), v2 on GPUs 4/5/6 (`controlnet_hybrid_v2_bestval`); 240 trials each; paper default hyperparams; seed=18. Results in Section 12.
4. **Cross-variant analysis COMPLETE**: see Section 12 for full results and comparison across all 4 variants.
5. **Open questions answered** (see Section 12 Analysis):
   - v2 improved training does outperform v1 at every scale, especially 1.3–1.5.
   - Hybrid (classical-planner) data does NOT generalise better — original diffusion-generated training data produces stronger ControlNet conditioning.
   - The self-distillation concern appears overstated; the original dataset remains the preferred training source.
6. **Next step**: regenerate hybrid dataset with `rrt_max_time ≥ 300s` to test whether higher-quality data closes the performance gap, OR investigate cross-attention conditioning as the next architecture iteration.

Full session notes: `docs/controlnet_training_session_2026_03_28.md`

### Benchmark Launch Details

- **Launch timestamp**: 2026-03-28 ~21:42
- **Bug (fixed before launch)**: checkpoint paths passed to `launch_controlnet_benchmark.py` must be absolute, not relative. Worker scripts `cd` to `scripts/inference/` before running, so relative paths resolve incorrectly. Fix: regenerate scripts with fully-qualified checkpoint paths.
- **v1 launch command**: `bash scripts/inference/gpu_scripts/hybrid_benchmark_v1/launch_controlnet_benchmark_tmux.sh`
- **v2 launch command**: `bash scripts/inference/gpu_scripts/hybrid_benchmark_v2/launch_controlnet_benchmark_tmux.sh`
- **Monitoring**: `tmux attach -t hybrid_v1_bench_workers` / `tmux attach -t hybrid_v2_bench_workers`

---

## 12. Hybrid ControlNet Benchmark Results (Completed 2026-03-28)

**Status: COMPLETE** — Both variants fully benchmarked across 6 scales (1.0–1.5), 4 agent counts (6/9/12/15), 10 trials each.

### Overview

Two hybrid-trained ControlNet checkpoints were benchmarked using the same protocol as the original `controlnet_bestval_v2` benchmark (Section 7):

| Run label | Checkpoint | Training data | Training code |
|-----------|-----------|---------------|---------------|
| `controlnet_hybrid_v1` | `logs_controlnet_hybrid/0/0/checkpoints/ema_controlnet_final_state_dict.pth` | hybrid (RRT+GPMP2) | original (unmodified) |
| `controlnet_hybrid_v2_bestval` | `logs_controlnet_hybrid_v2/0/0/checkpoints/ema_controlnet_best_val_state_dict.pth` | hybrid (RRT+GPMP2) | improved (v2: shuffle, cosine LR, stratified val, best-val ckpt, AMP) |

Both runs used `--controlnet_use_paper_defaults`, `--seed 18`, `--control_scale 1.0`, `--runtime_limit 180`.
The primary comparison pair from prior sections remains `base` (WINNING_HYPERPARAMS) and `controlnet_bestval_v2` (original ControlNet, paper defaults).

---

### controlnet_hybrid_v1 — Full Results by Scale

#### Scale 1.0

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.9 | 1.44 | 9.52 | 0.944 | 3.081 | 0.090 | no_solution=0.1 |
| 9 | 0.5 | 1.60 | 14.13 | 0.956 | 2.995 | 0.081 | no_solution=0.5 |
| 12 | 0.3 | 9.00 | 25.15 | 0.944 | 3.243 | 0.108 | no_solution=0.7 |
| 15 | 0.0 | — | — | — | — | — | no_solution=1.0 |

#### Scale 1.1

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.8 | 2.63 | 10.75 | 0.917 | 3.007 | 0.089 | no_solution=0.2 |
| 9 | 0.6 | 9.83 | 21.90 | 0.963 | 3.285 | 0.109 | no_solution=0.4 |
| 12 | 0.2 | 21.00 | 36.27 | 1.000 | 3.498 | 0.130 | no_solution=0.8 |
| 15 | 0.0 | — | — | — | — | — | no_solution=0.9, runtime_limit=0.1 |

#### Scale 1.2

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.7 | 1.86 | 9.91 | 0.976 | 3.141 | 0.084 | no_solution=0.3 |
| 9 | 0.6 | 17.50 | 29.92 | 1.000 | 3.501 | 0.121 | no_solution=0.4 |
| 12 | 0.2 | 21.00 | 36.35 | 1.000 | 3.624 | 0.119 | no_solution=0.8 |
| 15 | 0.0 | — | — | — | — | — | no_solution=1.0 |

#### Scale 1.3

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 | 2.00 | 10.18 | 0.983 | 3.542 | 0.101 | - |
| 9 | 0.6 | 5.00 | 17.03 | 1.000 | 3.572 | 0.114 | no_solution=0.4 |
| 12 | 0.2 | 2.50 | 17.81 | 1.000 | 3.356 | 0.088 | no_solution=0.7, runtime_limit=0.1 |
| 15 | 0.0 | — | — | — | — | — | no_solution=1.0 |

#### Scale 1.4

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 | 1.80 | 9.86 | 1.000 | 3.639 | 0.103 | - |
| 9 | 0.8 | 5.88 | 18.26 | 0.986 | 3.712 | 0.113 | no_solution=0.2 |
| 12 | 0.3 | 16.67 | 33.25 | 1.000 | 3.745 | 0.132 | no_solution=0.7 |
| 15 | 0.3 | 28.67 | 47.80 | 1.000 | 3.908 | 0.144 | no_solution=0.7 |

#### Scale 1.5

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.9 | 1.33 | 9.14 | 1.000 | 3.726 | 0.102 | no_solution=0.1 |
| 9 | 0.6 | 2.83 | 14.50 | 1.000 | 3.767 | 0.109 | no_solution=0.4 |
| 12 | 0.9 | 18.44 | 34.49 | 0.981 | 4.010 | 0.140 | no_solution=0.1 |
| 15 | 0.5 | 40.60 | 61.03 | 0.987 | 4.180 | 0.161 | no_solution=0.5 |

---

### controlnet_hybrid_v2_bestval — Full Results by Scale

#### Scale 1.0

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.8 | 1.75 | 10.01 | 0.896 | 3.072 | 0.092 | no_solution=0.2 |
| 9 | 0.7 | 2.43 | 15.04 | 0.968 | 3.014 | 0.082 | no_solution=0.3 |
| 12 | 0.2 | 4.00 | 20.84 | 1.000 | 3.143 | 0.092 | no_solution=0.8 |
| 15 | 0.1 | 49.00 | 67.99 | 0.933 | 3.373 | 0.127 | no_solution=0.9 |

#### Scale 1.1

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.9 | 3.33 | 11.60 | 0.981 | 3.116 | 0.088 | no_solution=0.1 |
| 9 | 0.7 | 13.14 | 24.63 | 1.000 | 3.276 | 0.106 | no_solution=0.3 |
| 12 | 0.2 | 13.00 | 28.78 | 1.000 | 3.379 | 0.110 | no_solution=0.8 |
| 15 | 0.1 | 94.00 | 110.33 | 1.000 | 4.004 | 0.186 | no_solution=0.9 |

#### Scale 1.2

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 | 1.40 | 9.43 | 1.000 | 3.196 | 0.086 | - |
| 9 | 0.8 | 6.75 | 18.63 | 1.000 | 3.344 | 0.101 | no_solution=0.2 |
| 12 | 0.2 | 20.00 | 35.85 | 1.000 | 3.631 | 0.130 | no_solution=0.8 |
| 15 | 0.2 | 20.50 | 41.60 | 0.967 | 3.585 | 0.118 | no_solution=0.8 |

#### Scale 1.3

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.9 | 1.33 | 9.40 | 0.963 | 3.497 | 0.096 | no_solution=0.1 |
| 9 | 0.9 | 5.33 | 17.33 | 1.000 | 3.444 | 0.101 | no_solution=0.1 |
| 12 | 0.5 | 6.80 | 23.29 | 1.000 | 3.538 | 0.107 | no_solution=0.5 |
| 15 | 0.3 | 15.33 | 34.89 | 1.000 | 3.607 | 0.114 | no_solution=0.7 |

#### Scale 1.4

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 0.9 | 1.67 | 9.43 | 1.000 | 3.666 | 0.104 | no_solution=0.1 |
| 9 | 0.9 | 2.56 | 14.14 | 1.000 | 3.610 | 0.103 | no_solution=0.1 |
| 12 | 0.7 | 6.86 | 23.47 | 0.988 | 3.704 | 0.111 | no_solution=0.3 |
| 15 | 0.5 | 34.60 | 51.87 | 0.973 | 3.818 | 0.146 | no_solution=0.5 |

#### Scale 1.5

| Agents | Success | CT Expansions | Planning Time (s) | Data Adherence | Path Length | Acceleration | Fail Mode |
|--------|---------|---------------|-------------------|----------------|-------------|--------------|-----------|
| 6 | 1.0 | 2.00 | 10.11 | 1.000 | 3.858 | 0.110 | - |
| 9 | 1.0 | 2.30 | 15.18 | 0.989 | 3.730 | 0.107 | - |
| 12 | 0.7 | 12.71 | 30.07 | 0.988 | 4.147 | 0.148 | no_solution=0.3 |
| 15 | 0.7 | 37.43 | 52.98 | 0.990 | 4.047 | 0.148 | no_solution=0.2, runtime_limit=0.1 |

---

### 15-Agent Comparison by Scale (All Four Variants)

This table extends the Section 7 primary comparison by adding the two hybrid variants.
`—` denotes zero successful trials for that slice (quality metrics are not meaningful).

| Scale | `base` Success | `base` CT | `cnv2` Success | `cnv2` CT | `v1` Success | `v1` CT | `v2` Success | `v2` CT |
|-------|---------------|-----------|----------------|-----------|-------------|---------|-------------|---------|
| 1.0 | 1.0 | 18.8 | 1.0 | 9.2 | 0.0 | — | 0.1 | 49.0 |
| 1.1 | 1.0 | 15.1 | 1.0 | 9.0 | 0.0 | — | 0.1 | 94.0¹ |
| 1.2 | 1.0 | 14.6 | 1.0 | 10.3 | 0.0 | — | 0.2 | 20.5 |
| 1.3 | 1.0 | 26.4 | 1.0 | 10.3 | 0.0 | — | 0.3 | 15.3 |
| 1.4 | 0.9 | 31.1 | 1.0 | 10.7 | 0.3 | 28.7 | 0.5 | 34.6 |
| 1.5 | 0.9 | 52.0 | 0.9 | 15.0 | 0.5 | 40.6 | 0.7 | 37.4 |

`cnv2` = `controlnet_bestval_v2` (original ControlNet, paper defaults).
`v1` = `controlnet_hybrid_v1`. `v2` = `controlnet_hybrid_v2_bestval`.

¹ Scale 1.1 / 15 agents for v2: only 1 of 10 trials succeeded; CT expansion value (94.0) reflects that single very-hard trial only.

---

### Updated Cross-Scale Means (All 24 Combinations, 1.0–1.5)

The `base` and `controlnet_bestval_v2` rows are carried forward from Section 7.
The two hybrid rows are new. Quality metrics (CT expansions, planning time, path length, acceleration, data adherence) are reported as arithmetic means over all 24 aggregated CSV rows; for fully-failed slices (`success_rate=0.0`) those fields are `0.0` in the CSV, which pulls the hybrid means down — see analysis below.

| Mode | Mean Success | Mean CT Exp | Mean Planning Time (s) | Mean Path Length | Mean Acceleration | Mean Data Adherence | Mean No-Solution Rate |
|------|--------------|-------------|------------------------|------------------|-------------------|---------------------|-----------------------|
| `base` | 0.988 | 11.03 | 30.01 | 3.777 | 0.153 | 0.950 | 0.004 |
| `controlnet_bestval_v2` | 0.992 | 5.13 | 25.82 | 3.561 | 0.137 | 0.932 | 0.008 |
| `controlnet_hybrid_v1` | 0.496 | 8.82 | 19.47 | 2.939 | 0.093 | 0.818 | 0.496 |
| `controlnet_hybrid_v2_bestval` | 0.621 | 14.93 | 28.62 | 3.533 | 0.113 | 0.985 | 0.375 |

**Note on data adherence and quality metric means for hybrid variants**: for v1, four slices (scales 1.0–1.3 at 15 agents) have zero successful trials, contributing 0.0 to every quality metric. Restricting to the 20 non-zero slices, v1's conditional data adherence is **0.982** — nearly identical to the original ControlNet. The low unconditional mean (0.818) is entirely a consequence of planning failures, not poor trajectory quality when the planner does succeed.

---

### Analysis

#### 1. Hybrid-Trained Models Perform Significantly Worse Than Original ControlNet

The most striking result is the large gap between the hybrid-data variants and both the base reference and the original ControlNet.

| Comparison | Mean Success Gap |
|------------|-----------------|
| `controlnet_bestval_v2` vs `base` | +0.004 (parity) |
| `controlnet_hybrid_v2_bestval` vs `base` | −0.367 |
| `controlnet_hybrid_v1` vs `base` | −0.492 |

Hybrid-trained models fail primarily through `no_solution`. The base model's failure mode is different: at high scales, it occasionally hits the runtime limit because CBS search grows too large. Hybrid models collapse earlier in the CBS tree rather than exhausting time.

#### 2. Improved Training (v2) Consistently Outperforms Baseline Training (v1)

The five v2 improvements (shuffle, cosine LR, stratified val, best-val checkpoint, AMP) produce clear gains at every scale where v1 also has non-zero success:

| Scale | v1 mean success (4 agents) | v2 mean success (4 agents) |
|-------|---------------------------|---------------------------|
| 1.0 | 0.425 | 0.450 |
| 1.1 | 0.400 | 0.475 |
| 1.2 | 0.375 | 0.550 |
| 1.3 | 0.450 | 0.650 |
| 1.4 | 0.600 | 0.750 |
| 1.5 | 0.725 | 0.850 |

The benefit is largest at the hardest scales (1.3–1.5), suggesting v2's improved training procedure extracts more generalisation signal from the multi-scale dataset.

#### 3. v2 Uniquely Succeeds at env_scale=1.5 with High Reliability

At `env_scale=1.5`, v2 achieves `1.0 / 1.0 / 0.7 / 0.7` success (6/9/12/15 agents), performing comparably to the original `controlnet_bestval_v2` (`1.0 / 1.0 / 0.9 / 0.9`). For 6 and 9 agents at the hardest scale, v2 matches the original ControlNet exactly.

The gap remains pronounced at 12 and 15 agents: v2 achieves 0.7/0.7 vs original 0.9/0.9.

#### 4. Data Adherence Confirms Conditioning Signal Is Working

When hybrid-trained models do succeed, trajectory quality remains high. Conditional data adherence (over successful slices only) is 0.982 for v1 and 0.985 for v2 — matching the original ControlNet's 0.932 or better. The SDF conditioning signal is learned correctly; the problem is that it does not guide the diffusion model strongly enough to reliably avoid `no_solution` failures at the CBS level.

#### 5. Why Hybrid Data May Underperform Diffusion-Generated Data

Several hypotheses explain why training on RRT+GPMP2 trajectories produces weaker ControlNet conditioning than training on diffusion-generated trajectories:

1. **Trajectory distribution mismatch**: classical-planner trajectories may lie outside the base model's learned trajectory manifold. During training, the ControlNet adapter corrects noise injected into out-of-distribution trajectories rather than learning to bias the base model's denoising direction.
2. **Self-distillation concern may be overstated**: the original concern was that diffusion-generated data introduces circularity. However, if the base model already generates near-optimal trajectories, training on them may teach the ControlNet adapter to refine on-manifold corrections, which is exactly the kind of signal it needs at inference time.
3. **Classical-planner diversity is lower**: the hybrid dataset was generated with `rrt_max_time=60s`, resulting in 33–63% context skip rates at larger scales. Fewer unique trajectory patterns per scale may produce a weaker multi-scale signal.

#### 6. Recommendation

- `controlnet_bestval_v2` (original ControlNet, diffusion-generated training data) remains the recommended checkpoint for all scaled Conveyor experiments.
- `controlnet_hybrid_v2_bestval` is a viable checkpoint for `env_scale ≥ 1.3` when only 6–9 agents are needed.
- The self-distillation concern motivating the hybrid dataset does not appear to justify the performance cost. The original training data source (`EnvConveyor2D-RobotPlanarDisk-multiscale/`) is the preferred training set for further ControlNet experiments.
- Next investigation priority: regenerate the hybrid dataset with `rrt_max_time ≥ 300s` and retrain v2-style to test whether improved data quality (fewer skipped contexts) closes the gap.
