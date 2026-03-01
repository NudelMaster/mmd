# ControlNet for MMD Trajectory Diffusion

ControlNet-style adapter for conditioning the MMD trajectory diffusion model on SDF (Signed Distance Field) grids, enabling generalization to scaled environments (1.0x–1.5x).

**Target**: EnvConveyor2D-RobotPlanarDisk

---

## 1. Problem

The original MMD model performs poorly on environments scaled beyond 1.0x. We add a ControlNet adapter that conditions trajectory generation on the environment's SDF grid, so the model can adapt to different obstacle scales without retraining from scratch.

**Key challenge**: Unlike original ControlNet (image→image, same domain), here we condition 1D trajectory sequences on 2D SDF grids — a cross-domain problem with no natural spatial correspondence.

---

## 2. Base TemporalUnet Architecture

Config: `state_dim=4`, `unet_input_dim=32`, `dim_mults=(1,2,4,8)`, `n_support_points=64`, `conditioning_type=None`, `self_attention=False`, `time_emb_dim=32`, `cond_dim=32`.

```
Input [B, 64, 4] → rearrange → [B, 4, 64]

ENCODER (4 levels, skip connections h[0..3])
  Level 0: [B,4,64]  → ResBlock×2(c_emb) → [B,32,64]  → Down → [B,32,32]   h[0]=[B,32,64]
  Level 1: [B,32,32] → ResBlock×2(c_emb) → [B,64,32]  → Down → [B,64,16]   h[1]=[B,64,32]
  Level 2: [B,64,16] → ResBlock×2(c_emb) → [B,128,16] → Down → [B,128,8]   h[2]=[B,128,16]
  Level 3: [B,128,8] → ResBlock×2(c_emb) → [B,256,8]  → Id   → [B,256,8]   h[3]=[B,256,8]

MID BLOCK: [B,256,8] → ResBlock → ResBlock → [B,256,8]

DECODER (3 levels — h[0] is NEVER consumed)
  Level 0: cat(x, h[3]) → [B,512,8]  → ResBlock×2 → Up → [B,128,16]
  Level 1: cat(x, h[2]) → [B,256,16] → ResBlock×2 → Up → [B,64,32]
  Level 2: cat(x, h[1]) → [B,128,32] → ResBlock×2 → Up → [B,32,64]

FINAL: [B,32,64] → Conv1d → [B,4,64] → rearrange → [B, 64, 4]
```

**Key observations**:
- Asymmetric: 4 encoder levels, 3 decoder levels. h[0] skip is never used.
- No attention blocks (`self_attention=False`, `conditioning_type=None`).
- `c_emb = t_emb` — only time conditioning, via FiLM modulation in ResBlocks.
- Two `dim_mults` variants exist: option 0 `(1,2,4)` for Highways/DropRegion/EmptyNoWait, option 1 `(1,2,4,8)` for Empty/Conveyor.

### Verified Base Model Config (All 5 Pretrained Models)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `conditioning_type` | `None` | `context=None` is safe |
| `time_emb_dim` / `cond_dim` | `32` | No context concat |
| `self_attention` | `False` | No attention blocks anywhere |
| `state_dim` | `4` | 2D position + 2D velocity |
| `n_support_points` | `64` | Trajectory horizon |

---

## 3. How ControlNet Works (Original)

The original ControlNet (Zhang et al., 2023) for Stable Diffusion:

1. **Freeze** the pretrained UNet (the "locked copy")
2. **Clone** the encoder + mid blocks as a trainable copy
3. **Zero-initialized convolutions** connect the trainable copy's outputs to the frozen decoder's skip connections
4. A **condition encoder** (trainable CNN) processes the hint (Canny edges, depth, etc.) and adds it to the input
5. At init, zero convs output zeros → the model starts from the base model's behavior

```
Condition (e.g. Canny edges) → hint encoder → add to input
                                                    ↓
                                        Trainable encoder copy
                                                    ↓
                                             Zero convolutions
                                                    ↓
                                    Inject residuals into frozen decoder skips
```

**Key principle**: same-domain conditioning. The condition (2D image) and output (2D image) share the same pixel grid. Our case breaks this — we condition 1D trajectories on 2D SDF grids.

---

## 4. Our Design

### Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Conditioning approach | **Approach B (Global FiLM)** | Simpler; uses existing FiLM mechanism; clean experimental signal |
| Encoder design | **Design 2 (SDF → time embedding)** | Avoids cross-domain mapping (2D→1D); no undefined spatial correspondence |
| Target environment | **EnvConveyor2D** | `dim_mults=(1,2,4,8)`; matches doc traces |
| Self-attention | **None** | Faithful copy of base model (which has `self_attention=False`) |
| Residual count | **3 down + 1 mid** | Matches 3 decoder levels; h[0] is never consumed |
| Scale parameter | **Removed** | SDF already encodes scaled geometry |

### Architecture

```
SDF [B,1,64,64]
     │
     ▼
MapSDFEncoder (trainable CNN, 250K params)
  [B,1,64,64] → Conv(1→32,s2) → Conv(32→64,s2) → Conv(64→256,s2) → Conv(256→256,1×1)
     │
     ▼
[B, 256, 8, 8] → global avg pool → [B, 256] → MLP → sdf_emb [B, 32]
     │
     ▼
cond_emb = t_emb + sdf_emb   [B, 32]
     │
     ├──────────────────────────────────────────┐
     ▼                                          ▼
 CONTROLNET ENCODER (trainable, 3.4M)       FROZEN BASE MODEL (3.9M)
 │                                           │
 │ Level 0: ResBlock×2(cond_emb), Down       │
 │ Level 1: ResBlock×2(cond_emb), Down       │
 │ Level 2: ResBlock×2(cond_emb), Down       │
 │ Level 3: ResBlock×2(cond_emb), Identity   │
 │ Mid:     ResBlock×2(cond_emb)             │
 │                                           │
 │ ZeroConv residuals:                       │ Injection:
 │   level 1 → [B, 64, 32]  ───────────────►│   skip h[1] += residual
 │   level 2 → [B, 128, 16] ───────────────►│   skip h[2] += residual
 │   level 3 → [B, 256, 8]  ───────────────►│   skip h[3] += residual
 │   mid     → [B, 256, 8]  ───────────────►│   mid output += residual
 │                                           │
 │ (Level 0: NO residual — h[0] unused)      │
```

### Parameter Counts

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| MapSDFEncoder (CNN + projection) | 250,848 | Yes |
| Control TimeEncoder | 8,352 | Yes |
| Control encoder (4 levels) | 1,667,072 | Yes |
| Control mid blocks | 1,330,688 | Yes |
| Zero convolutions (3 down + 1 mid) | 152,256 | Yes |
| **Total ControlNet** | **3,409,216** | **Yes** |
| Base TemporalUnet | 3,954,052 | Frozen |

Counts above were re-validated from instantiated modules (EnvConveyor2D settings: `dim_mults=(1,2,4,8)`, `state_dim=4`, `n_support_points=64`):

- `trainable_from_wrapper = 3,409,216`
- `num_trainable_tensors = 138`

### Verified (Smoke Test)

- All shapes correct end-to-end
- Forward + backward pass succeed
- 138 ControlNet params get gradients, 0 base model params get gradients
- Loss computation works

---

## 5. Data Strategy

### Core Principle: Decouple SDF Resolution from Trajectory Resolution

SDF pre-computation is cheap (~5s/scale). Trajectory generation is expensive (full diffusion + guidance). They have different granularities.

| Data Type | Resolution | Count | Storage |
|-----------|-----------|-------|---------|
| **SDF cache** | 0.01 increments | 51 SDFs (1.00–1.50) | ~32MB |
| **Trajectory data** | 6 discrete scales | ~500/scale, ~3K total | TBD |

### SDF Cache

Pre-computed by `scripts/generate_data/precompute_sdfs.py`:

```bash
# Default: 51 SDFs at 0.01 increments
python scripts/generate_data/precompute_sdfs.py

# Just 6 training scales
python scripts/generate_data/precompute_sdfs.py --scales 1.0 1.1 1.2 1.3 1.4 1.5
```

Pipeline: `EnvConveyor2D(scale=X)` → `GridMapSDF` auto-computes → save `sdf_tensor [400,400]`.
At training time: `resize_sdf()` to `[1, 64, 64]`.

Output: `data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache/sdf_scale_{X:.2f}.pt`

### Trajectory Data

Generated by base model + guidance at 6 discrete scales. Directory structure:

```
data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale/
    scale_1.00/
        0/trajs-free.pt    # [N, 64, 4]
    scale_1.10/
        0/trajs-free.pt
    ...
    scale_1.50/
        0/trajs-free.pt
```

Inference for data generation: if scale is provided, use ControlNet (once trained); otherwise use base model + guidance.

### Training-Time Sampling

1. Sample trajectory (has associated scale, e.g. 1.20)
2. Look up pre-computed SDF from dense cache (`sdf_scale_1.20.pt`)
3. Return `{traj_normalized, hard_conds, sdf_grid}` to training loop

At inference, model can generalize to any of the 51 pre-computed SDFs — including scales it never saw trajectories for.

---

## 6. Alternative Designs (For Advisor Discussion)

### Design 1: Hint at Input (Original ControlNet Pattern)

Encode SDF into trajectory shape `[B, 4, 64]`, add to input: `x_ctrl = trajectory + hint`.

**Deferred because**: The 64 trajectory positions are time steps, not spatial locations. Projecting a 2D grid onto a 1D temporal sequence has no natural correspondence — unlike original ControlNet where edges and images share pixel grids.

**Interesting variant**: Positional SDF lookup — at each waypoint, sample SDF at that (x,y) position. Physically meaningful but complex (differentiable grid sampling, changes per denoising step).

### Approach A: Cross-Attention Conditioning

Encode SDF to `[B, 64, 256]` context tokens. Each trajectory position cross-attends to SDF spatial regions via SpatialTransformer.

**Deferred because**: All pretrained models have `self_attention=False` and `conditioning_type=None` — no attention blocks exist. Adding attention to the ControlNet branch creates an architectural mismatch with the frozen base model, breaking the ControlNet principle of "trainable copy mirrors frozen copy."

**If Approach B shows benefit**, Approach A becomes a motivated follow-up with richer spatial conditioning.

| Aspect | Approach A (Cross-Attention) | Approach B (Global FiLM) |
|--------|------------------------------|--------------------------|
| SDF context | `[B, 64, 256]` tokens | `[B, 32]` vector |
| Mechanism | Cross-attention per level | FiLM in ResNet blocks |
| Spatial specificity | Per-position | Global (same for all) |
| Extra params | ~500K (SpatialTransformers) | ~20K (projection MLP) |
| Complexity | Higher | Lower |
| Base model match | Mismatch (adds attention) | Match (same FiLM) |

---

## 7. Implementation Status

### Files

| File | Status | Description |
|------|--------|-------------|
| `mmd/datasets/controlnet_dataset.py` | **Done** | `ControlNetTrajectoryDataset` with scale-aware loading + `sdf_grid` output |
| `mmd/models/helpers/map_encoder.py` | **Done** | `MapSDFEncoder` (Design 2), `resize_sdf()` |
| `mmd/models/diffusion_models/controlnet.py` | **Done** | `ZeroConv1d`, `MMDControlNet`, `ControlledDiffusionModel` |
| `mmd/models/diffusion_models/temporal_unet.py` | **Done** | Injection ports (backward-compatible) |
| `mmd/models/diffusion_models/__init__.py` | **Done** | Exports |
| `scripts/train_diffusion/train_controlnet.py` | **Done** | Training script now uses `ControlNetTrajectoryDataset` + `sdf_cache_dir` |
| `scripts/generate_data/precompute_sdfs.py` | **Done** | SDF pre-computation |
| `scripts/generate_data/generate_multiscale_data.py` | **Done (rewritten)** | Multi-scale trajectory generation aligned with `MMDParams` + inference defaults |

### Remaining (Updated)

| Priority | Task | Notes |
|----------|------|-------|
| **High** | Inference integration with MPD | Wire `ControlledDiffusionModel` into rollout/sampling path |
| **High** | Base vs ControlNet evaluation | Compare metrics across scales 1.0 to 1.5 |
| **Medium** | Overfitting mitigation study | Current run shows widening train/val gap after ~30K steps |
| **Low** | Trainable vs frozen encoder optimization | Works as-is; optimize later if needed |

### Full Multi-Scale Generation (Completed)

Executed command:

```bash
python scripts/generate_data/generate_multiscale_data.py \
    --env_id EnvConveyor2D \
    --model_id EnvConveyor2D-RobotPlanarDisk \
    --scales 1.0 1.1 1.2 1.3 1.4 1.5 \
    --num_trajs_per_scale 500
```

Generated dataset summary:

- Output root: `data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale/`
- Scales generated: `scale_1.00`, `scale_1.10`, `scale_1.20`, `scale_1.30`, `scale_1.40`, `scale_1.50`
- Per-scale tensor shape: `trajs-free.pt` is `[500, 64, 4]` (`float32`)
- Total trajectories: `3000` (6 × 500)
- Args consistency verified:
  - `obstacle_cutoff_margin: 0.02`
  - `threshold_start_goal_pos: 1.0`
  - `env_scale` matches each scale directory
- SDF cache coverage verified for all training scales (`sdf_scale_1.00.pt` ... `sdf_scale_1.50.pt`)
- `ControlNetTrajectoryDataset` load test and DataLoader collation test passed

### Args Compatibility Validation (Completed)

Concern addressed: original dataset args came from a different (classical-planner) generation script, so we validated which fields are actually required for current MPD-based ControlNet training.

- `obstacle_cutoff_margin` and `threshold_start_goal_pos` are now written to match the original EnvConveyor2D dataset conventions:
  - `obstacle_cutoff_margin: 0.02`
  - `threshold_start_goal_pos: 1.0`
- Updated in both:
  - top-level dataset args: `...-multiscale/0/args.yaml`
  - per-scale args: `...-multiscale/scale_X.XX/0/args.yaml`
- Validation outcome for training relevance:
  - Diffusion/ControlNet loss uses trajectory tensors and SDF grids directly.
  - `args.yaml` fields are mainly consumed to construct `PlanningTask` (collision/eval context), not to compute training loss.
  - Keeping these values aligned still avoids silent drift in evaluation/inference behavior.

### Smoke Regeneration (Completed)

Re-ran smoke generation after args fix:

```bash
python scripts/generate_data/generate_multiscale_data.py \
    --env_id EnvConveyor2D \
    --robot_id RobotPlanarDisk \
    --model_id EnvConveyor2D-RobotPlanarDisk \
    --scales 1.0 \
    --num_trajs_per_scale 20 \
    --n_samples 8 \
    --n_guide_steps 5
```

Generated output verifies corrected args values:

- `data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale/0/args.yaml`
- `data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale/scale_1.00/0/args.yaml`

Both contain:

```yaml
env_scale: 1.0
obstacle_cutoff_margin: 0.02
threshold_start_goal_pos: 1.0
```

### Full ControlNet Training (Completed)

Full training was executed on the generated multi-scale dataset:

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_full \
    --batch_size 32 \
    --lr 1e-4 \
    --num_train_steps 100000 \
    --steps_til_summary 100 \
    --steps_til_ckpt 10000 \
    --wandb_mode disabled
```

Key outcomes:

- Total executed steps: `96,030` (derived by epoch conversion from `num_train_steps=100000`)
- Wall-clock time: about `69 minutes` (`19:23 -> 20:32` from checkpoint timestamps)
- Throughput: about `23.8 it/s` (`~1430 steps/min`) at `batch_size=32`, `use_amp=False`
- Train loss: `1.049 -> 0.078` (min observed: `0.052`)
- Validation loss: `1.002 -> 0.134` (min observed: `0.097` at step ~34,700)
- Best saved checkpoint nearest the minimum val step: `checkpoints/controlnet_epoch_0333_iter_030000_state_dict.pth`
- EMA checkpoint saved: `checkpoints/ema_controlnet_final_state_dict.pth`

Observed behavior:

- Validation improves quickly early, then plateaus around `0.12-0.14` after ~20K steps
- Train loss keeps decreasing after validation plateaus
- Train/val gap grows over time, indicating mild overfitting on the 3K-trajectory dataset

### WandB Logging Fixes Applied

Two instrumentation issues were fixed in `scripts/train_diffusion/train_controlnet.py`:

1. **Double `wandb.init()` fixed**
   - `@single_experiment_yaml` already initializes wandb.
   - Explicit `wandb.init()` in the experiment function was removed.
   - Script now uses `wandb.config.update(...)` only.

2. **Validation loss logging fixed**
   - Previously only `train_loss` was logged.
   - Now both `train_loss` and `val_loss` are logged in a merged `wandb.log(...)` call.

### Dataset Integration Implemented

`ControlNetTrajectoryDataset` is now implemented with explicit scale-aware loading:

1. Walks `scale_X.XX/*/trajs-free.pt` directories explicitly (not generic `os.walk` over all folders)
2. Parses `X.XX` scale from directory names and stores per-trajectory scale mapping
3. Loads required SDF files from cache (`sdf_scale_{X:.2f}.pt`)
4. Resizes native SDF grids to `[1, 64, 64]` via `resize_sdf()` at dataset init
5. Returns `{traj_normalized, task_normalized, hard_conds, sdf_grid}` in `__getitem__`

This closes the training-time wiring gap where `batch_dict['sdf_grid']` was missing.

### Validation Run (Pre-Data-Generation)

The following validations were run after implementation and before any data generation:

- Python compile check for modified files (`py_compile`) to catch syntax/import issues
- Import resolution check to verify `ControlNetTrajectoryDataset` is exported via `mmd.datasets`
- Training script wiring check to ensure `get_dataset()` now requests `ControlNetTrajectoryDataset`

No data-generation scripts were executed in this validation phase.

### Bugs Fixed

| Bug | Fix |
|-----|-----|
| `batch_dict[dataset.field_key_traj]` → KeyError | Use `f'{dataset.field_key_traj}_normalized'` |
| `controlnet_hint` always `None` | Use `batch_dict['sdf_grid']` from dataset |
| Concern about `context=None` crash | Verified: `conditioning_type=None` for all models, safe |

---

## References

- **ControlNet Paper**: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models"
- **Base TemporalUnet**: `mmd/models/diffusion_models/temporal_unet.py`
- **MapSDFEncoder**: `mmd/models/helpers/map_encoder.py`
- **ControlNet**: `mmd/models/diffusion_models/controlnet.py`
- **Training Script**: `scripts/train_diffusion/train_controlnet.py`
- **SDF Pre-computation**: `scripts/generate_data/precompute_sdfs.py`

---

## 8. Workflow & Commands

### 1. Pre-compute SDF Cache (Ready)
Generate the dense SDF cache (51 grids, 1.00 to 1.50) used for conditioning.

```bash
# From project root
python scripts/generate_data/precompute_sdfs.py \
    --env_id EnvConveyor2D \
    --scale_min 1.0 --scale_max 1.5 --scale_step 0.01
```

### 2. Generate Multi-Scale Trajectories (Ready)
Generate training trajectories at discrete scales using the **pretrained MMD diffusion model** (as a prior).
*Implemented in `scripts/generate_data/generate_multiscale_data.py` and rewritten to match the inference pipeline parameter wiring (`MMDParams`, `apply_overrides`, repo-root `TRAINED_MODELS_DIR`).*

```bash
python scripts/generate_data/generate_multiscale_data.py \
    --env_id EnvConveyor2D \
    --model_id EnvConveyor2D-RobotPlanarDisk \
    --scales 1.0 1.1 1.2 1.3 1.4 1.5 \
    --num_trajs_per_scale 500
```

Status: **Executed successfully** for all 6 scales; output validated as training-ready.

Optional overrides (same style as inference launcher):

```bash
python scripts/generate_data/generate_multiscale_data.py \
    --model_id EnvConveyor2D-RobotPlanarDisk \
    --scales 1.0 1.1 1.2 1.3 1.4 1.5 \
    --num_trajs_per_scale 500 \
    --weight_grad_cost_collision 0.02 \
    --weight_grad_cost_smoothness 0.08 \
    --weight_grad_cost_constraints 0.2 \
    --weight_grad_cost_soft_constraints 0.02 \
    --start_guide_steps_fraction 0.5 \
    --n_guide_steps 20
```

#### Data Analysis & Tradeoffs
- **Original Dataset**: 10,000 trajectories (all at scale 1.0x).
- **ControlNet Dataset**: 3,000 trajectories (6 scales × 500/scale).
- **Tradeoff**: 500/scale is 20x sparser than the original training data.
  - *Pros*: Faster generation (diffusion is slow); sufficient for residual learning.
  - *Cons*: Risk of overfitting to specific start/goal pairs; potentially poor coverage of 1.5x environments.
  - *Plan*: Start with 500/scale. If generalization is poor, increase to 1,000–2,000.

### 3. Train ControlNet (Ready)
Train the adapter using the pretrained base model and the multi-scale dataset.

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet \
    --batch_size 32 \
    --lr 1e-4
```

#### Training command presets (recommended variants)

**A) Short smoke test (correctness only, quick turnaround)**

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_smoke \
    --batch_size 16 \
    --lr 1e-4 \
    --num_train_steps 40 \
    --steps_til_summary 10 \
    --steps_til_ckpt 20 \
    --wandb_mode disabled
```

Use this to validate data loading, forward/backward, validation pass, and checkpoint writing.

**B) Standard full training run (default baseline)**

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_full \
    --batch_size 32 \
    --lr 1e-4 \
    --num_train_steps 100000 \
    --steps_til_summary 100 \
    --steps_til_ckpt 10000 \
    --wandb_mode online
```

Use this as the primary experiment configuration for the 3K multi-scale dataset.

**C) Memory-safe run (smaller GPU memory footprint)**

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_bs16 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_train_steps 100000 \
    --steps_til_summary 100 \
    --steps_til_ckpt 10000 \
    --wandb_mode online
```

Use this when batch size 32 is unstable/OOM; same optimizer settings, smaller batch.

**D) Throughput-focused run (mixed precision)**

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_amp \
    --batch_size 32 \
    --lr 1e-4 \
    --use_amp True \
    --num_train_steps 100000 \
    --steps_til_summary 100 \
    --steps_til_ckpt 10000 \
    --wandb_mode online
```

Use this for faster training on compatible GPUs; verify numerical stability against run B.

Notes on differences:

- `batch_size`: main memory/performance knob (`32` faster if it fits, `16` safer).
- `num_train_steps`: target step budget used to derive epochs from dataset size.
- `use_amp`: enables mixed precision (`torch.autocast`) to improve throughput.
- `steps_til_summary` / `steps_til_ckpt`: monitoring/checkpoint frequency (lower values = denser logging, more I/O).
- `wandb_mode`: `disabled` for smoke testing, `online` for full tracked experiments.

### 4. Next Steps (Immediate)

1. Integrate ControlNet into MPD inference path (scale/SDF-conditioned denoising at test time).
2. Run base vs ControlNet evaluation at scales `1.0` to `1.5`:
   - base model + guidance (baseline)
   - base + ControlNet (conditioned on SDF)
3. Report per-scale metrics (collision-free rate, quality/smoothness, success rate).
4. If overfitting persists in evaluation, test mitigations:
   - increase trajectory diversity per scale (`n_samples` lower during generation)
   - early stopping around the best validation region (~30K-40K steps)
   - regularization/data augmentation as needed

Recommended first inference checkpoint order:

1. `checkpoints/ema_controlnet_final_state_dict.pth` (usually best generalization)
2. `checkpoints/controlnet_epoch_0333_iter_030000_state_dict.pth` (nearest best-val region)
3. final non-EMA checkpoint (for ablation)

---

## 9. Training Time Estimates (Measured + Projected)

### Measured Baseline (Run B)

Measured on the completed full run (`logs_controlnet_full/0`):

- Configuration: `batch_size=32`, `use_amp=False`
- Effective executed steps: `96,030`
- Wall-clock: about `69 minutes`
- Throughput: about `23.8 it/s` (`~1430 steps/min`)
- Rule of thumb from measured run: about `7.0 minutes per 10,000 steps`

### Preset Runtime Estimates

The table below gives practical estimates for the four command presets in Section 8.3.

| Preset | Batch Size | AMP | Target Steps | Time Estimate | Basis |
|--------|------------|-----|--------------|---------------|-------|
| A (Smoke) | 16 | No | 40 | `< 1 min` (typically 10-30 s) | direct scaling from measured throughput |
| B (Standard) | 32 | No | 100,000 | about `70 min` (measured run finished in 69 min at 96,030 steps) | measured |
| C (Memory-safe) | 16 | No | 100,000 | about `90-100 min` | projected (smaller batch + more dataloader/validation overhead) |
| D (AMP) | 32 | Yes | 100,000 | about `50-60 min` | projected (typical AMP speedup ~1.2-1.4x) |

Important caveats:

- Exact time depends on GPU model, clock state, and system load.
- Validation cadence (`steps_til_summary`) and number of validation batches add overhead.
- `num_train_steps` is converted to epochs and then run as full epochs (`get_num_epochs`), so actual executed steps can differ from the target.
- Use run B as the anchor for this machine; recalibrate if hardware changes.

---

## 10. Training Script Workflow (`train_controlnet.py`)

This section explains the actual execution path of `scripts/train_diffusion/train_controlnet.py`.

### Entry and Experiment Setup

1. `experiment()` is decorated with `@single_experiment_yaml`.
2. The decorator handles results-directory setup and wandb startup.
3. Script updates wandb metadata via `wandb.config.update(...)` (no explicit `wandb.init()`).
4. Random seed is fixed (`fix_random_seed(seed)`), and `tensor_args` are built.

### Dataset Loading

1. `get_dataset(...)` is called with `dataset_class='ControlNetTrajectoryDataset'`.
2. Dataset returns per-batch fields including:
   - trajectory: `{field_key_traj}_normalized`
   - `hard_conds`
   - `sdf_grid` (`[B, 1, 64, 64]`)
3. Train/val subset indices are saved under the run directory.

### Model Construction

1. Build `TemporalUnet` with settings matching pretrained model (`dim_mults=(1,2,4,8)` for Conveyor).
2. Wrap in `GaussianDiffusionModel`.
3. Load pretrained checkpoint (`ema_model_current_state_dict.pth` by default).
4. Create `MMDControlNet(base_model=diffusion_model.model, cond_dim=32, ...)`.
5. Wrap both inside `ControlledDiffusionModel`.
6. Base diffusion model is frozen; optimizer receives only ControlNet parameters.

### Training Loop

For each batch:

1. Move batch to device (`dict_to_device`).
2. Compute loss via `controlnet_loss_fn(...)`, which calls:
   - `controlled_model.loss(traj, hard_conds, sdf_grid)`
3. Backpropagate, apply gradient clipping, optimizer step, optional AMP scaler updates.
4. Update EMA ControlNet copy every `update_ema_every` steps.
5. Every `steps_til_summary`:
   - print train metrics
   - run short validation pass (up to 11 val mini-batches)
   - log `train_loss` and `val_loss` to wandb
6. Every `steps_til_ckpt`:
   - save ControlNet checkpoint(s)
   - save EMA state dict
   - save loss arrays (`train_losses.npy`, `val_losses.npy`)

### Epoch/Step Accounting

`num_train_steps` is converted with:

```
epochs = ceil(num_train_steps * batch_size / dataset_len)
```

Implication: training completes whole epochs, so actual step count may be slightly lower or higher than the requested `num_train_steps` target.

### Artifacts Saved Per Run

- `args.yaml` (full run configuration)
- `checkpoints/controlnet_current_state_dict.pth`
- versioned checkpoints: `controlnet_epoch_XXXX_iter_XXXXXX_state_dict.pth`
- `checkpoints/ema_controlnet_current_state_dict.pth` and final EMA state dict
- `checkpoints/train_losses.npy`, `checkpoints/val_losses.npy`
- split indices (`train_subset_indices.pt`, `val_subset_indices.pt`)

---

## 11. WandB Usage Guide

### Enable/Disable

- Disabled (local or smoke): `--wandb_mode disabled`
- Online tracking: `--wandb_mode online`

Example full run with wandb enabled:

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache \
    --results_dir logs_controlnet_full_wandb \
    --batch_size 32 \
    --lr 1e-4 \
    --num_train_steps 100000 \
    --steps_til_summary 100 \
    --steps_til_ckpt 10000 \
    --wandb_mode online
```

### Expected Project Metadata

- `wandb_project`: `mmd_controlnet`
- `wandb_entity`: `scoreplan`

These defaults come from the experiment signature and can be overridden if needed.

### What Gets Logged

At each summary step (`steps_til_summary`):

- `train_loss`
- `val_loss` (when validation dataloader is enabled)
- step index (`train_steps_current`) as wandb step

### Common Pitfalls

1. Do not add manual `wandb.init()` in this script; decorator already initializes wandb.
2. Keep `steps_til_summary` reasonable (too small adds overhead and noisy curves).
3. Ensure network/auth is configured before `--wandb_mode online` runs.

### Recommended Tracking Practice

For reproducible comparisons, keep one run per clear configuration change:

- baseline full run (B)
- memory-safe comparison (C)
- AMP comparison (D)
- any overfitting mitigation run (data size, early stopping, regularization)

Name `results_dir` accordingly so local artifacts align with wandb runs.
