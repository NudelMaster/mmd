# ControlNet for MMD Trajectory Diffusion Architecture and Integration

This document covers the ControlNet architecture design, implementation status, training workflow, and inference integration for MMD trajectory diffusion.
Benchmark plans, results, and follow-up experiment analysis live in `CONTROLNET_EXPERIMENTS.md`.

ControlNet-style adapter for conditioning the MMD trajectory diffusion model on SDF (Signed Distance Field) grids, enabling generalization to scaled environments (1.0x-1.5x).

**2026-04-24 spatial redesign status**: the original `MapSDFEncoder -> global pool -> cond_emb = t_emb + sdf_emb` path is now treated as `legacy_global` checkpoint compatibility. New ControlNet training should use `conditioning_mode='spatial'`: a 2-channel `[sdf_cur, delta_sdf]` SDF cache is encoded by `SDFSpatialPyramid2D`, queried at current noisy waypoint positions with `grid_sample`, and injected into copied ControlNet encoder/mid blocks through zero-initialized map adapters before the existing zero-conv residual bridges. This restores the ControlNet/T2I-adapter invariant that spatial hints enter at matching feature resolutions while the frozen backbone remains unchanged at initialization.

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

## 4. Current Design: Spatial SDF ControlNet

The active ControlNet design is now a spatial SDF adapter that follows the T2I-adapter/ControlNet invariant more closely than the first global FiLM prototype: spatial hints are encoded as feature maps, sampled at the current noisy waypoint positions, and injected at matching ControlNet feature resolutions while the frozen base UNet remains unchanged at initialization.

The core architectural shift is:

```text
Old:
SDF grid -> global pooled embedding -> cond_emb = t_emb + sdf_emb

New:
2-channel SDF grid -> spatial feature pyramid -> waypoint-local grid_sample -> zero-init adapters inside copied ControlNet branch
```

### Motivation

The old global FiLM path collapses geometry into one vector. That can say "this map is different", but it loses where obstacles changed relative to the current trajectory. Spatial conditioning should reshape the learned trajectory prior locally, not overpower it with generic map guidance.

The current design keeps spatial hints at matching feature resolutions, which is closer to the original ControlNet/T2I-adapter pattern. The extra cross-domain step is the waypoint sampler: it converts 2D map features into 1D temporal feature sequences by sampling the map at the current noisy waypoint coordinates.

### High-Level Architecture

```text
SDF cache [B,2,256,256]
  │
  ▼
SDFSpatialPyramid2D
  ├── P32 [B,64,32,32]
  ├── P16 [B,128,16,16]
  └── P8  [B,256,8,8]

x_t [B,64,4]
  │
  └── x_t[..., :2]
       -> detach/clamp
       -> unnormalize with traj_pos_mins/maxs
       -> normalize with sdf_limits
       -> grid_sample coords

grid_sample(P32) -> S32 [B,64,32]
grid_sample(P16) -> S16 [B,128,16]
grid_sample(P8)  -> S8  [B,256,8]

S32/S16/S8
  -> zero_map_adapters
  -> copied ControlNet encoder/mid
  -> zero_conv residuals
  -> frozen TemporalUnet decoder/mid injection
```

### Conditioning Modes

| Mode | Purpose | Cache | Encoder | Status |
|------|---------|-------|---------|--------|
| `legacy_global` | Old checkpoint compatibility | single-channel SDF cache, or old map tensors through global encoder | `MapSDFEncoder` | legacy |
| `spatial` | New architecture | `[sdf_cur, delta_sdf]`, 256x256 | `SDFSpatialPyramid2D` + `WaypointFeatureSampler` | current |

Default split:

- `MMDControlNet` defaults to `legacy_global` to preserve old-checkpoint construction compatibility.
- `scripts/train_diffusion/train_controlnet.py` defaults to `spatial` for new training.
- Inference launchers default to `legacy_global` unless the run explicitly selects `--controlnet_conditioning_mode spatial`.

### 2-Channel SDF Cache Contract

Spatial mode uses the current cache produced by `scripts/generate_data/precompute_sdf_channels.py`.

| Field | Shape/Value | Meaning |
|-------|-------------|---------|
| `channels` | `[2,256,256]` | SDF input tensor |
| `channels[0]` | `[256,256]` | `sdf_cur` |
| `channels[1]` | `[256,256]` | `delta_sdf = sdf_cur - sdf_ref` |
| `limits` | `[2,2]` | workspace lower/upper bounds |
| `target_size` | `256` | spatial grid resolution |
| `channel_order` | `['sdf_cur', 'delta_sdf']` | validation contract |
| `version` | `1` | cache format version |

Producer:

```text
scripts/generate_data/precompute_sdf_channels.py
```

Consumers:

```text
mmd/datasets/controlnet_dataset.py
mmd/planners/single_agent/mpd.py
```

The reference SDF is sampled from `EnvConveyor2D(scale=1.0)`. The delta channel is zero at the reference scale and highlights scale-dependent geometry changes at larger scales.

### Coordinate Conversion

The waypoint sampler does not treat normalized trajectory coordinates as SDF-grid coordinates. It performs the explicit conversion below:

```text
x_t[..., :2] normalized
-> detach
-> clamp to [-1,1] for lookup only
-> world = (x_norm + 1) / 2 * (traj_pos_maxs - traj_pos_mins) + traj_pos_mins
-> grid = 2 * (world - limits_min) / (limits_max - limits_min) - 1
```

Rationale:

- Detach avoids making conditioning lookup a second optimization route through the noisy trajectory coordinates.
- Clamp stabilizes high-noise diffusion steps while only affecting the map lookup path.
- Trajectory normalization uses dataset empirical mins/maxs, not `env.limits`; direct normalized-coordinate lookup would silently sample wrong SDF locations.
- `align_corners=True` matches endpoint-inclusive SDF cache generation with `torch.linspace`.
- `padding_mode='border'` avoids injecting fake zero features when noisy high-t samples leave the map.

### Spatial Pyramid Encoder

`SDFSpatialPyramid2D` encodes `[sdf_cur, delta_sdf]` maps without destroying spatial locality.

| Layer | Output |
|-------|--------|
| input | `[B,2,256,256]` |
| conv128 | `[B,32,128,128]` |
| conv64 | `[B,64,64,64]` |
| `P32` | `[B,64,32,32]` |
| `P16` | `[B,128,16,16]` |
| `P8` | `[B,256,8,8]` |

It is implemented in `mmd/models/helpers/map_encoder.py` and uses stride-2 `Conv2d`, `GroupNorm`, and `SiLU`. `P32/P16/P8` match the true temporal feature sites `[B,64,32]`, `[B,128,16]`, and `[B,256,8]` after waypoint sampling.

### Waypoint Feature Sampling

`WaypointFeatureSampler` converts 2D pyramid maps into 1D temporal feature sequences by sampling at the current noisy trajectory waypoint positions.

| Pyramid Feature | Sample Length | Output | Injection Site |
|-----------------|--------------:|--------|----------------|
| `P32` | 32 | `S32 [B,64,32]` | level 1 |
| `P16` | 16 | `S16 [B,128,16]` | level 2 |
| `P8` | 8 | `S8 [B,256,8]` | level 3 |
| `P8` | 8 | `S8 [B,256,8]` | mid |

The trajectory coordinate sequence is linearly interpolated from 64 waypoints to 32/16/8 lookup lengths. This approximates temporal locality through the stride-2 encoder and makes conditioning path-dependent: each denoising step samples map features at the current `x_t` waypoints.

### ControlNet Branch Injection

Spatial features are injected inside the copied ControlNet branch, not directly into the frozen base model.

```text
Level 0:
  copied ResBlock 4 -> 32
  copied ResBlock 32 -> 32
  no spatial injection
  no residual output because h[0] is unused

Level 1:
  copied ResBlock 32 -> 64
  + zero_map_adapter(S32)
  copied ResBlock 64 -> 64
  zero_conv_down -> h[1] residual

Level 2:
  copied ResBlock 64 -> 128
  + zero_map_adapter(S16)
  copied ResBlock 128 -> 128
  zero_conv_down -> h[2] residual

Level 3:
  copied ResBlock 128 -> 256
  + zero_map_adapter(S8)
  copied ResBlock 256 -> 256
  zero_conv_down -> h[3] residual

Mid:
  copied mid_block1
  + zero_map_adapter_mid(S8)
  copied mid_block2
  zero_conv_mid -> mid residual
```

Injection happens after `resnet1` because before `resnet1` the channel dimensions do not match sampled map features. After `resnet1`, the trajectory feature channel count exactly matches the sampled map channel count at each level.

### Zero-Initialization Invariants

The implementation preserves the core ControlNet invariants:

| ControlNet Principle | Implementation |
|----------------------|----------------|
| Frozen base model | Base `TemporalUnet` unchanged |
| Trainable copied branch | Control encoder/mid blocks copied from base |
| Condition encoder | `SDFSpatialPyramid2D` |
| Zero-effect initialization | zero map adapters + zero global gate + zero residual convs |
| Residual injection | existing `down_block_additional_residuals` and `mid_block_additional_residual` hooks |
| Same noisy input to both branches | `x_t` feeds base and ControlNet branch |

Important initialization details:

- base encoder/mid weights are copied into the ControlNet branch
- `zero_map_adapters_*` start at zero
- `zero_conv_*` residual bridges start at zero
- optional global context gate starts at zero
- a fresh wrapper output equals the base output at initialization

The staged-gradient behavior is expected with double zero initialization: outer zero convs receive gradients first, map adapters receive gradients after zero convs update, and the SDF pyramid receives gradients after map adapters update.

### Training Wiring

Training entry point:

```text
scripts/train_diffusion/train_controlnet.py
```

Current spatial defaults:

```text
controlnet_conditioning_mode='spatial'
sdf_in_channels=2
use_zero_gated_global_context=True
```

Required spatial cache input:

```text
--sdf_channels_cache_dir <path-to-2ch-cache>
```

Batch fields consumed by the spatial loss path:

```text
traj_normalized
hard_conds
sdf_grid
sdf_limits
sdf_scale
traj_pos_mins
traj_pos_maxs
```

The diffusion loss remains unchanged. Only the conditioning inputs changed: `ControlledDiffusionModel.loss(...)` now receives `sdf_grid`, `sdf_limits`, `traj_pos_mins`, and `traj_pos_maxs` so waypoint sampling can be computed in the correct coordinate frame.

Training saves `controlnet_config.yaml` so inference can reconstruct the correct architecture before loading checkpoints. The config records the conditioning mode, SDF channel count, pyramid channels, normalizer stats, and SDF metadata.

### Inference Wiring

Inference is wired through:

```text
mmd/planners/single_agent/mpd.py
mmd/models/diffusion_models/controlnet.py::ControlNetInferenceWrapper
```

Spatial inference behavior:

- `MPD` loads the SDF channels cache for the current `env_scale`.
- spatial mode validates `version`, `channel_order`, `target_size`, and `limits` metadata.
- `ControlNetInferenceWrapper` computes `P32/P16/P8` once because the map is fixed for one planner instance.
- The wrapper still samples cached pyramid maps every denoising step because `x_t` changes every step.
- residuals are scaled by `control_scale`.
- the frozen base UNet is called with `down_block_additional_residuals` and `mid_block_additional_residual`.

### Files Added or Updated

| File | Role |
|------|------|
| `mmd/models/helpers/map_encoder.py` | spatial pyramid, sampler, coordinate helpers |
| `mmd/models/diffusion_models/controlnet.py` | conditioning modes, spatial branch, wrappers |
| `mmd/datasets/controlnet_dataset.py` | SDF metadata validation and batch fields |
| `scripts/train_diffusion/train_controlnet.py` | spatial training defaults and metadata saving |
| `mmd/planners/single_agent/mpd.py` | spatial inference cache loading and wrapper setup |
| `mmd/common/experiments/experiments.py` | config propagation |
| `scripts/inference/inference_multi_agent.py` | trial-to-MPD propagation |
| `scripts/inference/launch_multi_agent_experiment.py` | CLI flag |
| `scripts/inference/launch_controlnet_evaluation.py` | evaluation CLI/metadata |
| `scripts/inference/launch_controlnet_benchmark.py` | benchmark command generation |
| `mmd/config/mmd_params.py` | default ControlNet config attrs |

### Verification Performed

Syntax verification target:

```bash
python3 -m py_compile \
    mmd/models/helpers/map_encoder.py \
    mmd/models/helpers/sdf_channels.py \
    mmd/models/diffusion_models/controlnet.py \
    mmd/datasets/controlnet_dataset.py \
    mmd/planners/single_agent/mpd.py \
    mmd/common/experiments/experiments.py \
    scripts/generate_data/precompute_sdf_channels.py \
    scripts/train_diffusion/train_controlnet.py \
    scripts/inference/inference_multi_agent.py \
    scripts/inference/launch_multi_agent_experiment.py \
    scripts/inference/launch_controlnet_evaluation.py \
    scripts/inference/launch_controlnet_benchmark.py
```

Spatial smoke tests covered:

- pyramid shapes passed
- sampler shapes passed
- residual shapes passed
- zero residuals at initialization passed
- wrapper output exactly equals base output at initialization passed
- legacy mode still constructs and forwards
- staged gradient test passed

### Next Implementation Steps

1. Build the new 2-channel SDF cache with `precompute_sdf_channels.py`.
2. Run a dataset load smoke test with `ControlNetTrajectoryDataset` and the new cache.
3. Run a short spatial ControlNet training smoke test.
4. Verify `controlnet_config.yaml` is saved and can reconstruct architecture.
5. Run a one-trial spatial MPD inference smoke test.
6. Only after architecture smoke passes, start real training.
7. Then benchmark against base and legacy checkpoints using matched seeds.

---

## 5. Legacy Design: Global FiLM ControlNet

This section documents the first ControlNet prototype and the completed experiments based on it. It is now the `legacy_global` mode in code, retained for loading and evaluating existing checkpoints. It should not be treated as the current design for new spatial ControlNet training.

### Design Choices

Legacy/global choices retained for historical context:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Conditioning approach | **Approach B (Global FiLM, legacy)** | Simpler; used existing FiLM mechanism; clean first experimental signal |
| Encoder design | **Design 2 (SDF → time embedding, legacy)** | Avoided cross-domain mapping (2D→1D); no undefined spatial correspondence |
| Target environment | **EnvConveyor2D** | `dim_mults=(1,2,4,8)`; matches doc traces |
| Self-attention | **None** | Faithful copy of base model (which has `self_attention=False`) |
| Residual count | **3 down + 1 mid** | Matches 3 decoder levels; h[0] is never consumed |
| Scale parameter | **Removed** | SDF already encodes scaled geometry |

### Legacy Architecture

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

### Legacy Parameter Counts

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

### Legacy Verified (Smoke Test)

- All shapes correct end-to-end
- Forward + backward pass succeed
- 138 ControlNet params get gradients, 0 base model params get gradients
- Loss computation works

---

## 6. Data Strategy

### Core Principle: Decouple SDF Resolution from Trajectory Resolution

SDF pre-computation is cheap (~5s/scale). Trajectory generation is expensive (full diffusion + guidance). They have different granularities.

| Data Type | Resolution | Count | Storage |
|-----------|-----------|-------|---------|
| **Legacy SDF cache** | 0.01 increments | 51 single-channel SDFs (1.00–1.50) | `sdf_scale_*.pt` |
| **Spatial SDF channels cache** | 0.01 increments | 51 two-channel maps (1.00–1.50) | `sdf_channels_scale_*.pt` |
| **Trajectory data** | 6 discrete scales | ~500/scale, ~3K total | `scale_X.XX/*/trajs-free.pt` |

### SDF Cache

The legacy single-channel cache is pre-computed by `scripts/generate_data/precompute_sdfs.py` and remains supported for `conditioning_mode='legacy_global'`:

```bash
# Default: 51 SDFs at 0.01 increments
python scripts/generate_data/precompute_sdfs.py

# Just 6 training scales
python scripts/generate_data/precompute_sdfs.py --scales 1.0 1.1 1.2 1.3 1.4 1.5
```

Pipeline: `EnvConveyor2D(scale=X)` → `GridMapSDF` auto-computes → save `sdf_tensor [400,400]`.
At training time: `resize_sdf()` to `[1, 64, 64]`.

Output: `data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache/sdf_scale_{X:.2f}.pt`

Spatial mode uses the 2-channel cache generated by `scripts/generate_data/precompute_sdf_channels.py`:

```bash
python scripts/generate_data/precompute_sdf_channels.py \
    --env_id EnvConveyor2D \
    --scale_min 1.0 --scale_max 1.5 --scale_step 0.01 \
    --target_size 256
```

Pipeline: `EnvConveyor2D(scale=X, precompute_sdf_obj_fixed=False)` → analytic grid sampling at `256x256` → save `{channels, limits, scale, target_size, channel_order, version}`.

Output: `data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256/sdf_channels_scale_{X:.2f}.pt`

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
2. Look up pre-computed SDF from cache:
   - spatial mode: `sdf_channels_scale_1.20.pt` with `[sdf_cur, delta_sdf]`
   - legacy mode: `sdf_scale_1.20.pt`
3. Return `{traj_normalized, hard_conds, sdf_grid, sdf_limits, traj_pos_mins, traj_pos_maxs}` to training loop

At inference, spatial mode can generalize to any of the 51 pre-computed 2-channel SDF maps — including scales it never saw trajectories for — because the map is sampled at the denoising-time waypoint positions.

---

## 7. Alternative Designs (For Advisor Discussion)

### Design 1: Hint at Input (Original ControlNet Pattern)

Encode SDF into trajectory shape `[B, 4, 64]`, add to input: `x_ctrl = trajectory + hint`.

**Deferred because**: The 64 trajectory positions are time steps, not spatial locations. Projecting a 2D grid onto a 1D temporal sequence has no natural correspondence — unlike original ControlNet where edges and images share pixel grids.

**Implemented variant**: Spatial mode now uses positional SDF feature lookup — at each denoising step, it samples SDF pyramid features at the current noisy waypoint positions. The remaining deferred part of this design is adding richer cross-attention rather than sampled additive adapters.

### Approach A: Cross-Attention Conditioning

Encode SDF to `[B, 64, 256]` context tokens. Each trajectory position cross-attends to SDF spatial regions via SpatialTransformer.

**Deferred because**: All pretrained models have `self_attention=False` and `conditioning_type=None` — no attention blocks exist. Adding attention to the ControlNet branch creates an architectural mismatch with the frozen base model, breaking the ControlNet principle of "trainable copy mirrors frozen copy."

**Current status**: spatial SDF feature lookup is now implemented without attention. Approach A remains a possible follow-up if sampled additive adapters are not expressive enough.

| Aspect | Approach A (Cross-Attention) | Approach B (Global FiLM) |
|--------|------------------------------|--------------------------|
| SDF context | `[B, 64, 256]` tokens | `[B, 32]` vector |
| Mechanism | Cross-attention per level | FiLM in ResNet blocks |
| Spatial specificity | Per-position | Global (same for all) |
| Extra params | ~500K (SpatialTransformers) | ~20K (projection MLP) |
| Complexity | Higher | Lower |
| Base model match | Mismatch (adds attention) | Match (same FiLM) |

---

## 8. Implementation Status

### Files

| File | Status | Description |
|------|--------|-------------|
| `mmd/datasets/controlnet_dataset.py` | **Done** | `ControlNetTrajectoryDataset` with scale-aware loading, legacy/spatial cache support, `sdf_grid` + spatial metadata output |
| `mmd/models/helpers/map_encoder.py` | **Done** | Legacy `MapSDFEncoder`, `resize_sdf()`, spatial `SDFSpatialPyramid2D`, `WaypointFeatureSampler` |
| `mmd/models/helpers/sdf_channels.py` | **Done** | Raw SDF channel helpers for `[sdf_cur, delta_sdf]` |
| `mmd/models/diffusion_models/controlnet.py` | **Done** | `ZeroConv1d`, `MMDControlNet`, `ControlledDiffusionModel`, `ControlNetInferenceWrapper`; supports `legacy_global` and `spatial` modes |
| `mmd/models/diffusion_models/temporal_unet.py` | **Done** | Injection ports (backward-compatible) |
| `mmd/models/diffusion_models/__init__.py` | **Done** | Exports |
| `scripts/train_diffusion/train_controlnet.py` | **Done** | Training script defaults to spatial mode and requires `sdf_channels_cache_dir` for spatial training |
| `scripts/generate_data/precompute_sdfs.py` | **Done** | SDF pre-computation |
| `scripts/generate_data/precompute_sdf_channels.py` | **Done** | 2-channel spatial SDF cache pre-computation |
| `scripts/generate_data/generate_multiscale_data.py` | **Done (rewritten)** | Multi-scale trajectory generation aligned with `MMDParams` + inference defaults |

### Status (Updated)

| Priority | Task | Notes |
|----------|------|-------|
| **Done** | Spatial SDF ControlNet implementation | `conditioning_mode='spatial'` path implemented for training and inference |
| **Done** | Inference integration with MPD | Completed; see Section 14 |
| **Done** | Base vs ControlNet evaluation | Completed; see `CONTROLNET_EXPERIMENTS.md` Sections 2 to 4 |
| **Done** | Scale 1.5 stress-test | Completed; see `CONTROLNET_EXPERIMENTS.md` Sections 6 to 7 |
| **Medium** | Spatial benchmark runs | Run spatial checkpoints through the benchmark/evaluation launchers with `--controlnet_conditioning_mode spatial` |
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

### Full Legacy ControlNet Training (Completed)

Full training was executed on the generated multi-scale dataset for the legacy-global adapter:

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
3. Loads required SDF files from cache:
   - legacy: `sdf_scale_{X:.2f}.pt`, resized to `[1, 64, 64]` via `resize_sdf()`
   - spatial: `sdf_channels_scale_{X:.2f}.pt`, validated as `[2, 256, 256]` with `channel_order=['sdf_cur', 'delta_sdf']`
4. Stores per-scale `sdf_limits`, `scale`, and `target_size` metadata
5. Returns `{traj_normalized, task_normalized, hard_conds, sdf_grid, sdf_limits, sdf_scale, traj_pos_mins, traj_pos_maxs}` in `__getitem__`

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
- **SDF Channels Pre-computation**: `scripts/generate_data/precompute_sdf_channels.py`

---

## 9. Workflow & Commands

### 1. Pre-compute Spatial SDF Channels Cache (Ready)
Generate the dense 2-channel SDF cache (51 maps, 1.00 to 1.50) used by the current spatial ControlNet.

```bash
# From project root
python scripts/generate_data/precompute_sdf_channels.py \
    --env_id EnvConveyor2D \
    --scale_min 1.0 --scale_max 1.5 --scale_step 0.01 \
    --target_size 256
```

Legacy global checkpoints still use the older single-channel cache from `precompute_sdfs.py`.

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
Train the spatial adapter using the pretrained base model, the multi-scale dataset, and the 2-channel SDF cache.

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_channels_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
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
    --sdf_channels_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
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
    --sdf_channels_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
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
    --sdf_channels_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
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
    --sdf_channels_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
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
- `controlnet_conditioning_mode`: use `spatial` for the current design; use `legacy_global` only for old checkpoints.
- `sdf_channels_cache_dir`: required for spatial training; must contain 2-channel `[sdf_cur, delta_sdf]` cache files.
- `steps_til_summary` / `steps_til_ckpt`: monitoring/checkpoint frequency (lower values = denser logging, more I/O).
- `wandb_mode`: `disabled` for smoke testing, `online` for full tracked experiments.

### 4. Next Steps (Immediate)

1. Train a spatial ControlNet checkpoint from the 2-channel SDF cache.
2. Run base vs spatial ControlNet evaluation at scales `1.0` to `1.5` with `--controlnet_conditioning_mode spatial`.
3. Report per-scale metrics (collision-free rate, quality/smoothness, success rate) in `CONTROLNET_EXPERIMENTS.md`.
4. Compare spatial results against the historical `legacy_global` runs before deciding whether to keep the optional zero-gated global context.

Recommended historical legacy checkpoint order when reproducing old runs:

1. `checkpoints/ema_controlnet_final_state_dict.pth` (usually best generalization)
2. `checkpoints/controlnet_epoch_0333_iter_030000_state_dict.pth` (nearest best-val region)
3. final non-EMA checkpoint (for ablation)

---

## 10. Training Time Estimates (Measured + Projected)

The measured values in this section are from the completed `legacy_global` run. They remain useful for rough sizing, but spatial mode adds the SDF pyramid and waypoint sampling path, so recalibrate runtime from the first spatial full run.

### Measured Baseline (Run B)

Measured on the completed full run (`logs_controlnet_full/0`):

- Configuration: `batch_size=32`, `use_amp=False`
- Effective executed steps: `96,030`
- Wall-clock: about `69 minutes`
- Throughput: about `23.8 it/s` (`~1430 steps/min`)
- Rule of thumb from measured run: about `7.0 minutes per 10,000 steps`

### Preset Runtime Estimates

The table below gives practical estimates for the four command presets in Section 9.3, anchored to the historical global run.

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

## 11. Training Script Workflow (`train_controlnet.py`)

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
   - `sdf_grid` (`[B, 2, 256, 256]` in spatial mode; `[B, 1, 64, 64]` in legacy mode)
   - `sdf_limits`
   - `traj_pos_mins`, `traj_pos_maxs`
3. Train/val subset indices are saved under the run directory.
4. The current script uses a scale-aware validation split: each scale contributes `max(1, floor(n_scale * 0.05))` validation samples.

### Model Construction

1. Build `TemporalUnet` with settings matching pretrained model (`dim_mults=(1,2,4,8)` for Conveyor).
2. Wrap in `GaussianDiffusionModel`.
3. Load pretrained checkpoint (`ema_model_current_state_dict.pth` by default).
4. Create `MMDControlNet(base_model=diffusion_model.model, cond_dim=32, conditioning_mode=controlnet_conditioning_mode, sdf_in_channels=sdf_in_channels, ...)`.
5. Wrap both inside `ControlledDiffusionModel`.
6. Base diffusion model is frozen; optimizer receives only ControlNet parameters.
7. Spatial mode validates `sdf_in_channels == 2` and requires `sdf_channels_cache_dir`.

### Training Loop

For each batch:

1. Move batch to device (`dict_to_device`).
2. Compute loss via `controlnet_loss_fn(...)`, which calls:
   - `controlled_model.loss(traj, hard_conds, sdf_grid, sdf_limits, traj_pos_mins, traj_pos_maxs)`
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

## 12. WandB Usage Guide

### Enable/Disable

- Disabled (local or smoke): `--wandb_mode disabled`
- Online tracking: `--wandb_mode online`

Example full run with wandb enabled:

```bash
python scripts/train_diffusion/train_controlnet.py \
    --pretrained_model_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk \
    --dataset_subdir EnvConveyor2D-RobotPlanarDisk-multiscale \
    --sdf_channels_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
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
- spatial ablations (`use_zero_gated_global_context`, `control_scale`, data size, early stopping, regularization)

Name `results_dir` accordingly so local artifacts align with wandb runs.

---

## 13. Training Data, Split, and Log Artifacts

This section records the exact data and logging setup used for the completed legacy-global ControlNet runs, so future comparisons are made against the correct historical baseline. Spatial training uses the same trajectory dataset but replaces the single-channel SDF cache with the 2-channel SDF channels cache.

### Exact Training Data Used

The completed ControlNet training run used the generated multi-scale dataset at:

`data_trajectories/EnvConveyor2D-RobotPlanarDisk-multiscale/`

It contains exactly `3000` trajectories total, generated by the **pretrained MMD diffusion model via MPD** (not by classical planners), with `500` trajectories for each of the six discrete scales:

| Scale directory | Trajectories | File |
|-----------------|-------------|------|
| `scale_1.00` | 500 | `scale_1.00/0/trajs-free.pt` |
| `scale_1.10` | 500 | `scale_1.10/0/trajs-free.pt` |
| `scale_1.20` | 500 | `scale_1.20/0/trajs-free.pt` |
| `scale_1.30` | 500 | `scale_1.30/0/trajs-free.pt` |
| `scale_1.40` | 500 | `scale_1.40/0/trajs-free.pt` |
| `scale_1.50` | 500 | `scale_1.50/0/trajs-free.pt` |

Each stored tensor is `float32` with shape `[500, 64, 4]` for that scale. The per-sample representation used in training is:

- normalized trajectory: `[64, 4]`
- hard conditions: start/goal constraints derived from the trajectory endpoints
- legacy `sdf_grid`: `[1, 64, 64]`, loaded from the precomputed single-channel SDF cache for that trajectory's scale
- spatial `sdf_grid`: `[2, 256, 256]`, loaded from the precomputed SDF channels cache for that trajectory's scale
- spatial metadata: `sdf_limits`, `traj_pos_mins`, `traj_pos_maxs`

The SDF cache used by the completed runs was:

`data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache/`

The current spatial cache path is:

`data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256/`

### Train/Validation Split Actually Used

The completed legacy run used `mmd/trainer/train_loaders.py` via `torch.utils.data.random_split` with:

- `val_set_size = 0.05`
- random seed fixed at the start of the experiment (`fix_random_seed(seed)` with `seed=0` for the completed runs)

For the 3000-sample dataset, this produces:

- train subset: `2850` samples
- validation subset: `150` samples

Important details:

- The split is **random over the full combined pool**, not stratified by scale.
- In expectation this gives roughly proportional scale coverage in validation (about `25` samples per scale), but the exact composition is whatever `random_split` produced for that seed.
- The exact split indices are saved to disk for reproducibility.

Current `train_controlnet.py` spatial/default training now builds a scale-aware split directly from `full_dataset.traj_scales`, so each scale contributes validation samples before `Subset` objects are created.

Saved split files for the full run:

- `logs_controlnet_full/0/train_subset_indices.pt`
- `logs_controlnet_full/0/val_subset_indices.pt`

Saved split files for the smoke run:

- `logs_controlnet_smoke/0/train_subset_indices.pt`
- `logs_controlnet_smoke/0/val_subset_indices.pt`

### How Validation Was Computed

Validation is run every `steps_til_summary` steps. For the completed full run this was every `100` training steps.

The validation loop in `train_controlnet.py` stops after `val_step >= 10`, so it can use at most `11` mini-batches. In the completed full run:

- validation set size: `150`
- batch size: `32`
- available validation batches: `ceil(150 / 32) = 5`

Therefore, **all validation samples were used at every validation evaluation** in the completed full run. The reported `val_loss` is the mean across those 5 batches.

### Step Accounting for the Completed Full Run

The requested command used:

- `num_train_steps = 100000`
- `batch_size = 32`

But the code converts this target into epochs using the full dataset length:

```text
epochs = ceil(num_train_steps * batch_size / dataset_len)
       = ceil(100000 * 32 / 3000)
       = 1067
```

Then training runs complete epochs over the actual train split:

- train subset size: `2850`
- train batches per epoch: `ceil(2850 / 32) = 90`
- actual executed steps: `1067 * 90 = 96030`

This is why the completed full run stopped at `96,030` steps rather than exactly `100,000`.

### Where the Training Logs and Artifacts Live

Two ControlNet training runs were kept locally:

#### Smoke run

`logs_controlnet_smoke/0/`

Purpose:

- verify dataset loading
- verify forward/backward pass
- verify checkpoint writing and EMA saving

Key config:

- `batch_size = 16`
- `num_train_steps = 40`
- `steps_til_summary = 10`
- `steps_til_ckpt = 20`
- `wandb_mode = disabled`

#### Full run

`logs_controlnet_full/0/`

This is the main completed training run referenced throughout this document.

Contents:

- `args.yaml`: exact experiment configuration, including git hash
- `train_subset_indices.pt`, `val_subset_indices.pt`: exact random split used
- `checkpoints/controlnet_current_state_dict.pth`: latest non-EMA ControlNet weights
- versioned checkpoints: `checkpoints/controlnet_epoch_XXXX_iter_XXXXXX_state_dict.pth`
- `checkpoints/ema_controlnet_current_state_dict.pth`: latest EMA snapshot during training
- `checkpoints/ema_controlnet_final_state_dict.pth`: final EMA checkpoint
- `checkpoints/train_losses.npy`: logged train losses
- `checkpoints/val_losses.npy`: logged validation losses
- `summaries/`: created by the training scaffold but empty for this run
- `logfile`: created by the launcher scaffold but empty for this run

Checkpoint timestamps show the full run lasted about `69 minutes` (`19:23 -> 20:32`).

### What Is in the Loss Files

For the completed full run:

- `checkpoints/train_losses.npy` contains `901` entries
- `checkpoints/val_losses.npy` contains `901` entries

Each entry stores a pair:

```python
(train_steps_current, {'controlnet_loss': value})
(train_steps_current, {'val_loss': value})
```

Because logging happened every `100` steps, these arrays cover steps `0, 100, 200, ..., 90000`.

Representative values from the full run:

- train loss: `1.049` at step `0` -> about `0.078` at step `90000`
- val loss: `1.002` at step `0` -> about `0.134` at step `90000`
- best observed validation loss: about `0.097` near step `34700`

### WandB Status for the Completed Full Run

Although the script now supports correct wandb logging, the completed full run documented here used:

- `wandb_mode = disabled`

So for that run, the authoritative logs are the local artifacts under `logs_controlnet_full/0/`, not WandB.

## 14. Inference Integration (Implemented)

This section documents the ControlNet inference integration that connects trained adapters to the existing MPD planner while preserving the original MMD inference stack. It supports both `legacy_global` checkpoints and the current `spatial` SDF ControlNet.

### Official ControlNet Reference and Principles

Before implementing inference, the official ControlNet repository (`lllyasviel/ControlNet`) was reviewed to re-check how ControlNet is connected at inference time.

Key principles from the official implementation:

1. The base diffusion model remains the main entry point used by the sampler.
2. The sampler itself is unaware of ControlNet internals.
3. The same noisy latent `x_t` is fed to both the ControlNet branch and the frozen base UNet.
4. ControlNet produces zero-initialized residuals that are injected into the frozen decoder path.
5. Inference supports a control strength multiplier (`control_scales` in the official repo).

Our implementation follows the same principles, adapted to the MMD codebase and to the cross-domain conditioning setting (SDF grid -> trajectory diffusion model).

### Final Inference Design in MMD

Instead of subclassing the full diffusion model like the official Stable Diffusion implementation, MMD uses a lighter integration because the existing code already exposes the needed hook points.

#### Existing MMD Hook Points

- `GaussianDiffusionModel` calls `self.model(x, t, context)` during DDPM and DDIM sampling.
- `TemporalUnet.forward(...)` already accepts:
  - `down_block_additional_residuals`
  - `mid_block_additional_residual`

This means we do **not** need to modify the sampler or the diffusion wrapper.

#### Wrapper Strategy

A new class was added:

- `mmd/models/diffusion_models/controlnet.py::ControlNetInferenceWrapper`

This wrapper replaces `diffusion_model.model` (the base `TemporalUnet`) inside `MPD`.

It presents the same interface as the base UNet:

```python
forward(x, time, context)
```

Internally it performs:

1. load the cached map for the current `env_scale`
2. for `legacy_global`, expand cached `sdf_emb` to batch size
3. for `spatial`, reuse cached SDF pyramid features and sample them at the current noisy waypoint positions
4. run `MMDControlNet(x, time, ...)`
5. scale residuals by `control_scale`
6. call the frozen base UNet with residual injection kwargs

This is functionally analogous to the official ControlNet `apply_model(...)` orchestration step, except that in MMD the orchestration is placed in a wrapper object rather than a subclass of the diffusion model.

### Why the Wrapper Is Faithful to Official ControlNet Principles

Official ControlNet uses:

- a frozen base UNet
- a trainable ControlNet encoder copy
- one orchestration point that combines both during inference

Our wrapper preserves all three:

- frozen base `TemporalUnet`
- trained `MMDControlNet`
- one orchestration point: `ControlNetInferenceWrapper.forward(...)`

So although the implementation pattern differs (composition instead of subclassing), the inference behavior follows the same ControlNet logic.

### SDF/Pyramid Cache at Inference

For one MPD planner instance, the environment scale is fixed, so the SDF grid is fixed as well.

Therefore the wrapper caches the fixed conditioning map once per planner instance.

For `legacy_global`, it:

- stores the resized SDF grid once
- computes `sdf_emb = map_encoder(sdf_grid)` once during initialization
- reuses that embedding for every denoising step

For `spatial`, it:

- stores the 2-channel `[sdf_cur, delta_sdf]` grid once
- stores `sdf_limits`, `traj_pos_mins`, and `traj_pos_maxs`
- computes `p32`, `p16`, and `p8` with `SDFSpatialPyramid2D` once during initialization
- reuses those pyramid feature maps while `WaypointFeatureSampler` re-samples them at each denoising step's current noisy waypoint positions

This avoids recomputing the map CNN/pyramid about 100 times per planning call while preserving the denoising-step dependence of waypoint feature lookup.

This is an intentional optimization specific to our use case. It does **not** change the ControlNet computation graph conceptually; it only removes repeated evaluation of a fixed conditioning input.

### Control Strength at Inference

The official ControlNet implementation exposes per-block `control_scales`.

To preserve the same experimental principle, MMD inference now supports:

- `control_scale: float = 1.0`

This uniformly scales all ControlNet residuals before they are injected into the base UNet.

This gives us a clean ablation knob:

- `control_scale = 0.0` -> effectively base model behavior
- `control_scale = 1.0` -> default ControlNet strength
- `control_scale > 1.0` -> stronger ControlNet influence

For now we use a single scalar rather than one scale per residual because the trajectory UNet has only 3 skip residuals + 1 mid residual, and a uniform scale is the simplest experimental control.

### Files Modified for Inference Integration

#### Core model code

- `mmd/models/diffusion_models/controlnet.py`
  - added `ControlNetInferenceWrapper`

- `mmd/planners/single_agent/mpd.py`
  - added optional args:
    - `controlnet_checkpoint_path`
    - `sdf_cache_dir`
    - `sdf_channels_cache_dir`
    - `control_scale`
    - `controlnet_conditioning_mode`
    - `use_zero_gated_global_context`
  - loads ControlNet checkpoint
  - loads cached SDF or SDF channels for current `env_scale`
  - wraps `diffusion_model.model` before `torch.compile(...)`

#### Experiment/config plumbing

- `mmd/common/experiments/experiments.py`
  - added ControlNet fields to experiment config objects

- `scripts/inference/inference_multi_agent.py`
  - passes ControlNet config from `test_config` into MPD
  - explicitly rejects ControlNet with `MPDEnsemble`

- `scripts/inference/launch_multi_agent_experiment.py`
  - added CLI args for ControlNet inference
  - added `--controlnet_conditioning_mode` and `--no_zero_gated_global_context`
  - added CLI arg for choosing `MPD` vs `MPDEnsemble`

- `scripts/inference/launch_controlnet_evaluation.py`
  - forwards ControlNet mode, cache path, checkpoint path, and zero-gated global-context flag into experiment configs

- `scripts/inference/launch_controlnet_benchmark.py`
  - generates benchmark commands with `--controlnet_conditioning_mode` and optional `--no_zero_gated_global_context`

- `mmd/planners/single_agent/mpd_ensemble.py`
  - added explicit guard so ControlNet args cannot be silently ignored

### Why `MPDEnsemble` Is Not Enabled Yet

ControlNet training was performed specifically for the `EnvConveyor2D` base model configuration (`dim_mults=(1,2,4,8)`).

`MPDEnsemble` may mix different model IDs and different UNet architectures. Silently applying one ControlNet checkpoint there would be unsafe and could fail by architecture mismatch or produce misleading results.

So for now:

- ControlNet inference is supported only for `MPD`
- `MPDEnsemble` raises a clear error if ControlNet args are supplied

This is safer and gives a cleaner experimental signal.

### Status After Integration

Implemented:

- wrapper-based ControlNet inference connection
- ControlNet checkpoint loading inside `MPD`
- SDF cache and SDF channels cache loading by `env_scale`
- uniform control strength scaling
- `legacy_global` and `spatial` ControlNet mode selection
- optional zero-gated global context in spatial mode
- experiment/CLI plumbing
- explicit unsupported-path guard for `MPDEnsemble`

Still required after this code integration:

1. train and benchmark spatial checkpoints against the completed legacy-global baseline

### Validation Executed

The validation commands below are the historical checks from the legacy-global integration. For the spatial implementation, the minimum verification target is:

```bash
python3 -m py_compile \
    mmd/models/helpers/map_encoder.py \
    mmd/models/helpers/sdf_channels.py \
    mmd/models/diffusion_models/controlnet.py \
    mmd/datasets/controlnet_dataset.py \
    mmd/planners/single_agent/mpd.py \
    scripts/generate_data/precompute_sdf_channels.py \
    scripts/train_diffusion/train_controlnet.py \
    scripts/inference/launch_controlnet_evaluation.py \
    scripts/inference/launch_controlnet_benchmark.py
```

Spatial smoke checks should verify:

- `precompute_sdf_channels.py` writes `sdf_channels_scale_*.pt` files with `channel_order == ['sdf_cur', 'delta_sdf']` and `version == 1`
- reference scale `1.00` has near-zero delta channel
- `ControlNetTrajectoryDataset` returns `sdf_grid [2,256,256]`, `sdf_limits`, `traj_pos_mins`, and `traj_pos_maxs`
- `MMDControlNet(conditioning_mode='spatial', sdf_in_channels=2)` produces 3 down residuals plus 1 mid residual with the same shapes as the frozen decoder injection points
- `MPD` accepts `sdf_channels_cache_dir`/2-channel cache and runs `ControlNetInferenceWrapper` with `controlnet_conditioning_mode='spatial'`

Historical legacy-global checks:

The modified files were syntax-checked with:

```bash
python3 -m py_compile mmd/models/diffusion_models/controlnet.py mmd/planners/single_agent/mpd.py mmd/planners/single_agent/mpd_ensemble.py mmd/common/experiments/experiments.py scripts/inference/inference_multi_agent.py scripts/inference/launch_multi_agent_experiment.py
```

#### Smoke Test 1: Base MPD Path (No ControlNet)

Exact command used:

```bash
python3 - <<'PY'
from pathlib import Path
from einops._torch_specific import allow_ops_in_compiled_graph
from mmd.planners.single_agent.mpd import MPD

allow_ops_in_compiled_graph()
base = Path('/home/dsi/yefimnu/diffusion_projects/mmd')
planner = MPD(
    model_id='EnvConveyor2D-RobotPlanarDisk',
    planner_alg='mmd',
    start_state_pos=None,
    goal_state_pos=None,
    use_guide_on_extra_objects_only=False,
    start_guide_steps_fraction=0.5,
    n_guide_steps=1,
    n_diffusion_steps_without_noise=1,
    weight_grad_cost_collision=2e-2,
    weight_grad_cost_smoothness=8e-2,
    weight_grad_cost_constraints=2e-1,
    weight_grad_cost_soft_constraints=2e-2,
    factor_num_interpolated_points_for_collision=1.5,
    trajectory_duration=5.0,
    device='cuda',
    debug=False,
    seed=18,
    results_dir='logs',
    trained_models_dir=str(base / 'data_trained_models'),
    n_samples=4,
    n_local_inference_noising_steps=3,
    n_local_inference_denoising_steps=3,
    env_scale=1.0,
    horizon=64,
)
result = planner(planner.start_state_pos, planner.goal_state_pos)
print('BASE_SMOKE_OK', result.trajs_final.shape, None if result.trajs_final_free is None else result.trajs_final_free.shape)
PY
```

Observed result:

- planner initialized successfully
- diffusion sampling completed successfully
- output shape: `trajs_final = [4, 64, 4]`
- collision-free trajectories found: `trajs_final_free = [1, 64, 4]`

#### Smoke Test 2: Legacy ControlNet MPD Path

Exact command used:

```bash
python3 - <<'PY'
from pathlib import Path
from einops._torch_specific import allow_ops_in_compiled_graph
from mmd.planners.single_agent.mpd import MPD

allow_ops_in_compiled_graph()
base = Path('/home/dsi/yefimnu/diffusion_projects/mmd')
planner = MPD(
    model_id='EnvConveyor2D-RobotPlanarDisk',
    planner_alg='mmd',
    start_state_pos=None,
    goal_state_pos=None,
    use_guide_on_extra_objects_only=False,
    start_guide_steps_fraction=0.5,
    n_guide_steps=1,
    n_diffusion_steps_without_noise=1,
    weight_grad_cost_collision=2e-2,
    weight_grad_cost_smoothness=8e-2,
    weight_grad_cost_constraints=2e-1,
    weight_grad_cost_soft_constraints=2e-2,
    factor_num_interpolated_points_for_collision=1.5,
    trajectory_duration=5.0,
    device='cuda',
    debug=False,
    seed=18,
    results_dir='logs',
    trained_models_dir=str(base / 'data_trained_models'),
    n_samples=4,
    n_local_inference_noising_steps=3,
    n_local_inference_denoising_steps=3,
    env_scale=1.2,
    horizon=64,
    controlnet_checkpoint_path=str(base / 'logs_controlnet_full/0/checkpoints/ema_controlnet_final_state_dict.pth'),
    sdf_cache_dir=str(base / 'data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_cache'),
    control_scale=1.0,
)
result = planner(planner.start_state_pos, planner.goal_state_pos)
print('CONTROLNET_SMOKE_OK', result.trajs_final.shape, None if result.trajs_final_free is None else result.trajs_final_free.shape)
PY
```

Observed result:

- ControlNet checkpoint loaded successfully
- cached SDF for `env_scale=1.20` loaded successfully
- wrapper path executed inside MPD inference successfully
- output shape: `trajs_final = [4, 64, 4]`
- collision-free trajectories found: `trajs_final_free = [4, 64, 4]`

These smoke tests validated that the legacy-global integration is operational. Full legacy benchmark evaluations are completed and documented in `CONTROLNET_EXPERIMENTS.md` Sections 2 to 5. Spatial inference uses the same wrapper and MPD plumbing with `--controlnet_conditioning_mode spatial` plus a 2-channel SDF cache.

Example spatial MPD launcher arguments:

```bash
python scripts/inference/launch_controlnet_evaluation.py \
    --modes controlnet \
    --controlnet_checkpoint_path <spatial_controlnet_checkpoint.pth> \
    --sdf_cache_dir data_trained_models/EnvConveyor2D-RobotPlanarDisk/controlnet/sdf_channels_cache_2ch_256 \
    --controlnet_conditioning_mode spatial \
    --control_scale 1.0
```

---

## Experiment Results and Follow-Up

`CONTROLNET_EXPERIMENTS.md` contains the benchmark plan, executed legacy-global runs, aggregate analysis, and the `env_scale=1.5` stress-test.

- Completed legacy experiments show that the simpler global FiLM-style adapter improved planning most as scale and agent count increased.
- The current implementation replaces that path for new training with spatial SDF feature lookup while keeping `legacy_global` for checkpoint compatibility.
- Cross-attention from Section 7 remains a possible follow-up after spatial adapter benchmarks.
- See `CONTROLNET_EXPERIMENTS.md` Sections 4 to 7 for the historical legacy-global evidence and recommendations.
