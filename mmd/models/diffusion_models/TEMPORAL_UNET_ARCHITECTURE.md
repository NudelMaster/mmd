# ControlNet Implementation Plan for MMD TemporalUnet

This document provides the implementation plan for adding ControlNet-style conditioning to the MMD trajectory diffusion model, enabling generalization to scaled environments.

---

## Problem Statement

The original MMD model performs poorly on environments scaled beyond 1.0x (tested up to 1.5x on the Conveyor map). We need the model to generalize across different environment scales.

## Solution: ControlNet Adapter

Add a ControlNet-style adapter that conditions the trajectory diffusion model on:
1. **Local SDF (Signed Distance Function) grids** - 2D spatial map of obstacles (64x64, 1 channel)
2. **Scale parameter** - scalar indicating environment scale factor

---

## Overview

| Component | Description |
|-----------|-------------|
| **Goal** | Enable trajectory diffusion model to condition on local SDF maps + scale parameter |
| **Architecture** | ControlNet-style: Locked copy (frozen) + Trainable copy (encoder + mid) + Zero convolutions |
| **Input Conditioning** | SDF grid 64x64 (1 channel) + Scale scalar (sinusoidal embedding) |
| **Compatibility** | Works with both `MPD` (single-agent) and `MPDEnsemble` (multi-agent) |

---

## TemporalUnet Architecture Summary

### Tensor Shapes

| Location | Shape | Notes |
|----------|-------|-------|
| Input trajectory | `[B, 64, 4]` | (batch, horizon, state_dim) |
| Encoder level 0 | `[B, 32, 64]` | Skip connection h[0] |
| Encoder level 1 | `[B, 64, 32]` | Skip connection h[1] |
| Encoder level 2 | `[B, 128, 16]` | Skip connection h[2] |
| Encoder level 3 | `[B, 256, 8]` | Skip connection h[3] |
| Mid block output | `[B, 256, 8]` | After mid_block2 |
| Output | `[B, 64, 4]` | Predicted noise |

### Architecture Flow

```
Input [B, 64, 4] → Rearrange → [B, 4, 64]
    ↓
ENCODER (4 levels with skip connections h[0..3])
    ↓
MID BLOCK
    ↓
DECODER (3 levels, concatenates with skip connections)
    ↓
Output [B, 64, 4]
```

---

## Detailed Forward Flow

This section traces through the `forward` method step-by-step with concrete shapes.

**Assumptions**: `state_dim=4`, `unet_input_dim=32`, `dim_mults=(1,2,4,8)`, `H=64`, `B=8`, `conditioning_type=None`

### Step 1: Input & Shape Extraction

```python
def forward(self, x, time, context, ...):
    b, h, d = x.shape
```

| Variable | Shape | Example |
|----------|-------|---------|
| `x` | `[B, H, state_dim]` | `[8, 64, 4]` |
| `time` | `[B]` | `[8]` |
| `context` | `[B, context_dim]` or `None` | `None` |

### Step 2: Time Embedding

```python
t_emb = self.time_mlp(time)  # TimeEncoder: sinusoidal + MLP
c_emb = t_emb
```

| Variable | Shape | Description |
|----------|-------|-------------|
| `time` | `[8]` | Diffusion timestep indices |
| `t_emb` | `[8, 32]` | Time embedding (`time_emb_dim=32`) |
| `c_emb` | `[8, 32]` | Conditioning embedding (just time for now) |

### Step 3: Transpose for Conv1d

```python
x = einops.rearrange(x, 'b h c -> b c h')
```

| Before | After |
|--------|-------|
| `[8, 64, 4]` | `[8, 4, 64]` |
| `[B, H, state_dim]` | `[B, state_dim, H]` |

### Step 4: Encoder (Downsampling Path)

```python
h = []
for resnet, resnet2, attn_self, attn_conditioning, downsample in self.downs:
    x = resnet(x, c_emb)
    x = resnet2(x, c_emb)
    x = attn_self(x)
    h.append(x)
    x = downsample(x)
```

| Level | Input Shape | After resnet | After resnet2 | After downsample | Skip Saved |
|-------|-------------|--------------|---------------|------------------|------------|
| 0 | `[8, 4, 64]` | `[8, 32, 64]` | `[8, 32, 64]` | `[8, 32, 32]` | `[8, 32, 64]` |
| 1 | `[8, 32, 32]` | `[8, 64, 32]` | `[8, 64, 32]` | `[8, 64, 16]` | `[8, 64, 32]` |
| 2 | `[8, 64, 16]` | `[8, 128, 16]` | `[8, 128, 16]` | `[8, 128, 8]` | `[8, 128, 16]` |
| 3 | `[8, 128, 8]` | `[8, 256, 8]` | `[8, 256, 8]` | `[8, 256, 8]`* | `[8, 256, 8]` |

*Level 3 has `Identity` instead of `Downsample1d` (is_last=True)

### Step 5: Middle Block

```python
x = self.mid_block1(x, c_emb)
x = self.mid_attn(x)
x = self.mid_block2(x, c_emb)
if mid_block_additional_residual is not None:
    x = x + mid_block_additional_residual  # ControlNet injection
```

| Step | Shape | Notes |
|------|-------|-------|
| Input | `[8, 256, 8]` | From encoder |
| After mid_block1 | `[8, 256, 8]` | ResBlock (256→256) |
| After mid_block2 | `[8, 256, 8]` | ResBlock (256→256) |

### Step 6: Decoder (Upsampling Path)

```python
for resnet, resnet2, attn_self, attn_conditioning, upsample in self.ups:
    skip = h.pop()
    if down_block_additional_residuals is not None:
        skip = skip + down_block_additional_residuals.pop()  # ControlNet
    x = torch.cat((x, skip), dim=1)
    x = resnet(x, c_emb)
    x = resnet2(x, c_emb)
    x = upsample(x)
```

| Level | x Shape | Skip (popped) | After concat | After resnet | After resnet2 | After upsample |
|-------|---------|---------------|--------------|--------------|---------------|----------------|
| 0 | `[8,256,8]` | `[8,256,8]` | `[8,512,8]` | `[8,128,8]` | `[8,128,8]` | `[8,128,16]` |
| 1 | `[8,128,16]` | `[8,128,16]` | `[8,256,16]` | `[8,64,16]` | `[8,64,16]` | `[8,64,32]` |
| 2 | `[8,64,32]` | `[8,64,32]` | `[8,128,32]` | `[8,32,32]` | `[8,32,32]` | `[8,32,64]` |

**Note:** h[0]=[8,32,64] is never used (decoder has only 3 levels)

### Step 7: Final Convolution

```python
x = self.final_conv(x)
```

| Layer | Input | Output |
|-------|-------|--------|
| `Conv1dBlock(32, 32, k=5)` | `[8, 32, 64]` | `[8, 32, 64]` |
| `Conv1d(32, 4, k=1)` | `[8, 32, 64]` | `[8, 4, 64]` |

### Step 8: Transpose Back

```python
x = einops.rearrange(x, 'b c h -> b h c')
return x  # [8, 64, 4] = [B, H, state_dim]
```

---

## Complete Flow Diagram

```
INPUT: x [8, 64, 4], time [8], context None
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  TIME EMBEDDING                                               │
│  time [8] ──► TimeEncoder ──► t_emb [8, 32]                  │
│  c_emb = t_emb [8, 32]                                        │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  TRANSPOSE FOR CONV1D                                         │
│  x: [8, 64, 4] ──► rearrange ──► [8, 4, 64]                  │
│     [B, H, C]                     [B, C, H]                   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  ENCODER                                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 0: [8,4,64]──►Res──►[8,32,64]──►Down──►[8,32,32]  │──► h[0]=[8,32,64]
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 1: [8,32,32]──►Res──►[8,64,32]──►Down──►[8,64,16] │──► h[1]=[8,64,32]
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 2: [8,64,16]──►Res──►[8,128,16]──►Down──►[8,128,8]│──► h[2]=[8,128,16]
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 3: [8,128,8]──►Res──►[8,256,8]──►Identity         │──► h[3]=[8,256,8]
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ [8, 256, 8]
┌───────────────────────────────────────────────────────────────┐
│  MIDDLE BLOCK                                                 │
│  [8,256,8]──►ResBlock──►[8,256,8]──►ResBlock──►[8,256,8]     │
│  (+ mid_block_additional_residual if ControlNet)              │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ [8, 256, 8]
┌───────────────────────────────────────────────────────────────┐
│  DECODER                                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 0: cat([8,256,8], h.pop()=[8,256,8])              │ │
│  │          [8,512,8]──►Res──►[8,128,8]──►Up──►[8,128,16]  │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 1: cat([8,128,16], h.pop()=[8,128,16])            │ │
│  │          [8,256,16]──►Res──►[8,64,16]──►Up──►[8,64,32]  │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Level 2: cat([8,64,32], h.pop()=[8,64,32])              │ │
│  │          [8,128,32]──►Res──►[8,32,32]──►Up──►[8,32,64]  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Note: h[0]=[8,32,64] is NEVER USED (asymmetric design)      │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ [8, 32, 64]
┌───────────────────────────────────────────────────────────────┐
│  FINAL CONV                                                   │
│  [8,32,64]──►Conv1dBlock──►[8,32,64]──►Conv1d──►[8,4,64]     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  TRANSPOSE BACK                                               │
│  x: [8, 4, 64] ──► rearrange ──► [8, 64, 4]                  │
│     [B, C, H]                     [B, H, C]                   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: x [8, 64, 4] = [B, H, state_dim]
        (Predicted noise or denoised trajectory)
```

---

## Key Observations

1. **Asymmetric Skip Usage**: The first encoder skip (`h[0]`) is never used because decoder has only 3 levels (`reversed(in_out[1:])`)

2. **Channel Doubling at Concat**: Decoder input channels double due to skip concatenation (e.g., 256+256=512)

3. **Time Conditioning**: `c_emb` is passed to every `ResidualTemporalBlock` for FiLM-style modulation

4. **Shape Preservation**: Input and output have identical shapes `[B, H, state_dim]`

5. **Temporal Resolution**: Reduced by 8x at bottleneck (64→8), fully restored by decoder

---

## Skip Connections Purpose

| Purpose | Explanation |
|---------|-------------|
| **Preserve Fine-Grained Details** | Downsampling loses temporal resolution. Skip connections carry high-resolution features directly to the decoder. |
| **Multi-Scale Feature Fusion** | Combines low-level details (encoder) with high-level semantics (bottleneck). |
| **Gradient Flow** | Provides shorter paths for gradients during backpropagation. |

```
WITHOUT Skip Connections:
 Input ──► Enc ──► Mid ──► Dec ──► Output
   │                                  │
   └──── Gradient must flow through ALL layers ────┘

WITH Skip Connections:
 Input ──► Enc1 ──► Enc2 ──► Mid ──► Dec2 ──► Dec1 ──► Output
             │        │               ▲        ▲
             │        └───────────────┘        │
             └─────────────────────────────────┘
             (gradients have SHORTER paths too)
```

---

## Implementation Phases

### Phase 1: Modify Base Architecture [COMPLETED]

**File**: `mmd/models/diffusion_models/temporal_unet.py`

**Changes made**:
1. Updated `forward()` signature to accept `down_block_additional_residuals` and `mid_block_additional_residual` parameters
2. Added mid-block residual injection after `mid_block2`
3. Added down-block residuals injection into skip connections in decoder loop

All changes are backward-compatible (default values are `None`).

### Phase 2: Create ControlNet Components

**New file**: `mmd/models/diffusion_models/controlnet.py`

Components to implement:
- `ZeroConv1d` - Zero-initialized 1D convolution for stable training start
- `SDFHintEncoder` - CNN to encode SDF grid [B, 1, 64, 64] → [B, 4, 64]
- `ScaleEncoder` - Sinusoidal embedding for scale scalar
- `MMDControlNet` - Main wrapper class with locked/trainable copies

### Phase 3: Data Pipeline Updates

**Files to modify**:
- `scripts/generate_data/generate_trajectories.py` - Save SDF grids and scale parameter
- `mmd/datasets/trajectories.py` - Load SDF grid and scale in `__getitem__`

**New file**:
- `scripts/generate_data/launch_generate_trajectories_multiscale.py` - Generate data at scales [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

### Phase 4: Training Setup

**New file**: `scripts/train_diffusion/train_controlnet.py`

Training script that:
- Loads pretrained TemporalUnet (frozen)
- Creates ControlNet adapter
- Trains only ControlNet parameters on multi-scale data

---

## ControlNet Architecture

```
CONDITION INPUTS
├── SDF Grid [B, 1, 64, 64] → SDFHintEncoder → [B, 4, 64]
└── Scale [B] → ScaleEncoder → [B, 32] → scale_proj → [B, time_emb_dim]

TRAINABLE CONTROL COPY
├── Input: x + sdf_hint
├── Time: t_emb + scale_emb
├── control_downs[0..3] → control_residuals[0..3]
└── control_mid → mid_residual

ZERO CONVOLUTIONS (5 total)
├── 4x ZeroConv1d for down_residuals
└── 1x ZeroConv1d for mid_residual

FROZEN LOCKED UNET
├── Encoder runs normally
├── Mid block + mid_residual injection
└── Decoder with skip + down_residuals injection
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `temporal_unet.py` | MODIFIED | Added injection ports for ControlNet residuals |
| `controlnet.py` | TO CREATE | ZeroConv1d, SDFHintEncoder, ScaleEncoder, MMDControlNet |
| `trajectories.py` | TO MODIFY | Load SDF grid and scale from data |
| `generate_trajectories.py` | TO MODIFY | Save SDF grid and scale parameter |
| `launch_generate_trajectories_multiscale.py` | TO CREATE | Multi-scale data generation |
| `train_controlnet.py` | TO CREATE | ControlNet training script |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SDF grid only (no gradients) | CNN encoders naturally learn edge detection in first layers |
| Sinusoidal scale embedding | Same approach as diffusion timesteps for smooth interpolation |
| Zero convolutions | Ensures ControlNet has no effect at training start (stable) |
| Residuals added to skip connections | Preserves locked model's encoder behavior |

---

## Training Strategy

1. **Freeze base model**: All TemporalUnet parameters frozen
2. **Train ControlNet only**: control_time_mlp, control_downs, control_mid_*, encoders, zero_convs
3. **Data**: Multi-scale trajectories (1.0x to 1.5x)
4. **Loss**: L2 between predicted and actual noise
5. **Initialization**: Zero convolutions ensure training starts from base model behavior

---

## Next Steps

1. Implement Phase 2: Create `controlnet.py`
2. Generate multi-scale training data (Phase 3)
3. Create and run training script (Phase 4)
4. Evaluate on scaled environments

---

# ControlNet SDF Conditioning: Detailed Analysis

This section provides a comprehensive analysis of two architectural approaches for conditioning the ControlNet on SDF (Signed Distance Field) grids and environment scale.

---

## Background: SDF Grid Dimensions

Based on validation of the actual codebase:

| Parameter | Value | Source |
|-----------|-------|--------|
| Environment limits | `[[-1, -1], [1, 1]]` | `env_highways_2d.py` |
| Physical map size | 2×2 units | `abs(1 - (-1))` |
| Default `sdf_cell_size` | 0.005 | `EnvBase.__init__` |
| **Native SDF grid** | **400×400** | `map_dim / cell_size` |
| Robot radius | 0.05 units | `mmd_params.py` |
| Robot diameter | 0.1 units | 5% of environment width |

### SDF Resolution vs Robot Size

| `sdf_cell_size` | Grid Size | Robot diameter in cells | Quality |
|-----------------|-----------|-------------------------|---------|
| 0.005 (default) | 400×400 | 20 cells | High precision |
| 0.01 | 200×200 | 10 cells | Good precision |
| 0.05 | 40×40 | 2 cells | Poor |

### Target Resolution for Neural Network

For ControlNet conditioning, we resize 400×400 → 64×64:
- **Memory efficient**: 640KB → 16KB per sample (40× smaller)
- **CNN-friendly**: Power of 2, standard image size
- **Sufficient detail**: Robot ≈ 3.2 pixels at 64×64 (captures geometry)
- **Manageable attention**: 64 tokens for cross-attention

---

## Domain Mismatch Consideration

The original ControlNet (for Stable Diffusion) operates in the **same domain**:
- Input: 2D image
- Condition: 2D image (edges, depth, pose)
- Output: 2D image

For trajectory diffusion, we have a **cross-domain** problem:
- Input/Output: 1D trajectory `[B, H, state_dim]` (sequence of waypoints)
- Condition: 2D SDF grid `[400, 400]`

This domain mismatch informs the choice of conditioning architecture.

---

## Existing Infrastructure in TemporalUnet

The base `TemporalUnet` already supports cross-attention when `conditioning_type == 'attention'`:

```python
# In encoder (downs) - line 86-87
SpatialTransformer(dim_out, attention_num_heads, attention_dim_head, depth=1,
                   context_dim=conditioning_embed_dim) if conditioning_type == 'attention' else None

# In mid block - line 97-98
self.mid_attention = SpatialTransformer(...) if conditioning_type == 'attention' else nn.Identity()

# In decoder (ups) - line 108-109
SpatialTransformer(dim_in, ..., context_dim=conditioning_embed_dim) 
    if conditioning_type == 'attention' else None
```

This means **cross-attention infrastructure already exists** and can be leveraged.

---

## Existing MapSDFEncoder

Located at `mmd/models/helpers/map_encoder.py`:

```python
class MapSDFEncoder(nn.Module):
    """
    Encodes SDF grid + scale into context tokens for cross-attention.
    
    Input:
        sdf_grid: [B, 1, 64, 64] - Resized SDF grid
        scale_scalar: [B] - Environment scale factor
    
    Output:
        context: [B, 64, 256] - 64 spatial tokens, 256-dim each
    """
    def __init__(self, hidden_dim=256, output_dim=256):
        # CNN: 64×64 → 32×32 → 16×16 → 8×8
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 32×32
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16×16
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),  # 8×8
            nn.SiLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )
        
        # Scale embedding
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, sdf_grid, scale_scalar):
        img_features = self.cnn(sdf_grid)  # [B, 256, 8, 8]
        flat_features = img_features.view(B, C, -1).permute(0, 2, 1)  # [B, 64, 256]
        scale_emb = self.scale_mlp(scale_scalar.unsqueeze(-1))  # [B, 256]
        context = flat_features + scale_emb.unsqueeze(1)  # Broadcast add
        return context  # [B, 64, 256]
```

---

# Approach A: Cross-Attention Conditioning

## Overview

This approach uses the `MapSDFEncoder` output as **context tokens** for cross-attention in ControlNet encoder blocks. Each trajectory position can attend to relevant SDF spatial locations.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SDF PREPROCESSING                                   │
│                                                                             │
│  SDF Grid [400×400]                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │   resize_sdf()  │  F.interpolate(bilinear)                               │
│  │   400→64        │                                                        │
│  └─────────────────┘                                                        │
│         │                                                                   │
│         ▼                                                                   │
│  SDF Grid [B, 1, 64, 64]     Scale [B]                                      │
│         │                       │                                           │
│         └───────────┬───────────┘                                           │
│                     ▼                                                       │
│  ┌──────────────────────────────────────┐                                   │
│  │          MapSDFEncoder               │                                   │
│  │  ┌────────────────────────────────┐  │                                   │
│  │  │ CNN: [B,1,64,64]               │  │                                   │
│  │  │      → [B,32,32,32]            │  │                                   │
│  │  │      → [B,64,16,16]            │  │                                   │
│  │  │      → [B,256,8,8]             │  │                                   │
│  │  │      → flatten → [B,64,256]    │  │                                   │
│  │  └────────────────────────────────┘  │                                   │
│  │  ┌────────────────────────────────┐  │                                   │
│  │  │ Scale MLP: [B] → [B,256]       │  │                                   │
│  │  └────────────────────────────────┘  │                                   │
│  │  context = flat_features + scale_emb │                                   │
│  │  Output: [B, 64, 256]                │                                   │
│  └──────────────────────────────────────┘                                   │
│                     │                                                       │
│                     ▼                                                       │
│              context tokens                                                 │
│               [B, 64, 256]                                                  │
│                                                                             │
└─────────────────────┼───────────────────────────────────────────────────────┘
                      │
                      │ (passed to all cross-attention blocks)
                      │
    ┌─────────────────┴─────────────────────────────────────────┐
    │                                                           │
    ▼                                                           ▼
┌───────────────────────────────────┐     ┌───────────────────────────────────┐
│       ControlNet Encoder          │     │        Frozen Base Model          │
│         (TRAINABLE)               │     │           (LOCKED)                │
│                                   │     │                                   │
│  Input: x [B, 64, 4]              │     │                                   │
│         │                         │     │                                   │
│         ▼ rearrange               │     │                                   │
│  x_ctrl [B, 4, 64]                │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 0:                    │  │     │                                   │
│  │   ResNet×2 (t_emb)          │  │     │                                   │
│  │   Self-Attention            │  │     │                                   │
│  │   Cross-Attention ◄─────────┼──┼─────┼── context [B,64,256]              │
│  │   ZeroConv → residual[0]    │──┼─────┼──► added to skip[3]               │
│  │   Downsample                │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 1:                    │  │     │                                   │
│  │   ResNet×2 (t_emb)          │  │     │                                   │
│  │   Self-Attention            │  │     │                                   │
│  │   Cross-Attention ◄─────────┼──┼─────┼── context [B,64,256]              │
│  │   ZeroConv → residual[1]    │──┼─────┼──► added to skip[2]               │
│  │   Downsample                │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 2:                    │  │     │                                   │
│  │   ResNet×2 (t_emb)          │  │     │                                   │
│  │   Self-Attention            │  │     │                                   │
│  │   Cross-Attention ◄─────────┼──┼─────┼── context [B,64,256]              │
│  │   ZeroConv → residual[2]    │──┼─────┼──► added to skip[1]               │
│  │   Downsample                │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 3:                    │  │     │                                   │
│  │   ResNet×2 (t_emb)          │  │     │                                   │
│  │   Self-Attention            │  │     │                                   │
│  │   Cross-Attention ◄─────────┼──┼─────┼── context [B,64,256]              │
│  │   ZeroConv → residual[3]    │──┼─────┼──► (unused, h[0] pattern)         │
│  │   Identity (is_last)        │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│         ▼                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Mid Blocks:                 │  │     │                                   │
│  │   mid_block1 (t_emb)        │  │     │                                   │
│  │   mid_self_attn             │  │     │                                   │
│  │   mid_cross_attn ◄──────────┼──┼─────┼── context [B,64,256]              │
│  │   mid_block2 (t_emb)        │  │     │                                   │
│  │   ZeroConv → mid_residual   │──┼─────┼──► added to mid output            │
│  └─────────────────────────────┘  │     │                                   │
│                                   │     │                                   │
└───────────────────────────────────┘     └───────────────────────────────────┘
```

## Forward Pass Pseudocode

```python
def forward(self, x, time, sdf_grid, scale):
    """
    Args:
        x: Noisy trajectory [B, H, state_dim] = [B, 64, 4]
        time: Diffusion timestep [B]
        sdf_grid: SDF tensor [B, 400, 400] or [B, 1, 400, 400]
        scale: Environment scale factor [B]
    
    Returns:
        Denoised trajectory [B, 64, 4]
    """
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: SDF Preprocessing
    # ═══════════════════════════════════════════════════════════════
    
    # Resize SDF: 400×400 → 64×64
    sdf_resized = resize_sdf(sdf_grid, target_size=64)  # [B, 1, 64, 64]
    
    # Encode SDF + Scale → Context tokens
    context = self.map_encoder(sdf_resized, scale)  # [B, 64, 256]
    #   └── 64 spatial tokens (from 8×8 feature map)
    #   └── 256-dim features per token
    #   └── Scale information added to all tokens
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Time Embedding (ControlNet has its own)
    # ═══════════════════════════════════════════════════════════════
    
    t_emb = self.control_time_mlp(time)  # [B, 32]
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Prepare Trajectory Input
    # ═══════════════════════════════════════════════════════════════
    
    x_ctrl = einops.rearrange(x, 'b h c -> b c h')  # [B, 4, 64]
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: ControlNet Encoder with Cross-Attention
    # ═══════════════════════════════════════════════════════════════
    
    down_residuals = []
    
    for level, (resnet1, resnet2, self_attn, cross_attn, downsample) in enumerate(self.control_downs):
        # ResNet blocks with time conditioning
        x_ctrl = resnet1(x_ctrl, t_emb)   # [B, dim_out, H']
        x_ctrl = resnet2(x_ctrl, t_emb)   # [B, dim_out, H']
        
        # Self-attention (trajectory attends to itself)
        x_ctrl = self_attn(x_ctrl)        # [B, dim_out, H']
        
        # Cross-attention (trajectory attends to SDF spatial tokens)
        # This is the KEY difference from Approach B
        x_ctrl = cross_attn(x_ctrl, context=context)  # [B, dim_out, H']
        #   └── Query: trajectory features [B, H', dim_out]
        #   └── Key/Value: SDF context [B, 64, 256]
        #   └── Each trajectory position can "look at" relevant SDF regions
        
        # Zero-conv residual (starts at zero for stable training)
        residual = self.zero_convs[level](x_ctrl)
        down_residuals.append(residual)
        
        # Downsample (except last level)
        x_ctrl = downsample(x_ctrl)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Mid Blocks with Cross-Attention
    # ═══════════════════════════════════════════════════════════════
    
    x_ctrl = self.mid_block1(x_ctrl, t_emb)      # [B, 256, 8]
    x_ctrl = self.mid_self_attn(x_ctrl)          # [B, 256, 8]
    x_ctrl = self.mid_cross_attn(x_ctrl, context=context)  # Cross-attend to SDF
    x_ctrl = self.mid_block2(x_ctrl, t_emb)      # [B, 256, 8]
    
    mid_residual = self.zero_conv_mid(x_ctrl)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Inject Residuals into Frozen Base Model
    # ═══════════════════════════════════════════════════════════════
    
    output = self.base_model(
        x, time, context=None,  # Base model runs without context
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual
    )
    
    return output  # [B, 64, 4]
```

## Tensor Shape Flow (Cross-Attention)

### SDF Preprocessing

| Step | Operation | Shape |
|------|-----------|-------|
| Input SDF | From environment | `[B, 400, 400]` |
| Reshape | Add channel dim | `[B, 1, 400, 400]` |
| Resize | Bilinear interpolate | `[B, 1, 64, 64]` |
| CNN Layer 1 | Conv2d stride=2 | `[B, 32, 32, 32]` |
| CNN Layer 2 | Conv2d stride=2 | `[B, 64, 16, 16]` |
| CNN Layer 3 | Conv2d stride=2 | `[B, 256, 8, 8]` |
| Flatten | Reshape + permute | `[B, 64, 256]` |
| Add Scale | + scale_emb | `[B, 64, 256]` |
| **Context Output** | | **`[B, 64, 256]`** |

### ControlNet Encoder

| Level | Input | After ResNets | After Cross-Attn | After Downsample | Residual |
|-------|-------|---------------|------------------|------------------|----------|
| 0 | `[B,4,64]` | `[B,32,64]` | `[B,32,64]` | `[B,32,32]` | `[B,32,64]` |
| 1 | `[B,32,32]` | `[B,64,32]` | `[B,64,32]` | `[B,64,16]` | `[B,64,32]` |
| 2 | `[B,64,16]` | `[B,128,16]` | `[B,128,16]` | `[B,128,8]` | `[B,128,16]` |
| 3 | `[B,128,8]` | `[B,256,8]` | `[B,256,8]` | `[B,256,8]` | `[B,256,8]` |

### Cross-Attention Details

At each level, the `SpatialTransformer` performs:

```
Query:  trajectory features  [B, H', dim]  → project → [B, H', d_head × n_heads]
Key:    context (SDF)        [B, 64, 256]  → project → [B, 64, d_head × n_heads]
Value:  context (SDF)        [B, 64, 256]  → project → [B, 64, d_head × n_heads]

Attention: softmax(Q @ K^T / sqrt(d)) @ V
Output: [B, H', dim]
```

**Attention matrix size**: `H' × 64` where H' = {64, 32, 16, 8} at each level

## Components to Implement/Modify

| File | Change | Description |
|------|--------|-------------|
| `map_encoder.py` | **ADD** | `resize_sdf(sdf_tensor, target_size=64)` function |
| `controlnet.py` | **MODIFY** | Add `MapSDFEncoder` as `self.map_encoder` |
| `controlnet.py` | **MODIFY** | `_create_control_encoder()` to include `SpatialTransformer` blocks |
| `controlnet.py` | **MODIFY** | `_create_control_mid()` to include cross-attention |
| `controlnet.py` | **MODIFY** | `forward()` to process SDF and pass context |

## Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Spatial Specificity** | Each trajectory waypoint can attend to specific SDF regions (e.g., "obstacle on the left") |
| **Multi-scale Attention** | Different encoder levels attend at different resolutions |
| **Proven Pattern** | Matches Stable Diffusion ControlNet architecture |
| **Natural Reasoning** | Enables "look here to avoid this" type of learning |
| **Existing Infrastructure** | `SpatialTransformer` already exists in codebase |

## Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **More Parameters** | ~500K additional params (4 SpatialTransformers + mid) |
| **Slower Training** | Attention has O(H' × 64) complexity per level |
| **More Complex** | More components to debug and tune |
| **Context Dim Matching** | Must match `context_dim=256` everywhere |

---

# Approach B: Global Conditioning

## Overview

This approach pools the `MapSDFEncoder` output into a **single global vector** and adds it to the time embedding. All trajectory positions receive the same SDF information.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SDF PREPROCESSING                                   │
│                                                                             │
│  SDF Grid [400×400]                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │   resize_sdf()  │  F.interpolate(bilinear)                               │
│  │   400→64        │                                                        │
│  └─────────────────┘                                                        │
│         │                                                                   │
│         ▼                                                                   │
│  SDF Grid [B, 1, 64, 64]     Scale [B]                                      │
│         │                       │                                           │
│         └───────────┬───────────┘                                           │
│                     ▼                                                       │
│  ┌──────────────────────────────────────┐                                   │
│  │          MapSDFEncoder               │                                   │
│  │  CNN + Scale MLP                     │                                   │
│  │  Output: [B, 64, 256]                │                                   │
│  └──────────────────────────────────────┘                                   │
│                     │                                                       │
│                     ▼                                                       │
│  ┌──────────────────────────────────────┐                                   │
│  │       Global Average Pooling         │                                   │
│  │   [B, 64, 256] → mean(dim=1)         │                                   │
│  │   Output: [B, 256]                   │                                   │
│  └──────────────────────────────────────┘                                   │
│                     │                                                       │
│                     ▼                                                       │
│  ┌──────────────────────────────────────┐                                   │
│  │       Projection MLP                 │                                   │
│  │   [B, 256] → [B, 32]                 │  Match time_emb_dim               │
│  └──────────────────────────────────────┘                                   │
│                     │                                                       │
│                     ▼                                                       │
│              sdf_embedding                                                  │
│               [B, 32]                                                       │
│                                                                             │
└─────────────────────┼───────────────────────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  t_emb [B,32] │
              │       +       │
              │ sdf_emb [B,32]│
              │       =       │
              │ cond_emb [B,32]│
              └───────────────┘
                      │
                      │ (passed to ALL ResNet blocks via FiLM)
                      │
    ┌─────────────────┴─────────────────────────────────────────┐
    │                                                           │
    ▼                                                           ▼
┌───────────────────────────────────┐     ┌───────────────────────────────────┐
│       ControlNet Encoder          │     │        Frozen Base Model          │
│         (TRAINABLE)               │     │           (LOCKED)                │
│                                   │     │                                   │
│  Input: x [B, 64, 4]              │     │                                   │
│         │                         │     │                                   │
│         ▼ rearrange               │     │                                   │
│  x_ctrl [B, 4, 64]                │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 0:                    │  │     │                                   │
│  │   ResNet×2 (cond_emb) ◄─────┼──┼─────┼── cond_emb [B,32]                 │
│  │   Self-Attention            │  │     │                                   │
│  │   (NO cross-attention)      │  │     │                                   │
│  │   ZeroConv → residual[0]    │──┼─────┼──► added to skip[3]               │
│  │   Downsample                │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 1:                    │  │     │                                   │
│  │   ResNet×2 (cond_emb) ◄─────┼──┼─────┼── cond_emb [B,32]                 │
│  │   Self-Attention            │  │     │                                   │
│  │   (NO cross-attention)      │  │     │                                   │
│  │   ZeroConv → residual[1]    │──┼─────┼──► added to skip[2]               │
│  │   Downsample                │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 2:                    │  │     │                                   │
│  │   ResNet×2 (cond_emb) ◄─────┼──┼─────┼── cond_emb [B,32]                 │
│  │   Self-Attention            │  │     │                                   │
│  │   (NO cross-attention)      │  │     │                                   │
│  │   ZeroConv → residual[2]    │──┼─────┼──► added to skip[1]               │
│  │   Downsample                │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Level 3:                    │  │     │                                   │
│  │   ResNet×2 (cond_emb) ◄─────┼──┼─────┼── cond_emb [B,32]                 │
│  │   Self-Attention            │  │     │                                   │
│  │   (NO cross-attention)      │  │     │                                   │
│  │   ZeroConv → residual[3]    │──┼─────┼──► (unused)                       │
│  │   Identity (is_last)        │  │     │                                   │
│  └─────────────────────────────┘  │     │                                   │
│         │                         │     │                                   │
│         ▼                         │     │                                   │
│  ┌─────────────────────────────┐  │     │                                   │
│  │ Mid Blocks:                 │  │     │                                   │
│  │   mid_block1 (cond_emb) ◄───┼──┼─────┼── cond_emb [B,32]                 │
│  │   mid_self_attn             │  │     │                                   │
│  │   (NO cross-attention)      │  │     │                                   │
│  │   mid_block2 (cond_emb) ◄───┼──┼─────┼── cond_emb [B,32]                 │
│  │   ZeroConv → mid_residual   │──┼─────┼──► added to mid output            │
│  └─────────────────────────────┘  │     │                                   │
│                                   │     │                                   │
└───────────────────────────────────┘     └───────────────────────────────────┘
```

## Forward Pass Pseudocode

```python
def forward(self, x, time, sdf_grid, scale):
    """
    Args:
        x: Noisy trajectory [B, H, state_dim] = [B, 64, 4]
        time: Diffusion timestep [B]
        sdf_grid: SDF tensor [B, 400, 400] or [B, 1, 400, 400]
        scale: Environment scale factor [B]
    
    Returns:
        Denoised trajectory [B, 64, 4]
    """
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: SDF Preprocessing
    # ═══════════════════════════════════════════════════════════════
    
    # Resize SDF: 400×400 → 64×64
    sdf_resized = resize_sdf(sdf_grid, target_size=64)  # [B, 1, 64, 64]
    
    # Encode SDF + Scale → Context tokens
    context = self.map_encoder(sdf_resized, scale)  # [B, 64, 256]
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Global Pooling (KEY DIFFERENCE from Approach A)
    # ═══════════════════════════════════════════════════════════════
    
    # Pool spatial tokens → single global vector
    sdf_global = context.mean(dim=1)  # [B, 256]
    
    # Project to time embedding dimension
    sdf_emb = self.sdf_projection(sdf_global)  # [B, 32]
    #   └── sdf_projection = nn.Sequential(
    #           nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 32))
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Combined Conditioning Embedding
    # ═══════════════════════════════════════════════════════════════
    
    t_emb = self.control_time_mlp(time)  # [B, 32]
    cond_emb = t_emb + sdf_emb  # [B, 32] - Combined time + SDF
    #   └── This single vector conditions ALL ResNet blocks
    #   └── All trajectory positions get the SAME global SDF info
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Prepare Trajectory Input
    # ═══════════════════════════════════════════════════════════════
    
    x_ctrl = einops.rearrange(x, 'b h c -> b c h')  # [B, 4, 64]
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: ControlNet Encoder (NO Cross-Attention)
    # ═══════════════════════════════════════════════════════════════
    
    down_residuals = []
    
    for level, (resnet1, resnet2, self_attn, _, downsample) in enumerate(self.control_downs):
        # ResNet blocks with COMBINED conditioning (time + SDF)
        x_ctrl = resnet1(x_ctrl, cond_emb)  # FiLM: scale & shift by cond_emb
        x_ctrl = resnet2(x_ctrl, cond_emb)
        
        # Self-attention only (trajectory attends to itself)
        x_ctrl = self_attn(x_ctrl)
        
        # NO cross-attention - 4th element is None or unused
        
        # Zero-conv residual
        residual = self.zero_convs[level](x_ctrl)
        down_residuals.append(residual)
        
        # Downsample
        x_ctrl = downsample(x_ctrl)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Mid Blocks (NO Cross-Attention)
    # ═══════════════════════════════════════════════════════════════
    
    x_ctrl = self.mid_block1(x_ctrl, cond_emb)  # [B, 256, 8]
    x_ctrl = self.mid_self_attn(x_ctrl)          # [B, 256, 8]
    # NO cross-attention here
    x_ctrl = self.mid_block2(x_ctrl, cond_emb)  # [B, 256, 8]
    
    mid_residual = self.zero_conv_mid(x_ctrl)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 7: Inject Residuals into Frozen Base Model
    # ═══════════════════════════════════════════════════════════════
    
    output = self.base_model(
        x, time, context=None,
        down_block_additional_residuals=down_residuals,
        mid_block_additional_residual=mid_residual
    )
    
    return output  # [B, 64, 4]
```

## Tensor Shape Flow (Global Conditioning)

### SDF to Global Embedding

| Step | Operation | Shape |
|------|-----------|-------|
| Input SDF | From environment | `[B, 400, 400]` |
| Resize | Bilinear interpolate | `[B, 1, 64, 64]` |
| MapSDFEncoder | CNN + scale | `[B, 64, 256]` |
| **Global Pool** | mean(dim=1) | **`[B, 256]`** |
| Projection | MLP | `[B, 32]` |
| Add to time | t_emb + sdf_emb | `[B, 32]` |

### ControlNet Encoder (Same as Approach A, but no cross-attention)

| Level | Input | After ResNets | After Self-Attn | After Downsample | Residual |
|-------|-------|---------------|-----------------|------------------|----------|
| 0 | `[B,4,64]` | `[B,32,64]` | `[B,32,64]` | `[B,32,32]` | `[B,32,64]` |
| 1 | `[B,32,32]` | `[B,64,32]` | `[B,64,32]` | `[B,64,16]` | `[B,64,32]` |
| 2 | `[B,64,16]` | `[B,128,16]` | `[B,128,16]` | `[B,128,8]` | `[B,128,16]` |
| 3 | `[B,128,8]` | `[B,256,8]` | `[B,256,8]` | `[B,256,8]` | `[B,256,8]` |

## Components to Implement/Modify

| File | Change | Description |
|------|--------|-------------|
| `map_encoder.py` | **ADD** | `resize_sdf(sdf_tensor, target_size=64)` function |
| `map_encoder.py` | **MODIFY** | Add `pool_global()` method or make it optional |
| `controlnet.py` | **MODIFY** | Add `MapSDFEncoder` as `self.map_encoder` |
| `controlnet.py` | **ADD** | `self.sdf_projection` MLP (256 → 32) |
| `controlnet.py` | **MODIFY** | `forward()` to compute `cond_emb = t_emb + sdf_emb` |

## Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Simpler Implementation** | No SpatialTransformer blocks needed |
| **Fewer Parameters** | Only ~50K additional params (projection MLP) |
| **Faster Training** | No attention computation overhead |
| **Matches Current ControlNet** | Current implementation already uses this pattern |
| **Easier to Debug** | Fewer moving parts |

## Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **No Spatial Specificity** | All trajectory points get identical SDF info |
| **Compressed Information** | 64 spatial tokens → 1 vector (loses detail) |
| **Harder to Learn Local Avoidance** | Can't learn "obstacle HERE, avoid THIS region" |
| **Less Expressive** | Global conditioning is fundamentally limited |

---

# Side-by-Side Comparison

## Architecture Differences

| Aspect | Approach A (Cross-Attention) | Approach B (Global) |
|--------|------------------------------|---------------------|
| **SDF Context** | `[B, 64, 256]` tokens | `[B, 32]` vector |
| **Conditioning Mechanism** | Cross-attention in each level | FiLM in ResNet blocks |
| **Per-Level SpatialTransformer** | Yes (4 + 1 mid) | No |
| **Information Flow** | Trajectory → attends to → SDF regions | SDF → global → all positions equally |

## Parameter Count Comparison

| Component | Approach A | Approach B |
|-----------|------------|------------|
| MapSDFEncoder | ~200K | ~200K |
| Projection MLP | - | ~20K |
| SpatialTransformer ×5 | ~500K | - |
| ControlNet Encoder | ~3M | ~3M |
| Zero Convolutions | ~10K | ~10K |
| **Total Additional** | **~3.7M** | **~3.2M** |

## Computational Cost

| Metric | Approach A | Approach B |
|--------|------------|------------|
| Forward pass ops | Higher (attention) | Lower |
| Memory usage | Higher (attention maps) | Lower |
| Training speed | Slower (~1.3-1.5×) | Faster (baseline) |
| Inference speed | Slower (~1.2×) | Faster (baseline) |

## Expressiveness

| Capability | Approach A | Approach B |
|------------|------------|------------|
| "Avoid obstacle at position X" | ✓ Can learn | ✗ Difficult |
| "Environment has obstacles" | ✓ | ✓ |
| "Scale is 1.3x" | ✓ | ✓ |
| "Navigate through corridor" | ✓ Explicit attention | ○ Implicit |
| "This region is free space" | ✓ Spatial tokens | ✗ Lost in pooling |

---

# Recommendation

## For Initial Implementation: Approach B (Global Conditioning)

**Rationale:**
1. **Lower risk** - Simpler to implement and debug
2. **Faster iteration** - Train and validate pipeline quickly  
3. **Baseline establishment** - Working baseline for comparison
4. **Upgrade path** - Can add cross-attention later if needed

## For Production/Best Results: Approach A (Cross-Attention)

**Rationale:**
1. **More expressive** - Spatial specificity matters for obstacle avoidance
2. **Proven pattern** - Matches successful SD ControlNet
3. **Infrastructure exists** - SpatialTransformer already in codebase
4. **Natural for planning** - "Look where obstacles are" reasoning

## Suggested Development Path

```
Phase 1: Implement Approach B (Global)
    └── Validate pipeline works end-to-end
    └── Establish baseline metrics
    └── Quick iteration on data/training

Phase 2: Upgrade to Approach A (Cross-Attention)
    └── Add SpatialTransformer to ControlNet encoder
    └── Compare against Approach B baseline
    └── Tune attention hyperparameters

Phase 3: Evaluate and Select
    └── Compare trajectory quality metrics
    └── Compare training stability
    └── Select final architecture
```

---

# Implementation Checklist

## Shared Components (Both Approaches)

- [ ] Add `resize_sdf()` function to `mmd/models/helpers/map_encoder.py`
- [ ] Create `ControlNetTrajectoryDataset` with SDF + scale fields
- [ ] Update training script to pass SDF through pipeline

## Approach B Specific

- [ ] Add `self.sdf_projection` MLP to `MMDControlNet`
- [ ] Modify `forward()` to compute `cond_emb = t_emb + sdf_emb`
- [ ] Test training loop

## Approach A Specific

- [ ] Add `self.map_encoder` to `MMDControlNet`
- [ ] Modify `_create_control_encoder()` to include `SpatialTransformer`
- [ ] Modify `_create_control_mid()` to include cross-attention
- [ ] Modify `forward()` to pass context to all cross-attention blocks
- [ ] Match `context_dim=256` with MapSDFEncoder output

---

# References

- **ControlNet Paper**: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models"
- **Base TemporalUnet**: `mmd/models/diffusion_models/temporal_unet.py`
- **MapSDFEncoder**: `mmd/models/helpers/map_encoder.py`
- **SpatialTransformer**: `mmd/models/layers/layers_attention.py`
- **Current ControlNet**: `mmd/models/diffusion_models/controlnet.py`
