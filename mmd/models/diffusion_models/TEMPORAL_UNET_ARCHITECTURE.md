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

## Motivation Behind `[B, 64, 256]` Context Shape

The `MapSDFEncoder` outputs a tensor of shape `[B, 64, 256]`. This section explains the reasoning behind each dimension.

### Breaking Down the Shape

| Dimension | Value | Meaning |
|-----------|-------|---------|
| `B` | Batch size | Standard batching |
| `64` | Number of spatial tokens | Sequence length for cross-attention |
| `256` | Feature dimension per token | Embedding size per spatial location |

### Why 64 Tokens?

The 64 comes from the CNN output spatial dimensions: **8×8 = 64**

```
Input SDF:     [B, 1, 64, 64]
                      ↓ Conv stride=2
Layer 1:       [B, 32, 32, 32]
                      ↓ Conv stride=2
Layer 2:       [B, 64, 16, 16]
                      ↓ Conv stride=2
Layer 3:       [B, 256, 8, 8]
                      ↓ Flatten spatial dims
Output:        [B, 256, 64]  →  permute  →  [B, 64, 256]
```

**Why 8×8 specifically?**
- Three stride-2 convolutions: 64 → 32 → 16 → 8 (standard CNN downsampling)
- Each of the 64 tokens represents a **8×8 = 64 pixel region** of the original 64×64 SDF
- In physical space: each token covers `(2.0/8) × (2.0/8) = 0.25 × 0.25` units of the environment
- This provides meaningful spatial decomposition while keeping attention cost manageable

### Why 256 Feature Dimension?

The 256-dimensional feature vector per token is chosen based on:

| Factor | Explanation |
|--------|-------------|
| **Attention compatibility** | `SpatialTransformer` in TemporalUnet uses `context_dim` for cross-attention key/value projections. Must match. |
| **Representational capacity** | 256 dims can encode rich spatial features (obstacle shapes, distances, local geometry) |
| **Standard practice** | 256 is common in transformer architectures (BERT-small, many vision transformers) |
| **Memory/compute tradeoff** | Larger = more expressive but slower attention; 256 is a good balance |

### Why This Format for Cross-Attention?

Cross-attention in transformers requires:
- **Query**: From the main sequence (trajectory) `[B, H', dim]`
- **Key/Value**: From the context (SDF) `[B, seq_len, context_dim]`

The `[B, 64, 256]` format provides:
- `64` = sequence length for Key/Value (number of spatial tokens)
- `256` = `context_dim` for projection layers

```
Cross-Attention at encoder level 0:
  Query:  trajectory [B, 64, 32]  →  project  →  [B, 64, d_head × n_heads]
  Key:    context    [B, 64, 256] →  project  →  [B, 64, d_head × n_heads]
  Value:  context    [B, 64, 256] →  project  →  [B, 64, d_head × n_heads]
  
  Attention matrix: [B, n_heads, 64, 64]  (trajectory_len × context_len)
```

### Spatial Correspondence

Each of the 64 tokens has a **spatial meaning** in the environment:

```
SDF Grid [64×64]           CNN Output [8×8]           Context Tokens [64]
┌────────────────┐         ┌──────────┐               
│ ░░░░ ░░░░ ░░░░ │         │ 0  1  2  │  →  Flatten  →  [0, 1, 2, ..., 63]
│ ░░░░ ░░░░ ░░░░ │    →    │ 8  9  10 │                    │
│ ░░░░ ░░░░ ░░░░ │         │ 16 17 18 │                    │
└────────────────┘         └──────────┘                    ▼
                                                    Token i encodes
                                                    region (i//8, i%8)
                                                    of the 8×8 grid
```

**Physical interpretation:**
- Token 0 represents the **top-left region** of the environment (around [-1, 1] to [-0.75, 0.75])
- Token 63 represents the **bottom-right region** (around [0.75, -0.75] to [1, -1])
- When the trajectory cross-attends to context, each waypoint can learn to "look at" relevant spatial regions

### Alternative Designs Considered

| Design | Shape | Pros | Cons |
|--------|-------|------|------|
| **Current (8×8)** | `[B, 64, 256]` | Good balance of detail and speed | Some spatial detail lost |
| **Larger (16×16)** | `[B, 256, 256]` | Finer spatial resolution | 4× attention cost |
| **Smaller (4×4)** | `[B, 16, 256]` | Very fast attention | Too coarse, loses spatial info |
| **Global only** | `[B, 256]` | No attention overhead | No spatial specificity (Approach B) |

### Summary

The `[B, 64, 256]` shape is chosen because:

1. **64 tokens** = 8×8 spatial grid from CNN downsampling, provides meaningful spatial decomposition of the environment
2. **256 features** = rich enough to encode obstacle geometry, distances, and local structure; matches common transformer conventions
3. **Sequence format** = ready for cross-attention with trajectory features in `SpatialTransformer`
4. **Balanced tradeoff** = good spatial detail (each token = 0.25×0.25 physical units) with manageable attention cost (64×64 attention matrix)

---

## Critical Design Consideration: Is the Scale Parameter Redundant?

### The Observation

Currently, the `MapSDFEncoder` takes both `sdf_grid` and `scale_scalar` as inputs. However, there's a fundamental question: **Is the scale parameter redundant given that the SDF already encodes the scaled geometry?**

### How Scale Affects the SDF

When the environment is instantiated with a scale parameter, the obstacle geometry is scaled **before** the SDF is computed:

```python
# In env_highways_2d.py (and similar environments)
class EnvHighways2D(EnvBase):
    def __init__(self, scale=1.0, ...):
        obj_list = [
            MultiBoxField(
                centers=np.array([...]),
                sizes=np.array([...]) * scale,  # ← Obstacles scaled here
                ...
            ),
        ]
        super().__init__(
            obj_fixed_list=[ObjectField(obj_list, 'highways2d')],
            precompute_sdf_obj_fixed=True,  # SDF computed from scaled obstacles
            ...
        )
```

### The Flow

```
env_scale = 1.5 (hyperparameter at inference)
       │
       ▼
┌─────────────────────────────────┐
│  Environment Constructor        │
│  EnvHighways2D(scale=1.5)       │
│                                 │
│  Obstacle sizes multiplied by   │
│  scale BEFORE SDF computation   │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  GridMapSDF                     │
│                                 │
│  SDF computed from SCALED       │
│  obstacle positions/sizes       │
│                                 │
│  Resulting SDF values already   │
│  reflect narrower corridors,    │
│  larger obstacles, etc.         │
└─────────────────────────────────┘
       │
       ▼
   sdf_tensor [400, 400]
   (geometry at scale 1.5 is ALREADY encoded)
```

### Implications

The SDF grid **fully encodes** the effect of scaling:
- At scale=1.0: Corridors have width X, SDF values reflect this
- At scale=1.5: Corridors are narrower (obstacles 1.5× larger), SDF values are smaller in corridors
- The model can learn appropriate trajectory behavior **purely from the SDF geometry**

### Arguments FOR Keeping Scale Parameter

| Argument | Explanation |
|----------|-------------|
| **Faster convergence** | Explicit scale might help the network learn faster as an auxiliary signal |
| **Ambiguity resolution** | Two different environments at different scales could theoretically have similar local SDF patterns |
| **Trajectory normalization** | If trajectories need to be normalized/denormalized by scale |
| **Ablation studies** | Can test if model uses scale vs. learns purely from geometry |
| **Future multi-scale** | If training on multiple scales simultaneously, scale helps distinguish |

### Arguments AGAINST Scale Parameter (Redundancy)

| Argument | Explanation |
|----------|-------------|
| **Already encoded in SDF** | Geometry fully captures scale through obstacle sizes |
| **Simpler architecture** | One less input to process, fewer parameters |
| **Purer conditioning** | Model learns from actual environment geometry, not a scalar hint |
| **No consistency risk** | Can't have mismatch between scale param and actual SDF |
| **Generalization** | Model learns to read geometry, not rely on explicit scale |

### Recommendation

**For initial implementation**: Remove the scale parameter and condition only on SDF.

**Rationale**:
1. The SDF already contains all geometric information about scale effects
2. Simpler is better for debugging and validation
3. If the model fails to generalize, scale can be added back as an experiment
4. This forces the model to learn from actual obstacle geometry

### Simplified MapSDFEncoder (Without Scale)

```python
class MapSDFEncoder(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 64→32
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32→16
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),  # 16→8
            nn.SiLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

    def forward(self, sdf_grid):
        """
        Args:
            sdf_grid: [B, 1, 64, 64] - Resized SDF grid (already encodes scale)
        
        Returns:
            context: [B, 64, 256] - Spatial tokens for cross-attention
                 OR: [B, 256] - Global vector (after pooling)
        """
        img_features = self.cnn(sdf_grid)  # [B, 256, 8, 8]
        B, C, H, W = img_features.shape
        flat_features = img_features.view(B, C, -1).permute(0, 2, 1)  # [B, 64, 256]
        return flat_features
```

### Optional: Keep Scale for Experiments

If we want to experiment with scale as an auxiliary signal, make it optional:

```python
class MapSDFEncoder(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=256, use_scale=False):
        super().__init__()
        self.use_scale = use_scale
        self.cnn = nn.Sequential(...)
        
        if use_scale:
            self.scale_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, output_dim)
            )

    def forward(self, sdf_grid, scale_scalar=None):
        img_features = self.cnn(sdf_grid)
        flat_features = img_features.view(B, C, -1).permute(0, 2, 1)
        
        if self.use_scale and scale_scalar is not None:
            scale_emb = self.scale_mlp(scale_scalar.unsqueeze(-1))
            flat_features = flat_features + scale_emb.unsqueeze(1)
        
        return flat_features
```

### Updated Architecture Diagrams

With this consideration, the architecture diagrams for both approaches should show scale as **optional/dashed**:

```
SDF Grid [400×400]
       │
       ▼
┌─────────────────┐
│   resize_sdf()  │
└─────────────────┘
       │
       ▼
SDF Grid [B, 1, 64, 64]     Scale [B] (OPTIONAL)
       │                       ╎
       └───────────┬───────────┘
                   ▼
┌──────────────────────────────────────┐
│          MapSDFEncoder               │
│  CNN: encodes SDF geometry           │
│  Scale MLP: (optional, dashed)       │
│  Output: [B, 64, 256]                │
└──────────────────────────────────────┘
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
- [ ] Decide on scale parameter: remove (recommended) or keep as optional
- [ ] Update `MapSDFEncoder` to make scale optional with `use_scale=False` default
- [ ] Create `ControlNetTrajectoryDataset` with SDF field (scale optional)
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

# Open Design Questions

1. **Scale parameter**: Remove entirely (simpler) or keep as optional (more flexible)?
   - **Recommendation**: Remove for initial implementation, add back if needed
   
2. **SDF resolution**: Keep 64×64 or experiment with other sizes?
   - **Recommendation**: Start with 64×64, sufficient for robot size

3. **Cross-attention vs Global**: Which approach first?
   - **Recommendation**: Start with Approach B (Global), upgrade to A if needed

---

# ControlNet Training and Deployment Strategy

This section provides comprehensive details on the training data pipeline, deployment architecture, and implementation strategy for the ControlNet extension.

---

## One ControlNet Adapter Per Environment Type

### Core Principle

Each environment type (EnvHighways2D, EnvConveyor2D, etc.) has its own pretrained diffusion model. We create **one ControlNet adapter per pretrained model**, where the adapter conditions that specific model on scaled versions of the **same** environment type.

### Why Not a Universal ControlNet?

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Universal ControlNet** | One adapter for all environments | Single model to maintain | Different environments have different geometry patterns; may confuse the adapter |
| **Per-Environment ControlNet** | One adapter per environment type | Adapter learns specific geometry patterns | Multiple adapters to train and store |

**Decision**: Per-environment ControlNet adapters.

**Rationale**:
1. Each pretrained model was trained specifically on one environment type
2. Geometry patterns differ significantly between environments (highways vs. conveyor vs. random obstacles)
3. Adapter specialization leads to better quality and faster convergence
4. Memory overhead is minimal (~3-5MB per adapter)

### Directory Structure

```
data_trained_models/
├── EnvHighways2D-RobotPlanarDisk/
│   ├── checkpoints/
│   │   └── ckpt_best.pth              # Original frozen TemporalUnet
│   └── controlnet/
│       ├── controlnet.pth             # Highways ControlNet adapter
│       └── sdf_cache/                 # Pre-computed SDFs (optional)
│           ├── sdf_scale_1.0.pt
│           ├── sdf_scale_1.1.pt
│           └── ...
│
├── EnvConveyor2D-RobotPlanarDisk/
│   ├── checkpoints/
│   │   └── ckpt_best.pth              # Original frozen TemporalUnet
│   └── controlnet/
│       ├── controlnet.pth             # Conveyor ControlNet adapter
│       └── sdf_cache/
│           └── ...
│
├── EnvRandomBox2D-RobotPlanarDisk/
│   ├── checkpoints/
│   │   └── ckpt_best.pth
│   └── controlnet/
│       └── controlnet.pth
│
└── ... (one per environment type)
```

### Loading Pattern

```python
def load_controlled_model(env_type: str, scale: float):
    """
    Load base model + ControlNet adapter for a specific environment type.
    
    Args:
        env_type: e.g., "EnvHighways2D-RobotPlanarDisk"
        scale: Environment scale factor (for SDF lookup)
    
    Returns:
        ControlledDiffusionModel ready for inference
    """
    base_path = f"data_trained_models/{env_type}"
    
    # Load frozen base model
    base_model = load_pretrained_unet(f"{base_path}/checkpoints/ckpt_best.pth")
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Load ControlNet adapter (trained for this specific environment)
    controlnet = load_controlnet(f"{base_path}/controlnet/controlnet.pth")
    
    # Optionally load pre-computed SDF
    sdf_grid = load_precomputed_sdf(f"{base_path}/controlnet/sdf_cache", scale)
    
    return ControlledDiffusionModel(base_model, controlnet), sdf_grid
```

---

## Training Data Strategy

### Philosophy: Minimal Data for Adapters

ControlNet adapters are designed to work with **significantly less data** than training from scratch. The frozen base model already contains rich trajectory priors; the adapter only needs to learn the conditioning signal.

| Training Type | Typical Data Size | Rationale |
|---------------|-------------------|-----------|
| Original TemporalUnet | ~100K trajectories | Learn complete trajectory distribution from scratch |
| ControlNet Adapter | ~1K-10K trajectories | Adapter only learns to "steer" existing prior |

**Exact data quantity**: Left open for experimentation. Start with ~1K trajectories per scale and increase if needed.

### Data Generation Pipeline

#### Step 1: Generate Trajectories on Scaled Environments

Use the existing MMD model **with guidance** to generate valid trajectories on scaled environments:

```python
def generate_multiscale_trajectories(env_class, scales, num_per_scale):
    """
    Generate training trajectories for ControlNet using existing MMD.
    
    Key insight: Even on scaled maps, MMD with guidance produces valid
    trajectories. The SDF collision guidance ensures feasibility.
    """
    all_trajectories = []
    
    for scale in scales:
        # Create scaled environment
        env = env_class(scale=scale, ...)
        
        # Use existing MMD with guidance (not the ControlNet)
        planner = MPD(
            model=base_model,  # Original pretrained model
            env=env,
            use_guidance=True,  # SDF collision guidance enabled
            ...
        )
        
        for _ in range(num_per_scale):
            # Sample start/goal
            start, goal = env.sample_valid_start_goal()
            
            # Plan trajectory (with guidance for feasibility)
            trajectory = planner.plan(start, goal)
            
            # Store trajectory + scale identifier
            all_trajectories.append({
                'trajectory': trajectory,  # [64, 4]
                'scale': scale,
                'start': start,
                'goal': goal,
            })
    
    return all_trajectories
```

**Why this works**:
- The base model generates reasonable trajectory priors even on scaled maps
- Guidance (SDF collision avoidance) ensures the trajectories are valid
- The ControlNet adapter learns to internalize this guidance into the model itself

#### Step 2: Define Discrete Scale Set

```python
# Discrete scale values for training
TRAINING_SCALES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Why discrete?
# - Finite set enables pre-computing SDFs
# - Clear interpolation targets for model generalization
# - Simplifies data management and batching
```

### Scale Distribution in Training

| Strategy | Description | Recommendation |
|----------|-------------|----------------|
| **Uniform** | Equal samples per scale | Start here |
| **Weighted** | More samples at harder scales (1.4, 1.5) | If uniform struggles |
| **Curriculum** | Start easy (1.0-1.2), gradually add harder | If training unstable |

```python
# Uniform distribution (recommended starting point)
scale_weights = {
    1.0: 1.0,
    1.1: 1.0,
    1.2: 1.0,
    1.3: 1.0,
    1.4: 1.0,
    1.5: 1.0,
}

# Example weighted distribution (if needed)
scale_weights_hard = {
    1.0: 0.5,   # Base scale - model already good here
    1.1: 0.8,
    1.2: 1.0,
    1.3: 1.2,
    1.4: 1.5,   # Harder - more samples
    1.5: 2.0,   # Hardest - most samples
}
```

---

## OPEN QUESTION: Discrete vs. Continuous Scale Sampling

**Status**: Needs discussion with advisors

This section explores whether to use discrete scale values or continuous sampling during training. This is an important design decision with significant implications for generalization.

### Current Approach: Discrete Scales

```python
TRAINING_SCALES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # 6 discrete values
```

### Alternative: Continuous Scale Sampling

```python
# Sample any scale in the continuous range
scale = np.random.uniform(1.0, 1.5)  # Infinite variety
```

### Tradeoffs

| Aspect | Discrete (6 values) | Continuous (1.0-1.5 range) |
|--------|---------------------|---------------------------|
| **Variety** | 6 unique geometries | Infinite variety |
| **Generalization** | Risk of memorizing 6 patterns | Better interpolation to unseen scales |
| **SDF availability** | Pre-computed, exact | Need to generate somehow |
| **Encoder efficiency** | Encode 6 SDFs once, index into results | Must encode varying SDFs per batch |
| **Training speed** | Faster (finite encoding set) | Slower (continuous encoding) |

### Options for Continuous Scales

#### Option A: Pre-compute Dense Grid

```python
# Pre-compute many scales (e.g., 51 values at 0.01 increments)
scales = np.arange(1.0, 1.51, 0.01)  # [1.00, 1.01, 1.02, ..., 1.50]

# At training: sample continuous scale, snap to nearest 0.01
continuous_scale = np.random.uniform(1.0, 1.5)
discrete_scale = round(continuous_scale, 2)  # Snap to nearest 0.01
sdf = sdf_cache[discrete_scale]
```

**Pros**:
- More variety than 6 (51 unique geometries)
- Still enables pre-computation
- Model sees diverse scales, learns smooth generalization
- 0.01 increment is effectively continuous from model's perspective

**Cons**:
- 51 SDFs per environment (~800KB storage - very manageable)
- Still technically discrete (but imperceptibly so)

#### Option B: Interpolate Between Pre-computed SDFs

```python
# Have 6 (or more) anchor SDFs
scale = 1.23  # Continuous sample

# Find neighbors: 1.2 and 1.3
lower, upper = 1.2, 1.3
alpha = (scale - lower) / (upper - lower)  # = 0.3

# Interpolate: sdf_1.23 ≈ (1-alpha) * sdf_1.2 + alpha * sdf_1.3
sdf_interpolated = (1 - alpha) * sdf_cache[1.2] + alpha * sdf_cache[1.3]
```

**Pros**:
- True continuous scales
- Minimal storage (only anchor SDFs)

**Cons**:
- SDF values represent signed distances - **unclear if interpolation is geometrically meaningful**
- Interpolated SDF might not be valid (e.g., obstacles could "blur")
- Needs experimental validation

#### Option C: Compute SDFs On-the-fly

```python
# Each training sample: random scale, compute fresh SDF
scale = np.random.uniform(1.0, 1.5)
env = EnvHighways2D(scale=scale)
sdf = env.compute_sdf()  # Expensive!
```

**Pros**:
- Exact SDFs for any scale
- True continuous sampling

**Cons**:
- Very slow training (SDF computation is expensive)
- Not practical for large-scale training

### Recommendation

**Hybrid approach: Dense discrete sampling (Option A)**

```python
# Pre-compute 51 SDFs (0.01 increments)
TRAINING_SCALES = np.arange(1.0, 1.51, 0.01).round(2).tolist()
# [1.0, 1.01, 1.02, ..., 1.49, 1.50]

# During training: sample "continuous", snap to nearest 0.01
def sample_scale():
    continuous = np.random.uniform(1.0, 1.5)
    return round(continuous, 2)
```

**Why this is effectively continuous**:
- The geometry difference between scale 1.23 and 1.24 is imperceptible
- Model learns smooth interpolation across the range
- Still benefits from pre-computation efficiency

### Questions for Advisors

1. **Is 51 discrete points sufficient?** Or should we use finer granularity (101 points at 0.005 increments)?

2. **Is SDF interpolation valid?** Worth experimenting to see if interpolated SDFs produce reasonable results?

3. **Scale sampling distribution**: Uniform [1.0, 1.5] or weighted toward harder scales?

4. **Generalization testing**: Should we test on scales outside training range (e.g., 1.6, 0.9)?

---

## OPEN QUESTION: Trainable Encoder and Efficient Encoding

**Status**: Needs discussion with advisors

This section addresses whether the `MapSDFEncoder` should be trainable (following original ControlNet) and how to efficiently handle encoding given our finite SDF set.

### Background: Original ControlNet Preprocessing

In the original ControlNet paper, the condition (Canny edges, depth, pose) goes through a **trainable hint encoder**:

```python
# From ControlNet paper - the "hint encoder" IS learnable
class ControlNet:
    def __init__(self):
        # This is TRAINABLE - learns which condition features matter
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_conv(128, 320),  # Projects to match UNet channels
        )
```

**Key insight**: The encoder learns to extract features from the condition that are relevant for the generation task.

### The Question: Should `MapSDFEncoder` Be Trainable?

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Frozen Encoder** | Pre-train or freeze encoder, only train ControlNet body | Can pre-compute encoded features | Limited adaptation to task |
| **Trainable Encoder** | Encoder is part of ControlNet, trained end-to-end | Learns task-specific features | Cannot pre-compute encoded features |

**Following ControlNet faithfully**: The encoder SHOULD be trainable.

**Rationale**:
1. The encoder needs to learn which SDF features matter for trajectory planning
2. "Obstacle at position X" needs to translate to "avoid this region in trajectory"
3. This mapping is task-specific and should be learned
4. Original ControlNet uses trainable preprocessing

### The Efficiency Question

If encoder is trainable, we cannot pre-compute encoded features (they change as weights update). But we still have only **N unique SDF grids** (where N = number of scales: 6, 51, etc.).

**Naive approach (wasteful)**:
```python
# Batch has B=64 samples, but many share the same scale
sdf_batch = [B, 1, 64, 64]  # Contains duplicates!

# Wastefully encodes the same SDF multiple times
encoded = encoder(sdf_batch)  # [B, 64, 256]

# If 10 samples have scale=1.3, we encode sdf_1.3 ten times!
```

**Efficient approach**:
```python
# Only N unique SDFs exist (N = 6 or 51 depending on granularity)
unique_sdfs = torch.stack([sdf_cache[s] for s in TRAINING_SCALES])  # [N, 1, 64, 64]

# Encode all N SDFs once per forward pass
all_encoded = encoder(unique_sdfs)  # [N, 64, 256] - only N forward passes!

# For each batch, just index into pre-encoded features
scale_indices = batch['scale_idx']  # [B] values in {0, 1, ..., N-1}
encoded_batch = all_encoded[scale_indices]  # [B, 64, 256] - just indexing!
```

### Design with Trainable Encoder + Efficient Indexing

```
┌─────────────────────────────────────────────────────────────────┐
│                   EACH FORWARD PASS                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 1: Encode ALL unique SDFs once                       │  │
│  │                                                           │  │
│  │ unique_sdfs [N, 1, 64, 64] ──► encoder ──► [N, 64, 256]   │  │
│  │                                  (trainable)              │  │
│  │                                                           │  │
│  │ N = number of unique scales (6, 51, or more)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 2: Index by batch scale indices                      │  │
│  │                                                           │  │
│  │ scale_indices [B] ──► gather ──► encoded_batch [B,64,256] │  │
│  │                        (no compute, just indexing)        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  Use encoded_batch [B, 64, 256] in ControlNet                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class ControlledDiffusionModel(nn.Module):
    def __init__(self, base_model, controlnet, sdf_cache: SDFCache):
        super().__init__()
        self.base_model = base_model  # Frozen
        self.controlnet = controlnet   # Trainable (includes MapSDFEncoder)
        
        # Cache of raw SDF grids: [N, 1, 64, 64]
        self.register_buffer(
            'unique_sdfs',
            torch.stack([sdf_cache.get_sdf(s) for s in sdf_cache.scales])
        )
        # Scale value to index mapping
        self.scale_to_idx = {s: i for i, s in enumerate(sdf_cache.scales)}
    
    def forward(self, x, time, scales):
        """
        Args:
            x: Noisy trajectory [B, 64, 4]
            time: Diffusion timestep [B]
            scales: Scale values for each sample [B]
        """
        # Step 1: Encode ALL unique SDFs once (N forward passes, not B)
        all_encoded = self.controlnet.map_encoder(self.unique_sdfs)  # [N, 64, 256]
        
        # Step 2: Index into encoded features by scale
        scale_indices = torch.tensor([self.scale_to_idx[s.item()] for s in scales])
        encoded_batch = all_encoded[scale_indices]  # [B, 64, 256]
        
        # Step 3: Rest of ControlNet forward
        residuals = self.controlnet(x, time, encoded_batch)
        
        # Step 4: Inject into frozen base model
        output = self.base_model(x, time, residuals=residuals)
        return output
```

### Summary: What Gets Trained

| Component | Frozen | Trainable |
|-----------|--------|-----------|
| Base TemporalUnet | ✓ | |
| ControlNet encoder/mid blocks | | ✓ |
| MapSDFEncoder (hint encoder) | | ✓ |
| Zero convolutions | | ✓ |

### Questions for Advisors

1. **Trainable vs. frozen encoder**: Should we follow ControlNet exactly (trainable) or freeze the encoder?

2. **Encoding frequency**: Is encoding N SDFs per forward pass acceptable, or should we explore caching strategies that work with trainable encoders?

3. **Encoder architecture**: Is a simple CNN sufficient, or should we consider pretrained vision encoders?

4. **Gradient flow**: Any concerns about gradients flowing through the indexing operation?

---

## Pre-computed SDF Grids

### Motivation

Since we have a **finite, discrete set** of scales and **static map geometries**, we can pre-compute all SDF grids before training. This provides:

1. **No runtime SDF computation** - Major speedup during training
2. **Consistent SDF values** - No floating-point variations between runs
3. **Simplified data pipeline** - Just look up by scale index

### Storage Strategy

**Decision**: Store full 400×400 SDFs, resize to 64×64 once before training (not every batch).

**Rationale**:
- 400×400 preserves maximum precision for collision checking
- 64×64 resize is a one-time cost at training start
- Dictionary lookup by scale is O(1) per batch

### Directory Structure for Pre-computed SDFs

```
precomputed_sdfs/
├── EnvHighways2D/
│   ├── sdf_scale_1.0.pt     # torch.save({'sdf': [400, 400], 'scale': 1.0})
│   ├── sdf_scale_1.1.pt
│   ├── sdf_scale_1.2.pt
│   ├── sdf_scale_1.3.pt
│   ├── sdf_scale_1.4.pt
│   └── sdf_scale_1.5.pt
│
├── EnvConveyor2D/
│   ├── sdf_scale_1.0.pt
│   └── ... (same structure)
│
└── ... (one folder per environment type)
```

### SDF Pre-computation Script

```python
def precompute_sdfs(env_class, env_name, scales, output_dir):
    """
    Pre-compute and save SDF grids for all scales of an environment.
    
    Run ONCE before training. Output is cached for all future training runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for scale in scales:
        # Create environment at this scale
        env = env_class(
            scale=scale,
            precompute_sdf_obj_fixed=True,  # Compute SDF
            sdf_cell_size=0.005,  # Native 400×400 resolution
            ...
        )
        
        # Extract SDF grid
        sdf_tensor = env.grid_map_sdf.sdf_tensor  # [400, 400]
        
        # Save full resolution
        save_path = f"{output_dir}/sdf_scale_{scale}.pt"
        torch.save({
            'sdf': sdf_tensor,
            'scale': scale,
            'shape': sdf_tensor.shape,
            'cell_size': 0.005,
        }, save_path)
        
        print(f"Saved: {save_path} - shape {sdf_tensor.shape}")
```

### SDF Loading and Caching at Training Start

```python
class SDFCache:
    """
    Load and cache all SDFs for a specific environment type.
    Resize to 64×64 once at construction, not per-batch.
    """
    
    def __init__(self, env_name: str, scales: list, sdf_dir: str, target_size: int = 64):
        """
        Args:
            env_name: e.g., "EnvHighways2D"
            scales: List of scale values [1.0, 1.1, ...]
            sdf_dir: Path to precomputed_sdfs/{env_name}/
            target_size: Resize target (64×64)
        """
        self.scale_to_sdf = {}
        
        for scale in scales:
            # Load full 400×400 SDF
            data = torch.load(f"{sdf_dir}/{env_name}/sdf_scale_{scale}.pt")
            sdf_full = data['sdf']  # [400, 400]
            
            # Resize to 64×64 ONCE (not every training loop)
            sdf_resized = self._resize_sdf(sdf_full, target_size)  # [1, 64, 64]
            
            # Store in dictionary: scale → resized SDF
            self.scale_to_sdf[scale] = sdf_resized
        
        print(f"SDFCache initialized: {len(scales)} scales, each {target_size}×{target_size}")
    
    def _resize_sdf(self, sdf_tensor, target_size):
        """Resize SDF from 400×400 to target_size×target_size."""
        # Add batch and channel dims: [400, 400] → [1, 1, 400, 400]
        sdf = sdf_tensor.unsqueeze(0).unsqueeze(0).float()
        
        # Bilinear interpolation
        sdf_resized = F.interpolate(
            sdf, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return sdf_resized.squeeze(0)  # [1, 64, 64]
    
    def get_sdf(self, scale: float) -> torch.Tensor:
        """
        Get pre-computed, pre-resized SDF for a given scale.
        O(1) dictionary lookup.
        
        Returns:
            SDF tensor [1, 64, 64]
        """
        return self.scale_to_sdf[scale]
    
    def get_batch_sdf(self, scales: torch.Tensor) -> torch.Tensor:
        """
        Get SDFs for a batch of scale values.
        
        Args:
            scales: Tensor of scale values [B]
        
        Returns:
            Batch of SDFs [B, 1, 64, 64]
        """
        sdfs = [self.get_sdf(s.item()) for s in scales]
        return torch.stack(sdfs, dim=0)  # [B, 1, 64, 64]
```

### Alternative: Scale Index Instead of Scale Value

For simpler lookups, use integer indices:

```python
# Mapping: index → scale
SCALE_INDEX_TO_VALUE = {
    0: 1.0,
    1: 1.1,
    2: 1.2,
    3: 1.3,
    4: 1.4,
    5: 1.5,
}

# In dataset, store scale_idx instead of scale
sample = {
    'trajectory': trajectory,  # [64, 4]
    'scale_idx': 3,            # Means scale=1.3
}

# In SDFCache, lookup by index
def get_sdf_by_idx(self, scale_idx: int) -> torch.Tensor:
    scale = SCALE_INDEX_TO_VALUE[scale_idx]
    return self.scale_to_sdf[scale]
```

---

## Training Batch Structure

### Batch Composition

All samples in a training batch are from the **same environment type** (since each ControlNet is environment-specific), but samples **differ in scale** and thus have different SDF grids.

```python
# Training batch for EnvHighways2D ControlNet
batch = {
    'trajectory': torch.Tensor,  # [B, 64, 4] - Trajectories from MMD on scaled Highways
    'sdf_grid': torch.Tensor,    # [B, 1, 64, 64] - Pre-computed SDF for each sample's scale
    'scale': torch.Tensor,       # [B] - Scale values (optional, for logging/debugging)
    'start': torch.Tensor,       # [B, 4] - Start positions (optional)
    'goal': torch.Tensor,        # [B, 4] - Goal positions (optional)
}
```

### Example Batch

```python
# Example: B=8 batch from EnvHighways2D ControlNet training
batch = {
    'trajectory': tensor([B=8, 64, 4]),  # 8 trajectories
    'sdf_grid': tensor([
        # Sample 0: scale=1.0, SDF from precomputed cache
        [1, 64, 64],  # Wide corridors
        # Sample 1: scale=1.3
        [1, 64, 64],  # Narrower corridors
        # Sample 2: scale=1.5
        [1, 64, 64],  # Narrowest corridors
        # ...
    ]),  # [8, 1, 64, 64]
    'scale': tensor([1.0, 1.3, 1.5, 1.1, 1.2, 1.4, 1.0, 1.5]),  # [8]
}
```

### Dataset Class

```python
class ControlNetTrajectoryDataset(Dataset):
    """
    Dataset for training ControlNet on a specific environment type.
    
    Key design:
    - Trajectories come from .npz file (generated by MMD with guidance)
    - SDFs come from pre-computed cache (not computed on-the-fly)
    - Scale is stored per sample but SDF lookup uses scale value
    """
    
    def __init__(
        self,
        trajectory_file: str,
        sdf_cache: SDFCache,
        tensor_args: dict,
    ):
        """
        Args:
            trajectory_file: Path to .npz with trajectories + scales
            sdf_cache: Pre-loaded SDFCache with resized grids
            tensor_args: {'device': ..., 'dtype': ...}
        """
        data = np.load(trajectory_file)
        
        self.trajectories = torch.tensor(data['trajectories'], **tensor_args)  # [N, 64, 4]
        self.scales = torch.tensor(data['scales'], **tensor_args)  # [N]
        self.sdf_cache = sdf_cache
        
        print(f"Loaded {len(self)} trajectories across scales: {torch.unique(self.scales).tolist()}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]  # [64, 4]
        scale = self.scales[idx]  # scalar
        
        # Look up pre-computed, pre-resized SDF (O(1))
        sdf_grid = self.sdf_cache.get_sdf(scale.item())  # [1, 64, 64]
        
        return {
            'trajectory': trajectory,
            'sdf_grid': sdf_grid,
            'scale': scale,
        }
```

### DataLoader Configuration

```python
# Training DataLoader
train_loader = DataLoader(
    dataset=ControlNetTrajectoryDataset(
        trajectory_file="data_trajectories/EnvHighways2D/multiscale_trajs.npz",
        sdf_cache=SDFCache("EnvHighways2D", TRAINING_SCALES, "precomputed_sdfs"),
        tensor_args=tensor_args,
    ),
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

---

## Training Loop Overview

```python
def train_controlnet(
    base_model: TemporalUnet,
    controlnet: MMDControlNet,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float = 1e-4,
):
    """
    Train ControlNet adapter while keeping base model frozen.
    
    Key points:
    - Base model is FROZEN (no gradients)
    - ControlNet is TRAINABLE
    - Loss is standard diffusion L2 loss
    - SDF is passed as conditioning
    """
    # Freeze base model
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    # ControlNet is trainable
    controlnet.train()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            trajectory = batch['trajectory']  # [B, 64, 4]
            sdf_grid = batch['sdf_grid']      # [B, 1, 64, 64]
            
            # Sample diffusion timesteps
            t = torch.randint(0, num_diffusion_steps, (B,), device=device)
            
            # Add noise to trajectory
            noise = torch.randn_like(trajectory)
            x_noisy = q_sample(trajectory, t, noise)  # Noisy trajectory
            
            # Forward through controlled model
            # ControlNet processes SDF and injects residuals into base model
            noise_pred = controlled_model(
                x=x_noisy,
                time=t,
                sdf_grid=sdf_grid,
                # scale parameter is OPTIONAL (SDF already encodes it)
            )
            
            # L2 loss between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass (only ControlNet params updated)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
    
    # Save trained ControlNet
    torch.save(controlnet.state_dict(), "controlnet.pth")
```

---

## Inference Strategy

**Status**: To Be Explored

This section will be expanded after the training pipeline is validated. Key considerations:

### Open Questions for Inference

| Question | Options | Notes |
|----------|---------|-------|
| **SDF source** | Pre-computed cache vs. compute on-the-fly | Pre-computed for training scales, on-the-fly for arbitrary scales |
| **Integration with MPD** | Modify `MPD.plan()` to use ControlNet | Need to update planner to pass SDF |
| **Guidance interaction** | ControlNet only vs. ControlNet + guidance | ControlNet may reduce need for guidance |
| **Multi-agent support** | `MPDEnsemble` with ControlNet | Same pattern, shared SDF |
| **Arbitrary scales** | Interpolation between discrete scales? | Model may generalize naturally |

### Preliminary Inference Flow

```python
def inference_with_controlnet(
    env_type: str,
    scale: float,
    start: torch.Tensor,
    goal: torch.Tensor,
):
    """
    Plan trajectory using ControlNet-conditioned model.
    
    Status: Conceptual - needs implementation and testing.
    """
    # Load models
    controlled_model, sdf_grid = load_controlled_model(env_type, scale)
    
    # If scale not in pre-computed set, compute SDF on-the-fly
    if sdf_grid is None:
        env = create_environment(env_type, scale)
        sdf_grid = compute_and_resize_sdf(env)
    
    # Initialize random trajectory
    x = torch.randn([1, 64, 4], device=device)
    
    # Diffusion sampling loop (DDPM or DDIM)
    for t in reversed(range(num_steps)):
        # Get noise prediction conditioned on SDF
        noise_pred = controlled_model(x, t, sdf_grid)
        
        # Denoise step
        x = denoise_step(x, noise_pred, t)
        
        # Optional: Apply guidance for additional constraints
        # x = guidance_step(x, env, start, goal)
    
    # Condition on start/goal
    x[:, 0, :] = start
    x[:, -1, :] = goal
    
    return x  # [1, 64, 4] - Planned trajectory
```

### Expected Inference Behavior

| Scenario | Expected Outcome |
|----------|------------------|
| **Scale in training set (1.0-1.5)** | Good performance, model has seen this geometry |
| **Scale slightly outside (1.6)** | Reasonable generalization expected |
| **Scale far outside (2.0)** | May need guidance assistance |
| **New environment type** | Requires training new ControlNet adapter |

---

## Summary: What and How

### What We're Building

| Component | Purpose | Status |
|-----------|---------|--------|
| **ControlNet Adapter** | Conditions trajectory diffusion on SDF geometry | Architecture documented |
| **Per-Environment Adapters** | One adapter per pretrained model | Strategy defined |
| **Pre-computed SDFs** | Cached 64×64 grids for fast training | Pipeline documented |
| **Trajectory Dataset** | Multi-scale trajectories from guided MMD | Strategy defined |
| **Training Script** | Train ControlNet while freezing base | Pseudocode provided |
| **Inference Integration** | Use ControlNet in MPD planning | To Be Explored |

### How We're Achieving It

| Phase | Action | Files Involved |
|-------|--------|----------------|
| **1. SDF Utilities** | Add `resize_sdf()`, make scale optional in `MapSDFEncoder` | `map_encoder.py` |
| **2. Pre-compute SDFs** | Script to generate and cache SDFs for all (env, scale) pairs | `scripts/generate_data/` |
| **3. Generate Trajectories** | Use existing MMD+guidance on scaled envs | `scripts/generate_data/` |
| **4. Dataset Class** | `ControlNetTrajectoryDataset` with SDF lookup | `mmd/datasets/` |
| **5. Update ControlNet** | Integrate `MapSDFEncoder`, implement Approach B | `controlnet.py` |
| **6. Training Script** | End-to-end training with frozen base | `scripts/train_diffusion/` |
| **7. Inference** | Integrate into MPD planner | To Be Explored |

### Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scale parameter | Remove (SDF encodes it) | Simpler, no redundancy |
| SDF storage | Full 400×400, resize once | Preserve precision, fast training |
| Conditioning approach | Start with B (Global), upgrade to A | Lower risk first |
| Adapter scope | One per environment | Specialization for quality |
| Training data | Minimal (~1K-10K per scale) | Adapters need less data |
| Scale lookup | Dictionary: scale → SDF | O(1) lookup |

---

# References

- **ControlNet Paper**: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models"
- **Base TemporalUnet**: `mmd/models/diffusion_models/temporal_unet.py`
- **MapSDFEncoder**: `mmd/models/helpers/map_encoder.py`
- **SpatialTransformer**: `mmd/models/layers/layers_attention.py`
- **Current ControlNet**: `mmd/models/diffusion_models/controlnet.py`
