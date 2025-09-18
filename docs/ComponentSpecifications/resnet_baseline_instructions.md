# Implementation Guide: ResNet Baseline for the Learned Planner (Sokoban)

**Audience:** Capable AI research engineer with code-edit permissions in an existing repo.  
**Goal:** Implement the **ResNet baseline** described in the same paper as the DRC model and wire it into the existing training/eval stack as a drop‑in alternative to `DRCPolicy`. This file fully specifies the architecture as used in that work.

> You have internet access to the paper’s HTML but not PDFs. This guide specifies the ResNet exactly enough to implement without the PDF.

---

## 0) Quick spec (TL;DR)

- **Type:** **Feed‑forward ResNet**, **non‑recurrent** (no hidden state, no “thinking steps” between env actions).
- **Blocks:** **9 residual blocks** total.
  - **Channels:** Blocks **1–2** use **32** channels; Blocks **3–9** use **64** channels.
  - **Each block structure:**  
    1) a **trunk conv** (no activation before it), then  
    2) **two parallel residual branches**, each `ReLU → Conv`, and  
    3) **sum**: `out = trunk + branch1 + branch2`.
- **Convs:** Use **3×3 kernels**, **padding=1**, **bias=True** throughout. No pooling, no striding; **preserve H×W**.
- **Input:** RGB observation tensor `[B, C_in=3, H, W]` (already normalized by /255 in the env).
- **Output heads:** Flatten → `Linear(H·W·C_last → 256)` → `ReLU` →
  - **Policy head:** `Linear(256 → |A|)`
  - **Value head:** `Linear(256 → 1)`
- **No boundary channel, no pool‑and‑inject, no recurrence.**

Default hyperparams below can be exposed via config knobs for sweeps.

---

## 1) File & API surface

Add a new module alongside your DRC implementation. Keep the same public API used by trainers/evaluators.

### 1.1 Proposed files

```
learned_planner/
  resnet_policy.py     # Public entrypoint (ResNetPolicy)
  resnet_blocks.py     # BasicBlock2Branch definition and small helpers
  heads.py             # Reuse ActorCriticHead if already present
```

### 1.2 Public constructor

```python
class ResNetPolicy(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],   # (C_in, H, W), usually (3, H, W)
        num_actions: int,
        block_channels: tuple[int, int, int] = (32, 32, 64),  # (B1-2, B1-2, B3-9) semantic aid
        num_blocks: int = 9,               # fixed by spec
        kernel_size: int = 3,
        proj_dim: int = 256,
    ): ...
```

**Forward signature (match DRC policy where possible):**

```python
def forward(
    self,
    obs: torch.Tensor,                    # [B, 3, H, W]
    state: None | tuple = None,           # ignored; returns None for compatibility
    **kwargs,
) -> tuple[PolicyValue, None]:
    ...
```

Return object should carry `policy_logits: [B, |A|]` and `value: [B, 1]`. State is **always `None`**.

---

## 2) Architecture details

### 2.1 Block topology (the “two-branch” residual block)

Each residual **block** takes `x ∈ ℝ^{B×C×H×W}` and produces `y` with the **same shape**; channel count `C` is constant *within* a block and set per block index (see §2.3).

**Inside one block:**

1. **Trunk conv:** `t = Conv2d(C, C, k=3, padding=1, bias=True)(x)`
2. **Branch A:** `a = Conv2d(C, C, k=3, padding=1, bias=True)(ReLU(t))`
3. **Branch B:** `b = Conv2d(C, C, k=3, padding=1, bias=True)(ReLU(t))`
4. **Sum:** `y = t + a + b`

Notes:
- No extra ReLU after the sum (keep parity with the paper’s wording: “(relu, conv) sub‑blocks … added back to the trunk”). If desired, expose `post_act=True|False` as a toggle, default **False**.
- No BatchNorm, no LayerNorm, no pooling/strides.
- Keep spatial resolution **H×W** fixed through every block.

### 2.2 Stem and tail

- **Stem:** A **single 3×3 Conv** to map **input channels** to the first block’s channel count (32). If your Block‑1 trunk also receives 3‑channel input directly, you can fold the stem into Block‑1’s trunk. For clarity, do:
  ```python
  stem = Conv2d(C_in, 32, k=3, padding=1, bias=True)
  x = stem(obs)
  ```

- **Blocks:** 9 blocks in sequence (see channel schedule below).

- **Heads (actor–critic):**
  ```python
  flat = x.view(B, -1)                               # [B, H*W*C_last]
  hid  = ReLU( Linear(H*W*C_last, 256)(flat) )       # projection
  policy_logits = Linear(256, num_actions)(hid)
  value         = Linear(256, 1)(hid)
  ```

### 2.3 Channel schedule

Let `C1=32`, `C2=32`, `C3..C9=64`. To change the channel count **between Block 2 and 3**:

- Insert a **channel‑increasing 1×1 Conv** *before* Block 3 to map 32→64 **without** changing H×W.
  ```python
  if C_prev != C_next:
      x = Conv2d(C_prev, C_next, k=1, padding=0, bias=True)(x)
  ```
- Alternatively, make **Block‑3’s trunk conv** have `in_channels=32, out_channels=64` and add a parallel **projection** on the skip/trunk path so shapes match:
  ```python
  t = Conv2d(32, 64, k=3, padding=1, bias=True)(x)
  proj = Conv2d(32, 64, k=1, bias=True)(x)
  # Then sum branches with t, and replace 't' with 'proj' in the final sum if you use a classic residual add.
  ```
Given the wording (“trunk conv … branches added back to the trunk”), prefer the **1×1 up‑projection of the trunk path** at the boundary (simple and faithful).

### 2.4 Kernel sizes and padding

- All residual block convs: `k=3, padding=1, stride=1, bias=True`.
- Any channel‑projection convs: `k=1, padding=0, stride=1, bias=True`.
- **Never change spatial size** inside the model. Sokoban uses fixed grids; downsampling would discard crucial spatial detail.

### 2.5 Initialization

- **Kaiming‑uniform** or **Kaiming‑normal (fan_in, nonlinearity='relu')** for Conv/Linear weights.
- **Zero biases** everywhere.
- Heads can use small std normal for stability if that’s your repo norm.

---

## 3) Implementation guide

### 3.1 Code sketch

```python
import torch
from torch import nn

class BasicBlock2Branch(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.trunk = nn.Conv2d(channels, channels, k, padding=p, bias=True)
        self.branch_a = nn.Conv2d(channels, channels, k, padding=p, bias=True)
        self.branch_b = nn.Conv2d(channels, channels, k, padding=p, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        t = self.trunk(x)
        a = self.branch_a(self.act(t))
        b = self.branch_b(self.act(t))
        return t + a + b

class ResNetBackbone(nn.Module):
    def __init__(self, c_in: int, k: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(c_in, 32, k, padding=k//2, bias=True)
        blocks = []
        # Blocks 1–2 @ 32 ch
        blocks += [BasicBlock2Branch(32, k) for _ in range(2)]
        # Channel jump to 64
        blocks += [nn.Conv2d(32, 64, 1, bias=True)]
        # Blocks 3–9 @ 64 ch
        blocks += [BasicBlock2Branch(64, k) for _ in range(7)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        return self.blocks(x)

class ResNetPolicy(nn.Module):
    def __init__(self, obs_shape, num_actions, proj_dim=256, kernel_size=3):
        super().__init__()
        c_in, H, W = obs_shape
        self.backbone = ResNetBackbone(c_in, kernel_size)
        self.H, self.W = H, W
        c_last = 64
        self.proj = nn.Linear(c_last * H * W, proj_dim)
        self.act = nn.ReLU(inplace=True)
        self.pi = nn.Linear(proj_dim, num_actions)
        self.v  = nn.Linear(proj_dim, 1)

    def forward(self, obs, state=None, **kwargs):
        x = self.backbone(obs)                       # [B, 64, H, W]
        flat = x.flatten(1)                          # [B, 64*H*W]
        hid  = self.act(self.proj(flat))             # [B, 256]
        return (self.pi(hid), self.v(hid)), None
```

> Keep your repo’s actual `PolicyValue`/return types and logging hooks. The code above is a minimal faithful skeleton.

### 3.2 Tests

1. **Shape test:** random obs `[B,3,H,W]` → logits `[B,|A|]`, value `[B,1]`.
2. **Determinism:** seed and check output repeats bit‑exact for same inputs.
3. **Grad flow:** add dummy loss and ensure `backward()` runs with non‑zero grads in backbone and heads.
4. **Parity check:** Verify you can swap from DRC to ResNet via config without touching trainer code (state is `None`).

---

## 4) Integration with training/eval

- **IMPALA parity:** Learning code is identical to DRC except there is **no recurrent state** to carry. Ensure the actor loop does not try to cache or feed a state.
- **Thinking steps:** Do **not** repeat observations for “thinking”; the ResNet does all computation in its single forward pass.
- **Resets:** No state reset is needed on episode `done=True`.
- **Logging:** Plot the ResNet line as the dotted baseline in any “thinking‑steps” charts; it should be flat w.r.t. extra thinking (by design).

---

## 5) Hyperparameters & defaults

- `num_blocks = 9`
- Channels: `[32, 32, 64, 64, 64, 64, 64, 64, 64]`
- `kernel_size = 3`
- `proj_dim = 256`
- Weight init: Kaiming (fan_in)
- Optimizer / LR / losses: identical to DRC runs in the repo

---

## 6) Definition of Done (DoD)

- ✅ Swappable with DRC via config; trainer/evaluator run unchanged.
- ✅ Unit tests for shapes, determinism, grad flow pass.
- ✅ Training sanity‑check run produces non‑degenerate policy/value and learning curves comparable in scale to prior ResNet baselines.
- ✅ Code documented: blocks, heads, and config knobs.

---

## 7) Rationale vs. alternatives

- **Why 2‑branch blocks?** The paper’s baseline uses a trunk plus **two parallel (ReLU→Conv) branches** summed back in the block. This is uncommon vs. standard ResNet BasicBlocks but easy to implement and matches their baseline.  
- **Why no pooling/downsampling?** Sokoban is a small fixed grid with critical local geometry; losing resolution harms performance.
- **Why 9 blocks & channel schedule?** Matches the described baseline (first two at 32 channels, remaining at 64).

---

## 8) FAQ / likely follow‑ups

- **Q:** Can I slot BatchNorm/LayerNorm?  
  **A:** No—omit to match the baseline. You can experiment later but keep parity first.

- **Q:** Different board sizes at eval time?  
  **A:** Supported, as long as `H·W·C_last` fits your Linear(·→256). If you vary H×W, use an adaptive pooling + Linear(·→256) instead of flatten, but keep the default as **flatten** for parity.

- **Q:** Can I fuse the two 3×3 conv branches into a single 3×3 that outputs 2C and then split?  
  **A:** Yes, as an optimization (less Python overhead). Keep math identical.

---

## 9) Minimal usage example

```python
policy = ResNetPolicy((3, H, W), num_actions=4)
(out, _), _ = policy(torch.randn(32, 3, H, W))
logits, value = out  # [32, 4], [32, 1]
```
