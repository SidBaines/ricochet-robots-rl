# Implementation Guide: Deep Repeating ConvLSTM (DRC) for the Learned Planner

**Audience:** Capable AI research engineer with code-edit permissions in an existing repo.  
**Goal:** Replace/upgrade the current `convlstm` implementation to match the Deep Repeating ConvLSTM (DRC) architecture used in *Planning in a recurrent neural network that plays Sokoban* (Taufeeque et al., 2024/2025) and compatible with the training/eval stack already in the repo.

> You have internet access and can read:
> - The HTML version of the 2024/2025 paper (arXiv:2407.15421).  
> - The reference implementation file path: `https://github.com/AlignmentResearch/learned-planner/blob/main/learned_planner/convlstm.py`.  
> You **cannot** read PDFs. This document therefore **fully specifies** the DRC module you should implement, including equations, wiring, shapes, and integration points.

---

## 0) Quick specification (TL;DR)

- **Backbone:** A stack of **L ConvLSTM layers** (default **L=3**) with shared spatial resolution and channel count **C** (default **C=32**).  
- **Repeating (thinking) ticks:** For **each environment step**, the ConvLSTM stack is unrolled for **R ticks** (default **R=3**). Hidden/cell states carry across ticks and across environment steps.  
- **Inputs per tick & per layer:** Concatenate (channel-wise) the following to produce each layer’s input:
  1. **Encoded observation** `E_t` (same for all layers at a given tick).  
  2. **Pool-and-inject features** from the **previous tick’s hidden state** of the **same layer**: channel-wise **linear combination** of **mean-pooled** and **max-pooled** maps, broadcast to H×W.  
  3. **Boundary channel** `B` (1 channel): ones on outer border, zeros elsewhere (not zero-padded in convs to preserve size semantics).  
  4. **(For layers > 0)** the **previous layer’s hidden state** `h^{ℓ-1}` **at the current tick**.  
- **Recurrent wiring:** Standard ConvLSTM dynamics **per layer** with a **non-standard output gate nonlinearity**: use **tanh** for the output gate activation (per Jozefowicz et al., 2015)—details in §2.2.  
- **Encoder:** Two **plain convs (no nonlinearity)** applied to the raw observation to produce `E_t`. Use padding to preserve spatial size.  
- **Heads:** After tick R at a given env step, take the **final top-layer hidden state** `h^{L}` (H×W×C), **flatten** (H⋅W⋅C → 256 via linear+ReLU), then two separate linear heads: **policy logits** (|A|) and **value** (scalar).  
- **Kernel sizes:** Use **3×3** conv kernels for all ConvLSTM gates and for the encoder by default.  
- **Init & numerical details:**  
  - **Forget gate bias += 1.0**.  
  - **Zero init** for other biases; truncated normal or Kaiming/lecun-like weight init is acceptable (match existing repo defaults if present).  
  - **Do not normalize advantages** in IMPALA step if you touch training.  
- **Shapes:** Maintain constant spatial size **H×W** across the stack; all layers use **C channels** for hidden and cell states.  
- **Device/Framework:** PyTorch, matching the repo.

---

## 1) File & API surface

Update or replace the repo’s `convlstm` module to export a **single DRC policy module** with a minimal, stable API used by training/eval code.

### 1.1 Proposed file layout

```
learned_planner/
  convlstm.py        # Public entrypoint (DRCPolicy)
  drc_core.py        # ConvLSTMCell, DRCStack, PoolAndInject helpers
  encoders.py        # SimpleConvEncoder (2 convs, no nonlinearity)
  heads.py           # ActorCriticHead (flatten -> 256 -> policy,value)
  utils_spatial.py   # boundary_channel(H,W), broadcast*, shape checks
```

> If the repo expects a different structure, adapt names but keep the separation of concerns.

### 1.2 Public constructor (example)

```python
class DRCPolicy(nn.Module):
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],   # (C_in, H, W)
        num_actions: int,
        hidden_channels: int = 32,
        num_layers: int = 3,               # L
        repeats_per_step: int = 3,         # R
        kernel_size: int = 3,
        pool_and_inject: bool = True,
        use_boundary_channel: bool = True,
        proj_dim: int = 256,
    ): ...
```

**Forward signature:**

```python
def forward(
    self,
    obs: torch.Tensor,  # [B, C_in, H, W]
    state: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]], # (h_list, c_list), each list len=L, tensors [B, C, H, W]
    repeats: Optional[int] = None          # override default R; use for “thinking steps” (same obs repeated)
) -> Tuple[PolicyValue, NewState]
```

Where `PolicyValue` carries `policy_logits: [B, |A|]` and `value: [B, 1]`.  
`NewState` carries updated `(h_list, c_list)` after the final repeat tick.

> The training loop is responsible for calling `forward` with the same `obs` multiple times to realize thinking steps. Your `forward` should support both `R` repeats (default) and an override `repeats` argument so callers can vary thinking at eval time.

---

## 2) Core architecture details

### 2.1 Encoder (two convs, no nonlinearity)

- **Input:** raw observation `x_t ∈ ℝ^{B×C_in×H×W}` (RGB tiles or similar).  
- **Block:**  
  - `conv1`: C_in → C_e (use C_e = hidden_channels by default) with **kernel 3, padding 1**, bias=True, **no activation**.  
  - `conv2`: C_e → C_e with **kernel 3, padding 1**, bias=True, **no activation**.  
- **Output:** `E_t ∈ ℝ^{B×C_e×H×W}`.  
- Notes: The paper’s encoder is “two convolutional layers (without nonlinearity)”. Keep shapes H×W unchanged.

### 2.2 ConvLSTM Cell (per-layer, per-tick)

Let layer index be `ℓ ∈ {1..L}`. At **tick k of env step t** you compute:

**Inputs to layer ℓ:**
- `E_t` (from encoder).  
- `PI_{t,k,ℓ}` (pool-and-inject features from **previous tick** `k-1` of the **same layer**; for `k=1` use the previous env step’s last tick or zeros if uninitialized).  
- `B` boundary channel `[B×1×H×W]` (constant per resolution; see §2.3).  
- If `ℓ>1`: the **current tick** hidden from previous layer `h_{t,k}^{ℓ-1}`.  

Concatenate channel-wise to form `X_{t,k}^{ℓ}`. Shape: `[B, C_in_layer, H, W]` where `C_in_layer = C_e + C_pi + (ℓ>1)*C + (use_boundary)*1`.

**ConvLSTM gates:** (all convs are **3×3**, padding=1, bias=True; **share no parameters across layers or ticks**)

Standard LSTM equations except **output gate uses tanh nonlinearity**:

```
i = sigmoid( W_i * X + U_i * h_prev + b_i )
f = sigmoid( W_f * X + U_f * h_prev + b_f )
g = tanh(    W_g * X + U_g * h_prev + b_g )        # candidate
o = tanh(    W_o * X + U_o * h_prev + b_o )        # <— nonstandard
c = f ⊙ c_prev + i ⊙ g
h = o ⊙ tanh(c)
```

- `h_prev, c_prev` are from **the same layer ℓ** at the **previous tick** (or previous env step).  
- Initialize `c_prev, h_prev` to zeros on first call.  
- **Forget gate bias**: add **+1.0** at init (i.e., set bias_f += 1.0).

**Why tanh for `o`?** The DRC variant reported uses tanh on the output gate (Jozefowicz et al., 2015 exploration). Keep this exact variant for fidelity.

### 2.3 Pool-and-Inject (PI)

- Given the **previous tick’s hidden** `h_prev ∈ ℝ^{B×C×H×W}`, compute:
  - `m = adaptive_avg_pool2d(h_prev, output_size=1) → ℝ^{B×C×1×1}`  
  - `M = adaptive_max_pool2d(h_prev, output_size=1) → ℝ^{B×C×1×1}`  
  - Concatenate along channels: `[B×(2C)×1×1]`.  
  - Pass through a **1×1 linear** (conv1x1) `A: 2C → C_pi` (default **C_pi = C**, so PI has same channels as hidden).  
  - **Broadcast** back to H×W: `PI = A([m;M])` repeated over spatial dims to `[B×C_pi×H×W]`.  
- This gives fast, global communication across the grid between ticks, matching “pool-and-inject”.

### 2.4 Boundary channel

- Create `B ∈ ℝ^{1×H×W}`: `B[y,x]=1` if `y∈{0,H-1} or x∈{0,W-1}`, else 0.  
- **Important:** When you apply padding inside ConvLSTM gates, treat `B` as **already encoding** the edge condition; do **not** zero-pad `B` differently. Just concatenate `B` as-is (a constant feature plane).

### 2.5 Layer stacking and tick unrolling

For each **tick k=1..R**:
- For `ℓ=1..L`:
  - Build `X_{t,k}^{ℓ}` by concatenating `E_t`, `PI_{t,k,ℓ}`, `B`, and (if ℓ>1) `h_{t,k}^{ℓ-1}`.
  - Run **ConvLSTMCellℓ** with `(X, (h_prevℓ, c_prevℓ))` → `(hℓ, cℓ)`.
- After finishing `ℓ=L`, you have `h_{t,k}^{L}`.  
- **Feedback wiring:** At the **next tick (k+1)**, each layer uses its **own** `h_{t,k}^{ℓ}` as `h_prevℓ`. The “last layer feeding back into first” is implemented naturally via (a) hidden state persistence across ticks and (b) `h^{ℓ-1}` input; do **not** explicitly feed `h^{L}` into `ℓ=1` **within the same tick**—instead, the top-layer influence propagates across ticks via the recurrent state and the pool‑and‑inject path.

After **R ticks**, pass `h_{t,R}^{L}` to the heads.

> **Thinking steps:** Repeating the same observation multiple times is achieved by calling `forward` with the same `obs` and setting `repeats` accordingly. The environment executes an action **after** the final tick’s policy is sampled/applied.

---

## 3) Heads (actor–critic)

- **Input:** `h_top = h_{t,R}^{L} ∈ ℝ^{B×C×H×W}`.  
- **Flatten:** reshape to `[B, C·H·W]`.  
- **Projection:** `Linear(C·H·W → 256)`, `ReLU`.  
- **Policy head:** `Linear(256 → |A|)` produces logits.  
- **Value head:** `Linear(256 → 1)` produces scalar value.

Keep heads simple and free of normalization. If the repo uses separate modules for heads, re-use them if compatible.

---

## 4) Shapes, padding, and dtype/device

- **All convs** should preserve **H×W** via `padding=kernel_size//2`.  
- Use the repo’s dtype/device conventions (e.g., autocast/mixed precision if enabled).  
- Provide a utility to **initialize** hidden/cell states given batch size and spatial dims:
  ```python
  def init_state(self, batch, H, W, device=None, dtype=None):
      h = [torch.zeros(batch, C, H, W, device=device, dtype=dtype) for _ in range(L)]
      c = [torch.zeros(batch, C, H, W, device=device, dtype=dtype) for _ in range(L)]
      return (h, c)
  ```
- Ensure **state detaches** between sequences as needed by the training loop (respect current repo practice).

---

## 5) Hyperparameters & sensible defaults

- `L=3` layers, `R=3` repeats per env step.  
- Hidden/Cell channels `C=32`.  
- Encoder channels `C_e=C`.  
- Kernel size `k=3`.  
- Pool-and-inject output channels `C_pi=C`.  
- Projection dim `=256`.  
- Forget‑gate bias += 1.0 at init.

Expose these via the policy config so experiments can sweep them.

---

## 6) Implementation checklist

1. **ConvLSTMCell** with **tanh output gate** variant (§2.2), modularized and unit-tested:
   - Forward pass returns `(h, c)`.
   - Parameter shapes: W_*: Conv2d(in_ch, C, k=3), U_*: Conv2d(C, C, k=3). (Alternatively fuse as a single Conv2d over concatenated `[X, h_prev]` producing `4C` channels—more efficient.)
   - Init forget gate bias += 1.0.

2. **Pool-and-inject** (§2.3):
   - Adaptive avg & max → concat → 1×1 conv to `C_pi` → broadcast to H×W.
   - Unit-test broadcasting and shape compatibility.

3. **Boundary channel** (§2.4): utility to cache per `(H,W)` on device; verify no shape drift through padding.

4. **Encoder** (§2.1): two convs, no activation, C_in→C→C, k=3, pad=1.

5. **DRCStack** (L layers, R repeats) (§2.5):
   - Concatenate inputs in right order: `[E_t, PI, (ℓ>1)*h_{ℓ-1}, B]`.
   - Maintain lists of states `(h_list, c_list)` across ticks.
   - Support `repeats` override in `forward`.

6. **Heads** (§3):
   - Flatten → 256 → (policy, value).  
   - Check logits shape `[B, |A|]` and value `[B,1]`.

7. **API glue** (§1.2):
   - `init_state`, `forward`, and convenience methods for thinking steps.  
   - Ensure compatibility with existing training/eval runners.

8. **Initialization & numerical** (§0):
   - Respect existing global inits; otherwise, use Kaiming-normal fan_in for conv weights; small std for linear layers; zero biases except forget gate.  
   - Optional: orthogonal init for heads as in the paper replication.

9. **Tests**:
   - Shape propagation test (random obs, random state).  
   - Determinism (fixed seeds).  
   - Gradient flow (loss.backward() completes).  
   - Quick overfit on tiny toy (e.g., predict a fixed dummy label) to validate learning.

---

## 7) Integration with the training loop

- **Thinking steps:** The trainer should be able to call `policy(obs, state, repeats=K)` with `K≥1` (default R). For **extra thinking**, call with higher `K` without changing env state (repeat the same `obs`).  
- **Rollouts:** Ensure the IMPALA/V-trace collector caches the hidden/cell states **per actor** and forwards them across time.  
- **Reset:** On episode reset (env `done=True`), zero the states for the affected batch indices.  
- **Device parity:** Ensure consistency with JAX loading utilities if used (the repo may have a JAX→Torch converter for weights; keep parameter naming predictable).

---

## 8) Reference equations (concise)

Let `⊛` denote 2D convolution. For the **fused** implementation, compute:

```
Z = Conv2d([X, h_prev], out_channels=4C, k=3, pad=1)
(ī, f̄, ḡ, ō) = chunk(Z, 4, dim=channel)

i = sigmoid(ī)
f = sigmoid(f̄)
g = tanh(ḡ)
o = tanh(ō)                  # nonstandard gate nonlinearity

c = f ⊙ c_prev + i ⊙ g
h = o ⊙ tanh(c)
```

Where `X = concat( E_t, PI(h_prev), B, (ℓ>1)*h_lower )`.

---

## 9) Edge cases & gotchas

- **Do not** add ReLU or BN in the encoder (paper says two convs **without nonlinearity**).  
- **Boundary channel** must not be destroyed by padding semantics—just concatenate a constant map and let conv padding be symmetric zeros.  
- **Hidden/state shapes** must be identical for all layers; ensure `E_t` and `h` have the same H×W. If the input obs is not H×W, add a pre-resize in the data pipeline **outside** the model.  
- **Parameter count** will be close but not identical to reported values if channel counts differ—favor fidelity to wiring over matching params exactly.  
- **Output gate tanh** is easy to miss; do not accidentally keep sigmoid there.  
- **Pool-and-inject linear mix** should be **learned** (1×1 conv), not a fixed average of mean/max.

---

## 10) Minimal usage example (pseudo-code)

```python
B, C_in, H, W = 32, 3, 10, 10
policy = DRCPolicy((C_in, H, W), num_actions=4, hidden_channels=32, num_layers=3, repeats_per_step=3)

state = policy.init_state(B, H, W, device=next(policy.parameters()).device)

# Single env step with 3 “thinking” ticks:
obs = torch.randn(B, C_in, H, W)
out, state = policy(obs, state)  # repeats defaults to 3
logits, value = out.policy_logits, out.value
```

---

## 11) Migration plan from the current crude `convlstm`

1. **Read current code** and list divergences vs. this spec: gate nonlinearity, missing PI, missing boundary channel, encoder activations, layer/tick wiring.  
2. **Refactor** into `ConvLSTMCell`, `DRCStack`, `DRCPolicy` without changing external trainer code—add small adapters if needed.  
3. **Add unit tests** for: (a) pool-and-inject, (b) boundary channel, (c) 3×3 fused gates, (d) repeats override.  
4. **Bench** a short IMPALA run to verify: loss decreases; extra thinking improves success rate on held-out puzzles.  
5. **Document** the policy config knobs so future sweeps (e.g., L,R,C) are trivial.

---

## 12) Open questions you may ask (if needed)

- Do we need to load weights from the JAX-trained model? If yes, confirm parameter naming & shapes for a converter.  
- Should we support variable grid sizes at eval (generalization to larger boards)? If yes, avoid hard-coding H,W and ensure PI & boundary-channel generation are dynamic.  
- Do we want mixed precision and torch.compile for speed? (Safe if training remains stable.)

---

## 13) Provenance & rationale (non-PDF sources)

- Taufeeque et al., 2024/2025 (HTML arXiv) describe the **DRC with L ConvLSTM layers repeated R times**, **pool‑and‑inject**, **boundary channel**, **two‑conv encoder (no nonlinearity)**, and **MLP 256→heads**.  
- The inspiration is the **Deep Repeating ConvLSTM** (Guez et al., 2019). We hard-code the exact architectural quirks called out in the HTML paper to avoid any PDF-only ambiguities.

---

## 14) Definition of Done (DoD)

- ✅ Unit tests pass; shapes stable across ticks and layers.  
- ✅ Extra thinking (`repeats>R`) measurably changes logits/value; no runtime errors.  
- ✅ Training/eval scripts run unmodified (or with trivial config changes).  
- ✅ Code is clear, documented, and matches the spec above.

---

## 15) Appendix: Practical tips

- Implement ConvLSTM gates with a **single fused conv** for speed.  
- Cache the **boundary channel tensor** per `(H,W,device)` in the module to avoid reallocation.  
- If you see exploding/vanishing, confirm forget-bias init and remove accidental activations in encoder.  
- For reproducibility, seed PyTorch and disable CuDNN nondeterminism if exact bitwise reproducibility is required.

