# Faithful DRC Reimplementation

Code: `src/groundup/drc.py` (`ConvLSTMCell`, `DRCNet`, `DRCState`).
Tests: `src/tests/test_drc_faithful.py` (CPU-only, ~1s).

This is a from-primary-sources reimplementation of the DRC (Deep Repeated
ConvLSTM) architecture from DeepMind's Sokoban model-free-planning paper. It
deliberately ignores the earlier `src/models/convlstm.py` /
`src/models/drc_policy.py` line of work in this repo.

## Sources

| Tag | Source |
|---|---|
| [G19] | Guez et al., *An Investigation of Model-Free Planning*, ICML 2019. arXiv:1901.03559 (read via https://ar5iv.labs.arxiv.org/html/1901.03559). Architecture: §2.1.1–2.1.2; ConvLSTM equations: App. 8.2.1; pool-and-inject: App. 8.1.2; Sokoban hyperparameters: App. 9.1–9.2. |
| [T24] | Taufeeque et al., *Planning in a recurrent neural network that plays Sokoban*, 2024. arXiv:2407.15421 (https://arxiv.org/html/2407.15421v2), appendix A. Faithful DRC(3,3) retraining used for mechanistic interpretability. |
| [TLP] | https://github.com/AlignmentResearch/train-learned-planner — the training code for [T24]. Ground truth files consulted directly: `cleanba/convlstm.py` (cell + stack), `cleanba/network.py` (heads, init, preprocessing), `cleanba/config.py::sokoban_drc33_59` (the faithful config). |
| [BKY] | Config of the actual trained-and-interpreted DRC(3,3) checkpoint (run `drc33/bkynosqi`), from `learned-planner/plot/learning_curves/drc_33/bkynosqi.cfg.json` in https://github.com/AlignmentResearch/learned-planner. This is the arbiter wherever [TLP] exposes multiple options. |

## Architecture (with citations)

One environment step of DRC(D, N):

1. **Encoder** — a small CNN embeds the observation once per step into a
   spatial map `i_t` ([G19] §2.1.1). For 80×80×3 pixel Sokoban: channels
   (32, 32), kernels (8, 4), strides (4, 2), SAME padding → 10×10×32
   ([G19] App. 9.1). [T24]/[BKY] instead use 10×10×3 tile observations with
   channels (32, 32), kernels (4, 4), strides (1, 1), **no** inter-layer
   ReLU → the same 10×10×32.
2. **ConvLSTM stack, N ticks** — D ConvLSTM modules are applied N times per
   step. "The encoded observation i_t is fed as an input not only at all
   ticks 1…N, but also at all depths 1…D" ([G19] App. 8). Module d at tick n
   receives `i_t`, `h_{d-1}^n` (module below, current tick), and its own
   `(h_d^{n-1}, c_d^{n-1})`. For d=1 the "below" input is the **top-down skip
   connection** `h_D^{n-1}` ([G19] App. 8.2.1; [TLP] `apply_cells_once` reads
   `carry[-1].h` before updating any layer).
3. **Cell** ([G19] App. 8.2.1, App. 9.1; [TLP] `ConvLSTMCell.__call__`):
   - All gate pre-activations come from **one fused convolution path with
     4×32 = 128 output channels**, kernel 3, stride 1, SAME padding ("all
     ConvLSTM modules use one convolution with 128 output channels, a kernel
     size of 3 and stride of 1. All cell states and hidden states have 32
     channels", [G19] App. 9.1). Concretely ([TLP]):
     `gates = Conv_ih([i_t, h_below, pool_inject]) + Conv_hh(h_own_prev) + Conv_fence(fence)`
     with bias only on `Conv_ih`.
   - Chunked `(g, i, f, o)` (candidate, input, forget, output — [TLP] calls
     these `i, j, f, o`): `c' = σ(f + forget_bias)⊙c + σ(i)⊙tanh(g)`;
     `h' = act_o(o)⊙tanh(c')`.
   - **Pool-and-inject** ([G19] App. 8.1.2): `m_d = [max_{H,W}(h_d), mean_{H,W}(h_d)]`,
     `p_d = Tile_{H,W}(W_p · m_d)`; `p_d^{n-1}` is appended to module d's
     input at tick n. Pools the module's **own** previous hidden state
     ([G19] notation; [TLP] `pool_and_inject="horizontal"`).
   - **Boundary "fence" feature** ([G19] §2.1.2): "we append a feature map to
     the input of the convolutional operators that has ones on the boundary
     and zeros inside."
4. **Readout** ([G19] §2.1.1 + App. 9.1; [T24] App. A): output is `h_D` after
   the last tick; "the encoded input i_t is also combined with o_t through a
   skip-connection" — elementwise **addition** in [TLP] (`skip_final`). The
   sum is flattened → Linear(256) → ReLU → two separate linear heads for
   policy logits and scalar value.

Everything above is implemented exactly, in `ConvLSTMCell` and
`DRCNet.forward`. Recurrent state is explicit: `DRCNet.initial_state(B)`
creates zeros ([TLP] `initialize_carry`), `forward(obs, state)` returns the
new state, `DRCNet.reset_state(state, done)` zeroes finished rows ([TLP]
zeroes the carry where `episode_starts`).

## Hyperparameters (paper Sokoban DRC(3,3) = constructor defaults)

| Hyperparameter | Value | Source |
|---|---|---|
| D (depth) / N (ticks) | 3 / 3 (DRC(1,1): 1 / 1) | [G19] §3, App. 9.1 |
| hidden/cell channels | 32 | [G19] App. 9.1 |
| fused gate conv | 128 out-channels (=4×32), kernel 3, stride 1, SAME | [G19] App. 9.1 |
| encoder (pixel, 80×80×3) | ch (32,32), k (8,4), s (4,2), SAME, bias | [G19] App. 9.1 |
| encoder (tiles, 10×10×3) — **default** | ch (32,32), k (4,4), s (1,1), SAME, bias, no ReLU | [T24] App. A, [BKY] |
| encoder → h_D skip | elementwise add | [G19] §2.1.1 (“combined”); [TLP] `skip_final=True`, [BKY] |
| readout MLP | flatten → 256 → ReLU | [G19] App. 9.1 (“single hidden layer MLP with 256 units”) |
| heads | Linear(256→A) and Linear(256→1) | [TLP] `Actor`/`Critic`; [T24] App. A |
| pool-and-inject | max+mean, per-channel projection, tiled | [G19] App. 8.1.2; [BKY] `pool_projection="per-channel"` |
| fence | ring of ones outside the grid, VALID conv, no bias | [G19] §2.1.2; [BKY] `fence_pad="valid"`; [T24] (“a 12×12 channel … not zero-padded”) |
| output gate activation | tanh (non-standard) | [BKY] `output_activation="tanh"`; [T24] App. A. **[G19] App. 8.2.1 writes σ** — see Ambiguities |
| forget-gate bias | 0.0 | [BKY] `forget_bias=0` |
| weight init | truncated normal (±2σ), std √(1/fan_in) (flax `lecun_normal`); heads orthogonal gain 1; biases 0 | [TLP]/[T24]; **not specified in [G19]** |
| input preprocessing | RGB / 255, outside the network | [TLP] `Policy._maybe_normalize_input_image` (not part of the core; not implemented here) |
| prev action / reward inputs | none | absent from [G19] and [TLP] |
| Sokoban actions | 4 (up/down/left/right) | [T24]; not explicit in the [G19] text we could extract |
| (training, for reference only) | IMPALA/V-trace; lr 4e-4 annealed, Adam, batch 32, BPTT unroll 20; logit L2 and weight L2 regularizers | [G19] App. 9.2; [TLP] config |

### Parameter count cross-check (pinned in tests)

`sokoban_drc33()` (obs 3×10×10, 4 actions, defaults) has **1,285,125**
parameters, matching [T24]'s reported 1.29M: encoder 1,568 + 16,416; per cell
110,720 (`W_ih`: 96→128 3×3 + bias) + 36,864 (`W_hh`: 32→128 3×3) + 1,152
(fence: 1→128 3×3) + 64 (pool projection 2×32) = 148,800, ×3 cells; MLP
819,456; actor 1,028; critic 257. [G19] Table 5's 2,042,376 for DRC(3,3) does
not match its own architecture description; [T24] confirmed with the original
authors that the paper's parameter counts contain typographical errors.

## Ambiguities in [G19] and how they were resolved

1. **Output-gate activation.** [G19] App. 8.2.1 writes `o = σ(…)`, but the
   faithful retraining that reproduces the paper's results ([BKY], described
   in [T24] as "a nonstandard tanh on the output gate", citing Jozefowicz et
   al. 2015) uses `tanh`. Default here: `tanh`, with
   `output_activation="sigmoid"` available for the literal paper equation.
   This is the one place where our default knowingly differs from the paper's
   printed equation (see Deviations).
2. **Fence placement.** "Ones on the boundary and zeros inside" appended "to
   the input of the convolutional operators" ([G19] §2.1.2) could mean ones on
   the outermost grid cells, or ones in the ring where zero-padding would be.
   [BKY]/[T24] use the latter: a (H+2)×(W+2) map convolved VALID (no bias).
   Default `fence="valid"`; `fence="same"` implements the other reading.
3. **Pool-projection shape.** [G19] App. 8.1.2's `W_p · [max, mean]^T` does
   not fix `W_p`'s shape. [TLP] offers `full` (dense 2C→C) and `per-channel`
   (a learned (w_max, w_mean) pair per channel); [BKY] used `per-channel`
   (default here; `pool_projection="full"` available).
4. **Encoder nonlinearity.** [G19] App. 9 lists ReLU as the default
   activation, but the faithful retraining's encoder has **no** ReLU between
   its two convolutions ([T24]: "two convolutional layers (without
   nonlinearity)"; [BKY] `use_relu=false`). Default `encoder_relu=False`;
   when True, ReLU is applied between encoder convs but never after the last
   one ([TLP] semantics).
5. **"Combined … through a skip-connection".** Addition vs concatenation is
   not stated in [G19]; [TLP] adds (`skip_final=True`), which also explains
   why the encoder output width must equal the hidden width (32).
6. **Bias placement in the fused gate conv.** [G19] writes one bias per gate;
   [TLP] puts the bias on the input convolution only (`W_hh` and fence conv
   bias-free), which is equivalent (biases would sum) and is what we do.
7. **Weight initialization** is unspecified in [G19]; we follow [TLP]/[T24]
   (lecun truncated normal everywhere; orthogonal gain-1 heads; zero biases;
   forget bias 0).
8. **Gate chunk order** is mathematically arbitrary; we fix it to [TLP]'s
   `(candidate, input, forget, output)` so weights are portable to/from their
   released checkpoints (modulo NHWC↔NCHW and flatten-order transforms).

## Known deviations from [G19]

1. **Output gate defaults to `tanh`, not the paper's printed `σ`** —
   deliberately following the verified faithful retraining [BKY]/[T24].
   `output_activation="sigmoid"` restores the printed equation. (No other
   knowing deviations from the paper's specification exist.)
2. **Default encoder geometry is the [T24] tile-observation variant**
   (4×4/stride-1, no ReLU) rather than the paper's 80×80-pixel 8×8-stride-4
   encoder, because this repo's observations are small symbolic grids (and
   per repo policy, structured planes must not pass through striding
   layers). The paper's pixel encoder is one constructor call away
   (`encoder_kernels=(8, 4), encoder_strides=(4, 2), encoder_relu=True`) and
   is shape-tested.
3. **Env adaptations, off by default / outside the core:** optional
   `action_mask` argument on `forward` (masked logits set to −1e9);
   observation preprocessing (e.g. /255 for RGB) is left to the caller, as in
   [TLP]. `observation_shape` and `n_actions` are required constructor
   arguments (our env: 16×16 boards, 16 actions).

## Open questions

- Whether DeepMind's original TF implementation used the `tanh` output gate
  (making [G19]'s σ a typo) or [T24] introduced it; [T24]'s correspondence
  with the original authors about parameter-count typos suggests insider
  knowledge, and their retrained network reproduces the paper's Sokoban
  results, but the original code is unpublished.
- [T24]'s appendix says the LSTM forget gate "has 1 added to it", yet the
  released config of their trained network ([BKY]) sets `forget_bias=0`. We
  default to 0 and expose `forget_bias`.
- The exact Sokoban action count in [G19] (4 vs 5 with no-op) is not stated
  in the text we could extract; [T24] uses 4.
- [G19] Table 5 parameter counts (2.04M/1.75M) cannot be reconciled with the
  described architecture (acknowledged typographical errors, per [T24]).
