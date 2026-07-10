# Work in progress
Contains a description, detailed specifications, discussion, notes, questions and answers, and any progress notes etc. all related to the current task. You should update this file whenever you have made changes or progress on a task, and list next step and what remains.

## Current task
We've implemented stage 2 of the development process, and we're in the process of checking it. At the moment, we're focussing on getting the training code to work properly. 
For the next task, I want to implement a new renderig mode of the board, and integrate it with the training pipeline so that the relevant model types can ingest it. 
I want the rendering to be as follows:
- An RGB image, of some fixed image size, which will have to be bigger than 2 times the board width/height (board size).
- The board has a single pixel for each board space and a single pixel for each boundary between board spaces
  - Note that this means that the image will follow an 'alternating' pattern in both rows and columns, where one row/col will correspond to possible board positions, then the next will correspond to cell boundaries.
  - It also means that, in a 'cell boundary' row, every other square will be useless (as it will correspond to a corner).
- There is a fixed color mapping for each possible arrangement:
  - Squares with robots in are the usual colour
  - Goal square for a given robot colour is a lighter version of that colour
  - Boundary cells with no wall present are eg. light grey (similar to the current RGB)
  - Boundary cells with a wall present are a dark grey
- The 'usable' board is in the centre of the image array, surrounded on all sides by black pixels
  - Exact centering might not be possible depending on parity of board size/image size

The rendering process itself should follow the previously implemented strategy of caching as much as possible (e.g. the surrounding pixels, default light gray spaces between board squares, anything else which seems sensible to cache) and then just overwriting the other stuff (robot position, wall positions) to speed up the rendering process as much as possible.

You are to implement this in the relevant files (both the rendering code itself and adding it to the training scripts/config), and then write a small script which will: load up a curriculum, plot some levels from the curriculum using this rendering, and save them in a temporary folder, just so I can inspect the output.

If you're at all uncertain, please ask questions or ask for clarification of specific points here before starting the implementation.

### Plan: Alternating-boundary RGB rendering (no code yet)

- **Goal**: Add a new RGB rendering style where each board cell and each inter-cell boundary is a single pixel, arranged in an alternating grid, centered on a fixed-size canvas with black padding. Integrate as an observation option for training and provide a quick preview script.

#### Implementation outline
- **Env support**:
  - Extend `obs_mode="rgb_image"` to support a new style via `render_rgb_config.style: "classic" | "cellgrid"` (default remains "classic").
  - Implement a cell-grid renderer that maps a H×W board to a content region of size (2H+1)×(2W+1):
    - Cell pixels at even-even indices; horizontal boundary pixels at odd-even; vertical boundary pixels at even-odd; corner intersections at odd-odd.
    - Place this content in the center of a fixed `FIXED_PIXEL_SIZE` canvas with black padding on all sides (respecting parity, choose top/left bias on ties).
  - Colors (overwritable by config):
    - Cells with robots: robot color (as today).
    - Goal cell for target robot: lighter of the robot color.
    - Boundary pixels: light gray if no wall; dark gray if wall present (outer borders are walls ⇒ dark gray).
    - Corner pixels (intersection points): neutral (question below).
  - Caching strategy:
    - Cache static background of size (2H+1)×(2W+1) with default boundary light-gray and outer borders dark-gray, keyed by (H, W, style, colors).
    - Overlay interior walls (dark-gray) once per layout as an "overlay" cache similar to current `_rgb_internal_walls_overlay`.
    - Overwrite dynamic elements (robots, goal) per frame.
  - Observation shape stays `(FIXED_PIXEL_SIZE, FIXED_PIXEL_SIZE, 3)` (or channels-first if requested). No resizing of the content; only centering with black padding.

- **Training integration**:
  - Keep `args.obs_mode == "rgb_image"`; pass `render_rgb_config.style="cellgrid"` from YAML (`env:` section) through to the env ctor.
  - Continue using `normalize_images=True` for rgb images (already wired in `train_agent.py`).
  - Add optional color/style knobs in config for reproducibility.

- **Preview script**:
  - Add `src/scripts/bank_curriculum_preview.py` (or reuse if present):
    - Load bank + curriculum config.
    - Sample N boards across a few levels.
    - Render frames using the cellgrid style and save PNGs to a temp/output dir.

#### Open questions / clarifications
- **Corner pixels**: Should corner intersections be black, light gray, or match neighboring boundary default? (Spec says "useless"; propose black.)
- **Robot overlap on boundaries**: Robots sit on cell pixels only (even-even); confirm no need to visually mark paths on boundaries.
- **Fixed canvas size**: Keep `FIXED_PIXEL_SIZE=128`? Any preference for another size (e.g., 96) for throughput? Currently `rgb_image` is keyed off this constant.
- **Color palette**: Keep existing robot colors and define explicit `light_gray=(200,200,200)`, `dark_gray=(40,40,40)` defaults? Any preferences?
- **Channel order**: Default channels-first for training (`channels_first=True` in curriculum wrappers). Okay to keep consistent?

#### Risks / notes
- The content region is sparse at fixed 128×128 for small boards; CNNs may underutilize space. We can later add an integer scale factor (e.g., 2×) if needed, but the initial implementation will use 1 pixel per cell/boundary per spec.
- Ensure caches include color/style in keys to avoid visual mismatches across runs.

#### Next steps
1. Implement `rgb_cell_image` as a new `obs_mode` with its own config.
2. Plumb `cell_obs_pixel_size` via CLI/YAML → wrappers → env.
3. Add preview to `src/scripts/bank_curriculum_preview.py` that saves PNGs.
4. Sanity-check shapes, centering, and color correctness; attach sample outputs.


## User feedback on current plan

### General comments

This seems pretty good. However, I have a couple of notes:
1) Prefer just making a new obs_mode entirely (rather than having it as a subset of rgb_image), called eg. `rgb_cell_image`
2) Yes, good find regarding the `src/scripts/bank_curriculum_preview.py` script. We can just add a little cell to this, previewing in the new mode.

### Answers to open questions:

- **Corner pixels**: Should corner intersections be black, light gray, or match neighboring boundary default? (Spec says "useless"; propose black.)
  - Make them match the boundary default (light gray, exact same shade)
- **Robot overlap on boundaries**: Robots sit on cell pixels only (even-even); confirm no need to visually mark paths on boundaries.
  - Yes, robots sit on cell pixels only and we expect the model to learn that they can only land on these so no need to mark paths
- **Fixed canvas size**: Keep `FIXED_PIXEL_SIZE=128`? Any preference for another size (e.g., 96) for throughput? Currently `rgb_image` is keyed off this constant.
  - Maybe we could add this as an optional pass in argument when the new `rgb_cell_image` mode is used, with default to 128. Make this separate from the `FIXED_PIXEL_SIZE` parameter currently used for `rgb_image` (since, as discussed above, we are treating these as two separate obs_modes)
- **Color palette**: Keep existing robot colors and define explicit `light_gray=(200,200,200)`, `dark_gray=(40,40,40)` defaults? Any preferences?
  - Lets define a new colour mapping (because, as mentioned, this will be treated as a new obs_mode) but make the robot colours the same.
- **Channel order**: Default channels-first for training (`channels_first=True` in curriculum wrappers). Okay to keep consistent?
  - Yes sounds good, consistency with previous code is good!

Please go ahead and start implementing.
Remember to document your progress regularly and thoroughly in this document.
Remember that you can stop at any point if you need to check things or ask for clarificaiton. 

### Config review: configs/resnet_cellrgb2.yaml

- Observations:
  - env.obs_mode is `rgb_cell_image` with `cell_obs_pixel_size: 40`. For 8×8 boards, content is 17×17, leaving large black padding. Not fatal, but wastes model capacity. Consider 32–36.
  - curriculum.enabled is `false` while `use_bank: true` is set. If the trainer expects the structured flag to flip curriculum, it may be ignored → you might be training on `v1` instead of the bank’s level 0. Verify that curriculum actually runs.
  - device is `cpu`. This limits throughput; not a direct cause of flat success per step, but slows iteration.
  - ent_coef is `0.05` (quite high). This can keep the policy too random and cap success around ~25% early.
  - model.type is `resnet`, but code may require an explicit `resnet: true` flag to select the ResNet policy; otherwise it falls back to `CnnPolicy`. Ensure the policy actually switches.

- Likely causes of slow learning:
  1) Excessive exploration (`ent_coef=0.05`) maintaining high entropy and suppressing exploitation.
  2) Curriculum possibly disabled (mismatch between `enabled: false` and `use_bank: true`), training on `v1` or on broader difficulty than intended.
  3) Input size/padding inefficiency (40×40 canvas with 17×17 content) lowering SNR for gradients.
  4) ResNet not actually active if the parser doesn’t map `model.type: resnet` to the `--resnet` flag.

- Recommendations (minimal set, in order):
  - Reduce entropy: set `ent_coef: 0.005` (or 0.01) and monitor.
  - Enable curriculum explicitly: set `curriculum.enabled: true` and confirm the training script picks it up (or add top-level `use_bank_curriculum: true`).
  - Confirm policy: ensure ResNet path is used (e.g., add `resnet: true` or align to training parser expectations).
  - Tidy input: try `cell_obs_pixel_size: 32` or `36` to reduce padding; keep channels_first consistent.
  - Throughput: switch `device: auto` (GPU) and consider `n_envs: 16` for more stable gradients.

## Progress log

- Implemented new `obs_mode` `rgb_cell_image` in `src/env/ricochet_env.py`:
  - Added fixed-canvas cell-grid observation generator with caching.
  - Corner pixels set to light gray; outer boundaries dark gray per spec.
  - Separate `cell_obs_pixel_size` (default 128) independent of classic `rgb_image`.
- Plumbed `cell_obs_pixel_size` through wrappers and training:
  - `src/env/criteria_env.py` and `src/env/curriculum.py` accept and forward `cell_obs_pixel_size`.
  - `src/scripts/train_agent.py` parser/env factories updated; normalize_images now applies to `rgb_cell_image` too.
- Extended `src/scripts/bank_curriculum_preview.py`:
  - Added a section to preview `rgb_cell_image` observations and save PNGs.

### Next
- Run a quick local preview to validate centering/colors and drop sample outputs.
- If colors or sizes need tweaks, expose in YAML (`env:`) under a `render_cell_config` block.


## User feedback on current progress
### General comments
Looks great! I ran and tested the new env, it seems really good. I'm noticing that the training is not proceeding quite as quickly as I'd like it to, so I'm now tweaking training params/setup to see what I can do.
Please can you go and look at the file `configs/resnet_cellrgb2.yaml` and tell me why you think it might not be learning very quickly? For reference, the success rate on the first level of the curriculum (1-2 step solves involving only a single robot moving) is pretty flat from the get-go, remaining around 25% (for solving within 10 steps) even after ~500K training steps. 
The models seem to learn on the trivial environments (`v0` and `v1`) which were set up just to ensure that the training pipeline is working.
Can you see anything obvious which might be hindering learning?

As always, remember to document your progress regularly and thoroughly in this document, and remember that you can stop at any point if you need to check things or ask for clarificaiton. 

## User feedback on current progress
Okay, thanks for the review. I've fixed a couple of these things, but it doesn't seem to be making a difference. I think it would be really helpful, in order to debug this, to have the metric logging working fully.
At the moment, I don't see any of the PPO metrics from stable baselines 3 logged to wandb, though some work was attempted at onne point to make this happen - I don't know why they're not being logged.
Similarly, there was code written at one point to log images/gifs/videos of example rollouts, but these do not appear in either wandb or tensorflow.

Please investigate and fix these, and let me know where I should see the (wandb or tensorboard, and in which tab/area of each)

Also, when you make notes on your progress, please append to the end of this file, so that this is kept as a chronological record of our work.

### Monitoring investigation and fixes

- Found MonitoringHub/backends wired in `src/scripts/train_agent.py`; SB3 metrics are forwarded via a custom writer to the hub, then to backends.
- Issue: configs often had both `monitoring.tensorboard: false` and `monitoring.wandb: false`, so hub existed with no backends → nothing visible. I added a guard: when no backends are enabled, the hub is disabled with a clear console message, avoiding silent "logs to nowhere".
- Rollout media: periodic videos/images come from `_HubTrajectoryLogger` (controlled by `eval.traj_record_freq`, default 2000); they are sent to the hub as `eval/trajectories/video0` (or `.../frame0`).

How to enable and where to look:
- To log to wandb (recommended): in your YAML under `monitoring:` set `wandb: true`, and set `wandb_project`, `wandb_entity` (and optionally `wandb_run_name`). Metrics appear under the Charts tab with keys like `train/*`, `rollout/*`, `time/*`, `eval/*`. Videos/images appear in the Media tab (Videos/Images) as `eval/trajectories/*`.
- To log to TensorBoard: set `tensorboard: true` in `monitoring:`. Launch TB pointing to `monitoring.log_dir` (default `artifacts/runs/ppo`). Metrics appear under Scalars; videos under the Videos plugin; images under Images.
- Ensure `eval.traj_record_freq > 0` to emit trajectory media periodically (default 2000). This uses a lightweight matplotlib renderer if env RGB is unavailable.

### Phase 2 reproducible benchmark (2026-07-10)

- Audited the local MPS path and moved the benchmark into an isolated Python 3.11.15 environment. Fixed the incompatible Python 3.11 NumPy pin, passed the configured PPO seed through fresh training, corrected asynchronous accelerator timing, and added a real MPS train/save/reload test.
- Added `configs/phase2_benchmark.yaml` and `src/scripts/run_phase2_benchmark.py`. The runner deterministically builds a BFS-labelled easy-puzzle fixture, freezes disjoint layout-hash train/validation/test suites, records exact fingerprints and runtime provenance, evaluates random/target-only/oracle controls, and runs three stock-CNN PPO seeds.
- Added fixed-budget training and validation curves, optimal-length/robots-moved strata, deterministic and stochastic results, Wilson intervals, action/no-movement diagnostics, throughput, an isolated evaluation RNG, and tiny-overfit and held-out acceptance gates.
- Clean MPS run at commit `1523008`: the eight-puzzle overfit gate passed at 82.5% stochastic success, but broad held-out PPO averaged 31.7%, below the 46.7% target-robot-only baseline. Deterministic PPO averaged 3.1% and collapsed toward no-movement actions.
- Full findings and reproduction instructions are in `docs/Phase2BenchmarkReport.md`. Phase 2 measurement infrastructure is complete; the held-out agent milestone remains failed, so this baseline should not yet be called successful or advanced unchanged into a larger curriculum.

### Held-out generalization achieved: planes obs + GridPlanesCnn (2026-07-10, branch `sid-learnable-agent`)

- Diagnosis of the failed stock-CNN baseline: SB3's NatureCNN (8×8-kernel, stride-4 first conv, needs ≥36×36 inputs) was being fed the 40×40 `rgb_cell_image` canvas, whose actual content is a 17×17 patch with single-pixel walls/robots centered in black padding. The stride-4 convolution aliases away exactly the single-cell geometry the task depends on, and the whole board is crushed to a 1×1×64 tensor before the linear head. The environment already had a lossless alternative: `obs_mode="image"` structured planes (4 wall channels + goal + target + one channel per robot) at native 8×8 resolution.
- Added `src/models/grid_cnn.py` (`GridPlanesCnn`): stride-1, same-padded 3×3 CNN (default 4 layers × 64 channels), flatten + linear projection — no pooling, so exact robot/goal/wall positions stay recoverable. ~1.26M policy parameters.
- Extended `Phase2Config` with `obs_mode`/`policy`/`features_dim`/`conv_channels`/`n_conv_layers`, factored a shared `_build_model`, and validated policy↔obs-mode combinations. Added contract tests in `src/tests/test_grid_cnn.py`.
- Experiment 1 (`configs/phase2_exp1_grid_cnn.yaml`, same fixture/split/baselines as the benchmark, seed 11, 204,800 steps, MPS ~2.4k steps/s):
  - Tiny-set overfit gate: 100% deterministic / 99.4% stochastic at 25,600 steps (stock CNN: 37.5%/82.5% at 51,200).
  - Held-out test: **62.5% deterministic / 66.9% stochastic** vs random 29.5% and target-robot-only 46.7%. Held-out gate **passed** (66.9% ≥ 46.7%). Deterministic no-movement rate fell from 91% (stock) to 54%; mean successful optimality gap 0.25 moves.
  - Validation curve rose monotonically after 25k steps (9.4% → 71.9% deterministic), i.e. genuine distribution learning, not memorization.
- Conclusion: the earlier failure was an observation-encoding × architecture mismatch, not a task/reward/PPO problem. Sparse reward, 51k budget, and the bank pipeline were all fine once the network could see the board.
- Experiment 2 (`configs/phase2_exp2_grid_cnn_3seed.yaml`, same recipe, full three-seed protocol): held-out test success **69.8% ± 6.5% deterministic / 75.0% ± 7.8% stochastic** across seeds 11/29/47 (per-seed stochastic 66.9%/82.5%/75.6%; pooled Wilson 70.9%–78.7%). Every seed individually clears the 46.7% target-robot-only gate; the formal held-out milestone is **passed** (`artifacts/phase2_experiments/exp2_grid_cnn_3seed/`).

### Phase 2b: harder fixture also cleared (2026-07-10)

- New deterministic fixture (`configs/phase2b_grid_cnn.yaml`, bank `artifacts/phase2b_bank`): 8×8, two robots, interior walls (1 edge-T + 1 central-L per quadrant), optimal lengths 1–4, robots_moved 1–2 — 8,922 puzzles across 7 strata (1,166 two-robot-plan puzzles), layout-hash split 6,736/1,096/1,090. Split-manifest strata checks now derive from the fixture's achievable strata instead of the Cartesian product (which would wrongly require impossible combos like length-1/robots-2).
- Baselines collapse on this distribution: random 14.1%, target-robot-only 16.6% (0% on two-robot puzzles) — real headroom for demonstrating skill.
- Overfit gate (28 puzzles): 89.3% stochastic — passed.
- Seed 11, 409,600 steps: held-out test **68.5% deterministic / 70.7% stochastic** (gate passed, 70.7% vs 16.6%). By stratum: single-robot plans essentially solved (**99–100%** at every length); two-robot blocker puzzles **23–30%** (vs 0–8% baselines). Validation still rising at the final checkpoint (47.3% → 63.6% deterministic between 205k and 410k).
- Next lever (phase 2c, running): two-robot bands were only ~13% of uniform training exposure; explicit band weights now lift them to ~41%, budget doubled to 819,200 steps, overfit gate skipped (already proven). Curriculum manager honours explicit `bands` on a level spec; `Phase2Config` gains `two_robot_band_weight` and `run_overfit_milestone`.

### Phase 2c result: two-robot weakness largely fixed (2026-07-10)

- `configs/phase2c_grid_cnn_2robot.yaml` (same phase2b fixture/split; two-robot exposure 13%→41%; 819,200 steps, seed 11): held-out test **87.0% deterministic / 84.8% stochastic** (gate 16.6%). By stratum: single-robot **100%/97%**, two-robot blocker puzzles **68%/67%** (phase2b: 23%/30%; heuristic baselines: 0–8%). Mean successful optimality gap 0.66 moves; deterministic no-movement rate 28.9% (was 91% for the broken stock baseline).
- At equal budget (409,600 steps) the reweighted run reached 76.4% deterministic validation vs 63.6% unweighted — the distribution shift, not just the longer budget, drives the gain. Validation was still improving at 819k.
- Next: scale the recipe to the real game footprint (16×16, 4 robots, 16 actions) as phase 3.

### Phase 3 (16×16, 4 robots): no-curriculum control fails; staged curriculum launched (2026-07-10)

- New fixture (`configs/phase3_16x16_grid_cnn.yaml`, `artifacts/phase3_bank`): 16×16, 4 robots, 2 edge-T + 2 central-L walls per quadrant, lengths 1–6, robots-moved 1–2 — 6,588 puzzles over 11 strata, split 4,933/830/825. `build_fixture_bank` gained a deterministic parallel mode (verified byte-identical fingerprints; 6× faster) after a 15-minute sequential build was killed.
- Baselines on this distribution: random 6.3%, target-robot-only 12.3%, oracle 100%.
- Control run (same grid_cnn recipe, 6 conv layers, weighted bands, 1.64M steps, no curriculum): **flat** — validation ≤3.1% throughout, final held-out test 1.6% det / 5.1% stoch. With ~41% of episodes being two-robot length-5/6 puzzles the policy never stumbles on reward; a pure exploration failure, not capacity (identical recipe reached 87% on 8×8).
- Response: staged length curriculum through the existing `BankCurriculumManager` (`curriculum_stage_max_lengths: [2, 4, 6]`, advance at 70% train success over a 500-episode window after ≥5,000 episodes per stage). `grid_cnn` also exposed as `model.type` in `train_agent.py`.
- Staged result (`configs/phase3b_16x16_staged.yaml`, 1.64M steps): the curriculum mechanism works — stage 0 (lengths 1–2) was mastered on the training distribution (70% window) and advanced to stage 1 at ~400k steps — but the run ended still at stage 1 (window 25%), and held-out test was **3.2% det / 7.0% stoch** (gate 12.3%: failed). Transfer exists only at the easiest stratum (length-1 held-out 53% stochastic); lengths 3+ are 0%. Training-suite gains did not transfer, i.e. the model memorized the ~600 stage-0 training layouts rather than generalizing.
- Interpretation: 16×16/4-robot is data/scale-starved, not mechanism-broken. Next steps, in order: (1) 10–50× fixture size via the parallel builder (10k attempts ≈ 3 min at 8 workers); (2) wider/deeper residual grid CNN; (3) 10M+ step budget with per-stage generalization eval; (4) behaviour-cloning warmstart from the bank's stored optimal plans; (5) only then multi-seed formal gating. All 8×8 milestones remain green and reproducible.
