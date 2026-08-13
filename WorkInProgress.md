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

### Phase 4: scaling data/model/budget alone is not enough at 16×16 (2026-07-10 evening)

- Scaled everything the phase-3 diagnosis asked for: 55,143-puzzle deep-labelled fixture (lengths 1–8, robots 1–3, BFS depth 8/600k nodes, parallel deterministic build), 6.1M-param `GridPlanesResNet` (stem + 6 residual blocks × 128ch), 4-stage length curriculum with a stall safety-valve (forced advance at 250k episodes/stage), 10.24M steps at a measured 716 steps/s (~4h on MPS), live W&B logging (`phase4-16x16-big-seed11`). Baselines on the harder distribution: random 4.4%, target-only 8.5%.
- En route: found and fixed an 83ms/reset bottleneck (`BankSampler._split_mask` used `iterrows()` over the 55k-row partition; vectorized with byte-identical split fingerprints; regression test `test_bank_split_perf.py`; lesson recorded in CLAUDE.md).
- Result: **held-out gate failed** — final stochastic test 3.3% vs 8.5% required; deterministic 0.8% with 95% no-movement (argmax collapse). Mid-run validation showed real learning (length-1 held-out peaked at 67% during stages 0–1), but stages advanced only via the safety-valve (windows 25%/18%/10%), and by the end the policy had regressed to ~random on the test mix while grinding on stage-2 distributions it couldn't solve.
- Diagnosis: at 16×16 with 16 actions and horizon 20, random success on length ≥3 is ~0–5%, so from-scratch PPO gets almost no reward signal past the easiest stage — the binding constraint is **exploration/credit assignment, not data, capacity, or wall-clock**. The safety-valve prevented stalling but demonstrated that advancing without mastery trades old skill for no new skill.
- Next lever (in progress): **behaviour-cloning warmstart** — every fixture puzzle carries its BFS-optimal action sequence; pretrain the policy supervised on optimal actions, then PPO fine-tune under the same benchmark/gates.

### Phase 4b RESULT: gate cleared — BC warmstart + PPO solves longer/harder 16×16 puzzles (2026-07-12 ~16:10)

- **Final held-out test: 28.8% stochastic / 24.2% deterministic — the 8.5% gate is passed at 3.4×** (from-scratch phase 4: 3.3%). By length (stoch): 100/53/47/29/20/24/12/7% for lengths 1–8 — unseen length-7 and length-8 puzzles solved for the first time in this project. By robots-moved: 53% single, 19% two-robot, 4% three-robot. Mean successful deterministic optimality gap 1.28 moves.
- Trajectory: BC 89.3% action accuracy → BC-alone rollout 30.6%/24.8% val → PPO dipped to 24.0% at 2M (transient value-fitting), recovered to 31.4% at 4.1M, peaked 33.1%/27.3% at 7.2M, ended 30.6%/24.0%.
- The curriculum advanced through **all stages via the 65% mastery threshold** — zero safety-valve firings (final stage window 63%). The phase-4 exploration pathology is gone.
- Conclusion: the working 16×16 recipe is **structured planes + residual grid CNN + BC warmstart from stored solver plans + staged-curriculum PPO**. Scale alone failed; imitation gave the prior; RL extended it to the hardest strata.
- Formal next steps: 3-seed protocol before claiming the milestone; fine-tune hyperparameters (the 2M dip and post-7.2M fade suggest lower lr/ent_coef could add several points; checkpoint selection at 7.2M would already report ~33%); then the DRC/recurrence comparison (docs/ResearchPlan.md).

### Phase 4b: BC warmstart — earlier notes (relaunched 2026-07-12 ~10:33; first run paused 2026-07-10 to rest the machine)

**State: code committed (`0fb6e2f`), run in progress. The harness saves weights only at the end of a run and has no mid-run resume, so a stop means rerunning from scratch (~5.5–6h). Full mechanism + hard-won lessons now written up in `docs/GridCnnGeneralizationReport.md` (Phase 4b + Architectures sections).**

- Implementation (committed): `Phase2Config` gains `bc_pretrain_epochs` / `bc_batch_size` / `bc_learning_rate` / `bc_max_puzzles`; `train_seed` replays each training puzzle's stored BFS plan and records a `(observation, action)` pair at **every state along the optimal trajectory** (not just the first move — ~211k pairs from 41,395 puzzles), pretrains the policy's action distribution with cross-entropy/NLL on MPS, then hands the warmstarted network to the normal staged-curriculum PPO loop. Requires `obs_mode: image`. Config: `configs/phase4b_bc_16x16.yaml` (25 BC epochs, batch 1024, lr 1e-3, then the exact phase-4 PPO setup: 10.24M steps, 4-stage curriculum, safety-valve 250k).
- **Non-obvious fix (in `_bc_pretrain`):** SB3 initializes the PPO action head at orthogonal gain 0.01, which nearly zeroes supervised gradients and pins BC accuracy at chance; `action_net` must be re-initialized at unit gain before cloning. Watch for this in any future supervised-into-SB3 work.
- **Result (2026-07-12): BC alone clears the 16×16 gate before any PPO.** BC reached 89.3% optimal-action accuracy; the step-0 checkpoint (BC weights, zero PPO) scores **30.6% stochastic / 24.8% deterministic held-out** on the full length-1–8 mix (gate 8.5%; from-scratch Phase 4 ended at 3.3%). Non-trivial at every length (stoch by len: 1→100, 2→58, 3→69, 4→33, 5→17, 6→17, 7→11, 8→6%). Confirms the exploration diagnosis decisively. PPO fine-tuning now in progress; **the critical watch is the first PPO checkpoint (1.02M) holding/beating 30.6% rather than catastrophically forgetting** — if it regresses, gentler fine-tune (lower lr / lower ent_coef / KL-to-prior).
- Evidence from the killed run (log `artifacts/phase2_experiments/phase4b_bc_16x16.log`): dataset extraction produced **210,947 pairs from 41,395 training puzzles in 19s**; BC accuracy climbed 15.0% → **65.0% by epoch 7 of 25** (random = 6.25%), loss 2.522 → 1.188, ~3 min/epoch. The supervised signal is strong — the policy can clearly learn the solver's first-move function, which is exactly the exploration prior PPO was missing.

**To resume (one command, ~5.5–6h total on this machine):**

```bash
nohup caffeinate -is .venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase4b_bc_16x16.yaml \
  --bank-dir artifacts/phase4_bank \
  --output-dir artifacts/phase2_experiments/phase4b_bc_16x16 \
  > artifacts/phase2_experiments/phase4b_bc_16x16.log 2>&1 &
```

(Delete or move the old `artifacts/phase2_experiments/phase4b_bc_16x16/` output dir first if it exists — the fixture bank `artifacts/phase4_bank` must be left untouched; its fingerprint validates on start. Expect: ~80 min BC, then ~4.5h PPO at ~630–715 steps/s, checkpoints at 1M/2M/4M/7M/10.24M, W&B run `phase4b-bc-16x16-seed11` in project `ricochet-robots-rl`.)

**What a continuing agent should watch / decide:**

- Success criterion: mean stochastic held-out test ≥ **8.5%** (target-robot-only baseline) — but the real ambition is far above it; phase 2c reached 10× baseline at 8×8.
- Early sanity: the step-0 checkpoint after BC should already beat the 4.4% random baseline on validation (BC alone should transfer). If it doesn't, BC→policy wiring is suspect.
- Watch `curriculum/window_success_rate` on W&B: with the BC prior, stages should now advance via the **65% mastery threshold**, not the safety-valve. Repeated safety-valve advancement = the phase-4 failure mode returning.
- Known risk: PPO fine-tuning can catastrophically forget the BC prior (entropy bonus + policy-gradient noise). If validation collapses after strong step-0 numbers, consider: lower learning rate for PPO (1e-4), lower ent_coef, or adding a KL-to-BC-prior penalty.
- If BC+PPO clears the gate: run the 3-seed protocol before claiming the milestone, then consider the DRC/recurrence comparison (docs/ResearchPlan.md) on top of this recipe.
- Background context every new agent should read first: `CLAUDE.md` (hard-won lessons: real-factory benchmarking, no iterrows, fingerprints are contracts, detached long jobs), `docs/GridCnnGeneralizationReport.md` (full experiment arc), and the phase-4 diagnosis above (exploration, not scale, is the 16×16 constraint).

### Performance-first RL remediation started (2026-07-17 13:29 EDT)

- Synthesized `docs/RLLearningInvestigationPlan.md` and
  `docs/Phase5ReverseCurriculumPlan.md` into
  `docs/RLLearningSynthesizedPlan.md`.
- Detailed live state, experiment commands, results, decisions, and all
  deferred work are tracked in `docs/RLLearningExecutionLog.md`; use that file
  for handoff.
- Locked priority: improve the BC→PPO feed-forward model first with explicit
  provenance, sealed evaluation, training-time movement masking, BC validation,
  and PPO stability screens. Reverse curriculum will be implemented and tested
  now but its full RL-only run is deferred.
- Baseline before edits: 88 passed, 9 intentionally skipped.

### RL remediation observability/evaluation foundation (2026-07-17 14:05 EDT)

- Added explicit policy provenance, evaluable per-checkpoint weights and
  metadata, validation-only runs, validation-selected reporting, puzzle-level
  bootstrap uncertainty, corrected W&B axes, SB3 optimizer metrics, and fixed
  training-probe activation/logit/value health.
- Sealed a fresh 375-layout confirmation suite at
  `artifacts/phase2_experiments/phase4_confirmation_manifest.json`, fingerprint
  `816d8583c7cd…`; it is disjoint from train, validation, and the exposed test
  suite by both puzzle ID and layout hash. No model has touched it.
- Gate: 92 passed, 9 intentionally skipped. Detailed handoff remains in
  `docs/RLLearningExecutionLog.md`.

### Training-time movement masking and BC stability (2026-07-17 14:12 EDT)

- Implemented end-to-end MaskablePPO movement masking, wrapper forwarding,
  mask-aware BC/evaluation, held-out BC best-epoch selection, exact shared BC
  checkpoint loading, cumulative optimizer schedules, KL guard support, and a
  reusable policy-health audit command.
- Real-factory mask overhead is 5.7% after optimizing the feasibility check;
  masked evaluation produces exactly zero blocked movements in tests.
- Historical audit machine-recorded the Phase 4 constant-logit collapse (zero
  between-state logit variance) and healthy state dependence in Phase 4b.
- Gate: 99 passed, 9 intentionally skipped. Next: reverse-curriculum machinery
  and its calibrated smoke tests; see `docs/RLLearningExecutionLog.md`.

### Calibrated reverse-curriculum smoke passed (2026-07-17 14:44 EDT)

- Implemented exact stored-plan suffix resets, deterministic weighted reverse
  bands, frontier-only mastery windows, baseline-calibrated thresholds, and
  explicit mastery-versus-forced transition telemetry.
- The 8×8 301k-step smoke passed all three stages via mastery with zero forced
  transitions. Validation improved from 10.9% deterministic / 21.2% stochastic
  at step 0 to 65.5% / 67.9%; final masked invalid/no-movement rates were zero.
- A bounded 16×16 no-BC dry run is next. The full 16×16 RL-only reverse run and
  formal RL-only/BC-only/BC→RL comparison remain explicitly deferred; details,
  commands, fingerprints, and the full deferred register are in
  `docs/RLLearningExecutionLog.md`.

### 16×16 reverse dry run complete (2026-07-17 14:58 EDT)

- The immutable Phase 4 fixture/split and baselines reproduced exactly. The
  200,704-step run sustained 662 training steps/s and emitted both planned
  checkpoints without touching the final test.
- Its d1 frontier reached 72.0%, below the calibrated 79.5% gate, so it stayed
  at level 0 with no forced transition. Masking held no-movement at zero;
  disabling masks counterfactually produced 80.7% stochastic no-movement at
  the endpoint and only 0.8% success.
- Reverse machinery is validated and full RL-only training remains deferred.
  Work now returns to the performance-first canonical BC→PPO screens.

### Canonical Phase 5 BC initialization launched (2026-07-17 15:07 EDT)

- Commit `b82f899`; detached screen session `phase5_bc` (launch-time screen PID
  28369, Python PID 28372). Config:
  `configs/phase5_canonical_bc_seed11.yaml`; log:
  `artifacts/phase2_experiments/phase5_canonical_bc_seed11.log`.
- It trains 25 mask-aware epochs, selects by held-out expert-state BC loss,
  saves the exact post-BC step-0 model, evaluates validation with five fixed
  repetitions, and does not touch the final test. Both PPO screens will load
  this same checkpoint; they must not start until its artifacts are audited.

### Canonical BC attempt 1 stopped and diagnosed (2026-07-17 15:33 EDT)

- W&B `bfh8kzng` was stopped after three epochs stayed at masked chance
  (train/validation losses ≈2.520; accuracies ≈8–9%). It produced no checkpoint
  and did not evaluate rollouts or the final test.
- Gradient and algorithm-class audits were healthy. An exact full-data audit
  changing only the supervised mask reproduced historical unmasked epoch 1
  exactly (loss 2.522, accuracy 15.0%), proving the BC-loss mask caused the
  stall. Invalid expert actions are still checked, and PPO rollout/evaluation
  stays masked; only BC cross-entropy now retains all 16 logits.
- Retry config `configs/phase5_canonical_bc_seed11_retry1.yaml` has a unique
  output/W&B identity, and both screen configs point to its future step-0 model.
  Gate: 110 passed, 9 skipped. Full evidence is in the execution log.

### Canonical BC retry 1 launched (2026-07-17 15:35 EDT)

- Commit `baecf88`; detached screen `phase5_bc_retry1` (launch-time Python PID
  31861); log `artifacts/phase2_experiments/phase5_canonical_bc_seed11_retry1.log`.
- It retains mask validation and masked rollout/evaluation but uses the
  empirically restored full-contrast BC loss. Epoch 1 should reproduce the
  exact 15.0% audit reference before the remaining epochs are allowed to run.

### Future RGB-sprite observation comparison (2026-07-17 16:35 EDT)

- Verified the original Sokoban setup in Guez et al.'s _An Investigation of
  Model-Free Planning_: 10×10 symbolic states were rendered with 8×8 RGB
  sprites into 80×80×3 images, then encoded back to a 10×10 spatial feature
  grid. This was a grid-aligned visual representation, not uncontrolled raw
  screen capture.
- Added a deferred, controlled comparison of current structured planes against
  a Sokoban-style RGB-sprite renderer. It will reuse the frozen data split,
  actions, rewards, masks, training/evaluation protocol, and budget; BC data
  must be regenerated for the new tensors. The first comparison will keep a
  feed-forward spatial encoder so RGB and DRC are not changed simultaneously.
- The old padded single-pixel `rgb_cell_image` + NatureCNN result remains a
  documented encoder mismatch, not a conclusion that RGB cannot work. Full
  rationale and resumption gates are in `docs/RLLearningExecutionLog.md`.

### BC/evaluation acceleration and canonical checkpoint (2026-07-17 17:21 EDT)

- Canonical retry completed and restored best BC epoch 12. Its step-0 masked
  validation is 30.6% deterministic / 35.5% stochastic; checkpoint SHA-256 is
  `d64eed2dcde7f3c84df0d56ece3d0d9392c92ba1939837439a402bee9f056181`.
- BC now caches validation tensors on-device and removes per-minibatch metric
  synchronizations and unused mask gathers. Evaluation runs one deterministic
  episode per fixed puzzle and batches model calls with per-episode stochastic
  RNG streams that are invariant to batch width.
- Exact-suite MPS measurement: 5.73× deterministic evaluator speedup and
  10.16× stochastic speedup at batch 64, with record-level parity. Full gate:
  114 passed, 9 intentionally skipped. Next: bounded PPO rollout-geometry
  comparison at `16×128`, `32×64`, and `64×32`.

### PPO rollout geometry retained at 16×128 (2026-07-17 17:34 EDT)

- Matched 102,400-step runs measured 622.6 steps/s (`16×128`), 603.4 (`32×64`),
  and 550.1 (`64×32`), all preserving a 2,048-transition buffer and reaching
  curriculum level 2 by mastery.
- Retain `16×128`: wider batches were slower on the serial `DummyVecEnv` and
  did not improve short-run validation. Next work package is exact resumable
  promotion, including optimizer, timesteps/schedules, RNG, and curriculum.

## Ground-up RL rebuild (branch `rl-ground-up`, 2026-07-18)

### Red-team + redesign (2026-07-18)

- Independent audit of the RL stack treating the game env as given; full
  rationale and change list in `docs/GroundUpRLRedesign.md`.
- Environment verified clean: 200/200 stored bank plans replay to the goal,
  50/50 BFS lengths re-derive, obs planes lossless, goal always an L (2 walls).
- Root cause of every from-scratch 16×16 failure quantified: masked-random
  success within horizon 20 by optimal length = 40/18/10/3/1/0.2/0.2/0.0%
  (L1–L8). Sparse terminal reward + a length curriculum that jumps over the
  cliff at L≥4 leaves PPO no learning signal. Not an environment bug.
- Two levers validated empirically before building: (a) potential-based
  shaping on the relaxed slide distance (admissible 0/300, mean d0/optimal
  0.52, 0.5 ms/layout; half-greedy walk lifts L5 success 4→33/250); (b)
  reverse starts along stored plans (masked-random success 51% at d=1).
- New pipeline in `src/groundup/`: lean fixed-layout env (canonical target-
  first identity, action masks, shaping, reverse starts, time channel),
  reverse-frontier curriculum, small residual trunk (1.4M params vs 6.1M),
  MaskablePPO, batched masked eval with baselines under the identical
  protocol. 11 correctness tests in `src/tests/test_groundup.py` including a
  differential test of the reimplemented dynamics against `ricochet_core`.

### E1 overfit gate PASSED (2026-07-18)

- 8 fixed 16×16 puzzles (lengths 1–4), uniform sampler, 150k steps, seed 11.
- 100% deterministic on all 8 by 60k steps (gate ≥95%); stochastic 100%.
- Throughput ~1,140 steps/s at 8 envs, ~1,700 at 32 envs (legacy: ~700).
- Artifacts: `artifacts/groundup/e1_overfit/`.

### E2 8×8 from-scratch PASSED, decisively (2026-07-18)

- phase2b bank (8×8, 2 robots, ET1/CL1, lengths 1–4), full new recipe, no BC:
  masked PPO + relaxed-distance shaping + reverse-frontier curriculum +
  canonical identity. Seed 11, 1M steps, 243 s wall-clock (~4k steps/s).
- Held-out val: 97.5% det / 96.8% stoch. Sealed test split (240 puzzles,
  never used for any decision): **99.6% det / 99.2% stoch; two-robot 100%**.
- Reference: legacy best on this bank was 87.0% det / 84.8% stoch at 819k
  steps with band reweighting (two-robot 68%). New recipe exceeds it on 3×
  less compute with a 2.25× smaller model (630k params).
- Baselines under identical protocol: masked-random 28.4%, target-only 42.4%.
- Curriculum frontier reached the final band (d=4) at ~150k steps; all band
  EMAs ≥0.98 at end. Artifacts: `artifacts/groundup/e2_8x8_scratch/`.

### E3 v1 stall diagnosed: Tanh head saturation (2026-07-18)

- E3 16×16 v1 (10M budget) stalled by 1.25M steps: frontier thrashing d1↔d2,
  d1 band EMA plateaued ~0.75 (masked-random is 0.51), val det 5%.
- Behavioral probe of the checkpoint: argmax matched the known optimal final
  move on only 38% of d=1 states, and **V(s) was constant (+0.3130, std
  0.000000) across 64 diverse states** — a dead value function.
- Mechanism confirmed directly: extractor features had exploded (mean 74,
  max 1139); 100% of value-head Tanh pre-activations were saturated. Cause:
  SB3 `MlpPolicy` heads default to Tanh; the legacy Phase 5 provenance hit the
  identical failure (their fix: `policy_activation: relu`).
- Fix: `activation_fn=nn.ReLU` heads + LayerNorm on the extractor output;
  regression test added (12 tests green). Lesson recorded in CLAUDE.md.
- E2 8×8 rerun with the fixed architecture (`e2_8x8_scratch_v2`) to confirm
  no regression, then E3 relaunch as v2.

### E3 v2 stall diagnosed: representational bottleneck at the readout (2026-07-18)

- With healthy heads (ReLU+LayerNorm), 16×16 v2 tripled v1's early val det
  (17.5% vs 9% at 254k) but then regressed to 10% by 1.25M with the frontier
  pinned at d=1 (band EMA ~0.7; masked-random is 0.51).
- Probes on the checkpoint: value function alive (std 0.15) but carries no
  information about start distance (means +0.27..+0.31 for d=1..5, should be
  strongly decreasing); policy puts only ~0.21 probability on the known
  optimal final move at d=1, flat in slide length (refutes the receptive-field
  hypothesis).
- Diagnosis: the flat global readout cannot efficiently learn the relational
  predicate "this slide ends on the goal" at 16×16 scale.
- Fix (v3): (a) env now offers stop-cell planes (one-hot landing cell per
  robot×direction) and a normalized relaxed-distance plane; (b) new
  `SpatialPolicy` pointer head — per-action logits read from trunk features at
  that action's landing cell (1×1 conv score maps contracted against stop
  planes), critic on pooled trunk features. No flatten anywhere; no
  location-specific parameters. 14 tests green.

### E2 v3 gate PASSED — spatial pointer head (2026-07-18)

- 8×8 with stop planes + distance plane + SpatialPolicy: 68% val det at 53k
  steps, 99% by 152k (v2 flat-head needed ~400k for 96%). Final 98.8% det /
  99.7% stoch with 314k params (half of v2). ~3,500 steps/s.
- E3 relaunched as v3 (16×16, spatial head), W&B run
  groundup-e3-16x16-scratch-v3.

### E3 v3 COMPLETE: 16×16 from scratch, 78% on the sealed test split (2026-07-18)

- Full run: 10M steps, 2.5 h wall-clock, 468k params, seed 11, no BC.
- Validation (400 puzzles): **78.3% det / 79.6% stoch**. By length (det):
  100/98/96/88/76/62/48/58% for L1–L8. By robots-moved: 90/68/60%.
- **Sealed test split (480 puzzles, never used for any decision): 78.1% det /
  80.9% stoch** — matches validation, no selection inflation. Baselines under
  the identical masked protocol: random 12.2%, target-only 23.1%.
- Reference: legacy from-scratch 3–7%; legacy BC→masked-PPO champion 39.7%
  det on a reused 121-puzzle val suite (sealed confirmation never run).
  Caveat: v3's observation is richer (stop/distance planes) — see the
  helper-removal ladder in docs/GroundUpRLRedesign.md §4.
- Curriculum frontier reached d=6 of 8; lengths 7–8 remain the gap (~50%).
  Still climbing at budget end — a longer run would likely add points.
- E4 ablation chain launched (2M steps each, sequential): no-shaping,
  no-reverse, no-planes (flat head), all else matched to v3. Reference:
  v3 curve was at ~63% @ 2.25M.

### E4 ablations complete: the spatial head + stop planes are the decisive lever (2026-07-18)

- Matched 2M-step arms, 16×16, seed 11, everything else fixed to the v3
  recipe. Validation det (v3 reference ≈63% at 2.25M from its curve):
  - no shaping:            60.3% (−3)
  - no reverse curriculum: 58.8% (−4)
  - no stop/distance planes + flat head (v2 recipe): **12.0% (−51)**
- Read: at this budget the levers are partially redundant — shaping and the
  reverse curriculum each add a few points when the planes/pointer head are
  present, but the spatially-grounded perception+readout is what makes 16×16
  learnable at all. For the helper-removal ladder (design doc §4) this
  quantifies what must be re-earned at step 2 (removing stop planes): expect
  to lean much harder on shaping, curriculum, deeper trunks, and compute.
- Experiment ladder closed: E1 overfit ✓, E2 8×8 (99.6% sealed test) ✓,
  E3 16×16 (78.1% sealed test, from scratch) ✓, E4 causal ablations ✓.
  Multi-seed confirmation of E3 remains open (single seed 11 so far).

## RGB + DRC tracks (2026-07-20)

### R1 8×8 RGB gate PASSED (2026-07-20)

- RGB Dict obs (40×40 px image at 5 px/cell + helper planes at cell
  resolution), SpatialRgbPolicy (cell-aligned patch embed → shared trunk →
  pointer head). Walls/goal/robots are perceived from pixels only.
- 1M steps, ~5 min, 331k params, ~2,950 st/s: **99.6% det / 98.6% stoch**
  held-out val — identical to the structured-planes result (99.6%).
- Example obs image saved for reference; canonical identity on (target =
  red), goal tinted with target colour. W&B: groundup-r1-8x8-rgb.

### DRC trainer landed; DRC 8×8 gate + RGB 16×16 chained (2026-07-20)

- src/groundup/drc_train.py: self-contained recurrent masked PPO for the
  faithful DRCNet (sequence replay from stored segment-initial states with
  in-replay done resets; collection==replay invariant tested; truncation
  bootstrap; masks stored/applied everywhere). 11 CPU tests green (49 total
  in the groundup suite).
- Chain launched: DRC(3,3) 8×8 gate (all helpers), then RGB 16×16 (10M).

### DRC(3,3) 8×8 gate: learnability PASSED, late-run collapse found (2026-07-20)

- All helpers on, 1M steps, ~810 st/s. val_det 0.54 @50k → 0.97 @251k →
  **1.00 @450–600k** — DRC + the recurrent trainer learn the game cleanly.
- Then a terminal post-convergence collapse between 600k and 650k (1.00 →
  0.37 → 0.04 by 850k, entropy 0.14→0.42, mean-KL guard never triggered).
  Diagnosis: near-zero advantage std at convergence amplifies outliers in
  per-minibatch advantage normalization; value-loss spikes through the shared
  ConvLSTM trunk are the co-suspect (0.45–0.50 spikes seen pre-collapse).
- Hardening delegated (keep-best checkpointing, advantage-std floor +
  normalized-advantage clamp, value clipping, max-KL telemetry); 8×8 re-gate
  queued after the RGB 16×16 run frees the GPU.
- RGB 16×16 (10M, all helpers) now training: W&B groundup-r1-16x16-rgb.

### R1 16×16 RGB COMPLETE: pixels match structured planes (2026-07-20)

- 10M steps, 3.0 h, 484k params, ~920 st/s. Validation 78.8% det / 80.3%
  stoch. **Sealed test: 77.1% det / 79.9% stoch** (planes run: 78.1% det —
  statistically indistinguishable on 480 puzzles).
- The board (walls/goal/robots) is perceived purely from 80×80 RGB pixels;
  only the helper planes (stop/distance/time) remain symbolic. Cell-aligned
  patch embedding avoids the legacy NatureCNN aliasing entirely.
- W&B: groundup-r1-16x16-rgb. Ladder implication: step 5 (planes→RGB) is
  de-risked; the remaining hard step is removing the stop planes (step 2).

### DRC(3,3) 8×8 gate v3 PASSED — stable end to end (2026-07-20)

- Fixed value-clipping bug (v2's clip-only form caused exponential value
  divergence via the truncation-bootstrap feedback loop; replaced with the
  PPO2 max-of-losses form, default off). Guards from v1's diagnosis retained
  (advantage-std floor + clamp, keep-best checkpoints, max-KL telemetry).
- v3 run: 99% val det at 251k, 100% at 600k, **stable 97–100% through 1M**
  (both prior collapse windows passed cleanly). Final report from best
  checkpoint: **99.2% det / 97.9% stoch**. 20 min wall-clock, ~830 st/s,
  998k params (faithful DRC(3,3)).
- DRC track: architecture + recurrent trainer validated at 8×8. W&B:
  groundup-drc33-8x8-gate-v3. Next candidate: DRC(3,3) 16×16 (all helpers).

### RGB renderer geometry fixes after visual review (2026-07-20)

- Sid's review of the example obs found four defects, all rooted in the
  H*k-px canvas clamping the final boundary line into the last cell row/col:
  open wall corners (central block), shrunken bottom/right-edge cells,
  off-centre undersized robot sprites, and goal tint bleeding onto the
  left grid line.
- Fix: canvas is now (H*k+1)x(W*k+1) so every boundary line owns a pixel
  row/col; wall segments span both corner pixels (junctions close); sprites
  and tint fill exactly the (k-1)^2 cell interior. Policy patch encoder is
  now kernel k+1 / stride k (each patch sees its cell plus both bounding
  lines; output grid = board grid).
- New regression test covers all four defects; 17 groundup tests green.
- NOTE: the completed R1 runs (99.6% 8×8 / 77.1% 16×16 sealed) used the old
  renderer; results stand for that obs version. A quick 8×8 RGB re-gate on
  the fixed renderer is queued after the DRC 16×16 run frees the GPU.

### Write-up checkpoint; DRC 16×16 in flight (2026-07-20)

- Consolidated results table added to docs/GroundUpRLRedesign.md §6.
- DRC(3,3) 16×16 (all helpers, 10M, 128 envs / 1 epoch after throughput
  benchmark) in progress: ~60% val det at ~4M steps, trailing the
  feedforward curve by ~6pp at matched steps so far. W&B run
  groundup-drc33-16x16 (xd0ucugn).
- Queued after it: 8×8 RGB re-gate on the fixed renderer.

### DRC(3,3) 16×16 COMPLETE: 64.8% sealed test — readout-limited (2026-07-21)

- 10M steps (wall-clock stretched to 15.7h by machine sleeps; ~2.8h awake).
  Best checkpoint 8.5M (63.5% val). Keep-best used for all final evals.
- **Sealed test: 64.8% det / 63.5% stoch** (suite of 480). By robots-moved
  (det): 89% / 39% / 41% — multi-robot coordination is the gap vs the
  feedforward pointer-head model (89/68/59, 78.1% overall).
- Learning curve ground slowly upward after an apparent plateau at ~60%
  (3.5M) to ~63% (8.5M); never approached the feedforward trajectory.
- Interpretation: DRC's paper-faithful flatten→FC readout is the binding
  constraint at 16×16 (the same bottleneck E4 measured at +51pp on the
  feedforward side). The recurrent ticks compensate substantially
  (vs the flat-head feedforward's 12% at 2M) but do not close the gap.
  Natural follow-up: DRC core + pointer readout hybrid (documented
  deviation), and/or 2-epoch training if sample efficiency is suspected.
- 8×8 RGB re-gate on the fixed renderer launched (groundup-r1-8x8-rgb-v2render).

### RGB re-gate on fixed renderer PASSED (2026-07-21)

- 8×8, fixed-geometry renderer (+1px canvas): 98.75% det / 99.7% stoch at 1M
  steps — matches the old-renderer gate (99.6%). Rendering fixes are
  performance-neutral; all future RGB runs use the corrected geometry.

### P1 launched: 12×12 plane-free learnability + landing-cell probes (2026-07-21)

- New 12×12/3-robot bank generated (ET2/CL2, lengths 1–8): 53,000 puzzles of
  60,000 attempts, fingerprint 68ea3ea6c85ed7eb; splits disjoint; 100/100
  sampled plans replay to the goal. Spot-check board sheet reviewed by Sid.
- Run: plane-free obs (walls/goal/robots/time only — no stop/distance
  planes), flat head (6 blocks, RF ±13 ≥ max ray 11), shaping + reverse
  curriculum + masking retained. 200M-step / ~1-day budget, snapshots every
  5M. W&B: groundup-p1-12x12-planefree.
- Question: how long until a conv net learns slide dynamics from RL alone at
  a scale between 8×8 (learned plane-free) and 16×16 (didn't, in 2M)?
- Instrument: linear landing-cell probes (src/groundup/probe_landing.py) on
  frozen trunk activations per checkpoint, stratified by slide length, with
  random-init control — tracks when/whether the ray-tracing circuit forms.

### P1 STOPPED at 181M by decision; full write-up (2026-07-23)

- Run: 12×12/3-robot, plane-free obs (no stop/distance planes, flat head),
  shaping + reverse curriculum + masking retained. Stopped at 181M of 200M
  (interrupted once at 95M by a machine crash; resumed via new resume_from).
- Task performance: fast climb to ~60% by 30M, then a long oscillating
  plateau (66–73%) from ~50M onward. Best val checkpoint: 73% at ~112M.
  **Sealed test at that checkpoint: 69.2% det / 68.3% stoch**; by length
  100/92/90/82/65/57/47/22% (L1–L8); by robots-moved 84/49/12%. Baselines
  under identical protocol: random 14.3%, target-only 26.0%.
- Probes (35 checkpoints, 5M cadence, linear landing-cell decodability from
  trunk activations, random-init control 14.8%):
  - blocked-move detection rose 19%→~26%; short slides (1–3) 10%→~18%;
  - **mid (4–7) never exceeded ~6%; long (8+) stayed at 2–4% for the entire
    181M steps** (chance 0.7%). No linearly-decodable ray-tracing circuit
    ever formed.
  - Trained-trunk decodability was BELOW the random-init control until
    ~35M (task-driven compression), and only marginally above after.
- Conclusions: (1) plane-free 12×12 is learnable to ~70% test with helpers,
  so scale alone isn't a hard wall; (2) task success and long-range dynamics
  decodability plateau TOGETHER — performance beyond ~70% appears gated on
  exactly the knowledge the probes show missing; (3) 181M steps of RL was not
  enough for the ray-tracing circuit to form spontaneously — supporting the
  auxiliary supervised landing-cell loss as the next lever (predict stop
  planes as an output head; labels free from the env; obs stays plane-free).
- Comparators at 12×12: none directly; 16×16 full-recipe = 78.1% test with
  stop planes vs 12% without (E4, 2M). P1 shows the without-planes ceiling
  rises a lot given 90× the budget, but stays ~9pp below the with-planes
  16×16 number even on the smaller board.
- Artifacts: artifacts/groundup/p1_12x12_planefree/ (curve, 35 probe JSONs,
  probe_curve.png, probe_summary.md, test_eval_best112M.json, snapshots).
  W&B: groundup-p1-12x12-planefree + groundup-p1_12x12_planefree-probes.

## 2026-07-25 — DRC 87M restart: adversarial investigation says the restart was bit-exactly inert

Independent investigation of whether pausing/restoring the exact-state 87M
checkpoint caused the DRC(3,3) plane-free collapse at 89M
(`artifacts/groundup/drc33_p1_12x12_planefree`, W&B `pupnnpq8`). Brief:
`docs/Drc87MRestartCollapseInvestigationBrief.md`. Report:
`docs/Drc87MRestartCollapseInvestigation.md`. Forensics:
`artifacts/forensics/drc87m_restart/`. Sealed test unopened; frozen trainer
(PID 86950) not signalled; no primary artifact touched.

- **Answer: no.** Not "probably coincidental" — measured inert. A restart split
  10+10 across two processes from the 87M checkpoint reproduces the continuous
  run bit-for-bit (1,520 field-value comparisons, 0 differences), and the loop
  is bit-deterministic on MPS (5 processes, identical iteration-1 digests).
- **Harness validated against production**: replaying the full 87M→88M window
  (122 iterations) in a fresh process reproduces `probe_model_88006656.pt`
  bit-identically and every recorded 88M metric to the last digit. So the
  production resume was exact and the resumed 87→91M trajectory *is* the
  uninterrupted counterfactual — the planned 80M Arm U/Arm R replay is moot.
- **Cause**: value-target scale runaway from ~82M, and `max_grad_norm=0.5`
  renormalises the *combined* gradient, so the exploding value term crowds the
  policy gradient out. Policy share of the update: 98.8% (80M) → 64.1% (85M) →
  **3.7% (87M)** → 0.14% (90M); `‖g_policy‖` never weakens (1.72→3.06), only its
  share. `clip_range_vf: null` left no brake.
- **New lessons worth keeping** (all in the report):
  - `evaluate_drc` calls `torch.manual_seed(0)`, which reseeds CPU **and** MPS.
    Consequence: every checkpoint stores the *same* RNG constant, and the
    exploration noise stream re-anchors every `eval_every` steps. Fix it.
  - `value_loss` is a terrible alarm here: it swings 13.7↔71.3 across 122
    consecutive iterations, and `explained_variance` stayed 0.81–0.84 while the
    loss grew 10⁴×. Use `V_mean`/`V_std` on a fixed probe set instead — `V_mean`
    had already gone *negative* (−2.66) by the pause.
  - `caffeinate -is` does not prevent clamshell sleep (machine slept
    16:37→21:08 on battery). Use `-dis`, and treat low battery as a stop, not a
    pause. Not implicated in the collapse: the 88→89M window ran awake at
    9.6 min/1M vs 9.8 pre-pause.
- **Tests**: the old exact-resume test resumed in-process, CPU-only (so the MPS
  `device_rng_state` branch was never executed), one iteration, no Adam-moment
  assertion, and same-seeded arms. Added 3 tests incl. a real `subprocess`
  restart and an MPS-gated `device_rng_state` test; fixed a pre-existing red
  assertion left by commit `96fcbff`. `pytest src/tests/test_drc_train.py -q`:
  22 passed.
- **Recovery point**: `model_best.pt` (t=79,003,648) is bit-identical to
  `probe_model_79003648.pt` and to the `best_state_dict` in all five
  checkpoints; re-evaluates to `val_det` 0.830. Rescue arms should start from
  `training_80003072.pt` (0.800, value loss 0.037, policy share 98.8%).

### Addendum — root cause of the DRC value divergence (same day)

The 87M investigation left "why did the value scale run away" open; it is now
answered, and it is a **bug in `drc_train.py`** (report §6.1).

- At horizon expiry the env returns an observation with **time channel = 0**.
  `collect_rollout` runs the value head on it and adds `0.99 × V` to the reward,
  then throws the observation away — so it is **never a regression target**, and
  `remaining == 0` appears in no stored observation (87M checkpoint `steps`
  histogram: 0…23, never 24). With `time_channel: true` the MDP is finite-horizon
  and fully observed, so `V(s, remaining=0) ≡ 0` — but nothing pins it there.
- Measured (fixed 16k probe set, time channel zeroed; `goal_reward` = 1.0):
  `V(expiry)` = −1.87 (5M), +0.35 (70M), **+0.71 (80M)**, −1.25 (85M),
  **−12.28 (87M)**, −91.9 (89M), −139.0 (91M). **Never near zero anywhere in the
  run**, and it tracks `value_loss` throughout. At 80M *timing out paid almost as
  well as solving the puzzle*; by 87M it was a 12× penalty.
- **The P1 feed-forward baseline never had this**: `src/groundup/env.py` emits no
  `TimeLimit.truncated`, and SB3 only bootstraps when both that key and
  `terminal_observation` are present. `drc_train.py:586` hand-rolled the
  bootstrap citing SB3 semantics it never received — a DRC-trainer regression,
  so P1-vs-DRC was not like-for-like on reward semantics.
- **Fix**: delete the `trunc_slots`/`trunc_obs` bootstrap in `collect_rollout`.
  Correct for `time_channel: true`, doesn't change the optimal policy. Insurance:
  Huber value loss so no value error can eat the whole `max_grad_norm` budget.
- **New telemetry for every run**: `V` at time-channel 0 (should be ~0 — the best
  and cheapest alarm), `V_mean`/`V_std` on a fixed probe set, and the
  policy-gradient share. All three led `val_det` by 5–20M steps; `value_loss` did
  not (it swings 13.7↔71.3 iteration-to-iteration).

### Truncation-bootstrap bug fixed, test-first (report §12)

- **Failing test first**: 2-move puzzles at `horizon=1` (every stored step
  truncates) with the critic pinned to a constant → pre-fix, **4.95 was injected
  into rewards of true magnitude ~0.01**, a ~600× corruption of every truncated
  step.
- **Fix**: `RecurrentPPO(bootstrap_on_truncation=None)` auto-detects from
  `EnvConfig.time_channel` — bootstrap only when the horizon is *hidden*.
  Requesting the bootstrap with `time_channel=True` now raises. Recorded in
  `provenance`/`run_history.jsonl`. **Not** a new config field, so the contract
  fingerprint `af7aae49…` is unchanged and all existing checkpoints still resume
  (87M pause checkpoint verified 16/16 post-fix).
- **It reverses the divergence, not just prevents it.** From the corrupted 87M
  state: `value_loss` 53.9 → 0.163 by iteration 10 → 0.115 at 60;
  `ret_absmax` ±38.9 → ±1.6 (the task's true scale); `approx_kl` → 0.004.
  Pre-fix the same checkpoint climbed to 336 by 89M. Policy-gradient share
  recovers 3.67% → 65.6%. So the run may be salvageable from 87M, not only 80M.
- **Watch the transient**: right after the fix the *instantaneous* value error
  rises before it falls (targets drop while V is still inflated) — iterations 1–5
  are the re-fit. `approx_kl` briefly reaches ~0.081 (> `1.5 × target_kl`), so a
  rescue from a late corrupted checkpoint can trip the early stop. Prefer 80M.
- **Similar-error audit**: `drc_train.py:600` was the only place a learned value
  entered the rewards. The SB3 paths are both correct — the groundup env emits
  neither key SB3 needs (so expiry is terminal), and the legacy env sets
  `TimeLimit.truncated` but has a clock-free observation (so its bootstrap is
  right). Both pairings are now pinned by tests, since breaking either would
  reintroduce the bug in the other trainer.
- Still open, reported not fixed: `evaluate_drc` global reseed; `_dmap_cache`
  `puzzle.seed` key; CPU↔MPS "exact resume". 204 passed, 9 skipped.

### Fix: `evaluate_drc` no longer reseeds the global RNG

Second of the three defects reported alongside the truncation-bootstrap fix.

- `evaluate_drc` called `torch.manual_seed(seed)`, which reseeds the CPU **and**
  device generators. A periodic eval runs immediately before every checkpoint, so
  this (a) made `torch_rng_state`/`device_rng_state` the same constant in every
  checkpoint of the DRC run and (b) re-anchored the training action-sampling
  stream to seed 0 every `eval_every` steps — exploration noise repeated instead
  of continuing.
- Now uses a local `torch.Generator`. `Categorical.sample()` accepts no
  `generator=`, so the stochastic branch draws explicitly via `torch.multinomial`
  on CPU, which also makes the action stream device-independent.
- Test `test_evaluate_drc_leaves_the_callers_rng_untouched` asserts both the CPU
  and (when available) MPS generator are byte-identical across an eval, and that
  results stay reproducible from the eval's own seed.
- **Consequence for future forensics:** checkpoints will no longer all carry the
  same RNG constant, and training trajectories after the first eval differ from
  pre-fix runs. Both are intended.

### Fix: `_dmap_cache` keyed on puzzle identity, not `puzzle.seed`

Third of the three reported defects (latent, not yet triggered).

- `RicochetEnv._distance_map` cached the shaping distance map under
  `puzzle.seed`, a **proxy** for identity. Correct only while every seed in the
  bank is unique (they are today, all 39,713). A collision would silently serve
  one puzzle another's distance map and corrupt the potential-based shaping
  reward with no error anywhere.
- Now keyed on the puzzle's index in the env's own table — a true unique identity
  for every puzzle the env may see, resolved via `_puzzle_index_by_id` exactly as
  `reset` already does. `self.puzzles` holds strong references, so those ids are
  pinned for the env's lifetime.
- **Rejected alternatives, both recorded because the reasoning is non-obvious:**
  `layout_hash` would be *worse* (many puzzles share a layout but differ in goal);
  keying on the map's actual determinants (`h_walls`, `v_walls`, `goal`) is
  unconditionally correct and cheap in CPU (~140 ns/call, <0.05% of a core at the
  real 3.4k calls/s) but its ~300-byte key would add on the order of **1 GB of
  RSS** across 256 envs caching thousands of puzzles each — the dmap caches are
  already a large share of the observed 2.1 GB.
- Value-neutral on the real bank, verified: a 1-iteration replay from the 87M
  checkpoint reproduces the pre-change digests bit-identically
  (`net=586ae985b23e1afa`, `opt=881ea2b955057fa1`, `value_loss=53.9287`).
- Test `test_distance_map_cache_is_keyed_on_puzzle_identity_not_seed` builds two
  puzzles that share a layout *and* a seed but have different goals, and asserts
  `dmap[goal] == 0` for each. Pre-fix the second got the first's map (`2`, not 0).

### Fix: refuse a silent cross-device "exact" resume

- `device: auto` resolves to mps or cpu per machine while the config records only
  `"auto"`, so the config-contract fingerprint is identical on an MPS box and a
  CPU-only box — a cross-device resume passed every check while being measurably
  non-exact (same 87M checkpoint: iteration-1 `value_loss` 18.1032 on cpu vs
  19.8086 on mps). (`device` itself *is* in the contract, so an explicit change
  was already caught; `auto` was the hole. Corrects an earlier note in the report.)
- `RecurrentPPO.state_dict` now records the **resolved** device;
  `load_state_dict` refuses a mismatch, with `allow_device_change=True` as an
  explicit opt-out whose error message points at `model_*.pt` for genuine
  migrations.
- Backward compatible: pre-existing checkpoints have no `device` key and load
  unchanged — verified on `training_pause_87007232.pt`, still resuming to
  `net=586ae985b23e1afa`. Safe because `load_state_dict` reads `algo_state` by
  name and only `contract` is compared for equality; the key was deliberately
  **not** put in `contract`, which would have made every existing checkpoint
  unloadable.
- 207 passed, 9 skipped.

### Hard break: pin the reward semantics in the fingerprint; drop schema-1 resume

Both compatibility compromises made while the old run's checkpoints still
mattered are now removed, since that run is being scratched.

- **`bootstrap_on_truncation` is now a `DRCTrainConfig` contract field**
  (`None` = auto-detect from `time_channel`), set explicitly to `false` in the
  registered 12x12 config. Previously it was derived in code only, which left
  runs before and after the truncation-bootstrap fix **sharing a config-contract
  fingerprint despite different reward semantics**. Pinned by
  `test_truncation_semantics_are_pinned_by_the_config_contract`.
  Contract fingerprint moves `af7aae49…` → `e7b3bf21…` (intended).
- **`algo_state` schema bumped 1 → 2, with `device` required.** Schema-1
  checkpoints are now rejected with a message pointing at `model_*.pt` for
  weights. They predate both the bootstrap fix and the recorded resolved device,
  so their optimizer/weights were shaped by inflated rewards and cannot be
  resumed exactly. `allow_device_change=True` now also prints a loud warning.
- Not changed: `_dmap_cache`'s int index key. That choice was driven by RSS
  (~1 GB across 256 envs for a content key), not compatibility, so the hard-break
  permission doesn't alter it.
- 209 passed, 9 skipped.

## 2026-07-26 — DRC plane-free v2 completed at 120M; aux-loss experiment planned

Full review: `docs/DrcPlaneFree120MReview.md`. Plan: `docs/DrcAuxLandingLossPlan.md`.

- **120,004,608 steps, 20.4 h.** Final validation (480 puzzles): **0.7667 det /
  0.7743 stoch**; baselines random 0.141, target-only 0.243. Sealed test
  **unopened**.
- **The truncation-bootstrap fix held for the whole run.** `value_loss` stayed in
  [0.076, 0.155] across all 120 eval rows; `explained_variance` rose 0.257 →
  0.677; 3 target_kl early stops out of 46,876 updates. It ran 31M steps past the
  point the corrupted attempt destroyed itself. The 96M battery pause is
  invisible in the curve.
- **Read 0.767, not the 0.810 `best_val_det`** — that is the max of ~120 draws on
  a 200-puzzle suite (SE 0.031), i.e. selection bias; the independent 480-suite
  scores those same weights at 0.767, matching the run's mean.
- **Not converged at the stop**: window means still climbing (0.743 → 0.753 →
  0.767, last-10 mean 0.776), LR schedule only 60% consumed. An extension is
  defensible on the curve alone.
- **vs P1**: 0.767 (val) vs 0.692 (P1 sealed test) — *different splits, not a
  like-for-like headline*. But the shape is informative: v2 is slightly worse at
  lengths 1–4 and much better at 6–8 (+18pp, +12pp, **+30pp at length 8**).
  That is where recurrence should help.
- **Probes: the ray-tracing circuit still does not form.** Blocked +0.586 and
  short +0.168 over random-init control, but mid **−0.033** and long **−0.063** —
  *below* control at every depth and tick, declining monotonically through
  training (long readout 0.054 → 0.004). Replicates and strengthens the P1
  finding. Same fixed 16k dataset (`e97ef31a…`) as P1 and the dead run.
- **Tension to keep in view**: v2 is far better than P1 at long puzzles while
  having *less* linearly-decodable long-slide knowledge than a random net. The
  decodability→performance link is unestablished; the aux-loss arm tests it.
- **Correction to my own earlier analysis**: I described the ConvLSTM as "fully
  saturated" from abs-max. Wrong — abs-max is a 0.4% tail. Median |c| is 0.20,
  mean tanh'(c) 0.82, only 2.7% of units effectively dead. The network is
  **plastic**. Annotated in the investigation report; it does not affect the
  causal conclusion there.
- **Aux-loss plan**: mid-training screen (A0 control + A1 aux, 120M→135M, ~2.5 h
  each) *before* any fresh 120M run, justified by the plasticity measurement. The
  control arm is mandatory because the run had not converged. Prerequisites:
  landing targets that don't change the obs, a separate optimizer for the new
  head so the main Adam state resumes exactly, and additive config-contract
  evolution. Log the policy-gradient share first — a new shared-trunk loss under
  `max_grad_norm` is exactly the failure mode we just diagnosed.

## 2026-07-26 (overnight) — aux landing-cell loss: mid-training screen

Arms A0 (control, no aux), A1 (lambda=1.0), A2 (lambda=0.3), all resuming the
identical v2 120M checkpoint `training_120004608.pt` and differing ONLY in
`aux_landing_coef`. Run concurrently so that a truncated night still leaves a
valid paired comparison at matched steps. Plan: `docs/DrcAuxLandingLossPlan.md`.

**Headline so far: the bottleneck hypothesis looks FALSIFIED, in its strong form.**
The aux loss installs the missing knowledge easily, and the policy gets *worse*.

- lambda sweep (`artifacts/forensics/aux/`): aux loss starts at **5.053** vs
  chance ln(144)=4.969 — the trunk reads at exactly chance, independently
  confirming the 120M probe result by a different route. Policy share of the
  trunk gradient: 93.3% (0.1) / 92.4% (0.3) / 87.2% (1.0) / 65.4% (3.0) /
  27.8% (10), against 98.8% healthy and 3.7% at the dying 87M. lambda=3 disrupts
  the policy (entropy 0.24→0.39); lambda=1 chosen.
- **A1 learns the aux task extremely well**: cross-entropy 5.05 → 2.02 → 1.33 →
  0.93 over 3M steps (~39% average probability on the true cell out of 144).
- **Decodability flips from below-control to clearly positive** (readout, A1 at
  122M vs A0 at 125M): mid **0.010 → 0.187** (gain −0.032 → **+0.144**), long
  0.005 → 0.073 (gain −0.062 → +0.006), short 0.240 → 0.804. So the knowledge the
  120M review found missing is installable, and quickly.
- **And val_det drops ~15pp and stays down**: 0.615 / 0.650 / 0.625 against A0's
  0.790 / 0.765 / 0.800 at matched steps, with the training-side curriculum
  frontier stuck at 4 vs A0's 7–8. No recovery over 3M steps.

Caveats being tested / to state:

1. Single lambda. **A2 (lambda=0.3) added mid-experiment** to test whether the
   disruption is strength-dependent — an adaptive addition to the pre-registered
   design, flagged as such.
2. **No head warmup**: the aux head is randomly initialised and attached at full
   lambda, so it backpropagates structured noise into the shared trunk early.
   Argues for a transient — but the drop persisted 3M steps after the head became
   accurate (aux 0.93), which argues against noise and for genuine interference.
3. `long` only reached gain +0.006, so the test is strong for mid, weaker for long.
4. A fourth outcome the plan did not pre-register: "decodability up, val_det
   DOWN". The plan anticipated up/flat. Down is a stronger falsification but also
   more consistent with capacity interference than with "knowledge is irrelevant".

### Aux-loss screen: RESULT — bottleneck hypothesis falsified (branch 2)

A2 (lambda=0.3) settles it, and corrects the premature reading I recorded above
from A1 alone.

At t=127M, A2 reaches nearly the same decodability as A1 did at 124M, at a
fraction of the policy cost (readout gain vs random-init control):

| arm | blocked | short | mid | long | val_det vs control |
|---|---|---|---|---|---|
| A0 (lambda=0) | +0.591 | +0.177 | **-0.033** | -0.062 | — |
| A2 (lambda=0.3) | +0.787 | +0.833 | **+0.279** | +0.025 | **-0.013** |
| A1 (lambda=1.0) | +0.785 | +0.858 | +0.365 | +0.031 | -0.150 |

- A2's mid-band raw accuracy rises monotonically **0.028 -> 0.322 (11.4x)** over
  6M steps, a swing of +0.31 in gain, while `val_det` stays within ~1pp of the
  control (mean -0.0125 over 8 matched evals, SD 0.0167, SE 0.0059).
- A0's decodability is **flat to +-0.001 over 14 evals / 14M steps**, so the
  120M probe result is a stable property, not drift. Strong control.
- So the missing knowledge is installable, cheaply, to a large degree — and
  performance does not improve. That is branch 2 of the pre-registered decision
  rule: **hypothesis falsified; the 20h from-scratch arm is not justified.**
- The lambda=1.0 arm adds that pushing harder does not help either: same
  knowledge, 15pp worse policy. There is no dose at which this pays.
- The cost is about gradient STRENGTH, not amount of knowledge acquired: both
  arms reached similar aux mastery (A1 0.713 at 124M, A2 0.789 at 128M) but paid
  15pp vs 1.25pp. Gentle installation is nearly free.

Corrections I made during the night, both recorded because they changed the
conclusion: (1) I called the hypothesis falsified off A1 alone when the 15pp drop
was in fact a lambda artefact — A2 shows the cost is avoidable; (2) the plan
pre-registered "decodability up + val flat/up" and A1 produced "up + down", which
I initially read as a stronger falsification when it was really a dose effect.

### Aux-loss screen COMPLETE — full result in docs/DrcAuxLandingLossResult.md

Both A0 (control) and A2 (lambda=0.3) completed the full 15M arm to
t=135,004,160. A1 (lambda=1.0) stopped early at 124,002,304 once its answer was
clear.

**Bottleneck hypothesis falsified.** A2 raised readout mid-band decodability by
**+0.588** over the control (−0.032 → **+0.555**, raw accuracy 0.028 → ~0.60, aux
cross-entropy 4.969 chance → 0.395) and performance did not improve: −0.0104 on
the 480-suite endpoint (0.56 SE) and −0.0220 over 15 matched periodic evals
(SE 0.0062). No systematic long-puzzle gain, which was the central prediction.

**The control was load-bearing.** A0 gained **+2.7pp from plain training alone**
(0.7708 → 0.7979; best periodic 0.81 → 0.84). Without it, A2's 0.7875 against the
120M baseline would have read as the aux loss *helping by +1.7pp* when it is 1pp
*behind* plain training. Exactly the false positive the plan warned about.

**Dose response**: A2 ended with MORE knowledge installed than A1 (+0.555 vs
+0.365) at ~1/12 the cost (−0.01 vs −0.150). So the 15pp collapse at lambda=1.0
is an optimisation transient from pushing a big auxiliary gradient through a
shared trunk, not the price of the knowledge.

**Harness trap found (§8 of the result doc, worth fixing):** a resumed branch that
never beats its parent's `best_val_det` inherits `best_state_dict` via `run_state`
and, with `report_eval_weights=best`, silently reports the PARENT's weights as its
own final result. A2 wrote no `model_best.pt` at all and its report.json quoted
v2's 118M number. Taken at face value that gave a spurious A2−A0 gap of −3.1pp,
3x the real one. Fixed in analysis by evaluating each arm's own model_final.pt.

**Retires** linear landing-cell decodability as a progress metric: it can be
driven near ceiling without touching task performance.

**Next**: extend the control to 200M (still improving, LR schedule 68% consumed);
do NOT run the from-scratch aux arm; fix the keep-best inheritance trap before
further branch arms.

## 2026-08-13 — priorities 0/1/2 complete: honest 200M DRC closeout and causal carry result

Full report: `docs/Drc200MCarryMechanismResult.md`. Registered observational
and causal protocols/results: `docs/DrcCarryStateStudy.md` and
`docs/DrcCarryInterventionStudy.md`.

- **Reporting fixed first** (`9ccf245`, `84d91f0`): headline evaluation now
  always reloads immutable final weights; periodic best is separately named;
  every score carries source run/timestep/path/kind/SHA and suite identity;
  semantic branches cannot inherit a parent's best silently; the hard suite is
  genuinely joint-stratified by length x robots moved. The exact 190M resume
  had no contract differences and legitimately reconstructed the 135M best.
- **Exact continuation complete:** `training_pause_190005248.pt` resumed to
  the legal boundary at **200,007,680**. Immutable final source SHA
  `8760ba70...`. Final validation: **82.08% det / 82.29% stoch** on 480;
  joint-hard validation **75.96%** on 470. The separately evaluated 135M
  periodic best was worse on all three (79.79% / 80.83% / 74.89%), confirming
  the registered final-only rule was the right closeout.
- **DRC-line test evaluated once under a prewritten decision:** only the 200M
  final checkpoint, test seed 17, 60 puzzles/length, 480 total, 1 det + 3 stoch.
  Suite fingerprint `9cf385d2...`; scientific payload SHA `85d351c2...`.
  Result: **80.42% det / 81.25% stoch**, versus frozen feed-forward P1
  **69.17% / 68.33%**: **+11.25/+12.92pp**. Hard-tail det gains are +15pp L5,
  +18.33pp L7, +30pp L8, +17.26pp two-robot, and +29.42pp on the small n=17
  three-robot cell. One seed and unequal trainer/compute budgets: recipe
  success, not architecture-only attribution. The suite was held out from DRC
  training/selection but was not globally untouched: exact reproduction of
  P1's scripted-baseline aggregates strongly evidences the same selection was
  used for P1, whose strata informed the DRC programme; legacy P1 predates
  suite fingerprints.
- **Real-rollout carry probes** (`d78aa83`, `97e606a`) ran at 9M, 59M, 135M,
  and 200M on the same 157-puzzle/20-cell validation suite with exact BFS
  labels, a leak-free 118/39 layout split, independent target probes, and
  architecture-matched random-network plus shuffled-label controls. Carry has
  strong action/history code and learned nonlinear distance/optimal/reducing
  action information. It does not reveal a clean explicit plan: final best
  distance R2 .229, optimal/reducing top-1 .418/.427, landing-cell accuracy
  only ~.10-.13, and future-success decoding is non-monotonic.
- **Causal carry study** (`e6c85b9`, `e3f91c6`) settles the main question.
  Resetting all carry every move costs **32--38pp** at all four checkpoints;
  final effect **-38.22pp**, paired 95% CI **[-46.50,-29.94]**. Deeper depths
  and both `h/c` matter descriptively. One-shot move-1/2/4/8 resets are tiny,
  consistent with reconstruction, redundancy, or local bypass but not
  distinguishing them. At 200M, exact-state coherent donor carry
  changes actions in **39/68** matched contexts but has **0.00pp** mean
  suffix-success effect, CI [-7.35,+7.35]: histories change decisions, but no
  directional mean effect was detected at this precision. Donor controls pass.
- **Conclusion:** the carry is a causally necessary, distributed,
  history-dependent dynamical policy state that is resilient to isolated
  zeroing; it is not presently a readable symbolic route or slide simulator.
- **Next:** do not extend seed 11 blindly beyond its registered 200M/floor-LR
  plateau. Prioritize (1) two-seed confirmation under the repaired final-only
  protocol, (2) a preregistered reset-recovery-timescale experiment, and (3)
  outcome-blind, prefix/probe-stratified exact-state patching. The RGB
  entry gate is met; start with a short correctness/throughput/learnability
  gate, not a full long run.
