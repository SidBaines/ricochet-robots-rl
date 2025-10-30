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
