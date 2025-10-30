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
