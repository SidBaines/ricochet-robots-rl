# Work in progress
Contains a description, detailed specifications, discussion, notes, questions and answers, and any progress notes etc. all related to the current task. You should update this file whenever you have made changes or progress on a task, and list next step and what remains.

## Current task
We've implemented stage 2 of the development process, and we're in the process of checking it. At the moment, we're focussing on getting the training code to work properly.
In src/scripts/bank_agent_rollout_cells.py, I want to make two changes:
- In plot_episode_interactive, I want to add a third graphic (alongside the board rendering and reward plot) which shows the value that the model assigns to the given state.
- I want to be able to save an episode as a gif. Please add add in a new cell at the bottom a funtion which will take level index and episode index, and then create and save both plotly and matplotlib style gifs. 

## Progress
- Episode recorder now captures per-state value estimates so the Plotly viewer has the data it needs.
- Interactive episode plot shows board, cumulative reward, and value estimate with synced slider controls, including a step counter overlay.
- Added `save_episode_gifs` helper that writes looping matplotlib and Plotly GIFs with step labels and a configurable end-of-loop pause (Plotly export requires kaleido).
- Tweaked GIF exporting to work with recent Pillow/Plotly versions (Pillow 10+ `textbbox`, dropped deprecated Plotly `animation_opts`).

## Next steps
- Verify Plotly GIF export once kaleido is available, or document installation if missing.
- Decide on preferred pause duration/overlay styling after reviewing generated examples.
