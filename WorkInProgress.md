# Work in progress
Contains a description, detailed specifications, discussion, notes, questions and answers, and any progress notes etc. all related to the current task. You should update this file whenever you have made changes or progress on a task, and list next step and what remains.

## Current task
We've implemented stage 2 of the development process, and we're in the process of checking it. At the moment, we're focussing on getting the training code to work properly. 
I want you to do two things:
1) Add way more metric logging (including the usual RL metrics, for example KL, entropy, clip fraction, etc.). Please think about and decide what to log, then implement it.
2) Rearrange the yaml config file so that the parameters come in blocks which are relevant (eg. resnet parameters indented under a heading which says 'resnet-params' or something, same with convlstm, drc, etc., then eval parameters indented in an eval section)

### Notes on metrics logging plan (implemented)
- Added forwarding of SB3 scalar logs to the monitoring hub for keys under `train/`, `rollout/`, and `time/`.
- This captures: `train/approx_kl`, `train/clip_fraction`, `train/entropy_loss`, `train/policy_gradient_loss`, `train/value_loss`, `train/explained_variance`, `train/learning_rate`, plus rollout means and FPS.
- Existing custom collectors remain (episode success window, action/no-op stats, curriculum events) and continue to log under `train/` and `curriculum/`.

### Notes on config restructuring (implemented)
- Config now supports a structured layout with sections: `env`, `algo`, `model`, `curriculum`, `eval`, `monitoring`, and existing `initialization`.
- A loader now flattens these sections into argparse-compatible keys for backward compatibility; old flat configs still work.
- Default config (`configs/train_defaults.yaml`) has been rewritten to the new sectioned layout.
