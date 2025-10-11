# Work in progress
Contains a description, detailed specifications, discussion, notes, questions and answers, and any progress notes etc. all related to the current task. You should update this file whenever you have made changes or progress on a task, and list next step and what remains.

## Current task
We've implemented stage 2 of the development process, and we're in the process of checking it. At the moment, we're focussing on getting the training code to work properly.
Look at the train_defaults.yaml, work out which bits of code are being executed, and help me understand why it's not improving the training performance (or, if it is having an effect, it's taking a very long time). 
It might be something to do with the chosen training hyperparamters, or it might be that the resnet is not set up correctly.
It's worth noting that the model does learn to solve level 0 of the curriculum with the small-cnn flag turned on, with obs-mode image, (not rgb_image). So it seems to be that switching to resnet makes it worse at training. 

Feel free to read and run any code you want in order to investigate.

## Notes – 2025-09-11
- `configs/train_defaults.yaml` enables PPO with bank curriculum (`use_bank_curriculum: true`) and the ResNet feature extractor (`resnet: true`, `small_cnn: false`).
- Bank curriculum replaces the standalone env sizing: despite `height: 8`/`width: 8`, level 0 in `configs/curriculum_config_default.json` feeds 16×16 boards with 4 robots via `BankCurriculumWrapper`/`CriteriaFilteredEnv` (`train_agent.py:336-381`, `env/curriculum.py:601-707`).
- Policy path taken: image observations, non-recurrent, ResNet extractor ⇒ `policy="CnnPolicy"` with `features_extractor_class=ResNetFeaturesExtractor` and `features_dim=args.features_dim` (`train_agent.py:661-704`, config sets `features_dim: 32`).
- Warm-start files listed in YAML are ignored because `init_mode` remains `"fresh"`; no parameters are loaded (`train_agent.py:432-503`, `train_agent.py:735-805`).
- Attempted to instantiate `ResNetFeaturesExtractor` in Python for sanity checks, but `torch` is not available in the current environment (ModuleNotFoundError).
- Previously, `models/resnet.py` used `BasicBlock2Branch` without an identity skip; each block returned `trunk(x) + branch_a(ReLU(trunk)) + branch_b(ReLU(trunk))`, so the "ResNet" behaved like a plain CNN with three conv paths per block.
- The original backbone stacked 2 + 7 of these blocks with no downsampling or normalization, then flattened the full 16×16 grid into a 16,384-dimension vector before a linear projection—~1.3M parameters versus ~67k in the SmallCNN.
- SmallCNN instead pools spatially (global average) before the linear layer (`models/policies.py:17-31`), which likely stabilises scale and improves sample efficiency on sparse binary grids.
- Implemented a lighter residual extractor: `ResidualBlock` now preserves an identity skip with GroupNorm-stabilised two-conv path, adds stage transitions with post-conv norms, and the head applies adaptive average pooling + LayerNorm before projection (`models/resnet.py:9-118`). This drops the projection fan-in to 64 and keeps activations well-scaled.

### Next steps
- When `torch` is available, run sanity checks on the new extractor (forward pass stats, gradient norms) and compare short training curves against `--small-cnn` with the updated PPO defaults.
- Monitor curriculum progression under the relaxed thresholds; tighten again if advancement becomes too noisy.
- Decide whether to bump `batch_size` to synergise with `n_envs=8` or leave it at 32 after observing optimisation stability.

### Hyperparameter observations
- Sample throughput is low: `n_envs: 2` and `n_steps: 128` ⇒ 256 transitions per update, while `batch_size: 32` and `n_epochs: 10` recycle the same batch 10 times (`train_defaults.yaml`, `train_agent.py:640-717`). SB3 defaults are `n_envs ≥ 8` and `n_epochs ≈ 4`; the current combo risks high gradient variance and slow wall-clock progress.
- `ent_coef: 0.01` is relatively large; combined with sparse rewards, this keeps the policy more random, delaying convergence on level 0.
- Bank curriculum advancement is strict: `curriculum_success_threshold: 0.95`, `curriculum_min_episodes: 100`, `curriculum_window_size: 200`, `curriculum_check_freq: 50`. Even a well-performing agent must clear 100+ episodes (with 95% success) before advancing, so apparent stagnation may just be slow statistics accumulation.
- Updated defaults to speed feedback: `n_envs` increased to 8, `ent_coef` trimmed to 0.001, `curriculum_min_episodes` lowered to 60, `curriculum_success_threshold` relaxed to 0.9, and `curriculum_window_size` reduced to 100 (`configs/train_defaults.yaml:21-24,29,52`).
- Current runs use a reduced learning rate with linear decay (user tweak); early signs show steady improvement but slower progression.

### Curriculum pacing thoughts
- Level 0 puzzles are 16×16 with 4 robots and optimal length 1–2 (`configs/curriculum_config_default.json:3-18`). With `curriculum_min_episodes: 60` and `success_rate_threshold: 0.9`, budget ~70–80 completed episodes after stabilisation before advancing (window of 100 episodes with ≥90% success).
- Level 1 shares the same board spec but needs optimal length 2–4 (`configs/curriculum_config_default.json:19-34`). Expect roughly double the time: solve rate typically drops to ~60% initially, so allow ~150-200 total episodes to nudge the window above 0.9.
- Levels 2–3 (longer optimal lengths and multiple robots moved) can take 2–4× longer again; plan for 300–500 episodes each, assuming the agent keeps improving steadily.
- With `n_envs: 8`, `n_steps: 128`, each PPO update sees 1,024 environment steps. If average episode length is ~12 steps, that is ~85 episodes per update; curriculum windows will thus refresh every ~1.2-1.5 updates.
