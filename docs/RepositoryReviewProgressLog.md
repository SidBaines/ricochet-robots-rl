# Repository Review Progress Log

Review started: 2026-07-10  
Repository: `ricochet-robots-rl`  
Baseline commit: `de1ee77` (`main`, matching `origin/main`)  
Baseline working tree: clean

This is a chronological investigation log. Findings here are deliberately provisional; the final assessment and prioritized recommendations will be kept in `docs/RepositoryReviewReport.md`.

## 2026-07-10 — Initial inventory

### Scope and approach

- Preserve the repository's starting state and separate review artifacts from product-code changes.
- Reconstruct the intended system from the README, configuration, current handoff, history, and implementation.
- Establish a reproducible local environment, run the existing checks and workflows, and then use targeted experiments to distinguish confirmed defects from code-review suspicions.
- Review not only conventional software quality but also the RL experiment design: observation/action semantics, reward and termination behavior, curriculum integrity, evaluation leakage, metrics, reproducibility, and training throughput.

### Initial observations

- The Git working tree was clean on `main` and matched `origin/main` at commit `de1ee77`.
- The latest commit is dated 2025-10-30. There are 148 tracked files. The repository is approximately 33 MB, of which about 15 MB is documentation and 8.1 MB is examples.
- No repository-local `AGENTS.md`, packaging metadata (`pyproject.toml`, `setup.py`, or equivalent), lockfile, CI workflow, coverage configuration, lint configuration, or type-checker configuration was found in the initial file inventory.
- Runtime and development dependencies are combined in an unpinned lower-bound-only `requirements.txt`, apart from `moviepy==1.0.3`. This makes environment reconstruction vulnerable to dependency drift and currently offers no declared Python version.
- The project contains substantial historical design material under `docs/AiChatLogs/`. That can explain decisions, but it is much larger than the maintained user-facing and engineering documentation and may make the current source of truth difficult to identify.
- `README.md` describes the environment, solver, training framework, profiling, and monitoring as complete, while the commit history explicitly says many tests still failed at the time of a prior cleanup. This inconsistency needs to be resolved empirically.
- `WorkInProgress.md` is a chronological handoff ending in monitoring work. It references `configs/resnet_cellrgb2.yaml`, but that file is absent from the current tree. This may be harmless stale documentation or evidence that an important experiment configuration was never committed.
- The default training config warm-starts from `examples/resnet_example.zip`; this makes the nominal default run dependent on a binary checkpoint rather than being a clean-from-scratch baseline.
- The README's quick start requires generating a puzzle bank before default training, but the default config also selects warm-start and bank curriculum. Setup cost and failure modes should be tested from a clean checkout.

### Next investigation

Read the maintained design documents and all production modules, construct a dependency/data-flow map, and identify specific invariants to test before installing dependencies.

## 2026-07-10 — Architecture and environment setup

### Implementation map

- `src/env/ricochet_core.py` holds the board value object and pure slide/action/goal mechanics.
- `src/env/ricochet_env.py` combines Gymnasium lifecycle, procedural board generation, four observation encodings, RGB rendering, and several caches in one 1,360-line module.
- `src/env/solver.py` provides BFS and A* over robot-position states.
- `src/env/puzzle_bank.py` persists solved puzzles as Parquet partitions plus a JSON manifest and provides cached sampling.
- `src/env/precompute_pipeline.py` generates, solves, and records bank entries.
- `src/env/criteria_env.py` and `src/env/curriculum.py` layer bank filtering and curriculum progression around short-lived inner environments.
- `src/scripts/train_agent.py` is an 1,811-line entry point combining config parsing, environment creation, policy selection, checkpoint initialization, vector normalization, evaluation, monitoring, media recording, profiling, training, and saving.
- `src/models/` includes a feed-forward ResNet, a small CNN, ConvLSTM/DRC feature extraction, a custom recurrent policy, and an sb3-contrib recurrent-policy implementation.
- `src/monitoring/` provides a backend-agnostic hub, TensorBoard/W&B backends, collectors, evaluation, solver caching, and trajectory capture.

### Reconstructed runtime flow

1. YAML is flattened into an `argparse.Namespace` by `train_agent.py`.
2. A standard, online-curriculum, or bank-curriculum factory is selected.
3. SB3's `make_vec_env` constructs multiple wrappers. Curriculum workers share one manager; bank workers each create a sampler but share the `PuzzleBank` object.
4. An inner `RicochetRobotsEnv` produces channel-first observations. In bank mode, a new inner fixed-layout environment is created on every reset.
5. PPO or RecurrentPPO is constructed and callbacks handle curriculum updates, evaluation, checkpointing, custom metrics, and trajectory media.

### Environment reconstruction

- Created ignored virtual environment `rlenv/` using the machine default Python 3.14.6 on arm64 macOS 26.5.1.
- `pip install -r requirements.txt` succeeded.
- Because the project specifies almost exclusively lower bounds, the resolver selected current packages including NumPy 2.5.1, Gymnasium 1.3.0, pytest 9.1.1, Torch 2.13.0, Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, pandas 3.0.3, and W&B 0.28.0.
- `torchaudio>=2.3` independently resolved to torchaudio 2.11.0 while Torch resolved to 2.13.0. The dependency specification does not guarantee a compatible Torch/audio pair, and torchaudio does not appear to be used by the source.
- Installing `jupyter` pulls a large notebook/server stack into every environment even though the tracked notebook-style files are plain Python `#%%` scripts.

## 2026-07-10 — Baseline test suite

Command: `rlenv/bin/python -m pytest -ra`

Result: **34 passed, 16 failed, 4 errors, 3 warnings**; 54 tests collected in 21.67 seconds.

Failure groups:

- Eight curriculum tests fail. Several mock-based wrapper tests no longer provide the concrete attributes the implementation introspects, while real/default curriculum cases also fail and require separate diagnosis.
- The custom `RecurrentActorCriticPolicy` assumes both `pi` and `vf` architectures contain at least two layers; tests intentionally pass a valid one-layer SB3-style architecture, causing `IndexError`. This causes four setup errors and five additional recurrent-test failures.
- `test_drc_features_extractor_shapes_cpu` expects recurrent state at input resolution, while the implementation returns state at the encoder's downsampled spatial resolution (5×5 for a 10×10 input). Either the implementation contract or the test is stale.
- Two policy-output tests expect `model.predict()` to return a Python/NumPy scalar; SB3 2.9 returns a zero-dimensional NumPy array for an unvectorized discrete observation. Those two failures look like test compatibility issues rather than application defects.
- The test warnings identify deprecated `torch.cpu.amp.autocast` usage.

Passing groups include the core environment mechanics, observation basics, solvability gating, bank sampling, precompute coverage accounting, solver behavior/optimality on the included fixtures, PPO smoke training, and the two trivial learning milestones.

The suite currently has no tests for the RGB-cell observation mode, RGB cache invalidation, Gymnasium reset RNG progression, config-to-inner-environment propagation, CLI entry points, train/eval separation, deterministic bank sampling, monitoring integration, or a default-config end-to-end run.

## 2026-07-10 — High-risk code paths identified for experiments

These are source-review findings pending or undergoing direct reproduction:

- **Random reset behavior:** `RicochetRobotsEnv.reset(seed=None)` recreates its RNG from `_base_seed` every time. Gymnasium's normal convention is to seed on an explicit seed and continue the RNG stream on subsequent unseeded resets. In a vectorized run this appears to restrict each worker to one procedurally generated board after its initial rank-specific seed.
- **Stale wall overlays:** the RGB and RGB-cell internal-wall overlays rebuild only when absent or when their shape changes. A same-sized new layout can therefore retain the previous layout's walls. Random reset clears only `_cached_wall_obs`, not either RGB overlay.
- **Bank configuration loss:** `CriteriaFilteredEnv` accepts `max_steps` but the argument is commented out at both inner-environment construction sites. It also has no fields for configured step/no-op penalties or goal reward. Consequently the bank curriculum silently uses the inner environment defaults rather than the training YAML for those parameters.
- **Bank seed loss:** `CriteriaFilteredEnv.reset(seed=...)` explicitly ignores the seed, and every sampler initializes from OS entropy. A configured experiment seed therefore does not make puzzle order reproducible.
- **Evaluation contamination:** the bank evaluation environment is built from the training `env_factory`, which closes over the same `BankCurriculumManager`. Evaluation episodes call `record_episode_result()` and can affect or advance the training curriculum. Training and evaluation also sample the same bank without a held-out split.
- **Callback cadence mismatch:** SB3 `EvalCallback` and `CheckpointCallback` frequencies count vector-environment callback calls, not aggregate timesteps. The raw config frequencies are not divided by `n_envs`, so with eight environments an advertised interval of 1,000 timesteps occurs at roughly 8,000 aggregate timesteps.
- **Duplicate trajectory work:** the training script installs an always-on `_HubTrajectoryLogger` and then installs another fixed-set or default trajectory callback. The first always uses the non-curriculum factory, so bank training can log rollouts from `env_mode` instead of the bank distribution; the second default non-bank path records rollouts but does not log the returned media.
- **Online curriculum shape changes:** default online levels change board size and robot count, which changes image-channel count and the action space. Existing vector environments and a constructed policy cannot safely change those spaces after advancement.
- **Config loading defects:** online JSON levels are passed as dictionaries to a manager expecting `CurriculumLevel` instances. Bank mode without an explicit curriculum config tries to slice `None` before the helper can supply defaults.
- **Precompute entry-point defect:** `src/scripts/precompute_bank.py` passes an unsupported `solver_key=` keyword to `run_precomputation()`, whose parameter is `solver_config`.
- **Puzzle-bank integrity/performance:** reads are cached but `add_puzzles()` does not invalidate the cached DataFrame; appends rewrite the complete Parquet partition; writes and manifest updates are non-atomic; no locking is present; and the manifest advertises `dedup.method: none`. This conflicts with the README claim that incremental runs avoid duplicate layouts.
- **Generator provenance:** `generator_rules_version` is stored and used in partition names but is not used to dispatch a versioned generator. Replays always call today's generator and merely fail a hash check after drift. Although layout blobs are stored, the bank training path does not fall back to them.

## 2026-07-10 — Targeted environment probes

### RNG progression (confirmed defect)

Constructed a seeded 8×8 random environment, called unseeded `reset()` twice, and compared both the exposed `board_seed` and complete board state.

- First `board_seed`: `33158374`
- Second `board_seed`: `33158374`
- Walls, robot positions, goal, and target were all identical.

This confirms that each seeded random worker repeats one puzzle indefinitely rather than producing a reproducible sequence of different puzzles. Explicitly reseeding should reproduce a board; an ordinary unseeded episode reset should not rewind the generator.

### RGB wall-cache invalidation (confirmed defect)

For both RGB modes, reset one environment with seed 101 and then seed 202. The actual board walls changed, but the cached internal-wall overlay remained byte-for-byte identical:

| Mode | Actual walls changed | Overlay changed | Overlay shape |
|---|---:|---:|---:|
| `rgb_image` | yes | no | 120×120×3 |
| `rgb_cell_image` | yes | no | 17×17×3 |

The agent can therefore receive robot/goal pixels from the current board over wall pixels from a previous board. Bank mode happens to avoid this by creating a fresh inner environment for every puzzle, but standard random training and any reused same-size environment are affected.

### API checker

SB3's `check_env` completed for `image`, `symbolic`, `rgb_image`, and `rgb_cell_image`. It warned that the float channel observation resembles an image but is not uint8/0–255, and that 40×40 is near the default CNN's minimum resolution. These are advisory policy-selection concerns; the checker does not exercise multi-reset cache correctness.

## 2026-07-10 — Model integration probes

### ResNet (confirmed unusable)

Directly constructed `ResNetFeaturesExtractor` and `PPO(ResNetPolicy, ...)` for 40×40 `rgb_cell_image` and 128×128 `rgb_image` observations. All constructions failed in the dummy forward pass:

```text
RuntimeError: The size of tensor a (39) must match the size of tensor b (38) ...
RuntimeError: The size of tensor a (127) must match the size of tensor b (126) ...
```

The 4×4 convolutions with padding 1 shrink spatial dimensions. The trunk output and its branch outputs therefore cannot be added. The current ResNet cannot begin a run, and there is no ResNet test.

### DRC/RecurrentPPO (confirmed unusable)

`RecurrentPPO(DRCRecurrentPolicy, ...)` constructs for both RGB modes, but the first `predict()` fails with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x64 and 256x9)`.

Source inspection found additional incompatible assumptions: the custom encoder shrinks spatial dimensions while hidden/fence tensors retain input dimensions; the constructed fence is cropped after its interior is zeroed and is consequently all zeros; state types do not follow sb3-contrib's actor/critic recurrent-state contract; and conversion to NumPy does not consistently detach tensors.

The alternative `ConvLSTMFeaturesExtractor` and custom `RecurrentActorCriticPolicy` are not wired into `train_agent.py`. `--convlstm` selects stock sb3-contrib `CnnLstmPolicy`; `--drc` selects the separate broken `DRCRecurrentPolicy`. The configured `features_dim`, convolution channels, LSTM channels/layers, and repeat count are commented out or otherwise ignored by the active DRC branch.

### Model-selection/config mismatch

Policy selection sets `tiny_grid=True` whenever `env_mode` is `v0` or `v1`, even when bank curriculum replaces that environment with 16×16 bank puzzles. The default config has `env.mode: v1` and `model.type: resnet`, so it actually chooses `MlpPolicy`, not ResNet. The flat `resnet_rgb.yaml` has the same mismatch and would flatten a 128×128×3 input into an MLP.

Inspection of `examples/resnet_example.zip` confirms it is not a ResNet checkpoint. It contains SB3's standard `ActorCriticPolicy`, a `(10,16,16)` float observation space, 16 actions, and MLP `pi`/`vf` heads. It was produced with Python 3.10.13, SB3 2.7.0, Gymnasium 1.2.0, NumPy 2.2.6, and Torch 2.8.0. Its name and README description are misleading, but it is plausibly compatible with the accidental MLP bank path in the structured default config.

## 2026-07-10 — Profiling and monitoring investigation

### Profiling is always enabled (confirmed performance defect)

The global profiler initializes with `_enabled = True`. `train_agent.py` calls `enable()` when profiling is requested but never calls `disable()` otherwise. Every ordinary environment operation therefore takes process/system memory snapshots and locks profiling state, including nested observation/render scopes.

A 3,000-step fixed-layout RGB-cell microbenchmark showed:

| Profiling state | Time | Environment steps/sec |
|---|---:|---:|
| disabled | 0.0387 s | 77,613 |
| enabled (the default) | 0.2989 s | 10,037 |
| disabled, repeat | 0.0349 s | 86,034 |

This is a roughly 7.7–8.6× slowdown in the isolated environment loop. Neural-network training will amortize some of it, but the default behavior is the opposite of the configuration and is an avoidable throughput cost.

### PPO metric forwarding to W&B is not installed (confirmed defect on SB3 2.9)

Before `learn()`, a newly constructed PPO model has no `_logger`, so `hasattr(model, "logger")` is false. The custom `_HubSB3Writer` append block is skipped. `learn()` subsequently configures a logger containing only `TensorBoardOutputFormat`. This explains why native `train/*`, `rollout/*`, and `time/*` metrics do not reach W&B through the custom hub.

`tensorboard_log=args.log_dir` is passed to PPO regardless of `monitor_tensorboard`, so SB3's native TensorBoard logging remains enabled even when the monitoring toggle says false. The toggle controls only the second custom `SummaryWriter`.

### Other monitoring issues

- `ActionUsageCollector` is registered but `hub.on_step()` is never called, so it receives no actions and produces no metrics.
- `MonitoringHub.log_video()` silently swallows backend exceptions. Backends flush but are not closed; the W&B backend never calls `wandb.finish()`.
- `FixedSetEvaluator`, `OptimalityGapEvaluator`, and `TrajectoryRecorder` call recurrent models without carrying state or episode-start masks. Their recurrent results and videos would not measure the model's intended memory behavior.
- The training script contains two independently scheduled trajectory callback paths. This duplicates expensive deterministic rollouts and makes it hard to establish which callback produced a media artifact.

## 2026-07-10 — Static quality and dependency checks

Installed review-only tools into `rlenv/` and ran them without modifying source.

- `pip check`: passed; the resolved environment has no declared dependency conflicts.
- `pip-audit -r requirements.txt`: no known vulnerabilities found at review time.
- Ruff 0.15.21: 181 findings across `src/` (91 import-placement, 37 empty f-strings, 31 unused imports, plus semicolon, unused-variable, lambda-assignment, and ambiguous-name findings). The production packages excluding scripts/tests still have 31 findings.
- `ruff format --check src`: 50 files would be reformatted; only two already match the formatter.
- mypy 2.2 with missing imports ignored: 184 errors. Several are directly meaningful, including the invalid DRC `RNNStates` construction, optional-inner-environment misuse, incorrect solver union handling, unsupported `hashlib.blake3`, and config/manager type mismatches.
- Source comments in `resnet.py` and `drc_policy.py` retain generated citation artifacts such as `:contentReference[oaicite:...]`, suggesting code was transferred from an answer without a cleanup/verification pass.

There is no committed Ruff, formatter, mypy, coverage, pre-commit, or CI configuration, so these are informational counts rather than failures against an agreed project standard.

## 2026-07-10 — Puzzle-bank generation smoke test

Command (review-only bank):

```text
python src/scripts/generate_curriculum_bank.py \
  --config configs/curriculum_config_default.json \
  --bank_dir artifacts/review_puzzle_bank \
  --target_per_level 5 --max_puzzles_global 200 \
  --chunk_per_spec 20 --solver bfs --max_depth 8 --max_nodes 20000 --verbose
```

Observed behavior:

- The controller attempted four 20-puzzle chunks in its first iteration because all four levels share the same `SpecKey`; it does not deduplicate the generation plan by spec.
- It reported 80 attempted/generated approximately. The bank stored 38 solved puzzles and reached raw level coverages `{0: 7, 1: 13, 2: 18, 3: 9}`.
- The script's final “Bank statistics” incorrectly printed zero puzzles and zero partitions. The outer script created a `PuzzleBank` before the controller created/updated a separate bank object, then printed the stale first object's manifest/cache.
- A newly opened `PuzzleBank` correctly reported 38 rows in one 31 KB Parquet partition.
- Running the older `precompute_bank.py` non-dry path failed immediately with `TypeError: run_precomputation() got an unexpected keyword argument 'solver_key'`.

The bank controller's level coverage overlaps by design (for example a two-move, one-robot solution counts for levels 0 and 1). This can be useful for sampling, but raw per-level targets are not counts of disjoint examples.

### Generator and solver property probes

- Generated 2,000 intended 16×16 boards. Every board had the expected 32 internal wall segments, valid borders, distinct robots, an unoccupied goal, and perpendicular walls at the goal.
- On 200 generated 4×4/two-robot boards, BFS and admissible-one A* both found solutions, returned valid plans, and agreed on optimal depth in all cases.

These probes support treating core board construction and search as the strongest part of the repository, while noting that broader limits/scale and exact Ricochet Robots wall-spec equivalence have not been formally proven.

## 2026-07-10 — Configuration propagation probes

### Online curriculum

Requested RGB-cell size 40, horizon 33, no no-op, step penalty −0.25, and goal reward 7.0 through the training arguments. The constructed inner environment actually used:

- RGB-cell size 128
- horizon 10
- no-op enabled
- step penalty −0.01
- goal reward 1.0
- five actions for a one-robot level instead of the requested four

The online wrapper copies only observation mode and channel order from the base factory. It also changes observation/action spaces across its default levels, which is incompatible with an already constructed vector environment and policy.

`OptimalLengthFilteredEnv` calls `super().reset(seed=seed)` on every filtering attempt. Combined with the base environment's RNG rewind, all 100 outer attempts inspect the same generated result; after exhaustion it returns the non-matching puzzle anyway.

### Bank curriculum

Requested the same non-default rewards/horizon plus no-op enabled. The constructed bank inner environment preserved RGB-cell size 40 but actually used:

- wrapper criterion horizon 2, but inner horizon 10 (the argument is commented out)
- no-op disabled, because `make_bank_curriculum_env_factory()` does not forward `args.include_noop`
- step penalty −0.01 and goal reward 1.0, because those settings have no propagation path
- 16 actions rather than the requested 17

Two independently constructed bank wrappers both reset with seed 123 for eight episodes and produced different puzzle-seed sequences. This confirms the experiment seed does not control bank sampling.

## 2026-07-10 — End-to-end bank training smoke test

Created a review-only structured config matching the default design but reducing the run to 256 timesteps, two environments, one PPO epoch, no evaluation/media/checkpoints, and the miniature bank. Training completed successfully in 1.87 seconds wall time.

Important result: despite `model.type: resnet`, the saved model is `ActorCriticPolicy` with `FlattenExtractor`, observation space `(10,16,16)`, and `Discrete(16)`. This empirically confirms that the bank run is an MLP due to the unrelated `env.mode: v1` tiny-grid branch.

TensorBoard output was split across two event streams:

- The custom monitoring writer placed `train/success_rate`, optimality, and custom episode metrics in an event file at the log root.
- SB3 placed PPO `train/*`, `rollout/*`, and `time/*` metrics under `PPO_1/`.

This fragmentation can make metrics appear missing depending on which run is selected in TensorBoard. The W&B PPO-forwarding writer remains uninstalled as shown in the logger probe.

The misleading `examples/resnet_example.zip` did load with `exact_match=True` into this accidental MLP architecture, explaining why the default warm-start path can work despite not being a ResNet.

## 2026-07-10 — Level-0 baseline calibration

Validated all seven miniature-bank level-0 entries by replaying their stored optimal action sequences; all seven reached the goal.

Ran a seeded uniform-random policy for 5,000 episodes through the actual bank wrapper and its accidental ten-step horizon:

- Overall success: **18.02%**
- One-step puzzles: **29.78%**
- Two-step puzzles: **6.34%**
- Mean episode length: 8.99

The previously observed flat rate around 25% is therefore close to a plausible random baseline, especially if the evaluation sample contains more one-step puzzles. Any future training report should always include random, target-robot-only, and oracle baselines stratified by optimal length; a single aggregate success curve can conceal both sampling mix and failure to improve over chance.

## 2026-07-10 — User-facing script checks

- `demo_cli.py` starts but prints `None` instead of a board because its environment does not set `render_mode="ascii"`.
- `evaluate_agent.py` rejects `rgb_image` and `rgb_cell_image`, so it cannot evaluate the repository's advertised RGB training configs. Its `--resnet` flag and computed `policy_type` are unused, and its fallback loop double-counts the final episode.
- `bank_curriculum_preview.py` says it previews each level but slices the loaded list to `[:1]`; it is executable notebook-state code rather than a parameterized CLI and broadly suppresses output errors.

## 2026-07-10 — Controlled RGB-cell learnability experiment

The broken custom model paths and accidental default MLP make the historical training results hard to interpret, so I ran one narrow experiment designed only to answer whether the current environment, action encoding, and sparse reward can support learning at all.

Setup:

- PPO with SB3's stock `CnnPolicy`/`NatureCNN`
- channel-first 40×40 RGB-cell observations and 16 actions (no no-op)
- eight CPU environments, 128 rollout steps, batch size 256, four epochs, learning rate 3e-4
- 50,176 actual timesteps (SB3 rounded the 50,000 request to complete a rollout)
- bank curriculum held at level 0, which contained seven verified training puzzles (five one-step and two two-step)
- ten-step horizon, no evaluation callbacks or media, and the global profiler explicitly disabled

The run took 54.8 seconds. Its trailing 1,000-episode curriculum success rate rose from chance-like behavior to approximately 80.6%; the final PPO rollout success metric was 0.80.

I then evaluated the saved policy on each of the seven training layouts:

- Stochastic action selection: **641/700 successes (91.6%)**, using 100 episodes per layout. Every layout had nonzero success; per-layout results ranged from 66% to 100%.
- Deterministic action selection: **4/7 layouts solved**. On each of the other three, the argmax policy repeated one ineffective action until truncation.

Interpretation:

- The observation/action/reward path is learnable in a memorization-scale setting when it is paired with a viable stock CNN and the profiler is disabled.
- This is not a generalization result: all evaluation layouts were also training layouts, the bank was tiny, and one run/seed was used.
- The deterministic/stochastic discrepancy is substantial. Future evaluation must report both deliberately (or justify one), rather than silently relying on the SB3 default used by a particular callback.
- This result also argues against reward shaping or a more complex recurrent architecture as the first recovery task. The immediate blockers are wiring correctness, data splits, reproducibility, and a trustworthy feed-forward baseline.

## 2026-07-10 — Review wrap-up

- No product source was changed. Temporary configs, banks, checkpoints, logs, videos, and plots were created only under ignored `artifacts/`; the installed `rlenv/` is also ignored.
- The two durable review outputs are this chronological log and `docs/RepositoryReviewReport.md`.
- The final report converts the evidence above into a severity-ranked defect list and a staged recovery plan. Its overall classification is: a recoverable research prototype with promising core mechanics and solver code, but experiment results are not currently trustworthy until the environment lifecycle, configuration propagation, model selection, and evaluation boundaries are repaired and tested.
