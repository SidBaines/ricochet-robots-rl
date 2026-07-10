# Repository Review Report

Review date: 2026-07-10  
Baseline: commit `de1ee77` on `main`, initially clean and matching `origin/main`  
Detailed evidence: [RepositoryReviewProgressLog.md](/Users/sidbaines/Documents/ricochet-robots-rl/docs/RepositoryReviewProgressLog.md)

## Executive assessment

This repository is a **recoverable research prototype**, not a dependable experimental platform in its current state.

The strongest foundation is the game layer: board construction, slide mechanics, basic procedural generation, BFS/A* solving, and optimal-plan metadata worked well in property probes. A small controlled PPO run also showed that the RGB-cell observation and action/reward path can learn a tiny puzzle set.

The present training results are nevertheless not trustworthy. Several independent defects mean a run does not necessarily train on the observations, puzzles, model, horizon, rewards, or evaluation distribution described by its YAML:

- an ordinary random-environment reset repeats the same board forever;
- RGB wall pixels can belong to the preceding board;
- curriculum wrappers silently drop environment parameters;
- the default “ResNet” run actually selects an MLP, while the real ResNet and DRC paths fail before useful inference;
- bank evaluation shares training data and, in one path, the mutable training curriculum manager;
- configured callback intervals are multiplied by the number of environments;
- profiling is on by default despite `profile: none` and reduces isolated environment throughput by about 8×.

I recommend preserving the core mechanics and solver, but rebuilding one narrow, tested training/evaluation vertical slice before resuming architecture or reward experiments. Historical learning curves should be treated as exploratory only and reproduced after that stabilization.

No product code was changed during this review. The durable additions are this report and the chronological progress log.

## What was exercised

| Area | Result |
|---|---|
| Dependency installation | Successful in an ignored Python 3.14.6 virtual environment; current lower bounds resolved to recent packages including Torch 2.13, Gymnasium 1.3, SB3 2.9, and NumPy 2.5 |
| Existing tests | 54 collected: **34 passed, 16 failed, 4 errored** |
| Gymnasium/SB3 API checks | All four observation modes passed `check_env`, with image-policy warnings |
| Generator properties | 2,000 intended 16×16 boards had valid borders, expected internal-wall counts, distinct robots, an unoccupied goal, and expected goal-wall geometry |
| Solver agreement | BFS and admissible-one A* solved 200 generated small boards and agreed on optimal depth |
| Puzzle-bank generation | Working main generator stored 38 solved puzzles, but exposed planning, summary, cache, provenance, and durability defects; older entry point crashed |
| Default-style training smoke | Completed 256 timesteps, but proved that the requested ResNet was actually a flattened MLP |
| Random baseline | 18.02% across 5,000 miniature-bank level-0 episodes; 29.78% on one-step and 6.34% on two-step puzzles |
| Controlled learnability run | Stock-CNN PPO reached 91.6% stochastic success on repeated episodes from its seven training puzzles after about 50k steps; deterministic success was 4/7 layouts |
| Static checks | 181 Ruff findings; 50/52 source files would be reformatted; 184 mypy findings with missing third-party imports ignored |
| Dependency health | `pip check` passed; `pip-audit -r requirements.txt` found no known vulnerabilities at review time |

The learnability run is deliberately an overfit test, not a generalization result. It establishes that a simple viable policy can learn the current task path; it does not validate the curriculum or reported experiment methodology.

## What is already worth keeping

### Core game and search logic

The separation in [ricochet_core.py](/Users/sidbaines/Documents/ricochet-robots-rl/src/env/ricochet_core.py) between a board value object, action application, and goal detection is a sound starting point. The generator and solver passed much stronger probes than the surrounding training stack. Optimal plans stored in the review bank replayed correctly.

The solver is simple rather than highly optimized—BFS copies board/path state and the node limit counts generated neighbors—but it is useful as an oracle for small depths, curriculum labels, evaluation, and imitation data.

### Puzzle metadata intent

The bank records generation specifications, solver configuration, hashes, seeds, solution lengths, robots moved, plans, and serialized layouts. That is the right kind of provenance for an RL benchmark. The implementation needs stronger versioning and storage semantics, but it should be repaired rather than replaced casually.

### Operational tooling

There are useful ingredients for checkpoint initialization, TensorBoard/W&B abstraction, trajectory recording, solver-backed metrics, and visual previews. A local trajectory-to-GIF/MP4 smoke test worked headlessly. These tools mainly need consolidation and contract tests.

## Critical findings: fix before further training

### P0. Environment episodes and observations are incorrect across resets

[RicochetRobotsEnv.reset()](/Users/sidbaines/Documents/ricochet-robots-rl/src/env/ricochet_env.py:266) recreates its random generator from `_base_seed` on every `reset(seed=None)`. A seeded worker therefore sees the same procedural puzzle every episode. Because construction also resolves a base seed when none is supplied, nominally unseeded environments repeat as well.

Direct reproduction returned the same `board_seed` (`33158374`) and identical complete state on two consecutive resets. The required behavior is:

- `reset(seed=N)` starts a reproducible random stream;
- later `reset()` calls advance that stream;
- calling `reset(seed=N)` again reproduces the stream from its beginning.

The RGB render paths have a separate correctness failure. [The cached overlay check](/Users/sidbaines/Documents/ricochet-robots-rl/src/env/ricochet_env.py:833) invalidates on pixel shape only, not wall layout. For both RGB modes, boards generated with seeds 101 and 202 had different walls while their overlays remained byte-identical. The policy can therefore observe current robots and goal on stale walls.

Required action: fix the RNG lifecycle and tie every layout-dependent cache to a layout fingerprint, or rebuild it at each board replacement. Add multi-reset tests comparing cached output against a freshly constructed reference environment.

### P0. Configuration is not the runtime source of truth

Bank and online curriculum wrappers reconstruct inner environments from partial argument sets. Important settings disappear silently:

- [CriteriaFilteredEnv](/Users/sidbaines/Documents/ricochet-robots-rl/src/env/criteria_env.py:64) comments out `max_steps` at both construction sites and has no path for configured rewards.
- The bank factory does not forward `include_noop`.
- The online wrapper carries only observation mode and channel order from the base environment. It drops cell-pixel size, horizon, no-op setting, rewards, and seed.
- Default online levels change board and action dimensions after policy construction.

A direct probe requesting RGB-cell size 40, horizon 33, no no-op, step penalty −0.25, and reward 7.0 produced an online inner environment with size 128, horizon 10, no-op enabled, penalty −0.01, reward 1.0, and the wrong action count. The equivalent bank probe preserved only the cell size; its horizon, reward, and action count were also wrong.

Required action: introduce one validated immutable environment specification and pass it intact through standard, bank, evaluation, fixed-set, media, and curriculum factories. Reject unknown keys and log a resolved specification plus actual observation/action spaces before training.

### P0. Advertised model selection is false or nonfunctional

The structured default says `model.type: resnet`, yet [policy selection](/Users/sidbaines/Documents/ricochet-robots-rl/src/scripts/train_agent.py:852) declares any `env.mode` of `v0`/`v1` a tiny grid. Bank curriculum later supplies 16×16 boards, but this unrelated flag forces `MlpPolicy`. The smoke checkpoint was an `ActorCriticPolicy` with `FlattenExtractor`, `(10,16,16)` observations, and 16 actions.

The warm-start artifact `examples/resnet_example.zip` is likewise a standard MLP checkpoint, not a ResNet. It loads only because it matches the accidental architecture.

The actual [ResNet](/Users/sidbaines/Documents/ricochet-robots-rl/src/models/resnet.py:7) cannot construct on either 40×40 RGB-cell or 128×128 RGB input: 4×4 convolutions shrink the residual branches to incompatible sizes. The DRC/RecurrentPPO path constructs but fails on its first prediction with incompatible matrix dimensions; its encoder/state geometry and recurrent-state contract have further problems. Several configured architecture fields are ignored.

Required action: choose a policy from the *constructed observation/action spaces and explicit model type*, then instantiate and validate it before starting a run. Temporarily remove or fail clearly on unsupported ResNet/DRC choices. Add a model matrix test covering construction, one prediction, and a short update for every advertised model/observation combination.

### P0. Evaluation is not independent

Bank evaluation can use [the training `env_factory`](/Users/sidbaines/Documents/ricochet-robots-rl/src/scripts/train_agent.py:1433), which closes over the same mutable curriculum manager. Evaluation episodes record results and may advance the training curriculum. Both sides sample the same bank, with no held-out split.

Online evaluation freezes only the level present when the callback is constructed, so it does not follow later curriculum progression. When normalization is enabled, evaluation statistics are copied only once instead of being synchronized during training. Recurrent evaluators do not carry recurrent state or episode-start flags.

Required action: create immutable, independently seeded train/validation/test manifests and separate managers. Evaluation must never mutate training state. Decide whether the headline metric is fixed-distribution test performance or current-level performance; ideally report both. Synchronize normalization immediately before each evaluation.

### P0. Profiling contradicts configuration and materially slows execution

The global [Profiler starts enabled](/Users/sidbaines/Documents/ricochet-robots-rl/src/profiling/profiler.py:68). The training script enables it when requested but never disables it otherwise. A fixed-layout RGB-cell microbenchmark fell from 77k–86k environment steps/sec to 10k steps/sec when profiling was enabled, a 7.7–8.6× isolated slowdown.

Required action: default profiling off; enable it only in an explicit scoped diagnostic run. Do not collect process memory on every nested environment operation.

## High-priority engineering and methodology findings

### P1. Metrics can be missing, duplicated, or scheduled at the wrong time

- PPO's logger does not exist before `learn()` in SB3 2.9, so [the W&B forwarding writer installation](/Users/sidbaines/Documents/ricochet-robots-rl/src/scripts/train_agent.py:1079) is skipped. Native PPO metrics do not reach the hub.
- SB3 TensorBoard logging is enabled through `tensorboard_log` even when the config toggle is false. Custom and native metrics go to separate event streams.
- `ActionUsageCollector` is registered but never fed step events.
- Two trajectory systems perform overlapping rollout work, and one uses the non-curriculum environment even during bank training.
- Evaluation and checkpoint callback frequencies count vector steps. Raw configured values are not divided by `n_envs`; with eight environments, `eval_freq: 1000` means roughly 8,000 aggregate environment steps.
- Monitoring broadly catches exceptions, so media/backend failures can look like successful logging.

Consolidate this into one callback graph with aggregate-timestep semantics. Add an integration test that runs a tiny training job and asserts exact metric names, callback counts, artifact locations, and clean backend shutdown.

### P1. Puzzle-bank generation is useful but not durable or reproducible

The main controller works, but all four default curriculum levels share one `SpecKey` and the planner generated a chunk for each level rather than once per unique spec. The final CLI summary used a stale bank object and printed zero despite 38 stored puzzles. The older [precompute entry point](/Users/sidbaines/Documents/ricochet-robots-rl/src/scripts/precompute_bank.py:77) passes the wrong keyword and crashes.

[PuzzleBank.add_puzzles()](/Users/sidbaines/Documents/ricochet-robots-rl/src/env/puzzle_bank.py:276) rewrites the full partition, does not invalidate its read cache, performs non-atomic data/manifest writes, has no locking, and explicitly records no deduplication. Repeated small chunks are therefore quadratic and unsafe under interruption or concurrent writers.

Generation uses unrecorded global randomness at the controller level. `generator_rules_version` appears in metadata but does not dispatch a frozen generator implementation; replay always uses current code and only discovers drift after a hash mismatch. Serialized layouts exist but training regeneration does not fall back to them.

Recommended bank contract:

- seed every generation and sampling stream explicitly;
- deduplicate on a declared fingerprint before commit;
- write immutable chunk files plus an atomically replaced manifest, or use a transactional store;
- validate counts/histograms from persisted data rather than mutable in-memory state;
- dispatch generator versions or treat the serialized layout as canonical;
- create split manifests keyed by `layout_hash` when wall-layout generalization is desired.

### P1. The current experiment protocol cannot establish generalization

The review's random policy achieved 18.02% aggregate success on the small level-0 bank. Historical flat success near 25% can therefore be chance-like, especially if one-step puzzles are overrepresented. Aggregate success also hides the large difference between one-step (29.78% random) and two-step (6.34% random) tasks.

The controlled CNN overfit showed a healthy signal—91.6% stochastic success on its seven training boards—but deterministic evaluation solved only four. This demonstrates why the next benchmark must include:

- disjoint train/validation/test layouts;
- success stratified by optimal length and number of robots moved;
- random, target-robot-only, and oracle baselines;
- deterministic and stochastic policy results identified explicitly;
- no-movement/invalid-action rate and episode length;
- multiple training seeds with uncertainty, not one best curve;
- the exact bank/split fingerprint, environment spec, package versions, Git SHA, and actual policy summary.

Sparse reward is not yet shown to be the primary problem. Do not tune reward shaping or introduce recurrent complexity until the fixed feed-forward baseline is measured correctly.

### P1. Project setup and regression protection are missing

There is no package metadata, supported Python range, lock/constraints file, CI, formatter/linter configuration, coverage threshold, or pre-commit setup. `requirements.txt` mixes training, notebooks, plotting, media, monitoring, testing, and unused audio dependencies under mostly unconstrained lower bounds. This happened to install on Python 3.14, but does not define a reproducible research environment.

The current tests omit the highest-risk seams and have 20 non-passing cases. Some failures are stale expectations, but real curriculum and recurrent-policy failures are mixed into the same red baseline.

Create `pyproject.toml` with a deliberately supported Python version, separated optional dependency groups, and a reproducible lock/constraints mechanism. Start CI with one pinned reference environment and make all tests green before expanding the matrix.

### P1. Training orchestration is too centralized

[train_agent.py](/Users/sidbaines/Documents/ricochet-robots-rl/src/scripts/train_agent.py) is 1,811 lines and defines configuration translation, factories, algorithms, initialization, callbacks, evaluation, monitoring, media, profiling, and saving. The environment module similarly combines generation, Gym lifecycle, four encodings, rendering, and caches. Duplicate curriculum data classes and rendering implementations have already drifted.

Decompose around testable contracts rather than merely splitting by file size:

1. parsed and validated `ExperimentConfig`;
2. `EnvironmentSpec` and environment factory;
3. immutable `PuzzleSplit`/sampler;
4. explicit policy builder;
5. evaluation protocol;
6. callback/monitoring builder;
7. run manifest and checkpoint lifecycle.

## Lower-priority cleanup

### P2. User interfaces and documentation

- `demo_cli.py` prints `None` because it does not request ASCII rendering.
- `evaluate_agent.py` rejects both RGB observation modes, ignores its ResNet policy choice, and has fallback accounting defects.
- `bank_curriculum_preview.py` previews only the first level and behaves like an executable notebook rather than a CLI.
- `roc_curves_plot.py` contains a developer-specific absolute path.
- Some “initial” scripts execute substantial work at import time.
- README status claims are substantially ahead of the verified implementation, and `WorkInProgress.md` references a missing config.
- No license file is present, so reuse terms are unclear.

Replace the user-facing path with a small set of tested commands: `bank generate/validate`, `train`, `evaluate`, and `rollout`. Update the README only after those commands run from a clean environment.

### P2. Static consistency

Ruff, formatting, and mypy expose a large cleanup backlog, including unused imports, imports after executable code, invalid typing contracts, and generated citation artifacts in model source. Adopt checks incrementally: establish a formatter and a focused lint baseline first, then tighten typing around configuration, boards, bank metadata, and policy factories where it protects runtime contracts.

## Recommended recovery sequence

### Phase 1 — Establish a truthful vertical slice

1. Declare and lock one supported environment, preferably the Python/SB3/Torch family selected intentionally rather than whatever lower bounds resolve today.
2. Make profiling opt-in.
3. Correct reset RNG progression and all layout-cache invalidation.
4. Introduce one `EnvironmentSpec`; propagate and assert every field in all factory paths.
5. Select the model from explicit config plus actual spaces. Disable broken model choices with clear errors.
6. Use one stock feed-forward CNN on fixed `(10,16,16)` spatial planes or the known-working 40×40 RGB-cell input. The structured planes avoid unnecessary rendering complexity; a small same-padded CNN with coordinate channels is a sensible first custom architecture.
7. Split a validated bank into immutable train/validation/test manifests, with independent deterministic samplers and managers.
8. Build one evaluation callback with correct vector-step cadence and synchronized normalization.
9. Record a resolved run manifest and make a tiny end-to-end run a CI test.

Do not begin by repairing DRC. The full board is observable, so recurrence is not required for the Markov state; a reliable feed-forward baseline is the necessary control.

### Phase 2 — Re-establish a benchmark

Run a compact sweep over at least several training seeds using fixed spaces and one curriculum dimension at a time. Report:

- held-out success and confidence intervals;
- results by optimal length and robots moved;
- deterministic and stochastic performance;
- random and simple heuristic baselines;
- action/no-movement distributions, episode lengths, and sample throughput;
- training versus held-out curves with a fixed evaluation budget.

Treat a solved tiny training set as an automated overfit milestone. Add a separate held-out milestone before calling a model successful.

### Phase 3 — Restore curriculum and bank scale

Keep board dimensions, observation channels, robot slots, and action dimensions fixed while difficulty changes. Curriculum should vary solution length, dependency on blocking robots, and perhaps board-density parameters without changing tensor contracts. If variable robot counts are important, use a fixed maximum plus masks.

Make progression depend only on a specified training window. Evaluate both the current level and a permanent held-out suite without writing into the manager. Harden bank generation and fingerprinting before producing a large corpus.

### Phase 4 — Architecture research

Only after Phase 2 is reproducible:

- implement and test a same-padded residual CNN that preserves spatial coordinates;
- compare structured planes with RGB-cell observations at equal evaluation budgets;
- measure whether masking moves that leave the board unchanged improves sample efficiency;
- consider behavior cloning or an auxiliary first-action objective from solver plans;
- add recurrence/DRC only against a task that genuinely benefits from memory or iterative computation, with complete sb3-contrib contract tests.

## Acceptance criteria for the stabilization milestone

| Contract | Minimum acceptance test |
|---|---|
| RNG | Two unseeded resets differ; reseeding reproduces the full same sequence across processes |
| Observations | Cached RGB outputs equal fresh-render reference outputs after arbitrary same-size layout changes |
| Config | Every environment field is asserted at the final inner environment for standard, bank, online, fixed-set, evaluation, and media paths |
| Spaces | Observation/action spaces stay constant for a run and match the saved checkpoint manifest |
| Models | Every advertised policy constructs, predicts, performs one learning update, saves, reloads, and predicts again—or is rejected at config validation |
| Evaluation | No evaluation operation changes training manager/sampler state; split membership is disjoint under the declared fingerprint |
| Cadence | Callback invocation points match aggregate timesteps for one and multiple vector environments |
| Monitoring | One smoke run exposes native PPO and custom metrics in each enabled backend without duplicate trajectories |
| Bank | Stored optimal plans replay; duplicate input is idempotent; interrupted writes leave the previous manifest readable |
| Benchmark | Random baseline plus multi-seed train and held-out metrics are produced from a committed small fixture/config |

## Suggested first implementation batch

The highest-value next coding session is a deliberately small batch:

1. fix profiler default, RNG progression, and RGB cache invalidation;
2. add regression tests for those three defects;
3. define and propagate `EnvironmentSpec` through the bank path;
4. replace default model selection with an explicit stock-CNN baseline and fail on unsupported model types;
5. create deterministic disjoint bank splits and an evaluation manager that cannot mutate training;
6. commit a tiny bank fixture or deterministic fixture builder plus a short overfit/evaluation smoke test;
7. make the existing test suite green, then add pinned CI.

That batch turns the repository from “runs, but its meaning is ambiguous” into a platform where subsequent RL choices can be evaluated. Monitoring consolidation, bank storage redesign, CLI cleanup, and new architectures can then proceed without invalidating the scientific baseline.

## Review limitations

- The investigation used one arm64 macOS machine, CPU training, and the current dependency resolution; no GPU/CUDA path was tested.
- The controlled training run used seven training layouts and one seed. It tested learnability/memorization only, not generalization.
- Large-bank throughput, concurrent bank writers, crash recovery, and long-running W&B service behavior were assessed by code inspection and focused probes rather than production-scale stress tests.
- The generated-board property checks establish internal consistency, not a formal proof that every procedural wall rule exactly matches the physical Ricochet Robots board standard.
- No attempt was made to reproduce every historical result because the confirmed experiment-integrity defects would make that compute difficult to interpret.
