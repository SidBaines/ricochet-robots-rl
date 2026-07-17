# Phase 5 plan — pure RL (no behaviour cloning) via a reverse start-state curriculum

**Status: PLAN — not yet started.**
**Author context: written 2026-07-17 by a planning agent after a full diagnosis of the phase 4/4b runs. Intended executor: an AI agent working in this repository. Follow it in order; each work package has explicit acceptance criteria. When something in the code contradicts this plan, trust the code, note the discrepancy in `WorkInProgress.md`, and adapt — line numbers below are approximate anchors, not contracts.**

---

## 0. Read this first

Before writing any code, read (in this order):

1. `CLAUDE.md` — hard-won lessons. The ones that will bite you in this plan:
   - Benchmark throughput on the **real training factory** (bank-backed resets), never a synthetic env.
   - **No `pandas.iterrows()`** in per-episode/per-reset paths.
   - **Fingerprints are contracts** — fixture and split fingerprints must be byte-identical after your changes.
   - **Launch long jobs detached** (`nohup caffeinate -is … &`); harness background tasks get killed at ~15–20 min on this machine.
2. `docs/GridCnnGeneralizationReport.md` — the full experiment arc (phases 1→4b).
3. The last ~150 lines of `WorkInProgress.md` — phase 4 and 4b entries.
4. Section 1 of this document (background), then the work packages.

Ground rules for the whole plan:

- Python via `.venv-phase2/bin/python`. Tests: `.venv-phase2/bin/python -m pytest src/tests/ -q`.
- **Never modify anything under `artifacts/phase4_bank/`** (or any other bank/split artifact). All start-state data is derived at runtime in memory.
- **Never evaluate the held-out test split during development.** Validation split only. The test split is touched exactly once, at the end of the full run (the harness already does this correctly).
- Every new `Phase2Config` field must default to "off"/`None` so that **all existing YAML configs reproduce byte-identical behaviour**. Backwards compatibility is an acceptance criterion of every work package.
- Record everything chronologically in `WorkInProgress.md` (append, never rewrite).
- Commit per work package with the test suite green.

---

## 1. Background and motivation (why this experiment exists)

### 1.1 Where the project stands

RL agents for Ricochet Robots (16×16 board, 4 robots, sliding moves; action = robot×direction, 16 discrete actions; the moved robot slides until blocked). Training runs through the deterministic benchmark harness `src/benchmarking/phase2.py`, driven by one YAML per experiment, on puzzle banks with BFS-labelled optimal solutions. Observations are structured binary planes (`obs_mode: image`), encoder is `GridPlanesResNet` (`src/models/grid_cnn.py`), algorithm is SB3 PPO 2.9.0 on MPS.

Reward (fixed in `Phase2Config.environment_spec`, `src/benchmarking/phase2.py` ≈160–174, implemented in `src/env/ricochet_env.py` ≈385–436): **−0.01 per step, +1.0 on reaching the goal, hard truncation at `max_steps=20`.** No shaping of any kind. Blocked moves (robot already flush against a wall/robot in that direction) are silent no-ops costing the normal −0.01. There is no action masking anywhere. Truncation-vs-termination bootstrapping is handled correctly by SB3 2.9 — that is *not* the problem.

Key prior results on the 16×16 `phase4_bank` (55,143 puzzles, optimal lengths 1–8, 1–3 robots moved; train split 41,395 puzzles; held-out baselines: random 4.4%, target-robot-only 8.5%):

| Run | Recipe | Held-out stochastic success |
|---|---|---:|
| Phase 4 | from-scratch PPO, staged **length** curriculum, 10.24M steps | **3.3%** (below random-ish; argmax collapse) |
| Phase 4b step 0 | behaviour cloning (BC) on stored BFS plans, **zero PPO** | **30.6%** |
| Phase 4b final | BC warmstart + 10.24M PPO steps | **30.6%** (peak 33.1% at 7.2M, never saved) |

### 1.2 The phase 4b post-mortem (diagnosis established 2026-07-17)

A full audit of the phase 4b artifacts and code established these facts — they motivate every design choice below:

1. **PPO contributed nothing.** Rollout reward was flat for ~10M steps (the curriculum reached its final stage at ~266k steps; from there `rollout/ep_rew_mean` random-walks in 0.44–0.57 with zero trend). Training-suite success *declined* from 68.3% (post-BC) to 61.8%. All competence came from BC.
2. **Zero-gradient batch composition.** Final-stage sampling is uniform over ~20 (length × robots-moved) difficulty cells. ~40% of episodes land on cells the policy solves <20% of the time — with a purely terminal reward these episodes all return exactly −0.20 and carry ~no advantage signal. Another ~30% are saturated easy cells. PPO sat at an equilibrium with almost no gradient-bearing data.
3. **From scratch (phase 4), the same mechanism is fatal earlier:** random success on length ≥3 at 16 actions × 20 steps is ~0–5%, so PPO never experiences reward on the frontier. The staged **length** curriculum did not fix this because mastering length ≤ k transfers weakly to *discovering* the first correct move of an unseen length-k+1 layout, and the advancement gate measured the stage *mixture* (dominated by easy lengths), not the frontier.
4. Secondary pathologies (fix them, but they are not the core): value head cold-start on a shared extractor after BC; no protection of the BC prior (constant lr 3e-4, `ent_coef` 0.01 constant, `clip_range` 0.2, `target_kl=None`); ~30% of the policy's actions in failing episodes are provably-null blocked moves (random = 36.7% — it barely learned to avoid them); only last-iterate weights were saved (the 33.1% checkpoint is lost); mid-run validation used 1 episode/puzzle (SE ≈ 4%, so dips/peaks were partly noise); SB3's `train/value_loss`, `explained_variance`, `approx_kl`, `clip_fraction` were computed but never logged.
5. **Generalization gap, separate problem:** 68% train vs 30% held-out. Not addressed by this plan (that is a data/augmentation axis); this plan is about whether *RL itself* can learn.

### 1.3 The question phase 5 answers

**Can the agent learn to play from scratch — no BC, gradient only ever from environment reward — given a well-designed curriculum?**

The scientific motivation: the project's end goal (`docs/ResearchPlan.md`) is mechanistic interpretability of learned *planning*. A BC-seeded policy inherits the BFS solver's move function, so "does the network learn to plan?" is contaminated — we may be interpreting a distilled solver. A pure-RL policy discovers its own strategy. In the design below the solver's plans are used **only to choose episode start states**, never to supervise actions — much weaker contamination.

### 1.4 Why a reverse (start-state) curriculum should work where the length curriculum failed

Every bank puzzle stores its BFS-optimal action sequence (`PuzzleMetadata.actions_encoded`). Replaying the first `L−d` actions of a length-`L` puzzle produces a board state exactly `d` moves from the goal (≤ `d` in edge cases — the suffix is a witness, so `d` is an upper bound on true distance; acceptable label noise). We curriculum over `d` ("distance-to-go"): start at `d=1`, grow to `d=8`, finish on the puzzles' true start states.

Mechanism (contrast with the failed length curriculum):

- At `d=1` even a random policy gets rewarded within a few steps (one correct action out of 16, up to 20 tries) → learning starts immediately from scratch, no BC needed.
- At stage `d=k`, one good move from a start state lands in a `d=k−1` state **of the same layout** — a distribution the value function was just trained on. Credit assignment reduces to one-step lookahead at every stage and does **not** depend on value generalization across unseen layouts (which is what broke the length curriculum: a length-k+1 puzzle's first move had to be *discovered by exploration* on a fresh layout).
- Difficulty is continuously tunable and the advancement gate can be computed **on the frontier band only**, fixing the phase-4 mixture-gate flaw.

This is the classic reverse-curriculum / "learn from demonstration *states*, not demonstration *actions*" method: Florensa et al. 2017, *Reverse Curriculum Generation for Reinforcement Learning*; Salimans & Chen 2018, *Learning Montezuma's Revenge from a Single Demonstration* (plain PPO + demo-state resets solved a canonical hard-exploration game).

One honest caveat to keep in mind when interpreting results: Sokoban-scale "from-scratch RL works" results (the DRC lineage) had denser reward (+1 per box). Ricochet Robots is all-or-nothing; the start-state curriculum is how we buy back that density without hand-designed shaping.

---

## 2. Goal, gates, and non-goals

**Primary deliverable:** one full 10.24M-step from-scratch PPO run on `phase4_bank` with the reverse curriculum and the instrumentation/stability fixes, plus analysis.

Gates (defined *before* running; judge honestly against them):

- **G0 (wiring, 8×8 smoke):** the smoke run advances ≥2 reverse stages via the frontier gate and all new W&B keys appear. If G0 fails, fix code — do not touch 16×16.
- **G1 (learning starts, 16×16):** the `d=1` frontier gate (window success ≥0.7) passes within the first ~150k episodes. `d=1` is so easy that failure here means a bug (reward wiring, start-state replay, masking of the gate), not a research finding. Stop and debug.
- **G2 (curriculum climbs):** all `d` stages clear via the mastery gate — **zero safety-valve firings** — within ~70% of the step budget.
- **G3 (headline):** held-out **stochastic** success on the final test, from the **validation-selected** checkpoint: ≥ 8.5% = pass (beats the target-only gate from scratch, which phase 4 could not do: 3.3%); ≥ 30.6% = BC parity (ambition).

Non-goals for phase 5 (do not scope-creep): the train→held-out generalization gap (data augmentation / bigger banks), 3-seed protocol (only after a single-seed success), DRC/recurrence, training-time action masking via `sb3-contrib` (that is a named contingency, not the core run).

---

## 3. Design overview

Seven work packages. WP1–WP3 are run-quality fixes justified by the post-mortem (useful for *every* future run). WP4–WP5 are the reverse curriculum itself. WP6 is optimizer hygiene. WP7 is configs + execution.

```
WP1  SB3 train-diagnostics → W&B          (observability)
WP2  per-checkpoint weights + best-val selection + 3 eval episodes
WP3  valid-action mask + masked-eval diagnostics
WP4  start-state machinery (layout override + plan-prefix replay)
WP5  reverse-curriculum levels + per-band frontier gating
WP6  lr schedule, ent_coef anneal, target_kl
WP7  configs, smoke run, dry run, full run, analysis
```

Integration points you will touch (verify current line numbers with grep before editing):

| Piece | File / symbol |
|---|---|
| PPO construction | `src/benchmarking/phase2.py` `_build_model` (≈739–768) |
| BC dataset replay pattern to imitate | `phase2.py` `_bc_dataset` (≈771–804) |
| Curriculum level/band construction | `phase2.py` `_level` / `_curriculum_levels` (≈203–250) |
| W&B rollout callback | `phase2.py` `_WandbRolloutCallback` (≈890–918) |
| Checkpoint loop / final save | `phase2.py` ≈1146–1253 |
| Eval episode loop | `phase2.py` `_episode` (≈513–575) |
| Per-reset puzzle sampling → env build | `src/env/criteria_env.py` `CriteriaFilteredEnv._create_current_env` (≈173–217) |
| Stage advancement manager | `src/env/criteria_env.py` `BankCurriculumManager` (≈443–558) |
| Episode-end success recording | `src/env/curriculum.py` `BankCurriculumWrapper.step` (≈723–738) |
| Env, `FixedLayout` dataclass | `src/env/ricochet_env.py` (`FixedLayout` at ≈39–47) |
| Slide/apply-action primitives | `src/env/ricochet_core.py` (`apply_action`, `slide_until_blocked` ≈78–123) |
| Layout reconstruction from bank row | `src/env/puzzle_bank.py` `create_fixed_layout_from_seed` (≈994–1033) |

---

## 4. Work packages

### WP1 — Log SB3 optimizer diagnostics to W&B

**Why:** phase 4b's `value_loss` / `explained_variance` / `approx_kl` / `clip_fraction` were computed by SB3 and thrown away (`verbose=0`, no bridge). Without them, none of the runs can be diagnosed.

**What:** extend `_WandbRolloutCallback` (`phase2.py`). In `_on_rollout_end`, SB3's logger still holds the metrics recorded by the *previous* iteration's `train()` call in `self.model.logger.name_to_value` (a dict). Forward every key that starts with `train/` to the existing W&B log call (they arrive with a one-iteration lag; that is fine — note it in a comment). Expected keys: `train/value_loss`, `train/policy_gradient_loss`, `train/entropy_loss`, `train/approx_kl`, `train/clip_fraction`, `train/explained_variance`, `train/loss`, `train/learning_rate`.

Log everything with an explicit `step=self.model.num_timesteps`. Known trap from phase 4b: W&B drops data whose step is lower than one already logged (BC logged steps 1–25, then the step-0 eval was silently discarded). With no BC phase this shouldn't recur, but keep all step arguments monotone in `num_timesteps` anyway.

**Tests:** unit test that runs PPO for 2–3 rollouts on a tiny env with a stub W&B run object (a list-appending fake) and asserts at least `train/approx_kl` and `train/explained_variance` were forwarded. Place next to the existing harness tests (`src/tests/test_phase2_benchmark.py` has patterns for building small configs).

**Accept:** test green; existing suite green; no behaviour change when `wandb_project` is unset.

### WP2 — Save checkpoint weights; select reported model by validation; 3 validation episodes

**Why:** phase 4b's best policy (33.1% at 7.2M steps) was never saved — only the faded last iterate was reported. And 1 episode/puzzle validation has SE ≈ 4% at p≈0.3, which polluted every mid-run judgement.

**What (all in `phase2.py`):**
1. In the checkpoint loop (≈1146–1161), after each checkpoint's evaluation, save `model.save(seed_dir / f"model_{actual_timesteps}")` when a new config flag `save_checkpoint_weights: true` is set. (~25–70 MB each; the output dir is gitignored; a dozen checkpoints is fine.)
2. After training, when a new flag `select_checkpoint_by_validation: true` is set: pick the checkpoint with the highest **validation stochastic** overall success rate, reload those weights, and run the final **test** evaluation on *that* model (in addition to, or instead of, the last iterate — report both numbers in `results.json` under `test_selected` and `test_final`, and make the report table show both with the selected one as headline). The test split is still evaluated only at the very end.
3. Change the phase-5 configs to `validation_episodes_per_puzzle: 3` (config already supports this; no code change).

**Tests:** extend the harness smoke test to assert checkpoint files exist and that `results.json` contains both test blocks with the selection metadata (`selected_checkpoint_timesteps`).

**Accept:** tests green; old configs (flags absent) produce the exact current behaviour.

### WP3 — Valid-action mask + masked-eval diagnostics

**Why:** a third of the action space is a silent no-op in any given state (random-policy no-movement rate 36.7%); the phase-4b policy never learned to avoid them (≈31% in failing episodes) and deterministic eval frequently locks into a repeated null argmax until truncation. We want (a) a reusable mask primitive, (b) masked eval numbers as a *diagnostic* (how much of the failure is null-action waste?). Training-time masking is deliberately **not** in the core run (new dependency, more risk) — see contingencies.

**What:**
1. `RicochetRobotsEnv.valid_action_mask(self) -> np.ndarray` (bool, shape `(action_space.n,)`): action `robot*4 + direction` is valid iff the robot's immediately adjacent cell in that direction is reachable (no wall between, not off-board, not occupied by another robot) — a slide moves ≥1 cell exactly under that condition. Implement against the board/wall representation used by `slide_until_blocked` in `src/env/ricochet_core.py`; do not duplicate wall math — reuse or factor out the single-step check. If `include_noop` were true the noop action would count as valid; harness envs have it false.
2. In `_episode` (`phase2.py`), when a new config flag `eval_mask_null_actions: true` is set and `controller == "model"`, replace `model.predict` with masked selection: get action logits/distribution via `model.policy.get_distribution(obs_tensor)` (see `_bc_pretrain` ≈832–846 for the obs-tensor pattern), set masked-out logits to `-inf`, then argmax (deterministic) or sample (stochastic).
3. Report masked results as **additional** summary keys (`*_masked`), never replacing the unmasked ones. The headline metric of every run stays unmasked for comparability with phases 2–4b.

**Tests:** (a) hand-construct a small board (see `src/tests/test_env_lifecycle.py` for construction patterns) and assert the mask matches brute force — for each action, apply it and check whether `robot_positions` changed; (b) an eval test with a uniform-random stub policy asserting masked eval never takes a no-movement action.

**Accept:** tests green; masking is opt-in; unmasked numbers unchanged.

### WP4 — Start-state machinery

**Why:** the core of the reverse curriculum: episodes must be able to start `d` moves from the goal, along the stored optimal plan.

**Design:** at reset time, after `CriteriaFilteredEnv` samples a puzzle and builds its `FixedLayout` (`create_fixed_layout_from_seed`), if the active band carries a `start_distance: d`, compute the robot positions after replaying the first `L−d` actions of `puzzle.actions_encoded` (`L = puzzle.optimal_length`), and build the env with `dataclasses.replace(layout, robot_positions=new_positions)`. `FixedLayout` is a plain dataclass (`ricochet_env.py` ≈39–47) with `robot_positions: Dict[int, Tuple[int, int]]`, so `replace` is safe; the layout-hash validation happens inside `create_fixed_layout_from_seed` *before* the substitution, so hash contracts are untouched.

**Implementation notes:**
- Replay at the board level with `ricochet_core.apply_action` on the board object the layout describes (≤7 applications per reset — negligible; do NOT build a throwaway `RicochetRobotsEnv` per reset just to replay). If, and only if, the dry-run throughput gate (WP7) fails, add an in-memory cache keyed `(layout_id, d) → robot_positions` built once at startup the way `_bc_dataset` iterates puzzles.
- Only puzzles with `L ≥ d` may be sampled for a `start_distance: d` band — enforce via the band's `min_optimal_length = d` (see WP5).
- Edge semantics: `d = L` means the original start (replay 0 actions). The state after `L−d` optimal moves is *at most* `d` moves from the goal (the plan suffix is a witness); true distance can occasionally be smaller. Document this in the docstring; it is acceptable curriculum-label noise.
- Sanity guard: assert `L - d >= 0` and that replaying the *full* plan from the offset state's prefix… (keep it simple: assert the prefix replay used exactly `L−d` actions and did not hit the goal early; if it did hit the goal early, the stored plan or replay is buggy — raise).

**Tests (new file `src/tests/test_reverse_start_states.py`):** for a handful of puzzles from a small fixture (the tests already build tiny banks — follow `src/tests/test_bank_sampler.py` patterns): for each `d` in `1..L`, build the offset layout, then step the env with the plan's **last `d` actions** and assert it terminates successfully in exactly `d` steps. Also assert `d = L` reproduces the original `robot_positions` exactly.

**Accept:** tests green; no change to any existing sampling path when `start_distance` is absent; `create_fixed_layout_from_seed` untouched.

### WP5 — Reverse-curriculum levels + per-band frontier gating

**Why:** the phase-4 advancement gate measured the stage *mixture* (dominated by mastered easy content) — it certified nothing about the frontier. Phase 5 gates on the frontier band alone.

**What:**

1. **Level construction** (`phase2.py`): new function `_reverse_curriculum_levels(config)`, used when new config key `curriculum_start_distances` (list of ints, e.g. `[1,2,...,8]`) is set (mutually exclusive with `curriculum_stage_max_lengths` — validate and raise if both are set). Stage `k` (1-indexed over that list) has bands for every `d` in `{1..k}`:
   ```python
   {"min_optimal_length": d, "max_optimal_length": config.max_optimal_length,
    "min_robots_moved": config.min_robots_moved, "max_robots_moved": config.max_robots_moved,
    "start_distance": d, "band_key": f"d{d}",
    "weight": config.curriculum_frontier_weight if d == k else 1.0}
   ```
   plus, when `reverse_final_full_starts: true`, one **final stage** appended that is exactly the existing `_level(config)` construction — per-(length, robots) cells, original start states, no `start_distance`. This makes the final training distribution identical to phase 4/4b's final stage and to the evaluation mix. Give its bands `band_key`s like `L{length}R{robots}`.
2. **Band plumbing:** `CriteriaFilteredEnv` must know which band produced the current episode (it already picks a band inside `sample_puzzles_stratified` — with `total_count=1` the band choice happens in `puzzle_bank.py` ≈759–775; the cleanest route is to sample the band index in `CriteriaFilteredEnv` itself, or have the sampler return the chosen band). Expose it as `info["curriculum_band"]` on the terminal step (and/or a `current_band_key` attribute the wrapper can read).
3. **Per-band windowed success** (`BankCurriculumManager`): keep the existing global window (other code depends on `window_success_rate`); add `band_windows: Dict[str, deque(maxlen=config.curriculum_frontier_window)]`. `BankCurriculumWrapper.step` (`curriculum.py` ≈723–738) passes the band key with each recorded episode.
4. **Advancement rule for reverse levels:** advance when — episodes at level ≥ `curriculum_min_episodes` AND the **frontier band's** window is full AND its mean ≥ `curriculum_success_threshold`. The frontier band of stage `k` is `f"d{k}"`. The final full-start stage has no frontier band and never advances (terminal). Keep the existing safety valve (`curriculum_max_episodes_per_stage`) — a firing is a *finding* (log it loudly to stdout and W&B), not a rescue.
5. **W&B:** log `curriculum/frontier_success_rate` (frontier band's window mean) and per-band means under `curriculum/band/{band_key}` from the rollout callback. These curves are the primary live view of whether the hypothesis is holding.

**Tests (extend `src/tests/test_curriculum.py` / new `test_reverse_curriculum.py`):**
- Level construction: stage list `[1,2,3]` → stage 2 has bands d1 (w=1) and d2 (w=frontier); final full-start stage appended when flagged; validation error when both curriculum modes are configured.
- Gating: feed the manager synthetic episodes where the mixture succeeds (easy bands at 100%) but the frontier is at 0.5 with threshold 0.7 → must NOT advance; raise frontier to 0.75 → advances. This is the exact phase-4 failure mode being prevented — make the test names say so.
- Band plumbing: wrapper→manager band keys arrive intact under the vec-env sharing pattern (manager shared across workers).

**Accept:** tests green; `curriculum_stage_max_lengths` configs behave exactly as before (byte-identical `results.json` structure on the existing smoke test); fingerprint/perf tests (`test_bank_split_perf.py`, `test_bank_sampler.py`) green.

### WP6 — Optimizer schedules

**Why:** phase 4b ran constant lr 3e-4 / constant `ent_coef` 0.01 / no KL guard for 10M steps; the policy drifted without constraint. From scratch we want early exploration and late stability.

**What (`phase2.py`):** three new optional config fields, all plumbed in `_build_model` or a small callback:
- `learning_rate_final` (float | null): when set, pass SB3 a linear schedule `lambda progress_remaining: final + (initial − final) * progress_remaining` (SB3's `progress_remaining` goes 1→0).
- `ent_coef_final` (float | null): when set, add a callback that linearly anneals `model.ent_coef` (a plain float attribute read by `PPO.train()` each update — mutating it works) from `ent_coef` to `ent_coef_final` over total timesteps.
- `target_kl` (float | null): pass through to the PPO constructor.

**Tests:** unit-test the schedule callable endpoints and the anneal callback (run a few tiny rollouts, assert `model.ent_coef` decreased and matches the linear formula at the recorded timesteps).

**Accept:** tests green; absent fields → PPO kwargs identical to today.

### WP7 — Configs, smoke run, dry run, full run

#### 7.1 Config: 8×8 smoke (`configs/phase5_smoke_8x8.yaml`)

Purpose: end-to-end wiring check in ~10–15 min (G0). Derive from `configs/phase2b_grid_cnn.yaml` (its bank `artifacts/phase2b_bank` is 8×8, 2 robots, lengths 1–4, robots-moved 1–2 — check the YAML for the exact spec fields and copy them; the spec must match the bank or the fingerprint check refuses to start). Then set/override:

```yaml
# reverse curriculum (lengths only go to 4 on this bank)
curriculum_start_distances: [1, 2, 3]
reverse_final_full_starts: true
curriculum_frontier_weight: 3.0
curriculum_success_threshold: 0.7
curriculum_frontier_window: 200
curriculum_min_episodes: 1000
curriculum_check_frequency: 100
curriculum_max_episodes_per_stage: 20000
# instrumentation
save_checkpoint_weights: true
select_checkpoint_by_validation: true
eval_mask_null_actions: true
validation_episodes_per_puzzle: 3
# schedules
learning_rate: 0.0003
learning_rate_final: 0.00003
ent_coef: 0.01
ent_coef_final: 0.001
target_kl: 0.03
# budget: tiny
training_seeds: [11]
checkpoints: [0, 100000, 300000]
# total steps: use the config's existing mechanism — the harness trains to max(checkpoints)
# no BC:
bc_pretrain_epochs: 0
wandb_project: ricochet-robots-rl
wandb_run_name: phase5-smoke-8x8
```

(Where this plan's key names differ from what you implemented, use yours — the YAML above is the *intent*.) Run it **foreground** (it is short):

```bash
.venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase5_smoke_8x8.yaml \
  --bank-dir artifacts/phase2b_bank \
  --output-dir artifacts/phase2_experiments/phase5_smoke_8x8
```

**G0 checklist (all must hold):** stdout shows stage advancement lines driven by the frontier gate; ≥2 reverse stages cleared; W&B run contains `train/approx_kl`, `train/explained_variance`, `curriculum/frontier_success_rate`, `curriculum/band/*`, `eval_val_*` and `*_masked` keys; checkpoint weight files exist; `results.json` has `test_selected`/`test_final`.

#### 7.2 Config: 16×16 full run (`configs/phase5_reverse_16x16.yaml`)

Copy `configs/phase4b_bc_16x16.yaml` (same board/bank/split/eval fields — the split fingerprint must come out identical to phase 4b's: `c0e97329…`, check `report.md` prints it) and change **only**:

```yaml
# --- remove/zero the BC block ---
bc_pretrain_epochs: 0            # pure RL — this is the whole point
# --- replace the length curriculum with the reverse curriculum ---
# (delete curriculum_stage_max_lengths)
curriculum_start_distances: [1, 2, 3, 4, 5, 6, 7, 8]
reverse_final_full_starts: true
curriculum_frontier_weight: 3.0
curriculum_success_threshold: 0.7
curriculum_frontier_window: 200
curriculum_min_episodes: 2000
curriculum_check_frequency: 100
curriculum_max_episodes_per_stage: 100000   # valve = finding, not rescue; 100k episodes ≈ 0.4–2M steps
# --- instrumentation (WP1–WP3) ---
save_checkpoint_weights: true
select_checkpoint_by_validation: true
eval_mask_null_actions: true
validation_episodes_per_puzzle: 3
# --- schedules (WP6) ---
learning_rate: 0.0003
learning_rate_final: 0.00003
ent_coef: 0.01
ent_coef_final: 0.001
target_kl: 0.03
# --- eval cadence: denser than 4b, and weights are saved at each ---
checkpoints: [0, 512000, 1024000, 2048000, 3072000, 4096000, 5120000,
              6144000, 7168000, 8192000, 9216000, 10240000]
training_seeds: [11]
wandb_run_name: phase5-reverse-16x16
```

Everything else (PPO batch geometry `n_envs: 16, n_steps: 128, batch_size: 1024, n_epochs: 2`, `gamma: 0.99`, `gae_lambda: 0.95`, `vf_coef: 0.5`, `max_grad_norm: 0.5`, model `grid_resnet` 6×128ch `features_dim: 512`, `max_steps: 20`, split seeds, suite sizes) stays **identical to phase 4b** so the comparison is clean.

#### 7.3 Dry run (16×16, ~10 min, foreground)

Before the long launch, run the full config with `checkpoints: [0, 200000]` in a throwaway copy (`configs/phase5_dryrun_16x16.yaml`, output dir `..._dryrun`). Verify:
- **Throughput ≥ 550 steps/s** on the real factory (phase 4b measured 716; the plan's replay-at-reset must cost ≈nothing). If below, profile resets first — see the `BankSampler._split_mask` lesson — and only then add the start-state cache from WP4.
- `d=1` frontier success is visibly climbing within the first ~50–100k steps (watch `curriculum/frontier_success_rate`). From-scratch random play at `d=1` already succeeds often (one correct action of 16 with up to 20 tries, plus generous slack), so the window should move quickly. If it is flat at random levels → bug (G1 logic applies already).
- Delete the dryrun output dir afterwards; leave the bank untouched.

#### 7.4 Full run (detached — mandatory on this machine)

```bash
mv artifacts/phase2_experiments/phase5_reverse_16x16 /tmp/old_phase5 2>/dev/null || true
nohup caffeinate -is .venv-phase2/bin/python src/scripts/run_phase2_benchmark.py \
  --config configs/phase5_reverse_16x16.yaml \
  --bank-dir artifacts/phase4_bank \
  --output-dir artifacts/phase2_experiments/phase5_reverse_16x16 \
  > artifacts/phase2_experiments/phase5_reverse_16x16.log 2>&1 &
```

Expected wall clock: ~4.5–5.5h at 600–716 steps/s (10.24M steps + 12 evals + BC-free). Watch the log file and the W&B run (`phase5-reverse-16x16` in project `ricochet-robots-rl`), not the process.

**Monitoring guide — expected shapes (write mid-run observations into `WorkInProgress.md`):**
- `curriculum/level` should climb through all 9 stages. Rough expectation if the hypothesis holds: early stages clear in tens-of-k episodes each; later stages slower. All stages cleared by ~6–7M steps leaves ~3M on the true task.
- `rollout/ep_rew_mean` will **drop at every stage advance** (harder starts) — this is expected and healthy; do not read sawtooths as regressions. The number that must trend up is `curriculum/frontier_success_rate` *within* a stage and the checkpoint validation curve across stages.
- `train/explained_variance`: should rise above ~0.5 early and stay there. Near 0 for long → value function is failing; expect everything else to stall.
- `train/approx_kl` should sit near/below 0.03 (the target_kl guard).
- Safety-valve firings print to stdout/log — each one is a G2 failure datum; note the stage and its frontier success at firing.
- Kill criteria (then debug rather than burn compute): frontier `d=1` flat at random after 150k episodes (G1); explained_variance still ≈0 at 1M steps; throughput collapsed below ~400 steps/s.

#### 7.5 Analysis and reporting (after the run)

1. **Gates:** state G1/G2/G3 outcomes explicitly, with numbers, in a new `WorkInProgress.md` entry (template below).
2. **Curves:** run the existing plotting scripts (`src/scripts/plot_training_by_difficulty.py`, `src/scripts/plot_difficulty_heatmap.py`) on the output dir — the per-checkpoint joint tables are persisted, so the across-training 2-D heatmap works now.
3. **Comparison to phase 4b** at matched checkpoints: overall + by-length + by-robots-moved validation curves; the interesting scientific comparison is pure-RL-final vs BC-step-0 vs BC+PPO-final.
4. **Plan-agreement metric (new, small script under `src/scripts/`):** for the selected checkpoint, measure the policy's argmax agreement with the stored optimal action over on-plan states (reuse `_bc_dataset` to get (obs, action) pairs) on (a) train split, (b) validation split. Phase 4b's BC policy scored 89.3% on train by construction. A pure-RL policy with high success but *lower* plan agreement is evidence it found different strategies — directly relevant to the interpretability motivation. Report all four numbers.
5. **Masked-vs-unmasked eval deltas** per checkpoint: quantifies how much residual failure is null-action waste (feeds the MaskablePPO contingency decision).
6. Update `docs/GridCnnGeneralizationReport.md` with a Phase 5 section mirroring the Phase 4b one (result first, mechanism, artifacts), and add the run to the artifacts table. Keep the headline honest: report the validation-selected checkpoint as headline with `test_final` alongside.

**`WorkInProgress.md` entry template:**

```markdown
### Phase 5: pure-RL reverse start-state curriculum (RESULT — <date>)
- Question: can from-scratch PPO (no BC) clear the 16×16 gate with a reverse curriculum? (Plan: docs/Phase5ReverseCurriculumPlan.md)
- G1 (d=1 learns): <pass/fail, episodes/steps to gate>
- G2 (all stages via mastery): <pass/fail; list any safety-valve firings with stage + frontier success>
- G3 (held-out): selected ckpt <steps>: <x.x%> stoch / <x.x%> det (gate 8.5%; BC-parity ambition 30.6%); last iterate: <x.x%>/<x.x%>
- Optimizer health: explained_variance <range>, approx_kl <range>, valve firings <n>
- Plan agreement (argmax vs stored optimal, train/val): <a%>/<b%> (BC reference: 89.3% train)
- Masked-eval delta at selected ckpt: <+x.x pts stoch, +y.y pts det>
- Interpretation + next lever: <honest read against §8 of the plan>
```

---

## 5. Validation checklist before the full run

Run all of these; every box must be checked:

- [ ] `.venv-phase2/bin/python -m pytest src/tests/ -q` fully green.
- [ ] Fingerprint contract: the phase-5 16×16 config's startup prints/records the **same split fingerprint** as phase 4b (`report.md` of `phase4b_bc_16x16` shows `c0e97329b109…`). If it differs, you broke the split path — stop.
- [ ] A pre-existing config (e.g. the harness smoke test fixture) still produces an identical run structure with all new flags absent.
- [ ] G0 smoke checklist (§7.1) fully passed.
- [ ] Dry-run throughput ≥550 steps/s and `d=1` learning visible (§7.3).
- [ ] Work committed per WP with tests, on the current working branch (`sid-learnable-agent` unless told otherwise).

---

## 6. Budget

| Item | Estimate |
|---|---|
| WP1–WP6 implementation + tests | 1–2 focused days |
| 8×8 smoke | ~10–15 min |
| 16×16 dry run | ~10 min |
| 16×16 full run | ~4.5–5.5 h wall clock, detached |
| Analysis + writeups | ~half a day |

---

## 7. Risks and mitigations

- **Reset-time replay slows the factory.** Mitigation: board-level `apply_action` replay (≤7 calls), throughput gate in the dry run, optional in-memory cache. Never `iterrows()`.
- **Frontier gate too strict/loose.** 0.7 on a 200-episode window is the starting point; if a stage oscillates just under the gate for >2× `curriculum_min_episodes` with clearly rising success, that is tuning, not failure — you may lower the threshold to 0.65 for a rerun, but record it. Do not fiddle thresholds mid-run.
- **Per-band windows under-filled for the frontier.** Frontier weight 3.0 over ≤8+ bands gives the frontier ~27%+ of episodes at late stages — 200-episode windows fill in <1k stage episodes. If you change band counts, re-check this arithmetic.
- **W&B step-monotonicity** silently drops data (bit phase 4b at step 0). Always pass `step=num_timesteps`.
- **MPS quirks / long-run stability:** the detached `caffeinate` launch is mandatory; the harness has no mid-run resume, so a killed run restarts from zero — do not "quickly restart the machine" during the run.
- **Interpretation risk — curriculum contamination:** even without BC, start-state selection leaks solver knowledge (which states lie on optimal paths). Say so in the writeup; the claim phase 5 can support is "no *action* supervision", not "no solver influence at all".

## 8. Contingency decision tree (if gates fail)

Ordered by cost; try one change at a time, each as a fresh config + `WorkInProgress.md` entry:

1. **G1 fails (d=1 never learns):** it's a bug, not research. Check in order: start-state replay correctness (WP4 tests on the *16×16* bank), reward received on success (log a handful of episodes), obs planes from offset starts (target mask/robot planes must reflect moved robots), band plumbing (is the gate reading the right band?).
2. **G2 fails at some k (valve fires):** the one-step-frontier mechanism is being diluted or the policy lacks capacity to represent the growing state distribution. Levers, in order:
   a. Raise `curriculum_frontier_weight` 3→5 (more frontier data per update).
   b. `n_epochs` 2→4 (more reuse per rollout; costs ~throughput — re-measure).
   c. Add **training-time masking** via `sb3-contrib` `MaskablePPO` + `ActionMasker` using WP3's `valid_action_mask` (pin `sb3-contrib` matching SB3 2.9.0; this shrinks effective branching 16→~10–11). This is the biggest hammer; it changes the algorithm class, so rerun the smoke first.
   d. Insert intermediate stages (e.g. repeat the stalled k with `min_optimal_length` restricted to shorter puzzles first).
3. **G2 passes but G3 lands between 8.5% and BC's 30.6%:** the run is a *success* for the hypothesis (from-scratch RL works; phase 4 got 3.3%) with a quantified gap to imitation. Next levers are the frontier-adaptive weighting on the final stage (weight cells by success×(1−success)) and the generalization axis (augmentation/bank size) — write them up as phase 5b candidates; do not bolt them onto this run.
4. **G3 ≥ 30.6%:** BC parity from pure RL. Proceed to the 3-seed protocol (seeds 11/29/47) before claiming anything, then the DRC comparison per `docs/ResearchPlan.md`.

## 9. Definition of done

- All WP tests merged and green; fingerprints unchanged; old configs behave identically.
- One completed 16×16 full run with all gates explicitly adjudicated in `WorkInProgress.md`.
- `docs/GridCnnGeneralizationReport.md` updated with the Phase 5 section and artifacts row.
- Plots + plan-agreement + masked-delta numbers produced.
- A short "what next" recommendation grounded in §8.

## 10. Quick reference — numbers you will compare against

| Quantity | Value |
|---|---|
| Bank | `artifacts/phase4_bank`, 55,143 puzzles, fingerprint prefix `3579113b230a` |
| Split | train 41,395; fingerprint `c0e97329b109…` (must reproduce) |
| Baselines (held-out, stochastic) | random 4.4%, target-only 8.5%, oracle 100% |
| Phase 4 (scratch, length curriculum) | 3.3% stoch final; stages advanced only by valve |
| Phase 4b BC-only (step 0) | 30.6% stoch / 24.8% det val; train-suite 68.3% |
| Phase 4b final (BC+PPO 10.24M) | 28.8% stoch / 24.2% det test; best val 33.1% @7.2M (unsaved) |
| Phase 4b throughput | 716 steps/s train; random no-movement rate 36.7% |
| Reward | −0.01/step, +1.0 goal, max 20 steps, 16 actions, no masking |
