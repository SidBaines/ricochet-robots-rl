# DRC Helper-Free Slide-Dynamics Research Plan and Execution Log

Date created: 2026-07-23
Status: complete on 2026-08-13; 200M closeout, sealed test, and carry-state
mechanism studies reported in `docs/Drc200MCarryMechanismResult.md`
Branch at creation: `rl-ground-up`

This document is both:

1. the pre-registered plan for the next research programme; and
2. the append-only execution log for implementation, training, analysis, and
   decisions made while following it.

Keep the plan sections stable once execution begins. Record deviations in the
execution log before acting on them, with the evidence and reasoning that
motivated the change. Never rewrite earlier log entries to make the history
look cleaner.

---

## 1. Executive summary

The latest P1 experiment asked whether a feed-forward convolutional policy
could learn Ricochet Robots slide dynamics without being told where each move
would stop. On 12x12 boards it reached 69.2% deterministic sealed-test
success, but plateaued after a fast initial climb. Linear probes found no
clear, linearly decodable representation of medium- or long-range landing
cells after 181M environment steps.

The next experiment will change the policy to a faithful DRC(3,3): a
convolutional recurrent network which performs repeated spatial computation
within each environment step and carries recurrent state between steps.
Everything possible about the P1 task will remain fixed. In particular, the
DRC will initially receive the same structured wall/goal/robot/time channels,
with no stop-cell or distance planes.

This structured-input experiment comes before an RGB experiment because it
isolates the principal scientific question:

> Does repeated recurrent spatial computation learn the missing slide-dynamics
> representation more effectively than the feed-forward residual CNN?

Changing both the architecture and the observation representation at once
would make either success or failure ambiguous. If the structured-input DRC
establishes a viable learning recipe, the planned follow-up will move that DRC
to corrected-renderer RGB observations without stop/distance hints, bringing
the work closer to the pixel-based Sokoban DRC setting.

The initial structured-input sweep is **60M steps**, not a short 25-30M pilot.
There will be no performance-based early stop before 60M. P1 showed long
plateaus and reached its best validation checkpoint only around 112M, so the
design must leave room for delayed learning. Stops before 60M are permitted
only for invalid science, numerical instability, broken logging/checkpointing,
or a machine/operational failure that requires repair and resume.

---

## 2. Terminology

The previous notes call P1 "plane-free." That phrase means **helper-plane
free**, not literally free of all planes.

### Structured helper-free observation

The primary DRC experiment receives:

- four wall channels;
- one goal channel;
- one channel per robot, with canonical target-first identity; and
- the remaining-time channel.

It does **not** receive:

- stop-cell planes telling it where each action lands; or
- the relaxed-distance plane used by the reward shaper.

The model must therefore infer slide endpoints from walls and robot
occupancy. Reward shaping may use the relaxed distance internally, but that
map is not part of the policy observation.

### RGB helper-free observation

The follow-up pixel experiment will receive the corrected-renderer RGB board
and no stop-cell or distance planes. Its treatment of the time signal must be
fixed and documented before its first learning run; the preferred
Sokoban-aligned condition is raw RGB with no extra observation channel.
Action masking, shaping, and the reverse curriculum remain training/protocol
features rather than visual hints about slide endpoints.

### Repeated computation versus temporal memory

DRC recurrence has two potentially useful effects:

- multiple computation ticks over the same current observation; and
- hidden state carried between environment steps.

The probe design will distinguish these where practical by testing both
zero-state single-observation inference and carried-state trajectory
inference.

---

## 3. Research questions and hypotheses

### Primary question

On the same 12x12 helper-free task as P1, does DRC(3,3) learn slide dynamics
more effectively than the feed-forward residual CNN?

### Secondary questions

1. Does a landing-cell representation emerge inside the DRC encoder or
   recurrent hidden state?
2. Does decodability increase across DRC ticks, consistent with iterative
   spatial computation?
3. If DRC improves task performance, are the gains concentrated in the
   expected hard strata: long optimal solutions and puzzles requiring two or
   three robots to move?
4. Does representation emergence precede, coincide with, or follow changes
   in task success?
5. After a structured-input recipe is understood, can the same DRC learn from
   RGB pixels without stop/distance hints?

### Pre-registered hypotheses

- **H1 — representation:** trained DRC recurrent states will show a sustained
  positive landing-cell decoding gain over matched random initialization,
  especially for mid-length (4-7 cell) and long (8+ cell) slides.
- **H2 — iterative computation:** landing-cell decodability will improve from
  the encoder baseline through successive DRC ticks, rather than being
  entirely present before recurrence.
- **H3 — task learning:** by matched environment-step budgets, DRC will reach
  higher validation success, reach P1-equivalent success earlier, or avoid
  P1's long 66-73% plateau.
- **H4 — hard-stratum specificity:** a real slide-dynamics improvement will
  be visible most clearly on optimal lengths 7-8 and puzzles requiring
  multiple robots, not merely on length-1 target-only moves.

Evidence against the hypothesis is also useful. A DRC that matches P1's
plateau while its hidden-state probes remain flat would show that recurrence
alone does not cheaply induce the desired circuit under this RL signal.

---

## 4. Fixed comparator: P1

Primary comparator artifacts:

- config: `configs/groundup_p1_12x12_planefree.yaml`;
- run: `artifacts/groundup/p1_12x12_planefree/`;
- best validation checkpoint: approximately 112M steps;
- sealed-test result from that checkpoint: 69.2% deterministic / 68.3%
  stochastic;
- sealed-test by optimal length: 100/92/90/82/65/57/47/22% for L1-L8;
- sealed-test by robots moved: 84/49/12% for one/two/three robots;
- random and target-only baselines: 14.3% and 26.0%;
- random-initialization probe control: 14.7% overall;
- trained P1 mid-slide decoding: never above approximately 6%;
- trained P1 long-slide decoding: approximately 2-4% throughout.

Useful deterministic-validation reference points from the fixed 200-puzzle
suite are:

| Nominal budget | P1 checkpoint at or before budget | Validation success |
| ---: | ---: | ---: |
| 5M | 4.00M | 35.0% |
| 10M | 9.00M | 48.0% |
| 15M | 14.00M | 51.5% |
| 20M | 19.00M | 57.5% |
| 30M | 29.00M | 62.5% |
| 40M | 39.00M | 61.5% |
| 50M | 49.00M | 65.0% |
| 60M | 59.00M | 66.5% |
| 75M | 74.00M | 67.0% |
| 100M | 99.00M | 64.0% |
| 112M | 111.00M | 71.5% |
| 150M | 149.00M | 69.0% |
| 177M | 176.01M | 67.0% |

These individual validation measurements are noisy. Compare full curves,
fixed-suite re-evaluations, hard-stratum breakdowns, and probe trajectories;
do not over-interpret a one-checkpoint lead.

---

## 5. Primary experimental contract

The primary run is a DRC counterpart to P1.

### Environment and data held fixed

- Bank:
  `artifacts/groundup_bank_12x12/size-12x12_numR3_ET2_CL2.parquet`
- Board: 12x12
- Robots: 3
- Optimal lengths: 1-8
- Split seeds: the existing `20260710` / `20260711` discipline
- Horizon: 24
- Step penalty: -0.01
- Goal reward: +1.0
- Potential-shaping coefficient: 0.1
- Discount: 0.99
- Canonical target-first identity: on
- Valid-action masking: on
- Reverse-frontier curriculum: on
- Remaining-time observation: on
- Stop planes: off
- Distance plane: off
- Seed: 11 for this exploratory comparison
- Validation and sealed-test selection discipline: unchanged

The stored solver plans may choose reverse-curriculum start states. They must
not supervise policy actions.

### Architecture

- Faithful DRC(3,3)
- Three recurrent modules, three ticks
- 32 hidden channels
- 3x3 ConvLSTM kernels
- Structured/tile encoder geometry: two 4x4 SAME, stride-1 convolutions,
  matching the already implemented faithful tile-observation DRC variant
- Paper-faithful flatten-to-MLP readout for the primary run

Do not add the pointer readout, an auxiliary landing-cell loss, or supplied
stop planes to the primary run. Those are later interventions and would
change the question.

### Training

Use the hardened recurrent masked-PPO trainer:

- truncation bootstrapping;
- recurrent sequence replay with state resets;
- advantage standard-deviation floor and clipping;
- correct PPO2 value-loss semantics, with value clipping off by default;
- keep-best validation checkpoint;
- target-KL telemetry/guard;
- linear learning-rate schedule.

Rollout geometry and number of environments will be chosen by a pre-run
throughput benchmark on the real 12x12 training factory. Candidate settings
must preserve whole recurrent sequences in each minibatch. Record every
candidate, throughput, memory use, and the chosen configuration before
training. Performance tuning must not alter the environment or scientific
condition.

### Evaluation

- Periodic validation every 1M steps
- Deterministic validation on the same fixed stratified suite as P1
- Model snapshot every 5M steps
- Best checkpoint selected using validation only
- Sealed test evaluated once, after the run/extension decision is complete
- Final deterministic and three-episode stochastic evaluation
- Per-length and per-robots-moved breakdowns
- Random and target-only baselines under the identical action-masked protocol

The sealed test must not be inspected at the 60M review if training may
continue.

---

## 6. Engineering prerequisites

Long training must not start until all prerequisites pass.

### WP1 — Crash-safe DRC checkpoints and resume

Add periodic training-state checkpoints and `resume_from` support to
`src/groundup/drc_train.py`.

A resumable checkpoint must include enough state to continue without silently
restarting the optimization or curriculum:

- network weights;
- optimizer state;
- total environment timesteps and learning-rate progress;
- sampler frontier EMA/count state;
- relevant random-number-generator states;
- recurrent rollout state and episode-start flags where required;
- environment state or a documented iteration-boundary reset strategy;
- config and observation/action contracts.

Acceptance:

- an interrupted tiny deterministic run and an uninterrupted run produce the
  same next update and metrics, within the normal deterministic tolerances;
- resumed timestep, evaluation, and snapshot schedules do not restart at
  zero;
- curve logs append without overwriting pre-crash history;
- incompatible config or observation contracts fail clearly.

Also retain lightweight weight-only snapshots for probes. Training-state
checkpoints and analysis snapshots should have unambiguous names.

### WP2 — Periodic immutable snapshots

Add `checkpoint_every`, set to 5M for the primary run. Never overwrite these
snapshots. Continue to maintain `model_latest`, `model_best`, and
`model_final`.

Each snapshot metadata record must include:

- exact timesteps;
- Git commit and dirty state;
- config fingerprint;
- bank/split fingerprint;
- policy parameter count;
- wall-clock time and effective steps/second.

### WP3 — DRC landing-cell probes

Extend or complement `src/groundup/probe_landing.py` with a DRC-compatible
probe path.

Use the same fixed state-generation protocol as P1:

- validation split;
- 16,000 states;
- seeded puzzle/start selection;
- 0-6 random valid moves before capture;
- all robot-direction landing-cell labels;
- blocked, short (1-3), mid (4-7), and long (8+) buckets;
- a held-out 25% probe-test partition;
- a matched random-initialization control.

Save the generated probe dataset or its complete fingerprint so every
checkpoint sees exactly the same examples.

Primary probe conditions:

1. encoder output before recurrent computation;
2. final-depth hidden map after tick 1;
3. final-depth hidden map after tick 2;
4. final-depth hidden map after tick 3;
5. final map used by the faithful skip/readout, if distinct.

The primary condition starts recurrent state at zero for every sampled board,
making it comparable to the feed-forward P1 probe and testing within-step
repeated computation. A secondary carried-state probe may replay the sampled
trajectory and preserve DRC state; label it separately and never substitute
it for the zero-state headline.

Use a spatially local linear probe analogous to P1's 1x1 convolution wherever
feature and label grids align. If encoder stride or feature geometry changes,
document the label projection/readout exactly and test it on synthetic
features. Probes measure linear decodability, not causal use.

Acceptance:

- checkpoint load and architecture reconstruction are tested;
- random-init controls are reproducible;
- tick-specific feature extraction does not alter normal policy outputs;
- a probe smoke test learns a synthetic linearly encoded landing label;
- output JSON has stable schema and feeds the existing/new curve collator.

### WP4 — Probe collation and monitoring

Produce, and update after every 5M snapshot:

- machine-readable per-checkpoint JSON;
- a table of overall/bucket/tick results;
- task-success and probe-decoding curves on a shared timestep axis;
- random-init reference lines;
- optional W&B synchronization.

The combined chart should make delayed representation emergence visible
without requiring manual comparison across dozens of files.

### WP5 — Configuration and regression coverage

Add a dedicated YAML config, expected name:

`configs/groundup_drc33_p1_12x12_planefree.yaml`

Before launch:

- run the complete repository test suite;
- verify stored optimal plans replay correctly on sampled bank puzzles;
- verify train/validation/test layout hashes are disjoint;
- verify no stop/distance channels appear in the observation;
- verify action masks match actual movement;
- record observation shape and channel semantics;
- perform a short end-to-end checkpoint/resume/probe smoke test.

---

## 7. Execution stages and decision rules

### Stage A — Implementation and correctness

Complete WP1-WP5. No research run begins while tests or provenance checks are
red.

### Stage B — Real-factory throughput selection

Benchmark plausible recurrent rollout geometries on the actual 12x12 bank
environment, not a synthetic Gym environment. Run enough updates to include
backpropagation through time.

Choose the fastest stable setting that:

- fits memory with headroom;
- preserves the intended DRC(3,3);
- uses whole recurrent sequences per minibatch;
- does not change sample semantics;
- produces healthy optimizer telemetry.

Expected order of magnitude from earlier DRC runs is 800-1,000 steps/second,
but this must be measured rather than assumed. At that rate:

- 60M steps requires approximately 17-21 awake hours;
- 120M requires approximately 33-42 awake hours;
- 180M requires approximately 50-63 awake hours.

Launch long jobs detached and protected against machine sleep. Record the
exact command and process/log locations.

### Stage C — Mandatory 60M initial sweep

Train from zero to at least 60M environment steps.

- Evaluate every 1M.
- Snapshot and probe every 5M.
- Do not stop for a performance plateau before 60M.
- Do not inspect the sealed test.
- Diagnose numerical instability immediately, but distinguish a trainer bug
  from a negative research result.

At 60M, produce a formal review containing:

- DRC and P1 validation curves through 60M;
- fixed-suite deterministic and stochastic re-evaluation of the
  validation-selected DRC checkpoint;
- per-length and per-robots-moved results;
- curriculum frontier trajectories;
- encoder/tick-wise landing-cell probe curves;
- trained-minus-random-control probe gains;
- optimizer health, throughput, and any resume incidents.

Reference: P1 was 66.5% deterministic validation at its checkpoint just below
60M.

### Stage D — Extension decision at 60M

The purpose of this review is allocation, not declaring a final scientific
result.

**Extend to at least 120M** if any of the following holds:

- DRC matches or exceeds P1's task curve at comparable steps;
- validation or hard-stratum performance is still materially improving;
- mid/long landing-cell decodability shows sustained growth over several
  checkpoints;
- later ticks show increasing decodability over the encoder/tick-1 baseline;
- results are ambiguous but still compatible with a delayed learning phase.

**Stop after the 60M review** only for an unequivocal negative or invalid run,
for example:

- task performance remains far below the P1 curve with no upward trend;
- curriculum progression is stuck well before the P1 frontier;
- mid/long probe results remain at matched-random levels across the sweep;
- optimizer/trainer pathology invalidates the trajectory and requires a
  corrected rerun.

If a trainer bug is found, do not label the run a negative research result.
Fix it, add a regression test, log the invalidated run, and restart or resume
only when scientifically defensible.

### Stage E — Review at 120M and possible final extension

P1 reached its best validation checkpoint around 112M, so 120M is the first
fair point for judging delayed improvement.

Extend toward 180-200M if task performance, hard-stratum performance, or
probe decodability is still improving, or if the result remains scientifically
ambiguous near the P1 ceiling. Otherwise end training and select the best
checkpoint using validation only.

### Stage F — Sealed test and conclusion

After the decision to stop training:

1. freeze the validation-selected checkpoint;
2. run the sealed deterministic and stochastic evaluations once;
3. compare with P1 overall and by difficulty strata;
4. collate the complete probe trajectory;
5. write the conclusion in this document and update
   `docs/GroundUpRLRedesign.md` plus `WorkInProgress.md`.

### Outcome labels

- **Task-performance success:** DRC produces a credible improvement over
  P1's 69.2% sealed-test result, supported by validation and hard-stratum
  results rather than a single noisy aggregate.
- **Representation success:** mid/long landing-cell decoding is sustainably
  above matched random initialization and P1, preferably increasing across
  DRC ticks.
- **Mechanistic success:** tick-wise or temporal interventions provide
  evidence that the decoded representation is produced or used by recurrent
  computation. Linear decoding alone does not establish this.
- **Useful negative:** a healthy, sufficiently long run matches or trails P1
  and shows no meaningful recurrent landing-cell representation.

Because this is seed 11 only, small differences are exploratory. A promising
headline should receive multi-seed confirmation before a strong general
claim.

---

## 8. Pixel follow-up

The RGB experiment is deliberately second.

### Entry gate

Proceed when the structured DRC run has:

- demonstrated that the implementation and recurrent trainer are healthy;
  and
- yielded a sufficiently interpretable positive, parity, or well-understood
  negative result to justify changing the observation representation.

A positive representation or task result is the strongest reason to proceed.
If structured DRC fails for an understood architectural reason, decide
explicitly whether RGB could plausibly address it before spending the compute.

### Proposed RGB condition

- Same 12x12 bank, splits, horizon, shaping, curriculum, masking, and
  canonical identity
- Corrected 61x61 RGB renderer at 5 pixels/cell
- No stop-cell or distance planes
- Prefer raw RGB only for alignment with the original pixel Sokoban DRC
- Faithful pixel encoder: channels (32,32), kernels (8,4), strides (4,2),
  inter-layer ReLU on
- DRC(3,3) core and faithful readout retained initially
- The structured run's trainer stability settings retained

Before the RGB run, decide and record whether the remaining-time signal is
removed for pure RGB or represented separately. Do not choose after seeing
early RGB learning results.

Run a short RGB correctness/learnability gate and real-factory throughput
benchmark before allocating a long budget. The RGB result is a follow-up
generalization/representation experiment, not the clean causal comparison
with P1.

---

## 9. Risks and interpretation constraints

### The faithful DRC readout may remain the bottleneck

The earlier 16x16 DRC reached 64.8% and struggled particularly on multi-robot
puzzles. Its flatten-to-MLP readout was suspected. Keep that readout for the
primary faithful comparison, but do not mistake readout failure for proof that
the recurrent core learned nothing; the internal probes are important.

A DRC-plus-pointer-head hybrid is a later ablation, not part of this primary
run.

### Linear probes can miss nonlinear or distributed representations

"Not linearly decodable" is narrower than "not represented." State conclusions
accordingly. If task performance improves without linear decodability,
consider nonlinear probes, causal ablations, or readout interventions only
after reporting the primary probe honestly.

### Carried recurrent state complicates comparison

A DRC may distribute knowledge over observation encoding and temporal state.
Zero-state probes preserve comparability with P1; carried-state probes test
how the deployed policy actually operates. Report both without combining
them.

### Architecture and trainer cannot be separated perfectly

The recurrent DRC uses a specialized recurrent PPO trainer and different
rollout geometry. The experiment compares the complete viable DRC training
system against the complete P1 feed-forward system while holding the task and
inputs fixed. Attribute differences cautiously.

### One seed is not a population estimate

Seed 11 is appropriate for exploratory mechanism discovery and direct
continuity with P1. It is insufficient for narrow performance claims.

### Compute and machine interruptions

The 60M initial sweep is likely a long job. Exact resume, immutable snapshots,
detached launch, machine-sleep prevention, and explicit artifact checks are
requirements rather than conveniences.

---

## 10. Expected artifacts

Suggested run directory:

`artifacts/groundup/drc33_p1_12x12_planefree/`

Expected contents:

- resolved `config.json`;
- run/provenance manifest;
- append-only `curve.jsonl`;
- detached stdout/stderr log;
- resumable training-state checkpoints;
- `model_5M.pt`, `model_10M.pt`, ... immutable probe snapshots;
- `model_latest.pt`, `model_best.pt`, `model_final.pt`;
- fixed probe dataset or fingerprinted dataset manifest;
- per-checkpoint probe JSON;
- consolidated probe table and chart;
- combined task/probe curve;
- 60M and later review Markdown/JSON;
- final validation report;
- sealed-test report, created only at the end;
- W&B run identifiers when enabled.

No generated artifact should be committed unless it is intentionally selected
for documentation. Plans, configs, tests, implementation, and concise result
write-ups should be committed.

---

## 11. Execution checklist

### Before implementation

- [x] Confirm this plan and experimental contract.
- [x] Record starting commit and repository status.
- [x] Confirm P1 artifacts and bank fingerprint are readable.

### Engineering

- [x] Implement crash-safe recurrent resume.
- [x] Implement immutable periodic DRC snapshots.
- [x] Implement DRC encoder/tick/hidden-state probes.
- [x] Implement probe collation and shared-axis charts.
- [x] Add structured helper-free DRC config.
- [x] Add/extend tests for every new contract.
- [x] Run complete test suite.

### Preflight

- [x] Validate bank plans and split disjointness.
- [x] Assert observation channels exclude stop/distance planes.
- [x] Run checkpoint/resume/probe smoke test.
- [x] Benchmark real-factory throughput candidates.
- [x] Freeze and record the selected run config.
- [x] Verify detached launch and machine-sleep protection.

### Initial sweep

- [x] Launch seed-11 run.
- [x] Monitor optimizer health and artifacts.
- [x] Probe every 5M snapshot, plus the added dense 1M series from 4M.
- [x] Do not performance-stop before 60M.
- [x] Complete formal 60M review.

### Extension and closeout

- [x] Record 60M extend/stop decision.
- [x] Complete 120M review.
- [x] Decide to extend the unmodified control toward 200M.
- [x] Freeze the final-only checkpoint rule (135M periodic best diagnostic only).
- [x] Run the sealed test once.
- [x] Write the final conclusion and update headline docs.
- [x] Decide the RGB follow-up entry gate is met; require a short gate before a long run.

---

## 12. Append-only execution log

Use this template for each entry:

```markdown
### YYYY-MM-DD HH:MM TZ — Short title

- Status before:
- Action:
- Reason:
- Exact command/config:
- Commit and repository state:
- Result/metrics:
- Artifacts:
- Interpretation:
- Decision and next action:
- Deviations from plan:
```

### 2026-07-23 — Plan created

- Status before: P1 stopped at 181M; best checkpoint approximately 112M;
  69.2% deterministic sealed-test success; no clear mid/long landing-cell
  circuit under the registered linear probe.
- Decision: run a structured helper-free DRC(3,3) comparison before changing
  to RGB.
- Reason: this isolates recurrence/repeated computation as the main variable.
  A pixel-first DRC would change architecture and perception simultaneously.
- Initial sweep: 60M steps, with no performance-based stop before that point.
- Follow-up: move to corrected-renderer RGB after the structured DRC result is
  understood and the pixel entry gate is reviewed.
- Implementation/training performed: none.

### 2026-07-23 — Engineering prerequisites implemented and tested

- Status before: the DRC trainer had keep-best weight checkpoints but no
  resumable optimizer/curriculum/environment state, no immutable analysis
  snapshots, and no DRC-compatible landing-cell probes.
- Action: implemented full recurrent-PPO state serialization, adaptive
  sampler and environment state restoration, immutable model/training
  snapshots, provenance manifests, and separate stop-target versus
  learning-rate-schedule horizons. Added zero-state DRC probes for the encoder,
  tick 1/2/3, and faithful readout map; fixed-dataset fingerprinting; automatic
  collation; and a snapshot-watching probe daemon.
- Reason: the mandatory 60M run must survive machine interruption, preserve a
  200M learning-rate horizon for possible extensions, and expose delayed
  recurrent representation formation every 5M steps.
- Tests: exact CPU interrupted-versus-uninterrupted continuation reproduces
  the next rollout, metrics, weights, and sampler state. Tick tracing
  reconstructs ordinary policy outputs exactly. Synthetic probe and
  collation tests pass. Full suite: 191 passed, 9 intentionally skipped.
- MPS smoke: a 128-step real-bank run produced evaluation, immutable model and
  training snapshots, provenance, final reports, and a resumable
  `training_final.pt`. The first MPS resume attempt exposed a device-mapped CPU
  RNG byte-tensor bug; RNG tensors are now normalized to CPU on restore. Exact
  continuation then resumed at t=128 and completed to t=256. The DRC probe and
  daemon processed all smoke snapshots and wrote the shared-axis chart/table.
- Artifacts:
  `artifacts/groundup/drc33_p1_12x12_smoke/`.
- Research training performed: none; smoke metrics are not scientific results.

### 2026-07-23 — Real-bank preflight passed

- Config: `configs/groundup_drc33_p1_12x12_planefree.yaml`.
- Bank build fingerprint: `68ea3ea6c85ed7eb`; file SHA-256:
  `0103a7b9ca2cf0f0fdba27f66cfd610f5c6954006986d48524c0a66395fb02b5`.
- Splits: 39,713 train / 6,677 validation / 6,610 sealed test; zero
  layout-hash overlap between every pair.
- Correctness: 100/100 sampled stored plans replayed successfully; 360/360
  sampled action-mask checks matched actual movement.
- Observation contract: `(9, 12, 12)` = four walls + goal + three robots +
  time; stop and distance channels absent. Actions: 12. DRC encoded/recurrent
  grid remains 12x12. Policy parameters: 1,650,701.
- Artifact:
  `artifacts/groundup/drc33_p1_12x12_planefree/preflight.json`.
- Decision: preflight gates passed.

### 2026-07-23 — Throughput geometry frozen

- Method: complete collect-plus-recurrent-update measurements on the real
  39,713-puzzle 12x12 training factory, with MPS synchronization around the
  timed region, two warm-up and four measured updates.
- Results:
  - 32 env / 32 steps / batch 256 / 2 epochs: 584 steps/s;
  - 64 / 32 / 512 / 1: 1,337 steps/s;
  - 128 / 32 / 1024 / 1: 1,573 steps/s;
  - 64 / 64 / 1024 / 1: 1,361 steps/s;
  - 256 / 32 / 2048 / 1: **1,742 steps/s**, RSS 2.1 GB;
  - 512 / 32 / 4096 / 1: 1,756 steps/s, RSS 3.3 GB.
- Decision: freeze 256 env / 32 steps / batch 2048 / 1 epoch. It improves
  throughput 10.7% over 128 envs; the extra 0.8% from 512 envs is not worth
  1.2 GB additional RSS and reduced operational headroom.
- Config updated with the frozen geometry. Expected 60M awake training time at
  the measured rate is approximately 9.6 hours before periodic evaluation and
  probe overhead.
- Artifacts:
  `artifacts/groundup/drc33_p1_12x12_planefree/throughput_benchmark.json`,
  `throughput_benchmark_256.json`, and `throughput_benchmark_512.json`.

### 2026-07-23 17:37 BST — Mandatory 60M sweep launched

- Status before: all engineering, test, preflight, and throughput gates green;
  repository clean at commit `1964f7d`.
- Exact config:
  `configs/groundup_drc33_p1_12x12_planefree.yaml`; seed 11; 60M current
  stop target; 200M schedule horizon; frozen 256/32/2048/1 geometry.
- Launch: detached GNU Screen session `drc33_p1_60m`, with
  `caffeinate -is` wrapping the Python trainer. The direct `nohup` attempt was
  immediately reaped by the command-execution host before Python started and
  produced no run state; it is an operational non-run.
- Verified live processes: Screen, `caffeinate`, and Python trainer all
  present. Trainer reported 39,713 train puzzles, fixed 200-puzzle periodic
  validation suite, 480-puzzle final validation suite, MPS device, DRC(3,3),
  and 1,650,701 parameters.
- W&B: project `ricochet-robots-rl`, run
  `groundup-drc33-p1-12x12-planefree`, ID `pupnnpq8`.
- Probe watcher: detached Screen session `drc33_p1_probes`, CPU priority
  reduced with two BLAS/OpenMP threads. It will create the fixed 16,000-state
  validation dataset, random-initialization control, and probe/collation
  outputs as immutable 5M snapshots arrive.
- Logs:
  `artifacts/groundup/drc33_p1_12x12_planefree/train.log` and
  `probe_daemon.log`.
- Sealed test: not loaded by the trainer and not evaluated.
- Next action: monitor first 1M evaluation/checkpoint health; do not
  performance-stop before 60M.

### 2026-07-23 18:17 BST — Probe cadence increased to every 1M evaluation

- Status before: the trainer evaluated and rewrote `model_latest.pt` every
  1M steps, but retained immutable analysis snapshots only every 5M. The
  detached probe worker therefore planned to probe at 5M intervals. The run
  had reached 3,006,464 steps; the 1M and 2M weights had already been
  overwritten, while the 3M weights remained available.
- Decision: retain a separate immutable probe-only copy of each 1M
  `model_latest.pt` checkpoint from 3M onward and run the registered
  zero-state landing-cell probe on every retained evaluation checkpoint.
  Keep the original 5M model/training-state snapshot schedule unchanged.
- Reason: 1M probe resolution makes it easier to distinguish gradual
  representation formation from a late discontinuous change and to align
  tick-wise decodability with task and curriculum transitions. Probe work is
  read-only, runs at reduced CPU priority, and does not affect the MPS
  training path.
- Scientific impact: this is a diagnostic-cadence enhancement, not an
  intervention in the optimizer, environment, curriculum, validation model
  selection, or sealed-test protocol. All 5M checkpoints remain part of the
  original registered series; additional 1M points will be labelled
  explicitly. No conclusion will treat the denser cadence itself as an
  independent experimental condition.
- Limitation: 1M and 2M model states cannot be reconstructed because
  `model_latest.pt` had already advanced. The dense series therefore begins
  at the exact 3M evaluation step.
- Next action: add and test an independent fast snapshot-retention watcher,
  update the probe worker to consume those snapshots without duplicate work,
  preserve the 3M model, and relaunch only the probe-side processes. Do not
  restart or otherwise alter the live trainer.

### 2026-07-23 18:20 BST — Dense probe retention activated at 4M

- Result since the cadence decision: the 4M evaluation landed while the
  retention code was being implemented and tested. It improved validation
  success from 39.0% to 45.5%, so both `model_latest.pt` and
  `model_best.pt` advanced to 4,005,888 steps. The 3M model could therefore
  not be recovered. Contrary to the intended next action above, the dense
  series begins at 4,005,888, not 3,006,464. The 1M-3M task metrics remain in
  `curve.jsonl`, but no representation result will be implied for their
  unavailable weights.
- Implementation: added a lightweight independent watcher that notices a
  completed curve row, verifies the exact step in the atomically written
  `model_latest.pt`, and retains it as
  `probe_model_<exact_steps>.pt`. The probe daemon now consumes registered
  `model_<steps>.pt` and dense `probe_model_<steps>.pt` files, deduplicating
  by exact step and preferring the registered snapshot.
- Verification: six focused DRC trace/probe/retention tests pass; Ruff check
  and format pass. The first retained checkpoint is
  `probe_model_4005888.pt`, with matching embedded timestep metadata.
- Live processes: retention Screen `drc33_p1_probe_snapshots` polls every
  five seconds; probe Screen `drc33_p1_probes` runs the fixed 16,000-state,
  30-epoch zero-state probes on low-priority CPU with two threads. The old
  idle probe worker was stopped before duplicate work. The MPS trainer
  process and Screen were not restarted or signalled.
- Training state at 4M: deterministic validation 45.5%; length 1-8 success
  88/84/64/44/20/24/20/20%; one/two/three-robot success
  66.4/18.8/14.3%; curriculum frontier 3. PPO telemetry remained healthy
  (KL 0.0072, explained variance 0.63, no KL early stop).
- Comparator: P1 was 35.0% at its 4.00M checkpoint, so DRC is 10.5
  percentage points ahead at this boundary. This remains an early,
  200-puzzle validation comparison rather than a conclusion.
- Next action: let the 4M probe finish, confirm the registered 5M immutable
  model and training-state snapshots, then retain and probe each subsequent
  1M evaluation without changing the mandatory 60M stop floor.

### 2026-07-23 18:29 BST — First registered 5M checkpoint verified

- Task trajectory: deterministic validation reached 54.5% at 5,005,312
  steps, up from 21.0/24.5/39.0/45.5% at the preceding 1M boundaries.
  Length 1-8 success was 88/80/84/48/44/52/24/16%; one/two/three-robot
  success was 74.3/30.0/14.3%. The reverse-curriculum frontier advanced to
  4.
- P1 comparison: the registered P1 checkpoint at or before 5M was 35.0%, so
  DRC is 19.5 percentage points ahead on the same fixed 200-puzzle
  validation protocol. This is an early curve comparison, not a sealed-test
  result.
- Optimizer health: learning rate 0.0002937, entropy 0.640, mean/max
  approximate KL 0.0150/0.0281, clip fraction 0.132, explained variance
  0.679, four gradient updates, and no target-KL early stop.
- Registered artifacts: `model_5005312.pt` (6.3 MB) and
  `training_5005312.pt` (54 MB) have matching exact-step, clean-launch Git,
  config, bank, split, observation, action, and parameter metadata.
  `training_latest.pt` records next evaluation 6M, next registered checkpoint
  10M, and best validation 54.5% at 5,005,312. The dense watcher correctly
  avoided creating a duplicate probe-only model because the registered model
  is canonical for this exact step.
- Probe contract: both 4M and 5M used the same 16,000-state validation dataset
  fingerprint
  `e97ef31a7e5c0bf133ee508b4ea8bbb27bf32a6a9a58893d9f41d1df5c48adfc`
  and the same matched random-initialization control.
- Probe result: despite strong task learning, there is not yet evidence for
  the hypothesized trained mid/long landing-cell representation. At 5M,
  encoder/tick-1/tick-2/tick-3/readout mid accuracies were
  1.7/3.0/2.4/2.7/1.9%, versus control 4.7/3.6/3.5/3.3/4.3%. Long accuracies
  were 2.0/5.2/4.1/3.9/2.4%, versus control
  6.2/6.7/7.2/6.9/6.7%. Tick 1 consistently improves long decoding over the
  trained encoder, but not above the architectural random control.
- Interpretation: early task learning and the registered representation
  hypothesis currently diverge. Continue measuring; do not reinterpret
  overall or short-slide decoding gains as mid/long slide-dynamics evidence.
- Decision: all 5M correctness and monitoring gates passed. Continue the
  uninterrupted mandatory 60M sweep, retaining and probing every 1M
  evaluation and preserving the original 5M exact-resume snapshot cadence.

### 2026-07-23 22:50 BST — 6M result and clamshell-sleep pause

- Task result: deterministic validation reached 57.5% at 6,004,736 steps,
  versus P1's 44.0% at 6,000,640. Length 1-8 success was
  92/92/72/60/40/32/24/48%; one/two/three-robot success was
  72.6/37.5/42.9%. The three-robot stratum contains only seven periodic
  puzzles, so 42.9% is 3/7 and should not be treated as a precise estimate.
- Health: learning rate 0.0002925, entropy 0.661, mean/max approximate KL
  0.0033/0.0064, clip fraction 0.036, explained variance 0.674, and no
  target-KL early stop. `probe_model_6004736.pt` was retained with exact
  matching metadata, and the 6M probe completed on the fixed dataset.
- Probe result: mid and long decoding remained below matched random
  initialization at every representation. At tick 1—the strongest long-slide
  point—trained mid/long accuracy was 3.29/5.65%, versus control
  3.60/6.66%. The encoder and readout gaps widened. Strong task learning is
  therefore still not accompanied by the pre-registered linear
  landing-cell signal.
- Operational incident: the 6M curve row and exact `training_latest.pt` were
  flushed at 18:37:32. macOS entered clamshell sleep at 18:37:35 and fully
  woke at 22:40:19 when the lid reopened. The original trainer, retention
  watcher, and probe worker remained present and resumed automatically;
  there was no process restart or checkpoint reload.
- Cause and impact: `caffeinate -is` prevents idle/system sleep while the lid
  is open but cannot override clamshell sleep. This paused computation for
  approximately four hours without advancing environment steps or changing
  optimizer, recurrent, environment, sampler, or RNG state. The machine was
  subsequently connected to AC and charging. The run remains scientifically
  valid.
- Telemetry caveat: trainer `elapsed_s` uses wall time, so all later reported
  aggregate steps/second values include this sleep interval and understate
  active throughput. Use the pre-sleep 1,665 steps/s measurement and
  step-to-step active intervals for compute comparisons; label post-sleep
  aggregate throughput accordingly.
- Decision: continue the same live run. Keep the lid open and AC connected
  during the remaining sweep; no training/config change is warranted.

### 2026-07-23 23:31 BST — 10M milestone

- Progress: the same live process reached 10,002,432 environment steps after
  waking, with no restart or checkpoint reload. Validation at 7M-10M was
  57.0/59.5/61.0/59.5%; the current best is 61.0% at 9,003,008.
- Matched P1 comparison: at 8M/9M/10M, DRC achieved
  59.5/61.0/59.5% versus P1's 44.5/48.0/45.0%, leads of
  15.0/13.0/14.5 percentage points. The early DRC advantage is sustained
  across several fixed-suite evaluations rather than depending on one peak.
- Difficulty at 10M: length 1-8 success was
  96/80/76/76/56/36/24/32%; one/two/three-robot success was
  73.5/43.8/14.3%. The reverse curriculum remained at frontier 4.
- Optimizer health at 10M: learning rate 0.0002875, entropy 0.497, mean/max
  approximate KL 0.0073/0.0214, clip fraction 0.062, explained variance
  0.645, four gradient updates, and no target-KL early stop. Value loss was
  3.15 on this update but was not accompanied by non-finite values, a KL
  spike, explained-variance collapse, or other instability.
- Checkpoints: registered `model_10002432.pt` and
  `training_10002432.pt` have matching exact-step metadata.
  `training_latest.pt` records algorithm step 10,002,432, best validation
  61.0% at 9,003,008, next evaluation 11M, and next registered checkpoint
  15M.
- Dense probes: the worker completed every 1M checkpoint through 9M and was
  processing 10M at this review. Tick-3 trained-minus-random mid/long gains
  from 4M through 9M remained negative:
  mid -1.01/-0.67/-0.57/-0.67/-0.74/-1.08 percentage points and long
  -2.64/-2.96/-2.96/-2.60/-3.26/-3.52 points. There is no registered
  representation success yet.
- Active throughput after wake was approximately 1,448 steps/s from 7M-8M,
  1,302 from 8M-9M, and 1,207 from 9M-10M while the low-priority CPU probe
  worker was active. The downward change will be monitored for thermal or
  resource contention; it does not affect the environment-step comparison.
- Decision: continue unchanged. Task evidence is strongly positive and probe
  evidence is currently negative, making the mandatory longer sweep and
  dense diagnostic trajectory informative.

### 2026-07-24 11:41 BST — Formal 60M review and extension decision

- Completion: the original run stopped normally at 60,006,400 steps and
  wrote final model, exact-resume, validation report, curve, and provenance
  artifacts. All 60 periodic validation rows and all 57 intended probe rows
  from 4M through 60M are present.
- Task result: best periodic validation was 76.5% at 56.001M, 57.008M, and
  59.007M, versus P1's 66.5% near 60M. The validation-selected 56M weights
  achieved 79.58% deterministic and 78.19% stochastic success on the larger
  fixed validation suite. Deterministic one/two/three-robot success was
  93.97/59.55/55.0%.
- Probe result: there was no sustained trained-minus-random mid/long gain at
  the encoder, recurrent ticks, or readout. The isolated best tick-1 gains
  were only +0.04 percentage points and were not sustained; all 60M
  mid/long comparisons were negative. Overall decoding gains were driven by
  blocked and short slides.
- Health: no non-finite values or KL early stops; median approximate KL
  0.0040, maximum 0.0150; final explained variance 0.821. Two clamshell-sleep
  pauses distorted wall-clock throughput but did not advance steps or
  restart the process.
- Sealed test: not evaluated.
- Persistence: created a flat 57-row CSV and SHA-256 manifest locally.
  Backfilled 57 `probe/*` history rows, a table, and a chart to original W&B
  run `pupnnpq8`. Uploaded and remotely verified probe artifact
  `drc33_p1_12x12_planefree-probes:v0` with aliases `60m` and `latest`;
  it contains the fixed dataset, control, consolidated outputs, validation
  context, and all checkpoint JSONs.
- Interpretation: task-performance success at 60M; registered representation
  hypothesis not supported. Recurrent benefit may rely on nonlinear,
  distributed, action-specific, or carried-state computation not captured by
  the local zero-state linear probe.
- Decision: **extend the same run to 120M**. This is required because DRC
  exceeds P1 at matched steps and continued to improve late in the sweep.
  Preserve the 200M schedule horizon and validation-only selection; keep the
  sealed test unopened.
- Full review: `docs/DrcPlaneFree60MReview.md` and
  `docs/DrcPlaneFree60MReview.json`.

### 2026-07-24 11:45 BST — Exact continuation launched to 120M

- Status before: 60M review complete; repository clean at commit `fede440`;
  immutable exact-state checkpoint `training_60006400.pt` verified at
  algorithm step 60,006,400 with matching scientific contract.
- Config: `total_steps: 120000000`,
  `schedule_total_steps: 200000000`, and `resume_from` set to the immutable
  60M checkpoint. Environment, model, optimizer, curriculum, rollout, probe,
  validation, and checkpoint settings are otherwise unchanged.
- Training launch: detached Screen `drc33_p1_120m`, wrapped in
  `caffeinate -is`. `WANDB_RUN_ID=pupnnpq8` and `WANDB_RESUME=must` append
  the continuation to the original W&B run rather than creating a fork.
- Trainer verification: exact resume reported t=60,006,400; 39,713 training
  puzzles; fixed 200/480 periodic/final validation suites; MPS DRC(3,3);
  1,650,701 parameters. Run history records clean commit `fede440`, unchanged
  scientific contract fingerprint
  `af7aae4967d56957deedb11647b46045164efa437e51e885f789873e92611719`,
  next evaluation 61M, and next registered snapshot 65M.
- Probe launch: detached target-aware Screens
  `drc33_p1_probe_snapshots_120m` and `drc33_p1_probes_120m`. They ignore the
  old 60M final marker, retain/probe every new 1M evaluation, update the local
  CSV/manifest/chart/table, and exit only after the 120M curve target and
  pending probes are complete.
- W&B verification: original run `pupnnpq8` is remotely `running`; the
  verified 60M probe artifact remains attached.
- Sealed test: still unopened.
- Operational note: launch was on battery at 93%. `caffeinate` protects
  against idle sleep but not clamshell sleep; keep the lid open and prefer AC
  power to avoid a harmless but time-consuming pause.
- Next action: verify the first resumed 61M curve row, dense snapshot, probe,
  schedule, and W&B history before settling into 120M monitoring.

### 2026-07-24 13:55 BST — Live probe W&B persistence repaired

- Audit: dense probe snapshots, checkpoint JSONs, CSV, manifest, Markdown
  table, and PNG had continued updating locally every 1M steps, reaching 68
  checkpoints through t=71,000,064. The original probe daemon had no live
  W&B writer; the 4–60M metrics and artifact on training run `pupnnpq8` were
  a one-off review backfill, so the remote probe plots had correctly stopped
  at 60M even while local probing continued.
- Fix: added an incremental, restart-safe W&B writer to the probe daemon.
  It records only previously unsynced checkpoints, refreshes the complete
  trajectory chart and table after each new 1M probe, and versions the
  checksum-backed probe artifact every 5M plus at final shutdown. Local
  `.drc_probe_wandb_state.json` and the remote run summary both record the
  last successful metric and artifact steps.
- W&B topology: because the active trainer was not originally initialized
  in W&B shared mode, a concurrent writer to that same run would be unsafe.
  Live probes therefore use dedicated run
  `groundup-drc33-p1-12x12-planefree-probes` (ID `bkvhg71c`), grouped under
  parent training run `pupnnpq8`. The final review will also consolidate the
  complete probe history back into the training run.
- Remote verification: run `bkvhg71c` is `running`; summary reports 68
  checkpoints and `probe/last_timesteps=71,000,064`; refreshed chart/table
  are logged; artifact
  `drc33_p1_12x12_planefree-probes:v1` has aliases `71m` and `latest` and
  metadata through 71,000,064.
- Validation: probe suite 7/7, DRC exact-resume suite 19/19, Ruff clean.
  Only the CPU probe daemon was restarted; GPU training, exact continuation,
  and the dense snapshot watcher were untouched.

### 2026-07-24 17:12 BST — Battery pause at 87M

- User-requested pause: stopped the MPS trainer, dense-snapshot watcher, and
  CPU probe workers; their Screen sessions and the `caffeinate` assertion are
  gone. Process audit found no remaining experiment compute.
- Exact restart point: `training_latest.pt` and immutable hard link
  `training_pause_87007232.pt` contain the full algorithm, optimizer,
  recurrent rollout, environment, curriculum, and RNG state at
  t=87,007,232. The curve also ends at t=87,007,232 (87 rows). Checkpoint
  SHA-256:
  `91c8e8ae197f027c7da457c98265d66cc8074ab8324c83051a723169b23c691e`.
- Last validation before pause: 77.0% deterministic on the fixed 200-puzzle
  suite. The 80–86M window remained 78.5–80.5%; no sealed-test evaluation
  was performed.
- Probe state: local probes and the live probe W&B run are synchronized
  through t=87,007,232. Both W&B clients flushed during shutdown.
- Process hygiene: the audit also found the pre-W&B probe daemon orphaned
  from the 13:55 restart. It had raced the replacement daemon on some
  already-deterministic probe checkpoints but did not create a second result
  file or change the fixed dataset. Both copies are now stopped; resume must
  launch exactly one probe daemon.
- Resume preparation: the config now points `resume_from` at
  `training_pause_87007232.pt`; the 120M target, 200M learning-rate schedule
  horizon, scientific contract, validation suites, and W&B run IDs remain
  unchanged. On request, relaunch the trainer with `WANDB_RUN_ID=pupnnpq8`
  and `WANDB_RESUME=must`, then one snapshot watcher and one probe daemon
  attached to probe run `bkvhg71c`; verify the exact t=87,007,232 load before
  leaving it unattended.

### 2026-07-24 21:10 BST — Exact continuation resumed from 87M

- Preflight: AC power connected, battery charged to 100%; repository clean at
  commit `96fcbff`; immutable pause checkpoint reloaded and verified at
  t=87,007,232 with scientific contract fingerprint
  `af7aae4967d56957deedb11647b46045164efa437e51e885f789873e92611719`.
- Trainer: detached Screen `drc33_p1_120m`, protected by `caffeinate -is`;
  MPS DRC(3,3), 1,650,701 parameters, 39,713-puzzle training split, fixed
  200/480 validation suites. W&B training run `pupnnpq8` resumed with
  `WANDB_RESUME=must`.
- Probes: launched exactly one dense-snapshot watcher and one CPU probe
  daemon. Probe run `bkvhg71c` resumed from its persisted 87M cursor, so it
  will append only new checkpoints rather than backfilling duplicates.
- Scope remains unchanged: continue to 120M under the preserved 200M
  learning-rate schedule horizon; validation-only selection; sealed test
  remains unopened. Next operational check is the first post-resume 88M
  evaluation, retained snapshot, local probe, and W&B append.

### 2026-07-24 22:56 BST — Value divergence diagnosed; trainer frozen at 91M

- Trigger: user noticed a sharp W&B performance fall after the 87M battery
  pause. Local artifacts confirmed a real collapse rather than a plotting
  issue: deterministic validation was 75.5% at 88M, then 36.0%, 31.5%, and
  27.5% at 89–91M. Curriculum frontier fell from 6 to 2 as a consequence.
- Immediate containment: sent `SIGSTOP` to the trainer after the exact
  t=91,004,928 checkpoint was atomically written. The trainer is frozen and
  consumes no compute while the already-saved probes finish; it has not been
  allowed to compound the failure.
- Resume audit passed. `training_pause_87007232.pt` and
  `probe_model_87007232.pt` have bit-identical model weights. Adam step
  advanced continuously from 42,480 at 87M to 43,941 at 90M and 44,429 at
  91M; learning rate followed the registered 200M horizon
  (0.00019124, 0.00018749, 0.00018624). Parameters and optimizer moments are
  finite, the 87M checkpoint contract is unchanged, and no W&B or runtime
  errors occurred. This is not evidence of an exact-resume failure.
- Pre-pause warning: value loss was 0.037 at 80M, then 0.306/0.711/1.718 at
  84/85/86M and **19.18 at 87M before the pause**. It remained 16.01 at 88M
  and jumped to 335.64 at the 89M task collapse, then 272.71/313.00 at
  90/91M. Mean/max KL rose but stayed below the configured early-stop
  threshold. The critic-driven gradients passed through the shared DRC trunk
  and damaged policy behavior.
- Diagnosis: genuine value-function divergence under the registered primary
  config, which deliberately had `clip_range_vf: null`. The timing next to
  the restart is coincidental; the exponential instability was already
  established by 84–87M. Landing probes did not form the hypothesized
  mid/long circuit during the collapse.
- Preserved recovery points: validation-selected model remains the 79M
  checkpoint at 83.0%. The latest exact training checkpoint before the
  divergence is `training_80003072.pt` (80.0% validation, value loss 0.037).
  The 85M checkpoint is not a clean rescue point because value loss had
  already risen to 0.711.
- Decision required: do not continue the corrupted 91M trajectory. Either
  close the primary run as a numerical-instability result, or launch a
  separately labelled rescue arm from the exact 80M state with the already
  implemented PPO2 value-loss stabilization (`clip_range_vf: 0.2`). The
  latter changes the optimizer contract and must not be represented as an
  uninterrupted primary continuation. Sealed test remains unopened.

## 2026-07-25 — Independent adversarial investigation of the 87M restart

Commissioned by `docs/Drc87MRestartCollapseInvestigationBrief.md`, which
explicitly did not accept the "coincidental" verdict recorded above. Full
report: **`docs/Drc87MRestartCollapseInvestigation.md`**. Forensic outputs in
`artifacts/forensics/drc87m_restart/`. Sealed test remains unopened; the frozen
trainer (PID 86950) was not signalled.

The "coincidental" verdict is **upheld, and upgraded from an inference to a
measurement.** The restart was not merely coincidental — it was bit-exactly
inert.

- **The restart is provably not causal.** Splitting a 20-iteration run from the
  87M pause checkpoint into 10+10 across two *separate processes* reproduces the
  continuous single-process run bit-for-bit: 1,520 field-value comparisons
  (net `state_dict`, full Adam state, `obs`, `episode_start`, recurrent h/c, all
  256 env PCG64 states, sampler EMA/counts, minibatch generator, CPU and MPS
  RNG, every PPO metric), **0 differences**. There is no first divergent state
  to report because there is no divergence.
- **The loop is bit-deterministic on MPS here.** Five independent processes
  resumed from `training_pause_87007232.pt` all produced iteration-1
  `net=a4f19a65d297fd28`, `value_loss=19.8086`.
- **The harness was validated against production.** Replaying the whole
  87M→88M window (122 iterations) in a fresh process lands on a `state_dict`
  bit-identical to `probe_model_88006656.pt` (`b37f47e638b35fb0`), with
  `value_loss` 16.011714935302734, `policy_loss`, `entropy`, `approx_kl`,
  `approx_kl_max`, `clip_fraction`, `explained_variance` and `lr` all matching
  the recorded 88M row to the last digit. So the production resume at 21:10 was
  itself an exact restore, and the resumed 87→91M trajectory **is** the
  uninterrupted counterfactual. The proposed 80M Arm U / Arm R replay is
  therefore unnecessary.
- **Why the earlier RNG audit was even stronger than it looked.**
  `evaluate_drc` calls `torch.manual_seed(0)`, which reseeds *both* the CPU and
  the MPS generator, and every checkpoint is written immediately after an
  evaluation. So the stored `torch_rng_state`/`device_rng_state` are the same
  constant in all five checkpoints (`1ccf17250133dec5` / `c29a1d9c25f640eb`) and
  provably equal to what the uninterrupted process held. Incidental defect worth
  fixing: this re-anchors the exploration noise stream every `eval_every` steps.
- **Not an evaluation artefact.** 15/15 retained snapshots (65M–91M) re-evaluate
  to *exactly* the recorded `val_det` on the fixed 200-puzzle suite
  (fingerprint verified), and the training-side curriculum EMAs collapsed too.
- **Mechanism, now measured rather than asserted.** `max_grad_norm=0.5`
  renormalises the *combined* gradient, so the exploding value term crowds the
  policy out. Policy-gradient share of the update direction and its effective
  post-clip magnitude: 98.8% / 0.494 at 80M → 64.1% / 0.320 at 85M → **3.7% /
  0.0184 at 87M** → 0.14% / 0.00068 at 90M. `‖g_policy‖` itself never weakens
  (1.72 → 3.06); only its share of the budget collapses. This explains both the
  `val_det` cliff and why it waited until 89M.
- **Better leading indicators than `value_loss`.** `value_loss` oscillates
  between 13.7 and 71.3 across 122 consecutive iterations, so 1M-spaced point
  samples are a poor instrument. `V_std`/`V_mean` on the fixed 16k-state probe
  set are deterministic functions of the weights and move smoothly: `V_mean`
  +0.97 (80M) → +0.21 (86M) → **−2.66 (87M, sign already wrong)** → −95.8
  (91M). A pre-pause-only fit of `log10(V_std)` on 82–87M (0.250 decades/1M,
  R²=0.952) predicts the post-pause scale within 2.6× throughout and within 10%
  by 91M — the collapse magnitude was already determined before the pause.
- **Operational finding.** `caffeinate -is` does **not** prevent clamshell
  sleep: the machine slept 16:37:45→21:08:23 (battery 5%→1%). This is why the
  pre-pause process never reached 88M. It is *not* implicated in the collapse —
  the 88M→89M window ran awake at 9.6 min/1M, identical to the pre-pause
  86M→87M window at 9.8 min/1M.
- **Code identity verified independently.** All 7 relevant source blobs
  (`drc_train`, `drc`, `env`, `curriculum`, `puzzles`, `distance`, `evaluate`)
  are byte-identical at `1964f7d`, `fede440`, `96fcbff`, `2817c10` and `HEAD`.
  The stale-snapshot incident left no residue. The probe subsystem writes no
  trainer input or output.
- **Recovery point confirmed.** `model_best.pt` (t=79,003,648) is bit-identical
  to `probe_model_79003648.pt` *and* to the `best_state_dict` inside all five
  checkpoints, and independently re-evaluates to `val_det` 0.830.
- **Test-coverage gap closed.** The existing exact-resume test could not have
  caught this hypothesis class: it resumed in the *same* process, ran CPU-only
  (so the MPS `device_rng_state` branch — the one that matters, since MPS
  actions come from the device generator — was never executed), checked one
  iteration, never asserted the Adam moments, and seeded both arms identically
  so a failed restore could compare equal by accident. Three tests added in
  `src/tests/test_drc_train.py`, including a genuine `subprocess` restart and an
  MPS-gated `device_rng_state` test. Also fixed a pre-existing red assertion in
  `test_primary_12x12_drc_config_is_helper_free_and_extension_safe` (commit
  `96fcbff` repointed `resume_from` without updating it). 22 passed.

Disposition recommended in the report: close the primary run as a
value-divergence result at 82–89M (not a restart casualty), record the 60–80M
plateau at `val_det` 0.79–0.83 as its actual outcome, keep the sealed test
unopened, and run any rescue from `training_80003072.pt` as a separately
labelled arm. The report recommends an arm that decouples the value and policy
gradient budgets ahead of the `clip_range_vf: 0.2` candidate, because §6
measured the coupling as the proximate cause; both remain scientific arms, not
evidence about the restart.

### Root cause found: the truncation bootstrap reads an untrained observation slice

Addendum to the investigation above (report §6.1). This was listed as unresolved
(U3) in the first draft and is now resolved. **It is a bug in
`drc_train.py`, not a hyperparameter choice.**

- `env.step` increments `_steps` before building the observation, so at horizon
  expiry the returned obs has **time channel exactly 0**. `collect_rollout`
  passes it to the value head and does
  `rewards[trunc_slots] += gamma * boot_values`, then **discards it** —
  `env.reset()` overwrites `self._obs[b]`, so it never enters `buf.obs` and is
  never a regression target. `remaining == 0` appears in **no** stored
  observation; the 87M checkpoint's per-env `steps` histogram spans 0…23, never
  24.
- Because `time_channel: true` puts remaining time *in the observation*, the MDP
  is finite-horizon and fully observed, so **`V(s, remaining=0) ≡ 0`**. Nothing
  pins it there.
- Measured on the fixed 16k probe set with the time channel zeroed, against
  `goal_reward = 1.0`:

  | t | 5M | 20M | 60M | 70M | 80M | 85M | 86M | **87M** | 89M | 91M |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `V(expiry)` mean | −1.87 | −0.85 | −0.34 | +0.35 | **+0.71** | −1.25 | −2.92 | **−12.28** | −91.9 | −139.0 |

  Never near zero, in the entire run, and its magnitude tracks `value_loss`
  throughout (0.45/0.095 at 70M; 12.29/19.18 at 87M). At 80M, **timing out was
  worth +0.70 — nearly as much as solving the puzzle**. By 87M it was a penalty
  twelve times the goal reward. This is why replay rollout returns reach ±150 on
  a task whose true returns are O(1).
- Positive feedback with no contraction guarantee: `V(expiry)` drifts → the drift
  becomes real reward → returns inflate → `V` grows everywhere → `V(expiry)`
  extrapolates further; and via report §6, `max_grad_norm=0.5` on the *combined*
  gradient lets the value term crowd the policy out (share 98.8% → 3.7% → 0.14%).
  A correct bootstrap is safe because the bootstrapped state is also a regression
  target; this one has no anchor.
- **The bootstrap was never in the reference path.** `src/groundup/env.py` emits
  no `TimeLimit.truncated` (zero occurrences), and SB3 bootstraps only when the
  info dict carries both `terminal_observation` and `TimeLimit.truncated`
  (`on_policy_algorithm.py:236–245`). So the feed-forward **P1 baseline correctly
  treats horizon expiry as terminal**; `drc_train.py:586` hand-rolled a bootstrap
  citing SB3 semantics it never actually received. This is a DRC-trainer
  regression against P1, and it means the P1-vs-DRC comparison was not
  like-for-like on reward semantics.

**Fix (recommended rescue Arm C, run before `clip_range_vf`):** delete the
`trunc_slots`/`trunc_obs` collection and the
`rewards[trunc_slots] += self.gamma * boot_values` block in `collect_rollout`.
With `time_channel: true` this is the correct finite-horizon treatment and does
not change the optimal policy. Falsifiable predictions: returns stay O(1),
`value_loss` stays O(0.01–0.1), policy-gradient share stays ~99%.

Cheap orthogonal insurance: Huber/smooth-L1 value loss, so no value error can
consume the whole grad-norm budget. Does not fix the root cause.

**Add to every future run's telemetry** — all three led `val_det` by 5–20M steps
whereas `value_loss` did not: (1) `V` at time-channel 0 on a fixed probe set
(should be ~0; best alarm, nearly free), (2) `V_mean`/`V_std` on that set,
(3) policy-gradient share of the update.

### Truncation-bootstrap bug fixed (test-first) — report §12

- **Failing test first.** `test_no_truncation_bootstrap_when_the_horizon_is_in_the_observation`
  forces the condition deterministically (2-move puzzles, `horizon=1`, so every
  stored step truncates) with the critic pinned to a constant, then asserts the
  buffered reward equals what `env.step` returned. Pre-fix, every truncated step
  had **4.95 injected into a reward of true magnitude ~0.01** — a ~600×
  corruption.
- **Fix** (`src/groundup/drc_train.py`, ~40 lines):
  `RecurrentPPO(bootstrap_on_truncation=None)` auto-detects from the envs'
  `config.time_channel` — bootstrap iff the horizon is *not* observable. Passing
  `True` while `time_channel=True` now raises, so the inconsistent combination
  cannot come back silently. Envs with no `config.time_channel` keep the
  SB3-compatible bootstrap. Resolved value printed at startup and recorded in
  `provenance` → `run_history.jsonl` and every snapshot.
- **Deliberately not a new config field**, which would change the contract
  fingerprint and make every existing checkpoint unresumable — blocking the
  rescue. `af7aae49…` is therefore unchanged, and `training_pause_87007232.pt`
  still restores 16/16 post-fix. Consequence: the fingerprint alone does not
  distinguish pre- from post-fix runs; the git commit in `run_history.jsonl` is
  the authority, and the provenance key records the semantics.
- **Verified from the corrupted 87M checkpoint** (the state that previously ran
  away to `value_loss` 336 by 89M):

  | iteration | 1 | 5 | 10 | 20 | 60 |
  |---|---|---|---|---|---|
  | `value_loss` | 53.93 | 4.81 | **0.163** | 0.137 | **0.115** |
  | `ret_absmax` | 38.9 | 24.1 | **1.25** | 1.52 | **1.59** |
  | `approx_kl` | 0.025 | 0.061 | 0.0074 | 0.042 | **0.0042** |

  Within 10 iterations (82k steps) returns collapse to the true O(1) scale and
  the value loss falls 2.5 orders of magnitude, then holds. **The fix reverses
  the divergence, not just prevents it** — so the run may be salvageable from
  87M, not only from 80M.
- **Gradient budget at re-converged weights** (the §6 prediction): policy share
  3.67% → **65.60%**, effective ‖g_policy‖ 0.0184 → **0.328** (18×). Not yet
  80M's 98.8% because the 87M weights remain mis-scaled and the ConvLSTM stays
  saturated (`c` abs-max 35–48).
- **Transient caveat:** removing the bootstrap drops targets to their true scale
  while `V` still predicts the inflated values, so the *first* iteration's error
  rises before falling (0.035→0.079 at 80M; 19.0→61.1 at 87M; 389→4003 at 90M).
  Iterations 1–5 are the re-fit, not the steady state. A rescue from a late
  corrupted checkpoint will also see `approx_kl` up to ~0.081, above
  `1.5 × target_kl`, briefly tripping the early stop — one more reason to start
  Arm C from the clean 80M state.
- **Similar-error audit.** `drc_train.py:600` was the only place any learned value
  entered the rewards in `src/groundup`. Verified correct and now guarded by
  tests: the GAE `last_values` bootstrap (in-distribution, masked by
  `episode_starts`); `train.py`/SB3 + groundup env (emits neither
  `terminal_observation` nor `TimeLimit.truncated`, both of which SB3 requires, so
  expiry is a real terminal — correct under `time_channel=True`); and legacy
  `src/env/ricochet_env.py` (does set `TimeLimit.truncated`, but its observation
  is clock-free, so its bootstrap is the correct treatment — pairing now pinned by
  `test_timelimit_truncated_is_paired_with_a_hidden_horizon`).
- **Reported but not fixed** (different failure mode, each needs its own change):
  the `evaluate_drc` global reseed; `_dmap_cache`'s `puzzle.seed` key (note that
  keying on `layout_hash` would be *worse* — layouts are shared across goals);
  the CPU↔MPS "exact resume" loophole.
- `pytest src/tests/ -q` → **204 passed, 9 skipped**.

## 2026-07-25 02:21 BST — first attempt scratched; clean run launched from step 0

The 91M trajectory of the first attempt is abandoned, not rescued. Its processes
(trainer PID 86950, snapshot watcher, probe daemon) were SIGKILLed — the trainer
was still SIGSTOPped, and it was killed rather than SIGCONTed so it could not
write anything on the way out. All 220 artifacts of that run are **preserved
untouched** in `artifacts/groundup/drc33_p1_12x12_planefree/`, including
`model_best.pt` (t=79,003,648, val_det 0.830,
sha256 90cf56ea02a5e28c3f6a61d2e037bc707eb2c6ec7ad406d1e6dd591bcc1d511d).

**Why from step 0 rather than Arm C from 80M.** The truncation bootstrap was
active from the first gradient step: `V` at horizon expiry measured −1.87 at 5M
and −4.18 at 10M against a `goal_reward` of 1.0, and was never near 0 anywhere in
the run. So `training_80003072.pt` is not a clean rescue point either — its
weights and Adam moments were shaped by inflated rewards throughout. There is no
uncontaminated checkpoint to resume, which makes a fresh run the only honest
option. It is also the first genuinely like-for-like comparison against the
feed-forward P1 baseline, whose SB3 path never bootstrapped.

Registered run:

- config `configs/groundup_drc33_p1_12x12_planefree.yaml`, `run_name`
  **`drc33_p1_12x12_planefree_v2`**, `resume_from: null`
- config-contract fingerprint **`e7b3bf2186299ec0a75ff408f97eec03e38719c3866e9c171e92391fa3f7d4ee`**
  (was `af7aae49…`; moved deliberately by the new `bootstrap_on_truncation`
  contract field, which now pins the reward semantics)
- `bootstrap_on_truncation: false` set **explicitly**, echoed in the startup
  banner and recorded in `provenance` → `run_history.jsonl` and every snapshot
- everything else unchanged from the registered design: seed 11, 256 envs,
  n_steps 32, batch 2048, 1 epoch, `total_steps` 120M against a 200M LR horizon,
  `clip_range_vf: null`, `target_kl: 0.05`, DRC(3,3), 1,650,701 params,
  obs `(9, 12, 12)`, no stop/distance planes, horizon 24
- W&B run **`yidjviev`** (`groundup-drc33-p1-12x12-planefree-v2`); the dead run
  `pupnnpq8` is left as-is
- bank sha256 `0103a7b9…` re-verified; preflight clean, splits disjoint, 100
  stored plans replayed, 360 mask checks
- **split fingerprints identical to the first attempt** (train
  `86c69a75…`, val `a5029fe4…`, periodic suite `741960337e…`), so curves are
  directly comparable
- launched detached in screen `drc33_p1_v2` under **`caffeinate -dis`** (`-d`
  added per the 2026-07-24 finding that `-is` does not block clamshell sleep),
  plus screen `drc33_p1_v2_snapshots` running the per-1M snapshot watcher — those
  retained snapshots are what made the 87M forensics possible
- entrypoint smoke-tested first at `total_steps=24576` with W&B disabled:
  checkpoints wrote at schema 2 with `device: mps`, provenance carried
  `bootstrap_on_truncation: False`, and `value_loss` fell 1.379 → 0.428 over the
  three iterations

Not launched: the live probe daemon. It needs a probe dataset built against this
run and adds CPU pressure; start it separately if the landing-cell probes are
wanted again.

Watch items, in order of usefulness:

1. `value_loss` should now stay O(0.01–0.1) for the whole run. Post-fix it is a
   meaningful alarm again, because the channel that inflated it is closed.
2. `curriculum/frontier` should climb rather than collapse.
3. `val_det` versus the first attempt at matched steps — the first attempt reached
   0.75–0.83 between 60M and 87M *despite* corrupted rewards, so it is a real
   baseline to beat, not a floor.

Still not implemented: the extra telemetry recommended in report §11.2 (`V_mean`/
`V_std` on a fixed probe set, and the policy-gradient share of the update). The
`V`-at-expiry alarm is now moot, since `V` is no longer evaluated there.

### 2026-08-13 — 200M closeout and recurrent-carry mechanism study complete

- **Status before:** the clean v2 DRC run had reached 120M; the no-aux A0
  control had reached 135M and was still improving. The reporting path could
  inherit and headline a parent branch's periodic best, the test split remained
  sealed, and the existing probes did not follow carry through real rollouts.
- **Reporting repair:** commits `9ccf245` and `84d91f0` make final weights the
  headline, report periodic best separately, bind every result to immutable
  source run/timestep/path/kind/SHA, reject unsafe semantic branches, and add a
  real joint `(optimal_length, robots_moved)` hard suite.
- **Continuation:** resumed exactly from `training_pause_190005248.pt` with no
  contract difference and completed at the legal 200,007,680-step rollout
  boundary. The final source SHA is `8760ba70...`; final validation was 82.08%
  deterministic / 82.29% stochastic and joint-hard validation was 75.96%.
  The separately evaluated 135M periodic best scored 79.79% / 80.83% / 74.89%
  on those same suites, so max-selection would have chosen the worse policy.
- **Frozen DRC-line test decision:** before final-report inspection, named only
  the 200M final checkpoint, test seed 17, 60 puzzles per length, and one
  deterministic plus three stochastic episodes. The 135M best was never
  tested. This split was held out from DRC training/selection; exact
  reproduction of P1's scripted-baseline aggregates strongly evidences the
  same selection was used for P1, whose legacy result predates suite
  fingerprints and informed this programme. Suite
  fingerprint: `9cf385d2...`; scientific result hash: `85d351c2...`.
- **Sealed result:** DRC 80.42% deterministic / 81.25% stochastic versus the
  frozen feed-forward P1 artifact's 69.17% / 68.33%, gains of +11.25/+12.92
  points. Deterministic gains increase through the hard tail: +15 points at
  length 5, +18.33 at length 7, +30 at length 8; +17.26 points on two-robot
  and +29.42 on the small 17-puzzle three-robot stratum.
- **Observational study:** commits `d78aa83` and `97e606a` add exact BFS labels,
  real-rollout `h/c` capture, independent linear/MLP target probes, leak-free
  layout splitting, and matched random-network/shuffled-label controls. Four
  checkpoints (9M/59M/135M/200M) show strong action/history decoding and
  learned nonlinear distance/optimal-action information, but only weak
  landing-cell/shortest-path-progress structure. Final result fingerprint:
  `37ef57e0...`.
- **Causal study:** commits `e6c85b9` and `e3f91c6` add continuous/depth/
  component/timing resets and exact-state matched donor patching. Full carry
  reset every decision costs 32--38 points at every checkpoint; final effect
  -38.22 points, paired 95% CI [-46.50, -29.94]. One-off resets are small.
  At 200M, coherent donor carry changes actions in 39/68 exact reconvergences
  but has 0.00-point mean suffix-success effect, CI [-7.35, +7.35]. All donor
  positive controls pass.
- **Interpretation:** task-performance success is achieved for this seed.
  Carry is causally necessary and history-dependent; its resilience to
  one-time zeroing is consistent with reconstruction, redundancy, or local
  bypass. Probes do not support a simple explicit route or slide simulator.
  This is a complete-recipe result, not an architecture-only or multi-seed
  claim.
- **Artifacts and report:** `artifacts/groundup/drc33_p1_aux_A0_control/`,
  `artifacts/groundup/drc_carry_study_v2/`, and
  `docs/Drc200MCarryMechanismResult.md`.
- **Decision:** do not extend seed 11 beyond the registered 200M horizon by
  default. Highest-value next work is multi-seed confirmation plus a
  preregistered carry-recovery-timescale intervention. Treat the known test as
  a fixed replication endpoint and reserve a new globally untouched suite for
  programme-level confirmation. The RGB entry gate is met, but begin with a
  correctness/throughput/short-learnability gate rather than a full long run.
