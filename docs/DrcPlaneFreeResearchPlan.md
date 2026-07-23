# DRC Helper-Free Slide-Dynamics Research Plan and Execution Log

Date created: 2026-07-23
Status: planned; no implementation or training started
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

- [ ] Confirm this plan and experimental contract.
- [ ] Record starting commit and repository status.
- [ ] Confirm P1 artifacts and bank fingerprint are readable.

### Engineering

- [ ] Implement crash-safe recurrent resume.
- [ ] Implement immutable periodic DRC snapshots.
- [ ] Implement DRC encoder/tick/hidden-state probes.
- [ ] Implement probe collation and shared-axis charts.
- [ ] Add structured helper-free DRC config.
- [ ] Add/extend tests for every new contract.
- [ ] Run complete test suite.

### Preflight

- [ ] Validate bank plans and split disjointness.
- [ ] Assert observation channels exclude stop/distance planes.
- [ ] Run checkpoint/resume/probe smoke test.
- [ ] Benchmark real-factory throughput candidates.
- [ ] Freeze and record the selected run config.
- [ ] Verify detached launch and machine-sleep protection.

### Initial sweep

- [ ] Launch seed-11 run.
- [ ] Monitor optimizer health and artifacts.
- [ ] Probe every 5M snapshot.
- [ ] Do not performance-stop before 60M.
- [ ] Complete formal 60M review.

### Extension and closeout

- [ ] Record 60M extend/stop decision.
- [ ] If extended, complete 120M review.
- [ ] Decide whether to extend toward 180-200M.
- [ ] Freeze validation-selected checkpoint.
- [ ] Run sealed test once.
- [ ] Write final conclusion and update headline docs.
- [ ] Decide whether the RGB follow-up entry gate is met.

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
