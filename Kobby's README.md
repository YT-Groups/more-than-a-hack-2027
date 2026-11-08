# Kobby's Meeting README

## Executive summary

Our current submission is a conservative, stable possession model:

- **302 full-match possessions**
- **Macro-F1: 0.8533**
- **Spell-F1: 0.5059**
- **17 predicted public-window possessions, matching the truth count**
- **46.90% dead-ball share**
- **50 unit tests passing**
- Submission format validation passes

The change from **431 to 302 possessions is a quality improvement because 431 came from a more permissive C2 control detector that produced substantially more fragmented output**. The cleaner C3 pipeline reduced possession fragmentation while preserving or improving public frame-level alignment.

This is not proof that 302 is the exact private target. It is the strongest defensible result we have from public score, label-free match proxies, and football rules.

## What we changed

### 1. Adopted C3 control instead of C2

The earlier C2 candidate considered a player in possession when:

- the nearest player was within range;
- the nearest opponent was not inside the contest radius;
- the candidate survived hysteresis and dwell.

C3 adds a relative-motion veto: the ball must move with the player rather than merely pass near or fly past them.

This was added because ball-player relative speed distinguishes controlled possession from:

- a pass skimming past a player;
- a rebound or deflection;
- a loose ball near a stationary player.

### 2. Added calibrated event boundaries

We added controlled refinement around dead-ball edges:

- visible ball-out or ball-loss events refine onset;
- visible kicks refine restart;
- when no event is visible, restart fallback uses match-calibrated evidence.
- dead-ball runs never span halftime boundaries.

### 3. Added Smart Tagging evidence

Smart Tagging is used only as a weak prior:

- set-piece windows relax the dead-ball detector;
- corner kicks, free kicks, and goal kicks provide restart-team hints;
- tagged windows are never treated as ground truth.
- restart timing from tags varies significantly by event type.

### 4. Improved possession state handling

The state machine:

- starts dead frames with no owner;
- preserves the last owner across uncertain live gaps;
- changes ownership only at confirmed control evidence;
- resets ownership per graded half;
- prevents short possessions from being merged across dead-ball stretches.

### 5. Added measurement tooling

We added:

- half-open dead-run extraction;
- per-half dead-ball summaries;
- overlap checks for tag windows;
- restart-timing classifications;
- public ground-truth edge latency reporting;
- dead-ball proxy metrics in the evaluation kit;
- synthetic tests for runs, overlaps, timing, clipping, control, and restart behavior.
- a targeted restart audit matching set-piece tags to predicted and public ground-truth dead stretches.

## Structural improvements now available

### 1. Dynamic restart alignment

The pipeline now calculates a physical restart candidate independently of fixed-tag timing. For each
coarse restart boundary, it searches the local time window and scores:

- ball acceleration;
- player activity rise;
- ball visibility;
- spatial separation between the nearest home and away players.

This is available through `python run_pipeline.py --audit-architecture`. It is diagnostic only and
does not move the stable submission boundaries.

The first full-match diagnostic run evaluated 140 restart boundaries. It found a median physical
confidence of 0.2883 and a median 1-frame displacement from the current coarse restart edge.
These values are research signals, not measured scoring improvements; no fixed offset or boundary
change is applied.

### 2. Expanded full-match proxies

The evaluation kit now reports possession duration percentiles (`p10`, `p50`, `p90`, and `p95`) in
addition to mean, median, dead share, short-possession share, and home live share. These guardrails
can expose fragmentation, distribution drift, and team-share changes without public truth.

### 3. Weighted evidence fusion

[yt/evidence.py](./yt/evidence.py) adds a reusable confidence fusion function for:

- spatial taker evidence;
- ball-exit evidence;
- Smart Tagging team hints.

The weights are exposed and auditable: spatial 50%, ball exit 35%, team tag 15%. This is not
currently used to change ownership and must be validated before becoming default behavior.

## Restart audit completed

We created a targeted audit for corners, free kicks, goal kicks, and kickoffs. It compares:

- Smart Tagging windows;
- predicted dead-ball stretches from the current submission;
- public ground-truth dead stretches;
- predicted versus truth restart frames;
- predicted restart team versus truth team;
- named-team tagging hints versus truth team.

The audit was run on the current `YT.csv` submission and wrote:

- [Restart audit results](./reports/restart_audit.csv)
- [Restart audit script](./scripts/restart_audit.py)

### What it found

- **33** targeted set-piece tags were audited.
- Public ground truth matched only two targeted tags in the five-minute sample: one goal kick and one free kick.
- Both public matches had the same predicted restart edge:
  - truth restart: frame 18389;
  - predicted restart: frame 18479;
  - latency: **+90 frames / 3.6 seconds**.
- Initial ownership after the truth restart was correct in **2/2** public matches.
- The current restart-team evidence mapping was correct in **2/2** public matches.
- Named-team Smart Tagging hints were correct in **1/2** public matches.

The mismatch sample is too small to support a default pipeline change. In particular, a tag window is coarse and may contain the wrong dead stretch when a tag overlaps multiple events. The current implementation therefore reports whether one match is unambiguous and exports ambiguous evidence without turning it into a model decision.

### What this means

The public restart audit does not yet show that restart timing is definitively wrong. It shows a consistent +3.6-second predicted-edge delay in only two public examples. It also shows that team evidence agrees on those examples, while named-tag hints are not reliable enough alone.

This supports continuing restart research, but not changing production restart logic based on the first audit alone. Any future restart change must improve public edge or transition results and survive label-free match checks, unit tests, and a larger replay audit.

## Scores before and after

| Variant | Macro-F1 | Spell-F1 | Full-match possessions | Assessment |
|---|---:|---:|---:|---|
| C2 control | 0.8502 | 0.5081 | 431 | Too fragmented |
| Initial C3 control | 0.8539 | 0.5059 | 297–299 | Cleaner, but boundary calibration improved it |
| Calibrated C3 pipeline | 0.8533 | 0.5059 | 302 | Current submission |

Important comparison:

- **431 to 302 was not one threshold change and should not be described as a direct A-to-B tuning experiment.**
- It reflects replacing the more permissive C2 detector and applying the calibrated C3 pipeline.
- Public possession count remained accurate: both the truth and current submission predicted 17 spells in the public window.
- The full-match mean possession length increased from approximately 8 seconds under the early C2-style output to approximately 11 seconds under the calibrated pipeline.

## Experiments that did not improve the solution

### Core-edge backdating

This moved dead-ball start and end edges to the first/last frame of detected control.

Result:

- Macro-F1 fell from 0.8533 to 0.8125
- Spell-F1 fell from 0.5059 to 0.4588
- Full-match possessions rose to 309

It was reverted. Fewer guessed switches with the wrong boundaries are better than more switches that disagree with football control.

### Pass-receiver experiment

A pass-flight and receiver detector was tested as an ownership handover signal.

Result:

- Macro-F1 0.8535
- Spell-F1 0.4914
- Possessions rose to 316

It was removed because it lowered spell-F1 and increased fragmentation.

### Conservative handover guard

We added a guard that rejects short backdated handover fragments.

Result:

- Macro-F1 unchanged at 0.8533
- Spell-F1 unchanged at 0.5059
- Possessions unchanged at 302
- Unit tests increased from 39 to 40

It is retained as a correctness safeguard, not presented as a measured public score gain.

### Conservative reception detector

An optional reception-aware pipeline was tested for pass and stable-reception handovers.

Layer-count audit:

- 1172 visible fast-ball flights;
- 925 passed passer-team evidence;
- 173 rejected by dead-ball overlap;
- 232 rejected by unstable or contested proximity;
- 89 rejected by insufficient C3 consistency;
- 445 final candidate handovers before duplicate frames were removed;
- 422 unique handover frames.

Public result:

- Baseline: Macro-F1 0.8533, Spell-F1 0.5059, 302 full-match proxy possessions.
- Reception candidate: Macro-F1 0.8538, Spell-F1 0.5059, 17 public possessions.
- Only 41 live frame labels changed across 21 runs.
- In the public window, 4 changed frames corrected the baseline and 0 regressed it.

The 0.0005 Macro-F1 change is below the repository's stated segment-level noise floor. Relaxing the 10-frame dwell to 3 frames produced 454 events but no public score change. Removing C3 control consistency raised predicted public fragmentation to 18 possessions and lowered Spell-F1 to 0.4914. The receiver experiment remains opt-in and C3 consistency remains mandatory.

## Why 302 should be considered an improvement

### 1. It is much less fragmented

431 possessions corresponds to a high number of ownership switches and short spells. The current 302 is substantially more stable while preserving team ownership through loose or uncertain play.

### 2. It is closer to the evidence-backed range

The public truth contains 17 possessions in a five-minute sample. Scaling that rate to 105.9 graded match minutes gives an approximate target of 360 possessions. This is only an extrapolation, but it is more useful than pass arithmetic:

- 302 is below 360
- 431 is further outside the plausible range
- neither public nor pass totals prove an exact private count

### 3. The public score does not reward more switches blindly

The core-edge experiment demonstrated that increasing the switch count to 309 harmed public results. A larger count is therefore not an improvement by itself.

### 4. It matches the challenge's loose-ball rule

The challenge says contested and loose phases retain the last owner, and only controlled possession transfers ownership. C3 is built around that rule; C2's permissive candidate behavior fragmented possession more aggressively.

## Current remaining weaknesses

1. Possession handovers are too dependent on nearest-player control.
2. A pass, touch, deflection, rebound, and reception may appear similar in raw tracking.
3. Restart ownership is often inferred without a visible ball or takes.
4. Public transition matching is weak:
   - transition F1 at ±12 frames: 0.10
   - transition F1 at ±25 frames: 0.15
5. The public truth is a small sample:
   - 17 possessions
   - bootstrap spell-F1 remains wide
6. Restart evidence is broad:
   - the public audit has only two usable matched restart edges;
   - the two matches both show a +90-frame / 3.6-second delay;
   - named team-tag accuracy is only 1/2 in that sample.

## Our next aims

### Priority 1: Confirm actual handovers

Build a handover event detector that requires evidence from all three layers:

1. ball leaves with sustained speed;
2. a player near the flight endpoint is stable through reception;
3. the receiving team retains proximity for more than one short tracking run.

Do not create a switch for:

- a ball skimming past a player;
- a deflection without subsequent control;
- an aerial duel without control;
- single-frame proximity noise.

### Priority 2: Improve restart ownership

For every detected restart:

- locate the actual first ball movement;
- estimate the nearest team within a confidence-separated radius;
- use first reception after the restart;
- combine tracking evidence with Smart Tagging team hints;
- keep Smart Tagging timing coarse, not exact.

Current restart-audit status:

- the audit has been implemented, run, and added to the repository report set;
- 33 targeted set-piece tags were processed;
- only two public truth matches are currently strong enough to score;
- those two public matches agree with the current ownership evidence and show equal +90-frame timing;
- no restart algorithm change is justified until more public matches or replay evidence are available.

### Priority 3: Stamp boundaries from events

Ownership changes should occur where evidence is:

- visible;
- sustained;
- physically compatible with a ball pass or reception;
- confirmed by at least two tracking frames or one continuous event.

## Target quality range

We should not tune toward a guessed final possession number.

| Signal | Current | Meeting target |
|---|---:|---:|
| Full-match possessions | 302 | Plausible exploratory range: 300–360 |
| Mean possession duration | 11.17 seconds | 11–14 seconds |
| Dead-ball share | 46.90% | Maintain roughly 47% |
| Home live share | 58.93% | Monitor only |
| Short possessions under 1 second | 5.96% | Reduce further without deleting real events |



## What counts as a real improvement

A new variant should be kept only when it shows multiple independent gains:

- public spell-F1 improves;
- public transition timing improves;
- possession count stays inside a plausible range;
- short possessions and fragmentation decrease;
- dead-ball share remains consistent;
- restart ownership agrees more strongly with independent event evidence;
- unit and sanity tests continue passing.

## Current validation status

- Current production submission: 302 full-match possessions.
- Public result: Macro-F1 0.8533, Spell-F1 0.5059.
- Public possession count: 17 predicted, 17 in truth.
- Dead-ball sanity review: 137 dead stretches, 46.90% dead share, no sub-2-second stretches.
- Restart audit: 33 set-piece tags reviewed; public restart evidence limited to two matches.
- Architectural proxy: dynamic restart audit covered 140 boundaries with median confidence 0.2883
  and median 1-frame diagnostic displacement.
- Possession duration proxy: p10 1.37 s, p50 7.70 s, p90 25.80 s, p95 33.04 s.
- Test status: 50 unit tests passing, including dead runs, overlaps, state-machine behavior,
  reception controls, dead-ball checks, loading, and restart-audit matching.

## One-sentence meeting justification

"We moved from a permissive C2 detector that produced 431 fragmented possessions to a calibrated C3 control model with event-refined boundaries that produces 302 stable possessions, with equal public spell-F1 and an improved public macro-F1."
