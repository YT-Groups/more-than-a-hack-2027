# Kobby's Meeting README

## Executive summary

Our current submission is v8, a conservative possession model with restart refinements grounded
in the Laws of the Game:

- **295 full-match proxy possessions**
- **Public Macro-F1: 0.8671**
- **Public Spell-F1: 0.5500**
- **15 public possessions predicted; 10 match truth at IoU 0.5**
- **47.04% dead-ball share**
- **63 unit tests passing**
- Submission format validation passes

The progression from **431 C2 possessions to 302 in v4 and 295 in v8** reflects a less fragmented C3 model plus later restart and tracking corrections. The public scores improved from v4 to v8, but the small public sample does not establish a private-match gain.

This is not proof that 295 is the exact private target. It is the strongest defensible result we have from public score, label-free match proxies, and football rules.

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

The audit was run on the earlier v4 `YT.csv` submission and wrote:

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
- The v4 restart-team evidence mapping was correct in **2/2** public matches.
- Named-team Smart Tagging hints were correct in **1/2** public matches.

The mismatch sample is too small to support a default pipeline change. In particular, a tag window is coarse and may contain the wrong dead stretch when a tag overlaps multiple events. The audit reports ambiguous matches rather than turning them into a model decision.

### What this means

The earlier v4 public restart audit does not show that restart timing is definitively wrong. It shows a consistent +3.6-second predicted-edge delay in only two public examples. It also shows that team evidence agrees on those examples, while named-tag hints are not reliable enough alone.

This supports continuing restart research, but not changing production restart logic based on the first audit alone. Any future restart change must improve public edge or transition results and survive label-free match checks, unit tests, and a larger replay audit.

## v8 kick-off and restart audit

A read-only audit of the v8 pipeline, from the tracking and Smart Tagging only, is in the
[audit report](./reports/law_checks_v8.md) and reproducible with
[`scripts/law_checks_v8.py`](./scripts/law_checks_v8.py). No ground truth and no human-produced
data about the match are used.

- **Kick-off owners do not depend on tag team hints.** At all five kick-offs (two half starts,
  three after goals) the owner and first live frame are the same with every Smart Tagging `Team`
  value blanked. Elsewhere, blanking them changes 844 graded frames (0.53%): restarts with no
  visible taker, where a named corner, free-kick or goal-kick tag supplies the team.
- **The Law 8 side check passes at every kick-off.** Before the tracklet correction, slot 45
  stands in the wrong half at both half starts; after it, no kick-off has a violation.
- **Every dead stretch over 60 s looks like a stoppage in the tracking**: players walking (median
  p90 speed 1.3–1.4 m/s) with the ball mostly unseen, including the two stretches no tag
  overlaps. The second-half pre-kick-off block is the exception (players warm up with the ball)
  and is dead by Law 8.
- `[57846, 62478)` contains a 138.6 s ball gap; the restart type at its end (goal kick, free kick
  or drop ball) cannot be told from the tracking.

## What changed from v4 to v8, and what drove the score

The latest changes are recorded in [the ablation log](./reports/ablation.csv) and [the public evaluation report](./reports/eval_public.md):

| Version | Change | Public Macro-F1 | Public Spell-F1 | Full-match proxy possessions |
|---|---|---:|---:|---:|
| v4 | Calibrated baseline | 0.8533 | 0.5059 | 302 |
| v5 | Use a gentle first touch to a teammate as restart evidence | 0.8550 | 0.5059 | 302 |
| v6 | Correct kick-off-illegal tracklet team labels | **0.8671** | **0.5500** | 298 |
| v7 | Align after-goal restarts to detected kick-offs | 0.8671 | 0.5500 | 298 |
| v8 | Merge the false live gap before an after-goal kick-off | 0.8671 | 0.5500 | 295 |

The largest measured public-score gain came from the **tracklet team correction**, not the later
restart-timing rules. Disabling only that correction returned the public score to Macro-F1 0.8550
and Spell-F1 0.5059. Disabling kick-off timing or the goal-to-kick-off merge left the public scores
unchanged. Those two rules change full-match labels, but their effect on private-match accuracy is
unknown.

### What the A-to-H frame ranges mean

In the team-correction ablation, these are **half-open frame ranges** where the no-correction
candidate labeled the frames `A` and v8 labels them `H`:

| Frames | Length | Interpretation |
|---|---:|---|
| `[11704, 11813)` | 109 frames (4.36 s) | Inside the public window; the documented false Athletic spell, corrected to Barcelona |
| `[114163, 114173)` | 10 frames (0.40 s) | Full-match correction, outside the public window |
| `[114296, 114349)` | 53 frames (2.12 s) | Full-match correction, outside the public window |

Together these spans account for **172 changed labels**. They are not three separate edits to
the CSV: they are the observed output differences when the same pipeline is run with the
tracklet-team fix on versus off. The fix identifies away goalkeeper slot 45 as standing roughly
46 m into the opponents' half at both half-start kick-offs, contrary to Law 8, and reassigns that
tracklet to the home team. Recomputing tracking features then changes team-control evidence.

At the two later spans, slot 45 itself is missing from the ball-proximity evidence on the inspected
frames. Thus those A-to-H changes are **downstream effects of changing that tracklet's team
membership in the detector**, not proof that the goalkeeper was physically tracking the ball in
those exact frames. Only the first 109-frame span is in the public scoring window; the other 63
changed frames are label-free full-match effects and cannot be called correct from public truth.

This distinction matters: the v6 score jump is a real improvement on the public sample, but its
cause is concentrated in one public possession correction. The later v7/v8 rules are law-based
structural cleanups, not additional measured public-score gains.

## Earlier score history

| Variant | Macro-F1 | Spell-F1 | Full-match possessions | Assessment |
|---|---:|---:|---:|---|
| C2 control | 0.8502 | 0.5081 | 431 | Too fragmented |
| Initial C3 control | 0.8539 | 0.5059 | 297–299 | Cleaner, but boundary calibration improved it |
| Calibrated C3 pipeline (v4) | 0.8533 | 0.5059 | 302 | Previous baseline |
| Laws of the Game refinements (v8) | 0.8671 | 0.5500 | 295 | Current submission |

Important comparison:

- **431 to 302 was not one threshold change and should not be described as a direct A-to-B tuning experiment.**
- It reflects replacing the more permissive C2 detector and applying the calibrated C3 pipeline.
- v4 predicted 17 public spells; v8 predicts 15 against 17 truth possessions, with 10 matched at IoU 0.5.
- The full-match mean possession length increased from approximately 8 seconds under the early C2-style output to 11.4 seconds in v8.

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

## Why the current v8 is promising, not proven

### 1. It is less fragmented than the early C2 output

The progression from 431 C2 possessions to 302 in v4 and 295 in v8 is substantially less
fragmented. Counts are label-free proxies, not evidence that every deleted possession was false.

### 2. It is closer to the evidence-backed range

The public truth contains 17 possessions in a five-minute sample. Scaling that rate to 105.9 graded match minutes gives an approximate target of 360 possessions. This is only an extrapolation, but it is more useful than pass arithmetic:

- 295 is below 360
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
4. Public transition matching remains a weakness. The older v4 audit reported:
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

Earlier v4 restart-audit status:

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
| Full-match possessions | 295 | Plausible exploratory range: 300–360 |
| Mean possession duration | 11.41 seconds | 11–14 seconds |
| Dead-ball share | 47.04% | Maintain roughly 47% |
| Home live share | 59.25% | Monitor only |
| Short possessions under 1 second | 5.42% | Reduce further only with evidence |

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

- Current production submission: v8, 295 full-match proxy possessions.
- Public result: Macro-F1 0.8671, Spell-F1 0.5500; 15 predicted possessions, 10 matched at IoU 0.5.
- Dead-ball sanity review: 134 stretches, 47.04% dead share, no sub-2-second stretches.
- Restart audit: 33 set-piece tags reviewed; public restart evidence limited to two matches.
- Kick-off audit: owners at all five kick-offs unchanged without tag team hints; no Law 8 side
  violation remains after the tracklet correction.
- Possession duration proxy: p10 1.47 s, p50 7.88 s, p90 26.31 s, p95 33.90 s.
- Team-fix ablation: disabling it returns public scores to 0.8550 / 0.5059; the other two law
  refinements did not change public scores in one-at-a-time ablations.
- Test status: 63 unit tests passing, including the audit helpers in
  [`tests/test_law_checks_v8.py`](./tests/test_law_checks_v8.py).
- `YT.csv` is unchanged by the audit (SHA-256 `903CD22327DF7CD37C09042F5E595C326BB8CA5FA2954B282E7DF4377E7669D4`;
  a Windows checkout made before the `.gitattributes` fix shows `73DE0195…`, the same file with CRLF line endings).

## One-sentence meeting justification

"We moved from a permissive C2 detector with 431 fragmented possessions to the v8 C3 pipeline with 295 full-match proxy possessions; the Law 8 tracklet correction improved the public sample, while later restart rules are principled but not independently score-validated."
