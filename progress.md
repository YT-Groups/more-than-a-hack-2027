# Project Progress

## Goal

Build an algorithm that labels every frame of the tactical Barcelona vs Athletic match as:

- `H` — home team in possession
- `A` — away team in possession
- `D` — dead ball
- `X` — no answer

The submission contains 199,319 rows. The ranking metric is `spellF1`, which scores possession
spells as intervals; `macroF1` is also reported. The repository's [challenge description](./CHALLENGE.md),
[evaluation guide](./EVALUATION.md), [data guide](./DATA.md), and [submission requirements](./SUBMISSION.md)
are the primary references.

## Working possession definition

The initial control hypothesis is:

- A team has candidate control when one of its players is close to the ball and no opponent is
  close enough to be considered challenging.
- A sustained candidate run confirms control.
- A confirmed handover is backdated to the first frame of the qualifying run.
- During loose-ball or uncertain intervals, possession should not change merely because of a
  single touch or a brief tracking fluctuation.
- Dead-ball detection is separate from control detection. No control evidence does **not** by
  itself mean the ball is dead.

Initial numerical guesses are a 2.0 m control-entry radius, 2.5 m release radius, 1.5 m opponent
contest radius, and 10-frame dwell (0.4 seconds at 25 fps). These are hypotheses, not established
values. Track slots are anonymous and may switch identities, so the prototype uses team-level
nearest-player distances rather than assuming a slot follows a player.

The first implementation deliberately does not infer `D`. It writes `X` outside confirmed `H`/`A`
control runs, meaning “unconfirmed,” not “dead ball.”

## Code written

### `starter/control_prototype.py`

Added an experimental ATD-based control detector:

1. Loads tactical tracking metadata and raw tracking data.
2. Maps player slots to home (`+1`) and away (`-1`) from the ATD metadata.
3. Converts normalized player and ball coordinates to metres using the match pitch dimensions.
4. Checks that frame IDs are unique, in range, and complete.
5. Cleans the ball track:
   - rejects implausible jumps using a maximum-speed threshold;
   - interpolates only short, internal gaps whose implied speed remains plausible;
   - assigns lower confidence to interpolated observations.
6. Finds the nearest tracked player for each team per frame.
7. Applies entry/release radii and the opponent contest threshold to form team candidates.
8. Confirms candidate runs meeting the dwell duration and backdates labels to the run's first
   frame.
9. Exposes the radii and dwell through command-line options.
10. Writes a standard `frame,label` CSV with `H`, `A`, or `X`; it intentionally emits no `D`.

Usage:

```powershell
python starter\control_prototype.py --out evaluation\results\control_prototype.csv
python evaluation\validate_submission.py evaluation\results\control_prototype.csv
python evaluation\run_evaluation.py
```

The default evaluator scores all CSVs under `evaluation/results/` against the public five-minute
sample.

### `starter/test_control_prototype.py`

Added synthetic unit tests for:

- confirming and backdating a sustained control run;
- leaving a run shorter than the dwell unconfirmed;
- blocking a candidate when the opposing team is too close;
- handover from one team to the other;
- interpolating a short ball gap but leaving a long gap missing;
- rejecting an implausible ball jump;
- writing zero-based frame numbers and supported output labels.

### `README.md`

Added the prototype to the starter-tools listing.

## Validation performed

### Reception boundary work

- Added `yt/reception.py` for conservative pass-flight and stable-reception evidence.
- Confirmed a same-team reception requires visible ball flight, a passer-team majority, sustained
  receiver proximity, opponent separation, and C3 control confirmation.
- Added synthetic tests for stable receptions and contested receptions.
- Added `--receptions` and machine-state support without enabling it by default.
- Real match result: 445 candidate events, 422 unique handover frames, 41 changed live labels.
- Public result with the feature enabled: Macro-F1 0.8538, Spell-F1 0.5059, 17 possessions.
- The baseline is unchanged at Macro-F1 0.8533, Spell-F1 0.5059, because the feature remains opt-in
  until the public gain is outside the repository's stated noise range.
- A 41-frame audit found 21 changed runs. Four changed public frames corrected baseline errors and
  none regressed; the public gain was isolated and did not affect Spell-F1.
- Layer 1 produced 1172 flights, 925 passed passer-team evidence, and the default had 445 final
  events / 422 unique handovers.
- Relaxing dwell from 10 to 3 frames changed 445 to 454 events and did not change public scores.
- Removing control consistency raised events to 542 but worsened public scores and fragmentation;
  this relaxation is rejected.

- `python -m unittest starter.test_control_prototype -v` — **7 tests passed**.
- `python -m py_compile starter\control_prototype.py starter\test_control_prototype.py` — passed.
- `python starter\control_prototype.py --help` — CLI options displayed successfully.
- `git diff --check` — passed.
- The generated `control_prototype.csv` was reported valid by the repository submission validator.

## Public-window experiment

The run reported the following results on the public sample (frames 11,185–18,684; 17 ground-truth
possessions):

| Submission | macroF1 | spellF1 | Predicted possessions | Mean duration |
|---|---:|---:|---:|---:|
| `baseline_smooth125` | 0.5812 | 0.4242 | 16 | 18.5 s |
| `control_prototype` | 0.3742 | 0.3886 | 18 | 16.7 s |
| `baseline_smooth50` | 0.6207 | 0.3256 | 26 | 10.8 s |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 | 7.6 s |
| `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0 s |

### How to read these results

- **macroF1** and **spellF1** are scores from 0 to 1; higher is better. For example, `0.3886`
  means the prototype's mean spell F1 across the five IoU thresholds was about 38.9%.
- **Predicted possessions** is the number of H/A spells after the scorer fills `X` gaps. The public
  truth has 17. The prototype predicts 18, while `baseline_smooth125` predicts 16. A similar count
  is useful but does not guarantee that the same possessions or boundaries were found.
- **Mean duration** is the average length of the predicted possession spells in seconds. The public
  truth averages 14.2 s. The prototype's 16.7 s is closer than `baseline_smooth50`'s 10.8 s, but
  duration alone does not say whether the team or timing is correct.
- **Smoothing suffixes** are half-window sizes in frames: `smooth25` votes over up to 51 frames
  (about 2.04 s total), `smooth50` over up to 101 frames (about 4.04 s), and `smooth125` over up
  to 251 frames (about 10.04 s). Boundary windows are shorter. `baseline_raw` does not smooth.
- The prototype's **0.3742 macroF1** is low because it emits no `D` and abstains frequently.
  In the scorer's confusion matrix on the 7,500 graded frames, 1,704 H frames and 981 A frames
  were correct; it correctly predicted no dead-ball frames because it never outputs `D`.
  Its 4,122 `X` predictions are 54.96% of the graded sample. These figures explain why it can
  retain a moderate spellF1 while being weak as a complete H/A/D frame labeler.
- Of the published rows, `baseline_smooth125` has the highest public spellF1 (0.4242), and the
  prototype is next (0.3886). The difference is 0.0356, about 3.6 percentage points. The
  prototype is 0.0630 above `baseline_smooth50` on this sample. Neither gap proves which method
  will do better on the full match.

This result is an initial diagnostic, **not evidence that the thresholds generalize**. The public
sample contains only 17 possessions; use it primarily to check data alignment, output format, and
implementation behavior rather than to select thresholds.

## Rating / scoring

The submission is ranked by **spellF1**, not by per-frame accuracy. The evaluator extracts maximal
`H` or `A` runs as possession spells, then matches predicted spells to ground-truth spells of the
same team one-to-one using temporal intersection-over-union (IoU). It computes F1 at IoU thresholds
0.3, 0.4, 0.5, 0.6, and 0.7, and averages those values. A split possession can therefore create
both a missed spell and an extra predicted spell, while a possession with a slightly inaccurate
boundary may still match if its overlap is sufficient.

For example, IoU is `overlap frames / union frames`. If the true spell is 100 frames and a
same-team prediction overlaps it for 80 frames, with a 20-frame union-only extension, IoU is
`80 / 120 = 0.667`: it qualifies at thresholds 0.3 through 0.6, but not 0.7. At each threshold,
the scorer computes `F1 = 2 × precision × recall / (precision + recall)` from the one-to-one
matches. The five threshold-specific F1 values are then averaged to produce spellF1.

**macroF1** is the first tie-break. It averages frame-level F1 for `H`, `A`, and `D`; each `X`
prediction on a graded frame counts as a miss. Accuracy is also reported but does not determine the
ranking. The evaluator additionally reports possession counts and durations, transitions, alive /
dead F1, team accuracy, and possession-share error to help diagnose failure modes.

Scores are decimals, not percentages: `0.01` is one percentage point and `0.04` is four percentage
points. The documented approximate full-match noise bands are 0.01 for macroF1 and 0.04 for
spellF1. They are not guarantees or universal significance cutoffs; the public sample is much
smaller, so its uncertainty is substantially larger. With only 17 true possessions, one possession
can shift the score by several points.

Ground-truth `X` frames are ungraded. Before temporal scoring, predicted `X` labels inside a
graded half are filled from the most recent prediction (with leading abstentions backfilled from
the first known label); this prevents an abstention inside a possession from splitting it into
multiple spells. Consequently, `X` is not a free way to mark uncertainty at a handover: the prior
team label is carried across it. The prototype's `X` regions also mean “control not confirmed,”
not “dead ball,” so its spellF1 should be read alongside its poor macroF1 and absence of `D`
predictions.

The public sample covers five minutes and only 17 ground-truth possessions. Its spellF1 can move
substantially when a single possession is gained or lost, so its leaderboard is useful for checking
that scoring and frame alignment work, but it is not a reliable estimate of full-match ranking.
The published full-match baseline floor is `baseline_smooth50` at spellF1 0.2433; that value is
not directly comparable to a public-window score.

Run the scorer from the repository root with:

```powershell
python evaluation\run_evaluation.py
```

It scores every CSV in `evaluation/results/` against the public sample by default. Validate the
exact candidate file before scoring:

```powershell
python evaluation\validate_submission.py evaluation\results\control_prototype.csv
```

The validator checks the file contract only; passing it does not mean the labels are accurate.

## Current limitations and cautions

- There is no dead-ball model yet, so `D` recall and overall label quality are not addressed.
- The control and contest radii, dwell, ball-gap limit, and speed limit are initial guesses.
- Team-level nearest-player proximity is only evidence of control; it cannot by itself distinguish
  a controlled touch from a rebound, pass, aerial duel, or other transient event.
- The detector currently requires consecutive candidate frames for dwell. Its short-gap ball
  interpolation helps only when the surrounding observations are plausible.
- Backdating a confirmed run is an approximation; it may place a boundary too early if the
  candidate evidence is noisy.
- The public-window `spellF1` is noisy and should not be optimized as though it were a full-match
  leaderboard.
- A prior attempt to run against the workspace's tactical ATD failed because the raw tactical
  tracking file was absent at that time. A subsequent run successfully produced and validated a
  199,319-row CSV, so the run environment or data availability changed. Confirm the data location
  when reproducing the experiment.

## Next steps

1. Reproduce the prototype run from a clean invocation and save its exact command/configuration.
2. Review the generated possession boundaries and transition report against the public sample,
   focusing on missed, spurious, and delayed handovers rather than just the aggregate score.
3. Add dead-ball detection as a separate module. Treat Smart Tagging set-piece intervals as
   approximate restart evidence, not exact frame boundaries or ground truth.
4. Add tests for dead/live behavior, ball gaps during a run, candidate flicker, and handover
   backdating.
5. Compare a small, pre-declared set of parameter variants; document that public-sample tuning may
   overfit and cannot establish full-match performance.
6. Validate every generated CSV and preserve the current baseline outputs for comparison.
