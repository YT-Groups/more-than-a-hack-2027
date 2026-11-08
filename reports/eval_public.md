# Evaluation results (public window)

Commit `923ff77`, file `YT.csv` (SHA-256 `c5141c43a78eecfd267a3b287994c3fc9dbdf3397e87b528982b72889390c36e`).
Scored with the organisers' scorer against the public 5-minute ground truth (frames 11,185–18,684,
17 possessions), next to the four shipped baselines:

```bash
python run_pipeline.py --out YT.csv
python evaluation/run_evaluation.py --results <folder with YT.csv and the baselines> --bootstrap 1000
```

## Headline

| submission | macroF1 | spellF1 | spellF1 90% interval (5th–95th pct) | possessions (truth 17) | mean length (truth 14.2 s) |
|---|---|---|---|---|---|
| **YT** | **0.8533** | **0.5059** | 0.3529–0.6588 | 17 | 13.9 s |
| baseline_smooth125 | 0.5812 | 0.4242 | 0.267–0.594 | 16 | 18.5 s |
| baseline_smooth50 (published floor) | 0.6207 | 0.3256 | 0.214–0.447 | 26 | 10.8 s |
| baseline_smooth25 | 0.6200 | 0.2868 | 0.196–0.385 | 36 | 7.6 s |
| baseline_raw | 0.5972 | 0.0378 | 0.019–0.060 | 131 | 2.0 s |

## What the numbers say

- **Dead ball is the big win.** D-class F1 is 0.859, against 0.146 to 0.420 for the baselines.
  Alive/dead F1 is 0.964.
- **The team in possession is mostly right.** Per-class F1 is H 0.852 and A 0.849. Team accuracy
  on frames both call live is 0.882. The home share is 0.537, against 0.573 in the truth.
- **Possessions are found, but their edges are loose.** spellF1 is 0.65 at IoU 0.3 and 0.4, then
  0.59, 0.41 and 0.24 at 0.5, 0.6 and 0.7. Within ±1 s, only 3 of the 20 true possession changes
  are matched (transition F1 0.15). This is the main weakness. Part of it is the ATD-versus-truth
  disagreements listed in `docs/YT.md` section 6. The first three changes of the window, for
  example, are 5–8 s late because the raw tracking shows the other team on the ball.
- **No flicker.** Fragmentation is 1.00: 21 predicted segments against 21 true.

## What this tells us

**What it confirms**

- **The pipeline is aligned.** Frame indexing, home and away, and what each label means are all
  right. An off-by-one or a team swap would have collapsed the scores.
- **The dead-ball detector works.** It is the largest single gain over the baselines.
- **We do not flicker.** 17 possessions against 17 true, with almost the same mean length. The
  baselines either invent possessions (26 to 131) or merge them.

**Where it points: the boundaries are a few seconds off**

- spellF1 is 0.65 when a 30% overlap is enough, but only 0.24 at 70%. So we usually find the right
  possession but start or end it a few seconds early or late. The ranking metric averages over all
  five thresholds, so sharper boundaries are where the remaining points are, not more possessions.
- Part of the error is in the data, not our method. Where the raw tracking shows the other team on
  the ball (the first three changes of the window), we are late by construction.
- The rest is our stoppage timing, especially the 83 restarts where the kick is not visible.

**What it cannot tell us**

- **Our final rank.** 17 possessions are too few: the scorer itself says gaps under about 0.33
  spellF1 are within noise here, and our 90% bootstrap interval (5th–95th percentile) is 0.35–0.66.
- **How we do on the hard parts of the match.** The ball is visible on 88% of frames in this
  window, against 59% over the full match.
- **Whether one of our versions beats another.** That is why versions were chosen by reasoning and
  label-free full-match checks, not by this score.

**Bottom line.** We are clearly above the published floor and the method is sound. Any further
effort before the feature freeze should go on restart and stoppage timing, not on control
detection. Such a change has to be justified by reasoning, because this window cannot confirm it.

## Full-match checks (no labels)

From `python run_pipeline.py`: 302 possessions (2.85 per minute), mean 11.2 s, median 7.7 s,
5.96% shorter than 1 s, 46.90% of graded frames dead, home share in play 58.93%, no X on graded
frames. Version history is in `reports/ablation.csv`.

## Conservative reception experiment

An optional pass-reception detector now produces ownership changes only after visible ball flight,
same-team passer evidence, stable receiver proximity, opponent separation, and confirmed control.
It is available with:

```bash
python run_pipeline.py --out YT.csv --receptions
```

On the real match, this conservative implementation found 445 candidate events and 422 unique
handover frames. Compared with the baseline submission it changed only 41 live-frame labels; the
public result changed from Macro-F1 0.8533 / Spell-F1 0.5059 to 0.8538 / 0.5059, while preserving
17 predicted possessions and the 302-possession full-match proxy count.

This is a small, unconfirmed frame-level improvement: the 0.0005 Macro-F1 difference is below the
repository's 0.01 segment-level noise floor and Spell-F1 did not change. The detector therefore
remains opt-in and is not evidence that private-match possession boundaries improve. It provides
the next measurable boundary research direction, with synthetic tests for stable receptions and
opponent contests, but requires broader public/private replay validation before becoming default.

### 41-frame audit

The conservative reception candidate changes 41 frames grouped into 21 two-frame runs. In the
public graded window, 4 of those frames correct baseline ownership errors and 0 regress it:

| frame | public truth | baseline | reception |
|---|---|---|---|
| 12475 | H | A | H |
| 12476 | H | A | H |
| 12702 | A | H | A |
| 12703 | A | H | A |

These are valid pass-position corrections, not public-screen false positives. The rest of the
41-frame effect lies outside the public window, where ground truth is not available.

### Layer 3 relaxation audit

The default real-match counts separate the candidate stages as follows:

| stage | count |
|---|---:|
| visible fast-ball flights | 1172 |
| flights with passer-team evidence | 925 |
| events rejected by dead-ball overlap | 173 |
| events rejected by unstable proximity or opponent contest | 232 |
| events rejected by control-consistency below 80% | 89 |
| final events before duplicate handover frames are removed | 445 |
| unique handover frames | 422 |

Relaxing the 10-frame receiver dwell to 3 frames adds only 9 final events (445 to 454) and leaves
public scores unchanged at 0.8533 Macro-F1 and 0.5059 Spell-F1. Removing the C3 control-consistency
layer is dangerous: it raises events to 542, fragmentations to 316 possessions, lowers public
Macro-F1 to 0.8527, lowers public Spell-F1 to 0.4914, and increases predicted public spells from
17 to 18. The 422-to-445 true confidence event loss is primarily because the extra candidates are
contested or C3-inconsistent, not because the original dwell is starving the detector.

## Dead-ball sanity check (no labels)

Generated from `YT.csv` with `python -m yt.sanity YT.csv`. The `frame` column was checked against
the row index and passed. Dead stretches were found independently per graded half, with exclusive
end frames.

| summary | result |
|---|---:|
| graded frames marked D | 46.90% |
| dead stretches | 137 |
| half 1 stretches | 59 |
| half 2 stretches | 78 |
| duration 10th percentile | 5.33 s |
| duration median | 12.24 s |
| duration 90th percentile | 46.06 s |
| maximum duration | 185.28 s |
| stretches under 2 s | 0 |
| stretches over 60 s | 8 |

The five longest stretches are:

| half | frame range | duration |
|---|---|---:|
| H1 | `[57846, 62478)` | 185.28 s |
| H2 | `[158438, 162615)` | 167.08 s |
| H1 | `[70096, 73894)` | 151.92 s |
| H2 | `[110808, 113734)` | 117.04 s (second-half pre-kickoff graded block) |
| H1 | `[86615, 88499)` | 75.36 s |

The first four ranges are plausible review targets because long stretches at the half boundaries,
half-time gap, or extended stoppages may be legitimate but should be visually or structurally
checked when match assets are available. There were no detected stretches below 2 seconds.

Smart Tagging overlap and containment checks are implemented in `yt/sanity.py`, including clipping
tag windows to match/frame bounds and half-open interval semantics. With the tactical XML now
present, 59 tag windows were checked; 74 dead stretches overlap a tag window and 63 stretches
longer than 3 seconds do not overlap any tag window. Because Smart Tagging is inferred rather than
ground truth, the untagged list is a review queue rather than evidence that a detected stoppage is
wrong.

Across the 59 windows, 55 set-piece windows contain a detected dead stretch:

| set-piece type | windows containing D |
|---|---:|
| Throw-in | 25/26 |
| Corner Kick | 11/12 |
| Goal Kick | 9/10 |
| Free Kick | 5/5 |
| Kick Off (After Goal) | 4/4 |
| Kick Off (Start) | 1/2 |

### Restart timing prior and long-stretch review

Restart timing was measured by taking each set-piece window's detected dead stretch whose exclusive
end falls inside the window, then reporting `dead_end - tag_start`. These are label-free timing
priors from Smart Tagging, not ground truth.

| set-piece type | single matches | median | p10 | p90 | ambiguous | no match |
|---|---:|---:|---:|---:|---:|---:|
| Throw-in | 18 | 7.52 s | 4.85 s | 16.27 s | 6 | 2 |
| Corner Kick | 9 | 5.56 s | 3.30 s | 7.51 s | 2 | 1 |
| Goal Kick | 3 | 7.80 s | 7.29 s | 14.10 s | 4 | 3 |
| Free Kick | 5 | 11.20 s | 3.89 s | 12.75 s | 0 | 0 |
| Kick Off (After Goal) | 2 | 4.62 s | 4.41 s | 4.83 s | 0 | 2 |
| Kick Off (Start) | 1 | 2.96 s | 2.96 s | 2.96 s | 0 | 1 |

These statistics use only windows with one dead-end match. Throw-ins, corners, and goal kicks have
wide or ambiguous distributions, so tags should remain coarse hints rather than exact flip priors.
Free kicks have complete coverage but still span 8.86 seconds between p10 and p90; they need video
review before being used as a narrow timing prior. Goal kicks have three misses and four ambiguous
windows. The kickoff samples are small and their no-match counts should not be interpreted as direct
detector misses without video review. The after-goal kickoffs median 4.62 seconds from tag start,
while the tagged second-half kickoff remains 2.96 seconds, showing the conventions differ.

The public ground truth provides direct start- and end-flip checks, independent of tag convention:

| truth D range | detector range | start latency | end latency |
|---|---|---:|---:|
| `[12851, 12989)` | `[12831, 13030)` | -20 frames (-0.80 s) | +41 frames (+1.64 s) |
| `[15889, 16400)` | `[15882, 16400)` | -7 frames (-0.28 s) | 0 frames |
| `[17380, 17838)` | `[17302, 17873)` | -78 frames (-3.12 s) | +35 frames (+1.40 s) |
| `[18027, 18389)` | `[18185, 18479)` | +158 frames (+6.32 s) | +90 frames (+3.60 s) |

The restored event-refined detector is not consistently shifted late by about 40 frames. Starts
range from 78 frames early to 158 frames late, while ends range from exact to 90 frames late.

The eight stretches over 60 seconds were reviewed against overlapping Smart Tagging and goal tags:

| half | frame range | duration | overlapping tags / review |
|---|---|---:|---|
| H1 | `[47898, 49740)` | 73.68 s | Barcelona goal and After-Goal kickoff; likely stoppage around the restart |
| H1 | `[57846, 62478)` | 185.28 s | Athletic goal kickoff and corner; candidate merged event context |
| H1 | `[70096, 73894)` | 151.92 s | Athletic goal and free kick; candidate extended goal-stoppage context |
| H1 | `[86615, 88499)` | 75.36 s | No overlapping goal or set-piece tag; candidate for merged live play |
| H2 | `[110808, 113734)` | 117.04 s | Start-of-second-half kickoff; candidate pre-kickoff/halftime dead block |
| H2 | `[158438, 162615)` | 167.08 s | Barcelona goal tag; candidate extended goal-stoppage context |
| H2 | `[175322, 176986)` | 66.56 s | No overlapping goal or set-piece tag; candidate for merged live play |
| H2 | `[180893, 182446)` | 62.12 s | Athletic goal kick; candidate stoppage/restart context |

## Full scorer output

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns
ranked on spell_f1_mean; gaps below ~0.33 are within noise (widest bootstrap interval, 17 possessions, 1000 draws)
spellF1 = mean possession-spell F1 over IoU 0.3/0.4/0.5/0.6/0.7; transF1/lat_f at +/-25 frames

rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  YT                         0.8533   0.5059     17    13.9   0.1500   -9.0   0.036  1.00
   2  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   4  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52

macroF1 would order the field differently -- these move, biggest first:
  baseline_smooth125         board #2   -> macroF1 #5    DOWN 3
  baseline_smooth50          board #3   -> macroF1 #2    up 1
  baseline_smooth25          board #4   -> macroF1 #3    up 1
  baseline_raw               board #5   -> macroF1 #4    up 1
  one that rises under macroF1 is trading possession structure for per-frame accuracy

YT
  macro F1 0.8533  (H 0.8520  A 0.8485  D 0.8594)   kappa 0.7685  acc 0.8523
  bootstrap macroF1 5-95 pct  0.7078 .. 0.8989   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.8894   0.8176   0.8520      3454
    A                 0.8228   0.8758   0.8485      2577
    D                 0.8287   0.8924   0.8594      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        2824     421     209       0
                          A         258    2257      62       0
                          D          93      65    1311       0
                          X       46528   31901   72928   40462
  transitions  gt 20  pred 20
      tau=12  P 0.1000  R 0.1000  F1 0.1000   matched 2  missed 18  spurious 18
      tau=25  P 0.1500  R 0.1500  F1 0.1500   matched 3  missed 17  spurious 17
  latency      mean -9.0  median -7.0  p90|lat| 17  ambiguous 0
  possessions  gt 17  pred 17  (+0)   mean duration 13.9 s vs 14.2 s (-2%)
  spell F1     mean 0.5059   @0.3 0.6471  @0.4 0.6471  @0.5 0.5882  @0.6 0.4118  @0.7 0.2353
      bootstrap 5-95 pct  0.3529 .. 0.6588   (17 possessions, 1000 draws)
      IoU 0.5   P 0.5882  R 0.5882  F1 0.5882   matched 10  missed 7  spurious 7
  alive/dead F1 0.9641   team acc  gt-alive 0.8425  both-alive 0.8821
  share        gt home 0.5727  pred 0.5365  err 0.0362
  fragmentation  21 / 21 = 1.00
  abstained    0 frames of X on graded ground truth (0.00%)

baseline_smooth125
  macro F1 0.5812  (H 0.7864  A 0.8108  D 0.1464)   kappa 0.5355  acc 0.7275
  bootstrap macroF1 5-95 pct  0.4480 .. 0.6654   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.6876   0.9184   0.7864      3454
    A                 0.7824   0.8413   0.8108      2577
    D                 1.0000   0.0790   0.1464      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        3172     282       0       0
                          A         409    2168       0       0
                          D        1032     321     116       0
                          X       78145   59353   54321       0
  transitions  gt 20  pred 16
      tau=12  P 0.1250  R 0.1000  F1 0.1111   matched 2  missed 18  spurious 14
      tau=25  P 0.1875  R 0.1500  F1 0.1667   matched 3  missed 17  spurious 13
  latency      mean -7.7  median -3.0  p90|lat| 18  ambiguous 0
  possessions  gt 17  pred 16  (-1)   mean duration 18.5 s vs 14.2 s (+30%)
  spell F1     mean 0.4242   @0.3 0.6061  @0.4 0.6061  @0.5 0.4242  @0.6 0.2424  @0.7 0.2424
      bootstrap 5-95 pct  0.2667 .. 0.5939   (17 possessions, 1000 draws)
      IoU 0.5   P 0.4375  R 0.4118  F1 0.4242   matched 7  missed 10  spurious 9
  alive/dead F1 0.8991   team acc  gt-alive 0.8854  both-alive 0.8854
  share        gt home 0.5727  pred 0.6247  err 0.0520
  fragmentation  17 / 21 = 0.81
  abstained    0 frames of X on graded ground truth (0.00%)

baseline_smooth50
  macro F1 0.6207  (H 0.7781  A 0.7615  D 0.3225)   kappa 0.5202  acc 0.7132
  bootstrap macroF1 5-95 pct  0.4602 .. 0.7418   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.6972   0.8804   0.7781      3454
    A                 0.7496   0.7738   0.7615      2577
    D                 0.6569   0.2138   0.3225      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        3041     413       0       0
                          A         419    1994     164       0
                          D         902     253     314       0
                          X       68595   52182   71042       0
  transitions  gt 20  pred 29
      tau=12  P 0.0690  R 0.1000  F1 0.0816   matched 2  missed 18  spurious 27
      tau=25  P 0.0690  R 0.1000  F1 0.0816   matched 2  missed 18  spurious 27
  latency      mean +4.0  median +4.0  p90|lat| 6  ambiguous 0
  possessions  gt 17  pred 26  (+9)   mean duration 10.8 s vs 14.2 s (-24%)
  spell F1     mean 0.3256   @0.3 0.5116  @0.4 0.4651  @0.5 0.3256  @0.6 0.2326  @0.7 0.0930
      bootstrap 5-95 pct  0.2140 .. 0.4465   (17 possessions, 1000 draws)
      IoU 0.5   P 0.2692  R 0.4118  F1 0.3256   matched 7  missed 10  spurious 19
  alive/dead F1 0.8990   team acc  gt-alive 0.8349  both-alive 0.8582
  share        gt home 0.5727  pred 0.6212  err 0.0485
  fragmentation  30 / 21 = 1.43
  abstained    0 frames of X on graded ground truth (0.00%)

baseline_smooth25
  macro F1 0.6200  (H 0.7620  A 0.7123  D 0.3857)   kappa 0.4878  acc 0.6912
  bootstrap macroF1 5-95 pct  0.4553 .. 0.7608   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.6881   0.8538   0.7620      3454
    A                 0.7181   0.7066   0.7123      2577
    D                 0.6106   0.2818   0.3857      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        2949     505       0       0
                          A         492    1821     264       0
                          D         845     210     414       0
                          X       65160   48767   77892       0
  transitions  gt 20  pred 39
      tau=12  P 0.1282  R 0.2500  F1 0.1695   matched 5  missed 15  spurious 34
      tau=25  P 0.1538  R 0.3000  F1 0.2034   matched 6  missed 14  spurious 33
  latency      mean +8.2  median +6.5  p90|lat| 18  ambiguous 0
  possessions  gt 17  pred 36  (+19)   mean duration 7.6 s vs 14.2 s (-47%)
  spell F1     mean 0.2868   @0.3 0.4528  @0.4 0.3774  @0.5 0.3019  @0.6 0.2264  @0.7 0.0755
      bootstrap 5-95 pct  0.1962 .. 0.3849   (17 possessions, 1000 draws)
      IoU 0.5   P 0.2222  R 0.4706  F1 0.3019   matched 8  missed 9  spurious 28
  alive/dead F1 0.8974   team acc  gt-alive 0.7909  both-alive 0.8271
  share        gt home 0.5727  pred 0.6283  err 0.0556
  fragmentation  40 / 21 = 1.90
  abstained    0 frames of X on graded ground truth (0.00%)

baseline_raw
  macro F1 0.5972  (H 0.7290  A 0.6430  D 0.4197)   kappa 0.4282  acc 0.6512
  bootstrap macroF1 5-95 pct  0.4300 .. 0.7357   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.6722   0.7962   0.7290      3454
    A                 0.6498   0.6364   0.6430      2577
    D                 0.5582   0.3363   0.4197      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        2750     699       5       0
                          A         551    1640     386       0
                          D         790     185     494       0
                          X       59873   47000   84946       0
  transitions  gt 20  pred 136
      tau=12  P 0.0368  R 0.2500  F1 0.0641   matched 5  missed 15  spurious 131
      tau=25  P 0.0809  R 0.5500  F1 0.1410   matched 11  missed 9  spurious 125
  latency      mean -3.1  median -10.0  p90|lat| 20  ambiguous 2
               windows overlap, so treat the mean as indicative
  possessions  gt 17  pred 131  (+114)   mean duration 2.0 s vs 14.2 s (-86%)
  spell F1     mean 0.0378   @0.3 0.0946  @0.4 0.0541  @0.5 0.0270  @0.6 0.0135  @0.7 0.0000
      bootstrap 5-95 pct  0.0189 .. 0.0595   (17 possessions, 1000 draws)
      IoU 0.5   P 0.0153  R 0.1176  F1 0.0270   matched 2  missed 15  spurious 129
  alive/dead F1 0.8920   team acc  gt-alive 0.7279  both-alive 0.7784
  share        gt home 0.5727  pred 0.6184  err 0.0457
  fragmentation  137 / 21 = 6.52
  abstained    0 frames of X on graded ground truth (0.00%)

```
