# Team YT: possession from automatic tracking

More Than a Hack 2027, Phase 1. Barcelona (H) vs Athletic Club (A).

**In one line:** players stop running when the ball is dead, and a team has the ball when one of
its players stays on it with the ball moving *with* him. Everything else is bookkeeping.

The submission `YT.csv` is rebuilt from the data by one command (section 8). No model is trained,
no labels of any kind are used, and nothing is tuned on the public window. Every threshold below
is either a physical quantity or read off a label-free statistic of the match, and is stated with
its source.

Tags: **[measured]** found by running code on the data, **[derived]** arithmetic,
**[assumed]** a choice we made and did not fit.

---

## 1. Approach

The ranking metric (spellF1) rewards recovering *whole possessions*, so the two things that matter
are where possessions start and end. A possession ends in one of two ways: the other team takes
control, or the ball goes dead. We detect the two separately and combine them in a small state
machine:

```
ATD + metadata ─► features ─┬─► dead-ball detector ─┐
Smart Tagging ──────────────┤                       ├─► state machine ─► H / A / D / X
                            └─► control detector ───┘
```

We chose rules over a learned model because there is nothing legitimate to learn from: hand
labels on this game are forbidden, the public 5 minutes are not a training set, and the private
truth is private. Rules also make the possession definition explicit, which is what the technical
document is asked to explain.

## 2. Data, and what we found in it

**Used:** the tactical ATD (`*_tactical_atd.txt`) and its metadata (`*_tactical_atd.xml`), and
the tactical Smart Tagging. **Dropped:** both videos (labels must not come from watching them and
we built no vision model), the homography (not needed for a tracking-only method), and everything
broadcast (not synchronised to the submission timeline).

Facts we checked on the data rather than assumed:

| | |
|---|---|
| ATD records / video frames | 199,319 / 199,319, `video_frame = frame_id - 1` [measured] |
| Home side | `Score/@idLocalTeam` = Barcelona. 24 home slots, 22 away, one goalkeeper slot each [measured] |
| Barcelona attacks | left to right in the first half (metadata) |
| Third ball value | ball speed (`BallChannelRef s`), NaN on 91% of visible frames, so unused [measured] |
| Ball visible, graded frames | 94,214 / 158,857 = 59.3% [measured; matches DATA.md] |
| Our baseline rebuild | `starter/baseline.py` output is byte-identical to the shipped baselines; public spellF1 0.0378 / 0.2868 / 0.3256 / 0.4242 as published [measured] |
| Second-half kick-off | graded block starts at 110,808, kick-off is at about 113,742 (detected 113,734). The 117 s before it are pre-match and labelled D [measured] |
| Smart Tagging time base | consistent in both halves: the kick-off tag starts 3.5 s (half 1) and 3.3 s (half 2) before the kick [measured] |

**The ball gaps are long, and mostly stoppages** (`reports/gaps.md`). The 150 gaps in the graded
frames have a median of 10.7 s; 85% of the missing frames are in gaps longer than 10 s and only
0.1% in gaps of 25 frames or fewer. So short-gap interpolation, the usual first fix, is close to
worthless here. But while the ball is missing the players are slow (median p90 player speed
1.58 m/s, against 3.41 m/s when the ball is visible): the tracker loses the ball mostly when it
is out of play. On frames our detector calls live the ball is missing only 14.1% of the time.

**The public window is unrepresentative** [measured]: the ball is visible on 88% of its frames,
against 59% over the match. Scores on it say little about the hard parts of the full match.

## 3. Dead ball

**Signal.** For every frame, the 90th percentile of the players' speeds, smoothed with a 1 s
centred median ("activity"). Over the whole match its histogram is clearly bimodal [measured]:
a stoppage mode at 1.25–1.75 m/s, a live mode around 3 m/s, and a valley at 2.0–2.25 m/s. No
labels are needed to see it.

**Core.** A stoppage *core* is a stretch where activity falls below 2.0 m/s (the valley) and
stays below 2.5 m/s (hysteresis), lasting at least 2 s, with cores less than 1 s apart merged.

**Veto for slow build-up.** Players also walk while a team circulates the ball at the back. A core
is dropped if, over its first half, the ball is visible, at least 3 m inside every line and moving
faster than 3 m/s on more than 60% of frames. A dead ball is still, missing, or out; a live one
keeps moving between players.

**Edges.** The core lags the whistle, so its edges are moved to events:

- *onset*: the last moment the ball crossed a line (tracked position more than 1 m outside the
  pitch; the ATD places the ball about 1.5 m inside the touchline even at throw-ins [measured]),
  else the moment the tracker lost the ball, else the peak of player activity in the 5 s before
  the core (the players start to decelerate at the whistle);
- *restart*: the last "kick" (ball speed from below 2 m/s to above 5 m/s) between 4 s before and
  2 s after the core ends. That last kick can be a second touch: a short corner or a throw-in
  reaches a team-mate, who then plays on. So an earlier, gentler first movement (above 2.5 m/s after
  the ball was still) is taken as the restart instead when the ball goes from one player to a
  *different* player at least 2 m away and the players speed up by at least 0.5 m/s in the next
  second. A ball repositioned by the same player, with everyone still walking, fails both tests.
  This moves 3 of 140 restarts earlier (0.24, 0.40 and 2.08 s; the last is a throw-in that the kick
  rule had timed at the receiver's touch).

**Self-calibration for edges we cannot see.** When no kick is visible (83 of 139 restarts), or the
deceleration peak sits at the very start of the search window (activity already falling, 22
onsets), the edge is placed using a lag measured on this match from the edges we *can* see. Where
the ball is seen to go out or be lost, the core starts a median of 46 frames (1.8 s) later. Where a
kick is visible, the core ends a median of 8 frames (0.3 s) after it [measured]. The ball itself is
the reference, and no labels are involved.

**Kick-offs.** The first kick from within 5 m of the centre spot after both teams have stood in
their own halves for 2 s. Everything in the half before it is dead.

**Laws of the Game.** Three more rules follow directly from the laws, apply to the whole match,
and are tied to no particular frame [measured effects]:

- *Law 8, kick-off timing.* Every kick-off, including the one after a goal, is taken from the centre
  mark with both teams in their own halves. Where that pattern appears within 5 s before or 2 s
  after a detected restart, the restart moves onto the kick. Applied blindly to the whole match,
  it fires only at the three after-goal kick-offs, each 1.0–2.8 s earlier, and agrees with the
  Smart Tagging kick-off windows (tag start + 3.5 s) to within 0.5 s. The fourth goal's kick-off
  shows no clear pattern and is left alone.
- *Laws 8 and 10, nothing live between a goal and its kick-off.* After a goal, play restarts only
  with a kick-off, and the celebration and the walk back take far longer than 30 s [assumed
  minimum]. So when a stoppage shorter than 30 s ends in a mid-half kick-off and follows another
  stoppage after a live gap under 30 s, the two are one stoppage. This removes two short false
  "live" spells while players walked back after a goal (75,052–75,366 and 162,615–162,850).
- *Law 8, team labels.* At a kick-off every player is in his own half, so a tracked slot more than
  5 m into the opponents' half (outside the centre circle) one second before a kick-off carries the
  wrong team label for that tracklet. The tracklet is moved to the other team and the features
  are rebuilt. It flags two tracklets of the away goalkeeper's slot at the two half-start
  kick-offs; both stand deep in the home half next to the home goalkeeper.

Two related rules were tested and not used. *Opponents of the last toucher restart after the ball
goes out* (Laws 9 and 15–17) agreed with the visible taker on only 9 of 18 restarts, because the
tracking cannot see who touched the ball last. *A ball placed in a goal area or at a corner names
the restarting team* (Laws 16–17) agreed 3 of 3 but changed no label, so the simpler pipeline was
kept. A broader version of the team-label rule (a goalkeeper slot near the other team's goal
belongs to the team defending it) was rejected. In every such stretch the real home goalkeeper was
visible too, so that slot was not following a goalkeeper, even though the rule would have raised
the public score.

**Smart Tagging as a prior.** Inside a tagged set-piece window, a shorter (1 s) and shallower
(below 2.5 m/s) dip also counts, but only if the ball is placed (below 0.5 m/s for 0.5 s) and only
where the base rule found nothing, so the tags can add a quick restart but never stretch or move
a stoppage we already found. Tags are automatic and not ground truth; this is the only way they
touch the dead mask.

**Label-free checks** [measured]: 55 of the 59 tagged set pieces overlap a detected stoppage (the
remaining ones include the opening kick-off, which is the start of the graded block). The dead
share of graded frames is 47.0%, which is plausible: LaLiga's effective playing time is
commonly reported to be low, around 52–55 minutes per match (a public figure we did not verify).

## 4. Control

Our definition: **a team has control when one of its players stays close to the ball for 0.4 s,
no opponent is challenging, and the ball is moving with the player.** Concretely, on frames with a
visible ball:

- *close*: the nearest player is within 2.0 m to take control and within 2.5 m to keep it;
- *uncontested*: no opponent within 1.5 m;
- *with the player*: the ball's velocity and the player's differ by less than 4 m/s (velocities
  from 5-frame smoothed positions; slot swaps only make this noisy on the swap frame);
- *held*: 10 consecutive frames.

A confirmed run is stamped from the frame the player first came within range, not from the frame
the relative-speed test first passed, because a ball arriving from a pass is still fast for a few
frames after the receiver reaches it. We use track slots only to read the team.

**Variants compared** (all with the same dead-ball detector and state machine, v4 settings otherwise):

| variant | rule | possessions (full match) | mean length | public macroF1 | public spellF1 [90% bootstrap interval] |
|---|---|---|---|---|---|
| C1 | nearest player within 2 m, 10-frame dwell | 415 | 8.1 s | 0.842 | 0.442 [0.295, 0.579] |
| C2 | C1 + contest test + hysteresis | 440 | 7.7 s | 0.850 | 0.508 [0.368, 0.649] |
| **C3** | C2 + ball moves with the player | **302** | **11.2 s** | 0.853 | 0.506 [0.353, 0.659] |

On the public window the variants are indistinguishable (the bootstrap intervals are 0.3 wide).
We chose C3 on two other grounds. First, it implements the challenge's own "not control" cases (a
touch without control, a deflection that does not intercept). Second, its possession rate is closer
to what the public truth implies once the dead share is accounted for. The public window has 3.4
possessions per minute with 80% of its frames live. At our full-match live share of 53.1%, that
scales to about 2.3 possessions per minute [derived]; C3 gives 2.9 and C2 4.2.

Where C2 and C3 disagree (11.6% of the frames both call live) the cause is almost always a short run that C2
accepts and C3 rejects [measured, full match]:

| frames | C2 | C3 | why |
|---|---|---|---|
| 40,413–40,708 | A | H | C2 accepts a 14-frame A run at 1.9 m with the ball 10.3 m/s faster than the player: a ball flying past. The ball then vanishes for 9 s and C2 holds the wrong team throughout |
| 76,433–76,858 | A | H | 19-frame A run at 2.1 m, relative speed 4.4 m/s, just before a 16 s ball gap. C3 keeps the H player who had it at 1.1 m, moving with it |
| 163,686–164,017 | H | A | an H player 1.1 m from the ball, but the ball leaves at 5 m/s relative to him: a deflection. C3 keeps the A possession that preceded it |
| 155,925–156,045 | H | A | an 11-frame H run at 1.9 m with relative speed 7.8 m/s and no opponent within 36 m: a pass skimming past a player |
| 126,213–126,357 | A (C2) | H (C1) | C1 vs C2: a 13-frame A run at the 2 m entry edge that C2 keeps through its 2.5 m hold radius and C1 drops |

**Sensitivity grid**, declared before scoring (dwell {8,10,12,14} × entry radius {1.5,2.0,2.5} ×
contest radius {1.0,1.5,2.0}, C2, run at v1 settings): public macroF1 stays within 0.829–0.852 in all 36 cells. The
full-match possession count moves from 290 to 507 and depends mostly on the entry radius, because a
larger radius accepts more balls passing near an opponent. We did not pick a cell by public score;
the defaults are the values we wrote down before running anything.

**Why the control detector abstains**, share of live graded frames (C3): control 29.8%, ball
missing 14.1%, nobody within 2 m (a pass in flight or a loose ball) 39.6%, contested 2.5%, ball not
moving with the player 12.6%, run shorter than the dwell 1.3%. Abstaining is not a problem in
itself, because the last owner keeps the ball.

## 5. State machine

1. A dead frame is `D`, with no owner.
2. In play, possession changes only on the first frame of a confirmed control run by the other
   team. The dwell is applied once, inside the control detector, so boundaries do not drift late.
3. Between control runs, including through ball gaps and loose or contested phases, the last
   owner keeps the ball (the challenge's rule for loose balls).
4. After a stoppage, the owner from the restart frame is the restarting team: the team of the
   player nearest the ball (within 3 m) around the restart kick. If the ball is not visible there,
   the team on the corner, free-kick or goal-kick tag is used. If neither is available, the team of
   the first control run is back-filled to the restart frame. On the 13 tagged restarts where we
   have both, our own estimate agrees with the tag on 11.
5. The owner resets at each half. Frames outside the graded halves are `X`.
6. A possession shorter than 1 s is merged into its neighbours only when both neighbours are the
   same team in the same live stretch. It is never merged across a dead ball, so `H D H` stays two
   possessions.
7. We never write `X` on a graded frame once an owner is known. Under spellF1 an `X` repeats the
   last label anyway, and under macroF1 it is a guaranteed miss.

## 6. Evaluation and what we did not do

Every version is logged in `reports/ablation.csv` with its commit and settings, its public-window
scores, and label-free proxies for the full match: dead share, possessions per minute, mean
possession length, share of possessions under 1 s, and in-play home share. We changed one thing at
a time.

| version | change | public macroF1 | public spellF1 | full-match possessions | dead share |
|---|---|---|---|---|---|
| published floor | baseline_smooth50 | 0.6207 | 0.3256 | | |
| v1_C2 | dead ball + C2 + state machine | 0.8502 | 0.5081 | 431 | 46.0% |
| v1_C3 | C3 control | 0.8539 | 0.5059 | 297 | 46.0% |
| v2 | second-half pre-kickoff dead | 0.8539 | 0.5059 | 293 | 47.5% |
| v3 | Smart Tagging prior and restart teams | 0.8539 | 0.5059 | 299 | 47.7% |
| v4 | self-calibrated edges where no event is visible | 0.8533 | 0.5059 | 302 | 46.9% |
| v5 | restart at a first touch to a different player | 0.8550 | 0.5059 | 302 | 46.9% |
| v6 | Law 8: tracklet team labels at kick-offs | 0.8671 | 0.5500 | 298 | 46.9% |
| v7 | Law 8: after-goal kick-off timing | 0.8671 | 0.5500 | 298 | 46.8% |
| v8 (submitted) | Laws 8+10: no live play between a goal and its kick-off | 0.8671 | 0.5500 | 295 | 47.0% |

v6's public gain comes from one corrected tracklet whose 109 changed frames happen to fall inside
the public window. v7 and v8 do not touch the public window (it contains no goal). Over the full
match v5–v8 change 777 graded frames (0.49%) and remove 7 false possession changes. On the public
window v4 matched 10 of the 17 true possessions with 7 false ones; v8 matches the same 10 with 5
false ones.

These are public-window numbers on 17 possessions, and the organisers measured that one missed
possession moves spellF1 by about six points. They show the pipeline is aligned and the labels mean
what we think. They are not evidence that one version beats another, and we did not use them that
way.

**Label-free validation on the full match.** The public window cannot rank versions, so we
also tested the whole match in three ways that need no labels [measured, on v4; v5–v8 change
0.49% of graded frames, and v8's Smart Tagging agreement is 81.57% overall and 82.14% outside the
public window, against 81.65% and 82.03% for v4]:

- *Agreement with Smart Tagging phases.* Inside the four possession-phase tags (progression,
  build-up, attacking transition, creating chances), our team matches the tagged team on 81.6% of
  57,529 live frames: 82.4% in the first half and 80.9% in the second. A random guess at our home
  share would score 51.6%, and always guessing Barcelona 69.9%. The tags are automatic and noisy.
  On the public window even the true labels agree with them only 76.0% of the time, and ours agree
  75.6%, so against the tags we behave like the truth does. Agreement is lowest in attacking
  transitions (70%), where a fixed 30 s tag window spans the change of possession. Each defending
  tag has the same window with the opposite team and gives identical numbers, which confirms that
  the team on a possession-phase tag is the team with the ball.
- *First half against second half.* The second half is choppier: 168 possessions against 134
  (3.2 against 2.5 per minute), a median length of 6.3 s against 8.8 s, 49.2% dead against 44.6%,
  78 stoppages against 59, and a home share in play of 55.5% against 62.1%. The tracking is equally
  good in both halves (ball visible on 59.3% of graded frames in each; 84% and 88% of live
  frames). Tag agreement is also almost the same (82% and 81%). So we read this as the match
  rather than the model, with one caveat: more of the second half's restarts have no visible kick
  (50 of 79, against 33 of 61), and those are timed by the calibrated lag.
- *Hiding parts of the ball tracking.* We hid random 2 s stretches of the visible ball, rebuilt
  everything and compared with the normal output, over three fixed seeds per level. With 5%, 10%
  and 20% of the visible ball removed, 0.8–1.2%, 1.4–1.8% and 3.2–4.2% of graded labels change,
  and the possessions agree with the unhidden ones at spellF1 0.97–0.98, 0.96 and 0.88–0.90. The
  model degrades gradually, and when it loses the ball it merges possessions (302 falls to 280–300)
  rather than inventing new ones.

These checks measure consistency and stability, not accuracy, because no labels are involved.

**Disagreements with the public truth** (the organisers asked to hear about them). We list them, and
did not change anything to remove them:

- **11,351–11,477.** The truth switches to H at 11,351. The ATD shows an away slot (one tracklet
  covering 9,632–36,673) dribbling the ball at 0.7–1.2 m with no home player closer than 2 m until
  11,470. Team counts are 10 against 10. We believe the uncorrected ball or that track differs from
  the corrected tracking here. This one case puts our first three boundaries 5–8 s late.
- **12,800–12,850.** The truth says A while a home player is 1.1 m from the ball and the nearest
  away player is 4.4 m away, just before a stoppage. This looks like the same kind of difference.
- **17,380.** The truth goes dead at 17,380, about a second after the players had already slowed
  to walking pace. Our onset (17,302) is 3 s early.
- **17,838.** The truth goes live on what looks like a short corner tap, about 1.4 s before the
  clear kick at 17,873 where v4 restarted. The first-touch rule now restarts at 17,863, still 1.0 s
  late: the tap happens while the tracker has lost the ball.

**Not done:**

- We did not estimate the ball position from player movement during gaps. Gaps in live play are
  rare (14% of live frames), and only 2 of 84 such gaps longer than 1 s end with the other team in
  control.
- We did not use the video in any way.
- We did not tune on the public window.

## 7. Limitations

- **Blind restarts.** 83 of 139 restarts have no visible kick. They are placed with the median
  kick-to-activity lag from the visible ones, but that lag varies (interquartile range about
  -0.7 to +1.2 s), and a late or early restart moves the start of the next possession.
  We tried to do better from the players alone, and tested it without labels. On the 52 restarts
  outside the public window where the kick is visible, we hid the ball and estimated the kick
  from player movement, fitting each method's offset on one half and testing it on the other
  [measured]. The current lag rule has a median error of 0.96 s (62% within 1 s). None of the
  four alternatives beat it once early calls are counted: when activity bottoms out (0.96 s),
  the steepest acceleration of the players (0.86 s, but 18 early calls over 1 s against 12),
  the players near the last-seen ball speeding up (1.20 s), or a jump in the share of running
  players (1.10 s). Neither did a median of three of them (0.96 s; better on 9 restarts, worse
  on 11). So we kept the lag rule. A restart's timing seems to need the ball, or the video.
- **Onset timing** depends on which event comes first. When players slow before the whistle (an
  advantage being played, or a slow attack stopped by a foul), the deceleration onset fires early.
- **The ATD ball near the touchline** sits about 1.5 m inside, so "out" uses a 1 m margin beyond
  the line. Some balls that leave the pitch are never seen to.
- **Slot teams are mostly trusted.** Apart from the kick-off law check (section 3), which corrects
  two tracklets, a tracklet that switches between players of opposite teams would put control on
  the wrong side. We measured how often the tracking shows signs of this, over the whole
  match [measured]. Of 3.25 million consecutive slot positions, 83 move more than 2 m in one frame
  (five times sprint speed) and none move more than 5 m. A home and an away slot stay within 0.5 m
  of each other for at least 1 s 51 times (1.1% of graded frames), mostly far from the ball (94
  frames within 3 m of it), which is consistent with tight marking rather than one player tracked
  twice. Only 4 of our 165 direct possession changes have either sign within 2 s near the ball.
  The two public-window disagreements in section 6 show neither sign, so the tracking cannot reveal
  them, and we built no correction for them.
- **Distances are 2D.** Aerial balls look close to players they fly over. The contest and
  relative-speed tests remove most of them, but not all.
- **With more time** we would add: (1) team-shape evidence (the team in possession spreads out and
  pushes up) to place restarts when the ball is invisible; (2) the challenge's "more than two
  consecutive passes" proxy as a second confirmation of a change of possession; (3) an explicit
  goal and celebration detector, so the long stoppages after goals have clean edges.

## 8. Reproduction

Python 3.13, `pip install -r requirements.txt` (numpy and pandas, pinned). Unpack the data bundle
into `data/` (or point `MTAH_DATA_DIR` at it), then:

```bash
python run_pipeline.py --out YT.csv
```

This builds the file, runs `evaluation/validate_submission.py` on it, and prints the public-window
scores and the full-match proxies. It takes about 20 s, and the first run also caches the parsed
tracking in `data/cache/`. `scripts/clean_clone_check.sh YT.csv` clones the repository into a
temporary folder, rebuilds the file from the raw tracking with `--no-cache` and compares SHA-256
hashes. It passes on the submitted commit. `python -m unittest discover tests` runs 63 unit tests.

Code map: `yt/load.py` (parsing and alignment), `yt/features.py`, `yt/dead.py`, `yt/control.py`,
`yt/tags.py`, `yt/laws.py` and `yt/teamfix.py` (Laws of the Game refinements), `yt/machine.py`,
`yt/evalkit.py` (scoring wrapper, proxies, ablation log), `yt/gaps.py` (gap report),
`run_pipeline.py`. `python run_pipeline.py --no-laws` switches the law refinements off (all of
them, or the ones named) to reproduce v5.

**Training data:** none. No labelled data of any kind, from this game or any other, is used.
