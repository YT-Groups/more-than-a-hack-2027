# v8 kick-off and restart audit (tracking and tags only)

A read-only audit of the v8 pipeline, run in memory by `python scripts/law_checks_v8.py`. It
reads only the tracking data and Smart Tagging; no ground truth and no human-produced data about
the match are used. No file is written, and `YT.csv` is unchanged.

The "team values blanked" run is the full pipeline with every Smart Tagging `Team` value set to
empty. Tag times and types are kept, so the set-piece timing prior still applies; only the named
restart-team fallback is removed.

## 1. Owner at every kick-off

| Kick-off frame | Kind | Owner (full v8) | Owner (team values blanked) | First live frame (both) |
|---:|---|---|---|---:|
| 11188 | first-half start | A | A | 11188 |
| 49714 | after a goal | A | A | 49714 |
| 75533 | after a goal | H | H | 75533 |
| 113734 | second-half start | H | H | 113734 |
| 163084 | after a goal | A | A | 163084 |

Kick-off owners come from the tracking alone: blanking every tag team value changes none of them.
Elsewhere, blanking the team values changes 844 graded frames (0.53%). Those are restarts
where no taker is visible and a corner, free-kick or goal-kick tag supplies the team, so the
named-team fallback does matter there.

## 2. Law 8 side check at every kick-off

The check flags a tracked slot standing more than 5 m into the opponents' half, outside the centre
circle, one second before the kick.

| Kick-off frame | Visible slots | Before the correction | After the correction |
|---:|---:|---|---|
| 11188 | 20 | slot 45, tracklet [0, 12662) | none |
| 49714 | 20 | none | none |
| 75533 | 22 | none | none |
| 113734 | 21 | slot 45, tracklet [105242, 114455) | none |
| 163084 | 22 | none | none |

The correction moves exactly these two tracklets; no violation remains at any kick-off.

## 3. Where Smart Tagging can influence the result

| Code path | Effect |
|---|---|
| `tags.set_pieces()`, `tags.prior_mask()` | Only `SET PIECES` rows enter. Their windows relax the dead-ball detector for quick restarts. `GOALS` rows never enter. |
| `tags.team_hints()`, `tags.fill_restart_teams()` | Named corner, free-kick and goal-kick teams fill a restart owner only where tracking found none. |
| `laws.align_kickoffs()`, `laws.merge_before_kickoffs()` | Use the tracking and the dead-ball segments only; no tag rows. |

## 4. Dead stretches longer than 60 s

| Frames | Length | Ball visible | Longest ball gap | Player activity (median p90 speed) | Above 2.5 m/s | Overlapping tags |
|---|---:|---:|---:|---:|---:|---|
| [47898, 49714) | 72.6 s | 5% | 42.9 s | 1.26 m/s | 3% | Goals, Kick Off (After Goal) |
| [57846, 62478) | 185.3 s | 13% | 138.6 s | 1.40 m/s | 2% | Corner Kick, Goals, Kick Off (After Goal) |
| [70096, 73894) | 151.9 s | 7% | 53.4 s | 1.29 m/s | 1% | Free Kick, Goals |
| [86615, 88499) | 75.4 s | 8% | 69.5 s | 1.43 m/s | 1% | none |
| [110808, 113734) | 117.0 s | 81% | 11.0 s | 1.80 m/s | 27% | Kick Off (Start) |
| [158438, 163084) | 185.8 s | 0% | 185.8 s | 1.34 m/s | 4% | Goals, Kick Off (After Goal) |
| [175322, 176986) | 66.6 s | 12% | 50.8 s | 1.44 m/s | 2% | none |
| [180893, 182446) | 62.1 s | 39% | 18.8 s | 1.31 m/s | 3% | Goal Kick |

In every stretch the players walk (median p90 speed 1.3–1.4 m/s, at most 4% of frames above
2.5 m/s) and the ball is mostly unseen, which is what a stoppage looks like in this tracking. The
exception is the second-half pre-kick-off block, where the players warm up with the ball (27%
above 2.5 m/s) and Law 8 makes it dead until the kick-off. The two stretches with no overlapping
tag show the same walking pattern, so they are treated as genuine stoppages.

[57846, 62478) contains a 138.6 s ball gap. When the ball reappears it is near the home goal area
with a home player close by, and the label turns H at 62478 on a visible kick. The tracking does
not show whether that restart was a goal kick, a free kick or a drop ball, so its type is not
classified.

## Commands

```bash
python scripts/law_checks_v8.py
python -m unittest discover tests
```
