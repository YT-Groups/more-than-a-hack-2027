"""Dead-ball detector.

Idea: after a whistle the players stop running. The 90th-percentile player speed, smoothed over
one second, is bimodal over the match (a stoppage mode near 1.5 m/s, a live mode near 3 m/s, a
valley near 2.1 m/s). A sustained stretch below the valley is a stoppage *core*. Its edges are
then moved to the events that actually start and end it:

  onset    the last moment the ball left the pitch, or was lost by the tracker, or (failing
           both) the frame at which the players started to decelerate;
  restart  the kick that puts the ball back in play: the ball goes from (nearly) still to
           moving fast close to the end of the core. Without a visible ball, a fixed lead
           ahead of the activity rise is used.

A missing ball on its own is never treated as dead (CLAUDE.md 6.6.3); it only places an onset
inside a stretch that the players already say is a stoppage.

All thresholds are set from physical reasoning or from label-free statistics of the match,
never fitted to the public ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import Features, rolling, runs
from .load import FPS, HOME, PITCH, Match


@dataclass(frozen=True)
class DeadConfig:
    enter: float = 2.0          # activity below this starts a core (m/s); match histogram valley
    leave: float = 2.5          # ... and must rise above this to end it (hysteresis)
    min_core: int = 2 * FPS     # shortest core kept (frames)
    merge_gap: int = FPS        # live gaps shorter than this between cores are absorbed
    onset_search: int = 5 * FPS     # look this far back from a core start for the onset event
    restart_search: int = 4 * FPS   # look this far back from a core end for the restart kick
    restart_after: int = 2 * FPS    # ... and this far forward
    out_margin: float = -1.0    # ball counts as out beyond this (m); ATD reads ~1.5 m inside
    still_v: float = 2.0        # ball "still" below this (m/s)
    kick_v: float = 5.0         # ball "kicked" above this (m/s)
    blind_restart_lead: int = FPS // 2  # restart this many frames before the activity rise
                                         # when no kick is visible
    live_veto: float = 0.6      # drop a core if, over its first half, the ball is visible, well
                                # inside the pitch and moving on more than this share of frames
    live_interior: float = -3.0  # "well inside": at least 3 m from every line
    live_v: float = 3.0          # "moving": faster than this (m/s)
    # inside a Smart Tagging set-piece window (a prior, not truth)
    prior_enter: float = 2.5
    prior_leave: float = 3.0
    prior_min_core: int = FPS
    prior_still: int = FPS // 2  # ball below placed_v for this many frames in the core
    placed_v: float = 0.5        # a placed ball (m/s)


@dataclass
class DeadResult:
    dead: np.ndarray                   # (n,) bool
    conf: np.ndarray                   # (n,) float
    segments: list = field(default_factory=list)   # [(onset, restart, onset_reason, restart_reason)]


def cores(activity: np.ndarray, cfg: DeadConfig, prior: np.ndarray | None = None,
          ball_still: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Low-activity stretches with hysteresis, merged across short gaps, min length applied.

    Inside `prior` (set-piece windows) the thresholds are relaxed, and a core shorter than
    min_core is kept if it is at least prior_min_core long and the ball is still for at least
    prior_still frames in it (a quickly taken restart).
    """
    base = [(a, b) for a, b in _hysteresis(activity, cfg.enter, cfg.leave, cfg.merge_gap)
            if b - a >= cfg.min_core]
    if prior is None or not prior.any():
        return base
    ball_still = np.zeros(len(activity), dtype=bool) if ball_still is None else ball_still
    taken = np.zeros(len(activity), dtype=bool)
    for a, b in base:
        taken[a:b] = True
    extra = []
    act = np.where(prior, activity, np.nan)
    for a, b in _hysteresis(act, cfg.prior_enter, cfg.prior_leave, cfg.merge_gap):
        # only quick restarts the base rule missed; never stretch a stoppage it already found
        if (not taken[a:b].any() and b - a >= cfg.prior_min_core
                and ball_still[a:b].sum() >= cfg.prior_still):
            extra.append((a, b))
    return sorted(base + extra)


def _hysteresis(activity: np.ndarray, enter: float, leave: float, merge_gap: int
                ) -> list[tuple[int, int]]:
    """Runs where activity drops below `enter` until it rises above `leave`, merged across
    gaps shorter than `merge_gap`. NaN activity (no players) is not evidence of a stoppage."""
    act = np.where(np.isnan(activity), np.inf, activity)
    state = np.zeros(len(act), dtype=bool)
    on = False
    for i in range(len(act)):  # inherently sequential; 2e5 steps is fast enough
        on = act[i] < (leave if on else enter)
        state[i] = on
    s, e = runs(state)
    merged: list[list[int]] = []
    for a, b in zip(s, e):
        if merged and a - merged[-1][1] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def find_onset(f: Features, start: int, lo: int, cfg: DeadConfig) -> tuple[int, str]:
    """Frame the stoppage began, searching [max(lo, start - onset_search), start]."""
    a = max(lo, start - cfg.onset_search)
    seg = slice(a, start + 1)
    vis = f.ball_vis[seg]
    out = np.nan_to_num(f.ball_out[seg], nan=-99.0) > cfg.out_margin
    # ball leaving the pitch: first frame of the last out-run
    if out.any():
        idx = np.flatnonzero(out & ~np.r_[False, out[:-1]])
        return a + int(idx[-1]), "ball_out"
    # tracker losing the ball: first missing frame of the last gap that reaches the core
    if vis.size and not vis[-1]:
        seen = np.flatnonzero(vis)
        if seen.size:
            return a + int(seen[-1]) + 1, "ball_lost"
    # players decelerating: peak of lightly smoothed activity in the window
    act = rolling(f.activity_raw[seg], 10, "mean")
    if np.all(np.isnan(act)):
        return start, "core"
    return a + int(np.nanargmax(act)), "decel"


def still_before(still: np.ndarray, window: int) -> np.ndarray:
    """True at i if any of still[i-window : i] is True (strictly before i)."""
    c = np.concatenate(([0], np.cumsum(still)))
    i = np.arange(len(still))
    return (c[i] - c[np.maximum(0, i - window)]) > 0


def find_restart(f: Features, end: int, hi: int, cfg: DeadConfig) -> tuple[int, str]:
    """Frame play restarted: last kick (still -> fast) in [end - restart_search, end + after]."""
    a = max(0, end - cfg.restart_search)
    b = min(hi, end + cfg.restart_after)
    v = f.ball_v[a:b]
    fast = np.nan_to_num(v, nan=0.0) > cfg.kick_v
    still = np.nan_to_num(v, nan=np.inf) < cfg.still_v
    # a kick is the first fast frame after a still frame within the previous second
    kicks = np.flatnonzero(fast & ~np.r_[False, fast[:-1]] & still_before(still, FPS))
    if kicks.size:
        return a + int(kicks[-1]), "kick"
    return max(a, end - cfg.blind_restart_lead), "blind"


def kickoff_frame(m: Match, f: Features, lo: int, hi: int, half: int,
                  centre_r: float = 5.0, own_half_share: float = 0.75, halfway_tol: float = 1.5,
                  kick_v: float = 4.0, lineup: int = 2 * FPS,
                  search: int = 10 * 60 * FPS) -> int | None:
    """First kick-off of a half: the ball near the centre spot and kicked within a second, after
    both teams have stood in their own halves for `lineup` frames (mean share of tracked players
    on the correct side at least `own_half_share`). Averaging over the line-up tolerates
    duplicated objects and the odd wrong-team slot, and rejects a pre-match kickabout.

    Own halves come from the metadata's first-half attack direction, swapped for half 2.
    Returns the kick frame, or None if no such moment is found in the first `search` frames.
    """
    L, W = PITCH
    home_left = (m.attack_dir_home_h1 == "left_to_right") == (half == 1)
    hi = min(hi, lo + search)
    x = m.players_xy[lo:hi, :, 0]
    seen = ~np.isnan(x)
    # players standing on the halfway line count for both sides
    in_left = x < L / 2 + halfway_tol
    in_right = x > L / 2 - halfway_tol
    home = m.team == HOME
    home_ok = in_left if home_left else in_right
    away_ok = in_right if home_left else in_left
    with np.errstate(invalid="ignore", divide="ignore"):
        h_ok = (home_ok & seen & home).sum(1) / (seen & home).sum(1)
        a_ok = (away_ok & seen & ~home).sum(1) / (seen & ~home).sum(1)
    share = np.fmin(np.nan_to_num(h_ok), np.nan_to_num(a_ok))
    c = np.concatenate(([0.0], np.cumsum(share)))
    i = np.arange(len(share))
    before = (c[i] - c[np.maximum(0, i - lineup)]) / np.maximum(1, i - np.maximum(0, i - lineup))
    formation = (before >= own_half_share) | ((i == 0) & (share >= own_half_share))
    near = np.hypot(m.ball_xy[lo:hi, 0] - L / 2, m.ball_xy[lo:hi, 1] - W / 2) <= centre_r
    v = np.nan_to_num(f.ball_v[lo:hi])
    kicked = v > kick_v
    for i in np.flatnonzero(near & formation):
        j = np.flatnonzero(kicked[i:i + FPS])
        if j.size:
            return lo + i + int(j[0])
    return None


def restart_teams(f: Features, segments: list, before: int = FPS // 2, after: int = FPS // 4,
                  radius: float = 3.0) -> dict:
    """Restart frame -> team of the taker (+1/-1), where the ball is visible near the restart.

    The taker is the team whose nearest player is closer to the ball, within `radius`, over
    [restart - before, restart + after]. Majority over visible frames; ties and no evidence
    give no entry.
    """
    out = {}
    for _, restart, _, why in segments:
        if why == "half_end":
            continue
        a, b = max(0, restart - before), restart + after
        dh, da = f.d_home[a:b], f.d_away[a:b]
        ok = f.ball_vis[a:b] & (np.fmin(dh, da) <= radius)
        if not ok.any():
            continue
        vote = np.sign(np.where(dh[ok] < da[ok], 1, -1).sum())
        if vote:
            out[int(restart)] = int(vote)
    return out


def detect(f: Features, graded: np.ndarray, cfg: DeadConfig = DeadConfig(),
           kickoffs: dict | None = None, prior: np.ndarray | None = None) -> DeadResult:
    """Dead mask over the match (graded frames only; outside is irrelevant).

    `kickoffs` maps a half's first graded frame to its kick-off frame; everything before the
    kick-off is dead (the second half's graded block starts about two minutes before it).
    `prior` marks set-piece windows from Smart Tagging (see cores()).
    """
    res = _detect_cores(f, graded, cfg, prior)
    for lo, k in (kickoffs or {}).items():
        if k is None or k - lo < 2 * FPS:
            continue  # the graded block already starts at the kick-off (half 1)
        res.dead[lo:k] = True
        res.conf[lo:k] = 1.0
        keep = [s for s in res.segments if s[1] <= lo or s[0] >= k]
        over = [s for s in res.segments if not (s[1] <= lo or s[0] >= k)]
        end = max([k] + [s[1] for s in over])
        why = "kickoff" if end == k else next(s[3] for s in over if s[1] == end)
        res.dead[lo:end] = True
        res.segments = sorted(keep + [(int(lo), int(end), "half_start", why)])
    res.dead &= graded
    return res


def _detect_cores(f: Features, graded: np.ndarray, cfg: DeadConfig,
                  prior: np.ndarray | None = None) -> DeadResult:
    n = len(graded)
    dead = np.zeros(n, dtype=bool)
    conf = np.zeros(n)
    act = np.where(graded, f.activity, np.nan)
    segs = []
    prev_end = 0
    hs, he = runs(graded)
    live_ball = (f.ball_vis & (np.nan_to_num(f.ball_out, nan=99.0) < cfg.live_interior)
                 & (np.nan_to_num(f.ball_v) > cfg.live_v))
    placed = np.nan_to_num(f.ball_v, nan=np.inf) < cfg.placed_v
    for (c0, c1) in cores(act, cfg, prior, placed):
        if live_ball[c0:c0 + max(1, (c1 - c0) // 2)].mean() > cfg.live_veto:
            continue  # slow build-up play, not a stoppage
        # half the core belongs to: clip searches to it
        h = np.searchsorted(he, c0, side="right")
        h_lo, h_hi = (hs[h], he[h]) if h < len(hs) else (0, n)
        lo = max(prev_end, h_lo)
        if c0 <= h_lo:
            onset, why_on = h_lo, "half_start"
        else:
            onset, why_on = find_onset(f, c0, lo, cfg)
        if c1 >= h_hi:
            restart, why_re = h_hi, "half_end"
        else:
            restart, why_re = find_restart(f, c1, h_hi, cfg)
        restart = max(restart, onset + 1)
        dead[onset:restart] = True
        depth = np.clip((cfg.leave - np.nanmean(act[c0:c1])) / cfg.leave, 0, 1)
        conf[onset:restart] = 0.5 + 0.5 * depth
        segs.append((int(onset), int(restart), why_on, why_re))
        prev_end = restart
    dead &= graded
    return DeadResult(dead=dead, conf=conf, segments=segs)
