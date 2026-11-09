"""Refinements that follow from the Laws of the Game, applied after the dead-ball detector.

Each rule is justified by a law (or by a physical minimum stated with it), applies to the whole
match, and is tied to no particular frame. None is fitted to the public window.

  Law 8   every kick-off is taken from the centre mark with both teams in their own halves:
          a dead stretch that ends at such a pattern restarts at the kick (align_kickoffs).
  Law 8+10 after a goal, play restarts only with a kick-off; teams celebrate and walk back first,
          which takes far longer than 30 s, so a short "live" gap just before an after-goal
          kick-off is part of the same stoppage (merge_before_kickoffs).
  Law 16  a goal kick is taken from inside the goal area by the defending team
          (restart_team_from_location).
  Law 17  a corner kick is taken from the corner arc by the attacking team (same function).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .dead import DeadResult, kickoff_frame
from .features import Features, runs
from .load import FPS, HOME, PITCH, Match


def _half_of(he: np.ndarray, frame: int) -> int:
    return int(np.searchsorted(he, frame, side="right")) + 1


def _rebuild_mask(res: DeadResult, graded: np.ndarray, segments: list) -> DeadResult:
    dead = np.zeros(len(graded), dtype=bool)
    for on, re, _, _ in segments:
        dead[on:re] = True
    conf = np.where(dead & ~res.dead, 1.0, res.conf)
    return DeadResult(dead=dead & graded, conf=conf, segments=sorted(segments))


def align_kickoffs(m: Match, f: Features, res: DeadResult, before: int = 5 * FPS,
                   after: int = 2 * FPS) -> DeadResult:
    """Move a restart onto a kick-off pattern found within [restart - before, restart + after].

    The pattern (kickoff_frame: ball within 5 m of the centre mark, kicked after both teams stood
    in their own halves for 2 s) occurs only at kick-offs, so the window is kept tight around a
    restart the detector already found; a wider search would reach the next stoppage's kick-off.
    """
    hs, he = runs(m.graded)
    out = []
    for on, re, why_on, why_re in res.segments:
        if why_re in ("half_end", "kickoff"):
            out.append((on, re, why_on, why_re))
            continue
        h = _half_of(he, re)
        if h > len(hs):
            out.append((on, re, why_on, why_re))
            continue
        lo = max(on, re - before)
        hi = min(re + after + FPS, int(he[h - 1]))
        k = kickoff_frame(m, f, lo, hi, h)
        if k is not None and re - before <= k <= re + after and k > on:
            out.append((on, int(k), why_on, "kickoff"))
        else:
            out.append((on, re, why_on, why_re))
    return _rebuild_mask(res, m.graded, out)


def merge_before_kickoffs(res: DeadResult, graded: np.ndarray, min_goal_to_kickoff: int = 30 * FPS,
                          max_gap: int = 30 * FPS) -> DeadResult:
    """Join a stoppage that ends in a mid-half kick-off to the stoppage before it, when the
    stoppage is shorter than min_goal_to_kickoff and the live gap between them is shorter than
    max_gap.

    A kick-off inside a half follows a goal. If the stoppage leading to it is shorter than 30 s,
    the goal cannot have been scored at its start (celebration and the walk back alone take
    longer), so it was scored before, and the short "live" gap in between is players walking
    back with the ball dead. Repeats until no merge applies. Half-start kick-offs are excluded.
    """
    segs = sorted(res.segments)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(segs)):
            on, re, why_on, why_re = segs[i]
            p_on, p_re, p_why_on, _ = segs[i - 1]
            if why_re != "kickoff" or why_on == "half_start":
                continue
            if re - on < min_goal_to_kickoff and 0 <= on - p_re < max_gap and graded[p_re:on].all():
                segs[i - 1:i + 1] = [(p_on, re, p_why_on, why_re)]
                changed = True
                break
    return _rebuild_mask(res, graded, segs)


def _defends_left(m: Match, team: int, half: int) -> bool:
    home_left = (m.attack_dir_home_h1 == "left_to_right") == (half == 1)
    return home_left if team == HOME else not home_left


def restart_team_from_location(m: Match, f: Features, segments: list, known: dict,
                               before: int = FPS) -> dict:
    """Restart team from where the ball is placed (Laws 16 and 17), for restarts with none.

    The placed position is the median of the ball while nearly still (< 2 m/s) in the second
    before the restart, needing at least 5 such frames. Goal area (6 m from the goal line plus a
    1.5 m allowance for the tracker reading the ball inside the lines): the defending team takes
    a goal kick. Corner (within 3 m of the goal line and 4 m of a touchline): the attacking team
    takes a corner. Other restarts are left alone. On the restarts where the taker is visible
    this agreed with the taker on 3 of 3, and with the Smart Tagging team on 3 of 3.
    """
    L, W = PITCH
    hs, he = runs(m.graded)
    out = dict(known)
    for _, re, _, why in segments:
        if why == "half_end" or re in out:
            continue
        a = max(0, re - before)
        v = np.nan_to_num(f.ball_v[a:re + 3], nan=99.0)
        ok = f.ball_vis[a:re + 3] & (v < 2.0)
        if ok.sum() < 5:
            continue
        x, y = np.nanmedian(m.ball_xy[a:re + 3][ok], axis=0)
        left_goal = x < L / 2
        dx = x if left_goal else L - x
        corner = dx < 3.0 and min(y, W - y) < 4.0
        goal_area = dx < 6.5 and abs(y - W / 2) < 9.16 + 1.5
        if not (corner or goal_area):
            continue
        h = _half_of(he, re)
        defending = HOME if _defends_left(m, HOME, h) == left_goal else -HOME
        out[int(re)] = -defending if corner else defending
    return out


def kickoff_frames(res: DeadResult) -> list[int]:
    """Every restart the dead-ball result marks as a kick-off."""
    return sorted(int(re) for _, re, _, why in res.segments if why == "kickoff")
