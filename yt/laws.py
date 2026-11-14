"""Refinements that follow from the Laws of the Game, applied after the dead-ball detector.

Each rule is justified by a law (or by a physical minimum stated with it), applies to the whole
match, and is tied to no particular frame. None is fitted to the public window.

  Law 8   every kick-off is taken from the centre mark with both teams in their own halves:
          a dead stretch that ends at such a pattern restarts at the kick (align_kickoffs).
  Law 8+10 after a goal, play restarts only with a kick-off; teams celebrate and walk back first,
          which takes far longer than 30 s, so a short "live" gap just before an after-goal
          kick-off is part of the same stoppage (merge_before_kickoffs).

A third rule was tested and dropped: Laws 16-17 (a ball placed in a goal area or at a corner arc
names the restarting team). It agreed with the visible taker and the Smart Tagging team on 3 of
3, but on this match it changed no label, so the simpler pipeline was kept.
"""

from __future__ import annotations

import numpy as np

from .dead import DeadResult, kickoff_frame
from .features import Features, runs
from .load import FPS, Match


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


def kickoff_frames(res: DeadResult) -> list[int]:
    """Every restart the dead-ball result marks as a kick-off."""
    return sorted(int(re) for _, re, _, why in res.segments if why == "kickoff")
