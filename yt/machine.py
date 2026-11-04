"""State machine: confirmed control + dead mask -> H / A / D / X per frame.

Rules (docs/YT.md section 5):
  1. dead frames are D, with no owner;
  2. in play, possession changes only at the first frame of a confirmed control run by the
     other team (the control detector already applied the dwell);
  3. between control runs the last owner keeps the ball;
  4. after a dead ball the owner is the restarting team if known, otherwise the team of the
     first control run, back-filled to the restart frame;
  5. the owner resets at the start of each half, and ungraded frames are X;
  6. a possession shorter than min_len is merged into its neighbours only when both are the
     same team inside the same live stretch (never across D);
  7. X is used only where no owner can be named at all (before any evidence in a half).
"""

from __future__ import annotations

import numpy as np

from .features import runs

CODE = {1: "H", -1: "A"}


def _live_segments(dead: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    s, e = runs(~dead[lo:hi])
    return [(lo + a, lo + b) for a, b in zip(s, e)]


def _fill_segment(ctrl: np.ndarray, a: int, b: int, start_owner: int) -> np.ndarray:
    """Owner per frame for one live stretch, given who owns it at frame a (0 = unknown)."""
    seg = ctrl[a:b]
    owner = np.zeros(b - a, dtype=np.int8)
    # run starts of confirmed control
    change = np.flatnonzero((seg != 0) & (np.r_[0, seg[:-1]] != seg))
    cur = start_owner
    if cur == 0 and change.size:
        cur = int(seg[change[0]])  # back-fill to the restart
    last = 0
    for c in change:
        owner[last:c] = cur
        if seg[c] != cur:
            cur = int(seg[c])
        last = c
    owner[last:] = cur
    return owner


def _merge_short(owner: np.ndarray, min_len: int) -> np.ndarray:
    """Absorb runs shorter than min_len whose two neighbours are the same team."""
    owner = owner.copy()
    changed = True
    while changed:
        changed = False
        v = owner
        bounds = np.flatnonzero(np.r_[True, v[1:] != v[:-1], True])
        for i in range(1, len(bounds) - 2):
            a, b = bounds[i], bounds[i + 1]
            left, right = v[bounds[i - 1]], v[b]
            if b - a < min_len and left == right and left != 0 and v[a] != left:
                owner[a:b] = left
                changed = True
                break
    return owner


def label(ctrl: np.ndarray, dead: np.ndarray, graded: np.ndarray, half: np.ndarray,
          restart_team: dict | None = None, min_len: int = 25) -> np.ndarray:
    """Per-frame labels as a numpy array of single characters."""
    n = len(ctrl)
    restart_team = restart_team or {}
    out = np.full(n, "X", dtype="<U1")
    hs, he = runs(graded)
    for lo, hi in zip(hs, he):
        owner = 0
        for a, b in _live_segments(dead, lo, hi):
            start = int(restart_team.get(a, 0))
            seg_owner = _fill_segment(ctrl, a, b, start)
            if not seg_owner.any() and owner:
                seg_owner[:] = owner  # nothing seen: keep the last owner
            seg_owner = _merge_short(seg_owner, min_len)
            for team, ch in CODE.items():
                out[a:b][seg_owner == team] = ch
            if seg_owner.any():
                owner = int(seg_owner[-1])
        d = dead[lo:hi]
        out[lo:hi][d] = "D"
    return out
