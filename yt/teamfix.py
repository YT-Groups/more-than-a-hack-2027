"""Correct tracklets whose team label breaks the laws of the game.

At a kick-off every player must be in his own half (Law 8). A tracked slot standing clearly in
the opponents' half one second before a kick-off therefore carries the wrong team label for that
tracklet (a contiguous run of the slot). On this match it finds two tracklets of the away
goalkeeper's slot standing deep in the home half at both half-start kick-offs.

Slots carry one team label per column, so frames are not relabelled in place: a flagged tracklet
moves to an extra column with the other team's label, and every other module keeps its interface.

Experimental and not used by run_pipeline.py. A broader rule (a goalkeeper slot near the other
team's goal belongs to the team defending that goal) was tested and rejected: in every such
stretch the real home goalkeeper was also visible, so the slot was not following a goalkeeper.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .features import runs
from .load import HOME, PITCH, Match


def defends_left(m: Match, team: int, half: int) -> bool:
    """True if `team` defends the x = 0 goal in `half` (1 or 2)."""
    home_left = (m.attack_dir_home_h1 == "left_to_right") == (half == 1)
    return home_left if team == HOME else not home_left


def kickoff_violations(m: Match, kickoffs: list[int], margin: float = 5.0,
                       circle: float = 9.15, before: int = 25) -> list[tuple[int, int, int]]:
    """Tracklets that stand clearly in the opponents' half one second before a kick-off.

    At a kick-off every player must be in his own half (Law 8), so a slot more than `margin`
    metres past the halfway line, outside the centre circle, has the wrong team label for that
    tracklet (a contiguous run of the slot). Returns (slot, tracklet_start, tracklet_end).
    """
    L, W = PITCH
    hs, he = runs(m.graded)
    out = []
    for k in kickoffs:
        h = int(np.searchsorted(he, k, side="right"))
        if h >= len(hs) or not hs[h] <= k < he[h]:
            continue
        t = k - before
        x, y = m.players_xy[t, :, 0], m.players_xy[t, :, 1]
        for j in np.flatnonzero(~np.isnan(x)):
            own_left = defends_left(m, int(m.team[j]), h + 1)
            past = (x[j] - L / 2) if own_left else (L / 2 - x[j])
            if past > margin and np.hypot(x[j] - L / 2, y[j] - W / 2) > circle:
                s, e = runs(~np.isnan(m.players_xy[:, j, 0]))
                i = int(np.searchsorted(e, t, side="right"))
                out.append((int(j), int(s[i]), int(e[i])))
    return sorted(set(out))


def fix_tracklets(m: Match, tracklets: list[tuple[int, int, int]]) -> Match:
    """Move each (slot, start, end) tracklet to an extra slot with the opposite team label."""
    if not tracklets:
        return m
    xy, sp = m.players_xy.copy(), m.player_speed.copy()
    new_xy, new_sp, new_team, new_gk = [], [], [], []
    for j, a, b in tracklets:
        col_xy = np.full((m.n, 2), np.nan)
        col_sp = np.full(m.n, np.nan)
        col_xy[a:b], col_sp[a:b] = xy[a:b, j], sp[a:b, j]
        xy[a:b, j], sp[a:b, j] = np.nan, np.nan
        new_xy.append(col_xy); new_sp.append(col_sp)
        new_team.append(-m.team[j]); new_gk.append(False)
    return dataclasses.replace(
        m,
        players_xy=np.concatenate([xy, np.stack(new_xy, axis=1)], axis=1),
        player_speed=np.concatenate([sp, np.stack(new_sp, axis=1)], axis=1),
        team=np.concatenate([m.team, np.array(new_team, dtype=m.team.dtype)]),
        is_gk=np.concatenate([m.is_gk, np.array(new_gk, dtype=bool)]),
    )
