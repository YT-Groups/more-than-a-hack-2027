"""Smart Tagging as evidence (never as truth).

Two uses:
  * set-piece windows are a prior for the dead-ball detector: inside one, a shorter and
    shallower dip in player activity counts as a stoppage if the ball is also still;
  * corners, free kicks and goal kicks name the taking team, which fills in the restarting team
    where the ball is not visible at the restart.

Windows are fixed-length (35 s for most set pieces, 5 s for kick-offs) and say a set piece
happened somewhere inside, not when. Team names are matched case-insensitively and the away
side is spelled "Atletic Club" in the file.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .load import AWAY, HOME, ROOT

sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402  (organisers' loader, standard library only)

TEAM = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY}
TEAM_NAMED_TYPES = {"corner kick", "free kick", "goal kick"}


def load() -> list[dict]:
    return load_smart_tagging.load()


def set_pieces(events: list[dict]) -> list[dict]:
    return [e for e in events if (e["code"] or "").strip().upper() == "SET PIECES"]


def prior_mask(events: list[dict], n: int) -> np.ndarray:
    """True inside any set-piece window."""
    mask = np.zeros(n, dtype=bool)
    for e in set_pieces(events):
        mask[max(0, e["start_frame"]):min(n, e["end_frame"])] = True
    return mask


def team_hints(events: list[dict]) -> list[tuple[int, int, int]]:
    """(start, end, team) for set pieces whose tag names the taking team."""
    out = []
    for e in set_pieces(events):
        typ = (e["tags"].get("Type") or "").strip().lower()
        team = TEAM.get((e["tags"].get("Team") or "").strip().lower())
        if typ in TEAM_NAMED_TYPES and team:
            out.append((e["start_frame"], e["end_frame"], team))
    return out


def fill_restart_teams(restart_team: dict, segments: list, hints: list) -> dict:
    """Add the tagged team for restarts with no estimate that fall inside a hinted window.

    Our own estimate (taker nearest the ball) is kept when present; on the 11 tagged restarts
    where both exist they agree on 10.
    """
    out = dict(restart_team)
    for _, restart, _, why in segments:
        if why == "half_end" or restart in out:
            continue
        for a, b, team in hints:
            if a <= restart < b:
                out[int(restart)] = team
                break
    return out
