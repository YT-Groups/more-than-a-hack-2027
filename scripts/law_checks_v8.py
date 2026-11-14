"""Read-only, in-memory audit of the v8 restart and kick-off rules, from tracking and tags only.

    python scripts/law_checks_v8.py

Prints, without writing any file:
  1. the owner at every detected kick-off restart, with the full pipeline and with every Smart
     Tagging `Team` value blanked (tag times and types kept), to show whether owners depend on
     tag team hints;
  2. the Law 8 side check at every kick-off, before and after the tracklet team correction;
  3. the tracklets the correction moved;
  4. every dead stretch longer than 60 s, with tracking-only evidence (ball visibility, longest
     ball gap, player activity) and the Smart Tagging types overlapping it.

No ground truth and no human-produced data about the match are read.
"""

from __future__ import annotations

import copy
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pipeline import build  # noqa: E402
from yt import dead, features, tags  # noqa: E402
from yt.load import AWAY, HOME, load_match  # noqa: E402
from yt.teamfix import kickoff_violations  # noqa: E402

FPS = 25


def team_name(team: int | None) -> str:
    return "H" if team == HOME else "A" if team == AWAY else "?"


def first_live(labels: np.ndarray, frame: int, limit: int = 5 * FPS) -> int | None:
    """First H or A label at or after `frame`, within `limit` frames."""
    for i in range(frame, min(len(labels), frame + limit)):
        if labels[i] in ("H", "A"):
            return i
    return None


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    s, e = features.runs(mask)
    return int((e - s).max()) if len(s) else 0


def main() -> None:
    match = load_match()
    f0 = features.compute(match)
    events = tags.load()
    labels, full = build(m=match, f=f0)

    blank = copy.deepcopy(events)
    for event in blank:
        event.setdefault("tags", {})["Team"] = ""
    with patch("run_pipeline.tags.load", return_value=blank):
        blank_labels, blanked = build(m=match, f=f0)

    starts, ends = features.runs(match.graded)
    half_starts = [k for h, (a, b) in enumerate(zip(starts, ends), start=1)
                   if (k := dead.kickoff_frame(full["match"], full["features"], int(a), int(b), h))]
    kick_restarts = [int(r) for _, r, _, why in full["dead"].segments if why == "kickoff"]

    print("1. OWNER AT KICK-OFFS (full pipeline / Smart Tagging team values blanked)")
    for frame in sorted(set(half_starts) | set(kick_restarts)):
        a, b = first_live(labels, frame), first_live(blank_labels, frame)
        print("   frame {}: owner {} / {}; first live frame {} / {}".format(
            frame, labels[a] if a is not None else "?", blank_labels[b] if b is not None else "?",
            a, b))
    changed = int((labels[match.graded] != blank_labels[match.graded]).sum())
    print("   graded frames that differ when tag team values are blanked:", changed)

    print("2. LAW 8 SIDE CHECK AT KICK-OFFS (raw tracking / after the tracklet correction)")
    for frame in sorted(set(half_starts) | set(kick_restarts)):
        print("   frame {}: visible slots {}; violations {} / {}".format(
            frame, int(np.isfinite(match.players_xy[frame - FPS, :, 0]).sum()),
            kickoff_violations(match, [frame]), kickoff_violations(full["match"], [frame])))

    print("3. TRACKLETS MOVED BY THE CORRECTION:", full["team_fixes"])

    print("4. DEAD STRETCHES OVER 60 S (tracking-only evidence)")
    f = full["features"]
    for start, end, why_on, why_re in full["dead"].segments:
        if end - start <= 60 * FPS:
            continue
        vis = f.ball_vis[start:end]
        act = f.activity[start:end]
        types = sorted({(e["tags"].get("Type") or e["code"] or "").strip()
                        for e in events if e["start_frame"] < end and e["end_frame"] > start
                        and (e["code"] or "").strip().upper() in ("SET PIECES", "GOALS")})
        print("   [{}, {}) {:.1f} s, {}/{}: ball visible {:.0f}%, longest ball gap {:.1f} s, "
              "player activity median {:.2f} m/s, above 2.5 m/s {:.0f}%; tags: {}".format(
                  start, end, (end - start) / FPS, why_on, why_re, 100 * vis.mean(),
                  longest_run(~vis) / FPS, np.nanmedian(act), 100 * np.nanmean(act > 2.5),
                  ", ".join(types) or "none"))


if __name__ == "__main__":
    main()
