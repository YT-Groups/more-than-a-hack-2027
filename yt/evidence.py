"""Architectural evidence fusion for restart and handover research.

This module is opt-in and does not change production labels by itself. It provides
dynamic restart alignment and weighted multi-source team confidence for audits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Features
from .load import FPS, Match


@dataclass(frozen=True)
class RestartEvidenceConfig:
    search_before: int = 2 * FPS
    search_after: int = 4 * FPS
    acceleration_threshold: float = 4.0
    activity_threshold: float = 0.5
    spatial_radius: float = 3.0


def dynamic_restart_alignment(
    m: Match,
    f: Features,
    center: int,
    cfg: RestartEvidenceConfig = RestartEvidenceConfig(),
) -> tuple[int | None, float, dict]:
    """Find the strongest physical restart candidate around a coarse boundary."""
    a = max(0, int(center) - cfg.search_before)
    b = min(m.n, int(center) + cfg.search_after)
    if b <= a + 2:
        return None, 0.0, {"frame": None, "reason": "insufficient_window"}

    v = np.nan_to_num(f.ball_v[a:b], nan=0.0)
    acceleration = np.abs(np.diff(v, prepend=v[0]))
    activity = np.nan_to_num(f.activity[a:b], nan=0.0)
    activity_rise = np.diff(activity, prepend=activity[0])

    near_home = np.nan_to_num(f.d_home[a:b], nan=np.inf)
    near_away = np.nan_to_num(f.d_away[a:b], nan=np.inf)
    spatial_margin = near_away - near_home
    spatial_quality = np.clip((cfg.spatial_radius - np.fmin(near_home, near_away))
                              / cfg.spatial_radius, 0.0, 1.0)

    ball_exit = np.clip(acceleration / cfg.acceleration_threshold, 0.0, 1.0)
    player_shift = np.clip(activity_rise / cfg.activity_threshold, 0.0, 1.0)
    visible = f.ball_vis[a:b].astype(float)
    spatial = spatial_quality * np.clip(np.abs(spatial_margin) / cfg.spatial_radius, 0.0, 1.0)

    score = np.nan_to_num(
        0.50 * ball_exit + 0.30 * player_shift + 0.20 * spatial * visible,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    local_best = int(np.argmax(score))
    frame = a + local_best
    confidence = float(np.clip(score[local_best], 0.0, 1.0))

    evidence = {
        "frame": int(frame),
        "confidence": confidence,
        "ball_acceleration_mps2": float(acceleration[local_best]),
        "activity_rise_mps": float(activity_rise[local_best]),
        "spatial_margin_m": float(abs(spatial_margin[local_best]))
        if np.isfinite(spatial_margin[local_best]) else None,
        "search": [int(a), int(b)],
    }
    if confidence <= 0.0:
        return None, 0.0, evidence
    return frame, confidence, evidence


def fuse_transition_team(
    spatial_team: int | None,
    spatial_confidence: float,
    ball_exit_team: int | None,
    ball_exit_confidence: float,
    tag_team: int | None,
    tag_confidence: float = 0.25,
) -> tuple[int | None, float, dict[str, float]]:
    """Fuse independent team signals into one sign and confidence."""
    weights = {
        1: 0.50 * spatial_confidence if spatial_team == 1 else 0.0,
        -1: 0.50 * spatial_confidence if spatial_team == -1 else 0.0,
    }
    for team, team_confidence, weight in (
        (ball_exit_team, ball_exit_confidence, 0.35),
        (tag_team, tag_confidence, 0.15),
    ):
        if team == 1:
            weights[1] += weight * float(np.clip(team_confidence, 0.0, 1.0))
        elif team == -1:
            weights[-1] += weight * float(np.clip(team_confidence, 0.0, 1.0))

    total = weights[1] + weights[-1]
    if total <= 1e-9:
        return None, 0.0, weights
    team = 1 if weights[1] > weights[-1] else -1
    confidence = float(abs(weights[1] - weights[-1]) / total)
    return team, confidence, weights
