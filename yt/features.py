"""Per-frame features shared by the control and dead-ball detectors.

Everything is indexed by 0-based video frame and measured in metres and m/s.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .load import FPS, HOME, PITCH, Match


def rolling(x: np.ndarray, window: int, how: str = "median") -> np.ndarray:
    """Centred rolling statistic that ignores NaN (min one observation)."""
    r = pd.Series(x).rolling(window, center=True, min_periods=1)
    return getattr(r, how)().to_numpy(copy=True)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of the True runs of a boolean array."""
    m = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


@dataclass
class Features:
    activity: np.ndarray        # p90 player speed, rolling median over 1 s
    activity_raw: np.ndarray    # p90 player speed per frame
    ball_vis: np.ndarray        # bool
    ball_v: np.ndarray          # ball speed from displacement, m/s, light smoothing
    ball_out: np.ndarray        # metres outside the pitch (negative = inside)
    d_home: np.ndarray          # nearest home player distance to ball
    d_away: np.ndarray          # nearest away player distance to ball
    n_objects: np.ndarray       # tracked objects per frame
    near_slot: np.ndarray | None = None  # slot nearest the ball (-1 if none); identity is only
    near_d: np.ndarray | None = None     # trusted over a second or two, never across a match
    ball_xy: np.ndarray | None = None    # (n, 2) metres, as loaded


def ball_speed(ball_xy: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Displacement speed (m/s), centred, from positions smoothed over `smooth` frames.

    The tracker's own ball speed channel is NaN on 91% of visible frames, so it is not used.
    """
    x = rolling(ball_xy[:, 0], smooth, "mean")
    y = rolling(ball_xy[:, 1], smooth, "mean")
    vis = ~np.isnan(ball_xy[:, 0])
    x[~vis] = np.nan
    y[~vis] = np.nan
    vx = np.gradient(x) * FPS
    vy = np.gradient(y) * FPS
    v = np.hypot(vx, vy)
    v[~vis] = np.nan
    return v


def nearest_by_team(ball_xy: np.ndarray, players_xy: np.ndarray, team: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the ball to the nearest home and nearest away player (NaN if none)."""
    d = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, intended
        d_home = np.nanmin(np.where(team == HOME, d, np.nan), axis=1)
        d_away = np.nanmin(np.where(team != HOME, d, np.nan), axis=1)
    return d_home, d_away


def compute(m: Match) -> Features:
    """All features for a match."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        act_raw = np.nanpercentile(m.player_speed, 90, axis=1)
    act = rolling(act_raw, FPS, "median")
    L, W = PITCH
    b = m.ball_xy
    out = np.maximum.reduce([-b[:, 0], b[:, 0] - L, -b[:, 1], b[:, 1] - W])
    d_home, d_away = nearest_by_team(b, m.players_xy, m.team)
    d = np.linalg.norm(m.players_xy - b[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    slot = np.argmin(d, axis=1)
    near_d = d[np.arange(len(d)), slot]
    slot = np.where(np.isfinite(near_d), slot, -1)
    return Features(
        activity=act, activity_raw=act_raw, ball_vis=~np.isnan(b[:, 0]),
        ball_v=ball_speed(b), ball_out=out, d_home=d_home, d_away=d_away,
        n_objects=(~np.isnan(m.players_xy[:, :, 0])).sum(1),
        near_slot=slot, near_d=near_d, ball_xy=b,
    )
