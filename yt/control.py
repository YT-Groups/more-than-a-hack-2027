"""Ball control detection: which team, if any, has a player on the ball.

Definition (CLAUDE.md section 4): a team has control when one of its players stays close to the
ball for a dwell period with no opponent challenging it. A confirmed run is stamped from its
first frame, so no second dwell is applied downstream.

Variants, compared in the technical document:

  C1  nearest player within r_in, dwell. No contest test, no hysteresis.
  C2  C1 + contest test (no opponent within contest_radius) + hysteresis (r_in to enter,
      r_out to stay). This is our definition.
  C3  C2 + the ball must move with the player: relative speed below rel_v.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .features import Features, rolling, runs
from .load import FPS, HOME, Match


@dataclass(frozen=True)
class ControlConfig:
    variant: str = "C3"
    r_in: float = 2.0
    r_out: float = 2.5
    contest_radius: float = 1.5
    dwell: int = 10
    rel_v: float = 4.0


@dataclass
class ControlResult:
    ctrl: np.ndarray        # (n,) int8 confirmed team: +1, -1, 0
    conf: np.ndarray        # (n,) float 0-1
    candidate: np.ndarray   # (n,) int8 per-frame candidate before the dwell
    reason: np.ndarray      # (n,) int8 abstention reason, see REASONS

REASONS = {0: "control", 1: "ball_missing", 2: "nobody_close", 3: "contested", 4: "short_run",
           5: "ball_not_with_player"}


def player_velocity(players_xy: np.ndarray, smooth: int = 5) -> np.ndarray:
    """(n, K, 2) velocity per slot from smoothed positions. Slot swaps make this noisy at the
    swap frame only, which the dwell absorbs."""
    n, k, _ = players_xy.shape
    out = np.empty_like(players_xy)
    for j in range(k):
        for c in range(2):
            x = rolling(players_xy[:, j, c], smooth, "mean")
            x[np.isnan(players_xy[:, j, c])] = np.nan
            out[:, j, c] = np.gradient(x) * FPS
    return out


def ball_velocity(ball_xy: np.ndarray, smooth: int = 5) -> np.ndarray:
    out = np.empty_like(ball_xy)
    vis = ~np.isnan(ball_xy[:, 0])
    for c in range(2):
        x = rolling(ball_xy[:, c], smooth, "mean")
        x[~vis] = np.nan
        out[:, c] = np.gradient(x) * FPS
    return out


def candidates(m: Match, f: Features, cfg: ControlConfig,
               pv: np.ndarray | None = None, bv: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame candidate team, distance of the nearest player, and abstention reason."""
    n = m.n
    d = np.linalg.norm(m.players_xy - m.ball_xy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    rows = np.arange(n)
    dn = d[rows, k]
    tn = m.team[k]
    d_opp = np.where(tn == HOME, f.d_away, f.d_home)
    d_opp = np.nan_to_num(d_opp, nan=np.inf)

    cand = np.zeros(n, dtype=np.int8)
    reason = np.full(n, 2, dtype=np.int8)
    reason[~f.ball_vis] = 1

    if cfg.variant == "C1":
        ok = f.ball_vis & (dn <= cfg.r_in)
        cand[ok] = tn[ok]
        reason[ok] = 0
        return cand, dn, reason

    rel_ok = np.ones(n, dtype=bool)
    if cfg.variant == "C3":
        if pv is None:
            pv = player_velocity(m.players_xy)
        if bv is None:
            bv = ball_velocity(m.ball_xy)
        rel = np.linalg.norm(bv - pv[rows, k], axis=1)
        rel_ok = ~(rel > cfg.rel_v)  # NaN relative speed does not veto

    # hysteresis: keep a team while its nearest player stays within r_out
    holding = 0
    for i in range(n):
        if not f.ball_vis[i]:
            holding = 0
            continue
        radius = cfg.r_out if holding == tn[i] else cfg.r_in
        if dn[i] > radius:
            holding = 0
            continue
        if d_opp[i] <= cfg.contest_radius:
            holding = 0
            reason[i] = 3
            continue
        if not rel_ok[i]:
            holding = 0
            reason[i] = 5
            continue
        holding = tn[i]
        cand[i] = holding
        reason[i] = 0
    return cand, dn, reason


def confirm(cand: np.ndarray, dwell: int) -> np.ndarray:
    """Keep candidate runs of at least `dwell` frames, stamped from their first frame."""
    out = np.zeros_like(cand)
    for team in (1, -1):
        s, e = runs(cand == team)
        for a, b in zip(s, e):
            if b - a >= dwell:
                out[a:b] = team
    return out


def backdate(ctrl: np.ndarray, proximity: np.ndarray) -> np.ndarray:
    """Extend each confirmed run back over the contiguous same-team proximity run it sits in.

    C3 rejects the first frames of a reception (ball still arriving at speed). Control is
    decided by C3 but timed from when the player first got close, as in C2.
    """
    out = ctrl.copy()
    for team in (1, -1):
        s, e = runs(proximity == team)
        for a, b in zip(s, e):
            if (ctrl[a:b] == team).any():
                first = a + int(np.argmax(ctrl[a:b] == team))
                out[a:first] = team
    return out


def detect(m: Match, f: Features, cfg: ControlConfig = ControlConfig(), **kw) -> ControlResult:
    cand, dn, reason = candidates(m, f, cfg, **kw)
    ctrl = confirm(cand, cfg.dwell)
    if cfg.variant == "C3":
        prox, _, _ = candidates(m, f, dataclasses.replace(cfg, variant="C2"))
        ctrl = backdate(ctrl, prox)
    reason = reason.copy()
    reason[(cand != 0) & (ctrl == 0)] = 4
    conf = np.where(ctrl != 0, np.clip(1.0 - dn / max(cfg.r_out, 1e-9), 0.0, 1.0), 0.0)
    return ControlResult(ctrl=ctrl, conf=conf, candidate=cand, reason=reason)
