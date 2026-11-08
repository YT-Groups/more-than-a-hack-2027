"""Conservative pass and reception evidence.

This module detects ball flight followed by stable same-team proximity. It deliberately
does not infer ownership from isolated nearest-player changes: a reception is an event only
when the tracking shows a passing team, visible flight, and a sustained receiver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Features, runs
from .load import FPS, Match


@dataclass(frozen=True)
class ReceptionConfig:
    flight_v: float = 7.0
    flight_min: int = 3
    flight_max: int = 75
    passer_search: int = 10
    receiver_search: int = 75
    receiver_radius: float = 2.5
    receiver_dwell: int = 10
    team_margin: float = 1.0
    minimum_confidence: float = 0.55
    require_control: bool = True


@dataclass
class ReceptionResult:
    handover: dict[int, int]
    confidence: dict[int, float]
    events: list[dict]
    counts: dict[str, int] | None = None


def detect(m: Match, f: Features, ctrl: np.ndarray | None = None,
           dead: np.ndarray | None = None, cfg: ReceptionConfig = ReceptionConfig()
           ) -> ReceptionResult:
    """Find high-confidence pass receptions and return frame -> receiving team."""
    n = m.n
    ball_v = np.nan_to_num(f.ball_v, nan=0.0)
    ball_vis = f.ball_vis.copy()
    fast = ball_vis & (ball_v >= cfg.flight_v)
    start, end = runs(fast)
    flights = [(a, b) for a, b in zip(start, end)
               if b - a >= cfg.flight_min and b - a <= cfg.flight_max]

    handover: dict[int, int] = {}
    confidence: dict[int, float] = {}
    events: list[dict] = []
    counts = {
        "flights": len(flights),
        "passer_candidates": 0,
        "dead_rejected": 0,
        "layer3_rejected": 0,
        "events": 0,
    }

    for flight_start, flight_end in flights:
        if dead is not None and bool(np.any(dead[flight_start:min(flight_end + 1, n)])):
            counts["dead_rejected"] += 1
            continue

        passer_team, passer_confidence = _passer_team(
            m, flight_start, cfg.passer_search)
        if passer_team == 0 or passer_confidence < cfg.minimum_confidence:
            continue
        counts["passer_candidates"] += 1

        reception = _stable_receiver(
            m, passer_team, flight_start, flight_end, ctrl, dead, cfg)
        if reception is None:
            counts["layer3_rejected"] += 1
            continue

        frame, receiver_confidence, receiver_player = reception
        event_confidence = float(min(passer_confidence, receiver_confidence))
        if event_confidence < cfg.minimum_confidence:
            continue
        handover[frame] = passer_team
        confidence[frame] = event_confidence
        counts["events"] += 1
        events.append({
            "frame": int(frame),
            "team": int(passer_team),
            "confidence": event_confidence,
            "flight_start": int(flight_start),
            "flight_end": int(flight_end),
            "receiver_player": int(receiver_player),
        })
    return ReceptionResult(handover=handover, confidence=confidence, events=events,
                           counts=counts)


def _passer_team(m: Match, flight_start: int, search: int) -> tuple[int, float]:
    a = max(0, flight_start - search)
    b = flight_start
    if b <= a:
        return 0, 0.0
    d = np.linalg.norm(m.players_xy[a:b] - m.ball_xy[a:b, None, :], axis=2)
    nearest_slot = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
    nearest_distance = d[np.arange(len(nearest_slot)), nearest_slot]
    nearest_team = m.team[nearest_slot]

    valid = np.isfinite(nearest_distance) & (nearest_distance <= 4.0)
    if not np.any(valid):
        return 0, 0.0

    home = nearest_team[valid] == 1
    away = nearest_team[valid] == -1
    n_home, n_away = int(home.sum()), int(away.sum())
    total = n_home + n_away
    if total == 0:
        return 0, 0.0
    share = max(n_home, n_away) / total
    team = 1 if n_home >= n_away else -1
    confidence = 0.5 + 0.5 * (share - 0.5) * 2
    return team, float(min(1.0, confidence))


def _stable_receiver(m: Match, team: int, flight_start: int, flight_end: int,
                     ctrl: np.ndarray | None, dead: np.ndarray | None,
                     cfg: ReceptionConfig) -> tuple[int, float, int] | None:
    n = m.n
    a = min(n, max(0, flight_end - 3))
    b = min(n, flight_end + cfg.receiver_search)
    if b - a < cfg.receiver_dwell:
        return None

    ball_xy = m.ball_xy[a:b]
    ball_vis = ~np.isnan(ball_xy[:, 0])
    distances = np.linalg.norm(m.players_xy[a:b] - ball_xy[:, None, :], axis=2)
    distances = np.where(np.isnan(distances), np.inf, distances)
    team_slots = np.where(m.team == team)[0]
    opp_slots = np.where(m.team == -team)[0]

    for frame_offset in range(0, b - a - cfg.receiver_dwell + 1):
        frame = a + frame_offset
        win = slice(frame_offset, frame_offset + cfg.receiver_dwell)
        if dead is not None and np.any(dead[frame:frame + cfg.receiver_dwell]):
            continue
        if not np.any(ball_vis[win]):
            continue

        d_team = distances[win][:, team_slots]
        d_opp = distances[win][:, opp_slots]
        same_nearest = np.nanmin(d_team, axis=1)
        opp_nearest = np.nanmin(d_opp, axis=1)
        close = (same_nearest <= cfg.receiver_radius) & np.isfinite(same_nearest)
        separated = np.isfinite(opp_nearest) & (same_nearest + cfg.team_margin <= opp_nearest)
        stable = close & (separated | ~np.isfinite(opp_nearest))
        if stable.mean() < 0.8:
            continue

        same_slots = np.argmin(d_team, axis=1)
        receiver_slot = int(team_slots[same_slots[stable.argmax()]])
        share = float(stable.mean())

        if cfg.require_control and ctrl is not None:
            confirmed = ctrl[frame:frame + cfg.receiver_dwell] == team
            ctrl_share = float(np.mean(confirmed))
            if ctrl_share < 0.8:
                continue
            share = min(share, ctrl_share)

        margin = np.nanmedian(opp_nearest[stable] - same_nearest[stable])
        confidence = 0.5 + 0.3 * share + 0.2 * np.clip(
            margin / cfg.team_margin, 0.0, 1.0)
        if confidence >= cfg.minimum_confidence:
            return frame, float(confidence), receiver_slot
    return None
