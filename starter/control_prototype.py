"""Prototype control detection from the tactical ATD.

This implements only the H/A control-evidence stage. It does not decide whether the ball is
in play, so the output uses X outside confirmed control runs rather than guessing D.

    python starter/control_prototype.py --out control_prototype.csv
    python evaluation/validate_submission.py control_prototype.csv
    python evaluation/run_evaluation.py --results .

Distances are measured in metres. A candidate must be within r_in and have no opponent within
contest_radius. A candidate run lasting dwell_frames is confirmed and backdated to its first
frame. Hysteresis retains an existing candidate out to r_out.
"""

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402


@dataclass(frozen=True)
class ControlConfig:
    r_in: float = 2.0
    r_out: float = 2.5
    contest_radius: float = 1.5
    dwell_frames: int = 10
    max_ball_gap: int = 10
    max_ball_speed: float = 35.0
    chunk_frames: int = 4096


def _team_signs(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    if any(team_id not in {teams[0].get("id"), teams[1].get("id")} for team_id in slots):
        raise ValueError("{}: a track slot has an unknown team".format(meta_path))
    return np.array([1 if team_id == home_id else -1 for team_id in slots], dtype=np.int8)


def load_tracking(game=GAME, data_dir=None):
    """Load ATD positions as metres, indexed by 0-based tactical video frame.

    Returns ball_xy (n, 2), players_xy (n, slots, 2), and signed team codes (slots,).
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    teams = _team_signs(meta_path)
    pitch_length, pitch_width = GAMES[game]["pitch"]
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players_xy = np.full((n_frames, len(teams), 2), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}:{}: expected three ATD fields".format(raw_path, line_number))
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                raise ValueError("{}:{}: invalid frame id".format(raw_path, line_number))
            if not 0 <= frame < n_frames or seen[frame]:
                raise ValueError("{}:{}: duplicate or out-of-range frame id".format(
                    raw_path, line_number))

            player_groups = parts[1].split(";")
            if len(player_groups) != len(teams):
                raise ValueError("{}:{}: expected {} player slots, found {}".format(
                    raw_path, line_number, len(teams), len(player_groups)))
            for slot, group in enumerate(player_groups):
                values = group.split(",", 2)
                if len(values) < 2 or values[0].startswith("N") or values[1].startswith("N"):
                    continue
                x, y = float(values[0]), float(values[1])
                if np.isfinite(x) and np.isfinite(y):
                    players_xy[frame, slot] = (x * pitch_length, y * pitch_width)

            values = parts[2].split(",", 2)
            if len(values) >= 2 and not values[0].startswith("N") and not values[1].startswith("N"):
                x, y = float(values[0]), float(values[1])
                if np.isfinite(x) and np.isfinite(y):
                    ball_xy[frame] = (x * pitch_length, y * pitch_width)
            seen[frame] = True

    missing = int((~seen).sum())
    if missing:
        raise ValueError("{}: missing {} of {} video frames".format(raw_path, missing, n_frames))
    return ball_xy, players_xy, teams


def clean_ball_track(ball_xy, config=ControlConfig()):
    """Reject implausible jumps and interpolate only short, plausible internal gaps.

    Returns the cleaned coordinates and a confidence weight: 1 for observed positions, a value
    below 1 for interpolated positions, and 0 where no usable position is available.
    """
    ball = np.asarray(ball_xy, dtype=np.float64).copy()
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy must have shape (frames, 2)")

    valid = np.isfinite(ball).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) > 1:
        steps = np.diff(indices)
        speeds = np.linalg.norm(np.diff(ball[indices], axis=0), axis=1) / steps
        implausible = indices[1:][speeds > config.max_ball_speed]
        valid[implausible] = False
        ball[~valid] = np.nan

    confidence = valid.astype(np.float32)
    indices = np.flatnonzero(valid)
    for left, right in zip(indices[:-1], indices[1:]):
        gap = right - left - 1
        if gap == 0 or gap > config.max_ball_gap:
            continue
        speed = np.linalg.norm(ball[right] - ball[left]) / (right - left)
        if speed > config.max_ball_speed:
            continue
        steps = np.arange(1, gap + 1, dtype=np.float64)[:, None] / (right - left)
        ball[left + 1:right] = ball[left] + steps * (ball[right] - ball[left])
        confidence[left + 1:right] = max(0.1, 1.0 - gap / (2.0 * (config.max_ball_gap + 1)))
    return ball, confidence


def detect_control(ball_xy, players_xy, player_team, config=ControlConfig()):
    """Return confirmed control (+1 home, -1 away), confidence, and raw candidates.

    A zero in confirmed control means this detector did not confirm a team; it is not a dead-ball
    label. Candidate runs are backdated only after they reach the configured dwell.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    players = np.asarray(players_xy, dtype=np.float64)
    teams = np.asarray(player_team, dtype=np.int8)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy must have shape (frames, 2)")
    if players.ndim != 3 or players.shape[0] != len(ball) or players.shape[2] != 2:
        raise ValueError("players_xy must have shape (frames, slots, 2)")
    if teams.shape != (players.shape[1],) or not np.isin(teams, (-1, 1)).all():
        raise ValueError("player_team must contain one +1 or -1 value per slot")
    if config.dwell_frames < 1 or config.chunk_frames < 1:
        raise ValueError("dwell_frames and chunk_frames must be positive")
    if not 0 < config.r_in <= config.r_out or config.contest_radius <= 0:
        raise ValueError("radii must satisfy 0 < r_in <= r_out and contest_radius > 0")

    ball, ball_confidence = clean_ball_track(ball, config)
    count = len(ball)
    candidates = np.zeros(count, dtype=np.int8)
    candidate_confidence = np.zeros(count, dtype=np.float32)
    home_slots, away_slots = teams == 1, teams == -1
    if not home_slots.any() or not away_slots.any():
        raise ValueError("both teams must have at least one track slot")

    previous = 0
    for start in range(0, count, config.chunk_frames):
        stop = min(count, start + config.chunk_frames)
        valid_ball = ball_confidence[start:stop] > 0
        delta = players[start:stop] - ball[start:stop, None, :]
        distances = np.linalg.norm(delta, axis=2)
        distances[~np.isfinite(distances)] = np.inf
        home_distance = np.min(np.where(home_slots, distances, np.inf), axis=1)
        away_distance = np.min(np.where(away_slots, distances, np.inf), axis=1)

        for offset, (home, away, visible) in enumerate(
                zip(home_distance, away_distance, valid_ball)):
            if not visible:
                previous = 0
                continue

            entry_home = home <= config.r_in and away >= config.contest_radius
            entry_away = away <= config.r_in and home >= config.contest_radius
            keep_home = (previous == 1 and home <= config.r_out
                         and away >= config.contest_radius)
            keep_away = (previous == -1 and away <= config.r_out
                         and home >= config.contest_radius)

            if keep_home:
                team, own, opponent = 1, home, away
            elif keep_away:
                team, own, opponent = -1, away, home
            elif entry_home and entry_away:
                team, own, opponent = ((1, home, away) if home < away
                                       else (-1, away, home))
            elif entry_home:
                team, own, opponent = 1, home, away
            elif entry_away:
                team, own, opponent = -1, away, home
            else:
                team = 0
                previous = 0
                continue

            frame = start + offset
            candidates[frame] = team
            proximity = np.clip(1.0 - own / config.r_out, 0.0, 1.0)
            separation = np.clip(
                (opponent - config.contest_radius) / config.r_out, 0.0, 1.0)
            candidate_confidence[frame] = (
                ball_confidence[frame] * proximity * separation)
            previous = team

    confirmed = np.zeros(count, dtype=np.int8)
    confidence = np.zeros(count, dtype=np.float32)
    boundaries = np.r_[0, np.flatnonzero(candidates[1:] != candidates[:-1]) + 1, count]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        team = candidates[start]
        if team and stop - start >= config.dwell_frames:
            confirmed[start:stop] = team
            confidence[start:stop] = candidate_confidence[start:stop]
    return confirmed, confidence, candidates


def write_control_csv(path, confirmed_control):
    """Write an exploratory H/A/X CSV; X means unconfirmed, not necessarily dead."""
    control = np.asarray(confirmed_control, dtype=np.int8)
    if control.ndim != 1 or not np.isin(control, (-1, 0, 1)).all():
        raise ValueError("confirmed_control must be a 1-D array containing only -1, 0, +1")
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        for frame, team in enumerate(control):
            writer.writerow((frame, "H" if team == 1 else "A" if team == -1 else "X"))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--r-in", type=float, default=ControlConfig.r_in,
                        help="candidate entry radius in metres (default: 2.0)")
    parser.add_argument("--r-out", type=float, default=ControlConfig.r_out,
                        help="hysteresis release radius in metres (default: 2.5)")
    parser.add_argument("--contest-radius", type=float, default=ControlConfig.contest_radius,
                        help="opponent proximity that blocks control (default: 1.5)")
    parser.add_argument("--dwell", type=int, default=ControlConfig.dwell_frames,
                        help="consecutive candidate frames to confirm control (default: 10)")
    args = parser.parse_args()

    config = ControlConfig(r_in=args.r_in, r_out=args.r_out,
                           contest_radius=args.contest_radius, dwell_frames=args.dwell)
    ball, players, teams = load_tracking()
    confirmed, confidence, _ = detect_control(ball, players, teams, config)
    write_control_csv(args.out, confirmed)
    frames = int(np.count_nonzero(confirmed))
    print("wrote {} frames to {}".format(len(confirmed), args.out))
    print("confirmed control: {} frames ({:.1f}%)".format(
        frames, 100.0 * frames / len(confirmed)))
    print("H {}, A {}, X {}".format(
        int(np.count_nonzero(confirmed == 1)),
        int(np.count_nonzero(confirmed == -1)),
        int(np.count_nonzero(confirmed == 0))))
    print("X includes all unconfirmed and dead/live-uncertain frames; no D labels are inferred.")
    print("mean confidence on confirmed frames: {:.3f}".format(
        float(confidence[confirmed != 0].mean()) if frames else 0.0))


if __name__ == "__main__":
    main()
