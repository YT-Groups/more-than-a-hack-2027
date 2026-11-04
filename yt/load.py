"""Load the tactical ATD into per-frame numpy arrays, indexed by 0-based video frame, in metres.

Raw line format (from the EPTS DataFormatSpecification in the metadata XML):

    <frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<x,y,s>   (ball last)

x and y are normalised 0-1, s is speed in m/s. Missing values are NaN. frame_id counts from 1,
so video_frame = frame_id - 1.

The parsed arrays are cached next to the data (data/cache/, gitignored) so later runs take
seconds instead of a minute. The cache key is the raw file's size and mtime.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]  # (length, width) in metres
HOME, AWAY = 1, -1


@dataclass
class Match:
    """Everything downstream needs, one entry per video frame."""

    ball_xy: np.ndarray       # (n, 2) metres, NaN where missing
    ball_speed: np.ndarray    # (n,) m/s as reported by the tracker
    players_xy: np.ndarray    # (n, K, 2) metres
    player_speed: np.ndarray  # (n, K) m/s
    team: np.ndarray          # (K,) +1 home, -1 away
    is_gk: np.ndarray         # (K,) bool, slot flagged Goalkeeper in the metadata
    graded: np.ndarray        # (n,) bool
    half: np.ndarray          # (n,) int8: 1, 2, or 0 outside both halves
    attack_dir_home_h1: str   # "left_to_right" or "right_to_left"

    @property
    def n(self) -> int:
        return len(self.ball_xy)


def parse_meta(meta_path: str) -> dict:
    """Team sign and goalkeeper flag per slot, plus attack directions.

    Home is taken from Score/@idLocalTeam, not from the order of the Team elements.
    """
    root = ET.parse(meta_path).getroot()
    score = root.find(".//Score")
    if score is None or not score.get("idLocalTeam"):
        raise ValueError("metadata has no Score/@idLocalTeam")
    home_id, away_id = score.get("idLocalTeam"), score.get("idVisitingTeam")

    direction = {}
    for t in root.iter("Team"):
        params = {p.findtext("Name"): p.findtext("Value") for p in t.iter("ProviderParameter")}
        direction[t.get("id")] = params.get("attack_direction_first_half")

    team, gk = [], []
    for p in root.iter("Player"):
        tid = p.get("teamId")
        if tid not in (home_id, away_id):
            raise ValueError("slot {} has unknown team {}".format(p.get("id"), tid))
        team.append(HOME if tid == home_id else AWAY)
        params = {q.findtext("Name"): q.findtext("Value") for q in p.iter("ProviderParameter")}
        gk.append(params.get("position_type") == "Goalkeeper")
    return {
        "team": np.array(team, dtype=np.int8),
        "is_gk": np.array(gk, dtype=bool),
        "attack_dir_home_h1": direction.get(home_id),
    }


def parse_raw_text(text: str, n_slots: int) -> np.ndarray:
    """Raw ATD text -> (n_lines, 1 + 3*n_slots + 3) float array. Checks every line's width."""
    width = 1 + 3 * n_slots + 3
    rows = [line.split(",") for line in text.replace(":", ",").replace(";", ",").split()]
    bad = [i for i, r in enumerate(rows) if len(r) != width]
    if bad:
        raise ValueError("{} lines have the wrong field count, first at line {}".format(
            len(bad), bad[0] + 1))
    return np.array(rows, dtype=np.float64)


def _parse_raw_fast(raw_path: str, n_slots: int) -> np.ndarray:
    """Same result as parse_raw_text, using numpy's C parser over the whole file."""
    width = 1 + 3 * n_slots + 3
    with open(raw_path) as f:
        text = f.read()
    n_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    flat = np.fromstring(text.replace(":", ",").replace(";", ",").replace("\n", ","),
                         dtype=np.float64, sep=",")
    if flat.size != n_lines * width:
        # fall back to the slow parser, which names the bad line
        return parse_raw_text(text, n_slots)
    return flat.reshape(n_lines, width)


def arrays_from_table(table: np.ndarray, n_frames: int, n_slots: int) -> dict:
    """Split the parsed table into ball and player arrays on the video-frame index (in metres)."""
    frame_id = table[:, 0].astype(np.int64)
    if np.any(frame_id < 1) or np.any(frame_id > n_frames):
        raise ValueError("frame_id outside 1..{}".format(n_frames))
    if len(np.unique(frame_id)) != len(frame_id):
        raise ValueError("duplicate frame_id")
    vf = frame_id - 1

    scale = np.array(PITCH)
    players = np.full((n_frames, n_slots, 3), np.nan)
    players[vf] = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = np.full((n_frames, 3), np.nan)
    ball[vf] = table[:, 1 + 3 * n_slots:]
    return {
        "players_xy": players[:, :, :2] * scale,
        "player_speed": players[:, :, 2],
        "ball_xy": ball[:, :2] * scale,
        "ball_speed": ball[:, 2],
    }


def graded_mask(n_frames: int, game: str = GAME) -> tuple[np.ndarray, np.ndarray]:
    """(graded bool, half int8) from games.half_video_blocks."""
    graded = np.zeros(n_frames, dtype=bool)
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
        half[first:last + 1] = h
    return graded, half


def _cache_path(raw_path: str) -> str:
    st = os.stat(raw_path)
    return os.path.join(DATA_DIR, "cache", "atd_{}_{}.npz".format(st.st_size, int(st.st_mtime)))


def load_match(game: str = GAME, data_dir: str | None = None, use_cache: bool = True) -> Match:
    """Parse (or read from cache) the tactical ATD and metadata into a Match."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    meta = parse_meta(meta_path)
    n_frames = source_frames("tactical", game)
    n_slots = len(meta["team"])

    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        arrays = arrays_from_table(_parse_raw_fast(raw_path, n_slots), n_frames, n_slots)
        if use_cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, **arrays)

    graded, half = graded_mask(n_frames, game)
    return Match(team=meta["team"], is_gk=meta["is_gk"], graded=graded, half=half,
                 attack_dir_home_h1=meta["attack_dir_home_h1"], **arrays)
