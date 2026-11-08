import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yt import reception  # noqa: E402
from yt.features import Features  # noqa: E402
from yt.load import Match  # noqa: E402


def make_match(n=120, receiver_team=1, contested=False):
    ball = np.full((n, 2), np.nan)
    ball[:100, :] = 20.0
    ball[20:100, 0] = np.linspace(20.0, 35.0, 80)
    ball[20:100, 1] = np.linspace(20.0, 25.0, 80)

    players = np.zeros((n, 2, 2), dtype=float)
    players[:, 0, :] = 10.0
    players[:, 1, :] = 40.0
    passer_slot = 0 if receiver_team == 1 else 1
    players[:20, passer_slot, :] = ball[:20, :]
    players[20:, 0 if receiver_team == 1 else 1, :] = ball[20:, :]
    if contested:
        players[20:, 1 if receiver_team == 1 else 0, :] = ball[20:, :]

    team = np.array([1, -1], dtype=np.int8)
    player_speed = np.full((n, 2), 4.0)
    return Match(
        ball_xy=ball,
        ball_speed=np.zeros(n),
        players_xy=players,
        player_speed=player_speed,
        team=team,
        is_gk=np.array([False, False]),
        graded=np.ones(n, dtype=bool),
        half=np.ones(n, dtype=np.int8),
        attack_dir_home_h1="left_to_right",
    )


class TestReception(unittest.TestCase):
    def test_stable_same_team_reception_is_detected(self):
        n = 120
        m = make_match(n, receiver_team=1)
        speed = np.zeros(n)
        speed[20:60] = 10.0
        speed[60:100] = 1.0
        f = Features(
            activity=np.full(n, 3.0),
            activity_raw=np.full(n, 3.0),
            ball_vis=~np.isnan(m.ball_xy[:, 0]),
            ball_v=speed,
            ball_out=np.full(n, -5.0),
            d_home=np.zeros(n),
            d_away=np.full(n, 5.0),
            n_objects=np.full(n, 2),
        )
        ctrl = np.zeros(n, dtype=np.int8)
        ctrl[55:100] = 1
        result = reception.detect(m, f, ctrl=ctrl)
        self.assertTrue(result.handover)
        self.assertTrue(all(team == 1 for team in result.handover.values()))

    def test_no_reception_when_opponent_contests_reception(self):
        n = 120
        m = make_match(n, receiver_team=1, contested=True)
        speed = np.zeros(n)
        speed[20:60] = 10.0
        f = Features(
            activity=np.full(n, 3.0),
            activity_raw=np.full(n, 3.0),
            ball_vis=~np.isnan(m.ball_xy[:, 0]),
            ball_v=speed,
            ball_out=np.full(n, -5.0),
            d_home=np.zeros(n),
            d_away=np.zeros(n),
            n_objects=np.full(n, 2),
        )
        result = reception.detect(m, f, ctrl=np.zeros(n, dtype=np.int8))
        self.assertFalse(result.handover)

    def test_control_consistency_can_be_disabled_for_experiment_only(self):
        n = 120
        m = make_match(n, receiver_team=1)
        speed = np.zeros(n)
        speed[20:60] = 10.0
        f = Features(
            activity=np.full(n, 3.0),
            activity_raw=np.full(n, 3.0),
            ball_vis=~np.isnan(m.ball_xy[:, 0]),
            ball_v=speed,
            ball_out=np.full(n, -5.0),
            d_home=np.zeros(n),
            d_away=np.full(n, 5.0),
            n_objects=np.full(n, 2),
        )
        contradictory_control = np.full(n, -1, dtype=np.int8)
        conservative = reception.detect(
            m, f, contradictory_control,
            cfg=reception.ReceptionConfig(require_control=True))
        relaxed = reception.detect(
            m, f, contradictory_control,
            cfg=reception.ReceptionConfig(require_control=False))
        self.assertFalse(conservative.handover)
        self.assertTrue(relaxed.handover)


if __name__ == "__main__":
    unittest.main()
