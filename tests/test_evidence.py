import unittest

import numpy as np

from yt.evidence import dynamic_restart_alignment, fuse_transition_team
from yt.features import Features
from yt.load import Match


class TestEvidenceArchitecture(unittest.TestCase):
    def test_dynamic_restart_aligns_to_physical_ball_exit(self):
        n = 100
        ball_v = np.zeros(n)
        ball_v[40:] = 12.0
        activity = np.zeros(n)
        activity[40:] = 4.0
        m = Match(
            ball_xy=np.zeros((n, 2)),
            ball_speed=np.zeros(n),
            players_xy=np.zeros((n, 22, 2)),
            player_speed=np.zeros((n, 22)),
            team=np.ones(22, dtype=np.int8),
            is_gk=np.zeros(22, dtype=bool),
            graded=np.ones(n, dtype=bool),
            half=np.ones(n, dtype=np.int8),
            attack_dir_home_h1="left_to_right",
        )
        f = Features(
            activity=activity,
            activity_raw=activity,
            ball_vis=np.ones(n, dtype=bool),
            ball_v=ball_v,
            ball_out=np.full(n, -10.0),
            d_home=np.full(n, 1.0),
            d_away=np.full(n, 4.0),
            n_objects=np.full(n, 22, dtype=np.int16),
        )
        frame, confidence, detail = dynamic_restart_alignment(m, f, 30)
        self.assertEqual(frame, 40)
        self.assertGreater(confidence, 0.8)
        self.assertEqual(detail["frame"], 40)

    def test_weighted_fusion_can_prefer_strong_spatial_evidence(self):
        team, confidence, weights = fuse_transition_team(
            spatial_team=1,
            spatial_confidence=0.9,
            ball_exit_team=-1,
            ball_exit_confidence=0.4,
            tag_team=-1,
            tag_confidence=0.25,
        )
        self.assertEqual(team, 1)
        self.assertGreater(confidence, 0.0)
        self.assertGreater(weights[1], weights[-1])

    def test_weighted_fusion_handles_disagreement(self):
        team, confidence, _ = fuse_transition_team(
            spatial_team=-1,
            spatial_confidence=0.8,
            ball_exit_team=1,
            ball_exit_confidence=0.8,
            tag_team=-1,
            tag_confidence=0.25,
        )
        self.assertIn(team, (1, -1))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
