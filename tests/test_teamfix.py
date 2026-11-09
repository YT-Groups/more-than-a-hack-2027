import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yt import teamfix  # noqa: E402
from yt.load import Match, PITCH  # noqa: E402


def match(n=200, x_slot1=30.0):
    """Two slots, home defends the left goal in half 1; kick-off at frame 100."""
    L, W = PITCH
    xy = np.full((n, 2, 2), np.nan)
    xy[50:150, 0] = [20.0, W / 2]          # home player in his own half
    xy[50:150, 1] = [x_slot1, W / 2]       # away-labelled slot
    graded = np.zeros(n, bool)
    graded[10:190] = True
    half = np.where(graded, 1, 0).astype(np.int8)
    return Match(ball_xy=np.full((n, 2), np.nan), ball_speed=np.full(n, np.nan),
                 players_xy=xy, player_speed=np.zeros((n, 2)), team=np.array([1, -1], np.int8),
                 is_gk=np.zeros(2, bool), graded=graded, half=half,
                 attack_dir_home_h1="left_to_right")


class TestKickoffLaw(unittest.TestCase):
    def test_away_slot_in_home_half_is_flagged_and_moved(self):
        m = match(x_slot1=30.0)               # 22.6 m into the home half
        bad = teamfix.kickoff_violations(m, [100])
        self.assertEqual(bad, [(1, 50, 150)])
        fixed = teamfix.fix_tracklets(m, bad)
        self.assertEqual(list(fixed.team), [1, -1, 1])
        self.assertTrue(np.isnan(fixed.players_xy[50:150, 1]).all())
        np.testing.assert_allclose(fixed.players_xy[50:150, 2, 0], 30.0)

    def test_slot_in_own_half_or_on_the_line_is_not_flagged(self):
        self.assertEqual(teamfix.kickoff_violations(match(x_slot1=80.0), [100]), [])
        self.assertEqual(teamfix.kickoff_violations(match(x_slot1=50.0), [100]), [])


if __name__ == "__main__":
    unittest.main()
