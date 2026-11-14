import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yt import laws  # noqa: E402
from yt.dead import DeadResult  # noqa: E402

FPS = 25


def result(segments, n=10000):
    dead = np.zeros(n, bool)
    for on, re, _, _ in segments:
        dead[on:re] = True
    return DeadResult(dead=dead, conf=dead.astype(float), segments=list(segments))


class TestMergeBeforeKickoffs(unittest.TestCase):
    def test_short_live_gap_before_an_after_goal_kickoff_is_dead(self):
        # stoppage 1000-1800, "live" 1800-2100 (12 s), stoppage 2100-2300 ending in a kick-off
        r = result([(1000, 1800, "decel", "calibrated"), (2100, 2300, "decel", "kickoff")])
        out = laws.merge_before_kickoffs(r, np.ones(10000, bool))
        self.assertEqual([s[:2] for s in out.segments], [(1000, 2300)])
        self.assertTrue(out.dead[1000:2300].all())

    def test_long_stoppage_before_kickoff_is_left_alone(self):
        # the stoppage before the kick-off is 40 s: long enough to hold the goal itself
        r = result([(1000, 1800, "decel", "calibrated"), (2100, 3100, "decel", "kickoff")])
        out = laws.merge_before_kickoffs(r, np.ones(10000, bool))
        self.assertEqual(len(out.segments), 2)

    def test_long_live_gap_is_left_alone(self):
        r = result([(1000, 1800, "decel", "calibrated"), (2700, 2900, "decel", "kickoff")])
        out = laws.merge_before_kickoffs(r, np.ones(10000, bool))
        self.assertEqual(len(out.segments), 2)

    def test_half_start_kickoff_is_not_merged(self):
        r = result([(1000, 1800, "decel", "calibrated"), (2100, 2300, "half_start", "kickoff")])
        out = laws.merge_before_kickoffs(r, np.ones(10000, bool))
        self.assertEqual(len(out.segments), 2)


if __name__ == "__main__":
    unittest.main()
