import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yt import control, machine  # noqa: E402


def possessions(labels):
    """Maximal runs of H or A, as (label, start, end)."""
    out = []
    for i, x in enumerate(labels):
        if x in "HA":
            if out and out[-1][0] == x and out[-1][2] == i:
                out[-1][2] = i + 1
            else:
                out.append([x, i, i + 1])
    return [tuple(p) for p in out]


def run(ctrl, dead=None, restart_team=None, min_len=5):
    n = len(ctrl)
    ctrl = np.array(ctrl, dtype=np.int8)
    dead = np.zeros(n, bool) if dead is None else np.array(dead, bool)
    graded = np.ones(n, bool)
    half = np.ones(n, np.int8)
    return "".join(machine.label(ctrl, dead, graded, half, restart_team=restart_team,
                                 min_len=min_len))


class TestMachine(unittest.TestCase):
    def test_short_blip_does_not_change_possession(self):
        # the dwell lives in control.confirm; together they must ignore a 2-frame blip
        cand = np.array([1] * 20 + [-1] * 2 + [0] * 20 + [1] * 10, dtype=np.int8)
        ctrl = control.confirm(cand, dwell=10)
        self.assertEqual(possessions(run(ctrl)), [("H", 0, 52)])

    def test_change_stamped_on_first_frame(self):
        ctrl = [1] * 20 + [0] * 10 + [-1] * 20
        out = run(ctrl)
        self.assertEqual(out[29], "H")
        self.assertEqual(out[30], "A")

    def test_dead_splits_same_team(self):
        ctrl = [1] * 20 + [0] * 10 + [1] * 20
        dead = [0] * 20 + [1] * 10 + [0] * 20
        out = run(ctrl, dead)
        self.assertEqual(out[20:30], "D" * 10)
        self.assertEqual(possessions(out), [("H", 0, 20), ("H", 30, 50)])

    def test_owner_backfills_to_restart(self):
        ctrl = [1] * 20 + [0] * 10 + [0] * 8 + [-1] * 12
        dead = [0] * 20 + [1] * 10 + [0] * 20
        out = run(ctrl, dead)
        self.assertEqual(out[30:50], "A" * 20)

    def test_restart_team_wins_over_later_control(self):
        ctrl = [1] * 20 + [0] * 10 + [0] * 8 + [-1] * 12
        dead = [0] * 20 + [1] * 10 + [0] * 20
        out = run(ctrl, dead, restart_team={30: 1})
        self.assertEqual(out[30:38], "H" * 8)
        self.assertEqual(out[38:50], "A" * 12)

    def test_min_length_merge_stops_at_dead(self):
        # short A between H and a dead ball: neighbours differ (H, D) so it stays
        ctrl = [1] * 20 + [-1] * 3 + [0] * 0
        dead = [0] * 23 + [1] * 5
        out = run(ctrl + [0] * 5, dead, min_len=5)
        self.assertEqual(possessions(out), [("H", 0, 20), ("A", 20, 23)])

    def test_min_length_merge_between_same_team(self):
        ctrl = [1] * 20 + [-1] * 3 + [1] * 20
        out = run(ctrl, min_len=5)
        self.assertEqual(possessions(out), [("H", 0, 43)])

    def test_no_x_on_graded_once_owner_known(self):
        ctrl = [0] * 5 + [1] * 10 + [0] * 30 + [-1] * 10
        out = run(ctrl)
        self.assertNotIn("X", out)

    def test_ungraded_is_x(self):
        n = 10
        out = machine.label(np.ones(n, np.int8), np.zeros(n, bool),
                            np.r_[np.zeros(5, bool), np.ones(5, bool)],
                            np.r_[np.zeros(5, np.int8), np.ones(5, np.int8)])
        self.assertEqual("".join(out), "XXXXXHHHHH")


if __name__ == "__main__":
    unittest.main()
