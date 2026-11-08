import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from yt.sanity import find_dead_runs, overlaps, restart_timings


class TestDeadRuns(unittest.TestCase):
    def test_three_runs(self):
        labels = np.array(list("HHDDHHDDDHHDDH"), dtype="<U1")
        self.assertEqual(find_dead_runs(labels, 0, len(labels)), [(2, 4), (6, 9), (11, 13)])

    def test_first_last_and_all(self):
        first = np.array(list("DDHH"), dtype="<U1")
        last = np.array(list("HHDD"), dtype="<U1")
        all_dead = np.array(list("DDDD"), dtype="<U1")
        no_dead = np.array(list("HHHH"), dtype="<U1")
        self.assertEqual(find_dead_runs(first), [(0, 2)])
        self.assertEqual(find_dead_runs(last), [(2, 4)])
        self.assertEqual(find_dead_runs(all_dead), [(0, 4)])
        self.assertEqual(find_dead_runs(no_dead), [])

    def test_single_frame_separator(self):
        labels = np.array(list("DDHDD"), dtype="<U1")
        self.assertEqual(find_dead_runs(labels), [(0, 2), (3, 5)])

    def test_offset_run(self):
        labels = np.array(list("HHHDDHH"), dtype="<U1")
        self.assertEqual(find_dead_runs(labels, 2, 7), [(3, 5)])


class TestOverlaps(unittest.TestCase):
    def test_endpoint_touch_is_not_overlap(self):
        self.assertFalse(overlaps(10, 20, 20, 30))

    def test_window_inside_another(self):
        self.assertTrue(overlaps(10, 30, 15, 20))
        self.assertTrue(overlaps(15, 20, 10, 30))


class TestRestartTiming(unittest.TestCase):
    def test_one_match_is_measured(self):
        timing = restart_timings(
            [(20, 30, 1)],
            [{"start": 10, "end": 50, "type": "Throw-in"}],
            1000,
        )
        self.assertEqual(timing["Throw-in"]["one"], [0.8])
        self.assertEqual(timing["Throw-in"]["many"], [])
        self.assertEqual(timing["Throw-in"]["none"], 0)

    def test_two_matches_are_ambiguous(self):
        timing = restart_timings(
            [(20, 30, 1), (35, 40, 1)],
            [{"start": 10, "end": 50, "type": "Throw-in"}],
            1000,
        )
        self.assertEqual(timing["Throw-in"]["one"], [])
        self.assertEqual(timing["Throw-in"]["many"], [[0.8, 1.2]])

    def test_no_match_and_clipped_window(self):
        timing = restart_timings(
            [(20, 30, 1)],
            [
                {"start": 40, "end": 50, "type": "Corner Kick"},
                {"start": 90, "end": 110, "type": "Goal Kick"},
            ],
            100,
        )
        self.assertEqual(timing["Corner Kick"]["none"], 1)
        self.assertEqual(timing["Goal Kick"]["one"], [])


if __name__ == "__main__":
    unittest.main()
