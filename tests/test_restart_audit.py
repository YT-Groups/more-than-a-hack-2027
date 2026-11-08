import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from scripts.restart_audit import nearest_dead_run
from yt.sanity import overlaps


class TestRestartAudit(unittest.TestCase):
    def test_endpoint_touch_is_not_overlap(self):
        self.assertFalse(overlaps(10, 20, 20, 30))

    def test_nearest_dead_run_prefers_overlap(self):
        truth_range = (100, 140)
        predicted = [(150, 190), (105, 145)]
        self.assertEqual(nearest_dead_run(truth_range, predicted), (105, 145))

    def test_window_inside_dead_run(self):
        self.assertTrue(overlaps(110, 120, 100, 140))


if __name__ == "__main__":
    unittest.main()
