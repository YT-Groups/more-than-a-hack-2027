import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.law_checks_v8 import first_live, longest_run  # noqa: E402


class TestAuditHelpers(unittest.TestCase):
    def test_first_live(self):
        labels = np.array(list("DDDXHHA"))
        self.assertEqual(first_live(labels, 0), 4)
        self.assertIsNone(first_live(labels, 0, limit=3))

    def test_longest_run(self):
        self.assertEqual(longest_run(np.array([1, 1, 0, 1, 1, 1, 0], bool)), 3)
        self.assertEqual(longest_run(np.zeros(4, bool)), 0)


if __name__ == "__main__":
    unittest.main()
