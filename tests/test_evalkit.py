import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yt import evalkit  # noqa: E402

FPS = 25


class TestProxies(unittest.TestCase):
    def setUp(self):
        # two "halves" of 300 frames with a gap; one 3 s dead stretch in the first, two in the second
        n = 800
        lab = np.full(n, "X", dtype="<U1")
        lab[0:300] = "H"
        lab[100:175] = "D"
        lab[400:700] = "A"
        lab[450:525] = "D"
        lab[600:650] = "D"
        g = np.zeros(n, bool)
        g[0:300] = True
        g[400:700] = True
        self.lab, self.g = lab, g

    def test_dead_stretches_respect_the_mask(self):
        first = np.zeros_like(self.g)
        first[0:300] = True
        second = np.zeros_like(self.g)
        second[400:700] = True
        self.assertEqual(evalkit.proxies(self.lab, first)["dead_stretch_count"], 1)
        self.assertEqual(evalkit.proxies(self.lab, second)["dead_stretch_count"], 2)
        self.assertEqual(evalkit.proxies(self.lab, self.g)["dead_stretch_count"], 3)

    def test_dead_stretch_lengths(self):
        p = evalkit.proxies(self.lab, self.g)
        self.assertAlmostEqual(p["dead_stretch_median_s"], 75 / FPS)
        self.assertEqual(p["dead_stretch_under_2s"], 0)


if __name__ == "__main__":
    unittest.main()
