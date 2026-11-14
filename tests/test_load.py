import os
import sys
import unittest
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import asset_paths  # noqa: E402
from yt import load  # noqa: E402

HAVE_DATA = os.path.isfile(asset_paths()["atd_raw"])

class TestAssetPaths(unittest.TestCase):
    def test_packaged_video_variant_is_resolved(self):
        with tempfile.TemporaryDirectory() as root:
            tactical = os.path.join(root, "tactical")
            os.makedirs(tactical)
            variant = os.path.join(tactical, "BAR-ATH_tactical-001.mp4")
            with open(variant, "wb"):
                pass
            resolved = asset_paths(data_dir=root)["video"]
            self.assertEqual(resolved, variant)

    def test_video_variant_is_not_used_when_exact_name_exists(self):
        with tempfile.TemporaryDirectory() as root:
            tactical = os.path.join(root, "tactical")
            os.makedirs(tactical)
            exact = os.path.join(tactical, "BAR-ATH_tactical.mp4")
            variant = os.path.join(tactical, "BAR-ATH_tactical-001.mp4")
            with open(exact, "wb"):
                pass
            with open(variant, "wb"):
                pass
            resolved = asset_paths(data_dir=root)["video"]
            self.assertEqual(resolved, exact)


class TestParse(unittest.TestCase):
    def test_parse_and_place(self):
        # 2 slots, 2 frames; frame ids 1 and 3 (frame 2 missing)
        text = ("1:0.5,0.5,1.0;NaN,NaN,NaN:0.1,0.2,NaN\n"
                "3:0.0,1.0,2.0;1.0,0.0,3.0:NaN,NaN,NaN\n")
        table = load.parse_raw_text(text, 2)
        self.assertEqual(table.shape, (2, 1 + 6 + 3))
        arr = load.arrays_from_table(table, n_frames=3, n_slots=2)
        L, W = load.PITCH
        np.testing.assert_allclose(arr["players_xy"][0, 0], [0.5 * L, 0.5 * W])
        np.testing.assert_allclose(arr["ball_xy"][0], [0.1 * L, 0.2 * W])
        self.assertTrue(np.isnan(arr["ball_xy"][1]).all())     # missing frame stays NaN
        self.assertEqual(arr["player_speed"][2, 1], 3.0)        # frame_id 3 -> video frame 2

    def test_wrong_width_rejected(self):
        with self.assertRaises(ValueError):
            load.parse_raw_text("1:0.5,0.5;NaN,NaN,NaN:0.1,0.2,NaN\n", 2)

    def test_duplicate_frame_rejected(self):
        table = load.parse_raw_text("1:1,1,1:1,1,1\n1:1,1,1:1,1,1\n", 1)
        with self.assertRaises(ValueError):
            load.arrays_from_table(table, 2, 1)


@unittest.skipUnless(HAVE_DATA, "match data not present")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load.load_match()

    def test_shapes(self):
        self.assertEqual(self.m.n, 199319)
        self.assertEqual(self.m.players_xy.shape, (199319, 46, 2))

    def test_graded(self):
        self.assertEqual(int(self.m.graded.sum()), 158857)
        self.assertTrue(self.m.graded[11185] and not self.m.graded[11184])
        self.assertTrue(self.m.graded[190197] and not self.m.graded[190198])

    def test_ball_visibility(self):
        vis = ~np.isnan(self.m.ball_xy[:, 0])
        self.assertEqual(int(vis[self.m.graded].sum()), 94214)   # DATA.md

    def test_home_is_barcelona_slots(self):
        self.assertEqual(int((self.m.team == load.HOME).sum()), 24)
        self.assertEqual(self.m.attack_dir_home_h1, "left_to_right")


if __name__ == "__main__":
    unittest.main()
