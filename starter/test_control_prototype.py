import csv
import tempfile
import unittest

import numpy as np

from starter.control_prototype import (
    ControlConfig,
    clean_ball_track,
    detect_control,
    write_control_csv,
)


class ControlPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.config = ControlConfig(dwell_frames=4, max_ball_gap=3, chunk_frames=8)
        self.teams = np.array([1, -1], dtype=np.int8)

    def test_confirmed_run_is_backdated(self):
        ball = np.zeros((6, 2), dtype=np.float64)
        players = np.full((6, 2, 2), np.nan)
        players[:, 0] = (0.5, 0.0)
        players[:, 1] = (8.0, 0.0)

        control, confidence, candidates = detect_control(
            ball, players, self.teams, self.config)

        np.testing.assert_array_equal(candidates, np.ones(6, dtype=np.int8))
        np.testing.assert_array_equal(control, np.ones(6, dtype=np.int8))
        self.assertTrue(np.all(confidence > 0))

    def test_short_run_is_not_confirmed(self):
        ball = np.zeros((3, 2), dtype=np.float64)
        players = np.array([[[0.5, 0.0], [8.0, 0.0]]] * 3)

        control, _, _ = detect_control(ball, players, self.teams, self.config)

        np.testing.assert_array_equal(control, np.zeros(3, dtype=np.int8))

    def test_nearby_opponent_blocks_candidate(self):
        ball = np.zeros((5, 2), dtype=np.float64)
        players = np.array([[[0.5, 0.0], [1.0, 0.0]]] * 5)

        control, _, candidates = detect_control(
            ball, players, self.teams, self.config)

        np.testing.assert_array_equal(candidates, np.zeros(5, dtype=np.int8))
        np.testing.assert_array_equal(control, np.zeros(5, dtype=np.int8))

    def test_control_can_change_teams(self):
        ball = np.zeros((8, 2), dtype=np.float64)
        players = np.empty((8, 2, 2), dtype=np.float64)
        players[:4, 0] = (0.5, 0.0)
        players[:4, 1] = (8.0, 0.0)
        players[4:, 0] = (8.0, 0.0)
        players[4:, 1] = (0.5, 0.0)

        control, _, _ = detect_control(ball, players, self.teams, self.config)

        np.testing.assert_array_equal(control, np.array([1] * 4 + [-1] * 4))

    def test_short_ball_gap_is_interpolated_but_long_gap_is_not(self):
        ball = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0],
                         [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan],
                         [np.nan, np.nan], [9.0, 0.0]])

        cleaned, confidence = clean_ball_track(ball, self.config)

        np.testing.assert_allclose(cleaned[1], [1.0, 0.0])
        self.assertGreater(confidence[1], 0)
        self.assertTrue(np.isnan(cleaned[3:7]).all())
        np.testing.assert_array_equal(confidence[3:7], np.zeros(4, dtype=np.float32))

    def test_implausible_ball_jump_is_rejected(self):
        ball = np.array([[0.0, 0.0], [100.0, 0.0]])

        cleaned, confidence = clean_ball_track(ball, self.config)

        self.assertTrue(np.isnan(cleaned[1]).all())
        self.assertEqual(confidence[1], 0)

    def test_csv_has_zero_based_frame_and_supported_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_control_csv(
                folder + "/prototype.csv", np.array([1, -1, 0], dtype=np.int8))
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows, [
            ["frame", "label"],
            ["0", "H"],
            ["1", "A"],
            ["2", "X"],
        ])


if __name__ == "__main__":
    unittest.main()
