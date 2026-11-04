import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yt import control, dead  # noqa: E402
from yt.features import Features, runs  # noqa: E402

FPS = 25


def features(n, activity, ball_v=None, vis=None, out=None):
    act = np.asarray(activity, float)
    vis = np.ones(n, bool) if vis is None else np.asarray(vis, bool)
    bv = np.zeros(n) if ball_v is None else np.asarray(ball_v, float)
    bv = np.where(vis, bv, np.nan)
    out = np.full(n, -10.0) if out is None else np.asarray(out, float)
    return Features(activity=act, activity_raw=act, ball_vis=vis, ball_v=bv,
                    ball_out=np.where(vis, out, np.nan), d_home=np.full(n, 5.0),
                    d_away=np.full(n, 5.0), n_objects=np.full(n, 22))


class TestRuns(unittest.TestCase):
    def test_runs(self):
        s, e = runs(np.array([0, 1, 1, 0, 1], bool))
        self.assertEqual(list(s), [1, 4])
        self.assertEqual(list(e), [3, 5])


class TestCores(unittest.TestCase):
    def test_hysteresis_and_min_length(self):
        cfg = dead.DeadConfig()
        act = np.r_[np.full(100, 4.0), np.full(30, 1.5), np.full(20, 2.3),
                    np.full(30, 1.5), np.full(100, 4.0)]
        # one core: 2.3 is above enter but below leave, so the stoppage continues
        self.assertEqual(dead.cores(act, cfg), [(100, 180)])

    def test_short_dip_ignored(self):
        act = np.r_[np.full(100, 4.0), np.full(20, 1.0), np.full(100, 4.0)]
        self.assertEqual(dead.cores(act, dead.DeadConfig()), [])

    def test_nan_activity_is_not_dead(self):
        act = np.r_[np.full(50, 4.0), np.full(200, np.nan), np.full(50, 4.0)]
        self.assertEqual(dead.cores(act, dead.DeadConfig()), [])


class TestDetect(unittest.TestCase):
    def test_restart_at_kick(self):
        n = 600
        act = np.r_[np.full(200, 4.0), np.full(200, 1.2), np.full(200, 4.0)]
        bv = np.r_[np.full(200, 10.0), np.full(190, 0.2), np.full(210, 12.0)]  # kick at 390
        f = features(n, act, ball_v=bv)
        r = dead.detect(f, np.ones(n, bool))
        self.assertEqual(len(r.segments), 1)
        onset, restart, _, why = r.segments[0]
        self.assertEqual((restart, why), (390, "kick"))
        self.assertTrue(r.dead[onset:restart].all() and not r.dead[restart])

    def test_onset_at_ball_out(self):
        n = 600
        act = np.r_[np.full(200, 4.0), np.full(200, 1.2), np.full(200, 4.0)]
        out = np.r_[np.full(170, -5.0), np.full(430, 2.0)]  # ball leaves at 170
        f = features(n, act, ball_v=np.full(n, 6.0), out=out)
        r = dead.detect(f, np.ones(n, bool))
        self.assertEqual(r.segments[0][:1] + r.segments[0][2:3], (170, "ball_out"))

    def test_moving_ball_vetoes_slow_play(self):
        n = 600
        act = np.r_[np.full(200, 4.0), np.full(200, 1.8), np.full(200, 4.0)]
        f = features(n, act, ball_v=np.full(n, 8.0))  # ball keeps moving inside the pitch
        self.assertFalse(dead.detect(f, np.ones(n, bool)).dead.any())

    def test_kickoff_marks_pre_kickoff_dead(self):
        n = 400
        f = features(n, np.full(n, 4.0), ball_v=np.full(n, 8.0))
        r = dead.detect(f, np.ones(n, bool), kickoffs={0: 300})
        self.assertTrue(r.dead[:300].all() and not r.dead[300:].any())


class TestTap(unittest.TestCase):
    """A first touch to a team-mate is the restart; a repositioned ball is not."""

    def scene(self, receiver_slot, act_after):
        n = 400
        kick = 300                     # v4's last still -> fast transition
        v = np.full(n, 0.2)
        v[200:215] = 4.0               # first movement at 200
        v[215:299] = 0.3               # ball at rest again
        v[300:] = 9.0                  # second movement (the "kick")
        xy = np.zeros((n, 2))
        xy[215:] = [6.0, 0.0]          # first movement carries the ball 6 m
        slot = np.zeros(n, int)
        slot[215:] = receiver_slot
        act = np.r_[np.full(200, 1.0), np.full(200, act_after)]
        f = features(n, act, ball_v=v)
        f.near_slot, f.near_d, f.ball_xy = slot, np.full(n, 0.5), xy
        return f, kick

    def test_throw_to_team_mate_is_the_restart(self):
        f, kick = self.scene(receiver_slot=7, act_after=2.0)
        self.assertEqual(dead.find_tap(f, kick, dead.DeadConfig()), 200)

    def test_same_player_repositioning_is_not(self):
        f, kick = self.scene(receiver_slot=0, act_after=2.0)
        self.assertIsNone(dead.find_tap(f, kick, dead.DeadConfig()))

    def test_no_speed_up_is_not(self):
        f, kick = self.scene(receiver_slot=7, act_after=1.1)
        self.assertIsNone(dead.find_tap(f, kick, dead.DeadConfig()))


class TestConfirm(unittest.TestCase):
    def test_dwell_and_first_frame(self):
        cand = np.array([0, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0], np.int8)
        out = control.confirm(cand, dwell=4)
        self.assertEqual(list(out), [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0])

    def test_backdate(self):
        ctrl = np.array([0, 0, 0, 1, 1, 0], np.int8)
        prox = np.array([0, 1, 1, 1, 1, 0], np.int8)
        self.assertEqual(list(control.backdate(ctrl, prox)), [0, 1, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
