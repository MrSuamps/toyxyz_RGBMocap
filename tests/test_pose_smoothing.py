"""Pose smoothing: log-space Gaussian of local rotations, hemisphere safety, easing, identity outside the changed rows,
joint groups (never the root), foot locks and store rows."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import pose_smoothing as ps  # noqa: E402
from mocap import self_contact_store as store  # noqa: E402

FPS = 30.
N = 60


def quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()[..., [3, 0, 1, 2]]


def jittery(seed=0):
    """Every joint swings slowly about x; joints 0, 5 and 19 also jitter by +-3 deg on alternate frames."""
    t = np.arange(N) / FPS
    q = np.zeros((N, 22, 4)); q[..., 0] = 1
    noise = np.where(np.arange(N) % 2, 1., -1.) * np.deg2rad(3)
    for j in range(22):
        swing = .3 * np.sin(t + j)
        q[:, j] = quat(np.c_[swing + (noise if j in (0, 5, 19) else 0), .1 * np.ones(N), np.zeros(N)])
    return q.astype(np.float32)


def row(**values):
    return dict(dict(id='s', joints=['RightLeg', 'RightForeArm'], window_frames=5, enabled=True, ranges=[(20, 40)], automatic=[], source='test'),
                **values)


def changed_mask(report):
    mask = np.zeros((N, 22), bool)
    for c in report['changed']:
        mask[c['start']:c['end'], c['joints']] = True
    return mask


class SeriesTest(unittest.TestCase):
    def test_constant_series_is_kept(self):
        q = np.tile(quat(np.array([.4, -.2, .7])), (20, 1))
        out = ps.smooth_series(q, 7)
        self.assertLess(np.abs(np.abs((out * q).sum(1)) - 1).max(), 1e-9)

    def test_reduces_alternating_jitter(self):
        q = jittery()[:, 5]
        out = ps.smooth_series(q, 5)
        self.assertLess(ps.angular_acceleration(out, FPS)[3:-3].max(), .2 * ps.angular_acceleration(q, FPS)[3:-3].max())

    def test_hemisphere_flips_do_not_matter(self):
        q = jittery()[:, 5].astype(float)
        flipped = q * np.where(np.arange(N) % 3 == 0, -1., 1.)[:, None]
        a, b = ps.smooth_series(q, 7), ps.smooth_series(flipped, 7)
        self.assertLess(np.abs(np.abs((a * b).sum(1)) - 1).max(), 1e-9)


class ApplyTest(unittest.TestCase):
    def run_rows(self, rows, foot_ranges=([], []), hand_written=False):
        q = jittery(); before = q.copy()
        report = ps.apply(q, FPS, rows, foot_ranges, hand_written)
        return before, q, report

    def test_disabled_rows_are_a_no_op(self):
        before, q, report = self.run_rows([row(enabled=False)])
        self.assertIsNone(report)
        self.assertTrue(np.array_equal(q, before))

    def test_only_listed_joints_and_window_change(self):
        before, q, report = self.run_rows([row()])
        mask = changed_mask(report)
        self.assertTrue(np.array_equal(q[~mask], before[~mask]))
        self.assertEqual(sorted({j for c in report['changed'] for j in c['joints']}), [5, 19])
        lo, hi = report['entries'][0]['window']
        self.assertEqual((lo, hi), (15, 45))
        self.assertTrue(np.array_equal(q[:lo], before[:lo]) and np.array_equal(q[hi:], before[hi:]))
        e = report['entries'][0]['angular_accel_max_deg_s2']
        self.assertLess(e['after']['RightLeg'], .3 * e['before']['RightLeg'])

    def test_ease_blends_in(self):
        before, q, report = self.run_rows([row()])
        change = np.rad2deg((ps.as_rotation(before[15:20, 5]).inv() * ps.as_rotation(q[15:20, 5])).magnitude())
        middle = np.rad2deg((ps.as_rotation(before[20:40, 5]).inv() * ps.as_rotation(q[20:40, 5])).magnitude())
        self.assertLess(change[0], .5 * middle.max())

    def test_root_never_changes(self):
        for joints in ('all-written', 'spine', 'legs', 'arms'):
            before, q, report = self.run_rows([row(joints=joints)], hand_written=True)
            self.assertTrue(np.array_equal(q[:, 0], before[:, 0]), joints)
        self.assertEqual(ps.joint_ids('all-written'), [1, 2, 4, 5, 7, 8])
        self.assertEqual(ps.joint_ids('all-written', True), [1, 2, 4, 5, 7, 8, 13, 14, 16, 17, 18, 19, 20, 21])
        self.assertEqual(ps.joint_ids('spine'), [3, 6, 9, 12, 15])

    def test_locked_leg_rows_stay_bit_identical(self):
        before, q, report = self.run_rows([row(joints='legs')], foot_ranges=([], [(25, 32)]))
        for j in (2, 5, 8, 11):
            self.assertTrue(np.array_equal(q[25:32, j], before[25:32, j]))
        self.assertFalse(np.array_equal(q[25:32, 4], before[25:32, 4]))
        self.assertEqual(report['entries'][0]['leg_frames_held_by_foot_locks'], 4 * 7)


class StoreTest(unittest.TestCase):
    CONTEXT = dict(frames=100, fps=30., sources={})

    def test_round_trip_ids_and_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            store.save_smoothing(folder, 'initial', [dict(joints='arms', ranges=[[10, 20]]), dict(joints=['Head'], ranges=[[10, 20]])],
                                 self.CONTEXT)
            rows, _, _ = store.read_smoothing(folder, 'initial', self.CONTEXT)
            self.assertEqual([(r['id'], r['window_frames']) for r in rows], [('smooth-arms', 5), ('smooth-head', 5)])
            store.save_placement(folder, 'initial', [dict(chain='LeftLeg', ranges=[[0, 4]])], self.CONTEXT)
            self.assertEqual(len(store.read_smoothing(folder, 'initial', self.CONTEXT)[0]), 2)
        for rows in ([dict(joints='arms', ranges=[[0, 5]]), dict(joints=['LeftHand'], ranges=[[3, 8]])], [dict(joints=['Hips'])],
                     [dict(joints='torso')], [dict(joints=['Head'], window_frames=2)], [dict(joints=[])]):
            with self.assertRaises(ValueError):
                store.normalize_smoothing(rows, 100)


if __name__ == '__main__':
    unittest.main()
