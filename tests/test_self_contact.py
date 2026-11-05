"""Self-contact solver on synthetic arrays: no-op behaviour, identity outside the window, easing, reach and reciprocal legs."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import self_contact as sc  # noqa: E402
from mocap.correction import fk  # noqa: E402

# SMPL-X neutral joints of a real actor (metres, Y up, facing +Z) and the registered palm offsets.
REST = np.array([[.003, -.33, .005], [.054, -.417, -.024], [-.054, -.428, -.021], [.001, -.226, -.014], [.107, -.814, -.038],
                 [-.097, -.808, -.04], [.007, -.097, -.017], [.068, -1.229, -.068], [-.082, -1.232, -.059], [-.003, -.049, .003],
                 [.111, -1.291, .048], [-.118, -1.294, .058], [-.012, .106, -.022], [.04, .03, -.003], [-.044, .03, -.009], [.01, .264, 0.],
                 [.146, .088, -.017], [-.135, .084, -.018], [.403, .016, -.051], [-.408, .043, -.04], [.659, .036, -.057], [-.661, .04, -.058]])
OFFSETS = np.array([[.052, -.023, -.026], [-.05, -.026, -.025]])
FPS = 30.
N = 90


def quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()[..., [3, 0, 1, 2]]


def clip():
    """Arms hanging with a slow forward swing and a bent elbow; legs slightly flexed."""
    t = np.arange(N) / FPS
    q = np.zeros((N, 22, 4)); q[..., 0] = 1
    swing = .25 * np.sin(2 * np.pi * t / 3)
    for h, (shoulder, elbow, sign) in enumerate(((16, 18, 1), (17, 19, -1))):
        q[:, shoulder] = quat(np.c_[swing, np.zeros(N), np.full(N, -sign * np.deg2rad(75))])
        q[:, elbow] = quat(np.tile([0., -sign * np.deg2rad(25), 0.], (N, 1)))  # forearm bends toward +Z (forward)
    q[:, 1] = quat(np.tile([-.2, 0, 0], (N, 1))); q[:, 2] = quat(np.tile([-.2, 0, 0], (N, 1)))
    q[:, 4] = quat(np.tile([.3, 0, 0], (N, 1))); q[:, 5] = quat(np.tile([.3, 0, 0], (N, 1)))
    root = np.c_[np.zeros(N), np.full(N, .95), .2 * t]
    return root.astype(np.float32), q.astype(np.float32)


def track(**values):
    base = dict(id='t', effector='RightHand', target='RightUpLeg', offset='auto', distance='auto', reciprocal='off', grip='off',
                enabled=True, ranges=[(30, 60)], automatic=[], source='test')
    return dict(base, **values)


def elbow_interior(root, q, h):
    j, _ = fk(REST, np.asarray(root, float), np.asarray(q, float))
    _, s, e, w = sc.ARMS[h]
    return sc.interior(j[:, s], j[:, e], j[:, w])


class SolverTest(unittest.TestCase):
    def run_tracks(self, tracks, foot_ranges=([], [])):
        root, q = clip(); before = q.copy()
        report = sc.apply(REST, root, q, OFFSETS, FPS, tracks, [list(r) for r in foot_ranges])
        return root, before, q, report

    def test_no_tracks_is_a_no_op(self):
        for tracks in ([], [track(enabled=False)], [track(ranges=[])]):
            _, before, q, report = self.run_tracks(tracks)
            self.assertIsNone(report)
            self.assertEqual(before.tobytes(), q.tobytes())

    def test_missing_store_file_returns_nothing(self):
        root, q = clip(); before = q.copy()
        with tempfile.TemporaryDirectory() as folder:
            result = sc.from_store(Path(folder), 'initial', {'frames': N, 'fps': FPS, 'sources': {}}, REST, root, q, OFFSETS, FPS, [])
        self.assertEqual(result, (None, None))
        self.assertEqual(before.tobytes(), q.tobytes())

    def test_contact_reached_and_untouched_frames_identical(self):
        root, before, q, report = self.run_tracks([track()])
        entry = report['tracks'][0]; changed = report['changed'][0]
        self.assertLess(entry['residual_max_m'], .01)
        self.assertEqual(changed['joints'], [14, 17, 19, 21])
        self.assertEqual((changed['start'], changed['end']), tuple(entry['window']))
        self.assertEqual(changed['start'], 30 - entry['ease_frames'])
        mask = np.zeros((N, 22), bool); mask[changed['start']:changed['end'], changed['joints']] = True
        self.assertEqual(before[~mask].tobytes(), q[~mask].tobytes())
        self.assertFalse(np.array_equal(before[mask], q[mask]))
        _, g0 = fk(REST, root.astype(float), before.astype(float)); _, g1 = fk(REST, root.astype(float), q.astype(float))
        wrist = Rotation.from_matrix(g0[:, 21].swapaxes(-1, -2) @ g1[:, 21]).magnitude()
        self.assertLess(np.rad2deg(wrist).max(), .01)
        self.assertLess(entry['wrist_world_change_max_deg'], 1e-6)

    def test_easing_has_no_elbow_fold(self):
        root, before, q, report = self.run_tracks([track()])
        entry = report['tracks'][0]
        a, b = 30, 60; lo, hi = entry['window']
        source = elbow_interior(root, before, 1); eased = elbow_interior(root, q, 1)
        ease = np.r_[lo:a, b:hi]
        bound = np.minimum(source[ease], min(eased[a], eased[b - 1]))
        self.assertGreater((eased[ease] - bound).min(), -3.)
        self.assertLess(entry['elbows']['RightForeArm']['ease_fold_deg'], 3.)
        self.assertLessEqual(entry['elbows']['RightForeArm']['interval_interior_max_deg'], sc.ELBOW_MAX_DEG + 1e-6)
        self.assertLess(np.abs(np.diff(eased[lo - 1:hi + 1])).max(), 15.)

    def test_ease_weights(self):
        w = sc.ease_weights(20, 50, 30, 40, 10)
        self.assertTrue(((w[:10] > 0) & (w[:10] < 1)).all() and ((w[20:] > 0) & (w[20:] < 1)).all())
        self.assertTrue((w[10:20] == 1).all())
        self.assertTrue((np.diff(w[:11]) > 0).all() and (np.diff(w[19:]) < 0).all())

    def test_slerp_ends_and_sign(self):
        q0 = quat(np.array([[.1, .2, .3]])); q1 = quat(np.array([[-.4, .5, .2]]))
        np.testing.assert_allclose(sc.slerp(q0, q1, np.array([0.])), q0, atol=1e-12)
        np.testing.assert_allclose(np.abs((sc.slerp(q0, -q1, np.array([1.])) * q1).sum()), 1, atol=1e-12)

    def test_reach_deficit(self):
        j, g = fk(REST, np.zeros((1, 3)), quat(np.zeros((22, 3)))[None])
        reach = sc.arm_reach(REST, sc.ARM_CONFIG['collar_limit_deg'])[1]
        direction = np.array([-1., 0, 0])
        for distance, expected in ((reach - .05, 0.), (reach + .10, .10)):
            goal = j[:, 17] + direction * distance + OFFSETS[1]
            deficit = sc.reach_deficit(REST, j, g, 1, goal, OFFSETS, sc.ARM_CONFIG['collar_limit_deg'])
            self.assertAlmostEqual(float(deficit[0]), expected, places=6)

    def test_unreachable_without_reciprocal_reports_shortfall(self):
        _, before, q, report = self.run_tracks([track(target='LeftFoot')])
        entry = report['tracks'][0]
        self.assertGreater(entry['reach_deficit_max_m'], .2)
        self.assertNotIn('reciprocal', entry)
        self.assertGreater(entry['residual_max_m'], .1)
        self.assertIn('Self-contact', report['warning'])
        self.assertEqual(before[:, [0, 3, 6, 9, 12, 15]].tobytes(), q[:, [0, 3, 6, 9, 12, 15]].tobytes())

    def test_reciprocal_leg_moves_toward_hand_and_respects_foot_lock(self):
        _, before, q, report = self.run_tracks([track(target='RightFoot', reciprocal='auto')])
        entry = report['tracks'][0]
        self.assertGreater(entry['reach_deficit_max_m'], 0)
        self.assertGreater(entry['reciprocal']['ankle_shift_max_m'], .05)
        self.assertLess(entry['reach_deficit_after_reciprocal_max_m'], .005)
        self.assertLess(entry['residual_max_m'], .02)
        self.assertEqual(report['changed'][0]['joints'], [2, 5, 8, 14, 17, 19, 21])
        self.assertEqual(before[:, [0, 1, 4, 7, 10, 11]].tobytes(), q[:, [0, 1, 4, 7, 10, 11]].tobytes())
        _, _, locked, report = self.run_tracks([track(target='RightFoot', reciprocal='auto')], ([], [(0, N)]))
        entry = report['tracks'][0]
        self.assertGreater(entry['reciprocal']['foot_lock_conflict_frames'], 0)
        self.assertEqual(entry['reciprocal']['ankle_shift_max_m'], 0.)
        self.assertIn('foot lock', report['warning'])

    def test_hands_meet_with_reciprocal(self):
        _, before, q, report = self.run_tracks([track(target='LeftHand', reciprocal='always', ranges=[(30, 60)])])
        entry = report['tracks'][0]
        self.assertEqual(report['changed'][0]['joints'], [13, 14, 16, 17, 18, 19, 20, 21])
        self.assertLess(entry['residual_max_m'], .02)
        for name in ('LeftForeArm', 'RightForeArm'):
            self.assertLess(entry['elbows'][name]['ease_fold_deg'], 3.)


if __name__ == '__main__':
    unittest.main()
