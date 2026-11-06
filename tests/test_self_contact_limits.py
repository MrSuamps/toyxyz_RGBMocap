"""Self-contact arm limits on every path, kept-rotation touches, ease windows, already-touching holds and the ease/offset CLI."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mocap import self_contact as sc, self_contact_store as store  # noqa: E402
from mocap.correction import fk  # noqa: E402
from test_contacts_feet_cli import BASE, CONTEXT, Project  # noqa: E402
from test_self_contact import FPS, N, OFFSETS, REST, clip, track  # noqa: E402
from test_skin_contact import synthetic_skin  # noqa: E402


def wrist_local(root, q, h):
    _, g = fk(REST, np.asarray(root, float), np.asarray(q, float))
    return g[:, 18 + h].swapaxes(-1, -2) @ g[:, 20 + h]


class ElbowTest(unittest.TestCase):
    def test_hinge_flexion_is_not_twist(self):
        """A pure hinge flexion reads as flexion with no twist; review's plain swing-twist reads ~100 deg on the left arm."""
        for h in (0, 1):
            axis = sc.elbow_hinge(REST, h)
            for degrees in (60., 120., 150.):
                bend = Rotation.from_rotvec(np.deg2rad(degrees) * axis).as_matrix()
                flex, off, twist = sc.elbow_parts(bend, REST, h)
                self.assertAlmostEqual(float(twist), 0., delta=.5)
                roll = Rotation.from_rotvec(np.deg2rad(30.) * sc.forearm_axis(REST, h)).as_matrix()
                self.assertAlmostEqual(float(sc.elbow_parts(bend @ roll, REST, h)[2]), 30., delta=.5)
        bend = Rotation.from_rotvec(np.deg2rad(150.) * sc.elbow_hinge(REST, 0)).as_matrix()
        self.assertGreater(abs(sc.twist_deg(bend, sc.forearm_axis(REST, 0))), 60.)

    def test_over_folded_elbow_opens_to_the_limit_keeping_the_hand(self):
        root, q = clip(); q = q.astype(float)
        bend = Rotation.from_rotvec(np.deg2rad(175.) * sc.elbow_hinge(REST, 1)).as_matrix()
        q[:, 19] = sc.matrix_quat(np.broadcast_to(bend, (N, 3, 3)).copy())
        _, g0 = fk(REST, root.astype(float), q)
        out, clamped = sc.clamp_elbows(REST, root.astype(float), q.copy(), [1])
        self.assertEqual(clamped, N)
        self.assertLessEqual(sc.elbow_flex(REST, out, 1).max(), sc.ELBOW_FOLD_DEG + .5)
        _, g = fk(REST, root.astype(float), out)
        self.assertLess(sc.rotation_angle(g[:, 21], g0[:, 21]).max(), 1e-4)

    def test_arm_pose_never_folds_past_the_limit(self):
        root, q = clip(); j, g = fk(REST, root.astype(float), q.astype(float))
        target = j[:, 17] + (j[:, 19] - j[:, 17]) * .05
        upper, forearm, E, W, short = sc.arm_pose(REST, j, g, sc.quat_matrix(q[:, 19].astype(float)), 1, target, np.zeros(N))
        flex = sc.hinge_flex(REST[19] - REST[17], (upper.swapaxes(-1, -2) @ forearm) @ (REST[21] - REST[19]),
                             sc.elbow_hinge(REST, 1))
        self.assertLessEqual(flex.max(), sc.ELBOW_FOLD_DEG + 1.)
        self.assertTrue((short > .01).all())


class WristLimitTest(unittest.TestCase):
    def test_weighted_limit_leaves_inside_and_zero_weight_rows(self):
        root, q = clip(); q = q.astype(float)
        q[:40, 21] = sc.matrix_quat(Rotation.from_rotvec(np.tile([0., 0., 1.6], (40, 1))).as_matrix())
        weights = np.r_[np.zeros(10), np.linspace(.1, 1, 30), np.ones(N - 40)]
        out = sc.limit_wrist_rows(REST, root.astype(float), q, 1, weights=weights)
        self.assertEqual(out[:10].tobytes(), q[:10].tobytes()); self.assertEqual(out[40:].tobytes(), q[40:].tobytes())
        local = wrist_local(root, out, 1)
        self.assertLess(sc.rotation_angle(local[39], sc.clamp_wrist(local[39:40], REST, 1)[0]), .1)

    def test_fixed_offset_keeps_the_wrist_and_elbow_in_range(self):
        """A fixed offset used to keep the hand's world rotation whatever the arm did; now the wrist ends inside its limits."""
        root, q = clip(); before = q.copy()
        tracks = [track(target='LeftUpLeg', offset=[.12, -.05, .08])]
        report = sc.apply(REST, root, q, OFFSETS, FPS, tracks, [[], []])
        entry = report['tracks'][0]; lo, hi = entry['window']
        local = wrist_local(root, q, 1)[lo:hi]
        self.assertLess(sc.rotation_angle(local, sc.clamp_wrist(local, REST, 1)).max(), 1.)
        twist = np.abs(sc.elbow_parts(sc.quat_matrix(q[lo:hi, 19].astype(float)), REST, 1)[2])
        self.assertLessEqual(twist.max(), sc.ELBOW_TWIST_DEG + 1.)
        if entry['wrists_turned']:
            self.assertEqual(report['wrist_changed'][0]['joint'], 21)
        self.assertEqual(q[:lo].tobytes(), before[:lo].tobytes())


class TouchTest(unittest.TestCase):
    def test_grip_off_keeps_the_hand_rotation_and_touches_with_its_nearest_skin(self):
        skin = synthetic_skin()
        root, q = clip(); before = q.copy()
        report = sc.apply(REST, root, q, OFFSETS, FPS, [track()], [[], []], skin=skin)
        entry = report['tracks'][0]
        self.assertEqual(entry['touch']['mode'], 'kept hand rotation')
        self.assertIn(entry['touch']['surface'], ('palm', 'back of the hand', 'side of the hand', 'fingers'))
        _, g0 = fk(REST, root.astype(float), before.astype(float)); _, g = fk(REST, root.astype(float), q.astype(float))
        a, b = track()['ranges'][0]
        # The hand only turns as far as the wrist limits need (the clip's kept rotation is in range on this arm).
        self.assertLess(sc.rotation_angle(g[a:b, 21], g0[a:b, 21]).max(), 30.)
        self.assertLess(entry['skin']['penetration_interval_max_m'], .006)
        self.assertLess(entry['skin']['skin_gap_max_m'], .015)

    def test_video_seeing_the_hand_elsewhere_lays_the_palm(self):
        skin = synthetic_skin(); root, q = clip()

        class Video:
            def hand(self, h, rows):
                return None, np.ones(len(rows)), None
        with mock.patch.object(sc, 'video_cost', lambda *a, **k: (None, np.full((len(a[6]), 1), 90.))):
            touch, report = sc.hand_touch(skin, REST, root[30:60].astype(float), q[30:60].astype(float), np.arange(30, 60), 1, 'UpLeg', 1, OFFSETS,
                                           Video())
        self.assertIsNone(touch); self.assertTrue(report['mode'].startswith('palm laid'))

    def test_already_touching_hold_is_left_as_captured(self):
        root, q = clip(); before = q.copy()
        with mock.patch.object(sc, 'hand_contact', lambda *a, **k: (np.zeros(len(a[1])), np.full(len(a[1]), .004), None, None)):
            report = sc.apply(REST, root, q, OFFSETS, FPS, [track(target='LeftHand')], [[], []], skin=object())
        self.assertIn('already touches', report['tracks'][0]['skipped'])
        self.assertEqual(report['changed'], []); self.assertEqual(q.tobytes(), before.tobytes())

    def test_push_directions_fall_back_off_the_measured_frames(self):
        vectors = np.zeros((9, 3)); vectors[3:6] = [0, 0, -.02]; fallback = np.tile([1., 0, 0], (9, 1))
        out = sc.push_directions(vectors, (np.arange(9) >= 3) & (np.arange(9) < 6), fallback, 30.)
        np.testing.assert_allclose(out[4], [0, 0, -1], atol=1e-9); np.testing.assert_allclose(out[0], [1, 0, 0])


class EaseTest(unittest.TestCase):
    def test_asymmetric_weights(self):
        w = sc.ease_weights(4, 16, 6, 12, 2, 4)
        self.assertEqual(w[2:8].tolist(), [1.] * 6)
        self.assertTrue((w[:2] > 0).all() and (w[:2] < 1).all() and (w[8:] > 0).all() and (w[8:] < 1).all())
        np.testing.assert_array_equal(sc.ease_weights(0, 20, 6, 12, 3), sc.ease_weights(0, 20, 6, 12, 3, 3))

    def test_room_to_the_same_hands_neighbours(self):
        a, b, c = track(id='a', ranges=[(10, 30)]), track(id='b', target='Head', ranges=[(32, 50)]), track(id='c', effector='LeftHand')
        work = [(a, (10, 30)), (b, (32, 50)), (c, (30, 60))]
        self.assertEqual(sc.ease_room(work, 0, 90), (90, 2)); self.assertEqual(sc.ease_room(work, 1, 90), (2, 90))
        self.assertIsNone(sc.ease_room(work, 2, 90))

    def test_consecutive_holds_do_not_ease_into_each_other(self):
        root, q = clip()
        tracks = [track(id='a', ranges=[(20, 40)]), track(id='b', target='LeftUpLeg', ranges=[(42, 60)])]
        report = sc.apply(REST, root, q, OFFSETS, FPS, tracks, [[], []])
        first, second = report['tracks']
        self.assertLessEqual(first['window'][1], 42); self.assertGreaterEqual(second['window'][0], 40)
        self.assertEqual(first['ease_frames_in_out'][1], 2)

    def test_fixed_ease(self):
        root, q = clip()
        report = sc.apply(REST, root, q, OFFSETS, FPS, [track(ease=3)], [[], []])
        self.assertEqual(report['tracks'][0]['window'], [27, 63]); self.assertEqual(report['tracks'][0]['ease_frames'], 3)


class EaseStoreTest(Project):
    def test_ease_round_trip_and_auto_is_not_written(self):
        raw = dict(effector='RightHand', target='Head', ranges=[[10, 20]])
        store.save(self.project, BASE, [raw], CONTEXT)
        applied = store.revision(self.project, BASE)
        data = json.loads((self.project / json.loads(store.state_path(self.project, BASE).read_text())['file']).read_text())
        self.assertNotIn('ease', data['tracks'][0])
        store.save(self.project, BASE, [dict(raw, ease='auto')], CONTEXT)
        self.assertEqual(store.revision(self.project, BASE), applied)
        store.save(self.project, BASE, [dict(raw, ease=4)], CONTEXT)
        self.assertNotEqual(store.revision(self.project, BASE), applied)
        self.assertEqual(store.read(self.project, BASE, CONTEXT)[0][0]['ease'], 4)
        for bad in (-1, 121, 2.5, True, 'fast'):
            with self.assertRaises(ValueError):
                store.normalize_tracks([dict(raw, ease=bad)], 421)

    def test_cli_negative_offset_and_ease(self):
        code, _, _ = self.run_cli('add', '--effector', 'RightHand', '--target', 'Head', '--frames', '10-20', '--offset', '-0.05,0,-0.14',
                                  '--ease', '4')
        self.assertEqual(code, 0)
        tracks = store.read(self.project, BASE, CONTEXT)[0]
        self.assertEqual((tracks[0]['offset'], tracks[0]['ease']), ([-.05, 0., -.14], 4))
        code, _, lines = self.run_cli('edit', '--id', tracks[0]['id'], '--offset', '-.1,-.2,.3', '--ease', 'auto')
        self.assertEqual(store.read(self.project, BASE, CONTEXT)[0][0]['offset'], [-.1, -.2, .3])
        self.assertIn('ease auto', ' '.join(lines))
        with self.assertRaises(Exception):
            self.run_cli('edit', '--id', tracks[0]['id'], '--ease', '500')


if __name__ == '__main__':
    unittest.main()
