"""Limb placement: store rows, 2D chain IK on a synthetic camera, easing, identity outside the changed rows, foot locks,
confidence gating, detector swaps, publish and the export audit exemptions."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import limb_placement as lp  # noqa: E402
from mocap import self_contact_store as store  # noqa: E402
from mocap.correction import fk  # noqa: E402

# SMPL-X neutral joints of a real actor (metres, Y up, facing +Z).
REST = np.array([[.003, -.33, .005], [.054, -.417, -.024], [-.054, -.428, -.021], [.001, -.226, -.014], [.107, -.814, -.038],
                 [-.097, -.808, -.04], [.007, -.097, -.017], [.068, -1.229, -.068], [-.082, -1.232, -.059], [-.003, -.049, .003],
                 [.111, -1.291, .048], [-.118, -1.294, .058], [-.012, .106, -.022], [.04, .03, -.003], [-.044, .03, -.009], [.01, .264, 0.],
                 [.146, .088, -.017], [-.135, .084, -.018], [.403, .016, -.051], [-.408, .043, -.04], [.659, .036, -.057], [-.661, .04, -.058]])
FPS = 30.
N = 80
# COCO-63 body keypoint -> SMPL-X joint used to synthesise detections.
COCO = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}


def quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()[..., [3, 0, 1, 2]]


def source_clip():
    """Standing, right knee bent 30 deg, a slight sway so frames differ."""
    t = np.arange(N) / FPS
    q = np.zeros((N, 22, 4)); q[..., 0] = 1
    q[:, 5] = quat(np.tile([np.deg2rad(30), 0, 0], (N, 1)))
    q[:, 4] = quat(np.c_[.1 + .05 * np.sin(t), np.zeros(N), np.zeros(N)])
    root = np.c_[.02 * np.sin(t), np.full(N, 1.25), np.zeros(N)]
    return root.astype(np.float32), q.astype(np.float32)


def truth(q, abduction_deg):
    """The right thigh abducted in the image plane (about world z), everything else as the source."""
    out = q.copy()
    out[:, 2] = quat(np.tile([0, 0, -np.deg2rad(abduction_deg)], (N, 1)))
    return out


def camera(root, q, score=.9, swap=()):
    """Pinhole camera 3.5 m in front looking at -Z, keypoints projected from the given pose."""
    R = np.tile(np.diag([1., -1., -1.]), (N, 1, 1)); C = np.array([0., 1.0, 3.5])
    T = np.tile(-np.diag([1., -1., -1.]) @ C, (N, 1)); K = np.array([[1000., 0, 500], [0, 1000., 500], [0, 0, 1]])
    cam = lp.Camera(np.zeros((N, 63, 3)), R, T, K)
    joints, _ = fk(REST, np.asarray(root, float), np.asarray(q, float))
    uv = cam.project(joints, np.arange(N))
    kp = np.zeros((N, 63, 3))
    for c, s in COCO.items():
        kp[:, c, :2] = uv[:, s]; kp[:, c, 2] = score
    for f in swap:
        for a, b in ((13, 14), (15, 16)):
            kp[f, [a, b]] = kp[f, [b, a]]
    cam.keypoints = kp
    return cam


def entry(**values):
    return dict(dict(id='rl-place', chain='RightLeg', strength=1., enabled=True, ranges=[(20, 60)], automatic=[], source='test'), **values)


def changed_mask(report):
    mask = np.zeros((N, 22), bool)
    for c in report['changed']:
        mask[c['start']:c['end'], c['joints']] = True
    return mask


class SolverTest(unittest.TestCase):
    def run_case(self, entries, kp_pose=None, foot_ranges=([], []), **camera_args):
        root, q = source_clip(); before = q.copy()
        cam = camera(root, truth(q, 60) if kp_pose is None else kp_pose, **camera_args)
        report = lp.apply(REST, root, q, FPS, entries, cam, foot_ranges)
        return root, before, q, report

    def test_disabled_rows_are_a_no_op(self):
        root, before, q, report = self.run_case([entry(enabled=False)])
        self.assertIsNone(report)
        self.assertTrue(np.array_equal(q, before))

    def test_fits_2d_and_keeps_everything_else_bit_identical(self):
        root, before, q, report = self.run_case([entry()])
        e = report['entries'][0]
        self.assertGreater(e['error_before_torso']['limb']['median'], .3)
        self.assertLess(e['error_after_torso']['limb']['median'], .03)
        mask = changed_mask(report)
        self.assertTrue(np.array_equal(q[~mask], before[~mask]))
        self.assertEqual(sorted({j for c in report['changed'] for j in c['joints']}), [2, 5, 8])
        self.assertTrue(all(c['world'] == [2, 5] for c in report['changed']))
        lo, hi = e['window']
        self.assertTrue(np.array_equal(q[:lo], before[:lo]) and np.array_equal(q[hi:], before[hi:]))
        # The foot keeps its source world rotation; the eased rows blend strictly between source and solve.
        _, w0 = fk(REST, root.astype(float), before.astype(float)); _, w1 = fk(REST, root.astype(float), q.astype(float))
        rows = mask.any(1)
        self.assertLess(np.abs(w0[rows, 8] - w1[rows, 8]).max(), 1e-5)
        self.assertGreaterEqual(e['bend_deg']['after_min'], lp.BEND_LIMITS['Leg'][0] - .5)
        self.assertLess(e['depth_change_max_m'], .1)

    def test_ease_is_monotonic_in_rotation_space(self):
        root, before, q, report = self.run_case([entry()])
        lo = report['entries'][0]['window'][0]
        from mocap.self_contact import rotation_angle
        change = rotation_angle(lp.quat_matrix(before[lo:20, 2]), lp.quat_matrix(q[lo:20, 2]))
        self.assertTrue(np.all(np.diff(change) > 0))
        self.assertTrue(np.all(change < rotation_angle(lp.quat_matrix(before[25:26, 2]), lp.quat_matrix(q[25:26, 2]))[0]))

    def test_strength_scales_the_change(self):
        _, b1, q1, _ = self.run_case([entry()])
        _, b2, q2, _ = self.run_case([entry(strength=.5)])
        from mocap.self_contact import rotation_angle
        full = rotation_angle(lp.quat_matrix(b1[40, 2]), lp.quat_matrix(q1[40, 2]))
        half = rotation_angle(lp.quat_matrix(b2[40, 2]), lp.quat_matrix(q2[40, 2]))
        self.assertAlmostEqual(float(half), float(full) / 2, delta=1.)

    def test_locked_foot_frames_stay_bit_identical(self):
        root, before, q, report = self.run_case([entry()], foot_ranges=([], [(30, 40)]))
        self.assertTrue(np.array_equal(q[30:40], before[30:40]))
        self.assertEqual(report['entries'][0]['foot_lock_conflict_frames'], 10)
        self.assertIn('foot lock', report['warning'])
        self.assertFalse(np.array_equal(q[45], before[45]))

    def test_low_confidence_skips(self):
        root, before, q, report = self.run_case([entry()], score=.3)
        self.assertTrue(np.array_equal(q, before))
        self.assertIn('skipped', report['entries'][0])
        self.assertEqual(report['changed'], [])

    def test_detector_swap_frames_use_the_other_side(self):
        root, q = source_clip()
        target = truth(q, 25)
        _, before, after, report = self.run_case([entry()], kp_pose=target, swap=(35, 36, 37))
        e = report['entries'][0]
        self.assertEqual(e['swapped_frames'], 3)
        self.assertLess(e['error_after_torso']['limb']['max'], .05)

    def test_pointe_foot_turns_with_the_placed_shin(self):
        root, q = source_clip(); before = q.copy()
        follow = np.zeros((N, 2)); follow[:, 1] = 1.
        report = lp.apply(REST, root, q, FPS, [entry()], camera(root, truth(q, 60)), ([], []), follow=follow)
        mask = changed_mask(report); rows = mask.any(1)
        # Full follow keeps the ankle's rotation relative to the shin (local), so the foot leaves its world rotation.
        self.assertLess(np.abs(q[rows, 8] - before[rows, 8]).max(), 1e-5)
        _, w0 = fk(REST, root.astype(float), before.astype(float)); _, w1 = fk(REST, root.astype(float), q.astype(float))
        self.assertGreater(np.abs(w0[rows, 8] - w1[rows, 8]).max(), .1)
        self.assertTrue(all(c['world'] == [2, 5, 8] for c in report['changed']))
        self.assertEqual(report['entries'][0]['foot_follow_frames'], int(rows.sum()))
        # A follow weight on the other foot only leaves the placed leg's foot at its world rotation.
        root, q = source_clip(); before = q.copy(); other = np.zeros((N, 2)); other[:, 0] = 1.
        lp.apply(REST, root, q, FPS, [entry()], camera(root, truth(q, 60)), ([], []), follow=other)
        _, w2 = fk(REST, root.astype(float), q.astype(float))
        self.assertLess(np.abs(w0[:, 8] - w2[:, 8]).max(), 1e-5)

    def test_follow_shin_weights(self):
        from mocap.foot_articulation import follow_shin
        root, q = source_clip(); moved = q.copy()
        moved[:, 5] = quat(np.tile([np.deg2rad(70), 0, 0], (N, 1)))
        _, w0 = fk(REST, root.astype(float), q.astype(float)); _, wm = fk(REST, root.astype(float), moved.astype(float))
        moved[:, 8] = lp.matrix_quat(wm[:, 5].swapaxes(-1, -2) @ w0[:, 8])
        weight = np.zeros(N); weight[10:] = .5; weight[40:] = 1.
        out = follow_shin(REST, root, q, moved, 5, 8, weight)
        self.assertTrue(np.array_equal(out[:10], moved[:10]))
        self.assertLess(np.abs(out[40:, 8] - q[40:, 8]).max(), 1e-6)
        _, wo = fk(REST, root.astype(float), out.astype(float))
        from mocap.self_contact import rotation_angle
        half = rotation_angle(w0[20:40, 8], wo[20:40, 8])
        self.assertTrue(np.allclose(half, 20., atol=.5))

    def test_arm_chain_keeps_wrist_world_and_declares_upper_joints(self):
        root, q = source_clip()
        target = q.copy(); target[:, 17] = quat(np.tile([0, 0, np.deg2rad(50)], (N, 1)))
        _, before, after, report = self.run_case([entry(chain='RightArm', id='ra')], kp_pose=target)
        self.assertEqual(sorted({j for c in report['changed'] for j in c['joints']}), [17, 19, 21])
        _, w0 = fk(REST, root.astype(float), before.astype(float)); _, w1 = fk(REST, root.astype(float), after.astype(float))
        self.assertLess(np.abs(w0[:, 21] - w1[:, 21]).max(), 1e-5)
        self.assertLess(report['entries'][0]['error_after_torso']['limb']['median'], .05)
        preserved = (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
        rows = lp.audit_exemptions({'placement': report}, preserved)
        self.assertTrue(rows)
        self.assertTrue(all(sorted(preserved[i] for i in ids) == [17, 19] for _, _, ids in rows))


class AuditTest(unittest.TestCase):
    def test_smoothing_rows_exempt_descendants(self):
        preserved = (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
        rows = lp.audit_exemptions({'smoothing': {'changed': [dict(joints=[9], start=3, end=9)]}}, preserved)
        self.assertEqual(rows[0][:2], (3, 9))
        self.assertEqual(sorted(preserved[i] for i in rows[0][2]), [9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        self.assertEqual(lp.audit_exemptions({}, preserved), [])

    def test_publish_writes_only_unwritten_rows(self):
        import torch
        output = {'body_pose': torch.zeros((10, 63))}
        pose = np.arange(10 * 22 * 3, dtype=np.float32).reshape(10, 22, 3)
        report = {'changed': [dict(joints=[2, 17], start=2, end=5)]}
        placed = lp.publish(output, pose, (report, None), (1, 2, 4, 5, 7, 8))
        body = output['body_pose'].numpy()
        self.assertTrue(np.array_equal(body[2:5, 48:51], pose[2:5, 17]))
        self.assertTrue(np.all(body[:, 3:6] == 0) and np.all(body[:2] == 0) and np.all(body[5:] == 0))
        self.assertTrue(np.array_equal(placed.numpy(), body))
        self.assertIsNone(lp.publish({'body_pose': torch.zeros((10, 63))}, pose, (None,), ()))


class StoreTest(unittest.TestCase):
    CONTEXT = dict(frames=100, fps=30., sources={})

    def test_round_trip_and_kept_by_other_saves(self):
        with tempfile.TemporaryDirectory() as folder:
            store.save_placement(folder, 'initial', [dict(chain='RightLeg', ranges=[[10, 20]])], self.CONTEXT)
            rows, revision, warning = store.read_placement(folder, 'initial', self.CONTEXT)
            self.assertEqual((rows[0]['id'], rows[0]['strength'], rows[0]['ranges'], warning), ('rl-place', 1., [(10, 20)], ''))
            store.save(folder, 'initial', [dict(effector='RightHand', target='RightFoot', ranges=[[5, 9]])], self.CONTEXT)
            store.save_articulation(folder, 'initial', [], self.CONTEXT)
            self.assertEqual(len(store.read_placement(folder, 'initial', self.CONTEXT)[0]), 1)
            self.assertTrue(store.clear(folder, 'initial'))
            self.assertEqual(len(store.read_placement(folder, 'initial', self.CONTEXT)[0]), 1)
            self.assertEqual(store.read(folder, 'initial', self.CONTEXT)[0], [])
            self.assertIsNone(store.save_placement(folder, 'initial', [], self.CONTEXT))
            self.assertFalse(any(Path(folder).iterdir()))

    def test_absent_rows_keep_the_payload_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            store.save(folder, 'initial', [dict(effector='RightHand', target='RightFoot', ranges=[[5, 9]])], self.CONTEXT)
            state = store._state(folder, 'initial')
            import json
            data = json.loads((Path(folder) / state['file']).read_text(encoding='utf-8'))
            self.assertEqual(sorted(data), ['revision', 'tracks', 'version'])
            self.assertEqual(store.read_placement(folder, 'initial', self.CONTEXT), ([], state['revision'], ''))

    def test_validation(self):
        bad = ([dict(chain='Tail', ranges=[[0, 5]])], [dict(chain='LeftArm', strength=2, ranges=[[0, 5]])],
               [dict(chain='LeftArm', ranges=[[0, 5]]), dict(chain='LeftArm', ranges=[[4, 8]])], [dict(chain='LeftArm', extra=1)])
        for rows in bad:
            with self.assertRaises(ValueError):
                store.normalize_placement(rows, 100)
        ok = store.normalize_placement([dict(chain='LeftArm', ranges=[[0, 5]]), dict(chain='LeftArm', ranges=[[4, 8]], enabled=False)], 100)
        self.assertEqual([r['id'] for r in ok], ['la-place', 'la-place-2'])


if __name__ == '__main__':
    unittest.main()
