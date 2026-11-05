"""Anatomy maths and checks (review), compare helpers and the render rasteriser, on synthetic rotations only."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.cli import anatomy, compare, render  # noqa: E402
from mocap.cli.smplx_body import rodrigues  # noqa: E402
from mocap.mixamo_rig import BODY_PARENTS  # noqa: E402


def rot(axis, degrees):
    return Rotation.from_rotvec(np.radians(degrees) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()


# A straight T-pose: legs down, arms along +X (left) / -X (right), like the SMPL-X rest.
REST = np.zeros((22, 3))
REST[[1, 2]] = [[0.1, -0.1, 0], [-0.1, -0.1, 0]]
REST[[4, 5]] = REST[[1, 2]] + [0, -0.4, 0]
REST[[7, 8]] = REST[[4, 5]] + [0, -0.4, 0]
REST[[10, 11]] = REST[[7, 8]] + [0, -0.05, 0.12]
for j, y in ((3, 0.1), (6, 0.2), (9, 0.3), (12, 0.45), (15, 0.55)):
    REST[j] = [0, y, 0]
REST[[13, 14]] = [[0.08, 0.4, 0], [-0.08, 0.4, 0]]
REST[[16, 17]] = [[0.18, 0.4, 0], [-0.18, 0.4, 0]]
REST[[18, 19]] = REST[[16, 17]] + [[0.28, 0, 0], [-0.28, 0, 0]]
REST[[20, 21]] = REST[[18, 19]] + [[0.25, 0, 0], [-0.25, 0, 0]]


def pose_from_locals(local):
    """FK of (N, 22, 3, 3) locals on REST -> anatomy.Pose."""
    n = len(local)
    world = np.empty_like(local)
    joints = np.empty((n, 22, 3))
    for j, p in enumerate(BODY_PARENTS):
        if p < 0:
            world[:, j], joints[:, j] = local[:, j], REST[j]
        else:
            world[:, j] = world[:, p] @ local[:, j]
            joints[:, j] = joints[:, p] + np.einsum('nij,j->ni', world[:, p], REST[j] - REST[p])
    return anatomy.Pose(joints, world, local, REST, None, None, None, 0, 'test')


def identity(n):
    return np.tile(np.eye(3), (n, 22, 1, 1))


class MathsTest(unittest.TestCase):
    def test_rodrigues_matches_scipy(self):
        v = np.random.default_rng(1).normal(size=(50, 3))
        np.testing.assert_allclose(rodrigues(v), Rotation.from_rotvec(v).as_matrix(), atol=1e-12)

    def test_swing_twist_split(self):
        axis = np.array([1.0, 0, 0])
        swing, twist = anatomy.swing_twist(rot(axis, 70)[None], axis)
        self.assertAlmostEqual(float(twist[0]), 70, places=6)
        self.assertLess(np.linalg.norm(swing), 1e-9)
        swing, twist = anatomy.swing_twist(rot([0, 0, 1], 40)[None], axis)
        self.assertAlmostEqual(float(twist[0]), 0, places=6)
        np.testing.assert_allclose(np.degrees(swing[0]), [0, 0, 40], atol=1e-6)
        combined = rot([0, 1, 0], -25) @ rot(axis, -50)
        swing, twist = anatomy.swing_twist(combined[None], axis)
        self.assertAlmostEqual(float(twist[0]), -50, places=6)
        np.testing.assert_allclose(np.degrees(swing[0]), [0, -25, 0], atol=1e-6)

    def test_hinge_bend_sign_follows_hinges(self):
        local = identity(3)
        local[0, 18] = rot([0, -1, 0], 30)     # left elbow flexion (HINGES axis 0,-1,0): forearm forward
        local[1, 18] = rot([0, -1, 0], -20)    # bent the wrong way
        local[2, 19] = rot([0, 1, 0], 45)      # right elbow flexion
        pose = pose_from_locals(local)
        bend, interior, off = anatomy.hinge_bend(pose.joints, pose.world, 16, 18, 20, (0, -1, 0))
        np.testing.assert_allclose(bend[:2], [30, -20], atol=1e-6)
        np.testing.assert_allclose(interior[:2], [150, 160], atol=1e-6)
        bend, interior, _ = anatomy.hinge_bend(pose.joints, pose.world, 17, 19, 21, (0, 1, 0))
        self.assertAlmostEqual(float(bend[2]), 45, places=6)
        # A rolled upper arm carries the hinge with it: the same local flexion is still flexion.
        local = identity(1)
        local[0, 16] = rot([1, 0, 0], 90)
        local[0, 18] = rot([0, -1, 0], 30)
        pose = pose_from_locals(local)
        self.assertAlmostEqual(float(anatomy.hinge_bend(pose.joints, pose.world, 16, 18, 20, (0, -1, 0))[0][0]), 30, places=6)

    def test_knee_flexion_positive(self):
        local = identity(1)
        local[0, 4] = rot([1, 0, 0], 60)
        pose = pose_from_locals(local)
        bend, interior, _ = anatomy.hinge_bend(pose.joints, pose.world, 1, 4, 7, (1, 0, 0))
        self.assertAlmostEqual(float(bend[0]), 60, places=6)
        self.assertAlmostEqual(float(interior[0]), 120, places=6)

    def test_wrist_angles_both_sides(self):
        # Left hand along +X, palm down: flexion turns the fingers to -Y (about -Z), radial toward the thumb (+Z, about -Y).
        flex, radial, twist, swing = anatomy.wrist_angles(np.stack([rot([0, 0, -1], 40), rot([0, -1, 0], 20), rot([1, 0, 0], 30)]), REST, 'left')
        np.testing.assert_allclose(flex, [40, 0, 0], atol=1e-6)
        np.testing.assert_allclose(radial, [0, 20, 0], atol=1e-6)
        np.testing.assert_allclose(twist, [0, 0, 30], atol=1e-6)
        flex, radial, _, _ = anatomy.wrist_angles(np.stack([rot([0, 0, 1], 40), rot([0, 1, 0], 20)]), REST, 'right')
        np.testing.assert_allclose(flex, [40, 0], atol=1e-6)
        np.testing.assert_allclose(radial, [0, 20], atol=1e-6)

    def test_wrist_excess(self):
        ratio, excess = anatomy.wrist_excess(np.array([40.0, 90.0, -80.0, 0.0]), np.array([0.0, 0.0, 0.0, -50.0]))
        np.testing.assert_allclose(ratio, [0.5, 90 / 80, 80 / 70, 50 / 40])
        np.testing.assert_allclose(excess, [0, 10, 10, 10], atol=1e-9)

    def test_fold_depth(self):
        values = np.r_[np.full(10, 160.0), 150, 120, 80, 120, 150, np.full(10, 160.0)]
        depth = anatomy.fold_depth(values, 10)
        self.assertAlmostEqual(depth.max(), 80)
        self.assertEqual(int(depth.argmax()), 12)
        self.assertEqual(anatomy.fold_depth(np.linspace(170, 90, 30), 10).max(), 0)

    def test_finger_angles(self):
        axis, palm = np.array([1.0, 0, 0]), np.array([0, -1.0, 0])
        flex, abduction, twist = anatomy.finger_angles(np.stack([rot([0, 0, -1], 50), rot([0, 1, 0], 10), rot(axis, 15)]), axis, palm)
        np.testing.assert_allclose(flex, [50, 0, 0], atol=1e-6)
        np.testing.assert_allclose(np.abs(abduction), [0, 10, 0], atol=1e-6)
        np.testing.assert_allclose(twist, [0, 0, 15], atol=1e-6)


class ChecksTest(unittest.TestCase):
    def frames(self, n):
        return np.arange(100, 100 + n)

    def test_locked_elbow_is_high(self):
        n = 30
        local = identity(n)
        local[:, 19] = rot([0, 1, 0], 40)
        local[5:25, 19] = np.eye(3)            # 20 frames at exactly 180 deg interior
        issues = anatomy.check_limbs(pose_from_locals(local), self.frames(n), 30.0)
        locked = [i for i in issues if i['check'] == 'locked_limb' and i['joints'] == ['right_elbow']]
        self.assertEqual(len(locked), 1)
        self.assertEqual(locked[0]['severity'], 'high')
        self.assertEqual(locked[0]['frames'], [105, 124])
        self.assertEqual(locked[0]['blender_frames'], [106, 125])
        self.assertIn('_draw', locked[0])
        # The resting left elbow (identity local) is straight too, at 180 deg over the whole clip.
        self.assertTrue(any(i['joints'] == ['left_elbow'] for i in issues if i['check'] == 'locked_limb'))

    def test_hyperextension_and_fold(self):
        n = 40
        bends = np.full(n, 30.0)
        bends[5:15] = -30.0                    # hyperextended for 10 frames
        bends[25:28] = [80.0, 110.0, 80.0]     # fold and reopen
        local = identity(n)
        for t, b in enumerate(bends):
            local[t, 18] = rot([0, -1, 0], b)
        issues = anatomy.check_limbs(pose_from_locals(local), self.frames(n), 30.0)
        hyper = [i for i in issues if i['check'] == 'hyperextension' and i['joints'] == ['left_elbow']]
        self.assertEqual([(i['frames'], i['severity']) for i in hyper], [([105, 114], 'high')])
        self.assertAlmostEqual(hyper[0]['metrics']['hyperextension_max_deg'], 30, places=5)
        fold = [i for i in issues if i['check'] == 'sudden_fold' and i['joints'] == ['left_elbow']]
        self.assertEqual(len(fold), 1)
        self.assertEqual(fold[0]['worst_frame'], 126)
        self.assertEqual(fold[0]['severity'], 'high')
        self.assertAlmostEqual(fold[0]['metrics']['fold_depth_deg'], 80, places=5)

    def test_sideways_fold_is_not_hyperextension(self):
        # A deeply flexed elbow whose bend plane is turned off the hinge (twisted elbow) reads bend ~ -170 deg.
        n = 20
        local = identity(n)
        local[:, 18] = rot([0, 1, 0], -8) @ rot([0, 0, 1], 150)
        pose = pose_from_locals(local)
        axis = next(limb[5] for limb in anatomy.limbs() if limb[0] == 'left_elbow')
        bend, _, off = anatomy.hinge_bend(pose.joints, pose.world, 16, 18, 20, axis)
        self.assertLess(bend.max(), -150)
        self.assertFalse(anatomy.backward(bend, off).any())
        issues = anatomy.check_limbs(pose, self.frames(n), 30.0)
        self.assertEqual([i for i in issues if i['check'] == 'hyperextension'], [])
        self.assertTrue(anatomy.backward(np.array([-10.0, -120.0]), np.array([3.0, 0.0])).all())

    def test_wrist_limit_and_twist(self):
        n = 20
        local = identity(n)
        local[:, 18] = rot([0, -1, 0], 30)
        local[:, 19] = rot([0, 1, 0], 30)
        local[4:16, 20] = rot([0, 0, -1], 100)   # left wrist flexed 100 deg (limit 80)
        local[4:16, 21] = rot([-1, 0, 0], 120)   # right wrist twisted 120 deg about the forearm
        pose = pose_from_locals(local)
        wrists = anatomy.check_wrists(pose, self.frames(n))
        self.assertEqual([(i['joints'], i['frames'], i['severity']) for i in wrists], [(['left_wrist'], [104, 115], 'high')])
        self.assertAlmostEqual(wrists[0]['metrics']['flexion_at_worst_deg'], 100, places=5)
        twist = anatomy.check_twist(pose, self.frames(n))
        self.assertEqual([(i['joints'], i['severity']) for i in twist], [(['right_wrist'], 'high')])
        self.assertAlmostEqual(abs(twist[0]['metrics']['twist_signed_at_worst_deg']), 120, places=5)

    def test_shoulder_and_hip_twist(self):
        n = 20
        local = identity(n)
        local[:, 17] = rot([0, 0, 1], 40) @ rot([-1, 0, 0], 60)  # right arm lowered, upper arm rolled 60: in range
        local[4:16, 17] = rot([0, 0, 1], 40) @ rot([-1, 0, 0], 165)  # rolled 165 deg about the humerus
        local[:, 1] = rot([0, -1, 0], 50)                          # left thigh rolled 50 (limit 60)
        local[6:12, 2] = rot([0, -1, 0], -75)                      # right thigh rolled 75
        twist = anatomy.check_twist(pose_from_locals(local), self.frames(n))
        found = {i['joints'][0]: (i['frames'], i['severity'], round(abs(i['metrics']['twist_signed_at_worst_deg']))) for i in twist}
        self.assertEqual(found, {'right_shoulder': ([104, 115], 'high', 165), 'right_hip': ([106, 111], 'medium', 75)})

    def swivelled(self, angles):
        """Right elbow bent 90 deg, the whole arm turned by angles (deg per frame) about the shoulder-wrist line."""
        local = identity(len(angles))
        local[:, 19] = rot([0, 1, 0], 90)
        axis = [-0.28, 0, 0.25]
        local[:, 17] = [rot(axis, a) for a in angles]
        return pose_from_locals(local)

    def test_joint_flip(self):
        frames = self.frames(40)
        smooth = self.swivelled(np.linspace(0, 60, 40))
        self.assertEqual(anatomy.check_joint_flip(smooth, frames, 30.0), [])
        steps = anatomy.limb_steps(smooth)['right_arm'][1]
        self.assertAlmostEqual(abs(float(steps['swivel'].sum())), 60, places=5)
        self.assertLess(np.abs(steps['twist']).max(), 1e-6)
        flip = self.swivelled(np.r_[np.zeros(20), np.linspace(0, 170, 5)[1:], np.full(16, 170.0)])
        issues = anatomy.check_joint_flip(flip, frames, 30.0)
        self.assertEqual([(i['joints'], i['severity']) for i in issues], [(['right_elbow'], 'high')])
        self.assertAlmostEqual(issues[0]['metrics']['swivel_window_max_deg'], 170, places=4)
        self.assertTrue(issues[0]['frames'][0] <= 121 <= issues[0]['frames'][1])
        # 30 deg per frame is fast but real at 30 fps; at 60 fps the same steps are twice as fast.
        quick = self.swivelled(np.r_[np.zeros(20), 30, 60, 90, np.full(17, 90.0)])
        self.assertEqual(anatomy.check_joint_flip(quick, frames, 30.0), [])
        self.assertEqual(len(anatomy.check_joint_flip(quick, frames, 60.0)), 1)

    def test_rank_prefixes(self):
        issues = [dict(check='wrist_limit', severity='medium', score=1), dict(check='locked_limb', severity='high', score=2),
                  dict(check='missed_contact', severity='high', score=5), dict(check='wrist_limit', severity='low', score=9)]
        anatomy.rank(issues)
        self.assertEqual([i['id'] for i in issues], ['MC1', 'LL1', 'WL1', 'WL2'])


def issue(check, frames, joints, severity='medium'):
    return dict(check=check, frames=list(frames), joints=joints, severity=severity, worst_frame=frames[0], message='', id=check[:2])


class CompareTest(unittest.TestCase):
    def test_issue_diff(self):
        before = [issue('missed_contact', (366, 419), ['right_hand', 'right_ankle'], 'high'), issue('wrist_limit', (355, 365), ['right_wrist']),
                  issue('jitter', (10, 12), ['Hips'])]
        after = [issue('wrist_limit', (348, 376), ['right_wrist'], 'high'), issue('locked_limb', (383, 420), ['right_elbow'], 'high'),
                 issue('jitter', (200, 210), ['Hips'], 'low')]
        diff = compare.issue_diff(before, after)
        self.assertEqual([i['check'] for i in diff['fixed']], ['missed_contact', 'jitter'])
        self.assertEqual([i['check'] for i in diff['new']], ['locked_limb', 'jitter'])
        self.assertEqual(len(diff['unchanged']), 1)
        self.assertEqual(diff['unchanged'][0]['severity'], 'worse')
        self.assertEqual([i['check'] for i in diff['new_high_or_medium']], ['locked_limb'])

    def params(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return dict(body_pose=rng.normal(size=(n, 63)) * 0.2, global_orient=rng.normal(size=(n, 3)) * 0.2, transl=rng.normal(size=(n, 3)),
                    betas=np.zeros((n, 10)), left_hand_pose=rng.normal(size=(n, 45)) * 0.1, right_hand_pose=rng.normal(size=(n, 45)) * 0.1)

    def test_untouched_identity(self):
        before = self.params(50)
        after = {k: v.copy() for k, v in before.items()}
        changed = [dict(joints=[17, 19, 21], start=20, end=30)]
        after['body_pose'][20:30, 3 * 18:3 * 19] += 0.3          # joint 19 inside its interval: allowed
        result = compare.untouched_identity(after, before, changed)
        self.assertTrue(result['ok'])
        self.assertEqual(result['untouched_frames'], 40)
        after['body_pose'][35, 3 * 18] += 1e-6                   # outside the interval
        after['body_pose'][25, 3 * 4] += 1e-3                    # joint 5 is not listed
        result = compare.untouched_identity(after, before, changed)
        self.assertFalse(result['ok'])
        self.assertFalse(result['outside_frames_identical'])
        self.assertFalse(result['unlisted_joints_identical'])
        self.assertEqual(sorted((v['joint'], v['frames']) for v in result['violations']), [(5, [[25, 25]]), (19, [[35, 35]])])

    def test_rotation_delta(self):
        before = self.params(10, 3)
        after = {k: v.copy() for k, v in before.items()}
        pose = after['body_pose'].reshape(10, 21, 3)
        pose[4, 18] = Rotation.from_matrix(Rotation.from_rotvec(pose[4, 18]).as_matrix() @ rot([0, 1, 0], 25)).as_rotvec()
        delta = compare.rotation_delta(after, before)
        self.assertAlmostEqual(float(delta['max_deg'][4]), 25, places=5)
        self.assertEqual(delta['joint'][4], 'RightForeArm')
        self.assertEqual(delta['changed'], [[4, 4]])
        self.assertLess(float(np.delete(delta['max_deg'], 4).max()), 1e-6)


class RasterTest(unittest.TestCase):
    def test_triangle_and_occlusion(self):
        uv = np.array([[2.0, 2], [18, 2], [2, 18], [0, 0], [20, 0], [0, 20]])
        depth = np.array([1.0, 1, 1, 2, 2, 2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        shade = np.array([1.0, 1, 1, 0.5, 0.5, 0.5])
        value, mask, zinv = render.rasterize(uv, depth, faces, shade, 24, 24)
        self.assertTrue(mask[5, 5] and mask[1, 1])
        self.assertFalse(mask[22, 22])
        self.assertAlmostEqual(value[5, 5], 1.0)     # nearer triangle wins
        self.assertAlmostEqual(value[1, 1], 0.5)     # only the far one covers this pixel
        self.assertAlmostEqual(zinv[5, 5], 1.0)

    def test_look_at_centres_target(self):
        camera = render.look_at([0, 1, 3], [0, 1, 0], 200, 100)
        uv, z = render.project(camera, np.array([[0.0, 1, 0], [0, 1.5, 0], [0.5, 1, 0]]))
        np.testing.assert_allclose(uv[0], [100, 50], atol=1e-9)
        self.assertLess(uv[1, 1], 50)                # world up is image up
        self.assertAlmostEqual(float(z[0]), 3.0)
        # Camera at +Z looking toward -Z (facing a person who faces +Z): their left, world +X, is on the image's right.
        self.assertGreater(uv[2, 0], 100)


if __name__ == '__main__':
    unittest.main()
