"""Skin contact: penetration measure on synthetic meshes, the solver's penetration loop and the video grip fit."""
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

from mocap import self_contact as sc  # noqa: E402
from mocap.correction import PARENTS, fk  # noqa: E402
from test_self_contact import FPS, N, OFFSETS, REST, clip, track  # noqa: E402
from test_self_contact_grip import hand_rest  # noqa: E402


def box(centre, half, n=4):
    """Closed axis-aligned box mesh (vertices, outward faces) with n x n quads per side."""
    centre, half = np.asarray(centre, float), np.asarray(half, float)
    grid = np.linspace(-1, 1, n + 1); vertices, faces, index = [], [], {}

    def vertex(p):
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(vertices); vertices.append(p)
        return index[key]
    for axis in range(3):
        u, w = [k for k in range(3) if k != axis]
        for sign in (-1., 1.):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3); p[axis] = sign; p[u], p[w] = grid[i + di], grid[j + dj]; quad.append(vertex(p))
                    for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                        a, b, c = (np.array(vertices[k]) for k in tri)
                        faces.append(tri if np.cross(b - a, c - a)[axis] * sign > 0 else (tri[0], tri[2], tri[1]))
    return centre + np.array(vertices) * half, np.array(faces)


class PenetrationTest(unittest.TestCase):
    def setUp(self):
        self.v, faces = box([0, 0, 0], [.05, .05, .05])
        self.segment = sc.Segment(faces, np.arange(len(self.v)))

    def measure(self, points):
        return sc.penetration(np.atleast_2d(points), self.v, self.segment)

    def test_inside_outside_touching(self):
        inside = self.measure([0, .03, 0])
        self.assertAlmostEqual(inside['depth'], .02, places=9); self.assertEqual(inside['inside'], 1)
        np.testing.assert_allclose(inside['push'], [0, .02, 0], atol=1e-9)
        outside = self.measure([0, .08, 0])
        self.assertEqual(outside['depth'], 0.); self.assertAlmostEqual(outside['gap'], .03, places=9)
        touching = self.measure([.01, .05, .02])
        self.assertEqual(touching['depth'], 0.); self.assertAlmostEqual(touching['gap'], 0., places=9)
        far = self.measure([0, .5, 0])
        self.assertEqual(far['depth'], 0.); self.assertTrue(np.isinf(far['gap'])); self.assertTrue(np.isnan(sc.signed_gap(far)))

    def test_corner_and_edge_points(self):
        # Closest features are an edge and a corner: the face normal sign is ambiguous there, the pseudo-normal is not.
        for point, depth in (([.049, .049, 0], .001), ([.06, .06, 0], 0.), ([.049, .049, .049], .001), ([.051, .051, .051], 0.)):
            self.assertAlmostEqual(self.measure(point)['depth'], depth, places=9, msg=point)

    def test_points_cloud_takes_the_deepest(self):
        result = self.measure([[0, .08, 0], [0, .04, 0], [0, 0, 0]])
        self.assertAlmostEqual(result['depth'], .05, places=9); self.assertEqual(result['inside'], 2)
        self.assertAlmostEqual(result['gap'], .03, places=9)

    def test_open_segment_is_capped(self):
        v, faces = box([0, 0, 0], [.05, .1, .05])
        keep = ~(v[faces][:, :, 1] <= -.1 + 1e-12).all(1)
        segment = sc.Segment(faces[keep], np.arange(len(v)))
        self.assertEqual(len(segment.loops), 1)
        closed = segment.closed(v)
        self.assertAlmostEqual(sc.winding(np.array([[0, -.09, 0]]), closed)[0], 1., places=6)
        self.assertAlmostEqual(sc.winding(np.array([[0, -.12, 0]]), closed)[0], 0., places=6)
        # Depth is measured to the real skin only (sides), never to the cap.
        self.assertAlmostEqual(sc.penetration(np.array([[0, -.09, 0]]), v, segment)['depth'], .05, places=6)

    def test_pseudo_normals_agree_with_the_winding_number(self):
        v, faces = box([0, 0, 0], [.05, .1, .05])
        keep = ~(v[faces][:, :, 1] <= -.1 + 1e-12).all(1)
        segment = sc.Segment(faces[keep], np.arange(len(v)))
        points = np.random.default_rng(0).uniform([-.07, -.12, -.07], [.07, .12, .07], (400, 3))
        sd, _ = sc.signed_distance(points, v, segment, near=.1)
        np.testing.assert_array_equal(sd < 0, np.abs(sc.winding(points, segment.closed(v))) > .5)

    def test_segment_distance(self):
        a, b = np.array([0., 0, 0]), np.array([1., 0, 0])
        self.assertAlmostEqual(sc.segment_distance(a, b, np.array([.5, 1, 0]), np.array([.5, 2, 0])), 1.)
        self.assertAlmostEqual(sc.segment_distance(a, b, np.array([2., 0, 1]), np.array([3., 0, 1])), np.sqrt(2))
        self.assertAlmostEqual(sc.segment_distance(a, a, np.array([0., 1, 0]), np.array([0., 1, 0])), 1.)

    def test_closest_point_regions(self):
        a, b, c = np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([0., 1, 0])
        points = np.array([[.2, .2, 1], [-1, -1, 0], [.5, -1, 0], [2, 2, 0]])
        q, region = sc.closest_points(points, a, b, c)
        np.testing.assert_allclose(q, [[.2, .2, 0], [0, 0, 0], [.5, 0, 0], [.5, .5, 0]], atol=1e-12)
        self.assertEqual(region[0], 6); self.assertTrue((region[1:] != 6).all())


def synthetic_skin():
    """Boxes skinned rigidly to the right hand, right arm, right leg and trunk of the test rest pose (55 joints)."""
    rest = hand_rest(55)
    parents = list(PARENTS) + [15, 15, 15] + [20 if k % 3 == 0 else 22 + k for k in range(15)] + [21 if k % 3 == 0 else 37 + k for k in range(15)]
    parts = [(21, rest[21] + OFFSETS[1], (.045, .03, .045)), (19, (rest[19] + rest[21]) / 2, (.11, .03, .03)),
             (17, (rest[17] + rest[19]) / 2, (.11, .04, .04)), (2, (rest[2] + rest[5]) / 2, (.07, .18, .07)),
             (5, (rest[5] + rest[8]) / 2, (.05, .19, .05)), (8, rest[8] + [0, -.03, .05], (.04, .03, .09)), (3, rest[3], (.14, .2, .1))]
    vertices, faces, weights = [], [], []
    for joint, centre, half in parts:
        v, f = box(centre, half)
        faces.append(f + sum(len(x) for x in vertices)); vertices.append(v)
        w = np.zeros((len(v), 55)); w[:, joint] = 1; weights.append(w)
    vertices, faces, weights = np.concatenate(vertices), np.concatenate(faces), np.concatenate(weights)
    return sc.Skin(vertices, weights, np.zeros((486, len(vertices) * 3)), parents, rest, np.zeros((55, 3)), faces)


class LoopTest(unittest.TestCase):
    def setUp(self):
        # These cases lay the palm on the skin, as when the video sees the captured hand pointing elsewhere.
        patcher = mock.patch.object(sc, 'hand_touch', lambda *a, **k: (None, dict(mode='palm laid on the skin')))
        patcher.start(); self.addCleanup(patcher.stop)

    def test_penetration_loop_pushes_the_hand_out(self):
        skin = synthetic_skin()
        root, q = clip()
        report = sc.apply(REST, root, q, OFFSETS, FPS, [track()], [[], []], skin=skin)
        entry = report['tracks'][0]; contact = entry['skin']
        self.assertTrue(entry['goal'].startswith('skin of right_thigh'))
        self.assertGreater(contact['depth_max_by_iteration_m'][0], .01)
        self.assertLessEqual(len(contact['depth_max_by_iteration_m']), sc.PUSH_ITERATIONS + 1)
        self.assertLess(contact['penetration_interval_max_m'], .006)
        self.assertLess(abs(contact['skin_gap_median_m']), .01)
        self.assertGreater(contact['push_max_m'], .01)

    def test_anchor_is_one_skin_point_on_the_target_bone(self):
        skin = synthetic_skin()
        root, q = clip(); before = q.copy()
        report = sc.apply(REST, root, q, OFFSETS, FPS, [track()], [[], []], skin=skin)
        entry = report['tracks'][0]; anchor = entry['anchor']
        self.assertEqual(anchor['joint'], 'RightUpLeg'); self.assertIn(anchor['side'], ('front', 'back', 'inner', 'outer', 'top', 'bottom'))
        self.assertTrue(entry['goal'].startswith('skin of right_thigh'))
        # The palm lies on the skin (palm normal against the anchor normal) and the wrist rows are declared.
        _, g = fk(REST, root.astype(float), q.astype(float))
        normal = g[30:60, 2] @ np.array(anchor['normal_local'])
        self.assertLess(sc.angle_between(g[30:60, 21] @ sc.PALM_NORMAL, -normal).max(), 10.)
        changed = report['changed'][0]
        self.assertEqual(report['wrist_changed'], [dict(joint=21, start=changed['start'], end=changed['end'])])
        self.assertEqual(entry['grip']['wrist'], 'palm')
        self.assertLess(entry['skin']['penetration_interval_max_m'], .001); self.assertEqual(entry['skin']['arm_penetrating_frames'], 0)
        outside = np.ones(N, bool); outside[changed['start']:changed['end']] = False
        self.assertEqual(q[outside].tobytes(), before[outside].tobytes())

    def test_arm_pose_flexes_the_elbow_the_right_way(self):
        from mocap.limb_placement import bend
        root, q = clip(); j, g = fk(REST, root.astype(float), q.astype(float))
        targets = j[:, 17] + np.random.default_rng(1).normal(size=(N, 3)) * .15 + [0, .1, .25]
        for swivel in (-2., 0., 2.5):
            upper, forearm, E, W, short = sc.arm_pose(REST, j, g, sc.quat_matrix(q[:, 19]), 1, targets, np.full(N, swivel))
            self.assertGreater(np.rad2deg(bend(j[:, 17], E, W, upper, np.array(sc.ARM_HINGES[1]))).min(), 0.)
            reached = short < 1e-9
            np.testing.assert_allclose(W[reached], targets[reached], atol=1e-6)

    def test_upper_twist_is_the_roll_about_the_humerus(self):
        collar = Rotation.from_rotvec([0, .3, .2]).as_matrix()
        axis = sc.unit(REST[19] - REST[17]); swing = Rotation.from_rotvec(.6 * sc.unit(np.cross(axis, [0, 1., 0]))).as_matrix()
        for degrees in (-150., -30., 0., 75., 170.):
            local = swing @ Rotation.from_rotvec(np.deg2rad(degrees) * axis).as_matrix()
            self.assertAlmostEqual(float(sc.upper_twist(REST, 1, collar, collar @ local)), degrees, places=6)
        cost = sc.upper_twist_cost(np.array([0., 90., 120., 179., -179.]), np.array([0., 90., 90., -179., 179.]))
        wrung = sc.UPPER_TWIST_WEIGHT * 89 + sc.UPPER_TWIST_SOURCE_WEIGHT * 2
        np.testing.assert_allclose(cost, [0, 0, sc.UPPER_TWIST_WEIGHT * 30 + sc.UPPER_TWIST_SOURCE_WEIGHT * 30, wrung, wrung])

    def test_swivels_span_the_upper_arm_twist(self):
        # Swivelling a bent arm round the shoulder-hand line rolls the humerus: some candidates are in range, some wring it.
        root, q = clip(); j, g = fk(REST, root.astype(float), q.astype(float))
        targets = j[:, 17] + [0, -.15, .3]; swivels = np.deg2rad(sc.HAND_SWIVELS_DEG)
        upper = sc.arm_pose(REST, j, g, sc.quat_matrix(q[:, 19]), 1, np.broadcast_to(targets[:, None], (N, len(swivels), 3)),
                            np.broadcast_to(swivels, (N, len(swivels))))[0]
        twist = sc.upper_twist(REST, 1, g[:, None, 14], upper)
        self.assertLess(np.abs(twist).min(1).max(), sc.UPPER_TWIST_DEG)
        self.assertGreater(np.abs(twist).max(1).min(), 150.)

    def test_hand_steps_keep_the_elbow_branch(self):
        # Two candidates per frame (one roll, two swivels): stay put, or turn the upper arm 180 deg over the clip.
        m = 13; turn = np.linspace(0, np.pi, m)
        upper = np.stack([np.broadcast_to(np.eye(3), (m, 3, 3)), Rotation.from_rotvec(turn[:, None] * [1., 0, 0]).as_matrix()], 1)[:, None]
        step = sc.hand_steps(upper, np.broadcast_to(np.eye(3), (m, 1, 3, 3)), FPS)
        fast = sc.ARM_STEP_WEIGHT * 15 + sc.ARM_SPEED_WEIGHT * (15 - sc.ARM_STEP_DEG_S / FPS) ** 2
        np.testing.assert_allclose(step(1), [[0, fast], [0, fast]], atol=1e-6)
        # The other branch, cheaper from frame 6 on, is not worth swinging the arm half way round.
        unary = np.zeros((m, 2)); unary[:6, 1] = 50; unary[6:, 0] = 1
        self.assertEqual(sc.viterbi(unary, step).tolist(), [0] * m)
        per_frame = [1 if i >= 6 else 0 for i in range(m)]
        self.assertEqual(sc.viterbi(unary, np.zeros((2, 2))).tolist(), per_frame)
        self.assertEqual(sc.viterbi(unary, lambda i: np.zeros((2, 2))).tolist(), per_frame)

    def test_without_skin_no_loop_runs(self):
        root, q = clip()
        entry = sc.apply(REST, root, q, OFFSETS, FPS, [track()], [[], []])['tracks'][0]
        self.assertNotIn('skin', entry); self.assertEqual(entry['goal'], 'joint offset')

    def test_push_step_is_smooth_and_zero_without_penetration(self):
        depth = np.zeros(40); depth[18:22] = .01
        direction = np.tile([0., 1., 0.], (40, 1))
        step = sc.push_step(depth, direction, np.ones(40), FPS)
        self.assertGreaterEqual(step[20, 1], .013 - 1e-9 - .003)
        self.assertEqual(np.abs(step[:5]).max(), 0.)
        self.assertLess(np.abs(np.diff(step[:, 1])).max(), .01)
        np.testing.assert_array_equal(sc.push_step(np.zeros(40), direction, np.ones(40), FPS), np.zeros((40, 3)))


class FakeVideo(sc.HandVideo):
    """A camera 3 m in front of the actor and keypoints drawn from a chosen right-hand rotation."""

    def __init__(self, n):
        rotation = np.tile(np.diag([1., -1., -1.]), (n, 1, 1)); eye = np.array([0., .5, 3.])
        translation = np.tile(-np.diag([1., -1., -1.]) @ eye, (n, 1))
        keypoints = np.zeros((n, 63, 3)); keypoints[..., 2] = .9
        keypoints[:, [5, 6, 11, 12], :2] = [[400, 300], [600, 300], [420, 600], [580, 600]]
        super().__init__(keypoints, rotation, translation, np.array([[1000., 0, 500], [0, 1000., 500], [0, 0, 1]]))

    def draw(self, rest, h, wrist_at, world, rows):
        start = 25 + 15 * h; base = 23 + 20 * h
        points = wrist_at[:, None] + np.einsum('tij,pj->tpi', world, rest[[start + 3, start, start + 6]] - rest[20 + h])
        uv = self.project(np.concatenate([wrist_at[:, None], points], 1), rows)
        self.keypoints[rows[:, None], [9 + h, base + 8, base + 4, base + 16], :2] = uv


class VideoGripTest(unittest.TestCase):
    def test_fit_follows_the_video_hand_inside_the_limits(self):
        rest = hand_rest(55); root, q = clip(); rows = np.arange(N)
        j, g = fk(REST, root.astype(float), q.astype(float))
        target = g[:, 19] @ sc.wrist_local(np.full(N, 40.), np.full(N, -10.), np.full(N, 50.), REST, 1)
        video = FakeVideo(N); video.draw(rest, 1, j[:, 21], target, rows)
        world, weight, report = sc.video_wrist(rest, 1, g[:, 19], j[:, 21], g[:, 21], video, rows, FPS)
        self.assertGreater(report['direction_error_source_median_deg'], 20.)
        self.assertLess(report['direction_error_median_deg'], 5.)
        f, r, p = sc.wrist_angles(g[:, 19].swapaxes(-1, -2) @ world, REST, 1)
        self.assertLessEqual(sc.swing_cone(f, r).max(), 1 + 1e-6); self.assertLessEqual(np.abs(p).max(), 80 + 1e-6)
        self.assertEqual(weight.min(), 1.)

    def test_frames_without_keypoints_keep_the_source_wrist(self):
        rest = hand_rest(55); root, q = clip(); rows = np.arange(N)
        j, g = fk(REST, root.astype(float), q.astype(float))
        video = FakeVideo(N); video.keypoints[:, 43:63, 2] = .1; video.keypoints[:, 10, 2] = .1
        world, weight, report = sc.video_wrist(rest, 1, g[:, 19], j[:, 21], g[:, 21], video, rows, FPS)
        self.assertEqual(report['confident_frames'], 0); self.assertEqual(weight.max(), 0.)
        self.assertLess(sc.rotation_angle(world, g[:, 21]).max(), 1e-6)

    def test_grip_auto_declares_the_turned_wrist(self):
        skin = synthetic_skin(); root, q = clip(); before = q.copy(); rows = np.arange(N)
        j, g = fk(REST, root.astype(float), q.astype(float))
        video = FakeVideo(N); video.draw(skin.rest, 1, j[:, 21], g[:, 19] @ sc.wrist_local(np.full(N, 50.), 0, np.full(N, 60.), REST, 1), rows)
        report = sc.apply(REST, root, q, OFFSETS, FPS, [track(grip='auto')], [[], []], skin=skin, video=video)
        entry = report['tracks'][0]; changed = report['changed'][0]
        # On a skin anchor the palm lies on the skin; grip auto rolls it toward the video hand.
        self.assertEqual((entry['grip']['mode'], entry['grip']['wrist']), ('auto', 'palm'))
        self.assertEqual(report['wrist_changed'], [dict(joint=21, start=changed['start'], end=changed['end'])])
        _, q_off = clip()
        off = sc.apply(REST, root, q_off, OFFSETS, FPS, [track()], [[], []], skin=skin, video=video)['tracks'][0]['grip']
        self.assertLessEqual(entry['grip']['direction_error_median_deg'], off['direction_error_median_deg'] + 1e-6)
        self.assertEqual(entry['wrist']['result']['frames_outside_limits'], 0)
        _, g1 = fk(REST, root.astype(float), q.astype(float)); _, g0 = fk(REST, root.astype(float), before.astype(float))
        outside = np.ones(N, bool); outside[changed['start']:changed['end']] = False
        self.assertEqual(sc.rotation_angle(g0[outside, 21], g1[outside, 21]).max(), 0.)


class VideoArmTest(unittest.TestCase):
    def test_elbow_position_separates_swivels_the_direction_cannot(self):
        # Two elbows in the same 2D direction from the shoulder; only the one on the video elbow costs nothing.
        rows = np.arange(2); video = FakeVideo(2); shoulder = np.array([[-.2, 1.4, 0.]] * 2)
        seen = np.array([[-.45, 1.4, 0.]] * 2); far = shoulder + (seen - shoulder) * .4
        video.keypoints[rows[:, None], [6, 8], :2] = video.project(np.stack([shoulder, seen], 1), rows)
        candidates = np.stack([seen, far], 1)
        direction = sc.video_arm_cost(video, 1, rows, shoulder, candidates)
        np.testing.assert_allclose(direction, 0., atol=1e-6)
        aligned = (np.zeros((2, 2)), np.full(2, 300.))
        position = sc.video_arm_cost(video, 1, rows, shoulder, candidates, aligned)
        self.assertLess(position[0, 0], 1e-6)
        self.assertGreater(position[0, 1], 5.)


class FingerCurlTest(unittest.TestCase):
    def test_fingers_stop_at_the_skin(self):
        skin = synthetic_skin()
        skin.groups['right_fingers'] = skin.groups['right_hand'][:8]
        root, q = clip(); rows = np.arange(N)
        curl, curled = sc.finger_weights(skin, root.astype(float), q.astype(float), rows, 1, ['right_thigh'], np.ones(N))
        self.assertEqual(curl.shape, (N,)); self.assertTrue(((curl >= 0) & (curl <= 1)).all())


if __name__ == '__main__':
    unittest.main()
