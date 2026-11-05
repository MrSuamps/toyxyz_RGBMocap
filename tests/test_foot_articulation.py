"""Foot articulation: store entries, pointe/flat/flexed/match solves, easing, identity outside windows, support and review check."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import foot_articulation as fa  # noqa: E402
from mocap import self_contact_store as store  # noqa: E402
from mocap.correction import fk  # noqa: E402

# SMPL-X neutral joints of a real actor (metres, Y up, facing +Z).
REST = np.array([[.003, -.33, .005], [.054, -.417, -.024], [-.054, -.428, -.021], [.001, -.226, -.014], [.107, -.814, -.038],
                 [-.097, -.808, -.04], [.007, -.097, -.017], [.068, -1.229, -.068], [-.082, -1.232, -.059], [-.003, -.049, .003],
                 [.111, -1.291, .048], [-.118, -1.294, .058], [-.012, .106, -.022], [.04, .03, -.003], [-.044, .03, -.009], [.01, .264, 0.],
                 [.146, .088, -.017], [-.135, .084, -.018], [.403, .016, -.051], [-.408, .043, -.04], [.659, .036, -.057], [-.661, .04, -.058]])
FPS = 30.
N = 80
MODEL = fa.FootModel.from_joints(REST)


def quat(rotvec):
    return Rotation.from_rotvec(rotvec).as_quat()[..., [3, 0, 1, 2]]


def clip():
    """Standing with a slow knee bend and a slightly plantarflexed, turned-out left foot."""
    t = np.arange(N) / FPS
    q = np.zeros((N, 22, 4)); q[..., 0] = 1
    q[:, 4] = quat(np.c_[.2 + .1 * np.sin(t), np.zeros(N), np.zeros(N)])
    q[:, 7] = quat(np.tile([.35, .2, .05], (N, 1)))
    q[:, 10] = quat(np.tile([.05, 0, 0], (N, 1)))
    q[:, 8] = quat(np.tile([.1, 0, 0], (N, 1)))
    root = np.c_[np.zeros(N), np.full(N, 1.3), .1 * t]
    return root.astype(np.float32), q.astype(np.float32)


def entry(**values):
    base = dict(id='e', foot='LeftFoot', mode='pointe', strength=1., profile='dancer', enabled=True, ranges=[(30, 50)], automatic=[], source='test')
    return dict(base, **values)


def angle(u, v):
    cos = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


class MathTest(unittest.TestCase):
    def test_pitch_adds_x_rotation(self):
        m0 = REST[10] - REST[7]
        for deg in (-20., 0., 35., 80.):
            self.assertAlmostEqual(float(fa.ankle_pitch(fa.rot_x(deg), m0)), deg, places=6)

    def test_smoothstep_ease_frames_inside_open_interval(self):
        w = fa.smoothstep_weights(10, 40, 16, 34, 6)
        self.assertTrue(np.all(w[6:24] == 1))
        self.assertTrue(np.all((w[:6] > 0) & (w[:6] < 1)) and np.all((w[24:] > 0) & (w[24:] < 1)))
        self.assertTrue(np.all(np.diff(w[:7]) > 0) and np.all(np.diff(w[24:]) < 0))


class SolverTest(unittest.TestCase):
    def solve(self, entries, views=None):
        root, q = clip(); before = q.copy()
        result = fa.apply(MODEL, root, q, entries, FPS, views)
        return root, before, q, result

    def test_no_entries_is_a_no_op(self):
        root, before, q, result = self.solve([entry(enabled=False)])
        self.assertIsNone(result)
        self.assertTrue(np.array_equal(q, before))

    def test_pointe_in_line_with_shin_and_identity_outside_window(self):
        root, before, q, result = self.solve([entry()])
        report = result.report['entries'][0]
        lo, hi = report['window']
        self.assertTrue(np.array_equal(q[:lo], before[:lo]) and np.array_equal(q[hi:], before[hi:]))
        others = [j for j in range(22) if j not in (7, 10)]
        self.assertTrue(np.array_equal(q[:, others], before[:, others]))
        met, toes = fa.in_line(MODEL, 0, root[30:50], q[30:50])
        self.assertLess(met.max(), .5); self.assertLess(toes.max(), .5)
        self.assertGreater(report['ankle_plantarflexion_after_deg']['min'], 45.)
        self.assertEqual(result.changed, [(lo, hi, 0)])
        self.assertTrue(np.all(result.pointe[30:50, 0] == 1) and np.all(result.pointe[:, 1] == 0))
        ease = result.pointe[lo:30, 0]
        self.assertTrue(np.all((ease > 0) & (ease < 1)))

    def test_ease_rotates_monotonically_between_source_and_target(self):
        root, before, q, result = self.solve([entry()])
        lo, _ = result.report['entries'][0]['window']
        p = fa.ankle_pitch(fa.quat_matrix(q[lo - 1:31, 7]), REST[10] - REST[7])
        self.assertTrue(np.all(np.diff(p) > 0))

    def test_default_profile_limits_plantarflexion(self):
        _, _, q, result = self.solve([entry(profile='default')])
        report = result.report['entries'][0]
        self.assertAlmostEqual(report['ankle_plantarflexion_after_deg']['max'], fa.LIMITS['default']['plantar'], places=3)
        self.assertEqual(report['limited_frames'], 20)
        self.assertIn('limited the pose', result.report['warning'])

    def test_flat_levels_the_sole(self):
        root, _, q, _ = self.solve([entry(mode='flat', profile='default')])
        j, g = fk(REST, root.astype(float), q.astype(float))
        heel, tip = MODEL.points(0, j[30:50, 7], g[30:50, 7], g[30:50, 10])
        self.assertLess(np.abs(heel[:, 1] - tip[:, 1]).max(), 1e-6)
        self.assertLess(np.abs(fa.toe_pitch(fa.quat_matrix(q[30:50, 10]), MODEL.tip[0])).max(), 1e-6)

    def test_flexed_reaches_the_dorsiflexion_limit(self):
        _, _, q, result = self.solve([entry(mode='flexed', profile='default')])
        p = fa.ankle_pitch(fa.quat_matrix(q[30:50, 7]), REST[10] - REST[7])
        self.assertTrue(np.allclose(p, -fa.LIMITS['default']['dorsi'], atol=1e-5))
        self.assertTrue(np.all(result.follow[30:50, 0] == 1) and np.all(result.pointe == 0))

    def test_strength_scales_the_rotation(self):
        _, before, full, _ = self.solve([entry()])
        _, _, half, _ = self.solve([entry(strength=.5)])
        m0 = REST[10] - REST[7]
        source, a, b = (fa.ankle_pitch(fa.quat_matrix(x[40, 7]), m0) for x in (before, full, half))
        self.assertGreater(b, source); self.assertLess(b, a)

    def test_match_recovers_pitch_from_a_side_camera(self):
        root, q = clip()
        truth = q.copy()
        truth[:, 7] = quat(np.tile([.9, .2, .05], (N, 1)))
        views = side_views(root, truth)
        result = fa.apply(MODEL, root, q, [entry(mode='match', ranges=[(20, 60)])], FPS, views)
        m0 = REST[10] - REST[7]
        p = fa.ankle_pitch(fa.quat_matrix(q[25:55, 7]), m0); target = fa.ankle_pitch(fa.quat_matrix(truth[25:55, 7]), m0)
        self.assertLess(np.abs(p - target).max(), 2.)
        self.assertGreater(result.report['entries'][0]['confident_fraction'], .9)

    def test_match_without_views_is_skipped(self):
        _, before, q, result = self.solve([entry(mode='match')])
        self.assertTrue(np.array_equal(q, before))
        self.assertIn('skipped', result.report['warning'])

    def test_follow_shins_keeps_the_local_pose_relative_to_the_corrected_knee(self):
        root, before, q, result = self.solve([entry()])
        articulated = q.copy()
        out = q.copy()
        knee = Rotation.from_rotvec([.3, .1, 0]) * Rotation.from_quat(q[:, 4][:, [1, 2, 3, 0]])
        out[:, 4] = knee.as_quat()[:, [3, 0, 1, 2]]
        geometry = StubGeometry()
        probes = np.zeros((N, 4, 3))
        self.assertTrue(result.follow_shins(geometry, root, q, probes, root, out))
        _, src = fk(REST, root.astype(float), q.astype(float)); _, cor = fk(REST, root.astype(float), out.astype(float))
        rows = slice(30, 50)
        world_new = src[rows, 4] @ fa.quat_matrix(q[rows, 7])
        world_want = cor[rows, 4] @ fa.quat_matrix(articulated[rows, 7])
        self.assertLess(np.abs(world_new - world_want).max(), 1e-5)
        self.assertTrue(np.array_equal(q[:, 10], articulated[:, 10]))


def side_views(root, truth):
    """A camera at +X looking along -X, keypoints projected from the truth pose."""
    j, g = fk(REST, root.astype(float), truth.astype(float))
    rotation = np.tile(np.array([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]]), (N, 1, 1))
    translation = np.c_[np.zeros(N), np.full(N, 0.), np.full(N, 3.)] - np.einsum('fij,fj->fi', rotation, np.c_[np.zeros(N), np.full(N, 1.), root[:, 2]])
    K = np.array([[1000., 0, 500], [0, 1000., 500], [0, 0, 1]])
    views = fa.Views(np.zeros((N, 63, 3)), rotation, translation, K)
    heel, tip = MODEL.points(0, j[:, 7], g[:, 7], g[:, 10])
    uv = views.project(np.stack([j[:, 7], tip, tip, heel], 1), np.arange(N))
    for k, index in enumerate(fa.KEYPOINTS[0]):
        views.keypoints[:, index, :2] = uv[:, k]; views.keypoints[:, index, 2] = .9
    other = views.project(j[:, [8]], np.arange(N))[:, 0]
    views.keypoints[:, fa.KEYPOINTS[1][0], :2] = other; views.keypoints[:, fa.KEYPOINTS[1][0], 2] = .9
    return views


class StubGeometry:
    """SoleGeometry.sample stand-in: zero probes and one vertex per toe-box index at the joint positions."""

    def sample(self, root, quat):
        j, _ = fk(REST, np.asarray(root, float), np.asarray(quat, float))
        return np.zeros((len(root), 4, 3)), np.stack([j[:, 10], j[:, 11]], 1)


class SupportTest(unittest.TestCase):
    def test_pointe_probes_blend_to_the_toe_box_and_toe_rows_publish(self):
        import torch
        root, q = clip()
        model = fa.FootModel(REST, MODEL.heel, MODEL.tip, [np.array([0]), np.array([1])])
        result = fa.apply(model, root, q, [entry()], FPS)
        probes = np.ones((N, 4, 3))
        result.support(StubGeometry(), root, q, probes)
        j, _ = fk(REST, root.astype(float), q.astype(float))
        self.assertTrue(np.allclose(probes[30:50, 0], j[30:50, 10]) and np.allclose(probes[30:50, 1], j[30:50, 10]))
        lo, hi = result.report['entries'][0]['window']
        self.assertTrue(np.all(probes[:lo] == 1) and np.all(probes[hi:] == 1) and np.all(probes[:, 2:] == 1))
        output = {'body_pose': torch.zeros((N, 63))}
        pose = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_rotvec().reshape(N, 22, 3).astype(np.float32)
        result.publish(output, pose)
        changed = output['body_pose'].abs().sum(1) > 0
        self.assertTrue(bool(changed[lo:hi].all()) and not bool(changed[:lo].any()) and not bool(changed[hi:].any()))
        self.assertTrue(np.array_equal(output['body_pose'][lo:hi, 27:30].numpy(), pose[lo:hi, 10]))
        self.assertEqual(float(output['body_pose'][:, 30:33].abs().sum()), 0.)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(); self.project = Path(self.folder.name)
        self.context = dict(frames=100, fps=30., sources={})
        self.track = dict(effector='RightHand', target='RightFoot', ranges=[[10, 20]])

    def tearDown(self):
        self.folder.cleanup()

    def test_normalize_defaults_ids_and_errors(self):
        entries = store.normalize_articulation([dict(foot='LeftFoot', mode='pointe', ranges=[[5, 9], [8, 12]]), dict(foot='LeftFoot', mode='pointe')], 100)
        self.assertEqual([e['id'] for e in entries], ['lf-pointe', 'lf-pointe-2'])
        self.assertEqual(entries[0]['ranges'], [(5, 12)]); self.assertEqual(entries[0]['profile'], 'default'); self.assertEqual(entries[0]['strength'], 1.)
        for bad in (dict(foot='LeftHand'), dict(foot='LeftFoot', mode='tiptoe'), dict(foot='LeftFoot', strength=2), dict(foot='LeftFoot', extra=1),
                    dict(foot='LeftFoot', profile='ballet')):
            with self.assertRaises(ValueError):
                store.normalize_articulation([bad], 100)
        with self.assertRaises(ValueError):
            store.normalize_articulation([dict(foot='RightFoot', ranges=[[0, 10]]), dict(foot='RightFoot', mode='flat', ranges=[[5, 15]])], 100)

    def test_save_keeps_articulation_and_clear_keeps_it_until_both_empty(self):
        self.assertIsNone(store.save_articulation(self.project, 'initial', [], self.context))
        self.assertFalse(store.state_path(self.project, 'initial').exists())
        first = store.save_articulation(self.project, 'initial', [dict(foot='LeftFoot', mode='pointe', ranges=[[0, 50]])], self.context)
        self.assertEqual(store.read(self.project, 'initial', self.context), ([], first, ''))
        store.save(self.project, 'initial', [self.track], self.context)
        tracks, _, _ = store.read(self.project, 'initial', self.context)
        feet, revision, _ = store.read_articulation(self.project, 'initial', self.context)
        self.assertEqual(len(tracks), 1); self.assertEqual(feet[0]['ranges'], [(0, 50)]); self.assertNotEqual(revision, first)
        self.assertTrue(store.clear(self.project, 'initial'))
        self.assertEqual(store.read(self.project, 'initial', self.context)[0], [])
        self.assertEqual(len(store.read_articulation(self.project, 'initial', self.context)[0]), 1)
        store.save_articulation(self.project, 'initial', [], self.context)
        self.assertFalse(store.state_path(self.project, 'initial').exists())

    def test_tracks_only_payload_has_no_articulation_key(self):
        import json
        store.save(self.project, 'initial', [self.track], self.context)
        state = json.loads(store.state_path(self.project, 'initial').read_text(encoding='utf-8'))
        self.assertNotIn('articulation', json.loads((self.project / state['file']).read_text(encoding='utf-8')))
        self.assertEqual(store.read_articulation(self.project, 'initial', self.context)[0], [])
        self.assertTrue(store.clear(self.project, 'initial'))
        self.assertFalse(store.state_path(self.project, 'initial').exists())

    def test_other_time_base_warns_and_applies_nothing(self):
        store.save_articulation(self.project, 'initial', [dict(foot='RightFoot', mode='flat', ranges=[[0, 50]])], self.context)
        feet, revision, warning = store.read_articulation(self.project, 'initial', dict(frames=90, fps=30.))
        self.assertEqual(feet, []); self.assertIsNotNone(revision); self.assertIn('foot articulation', warning)

    def test_from_store_without_file_touches_nothing(self):
        root, q = clip(); before = q.copy(); probes = np.zeros((N, 4, 3))
        self.assertIsNone(fa.from_store(self.project, 'initial', dict(frames=N, fps=FPS), None, root, q, probes, FPS))
        self.assertTrue(np.array_equal(q, before))
        self.assertFalse(probes.any())


class ReviewCheckTest(unittest.TestCase):
    def context(self, truth_q, model_q):
        """A review-like Context: keypoints from truth_q, model joints/rotations from model_q (side camera)."""
        root, _ = clip()
        views = side_views(root, truth_q)
        j, g = fk(REST, root.astype(float), model_q.astype(float))
        kp = views.keypoints.copy()
        tj, _ = fk(REST, root.astype(float), truth_q.astype(float))
        knees = views.project(tj[:, [4, 5]], np.arange(N))
        kp[:, 13, :2], kp[:, 14, :2] = knees[:, 0], knees[:, 1]; kp[:, 13:15, 2] = .9
        kp[:, fa.KEYPOINTS[1], 2] = 0
        ctx = SimpleNamespace(keypoints=kp, joints=j, offset=np.zeros((N, 2)), torso=np.full(N, 200.), frames=np.arange(N), fps=FPS,
                              rotation=views.rotation, translation=views.translation, K=views.K, video=lambda i: int(i),
                              project=lambda p: views.project(p, np.arange(N)))
        local = fa.quat_matrix(model_q)
        pose = SimpleNamespace(rows=lambda frames: SimpleNamespace(rest=REST, world=g, local=local))
        return ctx, pose

    def test_matching_foot_has_no_issue_and_flat_foot_under_pointe_is_flagged(self):
        from mocap.cli import foot_orientation as fo
        _, q = clip()
        pointe = q.copy(); pointe[:, 7] = quat(np.tile([1.25, 0, 0], (N, 1))); pointe[:, 10] = quat(np.tile([.35, 0, 0], (N, 1)))
        ctx, pose = self.context(pointe, pointe)
        data = fo.series(ctx, pose)['left']
        self.assertLess(np.nanmax(data['direction_error_deg']), 1e-3)
        self.assertEqual([i for i in fo.check(ctx, pose) if i['effector'] == 'LeftFoot'], [])
        flat = q.copy(); flat[:, 7] = quat(np.tile([0., 0, 0], (N, 1)))
        ctx, pose = self.context(pointe, flat)
        issues = [i for i in fo.check(ctx, pose) if i['effector'] == 'LeftFoot']
        self.assertTrue(issues)
        self.assertEqual({i['check'] for i in issues}, {'foot_orientation'})
        self.assertIn('pointe', {i['suggestion']['mode'] for i in issues})
        stats = fo.statistics(ctx, pose)['left']
        self.assertGreater(stats['direction_error_deg']['median'], 30.)


if __name__ == '__main__':
    unittest.main()
