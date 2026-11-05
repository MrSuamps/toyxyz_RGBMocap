"""Review checks on synthetic data: sole floor penetration, declared contacts inside the target, the pointe rule of
foot_orientation; render frame selection and framing; the CLI camera-grounding notice."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.cli import anatomy, foot_orientation as fo, foot_support, main, render, review  # noqa: E402
from mocap.cli.smplx_body import Body  # noqa: E402


def context(n, first=100):
    return SimpleNamespace(frames=np.arange(first, first + n), video=lambda i: int(first + i), fps=30.0)


def mesh_pose(n):
    """A pose whose mesh is a placeholder (sole heights are patched in)."""
    mesh = SimpleNamespace(index=np.arange(n))
    return SimpleNamespace(mesh=mesh, rows=lambda frames: SimpleNamespace(mesh=mesh))


class FloorPenetrationTest(unittest.TestCase):
    def test_flat_floor_flags_soles_below_it(self):
        n = 20
        sole = np.full((n, 2), 0.068)
        sole[5:10, 1] = 0.068 - 0.03
        floor = review.Floor(0.068, 'corrected_initial_correction.json')
        with mock.patch.object(foot_support, 'sole_heights', return_value=sole):
            issues = review.check_penetration(context(n), np.zeros((n, 2), bool), floor, Path('.'), 'corrected_initial', mesh_pose(n))
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual((issue['frames'], issue['joints'], issue['severity']), ([105, 109], ['right_sole'], 'medium'))
        self.assertAlmostEqual(issue['metrics']['max_depth_m'], 0.03)
        self.assertIn('below the floor', issue['message'])

    def test_multi_level_uses_each_stance_height(self):
        n = 30
        sole = np.full((n, 2), 0.2)
        sole[10:20, 0] = 0.12              # second stance on a lower level: no penetration
        sole[22:26, 0] = 0.06              # 6 cm under the third stance
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            segments = np.array([[0, 100, 110, 0, 0.2, 0], [0, 110, 120, 0, 0.12, 0], [0, 120, 130, 0, 0.12, 0], [1, 100, 130, 0, 0.2, 0]], float)
            np.savez(project / 'corrected_initial_contacts.npz', segments=segments)
            floor = review.Floor(0.0, 'corrected_initial_correction.json', 'multi_level')
            level = review.surface_levels(context(n), project, 'corrected_initial', floor, sole)
            self.assertTrue(np.allclose(level[:10, 0], 0.2) and np.allclose(level[10:, 0], 0.12))
            with mock.patch.object(foot_support, 'sole_heights', return_value=sole):
                issues = review.check_penetration(context(n), np.zeros((n, 2), bool), floor, project, 'corrected_initial', mesh_pose(n))
        self.assertEqual([(i['frames'], i['severity']) for i in issues], [([122, 125], 'high')])
        self.assertAlmostEqual(issues[0]['metrics']['max_depth_m'], 0.06)
        self.assertEqual(issues[0]['metrics']['surface'], 'stance')

    def test_without_contacts_multi_level_flags_nothing_and_estimated_floor_follows_the_soles(self):
        n = 10
        sole = np.full((n, 2), 0.3)
        with tempfile.TemporaryDirectory() as folder:
            level = review.surface_levels(context(n), Path(folder), 'k', review.Floor(0.0, 'k_correction.json', 'multi_level'), sole)
        self.assertTrue(np.isnan(level).all())
        level = review.surface_levels(context(n), Path('.'), 'k', review.Floor(0.0, 'estimated'), sole)
        self.assertTrue(np.allclose(level, 0.3))

    def test_sole_heights_match_the_body_vertices(self):
        rng = np.random.default_rng(1)
        v, n = 60, 3
        body = Body.__new__(Body)
        body.template = rng.normal(size=(v, 3))
        body.shapedirs = rng.normal(scale=0.01, size=(v, 3, 20))
        body.posedirs = rng.normal(scale=0.01, size=(486, 3 * v))
        weights = rng.random((v, 55)) * 0.1
        weights[np.arange(v), rng.choice([0, 7, 8, 10, 11], size=v)] += 1
        body.weights = weights / weights.sum(1, keepdims=True)
        local = Rotation.from_rotvec(rng.normal(scale=0.3, size=(n * 55, 3))).as_matrix().reshape(n, 55, 3, 3)
        state = dict(shapes=rng.normal(size=(n, 20)), local=local, world=local.copy(), rest=rng.normal(size=(n, 55, 3)),
                     joints=rng.normal(size=(n, 55, 3)), transl=rng.normal(size=(n, 3)))
        mesh = SimpleNamespace(body=body, state=state, index=np.arange(n))
        heights = foot_support.sole_heights(mesh, chunk=2)
        dominant = body.weights.argmax(1)
        for i in range(n):
            y = body.vertices(state, i)[:, 1]
            self.assertAlmostEqual(heights[i, 0], y[np.isin(dominant, (7, 10))].min())
            self.assertAlmostEqual(heights[i, 1], y[np.isin(dominant, (8, 11))].min())


class ContactInsideTest(unittest.TestCase):
    def issues(self, depth, gap):
        n = len(depth)
        pair = ('right_hand', ('head', 'torso'))
        track = dict(id='rh-head', effector='RightHand', target='Head', ranges=[(100, 100 + n)])
        mesh = SimpleNamespace(measure=lambda pairs: {pair: (np.asarray(depth), np.zeros(n, int), np.asarray(gap))})
        with mock.patch.object(anatomy, 'contact_tracks', return_value=[(track, pair)]):
            return anatomy.check_contact_gap(SimpleNamespace(mesh=mesh), np.arange(100, 100 + n))

    def test_hand_held_inside_the_target_is_flagged_below_the_penetration_threshold(self):
        issues = self.issues(np.full(6, 0.008), np.full(6, -0.007))
        self.assertEqual(len(issues), 1)
        self.assertEqual((issues[0]['check'], issues[0]['severity']), ('contact_gap', 'medium'))
        self.assertTrue(issues[0]['metrics']['inside'])
        self.assertIn('inside the head', issues[0]['message'])

    def test_hand_on_the_skin_is_not_flagged(self):
        self.assertEqual(self.issues(np.full(6, 0.002), np.full(6, -0.001)), [])


def foot_data(n, **values):
    data = dict(direction_error_deg=np.full(n, 3.0), heel_toe_error_deg=np.full(n, 5.0), heel_lift_deg=np.full(n, -5.0),
                observed_in_line_deg=np.full(n, 10.0), model_in_line_deg=np.full(n, 8.0), observed_length_ratio=np.full(n, 0.8),
                plantarflexion_deg=np.full(n, 5.0), toe_flexion_deg=np.zeros(n), tip_error_torso=np.full(n, 0.05),
                ok=np.ones(n, bool), joints=(7, 10), rotations=True)
    data.update(values)
    return {'left': data}


class PointeRuleTest(unittest.TestCase):
    def check(self, data):
        n = len(data['left']['ok'])
        with mock.patch.object(fo, 'series', return_value=data):
            return [i for i in fo.check(context(n)) if i['suggestion']['mode'] == 'pointe']

    def test_wide_stance_matching_the_video_is_not_pointe(self):
        self.assertEqual(self.check(foot_data(20)), [])

    def test_raised_heel_or_off_direction_is_pointe(self):
        lift = np.full(20, -5.0)
        lift[2:14] = 25.0
        issues = self.check(foot_data(20, heel_lift_deg=lift))
        self.assertEqual([(i['frames'], i['severity']) for i in issues], [([102, 113], 'medium')])
        self.assertIn('heel 25 deg higher', issues[0]['message'])
        error = np.full(20, 3.0)
        error[5:9] = 20.0
        issues = self.check(foot_data(20, direction_error_deg=error))
        self.assertEqual([(i['frames'], i['severity']) for i in issues], [([105, 108], 'low')])


class SheetFramesTest(unittest.TestCase):
    def test_default_frames_spread_over_the_hold_not_its_gaps(self):
        frames, outside = render.sheet_frames(None, [(0, 17), (165, 174)], 449)
        self.assertEqual(len(frames), render.SHEET_FRAMES)
        self.assertTrue(all(f < 17 or 165 <= f < 174 for f in frames))
        self.assertEqual((frames[0], frames[-1], outside), (0, 173, []))

    def test_requested_frames_keep_the_ones_inside_the_hold(self):
        self.assertEqual(render.sheet_frames('380,385,430', [(383, 389)], 449), ([385], []))
        frames, outside = render.sheet_frames('300-400', [(383, 389)], 449)
        self.assertTrue(all(383 <= f < 389 for f in frames) and outside == [])
        self.assertEqual(render.sheet_frames('361,430', [(383, 389)], 449), ([361, 430], [361, 430]))
        self.assertEqual(render.sheet_frames('10,20', [], 449, [(15, 25)]), ([20], []))
        self.assertIsNone(render.sheet_frames(None, [], 449))

    def test_frame_list(self):
        self.assertEqual(render.frame_list('30,10,20,10', 100), [10, 20, 30])
        self.assertIsNone(render.frame_list('10-20', 100))
        with self.assertRaises(render.CliError):
            render.frame_list('10,200', 100)


class SheetSideTest(unittest.TestCase):
    def test_close_view_looks_from_the_hand_side(self):
        obs = SimpleNamespace(rotation=np.repeat(np.eye(3)[None], 3, 0))
        axis = render.side_axis(obs, 0)
        joints = np.zeros((3, 55, 3))
        joints[:, 0] = [0, 1, 0]
        joints[:, 21] = joints[:, 43] = [0, 1, 0] + 0.2 * axis
        track = dict(effector='RightHand', target='Hips')
        self.assertEqual(render.hand_side(dict(joints=joints), track, [0, 1, 2], obs), 1.0)
        joints[:, 21] = joints[:, 43] = [0, 1, 0] - 0.2 * axis
        self.assertEqual(render.hand_side(dict(joints=joints), track, [0, 1, 2], obs), -1.0)


class BodyFramingTest(unittest.TestCase):
    def layer(self, drop, role):
        rng = np.random.default_rng(3)
        joints = np.zeros((4, 55, 3))
        joints[:, :22] = render_rest() + [0, -drop, 0] + rng.normal(scale=0.01, size=(4, 22, 3))
        world = np.repeat(np.eye(3)[None, None], 4, 0).repeat(55, 1)
        return render.Layer(role, role, dict(joints=joints, world=world), None)

    def test_both_layers_fit_when_correction_moved_the_root(self):
        layers = [self.layer(0.0, 'reference'), self.layer(0.37, 'result')]
        frames = np.arange(4)
        renderer = render.Renderer(SimpleNamespace(faces=np.zeros((0, 3), int)), layers, frames, [], None, 900, [])
        width, height = render.BODY_PANEL
        for frame in frames:
            centre = np.mean([layer.state['joints'][frame, :22].mean(0) for layer in layers], 0)
            for direction in (renderer.forward, renderer.left):
                camera = render.look_at(centre + direction * renderer.distance + np.array([0, 0.15, 0]), centre, width, height)
                for layer in layers:
                    # The skin reaches about 0.2 m past the head and below the ankle joints.
                    joints = layer.state['joints'][frame, :22]
                    uv, _ = render.project(camera, np.concatenate([joints + [0, 0.2, 0], joints - [0, 0.2, 0]]))
                    self.assertTrue((uv[:, 1] > 0).all() and (uv[:, 1] < height).all())

    def test_one_layer_keeps_its_framing(self):
        layer = self.layer(0.0, 'result')
        frames = np.arange(4)
        renderer = render.Renderer(SimpleNamespace(faces=np.zeros((0, 3), int)), [layer], frames, [], None, 900, [])
        joints = layer.state['joints'][frames, :22]
        extent = float(np.abs(joints - joints.mean(1, keepdims=True)).max()) + 0.25
        self.assertAlmostEqual(renderer.distance, extent / np.tan(np.radians(render.FOV) / 2) * 1.05)


def render_rest():
    from test_anatomy import REST
    return REST + [0, 0.95, 0]


class GroundingNoticeTest(unittest.TestCase):
    def test_refused_and_ignored_camera_grounding_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            takes = []
            for name, entry in (('applied', {'camera_centre': [0, 0, 0], 'max_xz_m': 0.1}),
                                ('refused', {'warning': 'Camera grounding needs one static ExMR camera, Flat Ground and foot lock.'}),
                                ('ignored', None)):
                take = Path(folder) / name
                take.mkdir()
                report = {'floor': 0.0} if entry is None else {'floor': 0.0, 'camera_grounding': entry}
                (take / 'corrected_initial_correction.json').write_text(json.dumps(report), encoding='utf-8')
                takes.append(take)
            jobs = [(stage, ['--project', str(take), '--result', 'corrected_initial']) for take in takes for stage in ('correct', 'export')]
            refused = main.grounding_refusals(jobs)
        self.assertEqual([take.name for take, _ in refused], ['refused', 'ignored'])
        self.assertIn('static ExMR camera', refused[0][1])


if __name__ == '__main__':
    unittest.main()
