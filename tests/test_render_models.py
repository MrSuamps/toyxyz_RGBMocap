"""render on FlowHMR (SMPL-H) and GVHMR: the SMPL-H body, FlowHMR params, the saved-camera adapter and the body choice."""
import os
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

from mocap.cli import render, smplx_body  # noqa: E402
from mocap.cli.pipeline import CliError  # noqa: E402
from mocap.self_contact import segment_vertices  # noqa: E402

HAND = [20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35]
SMPLH_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] + HAND + [p + 1 if p == 20 else p + 15 for p in HAND]


def write_smplh(path, vertices=80, seed=0):
    """A tiny random model with the AMASS SMPL-H npz layout (52 joints, 16 betas)."""
    rng = np.random.default_rng(seed)
    regressor = rng.random((52, vertices))
    weights = rng.random((vertices, 52)) ** 4
    kintree = np.stack([np.array(SMPLH_PARENTS, np.int64) % 2 ** 32, np.arange(52)]).astype(np.uint32)
    np.savez(path, v_template=rng.normal(size=(vertices, 3)), shapedirs=rng.normal(scale=0.01, size=(vertices, 3, 16)),
             J_regressor=regressor / regressor.sum(1, keepdims=True), weights=weights / weights.sum(1, keepdims=True),
             kintree_table=kintree, f=rng.integers(0, vertices, (40, 3)), posedirs=np.zeros((vertices, 3, 459)))


def rot6d(matrices):
    """FlowHMR layout: the first two rotation-matrix columns, interleaved per row (view(..., 3, 2))."""
    return np.asarray(matrices)[..., :, :2].reshape(np.shape(matrices)[:-2] + (6,))


def reference_lbs(path, rotations, betas, trans):
    """Plain 52-joint SMPL-H LBS without pose correctives, independent of SmplhBody."""
    data = np.load(path)
    shaped = data['v_template'] + data['shapedirs'] @ betas
    rest = data['J_regressor'] @ shaped
    world, joints = np.empty((52, 3, 3)), np.empty((52, 3))
    for j, p in enumerate(SMPLH_PARENTS):
        world[j] = rotations[j] if p < 0 else world[p] @ rotations[j]
        joints[j] = rest[j] if p < 0 else joints[p] + world[p] @ (rest[j] - rest[p])
    offset = joints - np.einsum('jab,jb->ja', world, rest)
    blend = np.einsum('vj,jab->vab', data['weights'], world)
    return np.einsum('vab,vb->va', blend, shaped) + data['weights'] @ offset + trans, joints + trans


class SmplhBodyTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'model.npz'
        write_smplh(self.path)
        self.body = smplx_body.SmplhBody(self.path)
        rng = np.random.default_rng(3)
        self.rotations = Rotation.from_rotvec(rng.normal(scale=0.6, size=(2 * 52, 3))).as_matrix().reshape(2, 52, 3, 3)
        self.betas = rng.normal(size=16)
        self.trans = rng.normal(size=(2, 3))
        self.params = dict(rot6d=rot6d(self.rotations)[None], shapes=self.betas.reshape(1, 1, 16), trans=self.trans[None])

    def tearDown(self):
        self.folder.cleanup()

    def test_rot6d_round_trip(self):
        np.testing.assert_allclose(smplx_body.rot6d_matrix(rot6d(self.rotations)), self.rotations, atol=1e-12)
        skewed = rot6d(self.rotations) * 1.7
        np.testing.assert_allclose(smplx_body.rot6d_matrix(skewed), self.rotations, atol=1e-12)

    def test_vertices_and_joints_match_plain_lbs(self):
        state = self.body.pose(self.params)
        for i in range(2):
            vertices, joints = reference_lbs(self.path, self.rotations[i], self.betas, self.trans[i])
            np.testing.assert_allclose(self.body.vertices(state, i), vertices, atol=1e-10)
            np.testing.assert_allclose(state['joints'][i, smplx_body.SMPLH_TO_SMPLX], joints, atol=1e-10)

    def test_smplx_layout(self):
        state = self.body.pose(self.params)
        self.assertEqual(state['joints'].shape, (2, 55, 3))
        # Jaw and eyes sit on the head and carry no skin, so SMPL-X joint indices (wrist 20, middle knuckle 28) mean the same.
        np.testing.assert_allclose(state['joints'][:, 22:25], np.repeat(state['joints'][:, 15:16], 3, 1))
        np.testing.assert_array_equal(self.body.weights[:, 22:25], 0)
        np.testing.assert_allclose(self.body.weights.sum(1), 1)
        self.assertEqual([self.body.parents[j] for j in (25, 26, 28, 40, 41, 22, 23)], [20, 25, 20, 21, 40, 15, 15])
        groups = segment_vertices(self.body.weights)
        dominant = np.asarray(np.load(self.path)['weights']).argmax(1)
        np.testing.assert_array_equal(groups['right_hand'], np.flatnonzero(np.isin(dominant, (21,) + tuple(range(37, 52)))))

    def test_rest_pose_is_the_shaped_template(self):
        params = dict(rot6d=rot6d(np.tile(np.eye(3), (1, 52, 1, 1))), shapes=self.betas, trans=np.zeros((1, 3)))
        data = np.load(self.path)
        np.testing.assert_allclose(self.body.vertices(self.body.pose(params), 0), data['v_template'] + data['shapedirs'] @ self.betas, atol=1e-12)


class ParamsTest(unittest.TestCase):
    def test_flowhmr_record_keeps_only_body_fields(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'hmr4d_results.pt'
            torch.save(dict(rot6d=torch.zeros(1, 4, 52, 6), shapes=torch.zeros(1, 1, 16), trans=torch.ones(1, 4, 3),
                            foot_contacts=torch.zeros(1, 4, 4), source_representation='smplh', flow_version=1), path)
            params = smplx_body.load_params(path)
            self.assertEqual(sorted(params), ['rot6d', 'shapes', 'trans'])
            self.assertEqual(params['trans'].dtype, np.float64)
            nested = Path(folder) / 'exmr.pt'
            torch.save(dict(smpl_params_global=dict(body_pose=torch.zeros(4, 63), transl=torch.ones(4, 3)), source='initial'), nested)
            self.assertEqual(sorted(smplx_body.load_params(nested)), ['body_pose', 'transl'])

    def test_gui_vertices_reads_the_native_smplh_surface(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder)
            surface = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
            np.save(project / 'initial_flow-x-surface.npy', surface)
            (project / 'initial_mesh.json').write_text('{"representation": "smplh", "frames_file": "initial_flow-x-surface.npy", '
                                                       '"asset": "missing.npz"}', encoding='utf-8')
            np.testing.assert_array_equal(smplx_body.gui_vertices(project, 'initial', 2), surface[2])
            self.assertIsNone(smplx_body.gui_vertices(project, 'corrected_initial', 2))


class BodyChoiceTest(unittest.TestCase):
    def test_flowhmr_uses_smplh_and_reports_a_missing_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            with mock.patch.object(smplx_body, 'smplh_path', lambda: path):
                with self.assertRaises(FileNotFoundError):
                    smplx_body.body_for(folder, 'flowhmr')
                session = SimpleNamespace(info=lambda message: None)
                with self.assertRaises(CliError) as caught:
                    render.load_body(session, folder, 'flowhmr')
                self.assertEqual(caught.exception.code, 3)
                write_smplh(path)
                body = smplx_body.body_for(folder, 'flowhmr')
                self.assertIsInstance(body, smplx_body.SmplhBody)
                self.assertEqual(render.body_name(body), 'SMPL-H')
                self.assertIn('no pose correctives', render.geometry(body))
        self.assertEqual(render.body_name(SimpleNamespace()), 'SMPL-X')
        self.assertEqual(render.geometry(SimpleNamespace()), 'SMPL-X neutral LBS with shape and pose correctives (GUI Mesh view)')


class SavedCameraTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        (self.project / 'input.mp4').write_bytes(b'')
        self.matrix = np.eye(4)
        self.matrix[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        self.matrix[:3, 3] = [0.0, 0.7, 0.0]
        self.K = np.array([[1000.0, 0, 360], [0, 1000.0, 640], [0, 0, 1]])

    def tearDown(self):
        self.folder.cleanup()

    def load(self, matrices, model='flowhmr', count=3, size=(720, 1280)):
        loaded = dict(cameras=[dict(K=self.K, size=(720, 1280), matrices=np.asarray(matrices))])
        with mock.patch('mocap.camera_tracks.load', return_value=loaded) as load, mock.patch.object(render, 'video_size', return_value=size), \
                mock.patch('mocap.camera_reference.reference', return_value={'schema': 2}) as reference:
            camera = render.saved_camera(self.project, model, count)
        return camera, load, reference

    def test_single_matrix_is_one_fixed_camera(self):
        camera, load, reference = self.load(self.matrix[None])
        self.assertEqual(camera.rotation.shape, (3, 3, 3))
        np.testing.assert_array_equal(camera.translation, np.tile([0.0, 0.7, 0.0], (3, 1)))
        self.assertIn('one fixed camera', camera.describe())
        self.assertIsNone(camera.keypoints)
        self.assertIs(camera.follow(self.project, 'corrected_initial'), camera)
        reference.assert_not_called()
        self.assertEqual(camera.video, self.project.resolve() / 'input.mp4')

    def test_gvhmr_reads_the_paired_reference_in_process(self):
        _, load, reference = self.load(self.matrix[None], 'gvhmr')
        reference.assert_called_once()
        self.assertEqual(load.call_args.kwargs['reference_cache'], {str(self.project.resolve()): {'schema': 2}})

    def test_per_frame_rotation_and_video_scaling(self):
        frames = np.repeat(self.matrix[None], 3, 0)
        frames[1, :3, :3] = Rotation.from_rotvec([0, 0.1, 0]).as_matrix() @ frames[1, :3, :3]
        camera, _, _ = self.load(frames, size=(360, 640))
        self.assertIn('per-frame rotation', camera.describe())
        np.testing.assert_allclose(camera.K, [[500.0, 0, 180], [0, 500.0, 320], [0, 0, 1]])
        np.testing.assert_allclose(camera.rotation[1], frames[1, :3, :3])

    def test_frame_count_must_match(self):
        with self.assertRaises(CliError):
            self.load(np.repeat(self.matrix[None], 2, 0))

    def test_camera_errors_become_cli_errors(self):
        with mock.patch('mocap.camera_tracks.load', side_effect=ValueError('FlowHMR placement is stale. Use Prepare Preview.')):
            with self.assertRaises(CliError) as caught:
                render.saved_camera(self.project, 'flowhmr', 3)
        self.assertIn('Prepare Preview', caught.exception.message)

    def test_exmr_keeps_the_review_observation(self):
        with mock.patch('mocap.cli.review.load_exmr', return_value='observation') as load:
            self.assertEqual(render.load_cameras(self.project, 'exmr', 'corrected_initial', 3), 'observation')
        load.assert_called_once_with(self.project, 'corrected_initial')


@unittest.skipUnless(os.environ.get('RGBMOCAP_FLOWHMR_PROJECT'), 'set RGBMOCAP_FLOWHMR_PROJECT to a FlowHMR model project (read only)')
class FlowhmrProjectTest(unittest.TestCase):
    def test_smplh_body_matches_the_gui_surface(self):
        project = Path(os.environ['RGBMOCAP_FLOWHMR_PROJECT'])
        body = smplx_body.body_for(project, 'flowhmr')
        state = body.pose(smplx_body.load_params(project / 'hmr4d_results.pt'))
        for frame in (0, len(state['joints']) - 1):
            self.assertLess(np.abs(body.vertices(state, frame) - smplx_body.gui_vertices(project, 'initial', frame)).max(), 1e-5)


if __name__ == '__main__':
    unittest.main()
