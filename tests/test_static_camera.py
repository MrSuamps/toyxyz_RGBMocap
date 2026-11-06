"""The one static ExMR camera: median centre / mean rotation of all frame pairings, shared by review, correction and the GUI."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mocap import scene_camera
from mocap.camera_reference import exmr_view, static_source_camera
from mocap.camera_tracks import projection, record

PELVIS = np.array([0.0, -0.3, 0.02])


def scene(frames=9, outlier=0.3, seed=3):
    """World / in-camera roots seen by one true camera (rotation, centre); frame 0's in-camera depth is off by `outlier`."""
    rng = np.random.default_rng(seed)
    camera = Rotation.from_euler('xyz', [8, -12, 2], degrees=True).as_matrix()
    centre = np.array([0.2, 1.4, -3.5])
    world_rotation = Rotation.from_rotvec(rng.normal(scale=0.6, size=(frames, 3))).as_matrix()
    world_translation = rng.normal(scale=0.4, size=(frames, 3))
    cam_rotation = camera @ world_rotation
    cam_translation = (world_translation + PELVIS - centre) @ camera.T - PELVIS
    cam_translation[0, 2] += outlier
    return camera, centre, world_rotation, cam_rotation, world_translation, cam_translation


def write_take(project, static=True, **options):
    """An ExMR take with the files the static camera reads."""
    import torch
    camera, centre, rw, rc, tw, tc = scene(**options)
    vector = lambda m: torch.tensor(Rotation.from_matrix(m).as_rotvec(), dtype=torch.float64)
    torch.save(dict(smpl_params_global=dict(global_orient=vector(rw), transl=torch.tensor(tw)),
                    smpl_params_incam=dict(global_orient=vector(rc), transl=torch.tensor(tc))), project / 'hmr4d_results.pt')
    np.savez(project / 'initial_joints.npz', joints=np.repeat((tw + PELVIS)[:, None], 22, 1))
    (project / 'exmr_camera.json').write_text(json.dumps({'identity': {'static': static}}), encoding='utf-8')
    np.save(project / 'exmr_camera.npy', np.tile(np.eye(3), (len(tw), 1, 1)))
    eye = np.eye(3).tolist()
    (project / 'exmr_camera_sources.json').write_text(json.dumps(dict(root='.', first=0, reference_rotation=eye, first_rotation=eye)),
                                                      encoding='utf-8')
    return camera, centre, rw, rc, tw, tc


class StaticCameraTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def test_median_pairing_ignores_an_outlier_frame(self):
        camera, centre, rw, rc, tw, tc = scene()
        matrix = scene_camera.static_paired_camera(rw, rc, tw, tc, PELVIS)
        np.testing.assert_allclose(matrix[:3, :3], camera, atol=1e-9)
        np.testing.assert_allclose(-matrix[:3, :3].T @ matrix[:3, 3], centre, atol=1e-9)
        frame0 = scene_camera.paired_camera(rw[0], rc[0], tw[0], tc[0], PELVIS)
        self.assertGreater(np.linalg.norm(-frame0[:3, :3].T @ frame0[:3, 3] - centre), 0.25)

    def test_every_consumer_uses_the_same_camera(self):
        from mocap.cli.review import exmr_camera_policy
        from mocap.correction import static_camera_centre
        import torch
        camera, centre, *_ = write_take(self.project)
        rotation, translation = exmr_view(self.project)
        np.testing.assert_allclose(rotation, camera, atol=1e-9)
        np.testing.assert_allclose(-rotation.T @ translation, centre, atol=1e-9)
        policy = exmr_camera_policy(self.project)
        np.testing.assert_array_equal(policy[0], rotation)
        np.testing.assert_array_equal(policy[1], translation)
        np.testing.assert_allclose(static_source_camera(self.project)[:3, :3], camera, atol=1e-9)
        source = torch.load(self.project / 'hmr4d_results.pt', weights_only=True)
        np.testing.assert_allclose(static_camera_centre(self.project, source, PELVIS), centre, atol=1e-9)

    def test_corrected_result_camera_follows_its_root_change(self):
        from mocap.cli.review import exmr_camera_policy
        camera, centre, rw, rc, tw, tc = write_take(self.project)
        drift = np.c_[np.zeros(len(tw)), -np.linspace(0, .2, len(tw)), np.zeros(len(tw))]
        with np.load(self.project / 'initial_joints.npz') as data:
            np.savez(self.project / 'corrected_initial_joints.npz', joints=data['joints'] + drift[:, None])
        rotation, translation = exmr_view(self.project, 'corrected_initial')
        np.testing.assert_allclose(-rotation.T @ translation, centre + np.median(drift, 0), atol=1e-9)
        np.testing.assert_array_equal(exmr_camera_policy(self.project, kind='corrected_initial')[1], translation)
        np.testing.assert_allclose(-exmr_view(self.project, 'initial')[0].T @ exmr_view(self.project, 'initial')[1], centre, atol=1e-9)

    def test_moving_and_missing_policies(self):
        self.assertIsNone(exmr_view(self.project))
        write_take(self.project, static=False)
        self.assertEqual(exmr_view(self.project), 'moving')
        from mocap.correction import static_camera_centre
        self.assertIsNone(static_camera_centre(self.project, None, PELVIS))

    def test_scene_receipt_records_the_pairing(self):
        write_take(self.project)
        self.assertEqual(scene_camera.scene_pairing(self.project), 'median')
        write_take(self.project, static=False)
        self.assertIsNone(scene_camera.scene_pairing(self.project))


class OverlayProjectionTest(unittest.TestCase):
    def test_overlay_track_replaces_the_rotation_track_for_the_video_overlay(self):
        matrices = np.tile(np.eye(4), (3, 1, 1))
        overlay = matrices.copy()
        overlay[:, :3, 3] = [[0, 0, 1], [0, 0, 2], [0, 0, 3]]
        camera = record('Camera 1', np.diag([500.0, 500.0, 1.0]), [640, 480], matrices, 10.0, 30.0, 'test', None, overlay)
        np.testing.assert_array_equal(projection(camera, 10.0 + 1 / 30), overlay[1])
        self.assertIsNone(projection(camera, 9.0))
        plain = record('Camera 1', np.diag([500.0, 500.0, 1.0]), [640, 480], matrices, 10.0, 30.0, 'test')
        np.testing.assert_array_equal(projection(plain, 10.0 + 2 / 30), matrices[2])
        with self.assertRaises(ValueError):
            record('Camera 1', np.diag([500.0, 500.0, 1.0]), [640, 480], matrices, 10.0, 30.0, 'test', None, overlay[:2])


if __name__ == '__main__':
    unittest.main()
