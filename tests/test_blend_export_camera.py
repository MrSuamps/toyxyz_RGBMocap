import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mocap import blend_export
from mocap.blend_support import Inputs
from mocap.cli import review

FRAMES = 6


def write_moving_exmr(project, rng):
    """A moving-camera ExMR take whose saved visual-odometry rotations disagree with the body's paired rotation."""
    world_orient = Rotation.random(FRAMES, random_state=1).as_rotvec()
    camera = Rotation.from_euler('y', np.linspace(0, 30, FRAMES), degrees=True)
    incam_orient = (Rotation.from_euler('x', rng.normal(0, 6, FRAMES), degrees=True) * camera * Rotation.from_rotvec(world_orient)).as_rotvec()
    world_t = np.c_[np.linspace(0, 1.5, FRAMES), np.full(FRAMES, .9), np.linspace(0, .5, FRAMES)]
    incam_t = np.c_[rng.normal(0, .05, FRAMES), np.full(FRAMES, .1), np.full(FRAMES, 3.)]
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    torch.save(dict(smpl_params_global=dict(global_orient=tensor(world_orient), transl=tensor(world_t)),
                    smpl_params_incam=dict(global_orient=tensor(incam_orient), transl=tensor(incam_t))), project / 'hmr4d_results.pt')
    pelvis = np.array([0., -.35, .02])
    joints = rng.normal(0, .3, (FRAMES, 22, 3)) + world_t[:, None]
    joints[:, 0] = world_t.astype(np.float32) + pelvis
    np.savez(project / 'initial_joints.npz', joints=joints)
    corrected = joints.copy()
    corrected[:, :, 1] -= np.linspace(0, .3, FRAMES)[:, None]
    corrected[:, :, 0] += .1
    np.savez(project / 'corrected_initial_joints.npz', joints=corrected)
    torch.save(torch.zeros(FRAMES, 63, 3), project / 'exmr_keypoints.pt')
    np.savez(project / 'tracking.npz', K=np.array([[1000., 0, 360], [0, 1000, 640], [0, 0, 1]]))
    (project / 'project.json').write_text(json.dumps(dict(fps=30.0, start=0.0, frames=FRAMES)), encoding='utf-8')
    (project / 'input.mp4').write_bytes(b'')
    (project / 'exmr_camera.json').write_text(json.dumps({'identity': {'static': False}}), encoding='utf-8')
    np.save(project / 'exmr_camera.npy', camera.as_matrix())
    eye = np.eye(3).tolist()
    (project / 'exmr_camera_sources.json').write_text(json.dumps(dict(root='.', first=0, reference_rotation=eye, first_rotation=eye)),
                                                      encoding='utf-8')
    return joints, corrected


class BlendExportCameraTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        self.joints, self.corrected = write_moving_exmr(self.project, np.random.default_rng(3))
        patcher = mock.patch('mocap.result_artifacts.artifacts_current', return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.folder.cleanup()

    def test_moving_camera_is_the_review_camera(self):
        """The exported moving camera must be the per-frame camera review/render project with, not the rotation track."""
        for kind in ('corrected_initial', 'initial'):
            track, follows = blend_export.person_camera(self.project, kind, FRAMES, 1., Inputs())
            self.assertEqual(follows, kind == 'corrected_initial')
            obs = review.load_exmr(self.project, kind)
            np.testing.assert_allclose(track[:, :3, :3], obs.rotation, atol=1e-9)
            np.testing.assert_allclose(track[:, :3, 3], obs.translation, atol=1e-9)
        rotations = np.load(self.project / 'exmr_camera.npy')
        self.assertGreater(np.abs(track[:, :3, :3] - rotations).max(), .05)

    def test_corrected_body_keeps_its_place_in_view(self):
        original, _ = blend_export.person_camera(self.project, 'initial', FRAMES, 1., Inputs())
        followed, _ = blend_export.person_camera(self.project, 'corrected_initial', FRAMES, 1., Inputs())
        view = lambda m, p: np.einsum('fij,fkj->fki', m[:, :3, :3], p) + m[:, None, :3, 3]
        np.testing.assert_allclose(view(followed, self.corrected[:, :1]), view(original, self.joints[:, :1]), atol=1e-9)

    def test_movement_scale_zero_fixes_the_original_centre(self):
        track, _ = blend_export.person_camera(self.project, 'corrected_initial', FRAMES, 0., Inputs())
        original, _ = blend_export.person_camera(self.project, 'initial', FRAMES, 1., Inputs())
        centres = -np.einsum('fji,fj->fi', track[:, :3, :3], track[:, :3, 3])
        np.testing.assert_allclose(centres, np.broadcast_to(centres[0], centres.shape), atol=1e-9)
        np.testing.assert_allclose(centres[0], -original[0, :3, :3].T @ original[0, :3, 3], atol=1e-9)

    def test_frame_count_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            blend_export.person_camera(self.project, 'initial', FRAMES + 1, 1., Inputs())


if __name__ == '__main__':
    unittest.main()
