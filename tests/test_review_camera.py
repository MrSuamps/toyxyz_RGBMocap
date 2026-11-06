import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mocap.cli import review


def write_camera(project, static):
    (project / 'exmr_camera.json').write_text(json.dumps({'identity': {'static': static}}), encoding='utf-8')
    np.save(project / 'exmr_camera.npy', np.tile(np.eye(3), (4, 1, 1)))
    eye = np.eye(3).tolist()
    (project / 'exmr_camera_sources.json').write_text(json.dumps(dict(root='.', first=0, reference_rotation=eye, first_rotation=eye)),
                                                      encoding='utf-8')


def write_joints(project, kind, joints):
    np.savez(project / f'{kind}_joints.npz', joints=joints)


class ReviewCameraTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        angle = np.radians(10)
        self.rotation = np.stack([np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) for a in angle * np.arange(4)])
        self.translation = np.array([[0.0, 1.0, 3.0], [0.1, 1.0, 3.0], [0.2, 1.1, 3.0], [0.3, 1.2, 3.0]])

    def tearDown(self):
        self.folder.cleanup()

    def test_static_camera_is_the_shared_static_camera(self):
        from mocap.camera_reference import exmr_view
        from test_static_camera import write_take
        camera, centre, *_ = write_take(self.project)
        rotation, translation = review.exmr_camera_policy(self.project)
        np.testing.assert_array_equal(rotation, exmr_view(self.project)[0])
        np.testing.assert_allclose(rotation, camera, atol=1e-9)
        np.testing.assert_allclose(-rotation.T @ translation, centre, atol=1e-9)

    def test_missing_policy_keeps_per_frame_cameras(self):
        self.assertIsNone(review.exmr_camera_policy(self.project))

    def test_moving_camera_follows_the_corrected_root(self):
        write_camera(self.project, False)
        self.assertEqual(review.exmr_camera_policy(self.project), 'moving')
        joints = np.random.default_rng(1).normal(size=(4, 22, 3))
        corrected = joints.copy()
        corrected[:, :, 1] -= np.linspace(0, 0.4, 4)[:, None]
        write_joints(self.project, 'initial', joints)
        write_joints(self.project, 'corrected_initial', corrected)
        obs = review.Observation(np.zeros((4, 63, 3)), self.rotation, self.translation, np.eye(3), Path('v.mp4'), 30.0, 'moving', None,
                                 self.translation)
        followed = obs.follow(self.project, 'corrected_initial')
        self.assertEqual(followed.follows, 'corrected_initial')
        before = np.einsum('fij,fkj->fki', obs.rotation, joints) + obs.translation[:, None]
        after = np.einsum('fij,fkj->fki', followed.rotation, corrected) + followed.translation[:, None]
        np.testing.assert_allclose(after, before, atol=1e-12)
        self.assertIs(followed.follow(self.project, 'initial').follows, None)
        np.testing.assert_array_equal(followed.follow(self.project, 'initial').translation, self.translation)

    def test_body_offset_flags_a_lasting_excess(self):
        offset = np.zeros((60, 2))
        offset[10:50, 1] = -40.0
        joints = np.zeros((60, 22, 3))
        write_joints(self.project, 'initial', joints)
        ctx = type('Ctx', (), {})()
        ctx.frames, ctx.fps, ctx.torso, ctx.offset = np.arange(60), 30.0, np.full(60, 100.0), offset
        ctx.obs = review.Observation(np.zeros((60, 63, 3)), np.tile(np.eye(3), (60, 1, 1)), np.tile([0.0, 0.0, 3.0], (60, 1)), np.eye(3),
                                     Path('v.mp4'), 30.0, 'fixed')
        ctx.video = lambda i: int(ctx.frames[i])
        ctx.placement_offset = lambda uv: (np.zeros((60, 2)), np.ones(60, bool))
        issues = review.check_body_offset(ctx, self.project, 'corrected_initial')
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['check'], 'body_offset')
        self.assertEqual(issues[0]['frames'], [10, 49])
        self.assertEqual(issues[0]['metrics']['direction'], 'below')
        self.assertEqual(issues[0]['severity'], 'high')
        ctx.offset = offset * 0.4
        self.assertEqual(review.check_body_offset(ctx, self.project, 'corrected_initial'), [])
        self.assertEqual(review.check_body_offset(ctx, self.project, 'initial'), [])


if __name__ == '__main__':
    unittest.main()
