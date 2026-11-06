"""Skin anchor on a held limb: the hand wrapped across the limb there should point as the video hand does."""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mocap import self_contact as sc  # noqa: E402
from mocap.correction import fk  # noqa: E402
from test_self_contact import FPS, N, OFFSETS, REST, clip, track  # noqa: E402
from test_skin_contact import FakeVideo, synthetic_skin  # noqa: E402


def draw_hand(video, rows, wrist, direction):
    """Right-hand keypoints at pixel wrist (2,), 80 px long along direction (2,), knuckle row across it."""
    d = np.asarray(direction, float) / np.linalg.norm(direction); across = np.array([-d[1], d[0]])
    wrist = np.asarray(wrist, float); middle = wrist + 80 * d
    video.keypoints[rows[:, None], [10, 51, 47, 59], :2] = [wrist, middle, middle + 20 * across, middle - 20 * across]


class WrapDegTest(unittest.TestCase):
    def test_wrapped_direction_against_the_video_hand(self):
        rows = np.arange(2); video = FakeVideo(2)
        # Vertical limb; a candidate facing the camera wraps the fingers sideways, one on the limb's side points them at the camera.
        goal = np.broadcast_to([0., .5, 0.], (2, 2, 3)); normals = np.broadcast_to([[0., 0., 1.], [1., 0., 0.]], (2, 2, 3))
        limb = np.broadcast_to([0., 1., 0.], (2, 2, 3)); weights = np.full(2, .5)
        draw_hand(video, rows, [500, 500], [-1, 0])
        np.testing.assert_allclose(sc.anchor_wrap_deg(video, 1, rows, weights, goal, normals, limb), [0., 90.], atol=1e-6)
        draw_hand(video, rows, [500, 500], [1, .2])
        self.assertAlmostEqual(sc.anchor_wrap_deg(video, 1, rows, weights, goal, normals, limb)[0], np.rad2deg(np.arctan(.2)), places=6)
        draw_hand(video, rows, [500, 500], [0, 1])
        np.testing.assert_allclose(sc.anchor_wrap_deg(video, 1, rows, weights, goal, normals, limb), [90., 90.], atol=1e-6)


class AnchorTest(unittest.TestCase):
    def anchor(self, weight):
        skin = synthetic_skin(); root, q = clip(); a, b = 30, 60; rows = np.arange(a, b)
        j, g = fk(REST, root.astype(float), q.astype(float))
        video = FakeVideo(N)
        palm = j[:, 21] + np.einsum('tij,j->ti', g[:, 21], OFFSETS[1])
        draw_hand(video, np.arange(N), [0, 0], [-1, 0])
        uv = video.project(palm[:, None], np.arange(N))[:, 0]
        video.keypoints[:, [10, 51, 47, 59], :2] += uv[:, None] - video.keypoints[:, [10], :2]
        tj, side, part = sc.parse_target('RightUpLeg')
        with mock.patch.object(sc, 'ANCHOR_WRAP_M_PER_DEG', weight):
            return sc.skin_anchor(skin, REST, root[a:b].astype(float), q[a:b].astype(float), rows, 1, tj, part, side, palm[a:b], video)[3]

    def test_the_wrap_term_picks_a_side_the_hand_can_wrap_as_in_the_video(self):
        # The source hand hangs beside the thigh and the video hand points sideways: the nearest point (on the outer corner)
        # wraps the fingers toward the camera, the wrap term moves the anchor off the outer side to where they point sideways.
        plain, wrapped = self.anchor(0.), self.anchor(sc.ANCHOR_WRAP_M_PER_DEG)
        self.assertGreater(plain['wrap_deg'], 45.); self.assertLess(plain['normal_local'][0], -.3)
        self.assertLess(wrapped['wrap_deg'], 15.); self.assertLess(abs(wrapped['normal_local'][0]), .1)
        self.assertGreater(wrapped['cost_m']['total'], plain['cost_m']['total'])

    def test_no_wrap_term_without_video_or_off_limbs(self):
        skin = synthetic_skin(); root, q = clip(); rows = np.arange(30, 60)
        j, g = fk(REST, root.astype(float), q.astype(float)); palm = j[30:60, 21]
        tj, side, part = sc.parse_target('RightUpLeg')
        report = sc.skin_anchor(skin, REST, root[30:60].astype(float), q[30:60].astype(float), rows, 1, tj, part, side, palm)[3]
        self.assertNotIn('wrap_deg', report)
        self.assertNotIn('Spine', sc.LIMB_PARTS); self.assertNotIn('Hand', sc.LIMB_PARTS)


if __name__ == '__main__':
    unittest.main()
