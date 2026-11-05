"""Self-contact grip: wrist limits, palm facing, declared wrist rows, finger curl write-back, and grip off = previous solver."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mocap import self_contact as sc  # noqa: E402
from mocap.correction import fk  # noqa: E402
from test_self_contact import FPS, N, OFFSETS, REST, clip, track  # noqa: E402

BASELINE_COMMIT = 'da61026'
LIMITS = sc.WRIST_LIMITS_DEG


def solve(tracks, foot_ranges=([], [])):
    root, q = clip(); before = q.copy()
    report = sc.apply(REST, root, q, OFFSETS, FPS, tracks, [list(r) for r in foot_ranges])
    return root, before, q, report


def wrist_angles(root, q, h):
    _, g = fk(REST, np.asarray(root, float), np.asarray(q, float))
    _, _, elbow, wrist = sc.ARMS[h]
    return sc.wrist_angles(g[:, elbow].swapaxes(-1, -2) @ g[:, wrist], REST, h)


def hand_rest(count):
    """Full rest joints (SMPL-X 55 or SMPL-H 52) with straight fingers continuing the forearms and thumbs forward."""
    rest = np.zeros((count, 3)); rest[:22] = REST; first = 25 if count == 55 else 22
    for h, sign in ((0, 1.), (1, -1.)):
        wrist = REST[20 + h]
        for finger in range(5):
            if finger == 4:
                base, step = wrist + [sign * .03, -.01, .03], np.array([sign * .015, 0., .025])
            else:
                base, step = wrist + [sign * .09, -.01, .03 - .02 * finger], np.array([sign * .03, 0., 0.])
            for segment in range(3):
                rest[first + 15 * h + 3 * finger + segment] = base + segment * step
    return rest


def load_baseline():
    """The committed self-contact solver before grip (git show), imported inside the mocap package; None without git."""
    try:
        source = subprocess.run(['git', 'show', f'{BASELINE_COMMIT}:mocap/self_contact.py'], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    folder = tempfile.mkdtemp(); path = Path(folder, 'self_contact_baseline.py'); path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('mocap._self_contact_baseline', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class GripOffTest(unittest.TestCase):
    def test_grip_off_matches_committed_solver(self):
        baseline = load_baseline()
        if baseline is None:
            self.skipTest('git history unavailable')
        cases = ([track()], [track(target='RightFoot', reciprocal='auto')], [track(target='LeftHand', reciprocal='always')],
                 [track(target='RightLeg', ranges=[(10, 25), (50, 70)])])
        for tracks in cases:
            root, q = clip(); old_q = q.copy()
            new = sc.apply(REST, root, q, OFFSETS, FPS, tracks, [[], []])
            old = baseline.apply(REST, root.copy(), old_q, OFFSETS, FPS, tracks, [[], []])
            self.assertEqual(q.tobytes(), old_q.tobytes(), tracks[0]['target'])
            self.assertEqual(new['changed'], old['changed'])
            self.assertEqual(new['wrist_changed'], []); self.assertEqual(new['fingers'], [])
            self.assertEqual([e['grip'] for e in new['tracks']], ['off'] * len(new['tracks']))

    def test_publish_without_fingers_is_a_no_op(self):
        params = {'right_hand_pose': torch.zeros(N, 45), 'rot6d': torch.zeros(1, N, 52, 6)}
        output = dict(params)
        self.assertEqual(sc.publish_fingers_smplx(output, params, {'fingers': []}, hand_rest(55), np.zeros((55, 3))), {})
        sc.publish_fingers_rot6d(output, {'fingers': []}, hand_rest(52))
        self.assertIs(output['right_hand_pose'], params['right_hand_pose'])
        self.assertEqual(output['rot6d'].abs().sum().item(), 0.)


class GripTest(unittest.TestCase):
    def test_wrist_angles_round_trip_and_clamp(self):
        rng = np.random.default_rng(3)
        for h in (0, 1):
            angles = np.c_[rng.uniform(-60, 70, 50), rng.uniform(-25, 15, 50), rng.uniform(-70, 70, 50)]
            angles = angles[sc.swing_cone(angles[:, 0], angles[:, 1]) <= 1]
            local = sc.wrist_local(angles[:, 0], angles[:, 1], angles[:, 2], REST, h)
            np.testing.assert_allclose(np.stack(sc.wrist_angles(local, REST, h), -1), angles, atol=1e-6)
            np.testing.assert_array_equal(sc.clamp_wrist(local, REST, h), local)
            wild = Rotation.random(200, random_state=h).as_matrix()
            f, r, p = sc.wrist_angles(sc.clamp_wrist(wild, REST, h), REST, h)
            self.assertLessEqual(sc.swing_cone(f, r).max(), 1 + 1e-6)
            self.assertTrue((p <= LIMITS['pronation'] + 1e-6).all() and (p >= -LIMITS['supination'] - 1e-6).all())

    def test_anatomical_signs(self):
        """Right hand (forearm -X, palm -Y): fingers toward the palm is flexion, toward the thumb (+Z) radial, thumb down pronation."""
        t = sc.forearm_axis(REST, 1)
        flexed = sc.wrist_local(30., 0., 0., REST, 1) @ t; radial = sc.wrist_local(0., 15., 0., REST, 1) @ t
        pronated = sc.wrist_local(0., 0., 40., REST, 1) @ sc.THUMB
        self.assertLess(flexed[1], -.3); self.assertGreater(radial[2], .1); self.assertLess(pronated[1], -.5)

    def test_grip_keeps_wrist_within_limits(self):
        for target, reciprocal in (('RightUpLeg', 'off'), ('RightFoot', 'auto'), ('LeftForeArm', 'off'), ('RightLeg', 'off')):
            root, _, q, report = solve([track(target=target, reciprocal=reciprocal, grip='auto')])
            entry = report['tracks'][0]; lo, hi = entry['window']
            f, r, p = (x[lo:hi] for x in wrist_angles(root, q, 1))
            self.assertLessEqual(sc.swing_cone(f, r)[30 - lo:60 - lo].max(), 1 + 1e-3, target)
            self.assertLessEqual(np.abs(p[30 - lo:60 - lo]).max(), 80 + .1, target)
            self.assertEqual(entry['wrist']['result']['frames_outside_limits'], 0, target)

    def test_palm_faces_reachable_limb(self):
        for target, reciprocal in (('RightUpLeg', 'off'), ('RightFoot', 'auto'), ('LeftForeArm', 'off')):
            _, _, _, off = solve([track(target=target, reciprocal=reciprocal)])
            _, _, _, grip = solve([track(target=target, reciprocal=reciprocal, grip='auto')])
            before, after = off['tracks'][0]['palm_facing_deg'], grip['tracks'][0]['palm_facing_deg']
            self.assertLess(after['after_max'], 30., target)
            self.assertLess(after['after_median'], before['after_median'], target)
            self.assertLess(grip['tracks'][0]['residual_max_m'], .02, target)

    def test_grip_twist_is_continuous(self):
        root, _, q, report = solve([track(target='RightFoot', reciprocal='auto', grip='auto')])
        lo, hi = report['tracks'][0]['window']
        _, g = fk(REST, root.astype(float), q.astype(float))
        self.assertLess(sc.rotation_angle(g[lo:hi - 1, 21], g[lo + 1:hi, 21]).max(), 20.)

    def test_wrist_rows_are_declared(self):
        root, before, q, report = solve([track(grip='auto')])
        changed = report['changed'][0]
        self.assertEqual(report['wrist_changed'], [dict(joint=21, start=changed['start'], end=changed['end'])])
        _, g0 = fk(REST, root.astype(float), before.astype(float)); _, g1 = fk(REST, root.astype(float), q.astype(float))
        turn = np.rad2deg(Rotation.from_matrix(g0[:, 21].swapaxes(-1, -2) @ g1[:, 21]).magnitude())
        outside = np.ones(N, bool); outside[changed['start']:changed['end']] = False
        self.assertEqual(turn[outside].max(), 0.)
        self.assertGreater(turn[~outside].max(), 1.)
        self.assertLess(Rotation.from_matrix(g0[:, 20].swapaxes(-1, -2) @ g1[:, 20]).magnitude().max(), 1e-6)

    def test_audit_exemptions(self):
        preserved = (0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21)
        report = dict(changed=[dict(joints=[14, 17, 19, 21], start=20, end=70)])
        self.assertEqual(sc.audit_exemptions(report, preserved), [(20, 70, [7, 9])])
        report['wrist_changed'] = [dict(joint=21, start=24, end=66)]
        self.assertEqual(sc.audit_exemptions(report, preserved), [(20, 70, [7, 9]), (24, 66, [11])])
        self.assertEqual(sc.audit_exemptions(None, preserved), [])
        self.assertEqual(sc.audit_exemptions(dict(changed=[], wrist_changed=[dict(joint=19, start=0, end=5)]), preserved), [])


class FingerTest(unittest.TestCase):
    SPEC = dict(hand=1, window=[20, 70], contact=[30, 60], ease=10, curl=1.)

    def test_grip_shape_curls_toward_the_palm(self):
        for count in (55, 52):
            rest = hand_rest(count); first = (25 if count == 55 else 22) + 15
            shape = sc.grip_shape(rest, 1)
            for finger in range(5):
                chain = [first + 3 * finger + k for k in range(3)]
                position = rest[chain[0]].copy(); world = np.eye(3)
                for k in range(3):
                    world = world @ shape[3 * finger + k]
                    step = rest[chain[k + 1]] - rest[chain[k]] if k < 2 else rest[chain[2]] - rest[chain[1]]
                    position = position + world @ step
                self.assertLess(position[1] - rest[chain[0], 1], -.005, (count, finger))

    def test_smplx_write_back(self):
        rest = hand_rest(55); mean = np.random.default_rng(1).normal(0, .1, (55, 3))
        params = {'right_hand_pose': torch.zeros(N, 45), 'left_hand_pose': torch.zeros(N, 45)}
        output = dict(params); report = {'fingers': [dict(self.SPEC)]}
        replaced = sc.publish_fingers_smplx(output, params, report, rest, mean)
        self.assertIs(replaced['right_hand_pose'], params['right_hand_pose'])
        self.assertEqual(params['right_hand_pose'].abs().sum().item(), 0.)
        self.assertIs(output['left_hand_pose'], params['left_hand_pose'])
        value = output['right_hand_pose'].numpy()
        self.assertEqual(np.abs(value[:20]).max(), 0.); self.assertEqual(np.abs(value[70:]).max(), 0.)
        self.assertGreater(np.abs(value[30:60]).max(), .3)
        full = Rotation.from_rotvec((value[40].reshape(15, 3) + mean[40:55]).astype(float)).as_matrix()
        np.testing.assert_allclose(full, sc.grip_shape(rest, 1), atol=1e-5)
        self.assertTrue(report['fingers'][0]['applied'])

    def test_smplx_pca_hands_are_skipped(self):
        params = {'right_hand_pose': torch.zeros(N, 12)}; output = dict(params); report = {'fingers': [dict(self.SPEC)]}
        self.assertEqual(sc.publish_fingers_smplx(output, params, report, hand_rest(55), np.zeros((55, 3))), {})
        self.assertIs(output['right_hand_pose'], params['right_hand_pose'])
        self.assertFalse(report['fingers'][0]['applied']); self.assertIn('45-D', report['fingers'][0]['note'])

    def test_rot6d_write_back(self):
        rest = hand_rest(52)
        rot6d = torch.from_numpy(np.tile(np.eye(3)[:, :2].reshape(6), (1, N, 52, 1)).astype(np.float32))
        output = {'rot6d': rot6d.clone()}; report = {'fingers': [dict(self.SPEC)]}
        sc.publish_fingers_rot6d(output, report, rest)
        changed = (output['rot6d'] != rot6d).any(-1)[0]
        self.assertFalse(changed[:20].any() or changed[70:].any() or changed[:, :37].any())
        self.assertTrue(changed[30:60, 37:52].all())
        np.testing.assert_allclose(sc.rot6d_matrix(output['rot6d'][0, 45, 37:52].double().numpy()), sc.grip_shape(rest, 1), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
