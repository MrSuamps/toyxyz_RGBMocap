"""`contacts suggest`: missed-contact naming, suggestions that do not stale the corrected result, hands sunk into the body."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import foot_contacts, self_contact_store as store  # noqa: E402
from mocap.cli import anatomy, contacts, review  # noqa: E402
from mocap.cli.main import build_parser  # noqa: E402

BASE, KIND = 'initial', 'corrected_initial'
SOURCES = {'hmr4d_results.pt': [1, 2]}
CONTEXT = {'frames': 421, 'fps': 30.0, 'sources': SOURCES}
FPS = 30.0


def track(**values):
    return dict(dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]]), **values)


class Session:
    def __init__(self, dry_run=False):
        self.dry_run, self.lines = dry_run, []

    def say(self, text):
        self.lines.append(text)

    info = say


# Missed-contact naming -------------------------------------------------------------------------------------------

class MissedContactNameTest(unittest.TestCase):
    def context(self, right_error):
        n = 10
        kp = np.zeros((n, 63, 3))
        kp[:, 9] = (100, 100, 1)        # left wrist
        kp[:, 10] = (110, 100, 1)       # right wrist, 0.05 torso away in 2D
        joints = np.zeros((n, 22, 3))
        joints[:, 16], joints[:, 18], joints[:, 20] = (-0.2, 1.4, 0), (-0.2, 1.1, 0.1), (0.0, 1.0, 0.0)
        joints[:, 17], joints[:, 19], joints[:, 21] = (0.2, 1.4, 0), (0.2, 1.1, 0.3), (0.1, 1.0, 0.4)
        err = np.full((n, len(review.POSE)), 0.1)
        err[:, review.POSE_INDEX['right_wrist']] = right_error
        return SimpleNamespace(keypoints=kp, joints=joints, torso=np.full(n, 200.0), err=err, frames=np.arange(n),
                               project=lambda points: points[..., :2] * 100, video=lambda i: int(i))

    def test_message_names_the_hand_it_pins_onto(self):
        for right_error, effector, target in ((0.5, 'RightHand', 'LeftHand'), (0.0, 'LeftHand', 'RightHand')):
            issues = [i for i in review.check_missed_contacts(self.context(right_error)) if i['target'].endswith('Hand')]
            self.assertEqual(len(issues), 1)
            issue = issues[0]
            self.assertEqual((issue['effector'], issue['target']), (effector, target))
            self.assertTrue(issue['message'].startswith(f'{effector} touches {target} in the video'), issue['message'])
            self.assertEqual(issue['joints'], ['left_wrist', 'right_wrist'])


# Applied revision ------------------------------------------------------------------------------------------------

class AppliedRevisionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        self.foot = foot_contacts.save_edit(self.project, BASE, None, CONTEXT)['revision']

    def tearDown(self):
        self.folder.cleanup()

    def state(self):
        return json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))

    def test_suggestions_alone_keep_the_revision_of_no_file(self):
        store.save(self.project, BASE, [track(ranges=[], automatic=[[10, 20]], source='review:a')], CONTEXT)
        self.assertIsNone(store.revision(self.project, BASE))
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), self.foot)
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT)[1], self.foot)
        # Settings of a suggestion-only track and a disabled track with intervals apply nothing either.
        store.save(self.project, BASE, [track(ranges=[], automatic=[[30, 40]], distance=0.1), track(target='Head', enabled=False)], CONTEXT)
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), self.foot)

    def test_only_applied_content_changes_the_revision(self):
        first = store.save(self.project, BASE, [track(automatic=[[10, 20]])], CONTEXT)
        self.assertEqual(store.revision(self.project, BASE), first)
        steps = (([track(automatic=[[50, 60]], source='review:b')], False),
                 ([track(automatic=[[50, 60]]), track(effector='LeftHand', target='Head', ranges=[], automatic=[[1, 9]])], False),
                 ([track(ranges=[[345, 400]])], True),
                 ([track(ranges=[[345, 400]], grip='auto')], True),
                 ([track(ranges=[[345, 400]], grip='auto'), track(effector='LeftHand', target='Head', ranges=[[1, 9]])], True),
                 ([track(ranges=[[345, 400]], grip='auto'), track(effector='LeftHand', target='Head', ranges=[[1, 9]], enabled=False)], True))
        for tracks, changes in steps:
            before = store.revision(self.project, BASE)
            payload = store.save(self.project, BASE, tracks, CONTEXT)
            self.assertEqual(store.revision(self.project, BASE) != before, changes, tracks)
            self.assertEqual(store.read(self.project, BASE, CONTEXT)[1], payload)
        applied = store.revision(self.project, BASE)
        store.save_articulation(self.project, BASE, [dict(foot='LeftFoot', mode='pointe', ranges=[[0, 10]])], CONTEXT)
        self.assertNotEqual(store.revision(self.project, BASE), applied)
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), f'{self.foot}|sc:{store.revision(self.project, BASE)}')

    def test_emptied_tracks_fall_back_to_no_self_contacts(self):
        store.save(self.project, BASE, [track()], CONTEXT)
        store.save(self.project, BASE, [track(ranges=[], automatic=[[345, 421]])], CONTEXT)
        self.assertIsNone(store.revision(self.project, BASE))

    def test_state_written_before_applied_keeps_its_revision(self):
        payload = store.save(self.project, BASE, [track(automatic=[[1, 5]])], CONTEXT)
        legacy = {k: v for k, v in self.state().items() if k != 'applied'}
        store.state_path(self.project, BASE).write_text(json.dumps(legacy), encoding='utf-8')
        self.assertEqual(store.revision(self.project, BASE), payload)
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), f'{self.foot}|sc:{payload}')
        store.save(self.project, BASE, [track(automatic=[[100, 120]])], CONTEXT)
        self.assertEqual(store.revision(self.project, BASE), payload)
        store.save(self.project, BASE, [track(ranges=[[300, 421]])], CONTEXT)
        self.assertNotIn(store.revision(self.project, BASE), (payload, None))

    def test_unreadable_slot_or_other_time_base_is_not_carried(self):
        store.save(self.project, BASE, [track()], CONTEXT)
        applied = store.revision(self.project, BASE)
        (self.project / self.state()['file']).write_text('{}', encoding='utf-8')
        store.save(self.project, BASE, [track()], CONTEXT)
        self.assertNotEqual(store.revision(self.project, BASE), applied)
        applied = store.revision(self.project, BASE)
        store.save(self.project, BASE, [track()], {**CONTEXT, 'fps': 25.0})
        self.assertNotEqual(store.revision(self.project, BASE), applied)

    def test_gui_snapshot_stays_applied_after_suggestions(self):
        try:
            from mocap.runtime_registry import activate
            activate(ROOT, 'gui')
        except RuntimeError:
            pass  # Development interpreters provide PySide6 themselves.
        try:
            from mocap.selfcontact_ui import snapshot
        except ImportError:
            self.skipTest('PySide6 is not available')
        store.save(self.project, BASE, [track()], CONTEXT)
        report = dict(contact_revision=foot_contacts.contact_revision(self.project, BASE))
        (self.project / f'{KIND}_correction.json').write_text(json.dumps(report), encoding='utf-8')
        store.save(self.project, BASE, [track(automatic=[[1, 30]]), track(target='Head', ranges=[], automatic=[[2, 8]])], CONTEXT)
        self.assertTrue(snapshot(self.project, BASE, KIND)['applied'])
        store.save(self.project, BASE, [track(ranges=[[1, 30]])], CONTEXT)
        self.assertFalse(snapshot(self.project, BASE, KIND)['applied'])


# CLI -------------------------------------------------------------------------------------------------------------

def issue(check, effector, target, first, last, score=1.0, severity='medium'):
    metrics = dict(depth_max_m=0.05, palm_speed_m_s_median=0.05) if check == 'sunk_hand' else dict(distance_3d_m_median=0.3)
    return dict(id=check[:2].upper() + '1', check=check, severity=severity, score=score, effector=effector, target=target,
                frames=[first, last], joints=['left_hand', 'torso'], metrics=metrics)


class SuggestCliTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        (self.project / f'{KIND}_sources.json').write_text(json.dumps(SOURCES), encoding='utf-8')
        self.write_report()
        target = SimpleNamespace(project=self.project, model='exmr', people_root=None)
        self.issues = [issue('missed_contact', 'LeftHand', 'Head', 100, 120, 3.0), issue('sunk_hand', 'LeftHand', 'Hips', 0, 15, 0.8)]
        for name, value in (('resolve_project', lambda *a: target), ('current_kind', lambda *a: KIND),
                            ('review_issues', lambda store, kind, session, rejected=None: [dict(i) for i in self.issues])):
            patcher = mock.patch.object(contacts, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.folder.cleanup()

    def write_report(self):
        report = dict(frames=421, fps=30.0, contact_revision=foot_contacts.contact_revision(self.project, BASE))
        (self.project / f'{KIND}_correction.json').write_text(json.dumps(report), encoding='utf-8')

    def run_cli(self, *argv):
        args = build_parser().parse_args(['contacts', *argv, '--project', str(self.project)])
        return args.handler(args, Session())[1]

    def test_suggest_keeps_the_result_current_until_accepted(self):
        summary = self.run_cli('suggest')
        self.assertFalse(summary['stale'])
        self.assertEqual([i['check'] for i in summary['issues']], ['missed_contact', 'sunk_hand'])
        tracks = {t['id']: t for t in store.read(self.project, BASE, CONTEXT)[0]}
        self.assertEqual((tracks['lh-head']['automatic'], tracks['lh-hips']['automatic'], tracks['lh-hips']['ranges']), ([(100, 121)], [(0, 16)], []))
        self.assertTrue(self.run_cli('list')['applied'])
        self.assertTrue(self.run_cli('edit', '--id', 'lh-hips', '--accept')['stale'])

    def test_suggest_beside_applied_tracks(self):
        self.run_cli('add', '--effector', 'RightHand', '--target', 'Head', '--frames', '200-220')
        self.write_report()
        self.assertFalse(self.run_cli('suggest')['stale'])
        self.issues = self.issues[:1]
        self.assertFalse(self.run_cli('suggest')['stale'])
        self.assertNotIn('lh-hips', [t['id'] for t in store.read(self.project, BASE, CONTEXT)[0]])
        self.assertTrue(self.run_cli('suggest', '--accept')['stale'])


# Hands sunk into the body ----------------------------------------------------------------------------------------

class FakeMesh:
    def __init__(self, n, values):
        self.n, self.values = n, values

    def measure(self, pairs):
        empty = (np.zeros(self.n), np.zeros(self.n, int), np.full(self.n, np.nan))
        return {pair: self.values.get((pair[0], pair[1][0]), empty) for pair in pairs}


def body(n):
    """Static standing skeleton (SMPL-X body order) with both hands resting at the hips."""
    joints = np.zeros((n, 22, 3))
    rest = {0: (0, 1.0, 0), 3: (0, 1.1, 0), 6: (0, 1.22, 0), 9: (0, 1.34, 0), 12: (0, 1.5, 0), 15: (0, 1.62, 0), 13: (-0.06, 1.42, 0),
            14: (0.06, 1.42, 0), 16: (-0.18, 1.42, 0), 17: (0.18, 1.42, 0), 18: (-0.3, 1.18, -0.05), 19: (0.3, 1.18, -0.05),
            20: (-0.2, 1.0, 0.02), 21: (0.2, 1.0, 0.02), 1: (-0.1, 0.92, 0), 2: (0.1, 0.92, 0)}
    for j, p in rest.items():
        joints[:, j] = p
    return joints, np.tile(np.eye(3), (n, 22, 1, 1))


def series(n, inside, depth=0.04, touching=()):
    d, gap = np.zeros(n), np.full(n, np.nan)
    d[inside[0]:inside[1] + 1] = depth
    gap[inside[0]:inside[1] + 1] = -depth
    for f in touching:
        gap[f] = 0.008
    return d, np.zeros(n, int), gap


class SunkHandTest(unittest.TestCase):
    N = 60

    def check(self, values, joints=None, world=None):
        base_joints, base_world = body(self.N)
        pose = SimpleNamespace(joints=base_joints if joints is None else joints, world=base_world if world is None else world,
                               mesh=FakeMesh(self.N, values))
        rejected = []
        return anatomy.check_sunk_hands(pose, np.arange(self.N), FPS, rejected), rejected

    def test_resting_hand_is_suggested_with_padding(self):
        issues, rejected = self.check({('left_hand', 'torso'): series(self.N, (10, 29), touching=(7, 8, 9, 30))})
        self.assertEqual(rejected, [])
        self.assertEqual(len(issues), 1)
        found = issues[0]
        self.assertEqual((found['check'], found['effector'], found['target'], found['severity']), ('sunk_hand', 'LeftHand', 'Hips', 'high'))
        self.assertEqual(found['frames'], [7, 30])
        self.assertEqual(found['metrics']['penetration_frames'], [[10, 29]])
        self.assertEqual(anatomy.rank(issues)[0]['id'], 'SH1')

    def test_short_shallow_and_swinging_runs_are_rejected(self):
        joints, _ = body(self.N)
        sweep = np.arange(self.N)[:, None] * np.array([-1.0 / FPS, 0, 0])
        joints[:, [19, 21]] += sweep[:, None]          # right forearm sweeps across the body at 1 m/s
        issues, rejected = self.check({('left_hand', 'torso'): series(self.N, (5, 8)), ('left_hand', 'head'): series(self.N, (20, 40), 0.015),
                                       ('right_hand', 'torso'): series(self.N, (30, 45))}, joints)
        self.assertEqual(issues, [])
        reasons = {(r['effector'], r['segment']): r['reason'] for r in rejected}
        self.assertIn('fewer than 6', reasons[('LeftHand', 'torso')])
        self.assertIn('less than 2 cm', reasons[('LeftHand', 'head')])
        self.assertIn('faster than 0.4 m/s', reasons[('RightHand', 'torso')])

    def test_fast_run_while_the_body_spins_is_a_weak_suggestion(self):
        from scipy.spatial.transform import Rotation
        _, world = body(self.N)
        world[:, 0] = Rotation.from_euler('y', np.radians(400.0) * np.arange(self.N) / FPS).as_matrix()
        issues, rejected = self.check({('right_hand', 'torso'): series(self.N, (20, 32))}, world=world)
        self.assertEqual(rejected, [])
        self.assertEqual((issues[0]['effector'], issues[0]['target'], issues[0]['severity']), ('RightHand', 'Hips', 'low'))
        self.assertIn('fast turn', issues[0]['message'])

    def test_short_lift_joins_and_head_is_the_head(self):
        joints, _ = body(self.N)
        joints[:, 20], joints[:, 18] = (-0.12, 1.62, 0.0), (-0.3, 1.5, 0.0)
        d1, inside, gap = series(self.N, (10, 20))
        d2 = series(self.N, (26, 40))[0]
        issues, _ = self.check({('left_hand', 'head'): (d1 + d2, inside, np.where(d2 > 0, -d2, gap))}, joints)
        self.assertEqual([(i['target'], i['frames']) for i in issues], [('Head', [10, 40])])
        self.assertEqual(issues[0]['metrics']['penetration_frames'], [[10, 20], [26, 40]])

    def test_targets_follow_the_palm_and_never_the_own_arm(self):
        joints, world = body(self.N)
        pose = SimpleNamespace(joints=joints, world=world)
        rows = np.arange(self.N)
        palm = anatomy.palm_points(pose, 0)
        self.assertEqual(anatomy.sunk_target(pose, 0, 'torso', palm, rows)[0], 'Hips')
        high = np.tile([0.12, 1.42, 0.02], (self.N, 1))
        self.assertEqual(anatomy.sunk_target(pose, 0, 'torso', high, rows)[0], 'RightShoulder')
        self.assertEqual(anatomy.sunk_target(pose, 1, 'torso', high, rows)[0], 'Spine2')
        self.assertEqual(anatomy.sunk_target(pose, 0, 'right_forearm', palm, rows), ('RightForeArm', 19))


if __name__ == '__main__':
    unittest.main()
