"""Self-contact store: file format, validation and the revision folded into foot-lock stale checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import foot_contacts, self_contact_store as store  # noqa: E402

BASE = 'initial'
CONTEXT = {'frames': 421, 'fps': 30.0, 'sources': {'hmr4d_results.pt': [1, 2]}}


def track(**values):
    return dict(dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]]), **values)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def footlock(self, ranges):
        return foot_contacts.save_edit(self.project, BASE, ranges, CONTEXT)['revision']

    def test_absent_file_keeps_old_revisions(self):
        self.assertIsNone(store.revision(self.project, BASE))
        self.assertIsNone(foot_contacts.contact_revision(self.project, BASE))
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT), (None, None, ''))
        automatic = self.footlock(None)
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), automatic)
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT), (None, automatic, ''))
        manual = self.footlock([[(0, 10)], [(5, 20)]])
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), manual)
        self.assertEqual(foot_contacts.footlock_revision(self.project, BASE), manual)
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT), ([[(0, 10)], [(5, 20)]], manual, ''))
        self.assertEqual(store.read(self.project, BASE, CONTEXT), ([], None, ''))

    def test_roundtrip_and_slot_alternation(self):
        tracks = [track(id='grab', ranges=[[400, 421], [345, 360], [355, 380]], automatic=[[366, 420]], distance=0.06,
                        offset=[0.01, 0, -0.02], reciprocal='always', grip='auto', source='review:x'),
                  track(effector='LeftHand', target='Head', enabled=False, ranges=[])]
        first = store.save(self.project, BASE, tracks, CONTEXT)
        state = json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))
        self.assertEqual(state['file'], 'selfcontact_initial_edit1.json')
        self.assertEqual({k: state[k] for k in ('frames', 'fps', 'sources', 'revision')}, {**CONTEXT, 'revision': first})
        read, revision, warning = store.read(self.project, BASE, CONTEXT)
        self.assertEqual((revision, warning), (first, ''))
        self.assertEqual(read[0]['ranges'], [(345, 380), (400, 421)])
        self.assertEqual(read[0]['automatic'], [(366, 420)])
        self.assertEqual((read[0]['distance'], read[0]['offset'], read[0]['reciprocal'], read[0]['grip']),
                         (0.06, [0.01, 0.0, -0.02], 'always', 'auto'))
        self.assertEqual(read[1]['id'], 'lh-head')
        self.assertEqual((read[1]['enabled'], read[1]['reciprocal'], read[1]['grip'], read[1]['offset'], read[1]['source']), (False, 'auto', 'off', 'auto', 'user'))
        second = store.save(self.project, BASE, read, CONTEXT)
        state = json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))
        self.assertEqual(state['file'], 'selfcontact_initial_edit0.json')
        self.assertNotEqual(first, second)
        self.assertEqual(store.read(self.project, BASE, CONTEXT)[0], read)
        store.save(self.project, BASE, read, CONTEXT)
        self.assertEqual(json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))['file'], 'selfcontact_initial_edit1.json')

    def test_revision_combines_and_changes_on_save(self):
        foot = self.footlock(None)
        sc = store.save(self.project, BASE, [track()], CONTEXT)
        combined = f'{foot}|sc:{sc}'
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), combined)
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT)[1], combined)
        self.assertEqual(foot_contacts.footlock_revision(self.project, BASE), foot)
        sc2 = store.save(self.project, BASE, [track(ranges=[[300, 421]])], CONTEXT)
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), f'{foot}|sc:{sc2}')
        foot2 = self.footlock([[(0, 10)], [(5, 20)]])
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), f'{foot2}|sc:{sc2}')
        self.assertEqual(foot_contacts.with_self_contacts(self.project, BASE, foot2), f'{foot2}|sc:{sc2}')
        # A store that applies nothing adds nothing, as if there were no file.
        empty = store.save(self.project, BASE, [], CONTEXT)
        self.assertEqual(store.read(self.project, BASE, CONTEXT), ([], empty, ''))
        self.assertIsNone(store.revision(self.project, BASE))
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), foot2)

    def test_without_footlock_state(self):
        sc = store.save(self.project, BASE, [track()], CONTEXT)
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), f'|sc:{sc}')
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT), (None, f'|sc:{sc}', ''))
        self.assertIsNone(foot_contacts.footlock_revision(self.project, BASE))

    def test_clear_restores_old_behaviour(self):
        foot = self.footlock(None)
        store.save(self.project, BASE, [track()], CONTEXT)
        store.save(self.project, BASE, [track()], CONTEXT)
        self.assertTrue(store.clear(self.project, BASE))
        self.assertEqual(sorted(p.name for p in self.project.iterdir() if p.name.startswith('selfcontact_')), [])
        self.assertIsNone(store.revision(self.project, BASE))
        self.assertEqual(foot_contacts.contact_revision(self.project, BASE), foot)
        self.assertEqual(foot_contacts.read_edit(self.project, BASE, CONTEXT), (None, foot, ''))
        self.assertFalse(store.clear(self.project, BASE))

    def test_frames_or_fps_mismatch_warns(self):
        sc = store.save(self.project, BASE, [track()], CONTEXT)
        for change in ({'frames': 400}, {'fps': 24.0}):
            tracks, revision, warning = store.read(self.project, BASE, {**CONTEXT, **change})
            self.assertEqual((tracks, revision), ([], sc))
            self.assertIn('different length or frame rate', warning)
        tracks, revision, warning = store.read(self.project, BASE, {**CONTEXT, 'sources': {'other': [3, 4]}})
        self.assertEqual((len(tracks), revision, warning), (1, sc, ''))

    def test_torn_payload_is_rejected(self):
        store.save(self.project, BASE, [track()], CONTEXT)
        state = json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))
        (self.project / state['file']).write_text(json.dumps({'version': 1, 'revision': 'other', 'tracks': []}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Incomplete self-contact save'):
            store.read(self.project, BASE, CONTEXT)

    def test_unknown_base(self):
        with self.assertRaises(ValueError):
            store.state_path(self.project, 'nonsense')


class NormalizeTest(unittest.TestCase):
    def bad(self, message, tracks, frames=421):
        with self.assertRaisesRegex(ValueError, message):
            store.normalize_tracks(tracks, frames)

    def test_validation_errors(self):
        self.bad('must be a list', {'effector': 'RightHand'})
        self.bad('must be an object', ['RightHand'])
        self.bad('unknown field', [track(colour='red')])
        self.bad('effector must be one of', [track(effector='RightFoot')])
        self.bad('target must be a body bone', [track(target='RightPinky1')])
        self.bad('on the RightHand arm itself', [track(target='RightForeArm')])
        self.bad('on the RightHand arm itself', [track(target='RightHand')])
        self.bad('on the LeftHand arm itself', [track(effector='LeftHand', target='LeftShoulder')])
        self.bad('offset must be', [track(offset=[0, 1])])
        self.bad('offset must be a finite number', [track(offset=[0, 1, float('nan')])])
        self.bad('distance must be >= 0', [track(distance=-0.1)])
        self.bad('distance must be a finite number', [track(distance='near')])
        self.bad('reciprocal must be one of', [track(reciprocal='yes')])
        self.bad('grip must be one of', [track(grip='always')])
        self.bad('enabled must be true or false', [track(enabled=1)])
        self.bad('outside the motion frames', [track(ranges=[[400, 422]])])
        self.bad('outside the motion frames', [track(ranges=[[10, 10]])])
        self.bad('outside the motion frames', [track(ranges=[[-1, 10]])])
        self.bad('must be integers', [track(ranges=[[1.0, 10]])])
        self.bad('invalid interval', [track(ranges=[[1, 2, 3]])])
        self.bad('automatic: interval', [track(automatic=[[0, 500]])])
        self.bad('id must be', [track(id='has space')])
        self.bad('Duplicate self-contact track id', [track(id='a'), track(id='a', effector='LeftHand')])
        self.bad('positive integer', [], 0)

    def test_defaults_ids_and_merging(self):
        result = store.normalize_tracks([track(ranges=[[10, 20], [20, 30], [5, 8]]), track(target='LeftToeBase', ranges=[]),
                                         track(ranges=[]), track(effector='LeftHand', target='RightHand', ranges=[[0, 5]])], 421)
        self.assertEqual([t['id'] for t in result], ['rh-rf', 'rh-ltb', 'rh-rf-2', 'lh-rh'])
        self.assertEqual(result[0]['ranges'], [(5, 8), (10, 30)])
        self.assertEqual(sorted(result[0]), sorted(store.KEYS))
        self.assertEqual((result[0]['offset'], result[0]['distance'], result[0]['reciprocal'], result[0]['grip'], result[0]['enabled']),
                         ('auto', 'auto', 'auto', 'off', True))

    def test_overlap_rejection(self):
        self.bad("both hold RightHand on frames 360-369",
                 [track(ranges=[[345, 370]]), track(target='RightLeg', ranges=[[360, 380]])])
        # Touching intervals, a disabled track, another hand or suggestions only do not conflict.
        store.normalize_tracks([track(ranges=[[345, 370]]), track(target='RightLeg', ranges=[[370, 380]])], 421)
        store.normalize_tracks([track(ranges=[[345, 370]]), track(target='RightLeg', ranges=[[360, 380]], enabled=False)], 421)
        store.normalize_tracks([track(ranges=[[345, 370]]), track(effector='LeftHand', ranges=[[360, 380]])], 421)
        store.normalize_tracks([track(ranges=[[345, 370]]), track(target='RightLeg', ranges=[], automatic=[[360, 380]])], 421)


class CommandLineTest(unittest.TestCase):
    def test_frames_parsing(self):
        from mocap.cli.contacts import inclusive, parse_frames, subtract
        from mocap.cli.pipeline import CliError
        self.assertEqual(parse_frames('345-420,500-510', 421 + 100), [(345, 421), (500, 511)])
        self.assertEqual(parse_frames('7', 10), [(7, 8)])
        for value in ('5-3', '0-10', 'a-b', ''):
            with self.assertRaises(CliError):
                parse_frames(value, 10)
        self.assertEqual(inclusive([(345, 421), (500, 501)]), '345-420,500')
        self.assertEqual(subtract([(0, 10), (20, 30)], [(5, 25)]), [(0, 5), (25, 30)])

    def test_lock_needed(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('rgbmocap_cli_entry', ROOT / 'cli.py')
        import os
        saved, environment, bytecode = sys.path[:], dict(os.environ), sys.dont_write_bytecode
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = saved
            os.environ.clear()
            os.environ.update(environment)
            sys.dont_write_bytecode = bytecode
        lock = module.lock_needed
        self.assertFalse(lock(['contacts', 'list', '--project', 'P']))
        self.assertFalse(lock(['--json', 'contacts', 'suggest', '--project', 'P']))
        self.assertTrue(lock(['contacts', 'suggest', '--project', 'P', '--accept']))
        for action in ('add', 'edit', 'remove'):
            self.assertTrue(lock(['contacts', action, '--project', 'P']))
        self.assertFalse(lock(['review', '--project', 'P']))
        self.assertTrue(lock(['correct', '--project', 'P']))


if __name__ == '__main__':
    unittest.main()
