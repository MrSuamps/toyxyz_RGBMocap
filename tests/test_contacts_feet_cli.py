"""`contacts feet` CLI, the before-self-contact freshness check and conditional self-contact code identity."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cli  # noqa: E402
from mocap import foot_contacts, self_contact_store as store  # noqa: E402
from mocap.cli import compare, contacts, render  # noqa: E402
from mocap.cli.main import build_parser  # noqa: E402
from mocap.cli.pipeline import CliError  # noqa: E402
from mocap.result_artifacts import file_stat  # noqa: E402

BASE, KIND = 'initial', 'corrected_initial'
SOURCES = {'hmr4d_results.pt': [1, 2]}
CONTEXT = {'frames': 421, 'fps': 30.0, 'sources': SOURCES}
MODULES = ('self_contact.py', 'self_contact_store.py', 'foot_articulation.py')


class Session:
    def __init__(self, dry_run=False):
        self.dry_run, self.lines = dry_run, []

    def say(self, text):
        self.lines.append(text)

    info = say


class Project(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        (self.project / f'{KIND}_sources.json').write_text(json.dumps(SOURCES), encoding='utf-8')
        self.write_report()
        target = SimpleNamespace(project=self.project, model='exmr', people_root=None)
        for name, value in (('resolve_project', lambda *a: target), ('current_kind', lambda *a: KIND)):
            patcher = mock.patch.object(contacts, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.folder.cleanup()

    def write_report(self, **extra):
        report = dict(frames=421, fps=30.0, contact_revision=foot_contacts.contact_revision(self.project, BASE), **extra)
        (self.project / f'{KIND}_correction.json').write_text(json.dumps(report), encoding='utf-8')

    def run_cli(self, *argv, dry_run=False):
        args = build_parser().parse_args(['contacts', *argv, '--project', str(self.project)])
        session = Session(dry_run)
        code, summary = args.handler(args, session)
        return code, summary, session.lines


class FeetCliTest(Project):
    def test_add_edit_remove_round_trip(self):
        code, summary, lines = self.run_cli('feet', 'add', '--foot', 'leftfoot', '--mode', 'pointe', '--frames', '100-150', '--profile', 'dancer')
        self.assertEqual(code, 0)
        self.assertEqual(summary['id'], 'lf-pointe')
        self.assertTrue(summary['stale'])
        entries, _, _ = store.read_articulation(self.project, BASE, CONTEXT)
        self.assertEqual((entries[0]['foot'], entries[0]['ranges'], entries[0]['profile']), ('LeftFoot', [(100, 151)], 'dancer'))
        self.run_cli('feet', 'edit', '--id', 'lf-pointe', '--add-frames', '170-180', '--remove-frames', '120-125', '--strength', '0.5',
                     '--mode', 'flexed')
        entries, _, _ = store.read_articulation(self.project, BASE, CONTEXT)
        self.assertEqual(entries[0]['ranges'], [(100, 120), (126, 151), (170, 181)])
        self.assertEqual((entries[0]['mode'], entries[0]['strength']), ('flexed', 0.5))
        _, summary, lines = self.run_cli('list')
        self.assertEqual([e['id'] for e in summary['articulation']], ['lf-pointe'])
        self.assertIn('lf-pointe: LeftFoot flexed', lines)
        self.run_cli('feet', 'remove', '--id', 'lf-pointe')
        self.assertFalse(store.state_path(self.project, BASE).exists())

    def test_dry_run_writes_nothing_and_errors_are_usage(self):
        self.run_cli('feet', 'add', '--foot', 'RightFoot', '--mode', 'flat', '--frames', '1-2', dry_run=True)
        self.assertFalse(store.state_path(self.project, BASE).exists())
        for argv in (('--frames', '0-421'), ('--frames', '1-2', '--strength', '2')):
            with self.assertRaises(CliError) as caught:
                self.run_cli('feet', 'add', '--foot', 'RightFoot', '--mode', 'flat', *argv)
            self.assertEqual(caught.exception.code, 2)
        self.run_cli('feet', 'add', '--foot', 'RightFoot', '--mode', 'flat', '--frames', '10-20')
        with self.assertRaises(CliError):
            self.run_cli('feet', 'add', '--foot', 'RightFoot', '--mode', 'pointe', '--frames', '15-30')

    def test_tracks_and_feet_keep_each_other(self):
        self.run_cli('add', '--effector', 'RightHand', '--target', 'RightFoot', '--frames', '345-420')
        self.run_cli('feet', 'add', '--foot', 'LeftFoot', '--mode', 'pointe', '--frames', '100-150')
        self.run_cli('feet', 'remove', '--all')
        self.assertEqual(len(store.read(self.project, BASE, CONTEXT)[0]), 1)
        self.run_cli('feet', 'add', '--foot', 'LeftFoot', '--mode', 'pointe', '--frames', '100-150')
        _, _, lines = self.run_cli('remove', '--all')
        self.assertTrue(any(line.startswith('Kept 1 foot articulation entry') for line in lines))
        self.assertEqual(len(store.read_articulation(self.project, BASE, CONTEXT)[0]), 1)

    def test_other_time_base_blocks_edits_and_remove_all_starts_over(self):
        self.run_cli('feet', 'add', '--foot', 'LeftFoot', '--mode', 'pointe', '--frames', '100-150')
        state = json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))
        store.state_path(self.project, BASE).write_text(json.dumps(dict(state, frames=300)), encoding='utf-8')
        with self.assertRaises(CliError):
            self.run_cli('feet', 'add', '--foot', 'RightFoot', '--mode', 'flat', '--frames', '1-2')
        _, _, lines = self.run_cli('feet', 'list')
        self.assertTrue(lines[0].startswith('warning: Saved foot articulation'))
        self.run_cli('remove', '--all')
        self.assertFalse(any(self.project.glob('selfcontact_*')))

    def test_lock_rules(self):
        self.assertFalse(cli.lock_needed(['contacts', 'feet', 'list', '--project', 'P']))
        self.assertTrue(cli.lock_needed(['contacts', 'feet', 'add', '--project', 'P']))
        self.assertTrue(cli.lock_needed(['contacts', 'feet', 'remove', '--all', '--project', 'P']))
        self.assertFalse(cli.lock_needed(['contacts', 'feet', 'edit', '--dry-run']))
        self.assertFalse(cli.lock_needed(['contacts', 'list', '--project', 'P']))


class BeforeCurrentTest(Project):
    def setUp(self):
        super().setUp()
        self.before = store.before_path(self.project, KIND)
        self.before.write_bytes(b'pose')
        self.write_report(self_contact=dict(before_stat=file_stat(self.before)))
        patcher = mock.patch('mocap.motion_tools.result_is_current', lambda project, kind, **_: self.current)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.current = True

    def test_stat_and_result_must_match(self):
        self.assertTrue(store.before_current(self.project, KIND))
        self.current = False
        self.assertFalse(store.before_current(self.project, KIND))
        self.current = True
        self.before.write_bytes(b'other pose')
        self.assertFalse(store.before_current(self.project, KIND))
        self.write_report()
        self.assertFalse(store.before_current(self.project, KIND))
        self.before.unlink()
        self.assertFalse(store.before_current(self.project, KIND))

    def test_compare_and_render_do_not_trust_a_stale_file(self):
        self.assertEqual(compare.reference(self.project, KIND, 'before-selfcontact')[0], self.before)
        self.before.write_bytes(b'leftover from an older correct')
        with self.assertRaises(CliError) as caught:
            compare.reference(self.project, KIND, 'before-selfcontact')
        self.assertIn('outdated', caught.exception.message)
        (self.project / f'{KIND}.pt').write_bytes(b'')
        body = SimpleNamespace(pose=lambda params: {})
        with mock.patch('mocap.cli.smplx_body.load_params', lambda path, key: {}):
            layers, note = render.load_layers(self.project, KIND, 'auto', body)
        self.assertEqual((layers[0].kind, layers[0].label), (BASE, 'initial (reference)'))
        self.assertIn('Comparing against the original instead', note)


class SheetFramesTest(unittest.TestCase):
    def test_ranges_then_suggestions_then_none(self):
        self.assertEqual(render.sheet_frames(None, [(10, 41)], 100), [10, 20, 30, 40])
        self.assertEqual(render.sheet_frames(None, [], 100, [(20, 51)]), [20, 30, 40, 50])
        self.assertIsNone(render.sheet_frames(None, [], 100, []))
        self.assertEqual(render.sheet_frames('5,7', [], 100), [5, 7])
        self.assertEqual(render.sheet_frames('0-30', [], 100), [0, 10, 20, 30])


class CodeIdentityTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.project = Path(self.folder.name)
        (self.project / 'project.json').write_text(json.dumps({'fps': 30}), encoding='utf-8')

    def tearDown(self):
        self.folder.cleanup()

    def worker_fingerprint(self, edited):
        from mocap import models, worker

        def code_hash(path):
            return ('edited-' if edited and path.name in MODULES else '') + path.name
        with mock.patch.object(worker, 'code_hash', code_hash), mock.patch.object(worker, 'resolve_motion_result', lambda *a: KIND), \
                mock.patch.dict(models.STAGE_REQUIREMENTS, {'correct': ()}):
            return worker.fingerprint('correct', self.project, KIND)

    def flow_identity(self, edited):
        from mocap import flow_correction

        def sha256(path):
            return ('edited-' if edited and path.name in MODULES else '') + path.name
        with mock.patch.object(flow_correction, 'sha256', sha256):
            return flow_correction.code_identity(self.project)

    def test_modules_count_only_with_a_self_contact_file(self):
        plain = self.worker_fingerprint(False)
        self.assertEqual(self.worker_fingerprint(True), plain)
        self.assertEqual(self.flow_identity(True), self.flow_identity(False))
        self.assertNotIn('self_contact.py', self.flow_identity(False))
        store.save(self.project, BASE, [dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]])], CONTEXT)
        tracked = self.worker_fingerprint(False)
        self.assertNotEqual(tracked, plain)
        self.assertNotEqual(self.worker_fingerprint(True), tracked)
        self.assertNotEqual(self.flow_identity(True), self.flow_identity(False))
        self.assertIn('self_contact_store.py', self.flow_identity(False))


if __name__ == '__main__':
    unittest.main()
