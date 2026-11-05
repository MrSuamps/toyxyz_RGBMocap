"""`contacts place` and `contacts smooth` CLI: limb placement and pose smoothing rows in the self-contact store."""
import io
import json
import sys
from contextlib import redirect_stderr
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cli  # noqa: E402
from mocap import self_contact_store as store  # noqa: E402
from mocap.cli.pipeline import CliError  # noqa: E402
from test_contacts_feet_cli import BASE, CONTEXT, Project  # noqa: E402


class PlaceCliTest(Project):
    def test_add_edit_remove_round_trip(self):
        code, summary, _ = self.run_cli('place', 'add', '--chain', 'rightleg', '--frames', '380-420', '--strength', '0.8')
        self.assertEqual((code, summary['id'], summary['action']), (0, 'rl-place', 'place add'))
        self.assertTrue(summary['stale'])
        rows, _, _ = store.read_placement(self.project, BASE, CONTEXT)
        self.assertEqual((rows[0]['chain'], rows[0]['ranges'], rows[0]['strength']), ('RightLeg', [(380, 421)], .8))
        self.run_cli('place', 'edit', '--id', 'rl-place', '--add-frames', '100-110', '--remove-frames', '400-405', '--chain', 'LeftArm',
                     '--disable')
        rows, _, _ = store.read_placement(self.project, BASE, CONTEXT)
        self.assertEqual((rows[0]['chain'], rows[0]['enabled']), ('LeftArm', False))
        self.assertEqual(rows[0]['ranges'], [(100, 111), (380, 400), (406, 421)])
        _, summary, lines = self.run_cli('place', 'list')
        self.assertEqual(summary['placement'][0]['ranges_inclusive'], '100-110,380-399,406-420')
        self.assertIn('rl-place: LeftArm  (disabled)', lines)
        self.run_cli('place', 'remove', '--id', 'rl-place')
        self.assertFalse(store.state_path(self.project, BASE).exists())

    def test_overlap_strength_and_dry_run(self):
        self.run_cli('place', 'add', '--chain', 'LeftLeg', '--frames', '1-2', dry_run=True)
        self.assertFalse(store.state_path(self.project, BASE).exists())
        self.run_cli('place', 'add', '--chain', 'LeftLeg', '--frames', '10-20')
        for argv in (('--chain', 'LeftLeg', '--frames', '15-30'), ('--chain', 'RightArm', '--frames', '1-2', '--strength', '1.5')):
            with self.assertRaises(CliError) as caught:
                self.run_cli('place', 'add', *argv)
            self.assertEqual(caught.exception.code, 2)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_cli('place', 'add', '--chain', 'LeftFoot', '--frames', '1-2')
        with self.assertRaises(CliError):
            self.run_cli('place', 'edit', '--id', 'nope', '--disable')


class SmoothCliTest(Project):
    def test_groups_joints_and_window(self):
        _, summary, _ = self.run_cli('smooth', 'add', '--group', 'arms', '--window', '7', '--frames', '200-240')
        self.assertEqual(summary['id'], 'smooth-arms')
        _, summary, _ = self.run_cli('smooth', 'add', '--frames', '10-20')
        self.assertEqual(summary['id'], 'smooth-all-written')
        _, summary, _ = self.run_cli('smooth', 'add', '--joints', 'spine1,neck', '--frames', '200-240')
        self.assertEqual(summary['id'], 'smooth-spine1')
        rows, _, _ = store.read_smoothing(self.project, BASE, CONTEXT)
        self.assertEqual([(r['joints'], r['window_frames']) for r in rows], [('arms', 7), ('all-written', 5), (['Spine1', 'Neck'], 5)])
        self.run_cli('smooth', 'edit', '--id', 'smooth-arms', '--group', 'spine', '--window', '9', '--frames', '300-310')
        rows, _, _ = store.read_smoothing(self.project, BASE, CONTEXT)
        self.assertEqual((rows[0]['joints'], rows[0]['window_frames'], rows[0]['ranges']), ('spine', 9, [(300, 311)]))
        with self.assertRaises(CliError):  # Overlaps smooth-spine1 on 230-235: Spine1 is in the spine group.
            self.run_cli('smooth', 'add', '--group', 'spine', '--frames', '230-235')
        for argv in (('--window', '2', '--frames', '1-2'), ('--joints', 'Hips', '--frames', '1-2')):
            with self.assertRaises(CliError) as caught:
                self.run_cli('smooth', 'add', *argv)
            self.assertEqual(caught.exception.code, 2)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_cli('smooth', 'add', '--joints', 'LeftArm', '--group', 'arms', '--frames', '1-2')
        self.run_cli('smooth', 'remove', '--all')
        self.assertEqual(store.read_smoothing(self.project, BASE, CONTEXT)[0], [])


class SectionsTest(Project):
    def test_list_shows_every_section_and_remove_all_keeps_rows(self):
        self.run_cli('add', '--effector', 'RightHand', '--target', 'RightFoot', '--frames', '345-420')
        self.run_cli('feet', 'add', '--foot', 'LeftFoot', '--mode', 'pointe', '--frames', '100-150')
        self.run_cli('place', 'add', '--chain', 'RightLeg', '--frames', '380-420')
        self.run_cli('smooth', 'add', '--group', 'arms', '--frames', '50-60')
        self.write_report(placement=dict(entries=[dict(id='rl-place', chain='RightLeg', start=380, end=421,
                                                      error_before_torso=dict(limb=dict(median=.31)),
                                                      error_after_torso=dict(limb=dict(median=.08)))]),
                          smoothing=dict(entries=[dict(id='smooth-arms', start=50, end=61, skipped='no joints')]))
        _, summary, lines = self.run_cli('list')
        self.assertEqual([len(summary[k]) for k in ('tracks', 'articulation', 'placement', 'smoothing')], [1, 1, 1, 1])
        self.assertIn('1 limb placement row on corrected_initial (frames inclusive):', lines)
        self.assertIn('    last correction: 380-420 2D error 0.31 -> 0.08 torso (median)', lines)
        self.assertIn('smooth-arms: arms', lines)
        self.assertIn('    last correction: 50-60 skipped: no joints', lines)
        _, _, lines = self.run_cli('remove', '--all')
        self.assertTrue(any(line.startswith('Kept 1 limb placement row;') for line in lines))
        self.assertTrue(any(line.startswith('Kept 1 pose smoothing row;') for line in lines))
        self.assertEqual(store.read(self.project, BASE, CONTEXT)[0], [])
        self.run_cli('place', 'remove', '--all')
        self.run_cli('smooth', 'remove', '--all')
        self.assertEqual(len(store.read_articulation(self.project, BASE, CONTEXT)[0]), 1)
        self.run_cli('feet', 'remove', '--all')
        self.assertFalse(any(self.project.glob('selfcontact_*')))

    def test_other_time_base_blocks_edits(self):
        self.run_cli('place', 'add', '--chain', 'RightLeg', '--frames', '380-420')
        state = json.loads(store.state_path(self.project, BASE).read_text(encoding='utf-8'))
        store.state_path(self.project, BASE).write_text(json.dumps(dict(state, frames=300)), encoding='utf-8')
        with self.assertRaises(CliError):
            self.run_cli('smooth', 'add', '--group', 'arms', '--frames', '1-2')
        _, _, lines = self.run_cli('place', 'list')
        self.assertTrue(lines[0].startswith('warning: Saved limb placement'))

    def test_lock_rules(self):
        for section in ('place', 'smooth'):
            self.assertFalse(cli.lock_needed(['contacts', section, 'list', '--project', 'P']))
            self.assertTrue(cli.lock_needed(['contacts', section, 'add', '--project', 'P']))
            self.assertTrue(cli.lock_needed(['contacts', section, 'remove', '--all', '--project', 'P']))
            self.assertFalse(cli.lock_needed(['contacts', section, 'edit', '--dry-run']))


if __name__ == '__main__':
    unittest.main()
