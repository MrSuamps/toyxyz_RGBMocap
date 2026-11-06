"""render --frames: N / A-B / mixed comma lists, --step inside ranges, and --sheet frame selection per track's hold."""
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

from mocap.cli import render  # noqa: E402
from mocap.cli.pipeline import CliError  # noqa: E402

# gomenDance: holds 0-3 + 96-102 and 384-455 (inclusive), stored end-exclusive.
SPLIT_HOLD = [(0, 4), (96, 103)]
LATE_HOLD = [(384, 456)]


class FrameItemsTest(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(render.frame_items('380-420', 500), [(380, 420)])
        self.assertEqual(render.frame_items('7', 500), [(7, 7)])
        self.assertEqual(render.frame_items('0,23,99', 500), [(0, 0), (23, 23), (99, 99)])
        self.assertEqual(render.frame_items('0-40, 88-108,', 500), [(0, 40), (88, 108)])
        self.assertEqual(render.frame_items('5,0-2', 500), [(5, 5), (0, 2)])
        self.assertIsNone(render.frame_items(None, 500))

    def test_bad_input_is_a_usage_error(self):
        for value in ('', ',', 'a-b', '1-2-3', '5-', '-5', '0-40,x', '10-5', '0-500', '0,500'):
            with self.subTest(value=value), self.assertRaises(CliError) as caught:
                render.frame_items(value, 500)
            self.assertEqual(caught.exception.code, render.EXIT_USAGE)

    def test_list_steps_inside_ranges(self):
        self.assertIsNone(render.frame_list('0-9', 500, 3))
        self.assertIsNone(render.frame_list(None, 500))
        self.assertEqual(render.frame_list('30,10,20,10', 500, 3), [10, 20, 30])
        self.assertEqual(render.frame_list('0-40,88-108', 500, 20), [0, 20, 40, 88, 108])
        self.assertEqual(render.frame_list('5,3,0-4', 500, 2), [0, 2, 3, 4, 5])


class RenderCommandFramesTest(unittest.TestCase):
    """cmd_render's frames up to the video read (stubbed to stop there): mixed lists are accepted and --step applies in ranges."""

    def wanted(self, frames, step=1):
        state = {'joints': np.zeros((500, 55, 3))}
        target = SimpleNamespace(project=Path('P'), model='exmr')
        args = SimpleNamespace(project='P', model=None, result='initial', sheet=False, views='camera', step=step, size=900, out='out',
                               compare=None, frames=frames, focus=None)
        session = SimpleNamespace(dry_run=False, json=False, info=mock.Mock(), say=mock.Mock())
        read = mock.Mock(side_effect=KeyboardInterrupt)
        with mock.patch.object(render, 'resolve_project', return_value=target), \
                mock.patch.object(render, 'read_json', return_value={'fps': 30}), \
                mock.patch.object(render, 'load_body', return_value=None), \
                mock.patch.object(render, 'load_layers', return_value=([render.Layer('After', 'result', state, 'initial')], None)), \
                mock.patch.object(render, 'contact_tracks', return_value=([], None)), \
                mock.patch.object(render, 'load_cameras', return_value=SimpleNamespace(video='video.mp4')), \
                mock.patch('mocap.cli.review.read_frames', read), self.assertRaises(KeyboardInterrupt):
            render.cmd_render(args, session)
        return sorted(read.call_args.args[1])

    def test_forms(self):
        self.assertEqual(self.wanted('10-16', 3), [10, 13, 16])
        self.assertEqual(self.wanted('9,3,5', 3), [3, 5, 9])
        self.assertEqual(self.wanted('0-40,88-108', 20), [0, 20, 40, 88, 108])
        self.assertEqual(self.wanted('0-4,7'), [0, 1, 2, 3, 4, 7])

    def test_bad_input_exits_with_usage(self):
        with self.assertRaises(CliError) as caught:
            self.wanted('0-40,88-')
        self.assertEqual(caught.exception.code, render.EXIT_USAGE)


class SheetFramesTest(unittest.TestCase):
    def test_default_spreads_over_the_hold(self):
        self.assertEqual(render.sheet_selection(None, SPLIT_HOLD, 500), ([0, 3, 99, 102], [], []))
        self.assertIsNone(render.sheet_selection(None, [], 500))
        self.assertEqual(render.sheet_selection(None, [], 500, [(10, 12)]), ([10, 11], [], []))

    def test_listed_frames_partly_in_the_hold_report_the_dropped_ones(self):
        self.assertEqual(render.sheet_selection('0,23,99,383', SPLIT_HOLD, 500), ([0, 99], [], [23, 383]))

    def test_no_listed_frame_in_the_hold_marks_them_outside(self):
        self.assertEqual(render.sheet_selection('0,23,99,383', LATE_HOLD, 500), ([0, 23, 99, 383], [0, 23, 99, 383], []))

    def test_range_takes_hold_frames_inside_it(self):
        self.assertEqual(render.sheet_selection('90-110', SPLIT_HOLD, 500), ([96, 98, 100, 102], [], []))
        self.assertEqual(render.sheet_selection('200-203', SPLIT_HOLD, 500), ([200, 201, 202, 203], [200, 201, 202, 203], []))

    def test_mixed_list(self):
        frames, outside, dropped = render.sheet_selection('0-40,88-108', SPLIT_HOLD, 500)
        self.assertEqual((frames, outside, dropped), ([0, 1, 2, 3, 96, 98, 100, 102], [], []))
        frames, outside, dropped = render.sheet_selection('0-40,200-260,383', SPLIT_HOLD, 500)
        self.assertEqual((frames, outside, dropped), ([0, 1, 2, 3], [], [200, 220, 240, 260, 383]))


class SheetCommandTest(unittest.TestCase):
    """cmd_sheet wiring with the body, cameras, video and drawing stubbed: sheet.json and the console note per track."""

    def run_sheet(self, frames):
        count = 500
        state = {'joints': np.zeros((count, 55, 3))}
        tracks = [dict(id='split', effector='RightHand', target='Spine', ranges=SPLIT_HOLD),
                  dict(id='late', effector='LeftHand', target='Spine', ranges=LATE_HOLD)]
        body = SimpleNamespace(pose=lambda params: state)
        camera = SimpleNamespace(keypoints=np.zeros(1), video='video.mp4', describe=lambda: 'stub')
        drawn = {}
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / 'project.json').write_text('{"fps": 30}', encoding='utf-8')
            out = project / 'sheet'
            args = SimpleNamespace(frames=frames, add=[], compare=None, out=str(out))
            session = SimpleNamespace(dry_run=False, json=False, info=mock.Mock(), say=mock.Mock())
            with mock.patch.object(render, 'load_body', return_value=body), \
                    mock.patch.object(render, 'contact_tracks', return_value=(tracks, None)), \
                    mock.patch.object(render, 'load_cameras', return_value=None), \
                    mock.patch.object(render, 'layer_cameras', side_effect=lambda cams, project, layers, kind: [camera] * len(layers)), \
                    mock.patch.object(render, 'body_name', return_value='SMPL-X'), \
                    mock.patch.object(render, 'contact_sheet_image',
                                      side_effect=lambda body, layers, track, frames, *rest: drawn.setdefault(track['id'], frames) and 'image'), \
                    mock.patch('mocap.cli.compare.reference', return_value=(project / 'ref.pt', 'key', 'original', 'initial')), \
                    mock.patch('mocap.cli.smplx_body.load_params', return_value={}), \
                    mock.patch('mocap.cli.review.read_frames', side_effect=lambda video, wanted: {f: None for f in wanted}), \
                    mock.patch('cv2.imwrite'):
                render.cmd_sheet(args, session, project, 'exmr', 'initial', 0.0)
            report = json.loads((out / 'sheet.json').read_text(encoding='utf-8'))
        lines = [call.args[0] for call in session.info.call_args_list]
        return report, lines, drawn

    def test_gomen_dance_lists_dropped_frames(self):
        report, lines, drawn = self.run_sheet('0,23,99,383')
        self.assertEqual(drawn, {'split': [0, 99], 'late': [0, 23, 99, 383]})
        self.assertEqual(report['outside_hold'], {'late': [0, 23, 99, 383]})
        self.assertEqual(report['dropped_outside_hold'], {'split': [23, 383]})
        notes = [line for line in lines if 'left out' in line]
        self.assertEqual(len(notes), 1)
        self.assertIn('split', notes[0])
        self.assertIn('23, 383', notes[0])

    def test_unchanged_selection_has_no_dropped_frames(self):
        report, lines, drawn = self.run_sheet('380-460')
        self.assertEqual(drawn['late'], [384, 408, 431, 455])
        self.assertEqual(report['dropped_outside_hold'], {})
        self.assertFalse([line for line in lines if 'left out' in line])


if __name__ == '__main__':
    unittest.main()
