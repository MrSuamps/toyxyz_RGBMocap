"""Pose (foot articulation) timeline rows, their store round trip and the overlay/label polish, under offscreen Qt.

Optional: RGBMOCAP_POSE_PROJECT=<copied workspace with an applied correction> runs the window test on real data (it saves
pose edits into that project, so use a copy); RGBMOCAP_SHOTS=<folder> keeps the screenshots.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from mocap.runtime_registry import activate
    activate(ROOT, 'gui')
except RuntimeError:
    pass  # Development interpreters provide PySide6 themselves.

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QDialog  # noqa: E402

from mocap.pose_rows import SECTIONS, PoseRows, read_sections  # noqa: E402
from mocap.selfcontact_rows import FIRST_ROW  # noqa: E402

APP = QApplication.instance() or QApplication([])
FRAMES = 421
CONTEXT = dict(frames=FRAMES, fps=30.)
SHOTS = Path(os.environ['RGBMOCAP_SHOTS']) if os.environ.get('RGBMOCAP_SHOTS') else None
FEET = SECTIONS[0]


def entry(foot='LeftFoot', mode='pointe', ranges=((0, 100),), **values):
    return dict(dict(foot=foot, mode=mode, ranges=[list(p) for p in ranges]), **values)


def model(entries=()):
    from mocap.self_contact_store import normalize_articulation
    rows = PoseRows(FEET)
    rows.load(normalize_articulation(list(entries), FRAMES), FRAMES, ('project', 'initial'))
    return rows


class PoseModelTest(unittest.TestCase):
    def test_rows_intervals_and_labels(self):
        rows = model([entry(ranges=[(0, 100), (300, 421)], profile='dancer'), entry(mode='flat', ranges=[(120, 200)]),
                      entry(foot='RightFoot', mode='match', ranges=[(10, 20)], strength=.5)])
        self.assertEqual((rows.row_count(), rows.row_label(0), rows.row_label(1)), (2, 'L pose', 'R pose'))
        self.assertEqual(rows.ranges(0), [(0, 100), (120, 200), (300, 421)])
        self.assertEqual(rows.describe(rows.interval_settings(0, 0)), 'Pointe dancer')
        self.assertEqual(rows.describe(rows.interval_settings(1, 0)), 'Match 2D 50%')
        self.assertEqual(rows.describe(rows.draw_settings(0)), 'Match 2D')

    def test_insert_uses_draw_settings_and_settings_split_entries(self):
        from mocap.self_contact_store import normalize_articulation
        rows = model([entry(ranges=[(0, 100), (300, 421)])])
        rows.set_draw_settings(0, mode='flexed')
        rows.set_interval(0, 1, None, (150, 160))
        self.assertEqual([(e['id'], e['mode'], e['ranges']) for e in rows.entries],
                         [('lf-pointe', 'pointe', [(0, 100), (300, 421)]), ('lf-flexed', 'flexed', [(150, 160)])])
        rows.set_settings(0, 2, mode='match', strength=.75)
        self.assertEqual(rows.interval_settings(0, 2), dict(mode='match', strength=.75, profile='default', enabled=True))
        self.assertEqual(rows.entries[0]['ranges'], [(0, 100)])
        self.assertEqual(rows.draw_settings(0)['mode'], 'match')
        rows.set_interval(0, 1, (150, 160), None)
        self.assertEqual([e['id'] for e in rows.entries], ['lf-pointe', 'lf-match'])
        self.assertEqual(normalize_articulation(rows.payload(), FRAMES), rows.entries)
        self.assertTrue(rows.undo())
        self.assertTrue(rows.undo())
        self.assertEqual(rows.ranges(0), [(0, 100), (150, 160), (300, 421)])
        self.assertTrue(rows.redo())
        self.assertEqual(rows.interval_settings(0, 2)['mode'], 'match')

    def test_overlap_and_invalid_settings_are_refused(self):
        rows = model([entry(ranges=[(0, 100)]), entry(mode='flat', ranges=[(200, 250)], enabled=False)])
        with self.assertRaises(ValueError):
            rows.set_interval(0, 0, (0, 100), (0, 210))  # Overlaps the disabled flat interval: still one pose per frame.
        with self.assertRaises(ValueError):
            rows.set_settings(0, 0, mode='sideways')
        with self.assertRaises(ValueError):
            rows.set_settings(0, 0, strength=2.)
        with self.assertRaises(ValueError):
            rows.set_settings(0, 0, colour='red')
        self.assertEqual(rows.ranges(0), [(0, 100), (200, 250)])
        self.assertFalse(rows.can_undo())

    def test_suggestions_accept_reset_and_clear(self):
        rows = model([entry(ranges=[(0, 50)], automatic=[(40, 80), (200, 240)]), entry(mode='flat', ranges=[], automatic=[(60, 120)])])
        self.assertEqual(rows.suggestions(0), [(50, 80, 0), (60, 120, 1), (200, 240, 0)])
        self.assertEqual(rows.accept(0, (200, 240)), 40)
        self.assertEqual(rows.accept(0), 70)  # Pointe takes 50-80 first; flat gets the rest of its suggestion.
        self.assertEqual(rows.ranges(0), [(0, 80), (80, 120), (200, 240)])
        self.assertEqual(rows.accept(0), 0)
        rows.clear_row(0)
        self.assertEqual(rows.ranges(0), [])
        self.assertEqual(len(rows.entries), 2)  # Entries with suggestions survive an empty row.
        rows.reset_row(0)
        self.assertEqual(rows.ranges(0), [(40, 80), (80, 120), (200, 240)])

    def test_section_without_modes_or_row_field(self):
        """A smoothing-like section: one row, no mode field, list and int parameters; only a Section spec is needed."""
        from mocap.pose_rows import Mode, Param, Section
        section = Section('smoothing', 'pose smoothing', None, ((None, 'Smooth'),), (Mode('smooth', 'Smooth', '#8fa8c0'),),
                          (Param('joints', 'Joints', 'list'), Param('window_frames', 'Window', 'int', low=3, high=61)),
                          dict(joints=['spine'], window_frames=9, enabled=True), mode_field=None)
        rows = PoseRows(section)
        rows.load([], FRAMES, ('project', 'initial'))
        rows.set_interval(0, 0, None, (10, 30))
        rows.set_draw_settings(0, window_frames=15)
        rows.set_interval(0, 1, None, (40, 50))
        self.assertEqual([(e['window_frames'], e['ranges']) for e in rows.entries], [(9, [(10, 30)]), (15, [(40, 50)])])
        self.assertNotIn('mode', rows.entries[0])
        self.assertEqual(rows.describe(rows.interval_settings(0, 1)), 'Smooth 15')
        rows.set_settings(0, 1, window_frames=9)
        self.assertEqual([e['ranges'] for e in rows.entries], [[(10, 30), (40, 50)]])

    def test_store_round_trip_keeps_tracks_and_history(self):
        from mocap import self_contact_store as store
        with tempfile.TemporaryDirectory() as folder:
            store.save(folder, 'initial', [dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]])], CONTEXT)
            rows = model([entry()])
            rows.set_interval(1, 0, None, (5, 9))
            from mocap.pose_rows import write_section
            write_section(folder, 'initial', 'articulation', rows.payload(), CONTEXT)
            sections = read_sections(folder, 'initial', CONTEXT)
            self.assertFalse(rows.load(sections['articulation']['entries'], FRAMES, ('project', 'initial')))
            self.assertTrue(rows.can_undo())
            self.assertEqual(store.read(folder, 'initial', CONTEXT)[0][0]['ranges'], [(345, 421)])
            store.save(folder, 'initial', [], dict(frames=FRAMES + 1, fps=30.))
            self.assertIn('different length', read_sections(folder, 'initial', CONTEXT)['articulation']['warning'])


class PoseTimelineTest(unittest.TestCase):
    def setUp(self):
        from mocap.timeline_editor import TimelineSlider
        self.slider = TimelineSlider(Qt.Horizontal)
        self.slider.resize(1000, 520)
        self.slider.setRange(0, FRAMES - 1)
        self.saved, self.pose_saved, self.foot = [], [], []
        self.slider.sc_listener = self.saved.append
        self.slider.pose_listener = lambda key, payload: self.pose_saved.append((key, payload))
        self.slider.contacts_changed.connect(self.foot.append)
        self.slider.set_contacts([[(10, 40)], [(60, 90)], [], []], editable=True, identity=('p', 'initial'), token=1)
        self.slider.contact_edit_enabled = True
        self.slider.set_self_contacts([dict(effector='RightHand', target='RightFoot', ranges=[[345, 380]], grip='auto')], FRAMES,
                                      ('p', 'initial'), editable=True)
        self.load([entry(ranges=[(0, 100), (300, 421)], profile='dancer'), entry(foot='RightFoot', ranges=[(0, 421)], profile='dancer'),
                   entry(mode='match', ranges=[], automatic=[(150, 220)])])

    def load(self, entries):
        from mocap.self_contact_store import normalize_articulation
        self.slider.set_pose_sections(dict(articulation=dict(entries=normalize_articulation(entries, FRAMES), warning='')), FRAMES,
                                      ('p', 'initial'))

    def tearDown(self):
        self.slider.deleteLater()

    @property
    def left(self):
        return self.slider.pose_first_row()

    def point(self, row, frame):
        return QPointF(self.slider.frame_x(frame), 28 + (row + 2) * self.slider.TRACK_HEIGHT + self.slider.TRACK_HEIGHT / 2)

    def mouse(self, kind, row, frame, buttons=Qt.LeftButton):
        position = self.point(row, frame)
        return QMouseEvent(kind, position, self.slider.mapToGlobal(position.toPoint()), Qt.LeftButton, buttons, Qt.NoModifier)

    def drag(self, row, first, last):
        self.slider.mousePressEvent(self.mouse(QMouseEvent.MouseButtonPress, row, first))
        self.slider.mouseMoveEvent(self.mouse(QMouseEvent.MouseMove, row, last))
        self.slider.mouseReleaseEvent(self.mouse(QMouseEvent.MouseButtonRelease, row, last, Qt.NoButton))

    def menu(self, row, frame, *path):
        """Open the context menu at row/frame and trigger the action at path (menu titles, then the action text)."""
        seen = []

        def run(menu, *args):
            seen.append([a.text() for a in menu.actions()])
            target = menu
            for name in path[:-1]:
                target = next(a.menu() for a in target.actions() if a.text() == name)
            if path:
                action = next(a for a in target.actions() if a.text() == path[-1])
                self.assertTrue(action.isEnabled() and target.isEnabled(), path)
                action.trigger()
            return None
        position = self.point(row, frame).toPoint()
        event = QContextMenuEvent(QContextMenuEvent.Mouse, position, self.slider.mapToGlobal(position))
        with mock.patch.object(self.slider, 'run_menu', run):
            self.slider.contextMenuEvent(event)
        return seen[0] if seen else None

    def test_layout_rows_follow_self_contacts(self):
        self.assertEqual(self.left, FIRST_ROW + 2)
        self.assertEqual(self.slider.pose_row(self.left)[1], 0)
        self.assertEqual(self.slider.pose_row(self.left + 1)[1], 1)
        labels = [model.row_label(r) for model, r in self.slider.pose_row_models()]
        self.assertEqual(labels, ['L pose', 'R pose', 'L arm 2D', 'R arm 2D', 'L leg 2D', 'R leg 2D', 'Smooth'])
        self.assertEqual(self.slider.pose_row(self.left + 5)[0].section.key, 'placement')
        self.assertEqual(self.slider.pose_row(self.left + 6), (self.slider.pose_models[2], 0))
        self.assertIsNone(self.slider.pose_row(self.left + 7))
        self.assertEqual(self.slider._track(self.left), [(0, 100), (300, 421)])
        self.assertEqual(self.slider._track(FIRST_ROW + 1), [])  # '+ Pair' row
        self.slider.set_compare_delta([0.] * FRAMES)
        self.assertEqual(self.slider.delta_row(), self.left + 7)
        self.assertGreaterEqual(self.slider.minimumHeight(), 28 + (2 + self.left + 8) * self.slider.TRACK_HEIGHT)
        self.slider.set_self_contacts(None)
        self.assertEqual(self.slider.pose_models[0].entries, [])
        self.assertIsNone(self.slider._contact_row(self.mouse(QMouseEvent.MouseMove, FIRST_ROW + 1, 50)))

    def test_drag_adds_with_row_mode_and_saves(self):
        self.drag(self.left, 120, 140)
        key, payload = self.pose_saved[-1]
        self.assertEqual(key, 'articulation')
        added = [e for e in payload if e['foot'] == 'LeftFoot' and e['mode'] == 'match']
        self.assertEqual(added[0]['ranges'], [[120, 141]])
        self.assertEqual(added[0]['automatic'], [[150, 220]])  # Joins the matching entry, which holds the suggestion.
        self.drag(self.left, 90, 130)  # Starts inside the first interval: moves it, stops at the next one.
        self.assertEqual(self.slider._track(self.left), [(20, 120), (120, 141), (300, 421)])
        self.assertEqual(self.saved, [])
        self.assertEqual(self.foot, [])

    def test_menu_mode_strength_and_undo_order(self):
        items = self.menu(self.left, 50)
        self.assertIn('Mode', items)
        self.assertIn('Strength', items)
        self.assertIn('Limits', items)
        self.menu(self.left, 50, 'Mode', 'Flat')
        self.assertEqual(self.slider.pose_models[0].interval_settings(0, 0)['mode'], 'flat')
        self.menu(self.left, 50, 'Strength', '50%')
        self.assertEqual(self.slider.pose_models[0].interval_settings(0, 0)['strength'], .5)
        self.drag(FIRST_ROW, 10, 30)
        self.drag(0, 100, 120)
        self.assertEqual(self.slider._undo_contacts[-4:], ['self-contact', 'self-contact', 'self-contact', (0, 1, None, (100, 121))])
        self.slider.undo_contact()
        self.slider.undo_contact()
        self.assertEqual(self.slider.sc_rows.tracks[0]['ranges'], [(345, 380)])
        self.assertEqual(self.slider.pose_models[0].interval_settings(0, 0)['strength'], .5)
        self.slider.undo_contact()
        self.assertEqual(self.slider.pose_models[0].interval_settings(0, 0), dict(mode='flat', strength=1., profile='dancer', enabled=True))
        self.assertEqual(self.pose_saved[-1][0], 'articulation')
        self.slider.redo_contact()
        self.assertEqual(self.slider.pose_models[0].interval_settings(0, 0)['strength'], .5)

    def test_menu_without_interval_sets_new_interval_mode(self):
        self.menu(self.left, 250, 'Mode', 'Pointe')
        self.assertEqual(self.pose_saved, [])
        self.slider.mouseDoubleClickEvent(self.mouse(QMouseEvent.MouseButtonDblClick, self.left, 250))
        self.assertEqual(self.slider._track(self.left), [(0, 100), (250, 259), (300, 421)])
        self.assertEqual(self.slider.pose_models[0].interval_settings(0, 1)['mode'], 'pointe')

    def test_suggestions_accept_and_reload_keeps_other_history(self):
        self.drag(FIRST_ROW, 10, 30)
        self.slider.mouseDoubleClickEvent(self.mouse(QMouseEvent.MouseButtonDblClick, self.left, 160))
        self.assertEqual(self.slider._track(self.left), [(0, 100), (150, 220), (300, 421)])
        self.menu(self.left, 160, 'Delete Selected Interval')
        self.menu(self.left, 160, 'Accept All Suggestions in Row')
        self.assertEqual(self.slider._track(self.left), [(0, 100), (150, 220), (300, 421)])
        self.assertEqual(self.slider._undo_contacts.count('self-contact'), 4)
        self.load([entry(ranges=[(0, 10)])])  # Changed on disk: pose history goes, the self-contact step stays.
        self.assertEqual(self.slider._undo_contacts.count('self-contact'), 1)
        self.slider.undo_contact()
        self.assertEqual(self.slider.sc_rows.tracks[0]['ranges'], [(345, 380)])

    def test_delete_key_and_read_only_section(self):
        self.slider.mousePressEvent(self.mouse(QMouseEvent.MouseButtonPress, self.left + 1, 200))
        self.slider.mouseReleaseEvent(self.mouse(QMouseEvent.MouseButtonRelease, self.left + 1, 200, Qt.NoButton))
        self.slider.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        self.assertEqual(self.slider._track(self.left + 1), [])
        self.slider.set_pose_sections(dict(articulation=dict(entries=[], warning='Saved foot poses belong to other motion')), FRAMES,
                                      ('p', 'initial'))
        self.assertFalse(self.slider._editable(self.left))
        self.drag(self.left, 120, 140)
        self.assertEqual(self.slider._track(self.left), [])

    def test_screenshot(self):
        import numpy as np
        delta = np.zeros(FRAMES, np.float32)
        delta[335:421] = np.linspace(0, .2, 86)
        self.load([entry(ranges=[(0, 100), (300, 421)], profile='dancer'), entry(mode='flat', ranges=[(100, 140)]),
                   entry(mode='match', ranges=[(230, 300)], strength=.6, automatic=[(150, 230)]),
                   entry(foot='RightFoot', mode='flexed', ranges=[(40, 80)], enabled=False),
                   entry(foot='RightFoot', ranges=[(120, 421)], profile='dancer')])
        self.slider.set_compare_delta(delta, 'vs before self-contacts')
        self.slider.resize(1100, self.slider.minimumHeight())
        self.slider.setValue(360)
        image = self.slider.grab()
        self.assertFalse(image.isNull())
        folder = SHOTS or Path(tempfile.gettempdir())
        folder.mkdir(parents=True, exist_ok=True)
        image.save(str(folder / 'timeline_pose_rows.png'))


PLACEMENT, SMOOTHING = (next(s for s in SECTIONS if s.key == key) for key in ('placement', 'smoothing'))


class PlacementSmoothingModelTest(unittest.TestCase):
    def test_placement_rows_per_chain_with_strength(self):
        from mocap.self_contact_store import normalize_placement
        rows = PoseRows(PLACEMENT)
        rows.load(normalize_placement([dict(chain='RightLeg', ranges=[[380, 421]])], FRAMES), FRAMES, ('project', 'initial'))
        self.assertEqual([rows.row_label(r) for r in range(rows.row_count())], ['L arm 2D', 'R arm 2D', 'L leg 2D', 'R leg 2D'])
        self.assertEqual(rows.ranges(3), [(380, 421)])
        rows.set_draw_settings(0, strength=.5)
        rows.set_interval(0, 0, None, (10, 30))
        self.assertEqual([(e['id'], e['chain'], e['strength']) for e in rows.entries], [('rl-place', 'RightLeg', 1.), ('la-place', 'LeftArm', .5)])
        self.assertEqual(rows.describe(rows.interval_settings(0, 0)), 'Match 2D 50%')
        self.assertNotIn('mode', rows.entries[1])
        self.assertEqual(normalize_placement(rows.payload(), FRAMES), rows.entries)

    def test_smoothing_one_row_groups_and_bone_lists(self):
        from mocap.self_contact_store import normalize_smoothing
        rows = PoseRows(SMOOTHING)
        rows.load([], FRAMES, ('project', 'initial'))
        self.assertEqual((rows.row_count(), rows.row_label(0), rows.describe(rows.draw_settings(0))), (1, 'Smooth', 'Smooth'))
        rows.set_interval(0, 0, None, (10, 30))
        rows.set_draw_settings(0, joints='arms', window_frames=9)
        rows.set_interval(0, 1, None, (40, 50))
        rows.set_settings(0, 0, joints=['LeftArm', 'LeftForeArm'])
        self.assertEqual([(e['id'], e['joints'], e['window_frames']) for e in rows.entries],
                         [('smooth-arms', 'arms', 9), ('smooth-la', ['LeftArm', 'LeftForeArm'], 5)])
        self.assertEqual(rows.describe(rows.interval_settings(0, 1)), 'Smooth arms 9f')
        self.assertEqual(rows.describe(rows.interval_settings(0, 0)), 'Smooth LeftArm,LeftForeArm')
        self.assertEqual(normalize_smoothing(rows.payload(), FRAMES), rows.entries)
        with self.assertRaises(ValueError):
            rows.set_settings(0, 0, joints=['Hips'])

    def test_store_round_trip_keeps_other_sections(self):
        from mocap import self_contact_store as store
        from mocap.pose_rows import write_section
        with tempfile.TemporaryDirectory() as folder:
            store.save(folder, 'initial', [dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]])], CONTEXT,
                       articulation=[entry()])
            write_section(folder, 'initial', 'placement', [dict(chain='LeftLeg', ranges=[[5, 9]])], CONTEXT)
            write_section(folder, 'initial', 'smoothing', [dict(joints='spine', window_frames=7, ranges=[[100, 120]])], CONTEXT)
            sections = read_sections(folder, 'initial', CONTEXT)
            self.assertEqual(sorted(sections), ['articulation', 'placement', 'smoothing'])
            self.assertEqual(sections['placement']['entries'][0]['ranges'], [(5, 9)])
            self.assertEqual(sections['smoothing']['entries'][0]['window_frames'], 7)
            self.assertEqual(len(sections['articulation']['entries']), 1)
            self.assertEqual(len(store.read(folder, 'initial', CONTEXT)[0]), 1)

    def test_settings_dialog_reads_group_or_typed_bones(self):
        from PySide6.QtWidgets import QSpinBox
        from mocap.pose_rows import SettingsDialog
        typed = {}

        def answer(dialog):
            boxes = dialog.findChildren(QComboBox)
            typed['items'] = [boxes[0].itemText(i) for i in range(boxes[0].count())]
            boxes[0].setCurrentText(typed['text'])
            dialog.findChildren(QSpinBox)[0].setValue(11)
            return QDialog.Accepted
        with mock.patch.object(QDialog, 'exec', answer):
            typed['text'] = 'legs'
            values = SettingsDialog.ask(None, SMOOTHING, dict(joints='arms', window_frames=5, enabled=True), 'Smooth')
            self.assertEqual(values, dict(joints='legs', window_frames=11, enabled=True))
            typed['text'] = 'LeftArm, LeftForeArm'
            values = SettingsDialog.ask(None, SMOOTHING, dict(joints='arms', window_frames=5, enabled=True), 'Smooth')
            self.assertEqual(values['joints'], ['LeftArm', 'LeftForeArm'])
        self.assertEqual(typed['items'], ['all-written', 'arms', 'legs', 'spine'])


class PlacementSmoothingTimelineTest(unittest.TestCase):
    point, mouse, drag, menu = PoseTimelineTest.point, PoseTimelineTest.mouse, PoseTimelineTest.drag, PoseTimelineTest.menu

    def setUp(self):
        from mocap.self_contact_store import normalize_articulation, normalize_placement, normalize_smoothing
        from mocap.timeline_editor import TimelineSlider
        self.slider = TimelineSlider(Qt.Horizontal)
        self.slider.resize(1100, 600)
        self.slider.setRange(0, FRAMES - 1)
        self.pose_saved = []
        self.slider.pose_listener = lambda key, payload: self.pose_saved.append((key, payload))
        self.slider.set_contacts([[(10, 40)], [(60, 90)], [], []], editable=True, identity=('p', 'initial'), token=1)
        self.slider.contact_edit_enabled = True
        self.slider.set_self_contacts([dict(effector='RightHand', target='RightFoot', ranges=[[345, 380]], grip='auto')], FRAMES,
                                      ('p', 'initial'), editable=True)
        sections = dict(articulation=[entry(ranges=[(0, 100), (300, 421)], profile='dancer'), entry(foot='RightFoot', mode='flat', ranges=[(120, 200)])],
                        placement=[dict(chain='RightLeg', ranges=[[380, 421]]), dict(chain='LeftArm', strength=.6, ranges=[[150, 230]]),
                                   dict(chain='RightArm', ranges=[], automatic=[[60, 110]])],
                        smoothing=[dict(joints='arms', window_frames=7, ranges=[[200, 260]]), dict(joints='all-written', ranges=[[20, 60]])])
        normalize = dict(articulation=normalize_articulation, placement=normalize_placement, smoothing=normalize_smoothing)
        self.slider.set_pose_sections({k: dict(entries=normalize[k](v, FRAMES), warning='') for k, v in sections.items()}, FRAMES,
                                      ('p', 'initial'))
        self.first = self.slider.pose_first_row()

    def tearDown(self):
        self.slider.deleteLater()

    def test_drag_on_smoothing_and_placement_rows_saves_their_sections(self):
        smooth, left_leg = self.first + 6, self.first + 4
        self.drag(smooth, 300, 320)
        key, payload = self.pose_saved[-1]
        self.assertEqual(key, 'smoothing')
        self.assertEqual([e['ranges'] for e in payload if e['joints'] == 'all-written'], [[[20, 60], [300, 321]]])
        self.drag(left_leg, 50, 70)
        key, payload = self.pose_saved[-1]
        self.assertEqual(key, 'placement')
        self.assertIn(('LeftLeg', [[50, 71]]), [(e['chain'], e['ranges']) for e in payload])
        self.slider.undo_contact()
        self.assertEqual(self.slider._track(left_leg), [])

    def test_menus_offer_settings_without_modes(self):
        items = self.menu(self.first + 6, 220)
        self.assertNotIn('Mode', items)
        self.assertIn('Joints', items)
        self.assertIn('Window', items)
        self.menu(self.first + 6, 220, 'Joints', 'legs')
        self.assertEqual(self.slider.pose_models[2].interval_settings(0, 1)['joints'], 'legs')
        self.menu(self.first + 6, 220, 'Window', '15f')
        self.assertEqual(self.pose_saved[-1][1][-1]['window_frames'], 15)
        self.assertIn('Strength', self.menu(self.first + 2, 180))
        self.menu(self.first + 3, 80, 'Accept This Suggestion')
        self.assertEqual(self.slider._track(self.first + 3), [(60, 110)])

    def test_screenshot(self):
        self.slider.resize(1100, self.slider.minimumHeight())
        self.slider.setValue(360)
        image = self.slider.grab()
        self.assertFalse(image.isNull())
        folder = SHOTS or Path(tempfile.gettempdir())
        folder.mkdir(parents=True, exist_ok=True)
        image.save(str(folder / 'timeline_placement_smoothing_rows.png'))


class PolishTest(unittest.TestCase):
    def test_labels_stack_without_overlap(self):
        from mocap.compare_overlay import layout_labels
        boxes = [QRectF(100, 50, 200, 18), QRectF(110, 55, 220, 18), QRectF(120, 52, 150, 18), QRectF(600, 52, 300, 18)]
        placed = layout_labels(boxes, 800)
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                self.assertFalse(a.intersects(b), (a, b))
        self.assertEqual(placed[0], boxes[0])
        self.assertLessEqual(placed[3].right(), 796)

    def test_labels_paint(self):
        from PySide6.QtGui import QColor, QImage, QPainter
        from mocap.compare_overlay import paint_labels
        image = QImage(500, 200, QImage.Format_ARGB32)
        image.fill(QColor('#303030'))
        painter = QPainter(image)
        paint_labels(painter, [(QPointF(60, 80), 'RH→RF  pending: Apply Correction  (33 cm joint gap)', QColor('#9a9a9a')),
                               (QPointF(70, 84), 'LH→RF  pending: Apply Correction  (93 cm joint gap)', QColor('#e0b050'))], 500)
        painter.end()
        folder = SHOTS or Path(tempfile.gettempdir())
        folder.mkdir(parents=True, exist_ok=True)
        image.save(str(folder / 'contact_labels_stacked.png'))

    def test_help_mentions_pose_and_self_contact_rows(self):
        from mocap.selfcontact_ui import TIMELINE_HELP
        for text in ('Self-contact rows', 'L/R pose', 'Pointe', 'grip', 'Apply Correction', 'leg 2D', 'Smooth:'):
            self.assertIn(text, TIMELINE_HELP)

    def test_pair_dialog_offers_grip_off_and_auto(self):
        from mocap.selfcontact_rows import PairDialog
        found = {}

        def inspect(dialog):
            boxes = dialog.findChildren(QComboBox)
            found['grip'] = [[box.itemData(i) for i in range(box.count())] for box in boxes]
            return QDialog.Rejected
        with mock.patch.object(QDialog, 'exec', inspect):
            PairDialog.ask(None, dict(effector='RightHand', target='RightFoot', distance='auto', reciprocal='auto', grip='auto', enabled=True))
        self.assertIn(['off', 'auto'], found['grip'])

    def test_io_thread_saves_sections_and_reads_them_back(self):
        from mocap import self_contact_store as store
        from mocap.selfcontact_ui import SelfContactIO
        with tempfile.TemporaryDirectory() as folder:
            store.save(folder, 'initial', [dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]])], CONTEXT)
            io = SelfContactIO()
            results = []
            io.done.connect(lambda token, result: results.append(result))
            io.failed.connect(lambda token, message, job: results.append(message))
            io.start()
            try:
                io.request(dict(action='save', project=folder, base='initial', kind='corrected_initial', context=CONTEXT, edit_seq=1,
                                section='articulation', entries=[entry()]))
                deadline = time.monotonic() + 10
                while not results and time.monotonic() < deadline:
                    APP.processEvents()
                    time.sleep(.01)
            finally:
                io.stop()
                io.wait()
            self.assertIsInstance(results[0], dict, results)
            self.assertEqual(results[0]['tracks'][0]['ranges'], [(345, 421)])
            self.assertEqual(results[0]['sections']['articulation']['entries'][0]['ranges'], [(0, 100)])
            self.assertEqual(results[0]['contact_revision'].split('|sc:')[1], store.revision(folder, 'initial'))


@unittest.skipUnless(os.environ.get('RGBMOCAP_POSE_PROJECT'), 'set RGBMOCAP_POSE_PROJECT to a copied workspace')
class WindowTest(unittest.TestCase):
    def wait(self, predicate, seconds=120):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return True
            time.sleep(.02)
        return False

    def test_project_pose_rows_save_and_mark_pending(self):
        from mocap.gui import MainWindow
        from mocap import self_contact_store as store
        window = MainWindow()
        try:
            window.resize(1600, 1000)
            window.show()
            project = Path(os.environ['RGBMOCAP_POSE_PROJECT'])
            window.load_project(project)
            slider = window.slider
            self.assertTrue(self.wait(lambda: slider.sc_visible and slider.pose_models[0].entries), window.status.text())
            self.assertIn('L/R pose', window.timeline_help_dialog.text())
            model = slider.pose_models[0]
            revision = window.self_contacts.context()[3].get('revision')
            first = slider.pose_first_row()
            slider._pose_settings(model, 0, 0, mode='match', strength=.5)
            self.assertTrue(self.wait(lambda: window.self_contacts.save_token is None and not window.contact_saving, 30))
            folder = window.self_contacts.context()[0]
            entries, _, _ = store.read_articulation(folder, 'initial', window.self_contacts.context()[3]['context'])
            self.assertIn(('match', .5), [(e['mode'], e['strength']) for e in entries if e['foot'] == 'LeftFoot'])
            self.assertNotEqual(window.self_contacts.context()[3].get('revision'), revision)
            self.assertTrue(slider._editable(first))
            shots = SHOTS or Path(tempfile.gettempdir())
            shots.mkdir(parents=True, exist_ok=True)
            slider.grab().save(str(shots / 'window_timeline_pose_rows.png'))
            window.grab().save(str(shots / 'window_pose_rows_offscreen.png'))
            slider.undo_contact()
            self.assertTrue(self.wait(lambda: window.self_contacts.save_token is None and not window.contact_saving, 30))
            entries, _, _ = store.read_articulation(folder, 'initial', window.self_contacts.context()[3]['context'])
            self.assertEqual({e['mode'] for e in entries}, {'pointe'})
        finally:
            window.shutdown_preview()
            window.deleteLater()
            APP.processEvents()


if __name__ == '__main__':
    unittest.main()
