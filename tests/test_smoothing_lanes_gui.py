"""Overlapping pose smoothing intervals (different joints, as `contacts smooth add` allows) stacked in lanes of the Smooth row.

RGBMOCAP_SHOTS=<folder> keeps the screenshot.
"""
import os
import sys
import tempfile
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

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mocap.pose_rows import SECTIONS, PoseRows  # noqa: E402

APP = QApplication.instance() or QApplication([])
FRAMES = 421
SHOTS = Path(os.environ['RGBMOCAP_SHOTS']) if os.environ.get('RGBMOCAP_SHOTS') else None
SMOOTHING = next(s for s in SECTIONS if s.key == 'smoothing')
# arms 100-200, legs 150-250 and spine 180-220 overlap; all-written 300-320 shares bones with arms and legs but not their frames.
ENTRIES = [dict(joints='arms', ranges=[[100, 200]]), dict(joints='legs', window_frames=9, ranges=[[150, 250]]),
           dict(joints='spine', ranges=[[180, 220]]), dict(joints='all-written', ranges=[[300, 320]])]


def normalized(entries):
    from mocap.self_contact_store import normalize_smoothing
    return normalize_smoothing(entries, FRAMES)


def model(entries=ENTRIES):
    rows = PoseRows(SMOOTHING)
    rows.load(normalized(entries), FRAMES, ('project', 'initial'))
    return rows


class SmoothingLanesModelTest(unittest.TestCase):
    def test_overlaps_stack_in_lanes_with_a_spare_lane(self):
        rows = model()
        self.assertEqual([rows.row_label(r) for r in range(rows.row_count())], ['Smooth', 'Smooth 2', 'Smooth 3', '+ Smooth'])
        self.assertEqual([rows.ranges(r) for r in range(rows.row_count())], [[(100, 200), (300, 320)], [(150, 250)], [(180, 220)], []])
        self.assertEqual([rows.is_spare(r) for r in range(4)], [False, False, False, True])
        self.assertEqual(rows.interval_settings(1, 0)['joints'], 'legs')
        self.assertEqual(rows.describe(rows.interval_settings(1, 0)), 'Smooth legs 9f')
        empty = model([])
        self.assertEqual((empty.row_count(), empty.row_label(0), empty.is_spare(0)), (1, 'Smooth', False))
        single = model(ENTRIES[:1])
        self.assertEqual([single.row_label(r) for r in range(single.row_count())], ['Smooth', '+ Smooth'])

    def test_edits_relayout_and_keep_the_store_rules(self):
        rows = model()
        rows.set_interval(1, 0, (150, 250), (260, 290))  # No longer overlaps: joins the first lane.
        self.assertEqual([rows.ranges(r) for r in range(rows.row_count())], [[(100, 200), (260, 290), (300, 320)], [(180, 220)], []])
        self.assertEqual(rows.locate((260, 290)), (0, 1))
        with self.assertRaises(ValueError):
            rows.set_interval(2, 0, None, (265, 280))  # The spare lane draws all-written: shares the leg bones on 260-290.
        rows.set_draw_settings(2, joints='spine')
        rows.set_interval(2, 0, None, (265, 280))
        self.assertEqual(rows.ranges(1), [(180, 220), (265, 280)])
        self.assertEqual(normalized(rows.payload()), rows.entries)
        rows.clear_row(1)  # Only this lane.
        self.assertEqual([rows.ranges(r) for r in range(rows.row_count())], [[(100, 200), (260, 290), (300, 320)], []])
        self.assertTrue(rows.undo())
        self.assertEqual(rows.ranges(1), [(180, 220), (265, 280)])

    def test_suggestions_live_in_the_first_lane(self):
        rows = model(ENTRIES + [dict(joints='spine', window_frames=7, ranges=[], automatic=[[60, 110], [240, 270]])])
        self.assertEqual([(s, e) for s, e, _ in rows.suggestions(0)], [(60, 100), (250, 270)])  # Minus every lane.
        self.assertEqual(rows.suggestions(1), [])
        self.assertFalse(rows.has_automatic(1))
        self.assertEqual(rows.accept(2), 0)
        self.assertEqual(rows.accept(0, (250, 270)), 20)
        self.assertEqual(rows.ranges(0), [(100, 200), (250, 270), (300, 320)])

    def test_unstacked_sections_keep_one_list_per_row(self):
        from mocap.self_contact_store import normalize_placement
        rows = PoseRows(next(s for s in SECTIONS if s.key == 'placement'))
        rows.load(normalize_placement([dict(chain='LeftArm', ranges=[[10, 20]])], FRAMES), FRAMES, ('project', 'initial'))
        self.assertEqual((rows.row_count(), rows.row_label(3), rows.is_spare(3)), (4, 'R leg 2D', False))
        rows.set_draw_settings(0, strength=.5)
        with self.assertRaises(ValueError):
            rows.set_interval(0, 1, None, (15, 30))


class SmoothingLanesTimelineTest(unittest.TestCase):
    def setUp(self):
        from mocap.timeline_editor import TimelineSlider
        self.slider = TimelineSlider(Qt.Horizontal)
        self.slider.resize(1100, 600)
        self.slider.setRange(0, FRAMES - 1)
        self.pose_saved, self.resized = [], []
        self.slider.pose_listener = lambda key, payload: self.pose_saved.append((key, payload))
        self.slider.sc_resized = lambda: self.resized.append(self.slider.minimumHeight())
        self.slider.set_contacts([[(10, 40)], [(60, 90)], [], []], editable=True, identity=('p', 'initial'), token=1)
        self.slider.contact_edit_enabled = True
        self.slider.set_self_contacts([], FRAMES, ('p', 'initial'), editable=True)
        self.base_height = self.slider.minimumHeight()
        self.slider.set_pose_sections(dict(smoothing=dict(entries=normalized(ENTRIES), warning='')), FRAMES, ('p', 'initial'))
        self.model = self.slider.pose_models[2]
        self.smooth = self.slider.pose_track(self.model, 0)

    def tearDown(self):
        self.slider.deleteLater()

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
        def run(menu, *args):
            target = menu
            for name in path[:-1]:
                target = next(a.menu() for a in target.actions() if a.text() == name)
            action = next(a for a in target.actions() if a.text() == path[-1])
            self.assertTrue(action.isEnabled(), path)
            action.trigger()
        position = self.point(row, frame).toPoint()
        with mock.patch.object(self.slider, 'run_menu', run):
            self.slider.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse, position, self.slider.mapToGlobal(position)))

    def lanes(self):
        return [self.slider._track(self.smooth + r) for r in range(self.model.row_count())]

    def test_lanes_are_timeline_rows_and_grow_the_widget(self):
        labels = [m.row_label(r) for m, r in self.slider.pose_row_models()][-4:]
        self.assertEqual(labels, ['Smooth', 'Smooth 2', 'Smooth 3', '+ Smooth'])
        self.assertEqual(self.lanes(), [[(100, 200), (300, 320)], [(150, 250)], [(180, 220)], []])
        self.assertEqual(self.slider.minimumHeight(), self.base_height + 3 * self.slider.TRACK_HEIGHT)
        self.assertEqual(self.resized[-1], self.slider.minimumHeight())
        self.assertIsNone(self.slider.pose_row(self.smooth + 4))

    def test_drag_in_a_lane_saves_and_the_selection_follows_the_interval(self):
        self.drag(self.smooth + 1, 200, 240)  # Legs 150-250 to 190-290: now starts after spine 180-220, so it moves down a lane.
        key, payload = self.pose_saved[-1]
        self.assertEqual(key, 'smoothing')
        self.assertIn(('legs', [[190, 290]]), [(e['joints'], e['ranges']) for e in payload])
        self.assertEqual(self.lanes(), [[(100, 200), (300, 320)], [(180, 220)], [(190, 290)], []])
        self.assertEqual(self.slider._selected_contact, (self.smooth + 2, 0))
        self.slider.undo_contact()
        self.assertEqual(self.lanes()[1], [(150, 250)])

    def test_spare_lane_adds_an_overlapping_interval(self):
        self.menu(self.smooth + 3, 120, 'Joints', 'legs')  # New intervals of the spare lane.
        self.drag(self.smooth + 3, 110, 130)
        self.assertIn(('legs', 5, [[110, 131]]), [(e['joints'], e['window_frames'], e['ranges']) for e in self.pose_saved[-1][1]])
        self.assertEqual(self.lanes()[1], [(110, 131), (150, 250)])
        self.assertEqual(self.slider._selected_contact, (self.smooth + 1, 0))
        self.assertEqual(self.model.row_label(self.model.row_count() - 1), '+ Smooth')

    def test_conflicting_overlap_is_reported_and_menus_edit_one_lane(self):
        reports = []
        self.slider.sc_status = reports.append
        self.drag(self.smooth + 3, 110, 130)  # all-written shares the arm bones of 100-200.
        self.assertIn('same bones', reports[-1])
        self.assertEqual(self.pose_saved, [])
        self.menu(self.smooth + 2, 200, 'Window', '15f')
        self.assertIn(('spine', 15, [[180, 220]]), [(e['joints'], e['window_frames'], e['ranges']) for e in self.pose_saved[-1][1]])
        self.assertEqual(self.slider._selected_contact, (self.smooth + 2, 0))
        self.slider.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
        self.assertEqual(self.lanes(), [[(100, 200), (300, 320)], [(150, 250)], []])
        self.menu(self.smooth + 1, 200, 'Clear Row')
        self.assertEqual(self.lanes(), [[(100, 200), (300, 320)], []])

    def test_screenshot(self):
        self.slider.resize(1100, self.slider.minimumHeight())
        self.slider.setValue(190)
        image = self.slider.grab()
        self.assertFalse(image.isNull())
        folder = SHOTS or Path(tempfile.gettempdir())
        folder.mkdir(parents=True, exist_ok=True)
        image.save(str(folder / 'timeline_smoothing_lanes.png'))


if __name__ == '__main__':
    unittest.main()
