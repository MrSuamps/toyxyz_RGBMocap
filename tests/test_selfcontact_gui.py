"""Self-contact timeline rows and the original-vs-corrected overlay, under offscreen Qt.

Optional: RGBMOCAP_SELFCONTACT_PROJECT=<workspace with an applied self-contact> runs the window test on real data (the
project's view settings may be updated, so use a copy); RGBMOCAP_SHOTS=<folder> keeps the screenshots.
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from mocap.runtime_registry import activate
    activate(ROOT, 'gui')
except RuntimeError:
    pass  # Development interpreters provide PySide6 themselves.

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from mocap.selfcontact_rows import FIRST_ROW, SelfContactRows, normalize_tracks, row_label  # noqa: E402

APP = QApplication.instance() or QApplication([])
FRAMES = 421
SHOTS = Path(os.environ['RGBMOCAP_SHOTS']) if os.environ.get('RGBMOCAP_SHOTS') else None


def track(**values):
    return dict(dict(effector='RightHand', target='RightFoot', ranges=[[345, 421]]), **values)


class RowModelTest(unittest.TestCase):
    def model(self, tracks=()):
        rows = SelfContactRows()
        rows.load(list(tracks), FRAMES, ('project', 'initial'))
        return rows

    def test_normalize_clamps_merges_and_names(self):
        tracks = normalize_tracks([track(ranges=[[400, 500], [10, 20], [15, 30], [30, 31], [-5, 2]], future='kept'),
                                   track(id=None), dict(effector='LeftHand', target='Head')], FRAMES)
        self.assertEqual(tracks[0]['ranges'], [(0, 2), (10, 31), (400, 421)])
        self.assertEqual([t['id'] for t in tracks], ['rh-rf', 'rh-rf-2', 'lh-head'])
        self.assertEqual(tracks[0]['future'], 'kept')
        self.assertEqual((tracks[2]['distance'], tracks[2]['reciprocal'], tracks[2]['grip'], tracks[2]['enabled']), ('auto', 'auto', 'off', True))
        self.assertEqual(row_label(tracks[0]), 'RH→RF')
        self.assertEqual(row_label(tracks[2]), 'LH→Hd')

    def test_normalized_tracks_pass_the_store(self):
        from mocap import self_contact_store as store
        tracks = normalize_tracks([track(ranges=[[1, 5], [5, 9]]), dict(effector='LeftHand', target='RightHand', ranges=[[2, 3]])], FRAMES)
        self.assertEqual([t['ranges'] for t in store.normalize_tracks(tracks, FRAMES)], [t['ranges'] for t in tracks])

    def test_add_remove_and_undo(self):
        rows = self.model()
        index = rows.add_pair('RightHand', 'RightFoot', distance=.04)
        self.assertEqual((index, rows.tracks[0]['id'], rows.tracks[0]['distance']), (0, 'rh-rf', .04))
        rows.add_pair('LeftHand', 'Head')
        rows.remove_row(0)
        self.assertEqual([t['id'] for t in rows.tracks], ['lh-head'])
        self.assertTrue(rows.undo())
        self.assertEqual([t['id'] for t in rows.tracks], ['rh-rf', 'lh-head'])
        self.assertTrue(rows.undo())
        self.assertTrue(rows.undo())
        self.assertEqual(rows.tracks, [])
        self.assertFalse(rows.undo())
        self.assertTrue(rows.redo())
        self.assertEqual([t['id'] for t in rows.tracks], ['rh-rf'])
        with self.assertRaises(ValueError):
            rows.add_pair('RightHand', 'RightForeArm')

    def test_intervals_settings_and_one_hand_rule(self):
        rows = self.model([track(ranges=[[100, 200]]), track(target='Head', ranges=[[300, 320]])])
        rows.set_interval(0, 1, None, (220, 240))
        rows.set_interval(0, 0, (100, 200), (90, 210))
        self.assertEqual(rows.tracks[0]['ranges'], [(90, 210), (220, 240)])
        with self.assertRaises(ValueError):
            rows.set_interval(1, 0, (300, 320), (230, 320))
        self.assertEqual(rows.tracks[1]['ranges'], [(300, 320)])
        self.assertEqual(rows.blockers(1), [(90, 210), (220, 240)])
        rows.update_settings(0, distance=.05, reciprocal='always', grip='auto')
        self.assertEqual((rows.tracks[0]['distance'], rows.tracks[0]['reciprocal'], rows.tracks[0]['grip']), (.05, 'always', 'auto'))
        rows.update_settings(1, enabled=False)
        rows.set_interval(1, 0, (300, 320), (230, 320))
        with self.assertRaises(ValueError):
            rows.update_settings(1, enabled=True)
        rows.set_interval(0, 1, (220, 240), None)
        self.assertEqual(rows.tracks[0]['ranges'], [(90, 210)])
        for _ in range(4):
            rows.undo()
        self.assertEqual(rows.tracks[0]['ranges'], [(90, 210), (220, 240)])

    def test_accept_suggestions_and_reset(self):
        rows = self.model([track(ranges=[[340, 350]], automatic=[[345, 352], [357, 361], [366, 420]]),
                           track(target='Head', ranges=[[358, 360]], automatic=[[100, 110]])])
        self.assertEqual(rows.suggestions(0), [(350, 352), (357, 361), (366, 420)])
        self.assertEqual(rows.accept(0, (357, 361)), 2)
        self.assertEqual(rows.tracks[0]['ranges'], [(340, 350), (357, 358), (360, 361)])
        rows.accept(0)
        self.assertEqual(rows.tracks[0]['ranges'], [(340, 352), (357, 358), (360, 361), (366, 420)])
        self.assertEqual(rows.suggestions(0), [(358, 360)])
        self.assertEqual(rows.accept(0), 0)
        rows.reset_row(1)
        self.assertEqual(rows.tracks[1]['ranges'], [(100, 110)])
        rows.undo()
        rows.undo()
        self.assertEqual(rows.tracks[0]['ranges'], [(340, 350), (357, 358), (360, 361)])

    def test_reload_keeps_history_only_for_identical_tracks(self):
        rows = self.model([track()])
        rows.update_settings(0, grip='auto')
        self.assertFalse(rows.load(rows.payload(), FRAMES, ('project', 'initial')))
        self.assertTrue(rows.can_undo())
        self.assertTrue(rows.load([track()], FRAMES, ('project', 'initial')))
        self.assertFalse(rows.can_undo())


class TimelineRowsTest(unittest.TestCase):
    def setUp(self):
        from mocap.timeline_editor import TimelineSlider
        self.slider = TimelineSlider(Qt.Horizontal)
        self.slider.resize(1000, 400)
        self.slider.setRange(0, FRAMES - 1)
        self.saved = []
        self.foot = []
        self.slider.sc_listener = self.saved.append
        self.slider.contacts_changed.connect(self.foot.append)
        self.slider.set_contacts([[(10, 40)], [(60, 90)], [], [(200, 230)]], editable=True, identity=('p', 'initial'), token=1)
        self.slider.contact_edit_enabled = True
        self.slider.set_self_contacts([track(ranges=[[345, 380]], automatic=[[345, 352], [366, 420]]),
                                       dict(effector='LeftHand', target='RightHand', ranges=[], automatic=[[100, 140]]),
                                       dict(effector='LeftHand', target='Head', ranges=[[300, 330]])],
                                      FRAMES, ('p', 'initial'), editable=True)

    def tearDown(self):
        self.slider.deleteLater()

    def point(self, row, frame):
        return QPointF(self.slider.frame_x(frame), 28 + (row + 2) * self.slider.TRACK_HEIGHT + self.slider.TRACK_HEIGHT / 2)

    def drag(self, row, first, last):
        start, end = self.point(row, first), self.point(row, last)
        for kind, position, buttons in ((QMouseEvent.MouseButtonPress, start, Qt.LeftButton), (QMouseEvent.MouseMove, end, Qt.LeftButton),
                                        (QMouseEvent.MouseButtonRelease, end, Qt.NoButton)):
            event = QMouseEvent(kind, position, self.slider.mapToGlobal(position.toPoint()), Qt.LeftButton, buttons, Qt.NoModifier)
            {QMouseEvent.MouseButtonPress: self.slider.mousePressEvent, QMouseEvent.MouseMove: self.slider.mouseMoveEvent,
             QMouseEvent.MouseButtonRelease: self.slider.mouseReleaseEvent}[kind](event)

    def test_rows_index_and_height(self):
        self.assertEqual(self.slider._contact_row(QMouseEvent(QMouseEvent.MouseMove, self.point(FIRST_ROW + 1, 50), QPointF(),
                                                              Qt.NoButton, Qt.NoButton, Qt.NoModifier)), FIRST_ROW + 1)
        self.assertGreaterEqual(self.slider.minimumHeight(), 28 + (2 + FIRST_ROW + 4) * self.slider.TRACK_HEIGHT)
        self.assertEqual(self.slider._track(FIRST_ROW), [(345, 380)])

    def test_drag_creates_interval_saves_and_undoes(self):
        self.drag(FIRST_ROW + 1, 20, 50)
        self.assertEqual(self.slider.sc_rows.tracks[1]['ranges'], [(20, 51)])
        self.assertEqual(self.saved[-1][1]['ranges'], [[20, 51]])
        self.drag(FIRST_ROW + 1, 310, 360)  # LH->Head holds the left hand on 300-329: refused.
        self.assertEqual(self.slider.sc_rows.tracks[1]['ranges'], [(20, 51)])
        self.drag(FIRST_ROW + 1, 280, 360)  # Stops where LH->Head starts.
        self.assertEqual(self.slider.sc_rows.tracks[1]['ranges'], [(20, 51), (280, 300)])
        self.slider.undo_contact()
        self.slider.undo_contact()
        self.assertEqual(self.slider.sc_rows.tracks[1]['ranges'], [])
        self.slider.redo_contact()
        self.assertEqual(self.slider.sc_rows.tracks[1]['ranges'], [(20, 51)])
        self.assertEqual(self.foot, [])

    def test_foot_rows_unchanged_and_shared_undo_order(self):
        self.drag(0, 100, 120)
        self.assertEqual(self.foot[-1][0], [(10, 40), (100, 121)])
        self.assertEqual(len(self.foot[-1]), 4)
        self.drag(FIRST_ROW, 10, 30)
        self.assertEqual(self.slider.sc_rows.tracks[0]['ranges'], [(10, 31), (345, 380)])
        self.slider.undo_contact()
        self.assertEqual(self.slider.sc_rows.tracks[0]['ranges'], [(345, 380)])
        self.assertEqual(self.slider.contacts[0], [(10, 40), (100, 121)])
        self.slider.undo_contact()
        self.assertEqual(self.slider.contacts[0], [(10, 40)])
        self.assertEqual(self.slider._redo_contacts, ['self-contact', (0, 1, None, (100, 121))])
        self.slider.set_contacts([[(10, 40)], [], [], []], editable=True, identity=('other', 'initial'), token=2)
        self.assertEqual(self.slider._redo_contacts, ['self-contact'])

    def test_double_click_accepts_suggestion(self):
        position = self.point(FIRST_ROW, 400)
        event = QMouseEvent(QMouseEvent.MouseButtonDblClick, position, self.slider.mapToGlobal(position.toPoint()), Qt.LeftButton,
                            Qt.LeftButton, Qt.NoModifier)
        self.slider.mouseDoubleClickEvent(event)
        self.assertEqual(self.slider.sc_rows.tracks[0]['ranges'], [(345, 420)])

    def test_screenshot(self):
        import numpy as np
        delta = np.zeros(FRAMES, np.float32)
        delta[335:421] = np.linspace(0, .2, 86)
        self.slider.set_compare_delta(delta, 'vs before self-contacts')
        self.slider.resize(1000, self.slider.minimumHeight())
        self.slider.setValue(400)
        image = self.slider.grab()
        self.assertFalse(image.isNull())
        folder = SHOTS or Path(tempfile.gettempdir())
        folder.mkdir(parents=True, exist_ok=True)
        image.save(str(folder / 'timeline_selfcontact_rows.png'))


@unittest.skipUnless(os.environ.get('RGBMOCAP_SELFCONTACT_PROJECT'), 'set RGBMOCAP_SELFCONTACT_PROJECT to a copied workspace')
class WindowTest(unittest.TestCase):
    def wait(self, predicate, seconds=120):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return True
            time.sleep(.02)
        return False

    def test_project_rows_and_original_layer(self):
        from mocap.gui import MainWindow
        window = MainWindow()
        try:
            window.resize(1600, 1000)
            window.show()
            window.load_project(Path(os.environ['RGBMOCAP_SELFCONTACT_PROJECT']))
            self.assertTrue(self.wait(lambda: window.slider.sc_visible and window.slider.sc_rows.tracks), window.status.text())
            self.assertTrue(self.wait(lambda: window.self_contacts.overlay.available()), window.self_contacts.overlay.message)
            overlay = window.self_contacts.overlay
            self.assertEqual(overlay.data['source'], 'before')
            self.assertGreater(float(overlay.data['delta'].max()), .05)
            self.assertEqual(float(overlay.data['delta'][:300].max()), 0.)
            window.self_contacts.show_original.setChecked(True)
            window.slider.setValue(400)
            self.wait(lambda: window.motion_view.index == 400, 10)
            folder = SHOTS or Path(tempfile.gettempdir())
            folder.mkdir(parents=True, exist_ok=True)
            window.motion_view.grab().save(str(folder / 'viewport_original_offscreen.png'))
            window.slider.grab().save(str(folder / 'timeline_project.png'))
            window.grab().save(str(folder / 'window_offscreen.png'))
        finally:
            window.shutdown_preview()
            window.deleteLater()
            APP.processEvents()


if __name__ == '__main__':
    unittest.main()
