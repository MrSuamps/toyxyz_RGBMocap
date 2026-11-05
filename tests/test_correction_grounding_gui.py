"""Motion Correction grounding in the GUI: shown, saved and restored like `correct --grounding`, under offscreen Qt.

Optional: RGBMOCAP_GROUNDING_PROJECT=<copied ExMR model project corrected with `correct --grounding camera`> runs the window
test on real data (it saves that project's correction settings, so use a copy).
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from mocap.runtime_registry import activate
    activate(ROOT, 'gui')
except RuntimeError:
    pass  # Development interpreters provide PySide6 themselves.

from PySide6.QtWidgets import QApplication  # noqa: E402

from mocap.correction_settings import GROUNDING, correction_options, effective_options  # noqa: E402
from mocap.workspace_ui import camera_grounding_note  # noqa: E402

APP = QApplication.instance() or QApplication([])
# The settings every project stored before grounding existed; a default GUI must keep writing exactly these.
LEGACY = {'auto_floor': True, 'floor': 0.0, 'preset': 'normal', 'pelvis_offset': 0.0, 'mode': 'flat', 'foot_lock': True, 'hand_lock': False}


def take(folder, static=True, manifest=None):
    """A model project with the files correction.static_camera_centre reads (static=None: no ExMR camera)."""
    folder = Path(folder)
    (folder / 'project.json').write_text(json.dumps(manifest or {}), encoding='utf-8')
    if static is not None:
        (folder / 'camera').mkdir()
        (folder / 'exmr_camera_sources.json').write_text(json.dumps(dict(root='camera')), encoding='utf-8')
        (folder / 'camera' / 'exmr_camera.json').write_text(json.dumps(dict(identity=dict(static=static))), encoding='utf-8')
    return folder


class GroundingNoteTest(unittest.TestCase):
    def note(self, static=True, manifest=None, people=False, **options):
        with tempfile.TemporaryDirectory() as folder:
            project = take(folder, static, manifest)
            if people:
                (project / 'people.json').write_text('{}', encoding='utf-8')
            return camera_grounding_note(project, correction_options(options))

    def test_static_exmr_camera_has_no_note(self):
        self.assertEqual(self.note(), '')
        self.assertEqual(camera_grounding_note(None, LEGACY), '')

    def test_fallbacks_explain_vertical_grounding(self):
        self.assertIn('Moving camera', self.note(static=False))
        self.assertIn('ExMR', self.note(static=None))
        self.assertIn('Multi Level', self.note(mode='multi_level'))
        self.assertIn('Foot', self.note(foot_lock=False))
        self.assertIn('People share one floor', self.note(people=True))


class GroundingSettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mocap.gui import MainWindow
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.window.shutdown_preview()
        cls.window.deleteLater()
        APP.processEvents()

    def tearDown(self):
        self.window.project = None
        self.window.restore_correction_options({})

    def test_default_vertical_keeps_the_legacy_settings_payload(self):
        box = self.window.correction_grounding
        self.assertEqual([box.itemData(i) for i in range(box.count())], list(GROUNDING))
        self.assertEqual(box.currentData(), 'vertical')
        self.assertEqual(json.dumps(self.window.selected_correction_options()), json.dumps(LEGACY))

    def test_restore_and_select_camera(self):
        self.window.restore_correction_options(dict(LEGACY, preset='weak', grounding='camera'))
        self.assertEqual(self.window.correction_grounding.currentText(), 'Camera')
        self.assertEqual(self.window.selected_correction_options(), dict(LEGACY, preset='weak', grounding='camera'))
        self.window.restore_correction_options(dict(LEGACY, preset='weak'))
        self.assertEqual(self.window.correction_grounding.currentData(), 'vertical')

    def test_cli_and_gui_store_the_same_settings(self):
        from mocap.cli.pipeline import build_settings, saved_settings
        self.assertEqual(build_settings(SimpleNamespace(), 'exmr')['correction_options'], self.window.selected_correction_options())
        cli = build_settings(SimpleNamespace(grounding='camera'), 'exmr')['correction_options']
        with tempfile.TemporaryDirectory() as folder:
            project = take(folder, manifest=dict(correction=cli, pipeline=dict(model='exmr')))
            self.window.restore_correction_options(json.loads((project / 'project.json').read_text(encoding='utf-8'))['correction'])
            self.window.project = project
            stored = (project / 'project.json').read_bytes()
            self.window.save_correction_options()
            self.assertEqual(json.loads((project / 'project.json').read_bytes()), json.loads(stored))
            state = json.loads((project / 'project.json').read_text(encoding='utf-8'))
            self.assertEqual(effective_options(project, state), cli)
            # A later CLI run without --grounding keeps what the GUI saved.
            self.assertEqual(build_settings(SimpleNamespace(), 'exmr', saved_settings(project, 'exmr'))['correction_options']['grounding'], 'camera')
            self.window.correction_grounding.setCurrentIndex(self.window.correction_grounding.findData('vertical'))
            state = json.loads((project / 'project.json').read_text(encoding='utf-8'))
            self.assertNotIn('grounding', state['correction'])
            self.assertEqual(build_settings(SimpleNamespace(), 'exmr', saved_settings(project, 'exmr'))['correction_options'], state['correction'])

    def test_multi_level_disables_and_moving_camera_explains_in_the_tooltip(self):
        with tempfile.TemporaryDirectory() as folder:
            self.window.project = take(folder, static=False)
            self.window.restore_correction_options(dict(LEGACY, grounding='camera'))
            self.assertIn('Moving camera', self.window.correction_grounding.toolTip())
            self.window.update_readiness()
            self.assertTrue(self.window.correction_grounding.isEnabled())
            self.window.restore_correction_options(dict(LEGACY, mode='multi_level', grounding='camera'))
            self.window.update_readiness()
            self.assertFalse(self.window.correction_grounding.isEnabled())
            self.assertEqual(self.window.correction_grounding.currentData(), 'camera')


@unittest.skipUnless(os.environ.get('RGBMOCAP_GROUNDING_PROJECT'), 'set RGBMOCAP_GROUNDING_PROJECT to a copied project')
class GroundingProjectTest(unittest.TestCase):
    def wait(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return True
            time.sleep(.02)
        return False

    def test_camera_corrected_project_shows_and_keeps_camera(self):
        from mocap.gui import MainWindow
        from mocap.motion_tools import result_is_current
        project = Path(os.environ['RGBMOCAP_GROUNDING_PROJECT'])
        stored = (project / 'project.json').read_bytes()
        self.assertEqual(json.loads(stored)['correction'].get('grounding'), 'camera')
        window = MainWindow()
        try:
            window.show()
            window.load_project(project)
            self.assertTrue(self.wait(lambda: window.project is not None and not window.busy), window.status.text())
            self.assertEqual(window.correction_grounding.currentData(), 'camera')
            self.assertEqual(window.grounding_note.text(), '')
            window.save_correction_options()
            self.assertEqual((project / 'project.json').read_bytes(), stored)
            self.assertTrue(result_is_current(project, 'corrected_initial'))
        finally:
            window.shutdown_preview()
            window.deleteLater()
            APP.processEvents()


if __name__ == '__main__':
    unittest.main()
