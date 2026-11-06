"""Main-window wiring for self-contact rows, Suggest Contacts and the original-vs-corrected comparison.

Store reads and writes run on one worker thread; suggestions run the CLI in a lock-free QProcess.
"""
import json
from collections import deque
from pathlib import Path
from threading import Condition

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QThread, QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QPushButton, QSpinBox

from .tooltips import tooltip_text

TIMELINE_HELP = (
    "Self-contact rows (e.g. RH→RF: right hand holds right foot), one per pair: edit intervals like locks.\n"
    "Hatched intervals are suggestions: double-click or right-click to accept them.\n"
    "Right-click: add a pair (+ Pair row), pair settings (distance, reciprocal, grip, ease), disable, reset, remove.\n\n"
    "L/R pose: foot articulation per interval (Match 2D, Pointe, Flat, Flexed).\n"
    "Drag to add an interval with the row's current mode; right-click to change mode, strength or limits (dancer allows full pointe).\n"
    "L/R arm 2D, L/R leg 2D: limb placement, fit the limb to the video's 2D keypoints (ExMR); right-click for strength.\n"
    "Smooth: calm rotation jitter, applied last; right-click for joints (a group or bones) and window frames (keep it small).\n"
    "Overlapping intervals (different joints) stack in Smooth 2, 3...; drag in the + Smooth lane to add one.\n\n"
    "Δ orig: largest joint change per frame against the original motion (Show Original in the 3D view).\n"
    "Edits save at once and share Ctrl+Z / Ctrl+Y with the locks; Apply Correction applies them to the motion.")


def snapshot(project, base, kind):
    from .foot_contacts import contact_revision
    report_path = Path(project) / f'{kind}_correction.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
    revision = contact_revision(project, base)
    return dict(contact_revision=revision, applied=report.get('contact_revision') == revision)


class SelfContactIO(QThread):
    """Sequential store access; reads coalesce to the latest request."""
    done = Signal(int, object)
    failed = Signal(int, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition = Condition()
        self.jobs = deque()
        self.stopping = False
        self.sequence = 0

    def request(self, job):
        with self.condition:
            self.sequence += 1
            if job['action'] == 'read':
                self.jobs = deque(j for j in self.jobs if j['action'] != 'read')
            self.jobs.append(dict(job, token=self.sequence))
            self.condition.notify()
            return self.sequence

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()

    def run(self):
        from . import self_contact_store as store
        from .pose_rows import read_sections, write_section
        while True:
            with self.condition:
                while not self.jobs and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                job = self.jobs.popleft()
            try:
                project, base, context = job['project'], job['base'], job['context']
                if job['action'] == 'save' and job.get('section'):
                    write_section(project, base, job['section'], job['entries'], context)
                elif job['action'] == 'save':
                    if job['tracks']:
                        store.save(project, base, job['tracks'], context)
                    else:
                        store.clear(project, base)
                tracks, revision, warning = store.read(project, base, context)
                self.done.emit(job['token'], dict(job, tracks=tracks, revision=revision, warning=warning,
                                                  sections=read_sections(project, base, context), **snapshot(project, base, job['kind'])))
            except Exception as error:
                self.failed.emit(job['token'], str(error), job)


class SelfContactController(QObject):
    def __init__(self, window):
        super().__init__(window)
        from .compare_overlay import CompareOverlay
        self.w = w = window
        self.edit_seq = 0
        self.save_token = None
        self.suggest_process = None
        self.warning = ''
        self.loaded_key = None
        self.io = SelfContactIO(self)
        self.io.done.connect(self.io_done)
        self.io.failed.connect(self.io_failed)
        self.io.start()
        self.overlay = CompareOverlay(w)
        w.motion_view.compare_overlay = self.overlay
        slider = w.slider
        slider.sc_listener = self.save
        slider.pose_listener = self.save_section
        slider.sc_actions = {'suggest': self.suggest}
        help_dialog = getattr(w, 'timeline_help_dialog', None)
        if help_dialog is not None:
            help_dialog.setText(help_dialog.text() + '\n\n' + TIMELINE_HELP)
        slider.sc_status = w.status.setText
        slider.sc_resized = lambda: QTimer.singleShot(0, self.fit_timeline)
        self.suggest_button = QPushButton('Suggest Contacts')
        self.suggest_button.setToolTip(tooltip_text('Find frames where a hand probably touches the body (review missed-contact check: touching in the video '
                                                    'but apart in 3D) or rests sunk inside it (sustained, slow penetration of the body mesh), '
                                                    'and show them as hatched suggestions on the self-contact rows. Accept them on the '
                                                    'timeline, then Apply Correction. Does not run AI models.'))
        self.suggest_button.clicked.connect(self.suggest)
        actions = w.reset_footlocks_button.parentWidget().layout()
        actions.insertWidget(actions.indexOf(w.reset_footlocks_button) + 1, self.suggest_button)
        header = w.bone_overlay.parentWidget().layout()
        self.show_original = QCheckBox('Show Original')
        self.show_original.setToolTip(tooltip_text('Draw the original motion in translucent blue with the corrected body: the pose before '
                                                   'self-contacts, or the uncorrected motion when there are none. Display only.'))
        self.show_original.toggled.connect(self.original_changed)
        self.opacity = QSpinBox()
        self.opacity.setRange(10, 90)
        self.opacity.setSingleStep(5)
        self.opacity.setValue(35)
        self.opacity.setSuffix('%')
        self.opacity.setToolTip('Opacity of the original layer.')
        self.opacity.valueChanged.connect(self.original_changed)
        position = header.indexOf(w.bone_overlay) + 1
        header.insertWidget(position, self.show_original)
        header.insertWidget(position + 1, self.opacity)

    # State ------------------------------------------------------------------------------------------------------------

    def context(self):
        """(project, base, kind, info) of the edited result, or None."""
        from .motion_tools import base_result
        w = self.w
        if w.project is None:
            return None
        base = base_result(w.active_result)
        info = w.contact_tracks.get(base)
        if info and w.info and (info['context']['frames'] != w.info['frames'] or info['context']['fps'] != w.info['fps']):
            info = None
        return (w.project, base, 'corrected_' + base, info) if info else None

    def suggesting(self):
        return self.suggest_process is not None

    def editable(self):
        return self.context() is not None and not self.suggesting() and not self.warning and self.save_token is None

    def sync(self):
        w = self.w
        slider = w.slider
        slider.set_self_contacts_editable(self.editable())
        exmr = w.model_choice.currentData() == 'exmr'
        ready = self.context() is not None and not w.busy and not w.contact_saving and not self.suggesting()
        self.suggest_button.setEnabled(ready and exmr)
        self.suggest_button.setText('Suggesting...' if self.suggesting() else 'Suggest Contacts')
        if not exmr:
            self.suggest_button.setToolTip(tooltip_text('Suggestions need ExMR: the missed-contact check reads its 2D keypoints.'))
        available = self.overlay.available() or bool(self.overlay.message)
        self.show_original.setEnabled(available or self.show_original.isChecked())
        self.opacity.setEnabled(self.show_original.isChecked())

    def fit_timeline(self):
        """A splitter keeps its old sizes when a child's minimum grows; give the timeline room for its new rows."""
        split, panel = getattr(self.w, 'workspace_split', None), getattr(self.w, 'timeline_panel', None)
        if split is None or panel is None or split.indexOf(panel) != 1:
            return
        sizes = split.sizes()
        need = panel.minimumSizeHint().height()
        if len(sizes) == 2 and sizes[1] < need:
            split.setSizes([max(0, sum(sizes) - need), need])

    def original_changed(self, *args):
        self.overlay.show = self.show_original.isChecked()
        self.overlay.opacity = self.opacity.value() / 100
        self.opacity.setEnabled(self.overlay.show)
        self.w.motion_view.update()
        if hasattr(self.w, 'camera_tools'):
            self.w.camera_tools.canvas.mirror()

    def refresh(self):
        """Reload rows and the comparison for the displayed result (called whenever foot/hand rows are refreshed)."""
        w = self.w
        current = self.context()
        if current is None:
            self.loaded_key = None
            w.slider.set_self_contacts(None)
            self.overlay.applied = False
        else:
            self.request_read()
        self.overlay.request(w.project, w.active_result, w.preview_data) if w.project is not None else self.overlay.clear()
        self.sync()

    def request_read(self):
        current = self.context()
        if current is None:
            return
        project, base, kind, info = current
        self.io.request(dict(action='read', project=project, base=base, kind=kind, context=info['context'], edit_seq=self.edit_seq))

    # Store results ----------------------------------------------------------------------------------------------------

    def matches(self, job):
        current = self.context()
        return current is not None and (str(current[0]), current[1]) == (str(job['project']), job['base'])

    def io_done(self, token, result):
        w = self.w
        saving = result['action'] == 'save'
        if saving and token == self.save_token:
            self.save_token = None
            w.contact_saving = False
            w.status.setText(self.saved_noun(result).capitalize() + ' saved. Apply Correction to update the motion and FBX.')
        elif saving:
            return
        elif result['edit_seq'] != self.edit_seq or self.save_token is not None:
            return  # A newer local edit owns the rows.
        if not self.matches(result):
            w.update_readiness()
            return
        project, base, kind, info = self.context()
        self.warning = result['warning']
        self.overlay.applied = result['applied']
        frames = int(info['context']['frames'])
        w.slider.set_self_contacts(result['tracks'], frames, (str(project), base), self.editable(), self.warning)
        w.slider.set_pose_sections(result.get('sections'), frames, (str(project), base))
        if info.get('revision') != result['contact_revision']:
            # The stale/pending notices compare this revision with the applied correction.
            w.contact_tracks[base] = {**info, 'revision': result['contact_revision']}
            if w.preview_data is not None:
                w.apply_preview()
        w.update_readiness()

    def io_failed(self, token, message, job):
        w = self.w
        if job['action'] == 'save' and token == self.save_token:
            self.save_token = None
            w.contact_saving = False
            w.status.setText(f'Cannot save {self.saved_noun(job)}: {message}')
            self.edit_seq += 1
            self.request_read()
        elif job['action'] == 'read' and self.matches(job):
            self.warning = 'Self-contacts cannot be read: ' + message
            w.slider.set_self_contacts([], 1, None, False, self.warning)
            w.slider.set_pose_sections(None)
        w.update_readiness()

    @staticmethod
    def saved_noun(job):
        from .pose_rows import SECTIONS
        key = job.get('section')
        return next((s.noun + 's' for s in SECTIONS if s.key == key), 'self-contacts')

    def save(self, tracks):
        self.request_save(tracks=tracks)

    def save_section(self, key, entries):
        """Pose rows: one section of the same store payload, through the same save queue and pending state."""
        self.request_save(section=key, entries=entries)

    def request_save(self, **fields):
        w = self.w
        current = self.context()
        if current is None or w.busy or w.contact_saving or self.suggesting():
            self.request_read()
            return
        project, base, kind, info = current
        w.stop_playback()
        self.edit_seq += 1
        w.contact_saving = True
        self.save_token = self.io.request(dict(action='save', project=project, base=base, kind=kind, context=info['context'],
                                               edit_seq=self.edit_seq, **fields))
        w.status.setText(f'Saving {self.saved_noun(fields)}...')
        w.update_readiness()

    # Suggestions --------------------------------------------------------------------------------------------------------

    def suggest_command(self, project):
        from .paths import ROOT
        python = ROOT / 'runtime' / 'python.exe'
        if not python.is_file():
            import sys
            python = Path(sys.executable)
        # `contacts suggest` without --accept takes no app lock, so it runs beside the open GUI.
        return str(python), ['-I', '-X', 'utf8', '-u', '-B', str(ROOT / 'cli.py'), 'contacts', 'suggest', '--project', str(project), '--json']

    def suggest(self):
        w = self.w
        current = self.context()
        if current is None or w.busy or w.contact_saving or self.suggesting():
            return
        program, args = self.suggest_command(current[0])
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUTF8', '1')
        environment.insert('PYTHONIOENCODING', 'utf-8')
        process.setProcessEnvironment(environment)
        process.setProperty('project', str(current[0]))
        process.finished.connect(lambda code, status, p=process: self.suggest_finished(p, code))
        process.errorOccurred.connect(lambda error, p=process: self.suggest_finished(p, -1) if error == QProcess.FailedToStart else None)
        self.suggest_process = process
        w.stop_playback()
        w.status.setText('Suggesting self-contacts from the review missed-contact and sunk-hand checks...')
        process.start(program, args)
        self.sync()

    def suggest_finished(self, process, code):
        if process is not self.suggest_process:
            return
        self.suggest_process = None
        output = bytes(process.readAllStandardOutput()).decode('utf-8', 'replace')
        errors = bytes(process.readAllStandardError()).decode('utf-8', 'replace').strip().splitlines()
        process.deleteLater()
        summary = None
        lines = output.strip().splitlines()
        # The --json summary may be indented over several lines; parse from the last top-level opening brace.
        for first in reversed([i for i, line in enumerate(lines) if line.startswith('{')]):
            try:
                value = json.loads('\n'.join(lines[first:]))
            except ValueError:
                continue
            if isinstance(value, dict):
                summary = value
                break
        if code == 0 and summary is not None:
            found = summary.get('issues', [])
            issues, sunk = len(found), sum(i.get('check') == 'sunk_hand' for i in found)
            self.w.status.setText(f'{issues} suggested self-contact interval(s). Accept them on the hatched timeline rows, then Apply Correction.'
                                  if issues else 'No missed or sunk hand contacts found.')
            self.w.log.append(f'Suggest Contacts: {issues - sunk} missed contact(s) and {sunk} hand(s) resting inside the body '
                              f'from {summary.get("reviewed", "the original motion")}.')
        else:
            message = (summary or {}).get('error') or (errors[-1] if errors else f'exit code {code}')
            self.w.status.setText('Suggest Contacts failed: ' + message)
        self.edit_seq += 1
        self.request_read()
        self.sync()

    def close(self):
        if self.suggest_process is not None:
            process, self.suggest_process = self.suggest_process, None
            process.kill()
            process.waitForFinished(2000)
        self.io.stop()
        self.io.wait()
        self.overlay.release_renderer(self.w.motion_view)
        self.overlay.close()
