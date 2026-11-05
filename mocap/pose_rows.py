"""Pose rows: per-interval pose modes (foot articulation, limb placement, smoothing) below the self-contact rows.

A section is one entry list in the self-contact store payload. Entries group the intervals that share a mode and settings;
one timeline row shows one value of the section's row field (e.g. one foot), and every interval in it carries the settings
of its entry. Frames are zero-based and end-exclusive. A new section plugs in with one `Section` in SECTIONS.
"""
import copy
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    color: str
    tip: str = ''


@dataclass(frozen=True)
class Param:
    """One per-interval setting. kind: float, int, choice, or list (comma-separated names, e.g. bones, or one of presets)."""
    name: str
    label: str
    kind: str = 'float'
    choices: tuple = ()
    low: float = 0.
    high: float = 1.
    step: float = .05
    decimals: int = 2
    percent: bool = False
    presets: tuple = ()
    suffix: str = ''

    def text(self, value):
        if self.kind == 'list':
            return value if isinstance(value, str) else ','.join(value)
        if self.percent:
            return f'{float(value) * 100:.0f}%'
        return f'{value}{self.suffix}'


@dataclass(frozen=True)
class Section:
    key: str            # payload list in the store
    noun: str           # 'foot pose'
    row_field: str      # entry field choosing the row; None puts every entry in one row
    rows: tuple         # (field value, row label) per timeline row
    modes: tuple
    params: tuple
    defaults: dict = field(default_factory=dict)
    reader: str = ''    # self_contact_store functions: read(project, base, context) -> (entries, revision, warning)
    writer: str = ''    # save(project, base, entries, context)
    normalizer: str = ''  # normalize(entries, frames) -> validated entries with ids
    mode_field: str = 'mode'
    # Single-row section whose intervals may overlap (the store's normalizer decides which): they stack in lanes, plus an empty
    # lane to draw an overlapping interval in.
    stack: bool = False

    def setting_keys(self):
        return ((self.mode_field,) if self.mode_field else ()) + tuple(p.name for p in self.params) + ('enabled',)

    def mode(self, name):
        return next((m for m in self.modes if m.name == name), self.modes[0])

    def store_function(self, name):
        from . import self_contact_store as store
        return getattr(store, name, None) if name else None


SECTIONS = (
    Section('articulation', 'foot pose', 'foot', (('LeftFoot', 'L pose'), ('RightFoot', 'R pose')),
            (Mode('match', 'Match 2D', '#7fb0d8', 'Fit ankle and toes to the video heel-toe direction; depth from the model'),
             Mode('pointe', 'Pointe', '#d98fb8', 'Full plantarflexion, toes in line with the shin; support on the toe tip'),
             Mode('flat', 'Flat', '#9fbf6a', 'Sole level with the floor'),
             Mode('flexed', 'Flexed', '#d9b25f', 'Ankle dorsiflexed, toes extended')),
            (Param('strength', 'Strength', percent=True, presets=(1., .75, .5, .25)),
             Param('profile', 'Limits', 'choice', ('default', 'dancer'))),
            dict(mode='match', strength=1., profile='default', enabled=True),
            'read_articulation', 'save_articulation', 'normalize_articulation'),
    Section('placement', 'limb placement', 'chain',
            (('LeftArm', 'L arm 2D'), ('RightArm', 'R arm 2D'), ('LeftLeg', 'L leg 2D'), ('RightLeg', 'R leg 2D')),
            (Mode('place', 'Match 2D', '#e0a070', 'Fit the arm or leg to the video\'s 2D keypoints (ExMR); depth stays near the source, '
                                                   'the hand or foot keeps its world rotation'),),
            (Param('strength', 'Strength', percent=True, presets=(1., .75, .5, .25)),),
            dict(strength=1., enabled=True), 'read_placement', 'save_placement', 'normalize_placement', mode_field=None),
    Section('smoothing', 'pose smoothing', None, ((None, 'Smooth'),),
            (Mode('smooth', 'Smooth', '#8fc7c0', 'Calm rotation jitter of the chosen joints; applied last, after the self-contacts'),),
            (Param('joints', 'Joints', 'list', presets=('all-written', 'arms', 'legs', 'spine')),
             Param('window_frames', 'Window', 'int', low=3, high=61, presets=(3, 5, 9, 15), suffix='f')),
            dict(joints='all-written', window_frames=5, enabled=True), 'read_smoothing', 'save_smoothing', 'normalize_smoothing',
            mode_field=None, stack=True),
)


def read_sections(project, base, context):
    """{section key: dict(entries, warning)} for every section the store supports; unreadable sections come back empty."""
    result = {}
    for section in SECTIONS:
        reader = section.store_function(section.reader)
        if reader is None:
            continue
        try:
            entries, _, warning = reader(project, base, context)
        except (OSError, ValueError, KeyError, TypeError) as error:
            entries, warning = [], f'Saved {section.noun}s cannot be read: {error}'
        result[section.key] = dict(entries=entries, warning=warning)
    return result


def write_section(project, base, key, entries, context):
    section = next(s for s in SECTIONS if s.key == key)
    section.store_function(section.writer)(project, base, entries, context)


def _key(values):
    return json.dumps(values, sort_keys=True)


class PoseRows:
    """Entries of one section with snapshot undo; every committed state passes the store's normalizer.

    Stricter than the store: no two intervals of one row overlap, enabled or not, so a row reads as one interval list. A stacked
    section is one row shown as lanes instead: overlapping intervals go to the first lane they fit in, earliest start first."""

    LIMIT = 100

    def __init__(self, section):
        self.section = section
        self.entries = []
        self.frames = 0
        self.identity = None
        self.draw = {}
        self._undo = []
        self._redo = []
        self._rows = None

    # State ------------------------------------------------------------------------------------------------------------

    def _set(self, entries):
        self.entries = entries
        self._rows = None

    def _normalize(self, entries):
        normalizer = self.section.store_function(self.section.normalizer)
        entries = normalizer(entries, self.frames) if normalizer is not None else entries
        return [{**copy.deepcopy(e), 'ranges': [tuple(p) for p in e['ranges']], 'automatic': [tuple(p) for p in e.get('automatic', [])]}
                for e in entries]

    def load(self, entries, frames, identity=None):
        """Replace the entries from storage; history survives only an unchanged reload of the same result. True if cleared."""
        entries = [{**copy.deepcopy(dict(e)), 'ranges': [tuple(p) for p in e.get('ranges', [])],
                    'automatic': [tuple(p) for p in e.get('automatic', [])]} for e in entries or []]
        same = identity == self.identity and frames == self.frames and entries == self.entries
        self.identity, self.frames = identity, frames
        if same:
            return False
        self._set(entries)
        self._undo.clear()
        self._redo.clear()
        return True

    def clear(self):
        self.load([], 0, None)

    def can_undo(self):
        return bool(self._undo)

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self.entries))
        self._set(self._undo.pop())
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self.entries))
        self._set(self._redo.pop())
        return True

    def _commit(self, entries):
        if self.frames < 1:
            raise ValueError('No motion to attach poses to')
        from .selfcontact_rows import clamp
        for entry in entries:
            entry['ranges'] = clamp(entry['ranges'], self.frames)
        entries = [e for e in entries if e['ranges'] or e.get('automatic')]
        entries = self._normalize(entries)
        for r in range(0 if self.section.stack else len(self.section.rows)):
            pieces = sorted((s, e) for entry in self._row_entries(entries, r) for s, e in entry['ranges'])
            for (s0, e0), (s1, e1) in zip(pieces, pieces[1:]):
                if s1 < e0:
                    raise ValueError(f'{self.row_label(r)} already has an interval on frames {s1 + 1}-{min(e0, e1)}')
        self._undo.append(copy.deepcopy(self.entries))
        del self._undo[:-self.LIMIT]
        self._redo.clear()
        self._set(entries)

    def payload(self):
        return [{**e, 'ranges': [list(p) for p in e['ranges']], 'automatic': [list(p) for p in e.get('automatic', [])]}
                for e in self.entries]

    # Rows -------------------------------------------------------------------------------------------------------------

    def _layout(self):
        """Per timeline row, sorted (start, end, entry index); a stacked section's rows are its lanes."""
        if self._rows is None:
            if not self.section.stack:
                self._rows = [sorted((s, e, i) for i, entry in enumerate(self.entries) if self._in_row(entry, row) for s, e in entry['ranges'])
                              for row in range(len(self.section.rows))]
            else:
                lanes = []
                for piece in sorted((s, e, i) for i, entry in enumerate(self.entries) for s, e in entry['ranges']):
                    lane = next((lane for lane in lanes if lane[-1][1] <= piece[0]), None)
                    if lane is None:
                        lanes.append([piece])
                    else:
                        lane.append(piece)
                self._rows = lanes + [[]]
        return self._rows

    def row_count(self):
        return len(self._layout())

    def row_label(self, r):
        label = self.section.rows[0 if self.section.stack else r][1]
        if not self.section.stack or r == 0:
            return label
        return f'{label} {r + 1}' if self.intervals(r) else '+ ' + label

    def is_spare(self, r):
        """The empty lane below a stacked section's intervals."""
        return self.section.stack and r > 0 and not self.intervals(r)

    def _in_row(self, entry, r):
        name = self.section.row_field
        return name is None or entry.get(name) == self.section.rows[r][0]

    def _row_entries(self, entries, r):
        return [e for e in entries if self._in_row(e, r)]

    def intervals(self, r):
        """Sorted (start, end, entry index) of one row."""
        layout = self._layout()
        return layout[r] if r < len(layout) else []

    def ranges(self, r):
        return [(s, e) for s, e, _ in self.intervals(r)]

    def _taken(self, r):
        """Frames suggestions of row r may not cover: the row, or every lane of a stacked section."""
        from .selfcontact_rows import merge
        return merge(p for lane in range(self.row_count()) for p in self.ranges(lane)) if self.section.stack else self.ranges(r)

    def locate(self, piece):
        """(row, position) of the interval equal to piece, else of the first one holding its first frame, else None."""
        found = None
        for r in range(self.row_count()):
            for position, (s, e, _) in enumerate(self.intervals(r)):
                if (s, e) == tuple(piece):
                    return r, position
                if found is None and s <= piece[0] < e:
                    found = r, position
        return found

    def settings(self, entry):
        return {k: entry.get(k, self.section.defaults.get(k)) for k in self.section.setting_keys()}

    def interval_settings(self, r, position):
        return self.settings(self.entries[self.intervals(r)[position][2]])

    def draw_settings(self, r):
        return dict(self.draw.get(r) or {k: self.section.defaults.get(k) for k in self.section.setting_keys()})

    def describe(self, settings):
        """Interval label: mode plus every setting that differs from the section default."""
        section = self.section
        mode = section.mode(settings.get(section.mode_field) if section.mode_field else None)
        parts = [mode.label] + [p.text(settings[p.name]) for p in section.params if settings.get(p.name) != section.defaults.get(p.name)]
        return ' '.join(parts) + ('' if settings.get('enabled', True) else ' (off)')

    def _place(self, entries, r, piece, settings):
        """Add piece to the row's entry with these settings, creating the entry if none matches."""
        wanted = _key(settings)
        for entry in self._row_entries(entries, r):
            if _key(self.settings(entry)) == wanted:
                entry['ranges'] = list(entry['ranges']) + [tuple(piece)]
                return
        entry = dict(id=None, **settings, ranges=[tuple(piece)], automatic=[], source='user')
        if self.section.row_field:
            entry[self.section.row_field] = self.section.rows[r][0]
        entries.append(entry)

    # Edits ------------------------------------------------------------------------------------------------------------

    def set_interval(self, r, position, before, after):
        """Insert (before None, using the row's draw settings), delete (after None) or replace interval number position."""
        entries = copy.deepcopy(self.entries)
        if before is None:
            self._place(entries, r, after, self.draw_settings(r))
        else:
            s, e, i = self.intervals(r)[position]
            ranges = list(entries[i]['ranges'])
            ranges.remove((s, e))
            if after is not None:
                ranges.append(tuple(after))
            entries[i]['ranges'] = ranges
        self._commit(entries)

    def set_settings(self, r, position, **changes):
        """Move one interval to the entry holding its new settings; the row's draw settings follow."""
        unknown = set(changes) - set(self.section.setting_keys())
        if unknown:
            raise ValueError(f'Unknown {self.section.noun} setting: ' + ', '.join(sorted(unknown)))
        entries = copy.deepcopy(self.entries)
        s, e, i = self.intervals(r)[position]
        settings = {**self.settings(entries[i]), **changes}
        entries[i]['ranges'] = [p for p in entries[i]['ranges'] if p != (s, e)]
        self._place(entries, r, (s, e), settings)
        self._commit(entries)
        self.draw[r] = {**settings, 'enabled': True}

    def set_draw_settings(self, r, **changes):
        self.draw[r] = {**self.draw_settings(r), **changes}

    def suggestions(self, r):
        """(start, end, entry index) automatic intervals not covered by the row."""
        from .selfcontact_rows import subtract
        if self.section.stack and r:
            return []  # A stacked section shows its suggestions in the first lane.
        taken = self._taken(r)
        return sorted((s, e, i) for i, entry in enumerate(self.entries) if self._in_row(entry, r)
                      for s, e in subtract(entry.get('automatic', []), taken))

    def accept(self, r, interval=None):
        """Copy suggestions (all, or those inside interval) into their entries; returns accepted frames."""
        from .selfcontact_rows import merge, subtract
        if self.section.stack and r:
            return 0
        entries = copy.deepcopy(self.entries)
        taken, total = self._taken(r), 0
        for i, entry in enumerate(entries):
            if not self._in_row(entry, r):
                continue
            pieces = subtract(entry.get('automatic', []), taken)
            if interval is not None:
                pieces = [(max(s, interval[0]), min(e, interval[1])) for s, e in pieces if s < interval[1] and interval[0] < e]
            if pieces:
                entry['ranges'] = list(entry['ranges']) + pieces
                taken = merge(taken + pieces)
                total += sum(e - s for s, e in pieces)
        if total:
            self._commit(entries)
        return total

    def reset_row(self, r):
        """Ranges become the automatic suggestions; earlier entries win where suggestions of two modes overlap."""
        from .selfcontact_rows import merge, subtract
        entries, taken = copy.deepcopy(self.entries), []
        for entry in entries:
            if self._in_row(entry, r):
                entry['ranges'] = subtract(entry.get('automatic', []), taken)
                taken = merge(taken + entry['ranges'])
        self._commit(entries)

    def clear_row(self, r):
        """Remove the row's intervals; in a stacked section only those of lane r."""
        entries = copy.deepcopy(self.entries)
        if self.section.stack:
            for s, e, i in self.intervals(r):
                entries[i]['ranges'] = [p for p in entries[i]['ranges'] if p != (s, e)]
        else:
            for entry in self._row_entries(entries, r):
                entry['ranges'] = []
        self._commit(entries)

    def has_automatic(self, r):
        if self.section.stack and r:
            return False
        return any(entry.get('automatic') for entry in self._row_entries(self.entries, r))


class SettingsDialog:
    """Mode, parameters and enabled flag of one pose interval (or of the row's new intervals)."""

    @staticmethod
    def ask(parent, section, settings, title):
        from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLineEdit,
                                       QSpinBox)
        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        form = QFormLayout(dialog)
        readers = {}
        if section.mode_field:
            mode = QComboBox()
            for item in section.modes:
                mode.addItem(item.label, item.name)
                mode.setItemData(mode.count() - 1, item.tip, 3)
            mode.setCurrentIndex(max(0, mode.findData(settings.get(section.mode_field))))
            form.addRow('Mode', mode)
            readers[section.mode_field] = mode.currentData
        for param in section.params:
            value = settings.get(param.name, section.defaults.get(param.name))
            if param.kind == 'choice':
                widget = QComboBox()
                for choice in param.choices:
                    widget.addItem(str(choice), choice)
                widget.setCurrentIndex(max(0, widget.findData(value)))
                readers[param.name] = widget.currentData
            elif param.kind == 'list' and param.presets:
                # A preset name (e.g. a bone group) or comma-separated names typed in.
                widget = QComboBox()
                widget.setEditable(True)
                widget.addItems(list(param.presets))
                widget.setCurrentText(param.text(value or ()))
                readers[param.name] = lambda w=widget, p=param: (w.currentText().strip() if w.currentText().strip() in p.presets
                                                                 else [v.strip() for v in w.currentText().split(',') if v.strip()])
            elif param.kind == 'list':
                widget = QLineEdit(param.text(value or ()))
                readers[param.name] = lambda w=widget: [v.strip() for v in w.text().split(',') if v.strip()]
            elif param.kind == 'int':
                widget = QSpinBox()
                widget.setRange(int(param.low), int(param.high))
                widget.setValue(int(value))
                readers[param.name] = widget.value
            else:
                widget = QDoubleSpinBox()
                scale = 100 if param.percent else 1
                widget.setRange(param.low * scale, param.high * scale)
                widget.setDecimals(0 if param.percent else param.decimals)
                widget.setSingleStep(param.step * scale)
                widget.setSuffix('%' if param.percent else '')
                widget.setValue(float(value) * scale)
                readers[param.name] = lambda w=widget, k=scale: round(w.value() / k, 4)
            form.addRow(param.label, widget)
        enabled = QCheckBox('Apply this interval')
        enabled.setChecked(bool(settings.get('enabled', True)))
        form.addRow(enabled)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        try:
            if dialog.exec() != QDialog.Accepted:
                return None
            return {**{name: read() for name, read in readers.items()}, 'enabled': enabled.isChecked()}
        finally:
            dialog.deleteLater()


class PoseRowEditor:
    """Timeline mixin for pose rows; track index t = pose_first_row() + n addresses pose row n, after the '+ Pair' row.

    The host (SelfContactEditor) provides placeholder_row(), sc_visible/sc_editable, _update_self_height() and the shared undo
    plumbing."""

    def init_pose_rows(self):
        self.pose_models = [PoseRows(section) for section in SECTIONS]
        self.pose_messages = {}
        # Window hook: pose_listener(section key, payload) saves one section.
        self.pose_listener = None

    def pose_first_row(self):
        return self.placeholder_row() + 1

    def pose_rows_total(self):
        return sum(model.row_count() for model in self.pose_models)

    def pose_row(self, t):
        """(model, row) of track index t, or None."""
        n = t - self.pose_first_row()
        if n < 0:
            return None
        for model in self.pose_models:
            if n < model.row_count():
                return model, n
            n -= model.row_count()
        return None

    def pose_row_models(self):
        return [(model, r) for model in self.pose_models for r in range(model.row_count())]

    def pose_track(self, model, r):
        """Track index t of row r of model."""
        return self.pose_first_row() + self.pose_row_models().index((model, r))

    def _select_piece(self, model, piece):
        """Follow an edited interval of a stacked section, whose lanes are laid out again after every edit."""
        found = model.locate(piece) if model.section.stack else None
        if found is not None:
            self._selected_contact = (self.pose_track(model, found[0]), found[1])

    def pose_writable(self, model):
        return self.sc_editable and not self.pose_messages.get(model.section.key)

    def set_pose_sections(self, sections, frames=0, identity=None):
        """sections: read_sections() output; None or a missing key empties the section."""
        for model in self.pose_models:
            info = (sections or {}).get(model.section.key)
            self.pose_messages[model.section.key] = info['warning'] if info else ''
            if model.load(info['entries'] if info else [], frames if info else 0, identity if info else None):
                self._drop_owner_history(model)
        if self._selected_contact is not None and self.pose_row(self._selected_contact[0]) is not None:
            self._selected_contact = None
        if self._foot_drag and self.pose_row(self._foot_drag['foot']) is not None:
            self._foot_drag = None
            self._foot_draft = None
        self._index_contacts()
        self._update_self_height()

    def pose_command(self, t, index, before, after):
        model, r = self.pose_row(t)
        self._selected_contact = (t, index) if after is not None else None
        if self._model_apply(model, model.set_interval, r, index, before, after) is None:
            self._selected_contact = None
            self._index_contacts()
        elif after is not None:
            self._select_piece(model, after)

    def _pose_suggestion(self, t, frame):
        model, r = self.pose_row(t)
        return next(((s, e) for s, e, _ in model.suggestions(r) if s <= frame < e), None)

    def _pose_accept(self, t, piece):
        model, r = self.pose_row(t)
        return self._model_apply(model, model.accept, r, piece)

    # Menu -------------------------------------------------------------------------------------------------------------

    def pose_menu(self, event):
        from PySide6.QtWidgets import QMenu
        row = self._event_row(event)
        found = self.pose_row(row) if self.sc_visible else None
        if found is None:
            return False
        model, r = found
        section = model.section
        position = event.position() if hasattr(event, 'position') else event.pos()
        frame = min(self.maximum(), self._edge_position(event)) if position.x() >= self.LEFT else None
        editable = self.contact_edit_enabled and self.pose_writable(model)
        index, _ = self._contact_hit(row, event)
        self._selected_contact = (row, index) if index is not None else None
        self.update()
        current = model.interval_settings(r, index) if index is not None else model.draw_settings(r)
        menu = QMenu(self)
        heading = f'{model.row_label(r)}: ' + (f'{model.describe(current)}, frames {model.intervals(r)[index][0] + 1}-'
                                               f'{model.intervals(r)[index][1]}' if index is not None else 'new intervals ' + model.describe(current))
        menu.addAction(heading).setEnabled(False)
        menu.addAction('Add Interval Here', lambda: self.add_contact(row, frame)).setEnabled(editable and frame is not None and index is None)
        menu.addAction('Delete Selected Interval', self.delete_contact).setEnabled(editable and index is not None)
        menu.addSeparator()
        change = lambda **values: self._pose_settings(model, r, index, **values)

        def submenu(title):
            # Parented explicitly: a submenu from QMenu.addMenu(str) can be deleted with its dropped Python wrapper.
            child = QMenu(title, menu)
            menu.addMenu(child)
            return child
        if section.mode_field:
            modes = submenu('Mode')
            for item in section.modes:
                action = modes.addAction(item.label, lambda name=item.name: change(**{section.mode_field: name}))
                action.setCheckable(True)
                action.setChecked(current.get(section.mode_field) == item.name)
                action.setToolTip(item.tip)
            modes.setToolTipsVisible(True)
            modes.setEnabled(editable)
        for param in section.params:
            values = param.choices if param.kind == 'choice' else param.presets
            if not values:
                continue
            sub = submenu(param.label)
            for value in values:
                action = sub.addAction(param.text(value), lambda v=value, name=param.name: change(**{name: v}))
                action.setCheckable(True)
                action.setChecked(current.get(param.name) == value)
            sub.setEnabled(editable)
        menu.addAction('Interval Settings...' if index is not None else 'New Interval Settings...',
                       lambda: self._pose_dialog(model, r, index)).setEnabled(editable)
        if index is not None:
            toggle = menu.addAction('Disable Interval' if current['enabled'] else 'Enable Interval',
                                    lambda: change(enabled=not current['enabled']))
            toggle.setEnabled(editable)
        menu.addSeparator()
        piece = self._pose_suggestion(row, frame) if frame is not None and index is None else None
        menu.addAction('Accept This Suggestion', lambda: self.accept_suggestion_at(row, frame)).setEnabled(editable and piece is not None)
        menu.addAction('Accept All Suggestions in Row', lambda: self._model_apply(model, model.accept, r)).setEnabled(
            editable and bool(model.suggestions(r)))
        menu.addAction('Reset Row to Automatic', lambda: self._model_apply(model, model.reset_row, r)).setEnabled(
            editable and model.has_automatic(r))
        menu.addAction('Clear Row', lambda: self._model_apply(model, model.clear_row, r)).setEnabled(editable and bool(model.ranges(r)))
        menu.addSeparator()
        menu.addAction('Undo', self.undo_contact).setEnabled(bool(self._undo_contacts))
        menu.addAction('Redo', self.redo_contact).setEnabled(bool(self._redo_contacts))
        self.run_menu(menu, event.globalPos())
        menu.deleteLater()
        return True

    def run_menu(self, menu, position):
        """Modal menu; tests replace it to trigger actions without a running event loop."""
        menu.exec(position)

    def _pose_settings(self, model, r, index, **values):
        """Selected interval: an undoable settings change. No interval: only the row's settings for new intervals."""
        if index is None:
            model.set_draw_settings(r, **values)
            self.contact_revision += 1
            self.update()
            return
        piece = model.intervals(r)[index][:2]
        if self._model_apply(model, model.set_settings, r, index, **values) is not None:
            self._select_piece(model, piece)

    def _pose_dialog(self, model, r, index):
        current = model.interval_settings(r, index) if index is not None else model.draw_settings(r)
        title = f'{model.row_label(r)} ' + (f'frames {model.intervals(r)[index][0] + 1}-{model.intervals(r)[index][1]}'
                                            if index is not None else 'new intervals')
        values = SettingsDialog.ask(self, model.section, current, title)
        if values is not None:
            changes = {k: v for k, v in values.items() if current.get(k) != v}
            if changes:
                self._pose_settings(model, r, index, **changes)

    # Painting ---------------------------------------------------------------------------------------------------------

    def paint_pose_rows(self, p):
        from bisect import bisect_right
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QBrush, QColor, QPen
        width = max(1, self.width() - self.LEFT - self.RIGHT)
        first = self.pose_first_row()
        for n, (model, r) in enumerate(self.pose_row_models()):
            t = first + n
            y = 28 + (t + 2) * self.TRACK_HEIGHT
            p.setPen(QColor('#8c8c8c' if model.is_spare(r) else '#aaaaaa'))
            p.drawText(QRectF(8, y, 53, self.TRACK_HEIGHT), Qt.AlignVCenter, model.row_label(r))
            text_rect = QRectF(self.LEFT + 8, y, max(0, width - 11), self.TRACK_HEIGHT)
            message = self.pose_messages.get(model.section.key)
            if not self.sc_visible:
                p.setPen(QColor('#777777'))
                p.drawText(text_rect, Qt.AlignVCenter, f'Apply Correction to edit {model.section.noun}s')
                continue
            intervals = model.intervals(r)
            p.save()
            p.setClipRect(QRectF(self.LEFT, y, width, self.TRACK_HEIGHT))
            for s, e, i in model.suggestions(r):
                if e < self.view_start or s > self.view_start + self.span():
                    continue
                hatch = QColor(model.section.mode(model.entries[i].get(model.section.mode_field)).color)
                hatch.setAlpha(170)
                rect = self._contact_rect(t, (s, e))
                p.fillRect(rect, QBrush(hatch, Qt.BDiagPattern))
                p.setPen(QPen(hatch, 1, Qt.DashLine))
                p.drawRect(rect)
            editable = self._editable(t)
            starts = [s for s, _, _ in intervals]
            ends = [e for _, e, _ in intervals]
            dragged = self._foot_drag['index'] if self._foot_drag and self._foot_drag['foot'] == t and self._foot_drag['original'] else None
            for index in range(bisect_right(ends, self.view_start), bisect_right(starts, self.view_start + self.span())):
                if index == dragged:
                    continue
                s, e, i = intervals[index]
                settings = model.settings(model.entries[i])
                color = QColor(model.section.mode(settings.get(model.section.mode_field)).color if settings['enabled'] else '#7d7d7d')
                tint = QColor(color)
                tint.setAlpha(90 if editable else 40)
                rect = self._contact_rect(t, (s, e))
                p.fillRect(rect, tint)
                p.setPen(QPen(color, 1))
                p.drawRect(rect)
                label = f'{s + 1}–{e}' if self._selected_contact == (t, index) else model.describe(settings)
                if rect.width() > p.fontMetrics().horizontalAdvance(label) + 8:
                    p.drawText(rect, Qt.AlignCenter, label)
            p.restore()
            if message:
                p.setPen(QColor('#e0b06c'))
                p.drawText(text_rect, Qt.AlignVCenter, message)
            elif not intervals:
                p.setPen(QColor('#777777'))
                hint = f'Drag to add {model.describe(model.draw_settings(r))}; right-click for ' + ('modes and settings' if model.section.mode_field
                                                                                                  else 'settings')
                if model.is_spare(r):
                    hint = (f'Drag to add {model.describe(model.draw_settings(r))} overlapping the rows above (other joints); '
                            'right-click for settings')
                p.drawText(text_rect, Qt.AlignVCenter, hint)
