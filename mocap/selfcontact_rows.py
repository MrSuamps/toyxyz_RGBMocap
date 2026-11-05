"""Self-contact timeline rows: K hand-to-body tracks edited below the foot/hand lock rows.

Track index t addresses every timeline contact row: 0-3 are the foot/hand locks, FIRST_ROW + i is self-contact track i.
Frames are zero-based and end-exclusive, as in the foot-lock editor and `self_contact_store`.
"""
import copy

FIRST_ROW = 4
SELF_UNDO = 'self-contact'
EFFECTORS = ('LeftHand', 'RightHand')
RECIPROCAL = ('off', 'auto', 'always')
GRIP = ('off', 'auto')
DEFAULTS = dict(offset='auto', distance='auto', reciprocal='auto', grip='off', enabled=True, source='user')
ROW_COLORS = ('#d9a650', '#62c0a0', '#d77f9e', '#9d97e3', '#bfbf62', '#6fb3d9')
ABBREVIATIONS = {'hips': 'Hip', 'spine': 'Sp', 'spine1': 'Sp1', 'spine2': 'Sp2', 'neck': 'Nk', 'head': 'Hd'}


def merge(ranges):
    """Sorted union; touching intervals merge, as the store does."""
    merged = []
    for start, end in sorted((int(s), int(e)) for s, e in ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def subtract(ranges, removed):
    result = []
    for start, end in ranges:
        pieces = [(start, end)]
        for a, b in removed:
            pieces = [p for s, e in pieces for p in ((s, min(e, a)), (max(s, b), e)) if p[0] < p[1]]
        result += pieces
    return result


def clamp(ranges, frames):
    return merge((max(0, int(s)), min(frames, int(e))) for s, e in ranges if min(frames, int(e)) > max(0, int(s)))


def bone_label(bone):
    from .self_contact_store import short_name
    name = short_name(bone)
    return ABBREVIATIONS.get(name, name.upper())


def row_label(track):
    return bone_label(track['effector']) + '→' + bone_label(track['target'])


def pair_id(effector, target, used):
    from .self_contact_store import short_name
    stem = short_name(effector) + '-' + short_name(target)
    name, count = stem, 2
    while name in used:
        name, count = f'{stem}-{count}', count + 1
    return name


def check_pair(effector, target):
    from .mixamo_rig import BODY_NAMES
    from .self_contact_store import arm_chain
    if effector not in EFFECTORS:
        raise ValueError(f'Effector must be LeftHand or RightHand, not {effector!r}')
    if target not in BODY_NAMES:
        raise ValueError(f'Unknown target bone {target!r}')
    if target in arm_chain(effector):
        raise ValueError(f'{target} is on the {effector} arm itself')


def normalize_tracks(tracks, frames):
    """Fill defaults, clamp/sort/merge every track's intervals to the motion and give each track a unique id.

    Keys this editor does not know are kept, so saving through the store never drops newer fields."""
    result, used = [], set()
    for raw in tracks:
        track = {**DEFAULTS, **copy.deepcopy(dict(raw))}
        track['ranges'] = clamp(raw.get('ranges', []), frames)
        track['automatic'] = clamp(raw.get('automatic', []), frames)
        if track.get('id') in used:
            track['id'] = None
        if track.get('id'):
            used.add(track['id'])
        result.append(track)
    for track in result:
        if not track.get('id'):
            track['id'] = pair_id(track['effector'], track['target'], used)
            used.add(track['id'])
    return result


def held(tracks, index):
    """Frames other enabled tracks hold with the same hand; one hand holds one target at a time."""
    track = tracks[index]
    if not track['enabled']:
        return []
    return merge(r for i, other in enumerate(tracks) if i != index and other['enabled'] and other['effector'] == track['effector']
                 for r in other['ranges'])


def conflicts(tracks):
    found = []
    for i, a in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            if not (a['enabled'] and b['enabled'] and a['effector'] == b['effector']):
                continue
            for s0, e0 in a['ranges']:
                for s1, e1 in b['ranges']:
                    if s0 < e1 and s1 < e0:
                        found.append((i, j, max(s0, s1), min(e0, e1)))
    return found


class SelfContactRows:
    """Editable K-track model with snapshot undo; every committed state passes the store's one-hand-one-target rule."""

    LIMIT = 100

    def __init__(self):
        self.tracks = []
        self.frames = 0
        self.identity = None
        self._undo = []
        self._redo = []

    def load(self, tracks, frames, identity=None):
        """Replace the tracks from storage; history survives only an unchanged reload of the same result. True if it was cleared."""
        tracks = normalize_tracks(tracks or [], frames)
        same = identity == self.identity and frames == self.frames and tracks == self.tracks
        self.identity, self.frames = identity, frames
        if same:
            return False
        self.tracks = tracks
        self._undo.clear()
        self._redo.clear()
        return True

    def clear(self):
        self.load([], 0, None)

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def _commit(self, tracks):
        clash = conflicts(tracks)
        if clash:
            i, j, start, end = clash[0]
            raise ValueError(f'{row_label(tracks[i])} and {row_label(tracks[j])} would both hold {tracks[i]["effector"]} on frames '
                             f'{start + 1}-{end}; one hand holds one target at a time')
        self._undo.append(copy.deepcopy(self.tracks))
        del self._undo[:-self.LIMIT]
        self._redo.clear()
        self.tracks = tracks

    def _copy(self):
        return copy.deepcopy(self.tracks)

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._copy())
        self.tracks = self._undo.pop()
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._copy())
        self.tracks = self._redo.pop()
        return True

    def add_pair(self, effector, target, **settings):
        check_pair(effector, target)
        tracks = self._copy()
        track = normalize_tracks([{**DEFAULTS, 'effector': effector, 'target': target, **settings}], self.frames)[0]
        track['id'] = pair_id(effector, target, {t['id'] for t in tracks})
        tracks.append(track)
        self._commit(tracks)
        return len(tracks) - 1

    def remove_row(self, index):
        tracks = self._copy()
        tracks.pop(index)
        self._commit(tracks)

    def update_settings(self, index, **changes):
        tracks = self._copy()
        track = tracks[index]
        unknown = set(changes) - {'target', 'distance', 'offset', 'reciprocal', 'grip', 'enabled'}
        if unknown:
            raise ValueError('Unknown self-contact setting: ' + ', '.join(sorted(unknown)))
        if 'target' in changes:
            check_pair(track['effector'], changes['target'])
        if changes.get('reciprocal', track['reciprocal']) not in RECIPROCAL or changes.get('grip', track['grip']) not in GRIP:
            raise ValueError('Invalid reciprocal or grip setting')
        distance = changes.get('distance', track['distance'])
        if distance != 'auto' and (isinstance(distance, bool) or not isinstance(distance, (int, float)) or not 0 <= distance < 10):
            raise ValueError('Distance must be "auto" or metres between 0 and 10')
        track.update(changes)
        if 'distance' in changes and distance != 'auto':
            track['distance'] = float(distance)
        if changes and track['source'] != 'user':
            track['source'] = 'user'
        self._commit(tracks)

    def set_interval(self, index, position, before, after):
        """Insert (before None), delete (after None) or replace interval number position of one track."""
        tracks = self._copy()
        ranges = list(tracks[index]['ranges'])
        if before is None:
            ranges.insert(position, tuple(after))
        elif after is None:
            ranges.pop(position)
        else:
            ranges[position] = tuple(after)
        tracks[index]['ranges'] = clamp(ranges, self.frames)
        self._commit(tracks)

    def suggestions(self, index):
        """Automatic intervals not yet accepted into the ranges."""
        track = self.tracks[index]
        return subtract(track['automatic'], track['ranges'])

    def accept(self, index, interval=None):
        """Copy suggestions (all, or those inside interval) into the ranges, skipping frames another track holds with the hand."""
        pieces = self.suggestions(index)
        if interval is not None:
            pieces = [(max(s, interval[0]), min(e, interval[1])) for s, e in pieces if s < interval[1] and interval[0] < e]
        free = subtract(pieces, held(self.tracks, index))
        if not free:
            return 0
        tracks = self._copy()
        tracks[index]['ranges'] = merge(list(tracks[index]['ranges']) + free)
        self._commit(tracks)
        return sum(e - s for s, e in free)

    def reset_row(self, index):
        """Ranges become the automatic suggestions, minus frames another track holds with the same hand."""
        tracks = self._copy()
        tracks[index]['ranges'] = subtract(tracks[index]['automatic'], held(tracks, index))
        self._commit(tracks)

    def blockers(self, index):
        return held(self.tracks, index)

    def payload(self):
        return [{**track, 'ranges': [list(p) for p in track['ranges']], 'automatic': [list(p) for p in track['automatic']]}
                for track in self.tracks]


class PairDialog:
    """Effector/target and solver settings for one self-contact pair."""

    @staticmethod
    def ask(parent, track=None):
        from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout
        from .mixamo_rig import BODY_NAMES
        from .self_contact_store import arm_chain
        dialog = QDialog(parent)
        dialog.setWindowTitle('Self-Contact Pair' if track is None else 'Self-Contact Settings: ' + row_label(track))
        form = QFormLayout(dialog)
        effector = QComboBox()
        for name in EFFECTORS:
            effector.addItem(name, name)
        target = QComboBox()

        def fill_targets():
            current = target.currentData() or (track['target'] if track else 'RightFoot')
            target.clear()
            for name in BODY_NAMES:
                if name not in arm_chain(effector.currentData()):
                    target.addItem(name, name)
            target.setCurrentIndex(max(0, target.findData(current)))
        effector.currentIndexChanged.connect(fill_targets)
        effector.setCurrentIndex(effector.findData(track['effector'] if track else 'RightHand'))
        fill_targets()
        effector.setEnabled(track is None)
        automatic = QCheckBox('Automatic distance')
        distance = QDoubleSpinBox()
        distance.setRange(0, 0.5)
        distance.setDecimals(3)
        distance.setSingleStep(.005)
        distance.setSuffix(' m')
        value = track['distance'] if track else 'auto'
        automatic.setChecked(value == 'auto')
        distance.setValue(.05 if value == 'auto' else float(value))
        distance.setEnabled(value != 'auto')
        automatic.toggled.connect(lambda on: distance.setEnabled(not on))
        reciprocal = QComboBox()
        for name, text in (('off', 'Off: only the arm moves'), ('auto', 'Auto: move the target limb when out of reach'), ('always', 'Always')):
            reciprocal.addItem(text, name)
        reciprocal.setCurrentIndex(reciprocal.findData(track['reciprocal'] if track else 'auto'))
        grip = QComboBox()
        for name in GRIP:
            grip.addItem(name.capitalize(), name)
        grip.setCurrentIndex(grip.findData(track['grip'] if track else 'off'))
        enabled = QCheckBox('Apply this pair')
        enabled.setChecked(track['enabled'] if track else True)
        form.addRow('Hand', effector)
        form.addRow('Holds', target)
        form.addRow(automatic)
        form.addRow('Palm distance', distance)
        form.addRow('Reciprocal', reciprocal)
        form.addRow('Grip', grip)
        form.addRow(enabled)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        try:
            if dialog.exec() != QDialog.Accepted:
                return None
            return dict(effector=effector.currentData(), target=target.currentData(),
                        distance='auto' if automatic.isChecked() else float(distance.value()),
                        reciprocal=reciprocal.currentData(), grip=grip.currentData(), enabled=enabled.isChecked())
        finally:
            dialog.deleteLater()


class SelfContactEditor:
    """Timeline mixin: self-contact rows share the foot-lock editor's drag, resize, delete and undo through track index t."""

    def init_self_contacts(self):
        self.sc_rows = SelfContactRows()
        self.sc_visible = False
        self.sc_editable = False
        self.sc_message = ''
        # Window hooks: sc_listener(payload) saves, sc_actions['suggest']() runs suggestions, sc_status(text) reports,
        # sc_resized() follows a row-count change.
        self.sc_listener = None
        self.sc_actions = {}
        self.sc_status = None
        self.sc_resized = None
        self.compare_delta = None
        self.compare_label = ''
        self._sc_height = None

    # Track access shared with the foot rows -------------------------------------------------------------------------

    def _track(self, t):
        if t < FIRST_ROW:
            return self.contacts[t] if t < len(self.contacts) else []
        return self.sc_rows.tracks[t - FIRST_ROW]['ranges']

    def _all_tracks(self):
        if not self.sc_visible:
            return list(self.contacts)
        return [self._track(t) for t in range(FIRST_ROW)] + [track['ranges'] for track in self.sc_rows.tracks]

    def _editable(self, t):
        return self.contact_edit_enabled and (t < FIRST_ROW or self.sc_editable)

    def self_contact_row(self, row):
        """Timeline row (0 = first contact row) of a self-contact track, or None."""
        if self.sc_visible and FIRST_ROW <= row < FIRST_ROW + len(self.sc_rows.tracks):
            return row
        return None

    def placeholder_row(self):
        return FIRST_ROW + len(self.sc_rows.tracks)

    def delta_row(self):
        return self.placeholder_row() + 1 if self.compare_delta is not None else None

    def self_contact_rows_total(self):
        return (len(self.sc_rows.tracks) + 1 if self.sc_visible else 1) + (1 if self.compare_delta is not None else 0)

    # Window-facing API ----------------------------------------------------------------------------------------------

    def set_self_contacts(self, tracks, frames=0, identity=None, editable=False, message=''):
        """tracks None hides the rows (no corrected result to attach them to)."""
        self.sc_message = message
        if tracks is None:
            self.sc_visible = False
            self.sc_rows.clear()
        else:
            self.sc_visible = True
            if self.sc_rows.load(tracks, frames, identity):
                self._drop_self_history()
        self.sc_editable = editable and tracks is not None
        if self._selected_contact is not None and self._selected_contact[0] >= FIRST_ROW:
            self._selected_contact = None
        if self._foot_drag and self._foot_drag['foot'] >= FIRST_ROW:
            self._foot_drag = None
            self._foot_draft = None
        self._index_contacts()
        self._update_self_height()

    def set_self_contacts_editable(self, editable):
        editable = bool(editable and self.sc_visible)
        if editable != self.sc_editable:
            self.sc_editable = editable
            self.contact_revision += 1
            self.update()

    def set_compare_delta(self, values, label=''):
        self.compare_delta = values
        self.compare_label = label
        self.contact_revision += 1
        self._update_self_height()
        self.update()

    def _update_self_height(self):
        rows = 2 + FIRST_ROW + self.self_contact_rows_total()
        height = max(190, 28 + rows * self.TRACK_HEIGHT + 6)
        if height != self._sc_height:
            self._sc_height = height
            self.setMinimumHeight(height)
            if self.sc_resized is not None:
                self.sc_resized()

    def _drop_self_history(self):
        self._undo_contacts[:] = [c for c in self._undo_contacts if c != SELF_UNDO]
        self._redo_contacts[:] = [c for c in self._redo_contacts if c != SELF_UNDO]

    def _report(self, text):
        if self.sc_status is not None:
            self.sc_status(text)

    def _self_changed(self):
        self._index_contacts()
        self._update_self_height()
        if self.sc_listener is not None:
            self.sc_listener(self.sc_rows.payload())

    def _self_apply(self, action, *args, **kwargs):
        """Run one model mutation as an undoable step; invalid edits are reported and leave the tracks unchanged."""
        if not self.sc_editable or not self.contact_edit_enabled:
            return None
        try:
            result = action(*args, **kwargs)
        except ValueError as error:
            self._report('Self-contact not changed: ' + str(error))
            self.update()
            return None
        self._undo_contacts.append(SELF_UNDO)
        del self._undo_contacts[:-100]
        self._redo_contacts.clear()
        self._self_changed()
        return True if result is None else result

    # Interval editing hooks used by the foot-lock editor ------------------------------------------------------------

    def _clamp_bounds(self, t, low, high, frame, original):
        """Drag limits for track t; self rows also stop at frames another track holds with the same hand. None refuses."""
        if t < FIRST_ROW:
            return low, high
        start, end = original if original is not None else (frame, frame + 1)
        for s, e in self.sc_rows.blockers(t - FIRST_ROW):
            if s < end and start < e:
                return None
            if e <= start:
                low = max(low, e)
            elif s >= end:
                high = min(high, s)
        return low, high

    def _add_limit(self, t, frame, high):
        bounds = self._clamp_bounds(t, 0, high, frame, None)
        return None if bounds is None else bounds[1]

    def self_contact_command(self, t, index, before, after):
        self._selected_contact = (t, index) if after is not None else None
        if self._self_apply(self.sc_rows.set_interval, t - FIRST_ROW, index, before, after) is None:
            self._selected_contact = None
            self._index_contacts()

    def undo_self_contact(self):
        if self.sc_rows.undo():
            self._selected_contact = None
            self._self_changed()
            return True
        return False

    def redo_self_contact(self):
        if self.sc_rows.redo():
            self._selected_contact = None
            self._self_changed()
            return True
        return False

    def suggestion_at(self, t, frame):
        if t is None or t < FIRST_ROW:
            return None
        return next(((s, e) for s, e in self.sc_rows.suggestions(t - FIRST_ROW) if s <= frame < e), None)

    def accept_suggestion_at(self, t, frame):
        piece = self.suggestion_at(t, frame)
        if piece is None or any(s <= frame < e for s, e in self._track(t)):
            return False
        accepted = self._self_apply(self.sc_rows.accept, t - FIRST_ROW, piece)
        if accepted is not None and not accepted:
            self._report('Suggestion not accepted: another contact already holds this hand on those frames.')
        return True

    # Menus ------------------------------------------------------------------------------------------------------------

    def _event_row(self, event):
        position = event.position() if hasattr(event, 'position') else event.pos()
        return int((position.y() - 28) // self.TRACK_HEIGHT) - 2

    def self_contact_menu(self, event):
        """Context menu for self-contact rows and the add-pair row; False leaves the event to the foot-lock menu."""
        from PySide6.QtWidgets import QMenu
        if not getattr(self, 'contacts_supported', True):
            return False
        row = self._event_row(event)
        track_row = self.self_contact_row(row)
        if track_row is None and row != self.placeholder_row():
            return False
        position = event.position() if hasattr(event, 'position') else event.pos()
        frame = min(self.maximum(), self._edge_position(event)) if position.x() >= self.LEFT else None
        editable = self.sc_editable and self.contact_edit_enabled
        menu = QMenu(self)
        if track_row is not None:
            i = track_row - FIRST_ROW
            track = self.sc_rows.tracks[i]
            index, _ = self._contact_hit(track_row, event)
            self._selected_contact = (track_row, index) if index is not None else None
            self.update()
            piece = self.suggestion_at(track_row, frame) if frame is not None and index is None else None
            title = menu.addAction(f'{row_label(track)}: {track["effector"]} holds {track["target"]}' + ('' if track['enabled'] else ' (off)'))
            title.setEnabled(False)
            add = menu.addAction('Add Contact Here', lambda: self.add_contact(track_row, frame))
            add.setEnabled(editable and frame is not None)
            remove = menu.addAction('Delete Selected Contact', self.delete_contact)
            remove.setEnabled(editable and self._selected_contact is not None)
            menu.addSeparator()
            one = menu.addAction('Accept This Suggestion', lambda: self.accept_suggestion_at(track_row, frame))
            one.setEnabled(editable and piece is not None)
            every = menu.addAction('Accept All Suggestions in Row', lambda: self._accept_row(i))
            every.setEnabled(editable and bool(self.sc_rows.suggestions(i)))
            menu.addSeparator()
            menu.addAction('Edit Pair Settings...', lambda: self._edit_pair(i)).setEnabled(editable)
            toggle = menu.addAction('Disable Pair' if track['enabled'] else 'Enable Pair',
                                    lambda: self._self_apply(self.sc_rows.update_settings, i, enabled=not track['enabled']))
            toggle.setEnabled(editable)
            menu.addAction('Reset Row to Automatic', lambda: self._self_apply(self.sc_rows.reset_row, i)).setEnabled(editable)
            menu.addAction('Remove Row', lambda: self._self_apply(self.sc_rows.remove_row, i)).setEnabled(editable)
            menu.addSeparator()
            undo = menu.addAction('Undo', self.undo_contact)
            undo.setEnabled(bool(self._undo_contacts))
            redo = menu.addAction('Redo', self.redo_contact)
            redo.setEnabled(bool(self._redo_contacts))
            menu.addSeparator()
        menu.addAction('Add Contact Pair...', self._add_pair).setEnabled(editable)
        suggest = self.sc_actions.get('suggest')
        menu.addAction('Suggest Contacts', suggest).setEnabled(suggest is not None and editable)
        menu.exec(event.globalPos())
        menu.deleteLater()
        return True

    def _accept_row(self, i):
        accepted = self._self_apply(self.sc_rows.accept, i)
        if accepted is not None and not accepted:
            self._report('No suggestion accepted: another contact already holds this hand on those frames.')

    def _add_pair(self):
        values = PairDialog.ask(self)
        if values is None:
            return
        effector, target = values.pop('effector'), values.pop('target')
        self._self_apply(self.sc_rows.add_pair, effector, target, **values)

    def _edit_pair(self, i):
        values = PairDialog.ask(self, self.sc_rows.tracks[i])
        if values is None:
            return
        values.pop('effector')
        track = self.sc_rows.tracks[i]
        changes = {key: value for key, value in values.items() if track.get(key) != value}
        if changes:
            self._self_apply(self.sc_rows.update_settings, i, **changes)

    # Painting ---------------------------------------------------------------------------------------------------------

    def paint_self_contact_rows(self, p):
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QBrush, QColor, QPen
        width = max(1, self.width() - self.LEFT - self.RIGHT)
        text_rect = lambda y: QRectF(self.LEFT + 8, y, max(0, self.width() - self.LEFT - 25), self.TRACK_HEIGHT)
        if self.sc_visible:
            for i, track in enumerate(self.sc_rows.tracks):
                t = FIRST_ROW + i
                y = 28 + (t + 2) * self.TRACK_HEIGHT
                color = QColor(ROW_COLORS[i % len(ROW_COLORS)])
                p.setPen(QColor('#aaaaaa') if track['enabled'] else QColor('#707070'))
                p.drawText(QRectF(8, y, 53, self.TRACK_HEIGHT), Qt.AlignVCenter, row_label(track))
                p.save()
                p.setClipRect(QRectF(self.LEFT, y, width, self.TRACK_HEIGHT))
                hatch = QColor(color)
                hatch.setAlpha(170)
                for s, e in self.sc_rows.suggestions(i):
                    if e < self.view_start or s > self.view_start + self.span():
                        continue
                    rect = self._contact_rect(t, (s, e))
                    p.fillRect(rect, QBrush(hatch, Qt.BDiagPattern))
                    p.setPen(QPen(hatch, 1, Qt.DashLine))
                    p.drawRect(rect)
                    if rect.width() > 70 and not any(a < e and s < b for a, b in track['ranges']):
                        p.drawText(rect, Qt.AlignCenter, 'Suggested')
                p.restore()
                if not track['enabled']:
                    color = QColor('#7d7d7d')
                self._paint_track_intervals(p, t, color.name(), 'Hold' if track['enabled'] else 'Off')
            y = 28 + (self.placeholder_row() + 2) * self.TRACK_HEIGHT
            p.setPen(QColor('#8c8c8c'))
            p.drawText(QRectF(8, y, 53, self.TRACK_HEIGHT), Qt.AlignVCenter, '+ Pair')
            hint = self.sc_message or ('Right-click to add a hand contact pair or suggest contacts' if not self.sc_rows.tracks else
                                       'Hatched: suggested. Double-click or right-click to accept')
            p.setPen(QColor('#e0b06c') if self.sc_message else QColor('#777777'))
            p.drawText(text_rect(y), Qt.AlignVCenter, hint)
        else:
            y = 28 + (self.placeholder_row() + 2) * self.TRACK_HEIGHT
            p.setPen(QColor('#aaaaaa'))
            p.drawText(QRectF(8, y, 53, self.TRACK_HEIGHT), Qt.AlignVCenter, 'Self')
            p.setPen(QColor('#777777'))
            p.drawText(text_rect(y), Qt.AlignVCenter, 'Apply Correction to edit self-contacts')
        if self.compare_delta is not None:
            self._paint_delta(p, 28 + (self.delta_row() + 2) * self.TRACK_HEIGHT, width)

    def _paint_delta(self, p, y, width):
        import numpy as np
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor
        values = self.compare_delta
        p.setPen(QColor('#aaaaaa'))
        p.drawText(QRectF(8, y, 53, self.TRACK_HEIGHT), Qt.AlignVCenter, 'Δ orig')
        if not len(values):
            return
        p.save()
        p.setClipRect(QRectF(self.LEFT, y, width, self.TRACK_HEIGHT))
        step = 2
        columns = np.arange(self.LEFT, self.width() - self.RIGHT, step)
        starts = np.floor(self.view_start + (columns - self.LEFT) * self.span() / width).astype(np.int64)
        ends = np.maximum(starts + 1, np.floor(self.view_start + (columns + step - self.LEFT) * self.span() / width).astype(np.int64))
        visible = (ends > 0) & (starts < len(values))
        starts, ends = np.clip(starts, 0, len(values) - 1), np.clip(ends, 1, len(values))
        top = self.TRACK_HEIGHT - 5
        for x, a, b, show in zip(columns, starts, ends, visible):
            if not show:
                continue
            value = float(values[a:b].max())
            if value < .002:
                continue
            height = max(1., min(1., value / .10) * top)
            color = QColor('#7fae7a') if value < .02 else QColor('#d9a650') if value < .06 else QColor('#d96b5f')
            color.setAlpha(200)
            p.fillRect(QRectF(float(x), y + 2 + top - height, step, height), color)
        p.restore()
        label = f'max {float(np.max(values)) * 100:.1f} cm' + (' | ' + self.compare_label if self.compare_label else '')
        box = p.fontMetrics().horizontalAdvance(label) + 12
        rect = QRectF(self.width() - self.RIGHT - box - 4, y + 5, box, self.TRACK_HEIGHT - 10)
        p.fillRect(rect, QColor(28, 28, 28, 210))
        p.setPen(QColor('#b0b0b0'))
        p.drawText(rect, Qt.AlignCenter, label)
