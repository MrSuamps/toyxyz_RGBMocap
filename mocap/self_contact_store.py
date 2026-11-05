"""Self-contact tracks (a hand held on another body part). Frames are zero-based, end-exclusive.

Stored like foot-lock edits: two alternating payload slots plus an atomic state file that commits the current one.
The same payload optionally holds foot `articulation` entries (pointe/flat/flexed/match per foot interval).
"""
import json
import math
import re
from pathlib import Path

VERSION = 1
EFFECTORS = ('LeftHand', 'RightHand')
RECIPROCAL = ('off', 'auto', 'always')
GRIP = ('off', 'auto')
KEYS = ('id', 'effector', 'target', 'offset', 'distance', 'reciprocal', 'grip', 'enabled', 'ranges', 'automatic', 'source')
ID_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
UNSIDED = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head')
FEET = ('LeftFoot', 'RightFoot')
MODES = ('match', 'pointe', 'flat', 'flexed')
PROFILES = ('default', 'dancer')
ARTICULATION_KEYS = ('id', 'foot', 'mode', 'strength', 'profile', 'enabled', 'ranges', 'automatic', 'source')


def _bases():
    from .foot_contacts import BASES
    return BASES


def _body_names():
    from .mixamo_rig import BODY_NAMES
    return BODY_NAMES


def arm_chain(effector):
    side = effector[:-4]
    return tuple(side + name for name in ('Shoulder', 'Arm', 'ForeArm', 'Hand'))


def short_name(bone):
    """LeftHand -> lh, RightToeBase -> rtb, Spine1 -> spine1."""
    if bone in UNSIDED:
        return bone.lower()
    side = 'l' if bone.startswith('Left') else 'r'
    rest = bone[4:] if side == 'l' else bone[5:]
    return side + ''.join(ch for ch in rest if ch.isupper()).lower()


def state_path(project, base):
    if base not in _bases():
        raise ValueError('Unknown self-contact base')
    return Path(project) / f'selfcontact_{base}_state.json'


def slot_names(base):
    return f'selfcontact_{base}_edit0.json', f'selfcontact_{base}_edit1.json'


def _state(project, base):
    path = state_path(project, base)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def revision(project, base):
    return _state(project, base).get('revision')


def _ranges(value, frames, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{label} must be a list of [start, end] intervals')
    pairs = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'{label}: invalid interval {pair!r}')
        start, end = pair
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
            raise ValueError(f'{label}: interval bounds must be integers, got {pair!r}')
        if not 0 <= start < end <= frames:
            raise ValueError(f'{label}: interval {start}-{end} is outside the motion frames 0-{frames} (end-exclusive)')
        pairs.append((start, end))
    merged = []
    for start, end in sorted(pairs):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number, got {value!r}')
    return float(value)


def _track(raw, frames, index):
    if not isinstance(raw, dict):
        raise ValueError(f'Self-contact track {index + 1} must be an object')
    unknown = sorted(set(raw) - set(KEYS))
    if unknown:
        raise ValueError(f'Self-contact track {index + 1}: unknown field(s) {", ".join(unknown)}')
    name = raw.get('id')
    label = f'Self-contact track {name!r}' if name is not None else f'Self-contact track {index + 1}'
    effector, target = raw.get('effector'), raw.get('target')
    if effector not in EFFECTORS:
        raise ValueError(f'{label}: effector must be one of {", ".join(EFFECTORS)}, got {effector!r}')
    if target not in _body_names():
        raise ValueError(f'{label}: target must be a body bone such as RightFoot, got {target!r}')
    if target in arm_chain(effector):
        raise ValueError(f'{label}: target {target} is on the {effector} arm itself')
    offset = raw.get('offset', 'auto')
    if offset != 'auto':
        if not isinstance(offset, (list, tuple)) or len(offset) != 3:
            raise ValueError(f'{label}: offset must be "auto" or three numbers (metres), got {offset!r}')
        offset = [_number(v, label + ' offset') for v in offset]
    distance = raw.get('distance', 'auto')
    if distance != 'auto':
        distance = _number(distance, label + ' distance')
        if distance < 0:
            raise ValueError(f'{label}: distance must be >= 0')
    reciprocal = raw.get('reciprocal', 'auto')
    if reciprocal not in RECIPROCAL:
        raise ValueError(f'{label}: reciprocal must be one of {", ".join(RECIPROCAL)}, got {reciprocal!r}')
    grip = raw.get('grip', 'off')
    if grip not in GRIP:
        raise ValueError(f'{label}: grip must be one of {", ".join(GRIP)}, got {grip!r}')
    enabled = raw.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError(f'{label}: enabled must be true or false')
    source = raw.get('source', 'user')
    if not isinstance(source, str):
        raise ValueError(f'{label}: source must be a string')
    if name is not None and (not isinstance(name, str) or not ID_PATTERN.fullmatch(name)):
        raise ValueError(f'{label}: id must be 1-64 letters, digits, ".", "_" or "-"')
    return dict(id=name, effector=effector, target=target, offset=offset, distance=distance, reciprocal=reciprocal, grip=grip,
                enabled=enabled, ranges=_ranges(raw.get('ranges', []), frames, label + ' ranges'),
                automatic=_ranges(raw.get('automatic', []), frames, label + ' automatic'), source=source)


def normalize_tracks(tracks, frames):
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError('Self-contact frame count must be a positive integer')
    if not isinstance(tracks, (list, tuple)):
        raise ValueError('Self-contact tracks must be a list')
    result = [_track(raw, frames, i) for i, raw in enumerate(tracks)]
    used = set()
    for track in result:
        if track['id'] is not None:
            if track['id'] in used:
                raise ValueError(f'Duplicate self-contact track id {track["id"]!r}')
            used.add(track['id'])
    for track in result:
        if track['id'] is None:
            stem = short_name(track['effector']) + '-' + short_name(track['target'])
            name, count = stem, 2
            while name in used:
                name, count = f'{stem}-{count}', count + 1
            track['id'] = name
            used.add(name)
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            if not (a['enabled'] and b['enabled'] and a['effector'] == b['effector']):
                continue
            for s0, e0 in a['ranges']:
                for s1, e1 in b['ranges']:
                    if s0 < e1 and s1 < e0:
                        raise ValueError(f'Self-contact tracks {a["id"]!r} and {b["id"]!r} both hold {a["effector"]} on frames '
                                         f'{max(s0, s1)}-{min(e0, e1) - 1}; one hand can hold one target at a time')
    return result


def _articulation(raw, frames, index):
    if not isinstance(raw, dict):
        raise ValueError(f'Foot articulation {index + 1} must be an object')
    unknown = sorted(set(raw) - set(ARTICULATION_KEYS))
    if unknown:
        raise ValueError(f'Foot articulation {index + 1}: unknown field(s) {", ".join(unknown)}')
    name = raw.get('id')
    label = f'Foot articulation {name!r}' if name is not None else f'Foot articulation {index + 1}'
    foot, mode, profile = raw.get('foot'), raw.get('mode', 'match'), raw.get('profile', 'default')
    if foot not in FEET:
        raise ValueError(f'{label}: foot must be one of {", ".join(FEET)}, got {foot!r}')
    if mode not in MODES:
        raise ValueError(f'{label}: mode must be one of {", ".join(MODES)}, got {mode!r}')
    if profile not in PROFILES:
        raise ValueError(f'{label}: profile must be one of {", ".join(PROFILES)}, got {profile!r}')
    strength = _number(raw.get('strength', 1.0), label + ' strength')
    if not 0 <= strength <= 1:
        raise ValueError(f'{label}: strength must be between 0 and 1')
    enabled = raw.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError(f'{label}: enabled must be true or false')
    source = raw.get('source', 'user')
    if not isinstance(source, str):
        raise ValueError(f'{label}: source must be a string')
    if name is not None and (not isinstance(name, str) or not ID_PATTERN.fullmatch(name)):
        raise ValueError(f'{label}: id must be 1-64 letters, digits, ".", "_" or "-"')
    return dict(id=name, foot=foot, mode=mode, strength=strength, profile=profile, enabled=enabled,
                ranges=_ranges(raw.get('ranges', []), frames, label + ' ranges'),
                automatic=_ranges(raw.get('automatic', []), frames, label + ' automatic'), source=source)


def normalize_articulation(entries, frames):
    """Validated foot articulation entries with ids filled in; one enabled mode per foot and frame."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError('Foot articulation frame count must be a positive integer')
    if not isinstance(entries, (list, tuple)):
        raise ValueError('Foot articulation must be a list')
    result = [_articulation(raw, frames, i) for i, raw in enumerate(entries)]
    used = set()
    for entry in result:
        if entry['id'] is not None:
            if entry['id'] in used:
                raise ValueError(f'Duplicate foot articulation id {entry["id"]!r}')
            used.add(entry['id'])
    for entry in result:
        if entry['id'] is None:
            stem = short_name(entry['foot']) + '-' + entry['mode']
            name, count = stem, 2
            while name in used:
                name, count = f'{stem}-{count}', count + 1
            entry['id'] = name
            used.add(name)
    for i, a in enumerate(result):
        for b in result[i + 1:]:
            if not (a['enabled'] and b['enabled'] and a['foot'] == b['foot']):
                continue
            for s0, e0 in a['ranges']:
                for s1, e1 in b['ranges']:
                    if s0 < e1 and s1 < e0:
                        raise ValueError(f'Foot articulations {a["id"]!r} and {b["id"]!r} both set {a["foot"]} on frames '
                                         f'{max(s0, s1)}-{min(e0, e1) - 1}; one foot can have one mode at a time')
    return result


def _payload(project, base, context):
    """(payload or None, state, warning) for the committed slot; None with a warning when the time base differs."""
    state = _state(project, base)
    if not state:
        return None, state, ''
    current = state.get('revision')
    try:
        same = int(state['frames']) == int(context['frames']) and float(state['fps']) == float(context['fps'])
    except (KeyError, TypeError, ValueError):
        same = False
    if not same:
        return None, state, 'belong to motion with a different length or frame rate; they are not applied. Review them.'
    filename = state.get('file')
    if filename not in slot_names(base):
        raise ValueError('Invalid self-contact edit file')
    data = json.loads((Path(project) / filename).read_text(encoding='utf-8'))
    if data.get('revision') != current:
        raise ValueError('Incomplete self-contact save; edit the self-contacts again')
    return data, state, ''


def read(project, base, context):
    """(tracks, revision, warning). No file: ([], None, ''). A different time base keeps the revision but no tracks."""
    data, state, warning = _payload(project, base, context)
    if warning:
        return [], state.get('revision'), 'Saved self-contacts ' + warning
    if data is None:
        return [], None, ''
    return normalize_tracks(data.get('tracks', []), int(context['frames'])), state.get('revision'), ''


def read_articulation(project, base, context):
    """(entries, revision, warning) for foot articulation, with the same rules as read()."""
    data, state, warning = _payload(project, base, context)
    if warning:
        return [], state.get('revision'), ('Saved foot articulation ' + warning) if _stored_articulation(project, base) else ''
    if data is None:
        return [], None, ''
    return normalize_articulation(data.get('articulation', []), int(context['frames'])), state.get('revision'), ''


def _stored_articulation(project, base):
    """Raw articulation entries of the committed slot, whatever its time base; [] when unreadable."""
    state = _state(project, base)
    filename = state.get('file')
    if filename not in slot_names(base):
        return []
    try:
        data = json.loads((Path(project) / filename).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    return data.get('articulation', []) if data.get('revision') == state.get('revision') else []


def save(project, base, tracks, context, articulation=None):
    """Write the payload to the slot not referenced by the current state, then commit the state. Returns the new revision.

    articulation=None keeps the stored foot articulation entries; pass a list (possibly empty) to replace them.
    """
    import uuid
    from .storage import atomic_json
    frames, fps = int(context['frames']), float(context['fps'])
    if frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid self-contact time base')
    tracks = normalize_tracks(tracks, frames)
    if articulation is None:
        articulation = _stored_articulation(project, base)
    articulation = normalize_articulation(articulation, frames)
    previous = _state(project, base)
    current = uuid.uuid4().hex
    first, second = slot_names(base)
    filename = first if previous.get('file') == second else second
    payload = [{**track, 'ranges': [list(p) for p in track['ranges']], 'automatic': [list(p) for p in track['automatic']]} for track in tracks]
    record = {'version': VERSION, 'revision': current, 'tracks': payload}
    if articulation:
        record['articulation'] = [{**e, 'ranges': [list(p) for p in e['ranges']], 'automatic': [list(p) for p in e['automatic']]} for e in articulation]
    atomic_json(Path(project) / filename, record)
    atomic_json(state_path(project, base), {'frames': frames, 'fps': fps, 'sources': dict(context.get('sources') or {}),
                                            'revision': current, 'file': filename})
    return current


def save_articulation(project, base, entries, context):
    """Replace the foot articulation entries, keeping the tracks. Removes the files when both end up empty."""
    tracks, _, warning = read(project, base, context)
    if warning and _state(project, base):
        raise ValueError(warning)
    entries = normalize_articulation(entries, int(context['frames']))
    if not tracks and not entries:
        clear(project, base, articulation=True)
        return None
    return save(project, base, tracks, context, articulation=entries)


def clear(project, base, articulation=False):
    """Remove all self-contact tracks. True if anything was removed.

    Stored foot articulation entries are kept (re-saved without tracks) unless articulation=True; only when nothing is
    left are the files deleted, so revisions fall back to foot-lock values only.
    """
    state = _state(project, base)
    kept = [] if articulation else _stored_articulation(project, base)
    if kept:
        data = json.loads((Path(project) / state['file']).read_text(encoding='utf-8'))
        save(project, base, [], state, articulation=kept)
        return bool(data.get('tracks'))
    removed = False
    for path in (state_path(project, base), *(Path(project) / name for name in slot_names(base))):
        if path.exists():
            path.unlink()
            removed = True
    return removed
