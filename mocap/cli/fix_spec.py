"""Animation fix specs for `rgbmocap-cli fix`: strict validation, a binary FBX skeleton probe and limb resolution.

Plain Python (no numpy, bpy or Qt) so the CLI and the Blender-side script (blender_fix.py) share the same checks.
"""
from __future__ import annotations

import difflib
import json
import math
import struct
import zlib
from array import array
from pathlib import Path

VERSION = 1
# Spec frame = file key index + 1: the numbering Blender shows after a default FBX import (anim_offset 1, integer fps).
FRAME_BASE = 1
WORLD = 'world'
SIDES = ('Left', 'Right')
# Effector (name without namespace and side) -> limb end bone, IK mid bone, IK root bone.
LIMBS = {'Hand': ('Hand', 'ForeArm', 'Arm'), 'Foot': ('Foot', 'Leg', 'UpLeg'), 'ToeBase': ('Foot', 'Leg', 'UpLeg')}
CHAIN = 2
# Arms may also turn the shoulder (clavicle) bone: chain 3 = Shoulder + Arm + ForeArm.
SHOULDER = 'Shoulder'
ROTATIONS = ('keep', 'follow')
TOP_KEYS = {'version', 'pins', 'smooth'}
PIN_KEYS = {'effector', 'target', 'frames', 'ease', 'distance', 'offset', 'chain', 'rotation'}
SMOOTH_KEYS = {'bones', 'frames', 'window'}


class SpecError(ValueError):
    """Invalid spec, or a spec that does not match the input file. The message is user-facing."""


class FbxError(ValueError):
    """The FBX cannot be probed without Blender (ASCII, damaged or without animation)."""


def _show(value):
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)


def _fail(where, text):
    raise SpecError(f'{where}: {text}' if where else text)


def _keys(item, allowed, required, where):
    if not isinstance(item, dict):
        _fail(where, f'expected an object, got {_show(item)}')
    unknown = sorted(set(item) - allowed)
    if unknown:
        _fail(where, f'unknown key {", ".join(map(_show, unknown))} (allowed: {", ".join(sorted(allowed))})')
    missing = [key for key in required if key not in item]
    if missing:
        _fail(where, 'missing ' + ', '.join(missing))


def _integer(value, where, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, f'expected an integer, got {_show(value)}')
    if minimum is not None and value < minimum:
        _fail(where, f'must be at least {minimum}, got {value}')
    return value


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(where, f'expected a number, got {_show(value)}')
    return float(value)


def _name(value, where):
    if not isinstance(value, str) or not value.strip():
        _fail(where, f'expected a bone name, got {_show(value)}')
    return value


def _frames(value, where):
    if not isinstance(value, list) or len(value) != 2:
        _fail(where, f'expected [first, last] frame numbers, got {_show(value)}')
    first, last = (_integer(item, where) for item in value)
    if first > last:
        _fail(where, f'first frame {first} is after last frame {last}')
    return [first, last]


def strip(name):
    """Bone name without a 'prefix:' namespace (mixamorig:RightHand -> RightHand)."""
    return name.rsplit(':', 1)[-1]


def effector_kind(name, where):
    """(side, part) of an effector name, e.g. ('Right', 'Hand')."""
    bare = strip(name)
    for side in SIDES:
        for part in LIMBS:
            if bare.casefold() == (side + part).casefold():
                return side, part
    allowed = ', '.join(side + part for side in SIDES for part in LIMBS)
    _fail(where, f'{_show(name)} is not an arm or leg end bone; use one of {allowed} (a namespace prefix is allowed)')


def limb_name(side, part):
    return f'{side.lower()} {"arm" if part == "Hand" else "leg"}'


def changed_window(frames, ease):
    """Frames a pin writes: the range plus ease-1 blend frames each side (frame first-ease has weight 0)."""
    margin = max(ease - 1, 0)
    return [frames[0] - margin, frames[1] + margin]


def ease_weight(frame, frames, ease):
    """Pin influence 0..1: 1 inside the range, smoothstep over `ease` frames outside it."""
    first, last = frames
    if first <= frame <= last:
        return 1.0
    gap = first - frame if frame < first else frame - last
    if ease <= 0 or gap >= ease:
        return 0.0
    x = 1.0 - gap / ease
    return x * x * (3.0 - 2.0 * x)


def validate(spec):
    """Check a parsed spec and return it normalized (defaults filled in)."""
    _keys(spec, TOP_KEYS, ('version',), 'spec')
    if _integer(spec['version'], 'version') != VERSION:
        _fail('version', f'unsupported version {spec["version"]}; expected {VERSION}')
    pins, smooth = spec.get('pins', []), spec.get('smooth', [])
    if not isinstance(pins, list):
        _fail('pins', 'expected a list')
    if not isinstance(smooth, list):
        _fail('smooth', 'expected a list')
    result = dict(version=VERSION, pins=[_pin(item, f'pins[{i}]') for i, item in enumerate(pins)],
                  smooth=[_smooth(item, f'smooth[{i}]') for i, item in enumerate(smooth)])
    _check_overlaps(result['pins'])
    return result


def _pin(item, where):
    _keys(item, PIN_KEYS, ('effector', 'target', 'frames', 'ease'), where)
    effector = _name(item['effector'], where + '.effector')
    effector_kind(effector, where + '.effector')
    target = _name(item['target'], where + '.target')
    pin = dict(effector=effector, target=target, frames=_frames(item['frames'], where + '.frames'),
               ease=_integer(item['ease'], where + '.ease', 0), distance=None, offset=None, chain=CHAIN, rotation='keep')
    if item.get('distance') is not None:
        pin['distance'] = _number(item['distance'], where + '.distance')
        if pin['distance'] < 0:
            _fail(where + '.distance', 'must not be negative')
        if target == WORLD:
            _fail(where + '.distance', 'needs a target bone; a "world" pin holds the effector on a point (use offset to move that point)')
    if item.get('offset') is not None:
        offset = item['offset']
        if not isinstance(offset, list) or len(offset) != 3:
            _fail(where + '.offset', f'expected [x, y, z], got {_show(offset)}')
        pin['offset'] = [_number(value, where + '.offset') for value in offset]
    if 'chain' in item:
        pin['chain'] = _integer(item['chain'], where + '.chain')
        arm = effector_kind(effector, '')[1] == 'Hand'
        if pin['chain'] not in ((CHAIN, CHAIN + 1) if arm else (CHAIN,)):
            allowed = '2 (upper arm + forearm) or 3 (also the shoulder)' if arm else '2 (thigh + shin)'
            _fail(where + '.chain', f'expected {allowed} for {strip(effector)}, got {pin["chain"]}')
    if 'rotation' in item:
        if item['rotation'] not in ROTATIONS:
            _fail(where + '.rotation', f'expected "keep" or "follow", got {_show(item["rotation"])}')
        pin['rotation'] = item['rotation']
    return pin


def _smooth(item, where):
    _keys(item, SMOOTH_KEYS, ('bones', 'frames', 'window'), where)
    bones = item['bones']
    if not isinstance(bones, list) or not bones:
        _fail(where + '.bones', 'expected a non-empty list of bone names')
    bones = [_name(bone, f'{where}.bones[{i}]') for i, bone in enumerate(bones)]
    if len({strip(bone) for bone in bones}) != len(bones):
        _fail(where + '.bones', 'lists a bone twice')
    return dict(bones=bones, frames=_frames(item['frames'], where + '.frames'), window=_integer(item['window'], where + '.window', 2))


def _check_overlaps(pins):
    # Pins on one limb are solved in order; overlapping windows would make the later one undo the earlier one.
    limbs = [limb_name(*effector_kind(pin['effector'], '')) for pin in pins]
    windows = [changed_window(pin['frames'], pin['ease']) for pin in pins]
    for i, window in enumerate(windows):
        for j in range(i):
            span = windows[j]
            if limbs[i] == limbs[j] and window[0] <= span[1] and span[0] <= window[1]:
                _fail(f'pins[{i}]', f'overlaps pins[{j}] on the {limbs[i]} (frames {window[0]}-{window[1]} incl. ease vs '
                                    f'{span[0]}-{span[1]}); split the ranges so their ease windows do not meet')


class Skeleton:
    """Bone names and parents (None for roots), in file order."""

    def __init__(self, parents):
        self.parents = dict(parents)

    def parent(self, name):
        return self.parents.get(name)

    def ancestors(self, name):
        chain = []
        while (name := self.parents.get(name)) is not None:
            chain.append(name)
        return chain

    def resolve(self, name, where):
        if name in self.parents:
            return name
        bare = strip(name)
        for same in (lambda bone: strip(bone) == bare, lambda bone: strip(bone).casefold() == bare.casefold()):
            matches = [bone for bone in self.parents if same(bone)]
            if len(matches) == 1:
                return matches[0]
            if matches:
                _fail(where, f'{_show(name)} is ambiguous: {", ".join(matches)}')
        close = difflib.get_close_matches(bare, [strip(bone) for bone in self.parents], 3, 0.6)
        hint = f'; did you mean {", ".join(close)}?' if close else ''
        _fail(where, f'no bone named {_show(name)} in the input ({len(self.parents)} bones){hint}')


class Clip:
    """Animated frame range in spec numbering, and the file frame rate."""

    def __init__(self, first, last, fps):
        self.first, self.last, self.fps = int(first), int(last), float(fps)

    def as_dict(self):
        return dict(first=self.first, last=self.last, frames=self.last - self.first + 1, fps=self.fps)


def _inside(frames, clip, where):
    if frames[0] < clip.first or frames[1] > clip.last:
        _fail(where, f'{frames[0]}-{frames[1]} is outside the animation (frames {clip.first}-{clip.last}; '
                     f'frame {FRAME_BASE} is the first key, as Blender numbers it after a default FBX import)')


def limb_bones(skeleton, effector, side, part, where):
    """(end, mid, root) bones of the limb an effector belongs to, checked against the expected names."""
    end_name, mid_name, root_name = LIMBS[part]
    end = effector if part != 'ToeBase' else skeleton.parent(effector)
    found = [end]
    for name in (mid_name, root_name):
        found.append(skeleton.parent(found[-1]) if found[-1] is not None else None)
    for bone, expected in zip(found, (end_name, mid_name, root_name)):
        if bone is None or strip(bone).casefold() != (side + expected).casefold():
            _fail(where, f'{effector}: expected the limb chain {side + root_name} > {side + mid_name} > {side + end_name}, '
                         f'found {" > ".join(str(bone) for bone in reversed(found))}')
    return tuple(found)


def plan(spec, skeleton=None, clip=None):
    """Resolve a validated spec against the input's bones and frame range (either may be None when unknown)."""
    pins = []
    for i, pin in enumerate(spec['pins']):
        where = f'pins[{i}]'
        side, part = effector_kind(pin['effector'], where + '.effector')
        if clip is not None:
            _inside(pin['frames'], clip, where + '.frames')
        window = changed_window(pin['frames'], pin['ease'])
        if clip is not None:
            window = [max(window[0], clip.first), min(window[1], clip.last)]
        entry = dict(pin, limb=limb_name(side, part), window=window)
        if skeleton is not None:
            effector = skeleton.resolve(pin['effector'], where + '.effector')
            end, mid, root = limb_bones(skeleton, effector, side, part, where + '.effector')
            chain = [root, mid]
            if pin['chain'] > CHAIN:
                shoulder = skeleton.parent(root)
                if shoulder is None or strip(shoulder).casefold() != (side + SHOULDER).casefold():
                    _fail(where + '.chain', f'chain 3 needs {side + SHOULDER} as the parent of {root}, found {shoulder}')
                chain.insert(0, shoulder)
            root = chain[0]
            entry.update(effector=effector, end_bone=end, chain_bones=chain)
            if pin['target'] != WORLD:
                target = skeleton.resolve(pin['target'], where + '.target')
                if root == target or root in skeleton.ancestors(target):
                    _fail(where + '.target', f'{target} is moved by this pin\'s own chain ({root} and its children); pick a bone outside the limb')
                entry['target'] = target
        pins.append(entry)
    smooth = []
    for i, item in enumerate(spec['smooth']):
        where = f'smooth[{i}]'
        if clip is not None:
            _inside(item['frames'], clip, where + '.frames')
        bones = item['bones']
        if skeleton is not None:
            bones = [skeleton.resolve(bone, f'{where}.bones[{j}]') for j, bone in enumerate(bones)]
            if len(set(bones)) != len(bones):
                _fail(where + '.bones', 'lists a bone twice')
        smooth.append(dict(item, bones=bones))
    return dict(resolved=skeleton is not None, clip=clip.as_dict() if clip is not None else None, pins=pins, smooth=smooth)


# Binary FBX probe ---------------------------------------------------------------------------------------------------

FBX_MAGIC = b'Kaydara FBX Binary  \x00'
KTIME_V7 = 46186158000
KTIME_V8 = 141120000
# FBX TimeMode -> frames per second (Blender's FBX_FRAMERATES; 14 is custom).
TIME_MODES = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 30 / 1.001, 9: 30 / 1.001, 10: 25.0, 11: 24.0,
              13: 24 / 1.001, 15: 96.0, 16: 72.0, 17: 60 / 1.001, 18: 120 / 1.001}
_SCALARS = {ord('Y'): '<h', ord('C'): '<?', ord('I'): '<i', ord('F'): '<f', ord('D'): '<d', ord('L'): '<q'}
_ARRAYS = {ord('f'): 'f', ord('d'): 'd', ord('l'): 'q', ord('i'): 'i', ord('b'): 'b'}


class _Array:
    """Array property, decoded only when used (meshes can be large)."""

    def __init__(self, code, encoding, raw):
        self.code, self.encoding, self.raw = code, encoding, raw

    def values(self):
        raw = zlib.decompress(self.raw) if self.encoding == 1 else bytes(self.raw)
        return array(self.code, raw)


def _property(data, pos):
    code = data[pos]
    pos += 1
    if code in _SCALARS:
        layout = _SCALARS[code]
        return struct.unpack_from(layout, data, pos)[0], pos + struct.calcsize(layout)
    if code in _ARRAYS:
        _, encoding, size = struct.unpack_from('<III', data, pos)
        pos += 12
        return _Array(_ARRAYS[code], encoding, data[pos:pos + size]), pos + size
    if code in (ord('S'), ord('R')):
        size = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        return bytes(data[pos:pos + size]), pos + size
    raise FbxError(f'unknown FBX property type {chr(code)!r}')


def _nodes(data, pos, end, header):
    nodes = []
    while pos < end:
        node_end, count, _, name_size = header.unpack_from(data, pos)
        if node_end == 0:
            break
        pos += header.size
        name = bytes(data[pos:pos + name_size]).decode('ascii', 'replace')
        pos += name_size
        props = []
        for _ in range(count):
            value, pos = _property(data, pos)
            props.append(value)
        nodes.append((name, props, _nodes(data, pos, node_end, header) if pos < node_end else []))
        pos = node_end
    return nodes


def _children(node, name):
    return [child for child in node[2] if child[0] == name]


def read_fbx(path):
    """(Skeleton, Clip) of a binary FBX, numbered like Blender's importer; raises FbxError when that is not possible."""
    data = memoryview(Path(path).read_bytes())
    if bytes(data[:len(FBX_MAGIC)]) != FBX_MAGIC:
        raise FbxError('not a binary FBX')
    version = struct.unpack_from('<I', data, 23)[0]
    try:
        top = _nodes(data, 27, len(data), struct.Struct('<QQQB' if version >= 7500 else '<IIIB'))
    except (struct.error, IndexError, zlib.error) as exc:
        raise FbxError(f'cannot parse the FBX ({exc})') from None
    sections = {node[0]: node for node in top}
    settings = {}
    for props in _children(sections.get('GlobalSettings', ('', [], [])), 'Properties70'):
        for prop in _children(props, 'P'):
            settings[prop[1][0]] = prop[1][4] if len(prop[1]) > 4 else None
    custom = settings.get(b'CustomFrameRate') or 25.0
    fps = TIME_MODES.get(settings.get(b'TimeMode'), custom)
    fps = fps if fps > 0 else 25.0
    names, parents, times = {}, {}, []
    for node in sections.get('Objects', ('', [], []))[2]:
        if node[0] == 'Model' and len(node[1]) >= 3 and node[1][2] in (b'LimbNode', b'Limb'):
            names[node[1][0]] = node[1][1].split(b'\x00\x01')[0].decode('utf-8', 'replace')
        elif node[0] == 'AnimationCurve':
            for key_time in _children(node, 'KeyTime'):
                values = key_time[1][0].values()
                if len(values):
                    times += [min(values), max(values)]
    for link in _children(sections.get('Connections', ('', [], [])), 'C'):
        if link[1][0] == b'OO' and link[1][1] in names and link[1][2] in names:
            parents[link[1][1]] = names[link[1][2]]
    if not names:
        raise FbxError('no skeleton found')
    if not times:
        raise FbxError('no animation keys')
    # Key i is frame i + FRAME_BASE (blender_fix.py undoes the importer's frame-rate rounding the same way).
    scale = fps / (KTIME_V8 if version >= 8000 else KTIME_V7)
    clip = Clip(round(min(times) * scale) + FRAME_BASE, round(max(times) * scale) + FRAME_BASE, fps)
    return Skeleton({name: parents.get(uid) for uid, name in names.items()}), clip
