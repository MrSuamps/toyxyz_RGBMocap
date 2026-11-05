"""Runs inside Blender (-b): apply an animation fix spec (IK pins, smoothing) to an FBX and export the result.

Started by `rgbmocap-cli fix` (mocap/cli/fixes.py) with one argument, a request JSON {input, output, result, spec}.
The result JSON ({ok, ...report} or {ok: false, error}) is written to request["result"].
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

PROGRESS = 'FIX: '
# Below this bend (perpendicular elbow/knee offset / limb length) the bend direction is too noisy to trust.
BENT = 0.01
# Margin keeping the two-bone solve away from the fully straight / folded singularities.
REACH_EPSILON = 1e-6
# Chain 3: the shoulder (clavicle) turns only when the arm alone cannot reach, until the arm would be this extended,
# and by at most SHOULDER_LIMIT (roughly the clavicle's elevation / protraction range).
SHOULDER_REACH = 0.995
SHOULDER_LIMIT = math.radians(30.0)


def _load_spec_module():
    # Like the CLI (-B), keep __pycache__ out of the install tree.
    sys.dont_write_bytecode = True
    path = Path(__file__).resolve().with_name('fix_spec.py')
    loader = importlib.util.spec_from_file_location('rgbmocap_fix_spec', path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    return module


SPEC = _load_spec_module()


class FixError(ValueError):
    """Input that cannot be fixed; reported to the user without a traceback."""


def progress(text):
    print(PROGRESS + text, flush=True)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def loc_rot(location, rotation):
    matrix = rotation.to_matrix().to_4x4() if isinstance(rotation, Quaternion) else rotation.to_4x4()
    matrix.translation = location
    return matrix


def frame_basis(axis, normal):
    """Orthonormal rotation with columns axis, normal, axis x normal (axis is perpendicular to normal)."""
    return Matrix((axis, normal, axis.cross(normal))).transposed()


def solve_two_bone(S, E, W, goal, bend):
    """Rotate a two-bone chain S-E-W so W reaches goal; `bend` is the original unit bend direction (E away from line S-W).

    The pole is the original mid joint pushed out along `bend`, so the new mid joint stays on the same side and the
    chain keeps its twist about the bend plane. With goal == W this returns identity rotations.
    Returns (upper rotation, lower rotation, new mid joint, reached end, clamped).
    """
    upper, lower = (E - S).length, (W - E).length
    axis = (W - S).normalized()
    normal = axis.cross(bend).normalized()
    reach = goal - S
    distance = reach.length
    new_axis = reach / distance if distance > 1e-9 else axis
    low, high = abs(upper - lower) + REACH_EPSILON, upper + lower - REACH_EPSILON
    clamped = not low <= distance <= high
    distance = min(max(distance, low), high)
    pole = E + bend * (upper + lower) - S
    new_bend = pole - new_axis * pole.dot(new_axis)
    new_bend = new_bend.normalized() if new_bend.length > 1e-9 else bend
    new_normal = new_axis.cross(new_bend).normalized()
    along = (upper * upper - lower * lower + distance * distance) / (2.0 * distance)
    height = math.sqrt(max(upper * upper - along * along, 0.0))
    new_E = S + new_axis * along + new_bend * height
    new_W = S + new_axis * distance
    R1 = frame_basis((new_E - S).normalized(), new_normal) @ frame_basis((E - S).normalized(), normal).transposed()
    R2 = frame_basis((new_W - new_E).normalized(), new_normal) @ frame_basis((W - E).normalized(), normal).transposed()
    return R1, R2, new_E, new_W, clamped


def shoulder_assist(C, S, goal, reach):
    """Smallest rotation about the shoulder head C that brings the arm root S within SHOULDER_REACH * reach of goal."""
    s, g = S - C, goal - C
    axis = s.cross(g)
    if s.length < 1e-9 or g.length < 1e-9 or axis.length < 1e-12:
        return Matrix.Identity(3)
    # |g - R s|^2 = |g|^2 + |s|^2 - 2|g||s|cos(angle(s, g) - turn); solve for the turn giving |g - R s| = r.
    r = SHOULDER_REACH * reach
    angle = s.angle(g)
    needed = (g.length_squared + s.length_squared - r * r) / (2.0 * g.length * s.length)
    if needed <= math.cos(angle):
        return Matrix.Identity(3)
    turn = angle if needed >= 1.0 else angle - math.acos(needed)
    return Matrix.Rotation(min(turn, SHOULDER_LIMIT), 3, axis.normalized())


class Channels:
    """Quaternion rotation F-curves of one pose bone, edited key by key."""

    def __init__(self, action, pose_bone, first, last):
        if pose_bone.rotation_mode != 'QUATERNION':
            raise FixError(f'{pose_bone.name}: rotation mode {pose_bone.rotation_mode} is not supported (expected FBX quaternion keys)')
        self.first, self.last = first, last
        path = pose_bone.path_from_id('rotation_quaternion')
        self.curves = []
        for index in range(4):
            curve = action.fcurves.find(path, index=index)
            if curve is None:
                # A static bone: one key holds its pose everywhere (constant extrapolation).
                curve = action.fcurves.new(path, index=index, action_group=pose_bone.name)
                curve.keyframe_points.insert(first, pose_bone.rotation_quaternion[index], options={'FAST'}).interpolation = 'LINEAR'
                curve.update()
            self.curves.append(curve)
        self._index()

    def _index(self):
        self.keys = [{round(point.co.x): i for i, point in enumerate(curve.keyframe_points) if abs(point.co.x - round(point.co.x)) < 1e-4}
                     for curve in self.curves]

    def value(self, frame):
        frame = min(max(frame, self.first), self.last)
        return Quaternion([curve.evaluate(frame) for curve in self.curves])

    def prepare(self, frames):
        """Key every frame in frames and one frame beyond each end at its current value, so other frames keep evaluating identically.

        FBX import keys are linear; a key inserted on a linear segment at its evaluated value leaves the segment unchanged.
        """
        first, last = max(min(frames) - 1, self.first), min(max(frames) + 1, self.last)
        inserted = False
        for curve, keys in zip(self.curves, self.keys):
            for frame in range(first, last + 1):
                if frame not in keys:
                    curve.keyframe_points.insert(frame, curve.evaluate(frame), options={'FAST'}).interpolation = 'LINEAR'
                    inserted = True
            curve.update()
        if inserted:
            self._index()

    def set(self, frame, rotation):
        current = Quaternion([curve.keyframe_points[keys[frame]].co.y for curve, keys in zip(self.curves, self.keys)])
        if current.dot(rotation) < 0.0:
            rotation = -rotation
        for curve, keys, value in zip(self.curves, self.keys, rotation):
            point = curve.keyframe_points[keys[frame]]
            point.co.y = point.handle_left.y = point.handle_right.y = value

    def finish(self):
        for curve in self.curves:
            curve.update()


class Fixer:
    def __init__(self, arm, action, first, last):
        self.arm, self.action, self.first, self.last = arm, action, first, last
        self.scene = bpy.context.scene
        self.unit = sum(arm.matrix_world.to_scale()) / 3.0  # meters per armature unit
        self.channels = {}

    def channel(self, name):
        if name not in self.channels:
            self.channels[name] = Channels(self.action, self.arm.pose.bones[name], self.first, self.last)
        return self.channels[name]

    def finish(self):
        for channels in self.channels.values():
            channels.finish()

    def sample(self, frames, names=None):
        """Armature-space pose matrices {frame: {bone: Matrix}}."""
        bones = [self.arm.pose.bones[name] for name in names] if names else list(self.arm.pose.bones)
        poses = {}
        for frame in sorted(set(frames)):
            self.scene.frame_set(frame)
            poses[frame] = {bone.name: bone.matrix.copy() for bone in bones}
        return poses

    def relative_rest(self, name):
        bone = self.arm.data.bones[name]
        return bone.parent.matrix_local.inverted() @ bone.matrix_local if bone.parent else bone.matrix_local.copy()

    def smooth(self, entry, base):
        first, last = entry['frames'][0] - base, entry['frames'][1] - base
        window = entry['window']
        radius, sigma = window // 2, window / 4.0
        kernel = [(k, math.exp(-k * k / (2.0 * sigma * sigma))) for k in range(-radius, radius + 1)]
        report = dict(bones=entry['bones'], frames=entry['frames'], window=window, max_change_deg={})
        for name in entry['bones']:
            channels = self.channel(name)
            # Kernel taps beyond the clip repeat its first / last frame (Channels.value clamps).
            original = {frame: channels.value(frame) for frame in range(first - radius, last + radius + 1)}
            channels.prepare(range(first, last + 1))
            largest = 0.0
            for frame in range(first, last + 1):
                q0 = original[frame]
                total = Quaternion((0.0, 0.0, 0.0, 0.0))
                for offset, weight in kernel:
                    q = original[frame + offset]
                    total = total + (q if q.dot(q0) >= 0.0 else -q) * weight
                edge = min(frame - first, last - frame) + 1
                blend = 1.0 if edge > window else smoothstep(edge / (window + 1.0))
                q = q0.slerp(total.normalized(), blend)
                largest = max(largest, math.degrees(q0.rotation_difference(q).angle))
                channels.set(frame, q)
            report['max_change_deg'][name] = round(largest, 4)
        self.finish()
        return report

    def pin(self, entry, base):
        """Solve one pin on the current animation; returns the state the report needs."""
        first, last = entry['frames'][0] - base, entry['frames'][1] - base
        frames = range(entry['window'][0] - base, entry['window'][1] + 1 - base)
        span = range(first, last + 1)
        root, mid = entry['chain_bones'][-2:]
        shoulder = entry['chain_bones'][0] if len(entry['chain_bones']) > 2 else None
        end, effector, target = entry['end_bone'], entry['effector'], entry['target']
        world = target == SPEC.WORLD
        parent = self.arm.data.bones[entry['chain_bones'][0]].parent
        names = set(entry['chain_bones']) | {end, effector} | ({parent.name} if parent else set()) | (set() if world else {target})
        pose = self.sample(frames, names)
        position = lambda frame, name: pose[frame][name].translation
        rotation = lambda frame, name: pose[frame][name].to_quaternion()

        offset = Vector(entry['offset']) / self.unit if entry['offset'] is not None else None
        if world:
            anchor = sum((position(frame, effector) for frame in span), Vector()) / len(span)
            reference = min(span, key=lambda frame: (position(frame, effector) - anchor).length)
            if offset is not None:
                anchor = anchor + offset
            local = None
        else:
            reference = min(span, key=lambda frame: (position(frame, effector) - position(frame, target)).length)
            local = pose[reference][target].inverted() @ position(reference, effector)
            if offset is not None:
                local = offset
            elif entry['distance'] is not None:
                if local.length < 1e-9:
                    raise FixError(f'{effector} and {target} coincide at frame {reference + base}; give an explicit offset')
                local = local.normalized() * (entry['distance'] / self.unit)

        def desired(frame, poses):
            """Wanted effector location and world rotation at frame for the target pose in poses."""
            if world:
                place = anchor
            else:
                place = poses[frame][target] @ local
            if entry['rotation'] == 'keep':
                turn = poses[frame][effector].to_quaternion()
            elif world:
                turn = rotation(reference, effector)
            else:
                turn = poses[frame][target].to_quaternion() @ rotation(reference, target).inverted() @ rotation(reference, effector)
            return place, turn

        bends = {}
        for frame in frames:
            S, E, W = position(frame, root), position(frame, mid), position(frame, end)
            axis = (W - S).normalized()
            bend = (E - S) - axis * (E - S).dot(axis)
            if bend.length > BENT * ((E - S).length + (W - E).length):
                bends[frame] = rotation(frame, root).inverted() @ bend.normalized()
        fallback = 0
        edited = entry['chain_bones'] + [end]
        rests = {name: self.relative_rest(name) for name in edited}
        channels = {name: self.channel(name) for name in edited}
        for item in channels.values():
            item.prepare(frames)
        failures = []
        shoulder_turn = 0.0
        for frame in frames:
            weight = SPEC.ease_weight(frame + base, entry['frames'], entry['ease'])
            S, E, W = position(frame, root), position(frame, mid), position(frame, end)
            axis = (W - S).normalized()
            if frame in bends:
                bend = rotation(frame, root) @ bends[frame]
            else:
                fallback += 1
                near = min(bends, key=lambda other: abs(other - frame)) if bends else None
                bend = rotation(frame, root) @ bends[near] if near is not None else pose[frame][mid].to_3x3().normalized().col[2]
            bend = (bend - axis * bend.dot(axis)).normalized()
            place, turn = desired(frame, pose)
            # The end bone carries the effector (itself, or ToeBase's parent Foot) with the effector's original local pose.
            relative = pose[frame][end].inverted() @ pose[frame][effector]
            wanted_end = loc_rot(place, turn) @ relative.inverted()
            goal = W.lerp(wanted_end.translation, weight)
            end_turn = rotation(frame, end).slerp(wanted_end.to_quaternion(), weight)
            new = {}
            assist = Matrix.Identity(3)
            if shoulder:
                C = position(frame, shoulder)
                assist = shoulder_assist(C, S, goal, (E - S).length + (W - E).length)
                S, E, W, bend = C + assist @ (S - C), C + assist @ (E - C), C + assist @ (W - C), assist @ bend
                new[shoulder] = loc_rot(C, assist @ pose[frame][shoulder].to_3x3().normalized())
                shoulder_turn = max(shoulder_turn, assist.to_quaternion().angle)
            R1, R2, new_E, new_W, clamped = solve_two_bone(S, E, W, goal, bend)
            if clamped and (new_W - goal).length * self.unit > 1e-4:
                failures.append((frame + base, (new_W - goal).length * self.unit))
            new[root] = loc_rot(S, R1 @ assist @ pose[frame][root].to_3x3().normalized())
            new[mid] = loc_rot(new_E, R2 @ assist @ pose[frame][mid].to_3x3().normalized())
            new[end] = loc_rot(new_W, end_turn)
            above = pose[frame][parent.name] if parent else Matrix.Identity(4)
            for name in edited:
                channels[name].set(frame, ((above @ rests[name]).inverted() @ new[name]).to_quaternion())
                above = new[name]
        self.finish()
        return dict(entry=entry, reference=reference, anchor=anchor if world else None, local=local, desired=desired,
                    frames=frames, span=span, failures=failures, fallback=fallback, shoulder_turn=shoulder_turn if shoulder else None)

    def pin_report(self, state, before, after, base):
        entry, span = state['entry'], state['span']
        effector, target, mid = entry['effector'], entry['target'], entry['chain_bones'][-1]
        world = target == SPEC.WORLD
        meters = lambda vector: vector.length * self.unit

        def distances(poses):
            values = {frame: meters(poses[frame][effector].translation - (state['anchor'] if world else poses[frame][target].translation))
                      for frame in span}
            middle = span[len(span) // 2]
            return dict(start=round(values[span[0]], 5), mid=round(values[middle], 5), end=round(values[span[-1]], 5),
                        min=round(min(values.values()), 5), max=round(max(values.values()), 5))

        errors = [meters(after[frame][effector].translation - state['desired'](frame, after)[0]) for frame in span]
        shift = max(meters(after[frame][mid].translation - before[frame][mid].translation) for frame in state['frames'])
        report = dict(effector=effector, target=target, frames=entry['frames'], ease=entry['ease'], rotation=entry['rotation'],
                      chain_bones=entry['chain_bones'], end_bone=entry['end_bone'], reference_frame=state['reference'] + base,
                      distance_before=distances(before), distance_after=distances(after),
                      max_error=round(max(errors), 5), mid_joint=mid, mid_joint_max_shift=round(shift, 5),
                      frames_changed=[state['frames'][0] + base, state['frames'][-1] + base], frames_changed_count=len(state['frames']),
                      reach_failures=dict(frames=len(state['failures']), max_residual=round(max((r for _, r in state['failures']), default=0.0), 5),
                                          first_frames=[frame for frame, _ in state['failures'][:10]]),
                      pole_fallback_frames=state['fallback'])
        if state['shoulder_turn'] is not None:
            report['shoulder_max_turn_deg'] = round(math.degrees(state['shoulder_turn']), 3)
        if world:
            report['anchor'] = [round(value * self.unit, 6) for value in state['anchor']]
            report['desired_distance'] = 0.0
        else:
            report['offset'] = [round(value * self.unit, 6) for value in state['local']]
            report['desired_distance'] = round(state['local'].length * self.unit, 6)
        return report


def rescale_keys(curve, factor):
    """Multiply key times by factor (handles keep their relative offsets scaled); keys landing within 0.01 of a frame snap to it."""
    points = curve.keyframe_points
    values = {}
    for name in ('co', 'handle_left', 'handle_right'):
        values[name] = np.empty(2 * len(points))
        points.foreach_get(name, values[name])
        values[name] = values[name].reshape(-1, 2)
    old = values['co'][:, 0].copy()
    new = old * factor
    near = np.round(new)
    new = np.where(np.abs(new - near) < 0.01, near, new)
    for name in ('handle_left', 'handle_right'):
        values[name][:, 0] = new + (values[name][:, 0] - old) * factor
    values['co'][:, 0] = new
    for name, array in values.items():
        points.foreach_set(name, array.ravel())
    curve.update()


def import_input(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Same options as the app's character import; leaf (*_end) bones stay real bones so they export unchanged.
    try:
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False, use_prepost_rot=True, ignore_leaf_bones=False, anim_offset=0.0)
    except RuntimeError as exc:
        raise FixError(f'Cannot import {path.name}: {str(exc).strip()}') from None
    arms = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    if len(arms) != 1:
        raise FixError(f'Expected one armature in the input, found {len(arms)}')
    arm = arms[0]
    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        raise FixError('The input armature has no animation')
    # The importer converts key times with the rounded frame rate (29.97 -> 30); the exporter writes frame / exact rate.
    # Rescaling puts file key i back on frame i, so a non-integer rate round-trips without resampling.
    factor = 1.0 / bpy.context.scene.render.fps_base
    if abs(factor - 1.0) > 1e-12:
        for act in bpy.data.actions:
            for curve in act.fcurves:
                rescale_keys(curve, factor)
    ranges = [act.frame_range for act in bpy.data.actions]
    return arm, action, int(round(min(r[0] for r in ranges))), int(round(max(r[1] for r in ranges)))


REST_VALUES = {'location': (0.0, 0.0, 0.0), 'rotation_quaternion': (1.0, 0.0, 0.0, 0.0), 'rotation_euler': (0.0, 0.0, 0.0),
               'rotation_axis_angle': (0.0, 0.0, 1.0, 0.0), 'scale': (1.0, 1.0, 1.0)}


def hold_rest_before(action, first):
    """Key the rest pose at first - 1 and make it current.

    Blender's FBX exporter bakes the scene range but writes bone node defaults (the rest pose a re-import uses when the
    file has no bind pose) from the pose at the frame current when export starts. A key at `first` holding its evaluated
    value keeps every exported frame unchanged.
    """
    for curve in action.fcurves:
        channel = curve.data_path.rsplit('.', 1)[-1]
        if not curve.data_path.startswith('pose.bones[') or channel not in REST_VALUES:
            continue
        # No key lies before `first` (the earliest key of all actions), so a key at `first` is the curve's first point.
        points = curve.keyframe_points
        if not len(points) or abs(points[0].co.x - first) > 1e-4:
            curve.keyframe_points.insert(first, curve.evaluate(first), options={'FAST'}).interpolation = 'LINEAR'
        curve.keyframe_points.insert(first - 1, REST_VALUES[channel][curve.array_index], options={'FAST'}).interpolation = 'CONSTANT'
        curve.update()
    bpy.context.scene.frame_set(first - 1)


def export_output(arm, path):
    bpy.ops.object.select_all(action='DESELECT')
    types = {'ARMATURE', 'MESH', 'EMPTY'}
    for obj in bpy.context.scene.objects:
        if obj.type in types:
            obj.select_set(True)
    bpy.context.view_layer.objects.active = arm
    # The app's character FBX export settings, except add_leaf_bones: the imported *_end bones are already present.
    bpy.ops.export_scene.fbx(filepath=str(path), use_selection=True, object_types=types, add_leaf_bones=False,
                             bake_anim=True, bake_anim_use_all_bones=False, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                             bake_anim_simplify_factor=0., use_mesh_modifiers=True, path_mode='COPY', embed_textures=True)


def run(request):
    started = time.perf_counter()
    source = Path(request['input'])
    progress('Importing ' + source.name)
    arm, action, first, last = import_input(source)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = first, last
    base = SPEC.FRAME_BASE
    fps = scene.render.fps / scene.render.fps_base
    skeleton = SPEC.Skeleton({bone.name: bone.parent.name if bone.parent else None for bone in arm.data.bones})
    plan = SPEC.plan(request['spec'], skeleton, SPEC.Clip(first + base, last + base, fps))
    fixer = Fixer(arm, action, first, last)
    watched = sorted({frame - base for pin in plan['pins'] for frame in range(pin['window'][0], pin['window'][1] + 1)})
    before = fixer.sample(watched)
    smooth = []
    for i, entry in enumerate(plan['smooth']):
        progress(f'Smoothing {i + 1}/{len(plan["smooth"])}: {", ".join(entry["bones"])} frames {entry["frames"][0]}-{entry["frames"][1]}')
        smooth.append(fixer.smooth(entry, base))
    states = []
    for i, entry in enumerate(plan['pins']):
        progress(f'Pin {i + 1}/{len(plan["pins"])}: {entry["effector"]} -> {entry["target"]} frames {entry["frames"][0]}-{entry["frames"][1]}')
        states.append(fixer.pin(entry, base))
    after = fixer.sample(watched)
    pins = [fixer.pin_report(state, before, after, base) for state in states]
    progress('Exporting')
    hold_rest_before(action, first)
    export_output(arm, request['output'])
    return dict(ok=True, input=str(source), armature=arm.name, bones=len(arm.data.bones), fps=fps,
                frames=[first + base, last + base], frame_base=base, pins=pins, smooth=smooth,
                blender_seconds=round(time.perf_counter() - started, 2))


def main():
    request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
    try:
        result = run(request)
    except SPEC.SpecError as exc:
        result = dict(ok=False, usage=True, error=f'Invalid fix spec: {exc}')
    except FixError as exc:
        result = dict(ok=False, error=str(exc))
    Path(request['result']).write_text(json.dumps(result, indent=2), encoding='utf-8')
    progress('Complete' if result['ok'] else 'Failed: ' + result['error'])


if __name__ == '__main__':
    main()
