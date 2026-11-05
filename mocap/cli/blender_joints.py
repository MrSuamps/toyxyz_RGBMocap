"""Blender-side FBX joint dump for `rgbmocap-cli review --motion`: per-frame world positions of the 22 body bone heads.

Run by the bundled Blender: blender -b --factory-startup --disable-autoexec --python blender_joints.py -- FBX OUT.npz
Writes OUT.npz (joints, frames, fps, bones) or OUT.json with an error message. bpy is imported only inside main().
"""
import json
import sys
from pathlib import Path

# FBX bone order == SMPL-X body joints 0-21 (bone head world position == joint position).
BONES = ('Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase',
         'RightToeBase', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand',
         'RightHand')


def strip(name):
    return name.rsplit(':', 1)[-1]


def fail(out, message, **extra):
    Path(out).with_suffix('.json').write_text(json.dumps(dict(error=message, **extra)), encoding='utf-8')
    print('REVIEW: ' + message, flush=True)
    raise SystemExit(2)


def main(argv):
    import bpy
    import numpy as np
    source, out = argv[0], argv[1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=source)
    except RuntimeError as error:
        fail(out, f'Blender could not import the FBX: {error}')
    wanted = {name.casefold(): name for name in BONES}
    best, found = None, {}
    for item in bpy.context.scene.objects:
        if item.type != 'ARMATURE':
            continue
        names = {}
        for bone in item.pose.bones:
            key = strip(bone.name).casefold()
            if key in wanted and wanted[key] not in names:
                names[wanted[key]] = bone.name
        if len(names) > len(found):
            best, found = item, names
    if best is None:
        fail(out, 'The FBX has no armature.')
    missing = [name for name in BONES if name not in found]
    if missing:
        fail(out, f'Armature {best.name} lacks body bones: {", ".join(missing)}', missing=missing, armature=best.name)
    action = best.animation_data.action if best.animation_data else None
    if action is None:
        fail(out, f'Armature {best.name} has no animation.')
    first, last = (int(round(value)) for value in action.frame_range)
    scene = bpy.context.scene
    bones = [best.pose.bones[found[name]] for name in BONES]
    joints = np.zeros((last - first + 1, len(BONES), 3), np.float64)
    for row, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        world = best.matrix_world
        for column, bone in enumerate(bones):
            p = world @ bone.head
            # Blender Z-up -> the motion's Y-up world: (x, y, z) -> (x, z, -y).
            joints[row, column] = (p.x, p.z, -p.y)
    fps = scene.render.fps / scene.render.fps_base
    np.savez(out, joints=joints, frames=np.arange(first, last + 1), fps=np.float64(fps), bones=np.array([found[name] for name in BONES]))
    print(f'REVIEW: {len(joints)} frames {first}-{last} at {fps:g} fps from {best.name}', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    main(sys.argv[sys.argv.index('--') + 1:])
