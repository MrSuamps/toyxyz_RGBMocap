"""Blender-side FBX pose dump for anatomy checks: per-frame world head positions and bone rotations of every bone.

Run by the bundled Blender: blender -b --factory-startup --disable-autoexec --python blender_pose.py -- FBX OUT.npz
Writes OUT.npz (names, parents, heads, rotations, rest_heads, rest_rotations, frames, fps) or OUT.json with an error.
Positions and rotations are converted to the motion's Y-up world; rotations keep the bone's own local axes.
"""
import json
import sys
from pathlib import Path


def strip(name):
    return name.rsplit(':', 1)[-1]


def fail(out, message):
    Path(out).with_suffix('.json').write_text(json.dumps(dict(error=message)), encoding='utf-8')
    print('POSE: ' + message, flush=True)
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
    armatures = [item for item in bpy.context.scene.objects if item.type == 'ARMATURE']
    if not armatures:
        fail(out, 'The FBX has no armature.')
    best = max(armatures, key=lambda item: len(item.pose.bones))
    action = best.animation_data.action if best.animation_data else None
    if action is None:
        fail(out, f'Armature {best.name} has no animation.')
    bones = list(best.pose.bones)
    names = [strip(bone.name) for bone in bones]
    index = {bone.name: i for i, bone in enumerate(bones)}
    parents = np.array([index[bone.parent.name] if bone.parent else -1 for bone in bones], np.int64)
    # Blender Z-up -> Y-up: (x, y, z) -> (x, z, -y).
    convert = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    def split(matrix):
        m = np.array(matrix, np.float64)
        rotation = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0, keepdims=True)
        return convert @ m[:3, 3], convert @ rotation

    world = np.array(best.matrix_world, np.float64)
    rest = [split(world @ np.array(bone.bone.matrix_local, np.float64)) for bone in bones]
    first, last = (int(round(value)) for value in action.frame_range)
    scene = bpy.context.scene
    heads = np.zeros((last - first + 1, len(bones), 3))
    rotations = np.zeros((last - first + 1, len(bones), 3, 3))
    for row, frame in enumerate(range(first, last + 1)):
        scene.frame_set(frame)
        world = np.array(best.matrix_world, np.float64)
        for column, bone in enumerate(bones):
            heads[row, column], rotations[row, column] = split(world @ np.array(bone.matrix, np.float64))
    fps = scene.render.fps / scene.render.fps_base
    np.savez(out, names=np.array(names), parents=parents, heads=heads, rotations=rotations,
             rest_heads=np.array([r[0] for r in rest]), rest_rotations=np.array([r[1] for r in rest]),
             frames=np.arange(first, last + 1), fps=np.float64(fps), armature=np.array(best.name))
    print(f'POSE: {len(heads)} frames {first}-{last} at {fps:g} fps, {len(bones)} bones from {best.name}', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    main(sys.argv[sys.argv.index('--') + 1:])
