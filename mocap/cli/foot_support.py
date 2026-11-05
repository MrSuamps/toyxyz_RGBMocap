"""`foot_slide` at the actual support point: on pointe frames the toe-box tip of the SMPL-X mesh, else review's joints.

On pointe the ankle and toe joints sit 4-15 cm above the toe box and swing with the shin, and `correct` may lose a lock
while the toe box stays on the floor (ExMR contact confidence drops on a toe-box support). So on pointe frames of a
project result, contact also covers dropouts of up to 0.5 s between two contacts of the same foot while the toe box stays
within 3 cm of the floor, and the slide is the toe-box tip's horizontal speed. numpy loads lazily.
"""
from __future__ import annotations

import json

DROPOUT_SECONDS = 0.5
ON_FLOOR_M = 0.03
SIDES = ('Left', 'Right')


def pointe_mask(project, kind, count):
    """(count, 2) frames inside an enabled pointe entry of `<kind>_correction.json`, or None without one."""
    import numpy as np
    try:
        report = json.loads((project / f'{kind}_correction.json').read_text(encoding='utf-8'))
        entries = report.get('foot_articulation', {}).get('entries', [])
    except (OSError, ValueError, AttributeError):
        return None
    mask = np.zeros((count, 2), bool)
    for entry in entries:
        if entry.get('mode') == 'pointe' and 'skipped' not in entry and entry.get('strength', 1) >= .5:
            mask[max(0, int(entry['start'])):min(count, int(entry['end'])), SIDES.index(entry['foot'][:-4])] = True
    return mask if mask.any() else None


def toe_boxes(body, shapes):
    """Per foot, the toe-box vertex ids `foot_articulation.FootModel.from_geometry` takes the pointe support from."""
    import numpy as np
    shaped = body.template + body.shapedirs @ shapes
    boxes = []
    for ankle, toe in ((7, 10), (8, 11)):
        ids = np.flatnonzero(body.weights[:, ankle] + body.weights[:, toe] > .6)
        toes = ids[body.weights[ids, toe] > .5]
        boxes.append(toes[shaped[toes, 2] >= np.quantile(shaped[toes, 2], .85)])
    return boxes


def posed_vertices(mesh, rows, ids):
    """(len(rows), len(ids), 3) posed vertices `ids` of the mesh rows, the maths of `smplx_body.Body.vertices` on a subset."""
    import numpy as np
    body, state = mesh.body, mesh.state
    index = mesh.index[rows]
    columns = (ids[:, None] * 3 + np.arange(3)).reshape(-1)
    shaped = body.template[ids][None] + np.einsum('vci,ni->nvc', body.shapedirs[ids], state['shapes'][index])
    features = (state['local'][index, 1:] - np.eye(3)).reshape(len(index), -1)
    posed = shaped + (features @ body.posedirs[:, columns]).reshape(len(index), -1, 3)
    world, transl = state['world'][index], state['transl'][index]
    offset = state['joints'][index] - transl[:, None] - np.einsum('njab,njb->nja', world, state['rest'][index])
    rotation = np.einsum('vj,njab->nvab', body.weights[ids], world)
    return np.einsum('nvab,nvb->nva', rotation, posed) + np.einsum('vj,nja->nva', body.weights[ids], offset) + transl[:, None]


def sole_heights(mesh, chunk=256):
    """(rows, 2) lowest Y of the left / right foot skin (vertices dominated by the ankle or toe joint) on every mesh row."""
    import numpy as np
    dominant = mesh.body.weights.argmax(1)
    feet = [np.flatnonzero(np.isin(dominant, joints)) for joints in ((7, 10), (8, 11))]
    ids = np.concatenate(feet)
    out = np.empty((len(mesh.index), 2))
    for start in range(0, len(mesh.index), chunk):
        rows = np.arange(start, min(start + chunk, len(mesh.index)))
        y = posed_vertices(mesh, rows, ids)[..., 1]
        out[rows, 0], out[rows, 1] = y[:, :len(feet[0])].min(1), y[:, len(feet[0]):].min(1)
    return out


def tips(mesh, rows, boxes):
    """(len(rows), 2, 3) toe-box tip per foot: mean XZ of the box, lowest Y (the correction's pointe support point)."""
    import numpy as np
    index = mesh.index[rows]
    vertices = posed_vertices(mesh, rows, np.concatenate(boxes))
    out, first = np.empty((len(index), 2, 3)), 0
    for side, box in enumerate(boxes):
        part = vertices[:, first:first + len(box)]
        first += len(box)
        out[:, side] = part.mean(1)
        out[:, side, 1] = part[..., 1].min(1)
    return out


def check_foot_slide(ctx, project, kind, contacts, floor, pose=None):
    """review's `foot_slide` issues, measured at the toe-box tip on the pointe frames of a project result with a mesh."""
    import numpy as np
    from . import review
    pointe = pointe_mask(project, kind, int(ctx.frames[-1]) + 1)
    mesh = None if pose is None or pose.mesh is None else pose.rows(ctx.frames).mesh
    if pointe is None or mesh is None or not pointe[ctx.frames].any():
        return review.check_foot_slide(ctx, contacts)
    pointe = pointe[ctx.frames]
    n = len(ctx.frames)
    tip = tips(mesh, np.arange(n), toe_boxes(mesh.body, mesh.state['shapes'][mesh.index[0]]))
    on_floor = tip[..., 1] - floor < ON_FLOOR_M
    locked, contacts = contacts, contacts.copy()
    for foot in (0, 1):
        for a, b in review.runs(~contacts[:, foot], 0, 1):
            end = b + 1
            if a == 0 or end == n or end - a > round(DROPOUT_SECONDS * ctx.fps):
                continue
            if pointe[a:end, foot].all() and on_floor[a:end, foot].all():
                contacts[a:end, foot] = True
    joint_speed = review.foot_speeds(ctx)
    tip_speed = np.linalg.norm(np.gradient(tip[..., [0, 2]], axis=0), axis=-1) * ctx.fps if n > 1 else np.zeros((n, 2))
    issues = []
    for foot, side in enumerate(SIDES):
        slowest = np.where(pointe[:, foot], tip_speed[:, foot], joint_speed[:, foot].min(1))
        flag = contacts[:, foot] & (slowest > review.SLIDE_SPEED)
        for a, b in review.runs(flag):
            span = slice(a, b + 1)
            held = pointe[span, foot].mean() >= .5
            if held:
                path, bone, name = tip[span, foot][:, [0, 2]], side + 'ToeBase', side + 'ToeBox'
            else:
                pivot = int(np.argmin(joint_speed[span, foot].mean(0)))
                joint = (7, 10)[pivot] if foot == 0 else (8, 11)[pivot]
                path, bone = ctx.joints[span, joint][:, [0, 2]], side + ('Foot', 'ToeBase')[pivot]
                name = bone
            drift = float(np.linalg.norm(path[-1] - path[0]))
            travel = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            peak = float(slowest[span].max())
            worst = a + int(np.argmax(slowest[span]))
            severity = 'high' if travel > 0.1 or peak > 0.5 else 'medium' if travel > 0.04 else 'low'
            info = review.interval(ctx, a, b)
            issues.append(dict(
                check='foot_slide', severity=severity, score=travel * 100, **info, worst_frame=ctx.video(worst),
                joints=[f'{side.lower()}_ankle', f'{side.lower()}_toe'], effector=bone,
                metrics=dict(frames_flagged=int(flag[span].sum()), peak_speed_mps=peak, mean_speed_mps=float(slowest[span].mean()),
                             travel_m=travel, net_drift_m=drift, held_point=name,
                             lock_dropout_frames=int((contacts[span, foot] & ~locked[span, foot]).sum())),
                message=f'{side.lower()} foot slides {travel * 100:.1f} cm during ground contact (up to {peak:.2f} m/s'
                        + (', toe-box tip on pointe)' if held else ')'),
                fix=dict(pins=[dict(effector=bone, target='world', frames=info['blender_frames'], ease=4)]),
                _draw=dict(smpl=((7, 10) if foot == 0 else (8, 11)), series={'speed m/s': slowest})))
    return issues
