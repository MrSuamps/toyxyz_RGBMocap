"""`foot_orientation` review check: the model's feet against the video's 2D foot keypoints.

Per foot and frame: 2D ankle->toe and heel->toe direction errors, heel lift against the model, observed foot-to-shin angle
in the image, the 2D foot length against its longest possible projection, model plantarflexion and toe flexion (degrees from
the rest foot), toe-tip error. Issues use review's format; `statistics` adds a stats block. numpy loads lazily.
"""
from __future__ import annotations

DIRECTION_DEG = 30.0           # ankle->toe direction error that flags a frame
IN_LINE_DEG = 20.0             # observed foot within this of the shin line (image) looks pointed
LONG_FOOT = 0.6                # observed 2D foot length / longest possible projection; shorter is foreshortened
POINTED_DEG = 50.0             # model plantarflexion below this on a pointed-looking foot is flagged
# ...but only where the projected model foot differs from the video: its ankle->toe line by more than POINTE_DISAGREE_DEG or
# the video heel->toe line points more than HEEL_LIFT_DEG further down (heel raised). A wide stance seen from the front looks
# in line with the shin too; there the flat model foot projects onto the video foot (measured 1-10 deg) and no pointe fits.
POINTE_DISAGREE_DEG = 15.0
HEEL_LIFT_DEG = 15.0
VISIBLE = 0.5
SIDES = ('left', 'right')
BONES = ('LeftFoot', 'RightFoot')
PREFIX = dict(foot_orientation='FO')


def _angle(u, v):
    import numpy as np
    cos = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _down(v):
    """Degrees a 2D image vector points below the horizontal (image y down)."""
    import numpy as np
    return np.degrees(np.arcsin(np.clip(v[..., 1] / np.maximum(np.linalg.norm(v, axis=-1), 1e-9), -1, 1)))


def series(ctx, pose=None):
    """Per-foot arrays over ctx.frames. Without a pose (rotations) the model foot is the ankle->toe joint line."""
    import numpy as np
    from ..foot_articulation import FEET, KEYPOINTS, FootModel, ankle_pitch, toe_pitch
    kp = ctx.keypoints
    sliced = pose.rows(ctx.frames) if pose is not None else None
    model = FootModel.from_joints(sliced.rest) if sliced is not None else None
    out = {}
    for side, (_, knee, ankle, toe) in enumerate(FEET.values()):
        a, big, small, heel = KEYPOINTS[side]
        joints = ctx.joints
        if model is not None:
            m_heel, m_tip = model.points(side, joints[:, ankle], sliced.world[:, ankle], sliced.world[:, toe])
            plantar = ankle_pitch(sliced.local[:, ankle], sliced.rest[toe] - sliced.rest[ankle])
            toes = toe_pitch(sliced.local[:, toe], model.tip[side])
        else:
            m_heel, m_tip = joints[:, ankle], joints[:, toe]
            plantar = toes = np.full(len(joints), np.nan)
        uv = ctx.project(np.stack([m_heel, m_tip, joints[:, knee], joints[:, ankle]], 1))
        both = (kp[:, small, 2] >= VISIBLE) & (kp[:, big, 2] >= VISIBLE)
        k_toe = np.where(both[:, None], (kp[:, big, :2] + kp[:, small, :2]) / 2, np.where((kp[:, big, 2] >= kp[:, small, 2])[:, None],
                                                                                        kp[:, big, :2], kp[:, small, :2]))
        toe_ok = np.maximum(kp[:, big, 2], kp[:, small, 2]) >= VISIBLE
        # The ankle (malleolus) -> toe line is the flagged direction: a raised heel's keypoint drifts, the malleolus does not.
        ok = toe_ok & (kp[:, a, 2] >= VISIBLE)
        heel_ok = toe_ok & (kp[:, heel, 2] >= VISIBLE) & (model is not None)
        shin_ok = ok & (kp[:, 13 + side, 2] >= VISIBLE)
        observed, modelled = k_toe - kp[:, a, :2], uv[:, 1] - uv[:, 3]
        error = np.where(ok, _angle(observed, modelled), np.nan)
        heel_error = np.where(heel_ok, _angle(k_toe - kp[:, heel, :2], uv[:, 1] - uv[:, 0]), np.nan)
        # Image y points down: a heel raised more than the model's tilts the video heel->toe line further down.
        heel_lift = np.where(heel_ok, _down(k_toe - kp[:, heel, :2]) - _down(uv[:, 1] - uv[:, 0]), np.nan)
        in_line = np.where(shin_ok, _angle(observed, kp[:, a, :2] - kp[:, 13 + side, :2]), np.nan)
        model_in_line = _angle(modelled, uv[:, 3] - uv[:, 2])
        # Longest possible 2D ankle->tip line: its 3D length seen square-on at the ankle's depth.
        cam = np.einsum('fij,fj->fi', ctx.rotation, joints[:, ankle]) + ctx.translation
        longest = np.linalg.norm(m_tip - joints[:, ankle], axis=-1) * ctx.K[0, 0] / np.maximum(cam[:, 2], 1e-6)
        length = np.where(ok, np.linalg.norm(observed, axis=-1) / np.maximum(longest, 1e-6), np.nan)
        tip_error = np.where(toe_ok, np.linalg.norm(uv[:, 1] + ctx.offset - k_toe, axis=-1) / ctx.torso, np.nan)
        out[SIDES[side]] = dict(direction_error_deg=error, heel_toe_error_deg=heel_error, heel_lift_deg=heel_lift, observed_in_line_deg=in_line,
                                model_in_line_deg=model_in_line, observed_length_ratio=length, plantarflexion_deg=plantar, toe_flexion_deg=toes,
                                tip_error_torso=tip_error, ok=ok, joints=(ankle, toe), rotations=model is not None)
    return out


def _issue(ctx, side, data, a, b, worst, flag, kind):
    import numpy as np
    from .review import interval
    span = slice(a, b + 1)
    count = int(flag[span].sum())
    err = data['direction_error_deg']; plantar = data['plantarflexion_deg']; toes = data['toe_flexion_deg']
    med = lambda values: float(np.nanmedian(values[span])) if np.isfinite(values[span]).any() else None
    metrics = dict(frames_flagged=count, direction_error_median_deg=med(err), direction_error_peak_deg=float(np.nanmax(err[span]))
                   if np.isfinite(err[span]).any() else None, observed_in_line_median_deg=med(data['observed_in_line_deg']),
                   model_in_line_median_deg=med(data['model_in_line_deg']), heel_toe_error_median_deg=med(data['heel_toe_error_deg']),
                   heel_lift_median_deg=med(data['heel_lift_deg']),
                   observed_length_ratio_median=med(data['observed_length_ratio']), plantarflexion_median_deg=med(plantar),
                   toe_flexion_median_deg=med(toes), tip_error_median_torso=med(data['tip_error_torso']))
    info = interval(ctx, a, b)
    foot = BONES[SIDES.index(side)]
    pf = '' if metrics['plantarflexion_median_deg'] is None else f", model plantarflexion {metrics['plantarflexion_median_deg']:.0f} deg"
    if kind == 'pointe':
        severity = 'medium' if count >= 10 else 'low'
        hit = flag[span]
        lift = data['heel_lift_deg'][span][hit]
        off = (f'heel {np.nanmedian(lift):.0f} deg higher than the model' if np.isfinite(lift).any() and np.nanmedian(lift) > HEEL_LIFT_DEG
               else f'model foot {np.nanmedian(err[span][hit]):.0f} deg off the video direction')
        message = (f'{side} foot looks pointed in the video (in line with the shin, {metrics["observed_in_line_median_deg"]:.0f} deg in the image; '
                   f'{off}) but the model foot is flatter{pf}')
        mode = 'pointe'
    else:
        peak = metrics['direction_error_peak_deg']
        severity = 'high' if count >= 10 and metrics['direction_error_median_deg'] >= 45 else 'medium' if peak >= 45 or count >= 10 else 'low'
        message = f'{side} foot points {metrics["direction_error_median_deg"]:.0f} deg (up to {peak:.0f}) off the video ankle->toe direction{pf}'
        mode = 'match'
    suggestion = dict(foot=foot, mode=mode, profile='dancer' if mode == 'pointe' else 'default', ranges=[[info['frames'][0], info['frames'][1] + 1]])
    hint = (f'add a `{mode}` foot articulation for {foot} on frames {info["frames"][0]}-{info["frames"][1]} '
            '(self_contact_store.save_articulation) and run `correct`')
    score = count * (metrics['direction_error_median_deg'] or 0.0) if kind != 'pointe' else count * 10.0
    return dict(check='foot_orientation', severity=severity, score=float(score), **info, worst_frame=ctx.video(worst),
                joints=[f'{side}_ankle', f'{side}_toe'], effector=foot, metrics=metrics, message=message, fix=None, hint=hint,
                suggestion=suggestion, _draw=dict(smpl=data['joints'], series={'foot dir err deg': err}))


def check(ctx, pose=None):
    """foot_orientation issues: ankle->toe direction runs past DIRECTION_DEG, and pointed-looking feet the model keeps flat."""
    import numpy as np
    from .review import runs
    issues = []
    for side, data in series(ctx, pose).items():
        err = data['direction_error_deg']
        flag = np.nan_to_num(err) > DIRECTION_DEG
        for a, b in runs(flag):
            worst = a + int(np.nanargmax(np.nan_to_num(err[a:b + 1], nan=-1)))
            issues.append(_issue(ctx, side, data, a, b, worst, flag, 'direction'))
        if not data['rotations']:
            continue
        disagrees = (np.nan_to_num(err) > POINTE_DISAGREE_DEG) | (np.nan_to_num(data['heel_lift_deg']) > HEEL_LIFT_DEG)
        pointed = ((np.nan_to_num(data['observed_in_line_deg'], nan=180.) <= IN_LINE_DEG) &
                   (np.nan_to_num(data['observed_length_ratio']) >= LONG_FOOT) & (data['plantarflexion_deg'] < POINTED_DEG) & disagrees)
        for a, b in runs(pointed):
            worst = a + int(np.argmin(data['plantarflexion_deg'][a:b + 1]))
            issues.append(_issue(ctx, side, data, a, b, worst, pointed, 'pointe'))
    return issues


def statistics(ctx, pose=None):
    """Median / p90 per foot of every series, for report stats."""
    import numpy as np
    out = {}
    for side, data in series(ctx, pose).items():
        row = {}
        for name, values in data.items():
            if name in ('ok', 'joints', 'rotations'):
                continue
            finite = values[np.isfinite(values)]
            row[name] = dict(median=float(np.median(finite)), p90=float(np.percentile(finite, 90)), samples=int(len(finite))) if len(finite) else None
        row['rotations'] = data['rotations']
        out[side] = row
    return out
