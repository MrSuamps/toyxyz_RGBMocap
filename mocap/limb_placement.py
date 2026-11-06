"""Limb placement ("match 2D"): per-interval IK of one arm or leg toward the video's 2D keypoints.

Runs in correction.run on the corrected pose, after the foot plan and hand lock and before self-contacts (so contacts are
solved on the placed limb). The upper joint gets a free rotation and the middle joint a hinge change about its
anatomical axis (`mixamo_rig.HINGES`); the hand or foot keeps its source world rotation, like self-contacts and leg IK.
Depth is not observable in one view: a prior keeps the middle and end joints near their source camera depth.
"""
import math
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

METHOD = 'Limb placement: per-frame chain IK to 2D keypoints with depth prior, hinge limits and rotation-space easing'
# chain: (parent, upper, middle, end) SMPL-X joints, COCO-63 (middle, end) keypoints, hinge axis in the upper joint frame
CHAINS = {'LeftLeg': ((0, 1, 4, 7), (13, 15), (1., 0., 0.)), 'RightLeg': ((0, 2, 5, 8), (14, 16), (1., 0., 0.)),
          'LeftArm': ((13, 16, 18, 20), (7, 9), (0., -1., 0.)), 'RightArm': ((14, 17, 19, 21), (8, 10), (0., 1., 0.))}
# Signed hinge bend (deg, 0 straight): knees and elbows never hyperextend and stay off the review's locked-limb band.
BEND_LIMITS = {'Leg': (5., 150.), 'Arm': (10., 150.)}
# Sapiens scores are raw heatmap maxima; confidence ramps from 0 at VISIBLE to 1 at VISIBLE + 0.15.
VISIBLE = .5
# Torso alignment keypoints (COCO, SMPL-X, weight), as review's 2D alignment: SMPL-X hip centres sit off the COCO hips.
ALIGN = ((5, 16, 1.), (6, 17, 1.), (11, 1, .3), (12, 2, .3))
DEPTH_SIGMA_M = .3            # camera-depth change costing as much as one torso length of 2D error
PRIOR_PER_RAD = .05
SMOOTH_WEIGHT = 3.            # second difference of the correction (rad) at 30 fps
LIMIT_PER_RAD = 20.
ROBUST_TORSO = .3             # soft-L1 scale of all residuals
EASE_DEG_PER_FRAME = 3.
MIN_EASE, MAX_EASE = 6, 24
COVERAGE_SECONDS = .15
LEG_JOINTS = ((1, 4, 7, 10), (2, 5, 8, 11))


def quat_matrix(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def matrix_quat(m):
    return Rotation.from_matrix(np.asarray(m).reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(np.shape(m)[:-2] + (4,))


def rotvec_matrix(v):
    """Rodrigues for (..., 3) rotation vectors."""
    v = np.asarray(v, float); angle = np.linalg.norm(v, axis=-1)[..., None, None]
    k = v / np.maximum(angle[..., 0], 1e-12)
    zero = np.zeros(v.shape[:-1])
    K = np.stack([np.stack([zero, -k[..., 2], k[..., 1]], -1), np.stack([k[..., 2], zero, -k[..., 0]], -1),
                  np.stack([-k[..., 1], k[..., 0], zero], -1)], -2)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def bend(upper, middle, end, upper_world, axis):
    """Signed hinge bend (rad) as anatomy.hinge_bend: 0 straight, positive flexed, negative hyperextended."""
    u = unit(middle - upper); v = unit(end - middle)
    h = np.einsum('tij,j->ti', upper_world, axis)
    h = unit(h - u * (h * u).sum(-1, keepdims=True))
    return np.arctan2((np.cross(u, v) * h).sum(-1), (u * v).sum(-1))


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_weights(lo, hi, a, b, ease):
    """Smoothstep up over [a-ease, a), 1 on [a, b), down over [b, b+ease); every ease frame is strictly inside (0, 1)."""
    t = np.arange(lo, hi, dtype=float); x = np.ones(len(t))
    x[t < a] = (t[t < a] - (a - ease - 1)) / (ease + 1)
    x[t >= b] = (b + ease - t[t >= b]) / (ease + 1)
    return smoothstep(x)


def lock_gate(locked, ease):
    """1 far from locked frames, 0 on them, smoothstep in between over `ease` frames."""
    if not locked.any():
        return np.ones(len(locked))
    index = np.flatnonzero(locked); t = np.arange(len(locked))
    distance = np.abs(t[:, None] - index[None]).min(1)
    return smoothstep(distance / (ease + 1.))


def slerp(q0, q1, w):
    from .self_contact import slerp as shortest
    return shortest(q0, q1, w)


class Camera:
    """Per-frame world -> OpenCV camera and COCO-63 keypoints; points are in the correction frame (shared floor applied)."""

    def __init__(self, keypoints, rotation, translation, K, transform=None):
        self.keypoints = np.asarray(keypoints, float); self.rotation = np.asarray(rotation, float)
        self.translation = np.asarray(translation, float); self.K = np.asarray(K, float)
        self.transform = np.eye(4) if transform is None else np.asarray(transform, float)

    @classmethod
    def load(cls, project, pelvis, transform=None, roots=None):
        """The cameras review projects with: a static ExMR take's one fixed camera (camera_reference.exmr_view), else per-frame
        pairings. Both follow the change of `roots` (correction frame) from the original root, as review's camera follows the
        corrected result's."""
        from .camera_reference import exmr_view
        from .foot_articulation import Views
        views = Views.load(project, pelvis, transform)
        rotation, translation = views.rotation, views.translation
        delta = None
        if roots is not None:
            with np.load(Path(project) / 'initial_joints.npz', allow_pickle=False) as data:
                original = np.asarray(data['joints'][:, 0], float)
            local = (np.asarray(roots, float) - views.transform[:3, 3]) @ views.transform[:3, :3]
            delta = local - original if local.shape == original.shape == translation.shape else None
        view = exmr_view(project, shift=delta)
        if isinstance(view, tuple):
            rotation = np.broadcast_to(view[0], rotation.shape).copy(); translation = np.broadcast_to(view[1], translation.shape).copy()
        elif view == 'moving' and delta is not None:
            translation = translation - np.einsum('fij,fj->fi', rotation, delta)
        return cls(views.keypoints, rotation, translation, views.K, views.transform)

    def camera(self, points, frames):
        local = (np.asarray(points, float) - self.transform[:3, 3]) @ self.transform[:3, :3]
        shape = local.shape; flat = local.reshape(shape[0], -1, 3)
        return (np.einsum('fij,fkj->fki', self.rotation[frames], flat) + self.translation[frames][:, None]).reshape(shape)

    def pixels(self, cam):
        z = cam[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = np.stack([self.K[0, 0] * cam[..., 0] / z + self.K[0, 2], self.K[1, 1] * cam[..., 1] / z + self.K[1, 2]], -1)
        uv[z <= 1e-6] = np.nan
        return uv

    def project(self, points, frames):
        return self.pixels(self.camera(points, frames))


def fill(values):
    good = np.isfinite(values)
    if not good.any():
        return values
    index = np.arange(len(values))
    return np.interp(index, index[good], values[good])


def weighted_median(values, weights):
    order = np.argsort(values); cumulative = np.cumsum(weights[order])
    return values[order][np.searchsorted(cumulative, cumulative[-1] / 2)]


def alignment(camera, joints, frames):
    """(offset px (m, 2), torso px (m,)) per frame, as review's Context: torso keypoints minus projected joints."""
    from scipy.ndimage import median_filter
    kp = camera.keypoints[frames]; uv = camera.project(joints, frames); m = len(frames)
    offset = np.full((m, 2), np.nan)
    for f in range(m):
        rows = [(c, s, w) for c, s, w in ALIGN if kp[f, c, 2] >= VISIBLE and np.isfinite(uv[f, s]).all()]
        if rows:
            delta = np.array([kp[f, c, :2] - uv[f, s] for c, s, _ in rows]); weights = np.array([w for _, _, w in rows])
            offset[f] = [weighted_median(delta[:, 0], weights), weighted_median(delta[:, 1], weights)]
    offset = np.stack([fill(offset[:, 0]), fill(offset[:, 1])], 1) if np.isfinite(offset[:, 0]).any() else np.zeros((m, 2))
    ok = (kp[:, [5, 6, 11, 12], 2] >= VISIBLE).all(1)
    torso = np.where(ok, np.linalg.norm((kp[:, 5, :2] + kp[:, 6, :2]) / 2 - (kp[:, 11, :2] + kp[:, 12, :2]) / 2, axis=1), np.nan)
    if np.isfinite(torso).any():
        torso = median_filter(fill(torso), size=9, mode='nearest')
    else:
        torso = np.linalg.norm((uv[:, 16] + uv[:, 17]) / 2 - (uv[:, 1] + uv[:, 2]) / 2, axis=1)
    return offset, np.maximum(np.nan_to_num(fill(torso), nan=1.), 1.)


def observations(camera, joints, frames, chain, offset, torso):
    """(targets px (m, 2, 2), confidence (m, 2), swapped (m,)) for the chain's middle and end keypoints.

    Frames whose left/right keypoints match the model clearly better when exchanged (detector swap) use the other side's.
    """
    (_, _, middle, end), (km, ke), _ = CHAINS[chain]
    other = ('Right' + chain[4:]) if chain.startswith('Left') else ('Left' + chain[5:])
    (_, _, om, oe), (okm, oke), _ = CHAINS[other]
    kp = camera.keypoints[frames]; uv = camera.project(joints, frames) + offset[:, None]
    seen = kp[:, [km, ke], :2]; seen_other = kp[:, [okm, oke], :2]
    score = np.minimum(kp[:, [km, ke], 2], kp[:, [okm, oke], 2])
    usable = (score >= VISIBLE).all(1)
    direct = (np.linalg.norm(seen - uv[:, [middle, end]], axis=-1) + np.linalg.norm(seen_other - uv[:, [om, oe]], axis=-1)).sum(1)
    cross = (np.linalg.norm(seen_other - uv[:, [middle, end]], axis=-1) + np.linalg.norm(seen - uv[:, [om, oe]], axis=-1)).sum(1)
    direct, cross = direct / (4 * torso), cross / (4 * torso)
    swapped = usable & np.isfinite(direct) & np.isfinite(cross) & (cross < .5 * direct) & (direct - cross > .15)
    targets = np.where(swapped[:, None, None], seen_other, seen)
    scores = np.where(swapped[:, None], kp[:, [okm, oke], 2], kp[:, [km, ke], 2])
    conf = np.clip((scores - VISIBLE) / .15, 0, 1) * np.isfinite(targets).all(-1)
    return np.nan_to_num(targets), conf, swapped


class ChainModel:
    """Chain FK from per-frame corrections x = (rotation vector of the upper joint in its parent frame, hinge angle)."""

    def __init__(self, rest, root, quat, chain):
        from .correction import fk
        (parent, upper, middle, end), _, axis = CHAINS[chain]
        self.joints, self.world = fk(rest, root, quat)
        local = quat_matrix(quat)
        self.parent_world = self.world[:, parent]; self.upper_pos = self.joints[:, upper]
        self.upper_local, self.middle_local = local[:, upper], local[:, middle]
        self.bone_upper = rest[middle] - rest[upper]; self.bone_middle = rest[end] - rest[middle]
        self.axis = np.asarray(axis, float)
        self.ids = (upper, middle, end)

    def locals(self, x):
        x = np.asarray(x, float).reshape(-1, 4)
        return rotvec_matrix(x[:, :3]) @ self.upper_local, rotvec_matrix(x[:, 3:] * self.axis) @ self.middle_local

    def pose(self, x):
        """(middle, end positions (m, 3) each, upper world rotations, bend rad)."""
        upper_local, middle_local = self.locals(x)
        upper_world = self.parent_world @ upper_local
        middle = self.upper_pos + np.einsum('tij,j->ti', upper_world, self.bone_upper)
        end = middle + np.einsum('tij,j->ti', upper_world @ middle_local, self.bone_middle)
        return middle, end, upper_world, bend(self.upper_pos, middle, end, upper_world, self.axis)


def sparsity(m, per_frame, smooth_rows):
    """Jacobian sparsity: per_frame residuals per frame on that frame's 4 variables, then 4 smoothness rows per inner frame."""
    rows = m * per_frame + smooth_rows * 4 * max(m - 2, 0)
    S = lil_matrix((rows, 4 * m), dtype=int)
    for t in range(m):
        S[t * per_frame:(t + 1) * per_frame, 4 * t:4 * t + 4] = 1
    base = m * per_frame
    for t in range(1, m - 1):
        for k in range(4):
            S[base + 4 * (t - 1) + k, [4 * (t - 1) + k, 4 * t + k, 4 * (t + 1) + k]] = 1
    return S


def solve_window(model, camera, frames, targets, conf, offset, torso, limits, fps):
    """Least-squares corrections (m, 4) for the window; frames without confident keypoints follow the smoothness."""
    m = len(frames)
    middle0, end0, _, bend0 = model.pose(np.zeros((m, 4)))
    z0 = camera.camera(np.stack([middle0, end0], 1), frames)[..., 2]
    lo = np.minimum(np.deg2rad(limits[0]), bend0); hi = np.maximum(np.deg2rad(limits[1]), bend0)
    weight = np.sqrt(conf)[..., None]
    smooth = SMOOTH_WEIGHT * (fps / 30.) ** 2

    def residuals(x):
        x = x.reshape(m, 4)
        middle, end, _, b = model.pose(x)
        cam = camera.camera(np.stack([middle, end], 1), frames)
        uv = np.nan_to_num(camera.pixels(cam)) + offset[:, None]
        r2d = (uv - targets) / torso[:, None, None] * weight
        rz = (cam[..., 2] - z0) / DEPTH_SIGMA_M
        rl = LIMIT_PER_RAD * (np.maximum(lo - b, 0) + np.maximum(b - hi, 0))
        per = np.concatenate([r2d.reshape(m, 4), rz, rl[:, None], PRIOR_PER_RAD * x], 1)
        second = smooth * (x[2:] - 2 * x[1:-1] + x[:-2]) if m > 2 else np.zeros((0, 4))
        return np.concatenate([per.ravel(), second.ravel()])

    per_frame = 4 + 2 + 1 + 4
    result = least_squares(residuals, np.zeros(4 * m), jac_sparsity=sparsity(m, per_frame, 1), method='trf', loss='soft_l1',
                           f_scale=ROBUST_TORSO, x_scale=1., max_nfev=200)
    return result.x.reshape(m, 4), result


def errors(camera, frames, middle, end, targets, conf, offset, torso):
    """Per-frame 2D error (torso units) of the middle and end joints; NaN where not confident."""
    uv = camera.project(np.stack([middle, end], 1), frames) + offset[:, None]
    e = np.linalg.norm(uv - targets, axis=-1) / torso[:, None]
    return np.where(conf > .5, e, np.nan)


def stats(values):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values):
        return dict(median=None, p90=None, max=None, samples=0)
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), max=float(values.max()), samples=int(len(values)))


def solve_interval(rest, roots, quats, fps, entry, interval, camera, foot_ranges):
    """Place one chain on one interval; returns ([(start, end, joints, values)] row runs or None, report entry)."""
    started = time.perf_counter(); n = len(roots); a, b = interval
    chain = entry['chain']; (_, upper, middle, end), _, _ = CHAINS[chain]
    report = dict(id=entry['id'], chain=chain, strength=entry['strength'], start=int(a), end=int(b))
    lo, hi = max(0, a - MAX_EASE), min(n, b + MAX_EASE)
    if hi - lo < 3:
        return None, dict(report, skipped='interval shorter than three frames')
    frames = np.arange(lo, hi)
    root = np.asarray(roots[lo:hi], float); start = np.asarray(quats[lo:hi], float)
    model = ChainModel(rest, root, start, chain)
    offset, torso = alignment(camera, model.joints, frames)
    targets, conf, swapped = observations(camera, model.joints, frames, chain, offset, torso)
    inside = slice(a - lo, b - lo)
    frame_conf = conf.max(1)
    if not (frame_conf[inside] > .5).any():
        return None, dict(report, skipped='no confident 2D keypoints for this limb in the interval')
    x, result = solve_window(model, camera, frames, targets, conf, offset, torso, BEND_LIMITS[chain[-3:]], fps)
    upper_new, middle_new = model.locals(x)
    solved = start.copy()
    solved[:, upper] = matrix_quat(upper_new); solved[:, middle] = matrix_quat(middle_new)
    from .self_contact import rotation_angle
    edge = max(rotation_angle(quat_matrix(start[[a - lo, b - lo - 1]][:, [upper, middle]]),
                              quat_matrix(solved[[a - lo, b - lo - 1]][:, [upper, middle]])).max(), 0.)
    ease = int(min(MAX_EASE, max(MIN_EASE, math.ceil(edge / EASE_DEG_PER_FRAME))))
    weights = np.zeros(hi - lo)
    w_lo, w_hi = max(lo, a - ease), min(hi, b + ease)
    weights[w_lo - lo:w_hi - lo] = ease_weights(w_lo, w_hi, a, b, ease)
    coverage = gaussian_filter1d(frame_conf, max(.5, fps * COVERAGE_SECONDS), mode='nearest')
    weights *= np.clip(coverage / .3, 0, 1) * entry['strength']
    conflict = 0
    if chain.endswith('Leg'):
        side = 0 if chain.startswith('Left') else 1
        locked = np.zeros(hi - lo, bool)
        for s, e in foot_ranges[side]:
            locked[max(s, lo) - lo:max(min(e, hi) - lo, 0)] = True
        conflict = int(locked[inside].sum())
        weights *= lock_gate(locked, MIN_EASE)
    weights[weights < 1e-6] = 0.
    final = start.copy()
    for j in (upper, middle):
        final[:, j] = slerp(start[:, j], solved[:, j], weights)
    changed = weights > 0
    if changed.any():
        # The hand or foot keeps its source world rotation (as self-contacts and the leg IK do); only its local rotation changes.
        from .correction import fk
        rows = np.flatnonzero(changed)
        _, world0 = fk(rest, root[rows], start[rows]); _, world1 = fk(rest, root[rows], final[rows])
        final[rows, end] = matrix_quat(world1[:, middle].swapaxes(-1, -2) @ world0[:, end])
    report.update(window=[int(w_lo), int(w_hi)], ease_frames=ease, solver=dict(cost=float(result.cost), evaluations=int(result.nfev),
                  status=int(result.status)), confident_fraction=float((frame_conf[inside] > .5).mean()),
                  swapped_frames=int(swapped[inside].sum()), foot_lock_conflict_frames=conflict)
    if not changed.any():
        report['seconds'] = time.perf_counter() - started
        return None, dict(report, skipped='nothing to change (no confidence or locked feet)')
    after = ChainModel(rest, root, final, chain)
    e0 = errors(camera, frames, model.joints[:, middle], model.joints[:, end], targets, conf, offset, torso)[inside]
    e1 = errors(camera, frames, after.joints[:, middle], after.joints[:, end], targets, conf, offset, torso)[inside]
    z0 = camera.camera(model.joints[:, [middle, end]], frames)[..., 2]; z1 = camera.camera(after.joints[:, [middle, end]], frames)[..., 2]
    b0 = np.rad2deg(model.pose(np.zeros((hi - lo, 4)))[3]); b1 = np.rad2deg(after.pose(np.zeros((hi - lo, 4)))[3])
    names = ('knee', 'ankle') if chain.endswith('Leg') else ('elbow', 'wrist')
    report.update(error_before_torso=dict(limb=stats(np.fmax(*e0.T)), **{names[k]: stats(e0[:, k]) for k in (0, 1)}),
                  error_after_torso=dict(limb=stats(np.fmax(*e1.T)), **{names[k]: stats(e1[:, k]) for k in (0, 1)}),
                  depth_change_max_m=float(np.abs(z1 - z0)[changed].max()), end_move_max_m=float(np.linalg.norm(
                      after.joints[:, end] - model.joints[:, end], axis=1)[changed].max()),
                  bend_deg=dict(before_min=float(b0[changed].min()), before_max=float(b0[changed].max()), after_min=float(b1[changed].min()),
                                after_max=float(b1[changed].max()), limits=list(BEND_LIMITS[chain[-3:]])),
                  rotation_change_max_deg=float(rotation_angle(quat_matrix(start[changed][:, [upper, middle]]),
                                                               quat_matrix(final[changed][:, [upper, middle]])).max()),
                  end_local_change_max_deg=float(rotation_angle(quat_matrix(start[changed][:, end]), quat_matrix(final[changed][:, end])).max()),
                  changed_frames=int(changed.sum()))
    from .foot_articulation import runs
    report['seconds'] = time.perf_counter() - started
    # Only rows with a positive weight are written: a zero-weight slerp still renormalises and would change bits.
    return [(lo + s, lo + e, [upper, middle, end], final[s:e]) for s, e in runs(np.flatnonzero(changed))], report


def apply(rest, out_roots, out_quats, fps, entries, camera, foot_ranges, emit=lambda *a, **k: None):
    """Place enabled rows in order, mutating out_quats; None when nothing is enabled."""
    work = [(e, r) for e in entries if e.get('enabled', True) for r in e['ranges']]
    if not work:
        return None
    started = time.perf_counter(); rest = np.asarray(rest, float)[:22]
    reports, changed, warnings = [], [], []
    for index, (entry, interval) in enumerate(work):
        if camera is None:
            reports.append(dict(id=entry['id'], chain=entry['chain'], start=int(interval[0]), end=int(interval[1]),
                                skipped='placement needs ExMR 2D keypoints and cameras'))
        else:
            updates, report = solve_interval(rest, out_roots, out_quats, fps, entry, interval, camera, foot_ranges)
            reports.append(report)
            for s, e, joints, values in updates or ():
                out_quats[s:e, joints] = values[:, joints].astype(out_quats.dtype)
                # world lists the joints whose world rotation changes: the hand or foot keeps its own.
                changed.append(dict(joints=[int(j) for j in joints], world=[int(j) for j in joints[:2]], start=int(s), end=int(e)))
            if report.get('foot_lock_conflict_frames'):
                warnings.append('%s %d-%d: a foot lock kept the leg still on %d frames.' % (
                    entry['chain'], interval[0], interval[1] - 1, report['foot_lock_conflict_frames']))
        if 'skipped' in reports[-1]:
            warnings.append('%s %d-%d skipped: %s.' % (entry['chain'], interval[0], interval[1] - 1, reports[-1]['skipped']))
        emit('progress', stage='correct', current=index + 1, total=len(work), phase='placement', message='Motion Correction: placing limbs')
    return dict(method=METHOD, entries=reports, changed=changed, seconds=time.perf_counter() - started,
                warning=' '.join('Limb placement: ' + w for w in warnings))


def from_store(project, base, context, rest, out_roots, out_quats, fps, segments, transform=None, emit=lambda *a, **k: None):
    """correction.run hook after the hand lock; None (nothing touched) without placement rows."""
    from .self_contact_store import read_placement
    entries, revision, warning = read_placement(project, base, context)
    if not entries and not warning:
        return None
    camera = None
    if any(e['enabled'] for e in entries):
        try:
            camera = Camera.load(project, np.asarray(rest, float)[0], transform, out_roots)
            if len(camera.keypoints) != len(out_roots):
                camera = None
        except (OSError, KeyError, ValueError, RuntimeError, TypeError):
            camera = None
    foot_ranges = [[(s['start'], s['end']) for s in segments if s['foot'] == f] for f in (0, 1)]
    report = apply(rest, out_roots, out_quats, fps, entries, camera, foot_ranges, emit) or dict(method=METHOD, entries=[], changed=[], warning='')
    report['revision'] = revision
    report['warning'] = ' '.join(filter(None, (report['warning'], warning)))
    return report


def publish(output, pose, reports, written):
    """Write changed rows of joints outside `written` into output['body_pose']; the body_pose after them, or None."""
    import torch
    rows = [(j, c['start'], c['end']) for report in reports if report for c in report.get('changed', []) for j in c['joints'] if j not in written]
    for j, s, e in rows:
        output['body_pose'][s:e, 3 * (j - 1):3 * j] = torch.from_numpy(np.asarray(pose[s:e, j]).copy())
    return output['body_pose'].clone() if rows else None


def descendants(joints):
    from .correction import PARENTS
    out = set(joints)
    for j in range(22):
        k = j
        while k >= 0 and k not in joints:
            k = PARENTS[k]
        if k >= 0:
            out.add(j)
    return out


def audit_exemptions(saved, preserved):
    """(start, end, column ids) of the export upper-body audit declared by placement and smoothing rows.

    A changed local rotation turns every descendant in the world, so rows exempt the changed joints and their descendants,
    unless they list the joints whose world rotation changes (`world`, placement keeps the hand's).
    """
    rows = []
    for key in ('placement', 'smoothing'):
        for c in ((saved or {}).get(key) or {}).get('changed', []):
            world = c['world'] if 'world' in c else sorted(descendants(c['joints']))
            ids = [preserved.index(j) for j in world if j in preserved]
            if ids:
                rows.append((c['start'], c['end'], ids))
    return rows
