"""Foot articulation: per-interval ankle and toe poses (pointe, flat, flexed, or fitted to the video's 2D foot direction).

Runs in correction.run on the source pose, before the foot plan, so sole probes, foot locks, grounding and the leg IK
(which keeps ankle world rotations) all see the articulated feet. On pointe the toe-box tip is the sole support.
"""
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

METHOD = 'Foot articulation: sagittal ankle/toe pitch per interval with rotation-space easing'
# side, knee, ankle, toe (SMPL-X joints)
FEET = {'LeftFoot': (0, 4, 7, 10), 'RightFoot': (1, 5, 8, 11)}
# Degrees from the SMPL-X rest foot (sole level, shin vertical). Ankle plantarflexion and toe flexion (curl down) are
# positive; dorsiflexion and toe extension negative. Pitch is about the parent's lateral (x) axis.
LIMITS = {'default': dict(plantar=50., dorsi=20., toe_flex=35., toe_extend=50.),
          'dancer': dict(plantar=90., dorsi=25., toe_flex=45., toe_extend=60.)}
FLEXED_TOE_DEG = -15.
VISIBLE = .5
# COCO-63 per side: ankle, big toe, small toe, heel
KEYPOINTS = ((15, 17, 18, 19), (16, 20, 21, 22))
MATCH_PRIOR = .05            # 2D degrees of error traded per degree of ankle change, keeps fits near the source
MATCH_STEP_DEG = 1.
EASE_DEG_PER_FRAME = 5.
MIN_EASE = 6
FOLLOW_PASSES = 2            # extra foot-plan solves that re-aim pointe/flexed ankles at the corrected shins
# Rest heel (ankle frame) and toe-tip (toe frame) points of the neutral SMPL-X per side, metres, and the rest
# ankle->toe length they belong to. Used to scale virtual foot points when no body model is loaded (review).
DEFAULT_HEEL = np.array([[.02728, -.05814, -.05191], [-.01089, -.0557, -.06092]])
DEFAULT_TIP = np.array([[-.01953, -.00939, .06811], [.02864, -.00642, .05856]])
DEFAULT_FOOT = np.array([.13990, .13813])


def quat_matrix(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def matrix_quat(m):
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(m.shape[:-2] + (4,))


def pitch(v):
    """Downward angle (deg) of vectors in their frame's y-z plane: +z forward = 0, -y down = 90; x rotation adds to it."""
    v = np.asarray(v, float)
    return np.rad2deg(np.arctan2(-v[..., 1], v[..., 2]))


def wrap(deg):
    return (np.asarray(deg, float) + 180.) % 360. - 180.


def rot_x(deg):
    a = np.deg2rad(np.asarray(deg, float)); c, s = np.cos(a), np.sin(a)
    m = np.zeros(a.shape + (3, 3)); m[..., 0, 0] = 1
    m[..., 1, 1] = c; m[..., 1, 2] = -s; m[..., 2, 1] = s; m[..., 2, 2] = c
    return m


def ankle_pitch(local_ankle, m0):
    """Plantarflexion (deg) of an ankle local rotation; m0 is the rest ankle->toe vector."""
    return wrap(pitch(np.einsum('...ij,j->...i', local_ankle, m0)) - pitch(m0))


def toe_pitch(local_toe, s0):
    """Toe flexion (deg) of a toe local rotation; s0 is the rest toe->tip vector."""
    return wrap(pitch(np.einsum('...ij,j->...i', local_toe, s0)) - pitch(s0))


def smoothstep_weights(lo, hi, a, b, ease):
    """1 on [a, b), smoothstep over ease frames on both sides; every ease frame is strictly inside (0, 1)."""
    t = np.arange(lo, hi, dtype=float); x = np.ones(len(t))
    x[t < a] = (t[t < a] - (a - ease - 1)) / (ease + 1)
    x[t >= b] = (b + ease - t[t >= b]) / (ease + 1)
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class FootModel:
    """Rest skeleton plus heel / toe-tip offsets; tip_index selects toe-box vertices of SoleGeometry.sample."""

    def __init__(self, rest, heel, tip, tip_index=None):
        self.rest = np.asarray(rest, float)[:22]
        self.heel = np.asarray(heel, float); self.tip = np.asarray(tip, float); self.tip_index = tip_index

    @classmethod
    def from_geometry(cls, geometry):
        heel, tip, index = [], [], []
        sizes = np.cumsum([0] + [len(g) for g in geometry.groups])
        for f, (_, _, ankle, toe) in enumerate(FEET.values()):
            first, last = sizes[2 * f], sizes[2 * f + 2]
            v = geometry.shaped[first:last]; w = geometry.weights[first:last]
            back = v[:, 2] <= np.quantile(v[:, 2], .06)
            low = back & (v[:, 1] <= np.median(v[back, 1]))
            toes = w[:, toe] > .5
            front = toes & (v[:, 2] >= np.quantile(v[toes, 2], .85))
            heel.append(v[low].mean(0) - geometry.rest[ankle]); tip.append(v[front].mean(0) - geometry.rest[toe])
            index.append(first + np.flatnonzero(front))
        return cls(geometry.rest, heel, tip, index)

    @classmethod
    def from_joints(cls, rest):
        """Offsets of the neutral body scaled to this skeleton's foot length (no mesh needed)."""
        rest = np.asarray(rest, float)
        scale = np.array([np.linalg.norm(rest[t] - rest[a]) for _, _, a, t in FEET.values()]) / DEFAULT_FOOT
        return cls(rest, DEFAULT_HEEL * scale[:, None], DEFAULT_TIP * scale[:, None])

    def vectors(self, side):
        _, knee, ankle, toe = list(FEET.values())[side]
        return self.rest[toe] - self.rest[ankle], self.tip[side], self.heel[side], self.rest[ankle] - self.rest[knee]

    def points(self, side, ankle_pos, ankle_world, toe_world):
        """World heel and toe-tip points for one foot."""
        m0, s0, h0, _ = self.vectors(side)
        toe_pos = ankle_pos + np.einsum('...ij,j->...i', ankle_world, m0)
        return ankle_pos + np.einsum('...ij,j->...i', ankle_world, h0), toe_pos + np.einsum('...ij,j->...i', toe_world, s0)


def limit_range(profile, source, key_hi, key_lo):
    """[lo, hi] per frame; a source already beyond the limit keeps its own value as the bound."""
    limits = LIMITS[profile]
    return np.minimum(-limits[key_lo], source), np.maximum(limits[key_hi], source)


def flat_delta(knee_world, v):
    """Smallest x rotation (deg) that levels the world y of knee-frame vectors v (the sole line)."""
    g = knee_world[:, 1, :]
    c = g[:, 0] * v[:, 0]; A = g[:, 1] * v[:, 1] + g[:, 2] * v[:, 2]; B = g[:, 2] * v[:, 1] - g[:, 1] * v[:, 2]
    radius = np.maximum(np.hypot(A, B), 1e-12); phi = np.arctan2(B, A)
    spread = np.arccos(np.clip(-c / radius, -1, 1))
    options = np.rad2deg(np.stack([phi + spread, phi - spread], 1))
    options = wrap(options)
    return options[np.arange(len(v)), np.abs(options).argmin(1)]


class Views:
    """ExMR 2D keypoints and per-frame cameras (world -> OpenCV camera), as review projects joints."""

    def __init__(self, keypoints, rotation, translation, K, transform=None):
        self.keypoints, self.rotation, self.translation, self.K = keypoints, rotation, translation, K
        self.transform = np.eye(4) if transform is None else np.asarray(transform, float)

    @classmethod
    def load(cls, project, pelvis, transform=None):
        import torch
        keypoints = np.asarray(torch.load(project / 'exmr_keypoints.pt', map_location='cpu', weights_only=True), float)
        record = torch.load(project / 'hmr4d_results.pt', map_location='cpu', weights_only=False, mmap=True)
        world, incam = record['smpl_params_global'], record['smpl_params_incam']
        with np.load(project / 'tracking.npz', allow_pickle=False) as data:
            K = np.asarray(data['K'], float)
        rw = Rotation.from_rotvec(world['global_orient'].numpy()).as_matrix()
        ri = Rotation.from_rotvec(incam['global_orient'].numpy()).as_matrix()
        wt, it = world['transl'].numpy().astype(float), incam['transl'].numpy().astype(float)
        rotation = ri @ rw.transpose(0, 2, 1)
        # SMPL translation rotates about the rest pelvis, so pivot both roots there (scene_camera.paired_camera).
        translation = it + pelvis - np.einsum('fij,fj->fi', rotation, wt + pelvis)
        return cls(keypoints, rotation, translation, K, transform)

    def project(self, points, frames):
        """Correction-frame points (N, ..., 3) on frames (N,) -> pixels (N, ..., 2); NaN behind the camera."""
        local = (np.asarray(points, float) - self.transform[:3, 3]) @ self.transform[:3, :3]
        shape = local.shape; flat = local.reshape(shape[0], -1, 3)
        cam = np.einsum('fij,fkj->fki', self.rotation[frames], flat) + self.translation[frames][:, None]
        z = cam[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = np.stack([self.K[0, 0] * cam[..., 0] / z + self.K[0, 2], self.K[1, 1] * cam[..., 1] / z + self.K[1, 2]], -1)
        uv[z <= 1e-6] = np.nan
        return uv.reshape(shape[:-1] + (2,))

    def foot(self, side, frames):
        """Observed ankle and toe-midpoint pixels plus a 0..1 confidence per frame."""
        ankle, big, small, _ = KEYPOINTS[side]
        k = self.keypoints[frames]
        score = np.minimum(k[:, ankle, 2], np.maximum(k[:, big, 2], k[:, small, 2]))
        toe = np.where((k[:, small, 2] >= VISIBLE)[:, None] & (k[:, big, 2] >= VISIBLE)[:, None], (k[:, big, :2] + k[:, small, :2]) / 2,
                       np.where((k[:, big, 2] >= k[:, small, 2])[:, None], k[:, big, :2], k[:, small, :2]))
        return k[:, ankle, :2], toe, np.clip((score - VISIBLE) / .15, 0, 1)

    def swapped(self, joints, frames):
        """Frames whose ankle keypoints look left/right swapped against the model (offset-free: ankle-to-ankle vectors)."""
        uv = self.project(joints[:, [7, 8]], frames)
        k = self.keypoints[frames][:, [KEYPOINTS[0][0], KEYPOINTS[1][0]], :2]
        model, seen = uv[:, 0] - uv[:, 1], k[:, 0] - k[:, 1]
        return np.linalg.norm(seen + model, axis=-1) < np.linalg.norm(seen - model, axis=-1)


def match_delta(model, side, views, frames, joints, world, local, toe_local, lo, hi, fps):
    """Ankle pitch change (deg) fitting the projected ankle->toe-tip direction to the video, and its smoothed confidence.

    The malleolus keypoint is the ankle joint; a raised heel's keypoint drifts, so the heel is not used.
    """
    m0, s0, _, _ = model.vectors(side)
    _, knee, ankle, _ = list(FEET.values())[side]
    n = len(frames)
    grid = np.arange(math.floor(float(lo.min())), math.ceil(float(hi.max())) + MATCH_STEP_DEG, MATCH_STEP_DEG)
    v_tip = np.einsum('tij,j->ti', local[:, ankle], m0) + np.einsum('tij,tjk,k->ti', local[:, ankle], toe_local, s0)
    tip = joints[:, ankle, None] + np.einsum('tij,kjl,tl->tki', world[:, knee], rot_x(grid), v_tip)
    uv = views.project(np.concatenate([joints[:, ankle, None], tip], 1), frames)
    d = uv[:, 1:] - uv[:, :1]
    obs_ankle, obs_toe, conf = views.foot(side, frames)
    o = obs_toe - obs_ankle
    cos = (d * o[:, None]).sum(-1) / np.maximum(np.linalg.norm(d, axis=-1) * np.linalg.norm(o, axis=-1)[:, None], 1e-9)
    error = np.rad2deg(np.arccos(np.clip(np.nan_to_num(cos, nan=-1.), -1, 1)))
    allowed = (grid[None] >= lo[:, None] - 1e-9) & (grid[None] <= hi[:, None] + 1e-9)
    cost = np.where(allowed, error + MATCH_PRIOR * np.abs(grid)[None], np.inf)
    best = cost.argmin(1)
    delta = grid[best]
    rows = np.arange(n)
    length = np.linalg.norm(d[rows, best], axis=-1)
    observed = np.linalg.norm(o, axis=-1)
    # A foreshortened or mislabelled foot (detector left/right swap) gives no usable direction.
    near = ~views.swapped(joints, frames)
    ratio = observed / np.maximum(length, 1e-6)
    conf = conf * near * np.clip((np.minimum(ratio, 1 / np.maximum(ratio, 1e-6)) - .25) / .25, 0, 1)
    conf = np.nan_to_num(conf * (error[rows, best] < 45.))
    sigma = max(.5, fps * .05)
    weight = gaussian_filter1d(conf, sigma, mode='nearest')
    smooth = gaussian_filter1d(conf * delta, sigma, mode='nearest') / np.maximum(weight, 1e-9)
    before = error[rows, np.abs(grid).argmin()]
    return np.where(weight > 1e-6, smooth, 0.), np.clip(weight, 0, 1), dict(direction_error_before=before, direction_error_fit=error[rows, best],
                                                                         confidence=conf)


def align(u, v):
    """Rotation vectors of the smallest rotations taking directions u to v (rows)."""
    u = u / np.linalg.norm(u, axis=-1, keepdims=True); v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.cross(u, v); s = np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = np.arctan2(s, (u * v).sum(-1, keepdims=True))
    return np.where(s > 1e-12, axis / np.maximum(s, 1e-12) * angle, 0.)


def turn(rotvec, local):
    """Pre-multiply local rotations by rotation vectors given in the parent frame."""
    return Rotation.from_rotvec(np.asarray(rotvec, float).reshape(-1, 3)).as_matrix().reshape(local.shape) @ local


def within(rotvec, local, measure, lo, hi):
    """Largest fraction t in [0, 1] (bisection) keeping measure(turn(t * rotvec)) inside [lo, hi]; (scaled rotvec, limited)."""
    ok = lambda t: (lambda p: (p >= lo - 1e-6) & (p <= hi + 1e-6))(measure(turn(rotvec * t[:, None], local)))
    full = ok(np.ones(len(local)))
    a, b = np.zeros(len(local)), np.ones(len(local))
    for _ in range(20):
        m = (a + b) / 2; good = ok(m)
        a = np.where(good, m, a); b = np.where(good, b, m)
    t = np.where(full, 1., a)
    return rotvec * t[:, None], ~full


def pitch_vector(deg):
    deg = np.asarray(deg, float)
    return np.stack([np.deg2rad(deg), np.zeros_like(deg), np.zeros_like(deg)], -1)


def targets(model, side, entry, roots, quats, frames, views, fps):
    """(ankle rotvec, toe rotvec, frame weights, info) for one entry; rotvecs pre-multiply the local rotations."""
    from .correction import fk
    _, knee, ankle, toe = list(FEET.values())[side]
    m0, s0, h0, shin = model.vectors(side)
    joints, world = fk(model.rest, roots, quats)
    local = quat_matrix(quats)
    p_src = ankle_pitch(local[:, ankle], m0); q_src = toe_pitch(local[:, toe], s0)
    q_lo, q_hi = limit_range(entry['profile'], q_src, 'toe_flex', 'toe_extend')
    p_lo, p_hi = limit_range(entry['profile'], p_src, 'plantar', 'dorsi')
    mode = entry['mode']; weight = np.ones(len(frames)); info = {}
    if mode == 'pointe':
        # Toes straight along the metatarsals; then the ankle->tip line onto the shin (full 3D alignment, no sickling).
        dq = align(np.einsum('tij,j->ti', local[:, toe], s0), np.broadcast_to(m0, (len(frames), 3)))
    elif mode == 'flat':
        dq = pitch_vector(-q_src)
    elif mode == 'flexed':
        dq = pitch_vector(FLEXED_TOE_DEG - q_src)
    else:
        dq = np.zeros((len(frames), 3))
    dq, limited = within(dq, local[:, toe], lambda r: toe_pitch(r, s0), q_lo, q_hi)
    toe_new = turn(dq, local[:, toe])
    reach = m0 + np.einsum('tij,j->ti', toe_new, s0)
    if mode == 'pointe':
        dp = align(np.einsum('tij,tj->ti', local[:, ankle], reach), np.broadcast_to(shin, (len(frames), 3)))
    elif mode == 'flat':
        dp = pitch_vector(flat_delta(world[:, knee], np.einsum('tij,tj->ti', local[:, ankle], reach - h0)))
    elif mode == 'flexed':
        dp = pitch_vector(-LIMITS[entry['profile']]['dorsi'] - p_src)
    else:
        delta, weight, info = match_delta(model, side, views, frames, joints, world, local, toe_new, p_lo - p_src, p_hi - p_src, fps)
        dp = pitch_vector(delta)
    dp, clipped = within(dp, local[:, ankle], lambda r: ankle_pitch(r, m0), p_lo, p_hi)
    return dp, dq, weight, dict(info, ankle_before=p_src, toe_before=q_src, limited=limited | clipped)


def in_line(model, side, roots, quats):
    """Angles (deg) of the ankle->toe and toe->tip segments from the shin direction; 0 means in line with the shin."""
    from .correction import fk
    _, knee, ankle, toe = list(FEET.values())[side]
    joints, world = fk(model.rest, roots, quats)
    _, tip = model.points(side, joints[:, ankle], world[:, ankle], world[:, toe])
    shin = joints[:, ankle] - joints[:, knee]

    def angle(u):
        cos = (u * shin).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(shin, axis=-1), 1e-12)
        return np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
    return angle(joints[:, toe] - joints[:, ankle]), angle(tip - joints[:, toe])


def solve_entry(model, roots, quats, entry, interval, fps, views):
    """Articulate one interval in place on quats; returns (changed row range, pointe weights, report entry)."""
    started = time.perf_counter(); n = len(roots); a, b = interval
    side, _, ankle, toe = FEET[entry['foot']]
    report = dict(id=entry['id'], foot=entry['foot'], mode=entry['mode'], profile=entry['profile'], strength=entry['strength'],
                  start=int(a), end=int(b))
    if entry['mode'] == 'match' and views is None:
        return [], None, dict(report, skipped='match needs ExMR 2D keypoints and cameras')
    frames = np.arange(a, b)
    dp, dq, _, _ = targets(model, side, entry, roots[a:b], quats[a:b], frames, views, fps)
    largest = np.rad2deg(max(np.linalg.norm(dp, axis=-1).max(), np.linalg.norm(dq, axis=-1).max()))
    ease = max(MIN_EASE, math.ceil(float(largest) / EASE_DEG_PER_FRAME))
    lo, hi = max(0, a - ease), min(n, b + ease)
    frames = np.arange(lo, hi)
    root = np.asarray(roots[lo:hi], float); start = np.asarray(quats[lo:hi], float)
    dp, dq, confidence, info = targets(model, side, entry, root, start, frames, views, fps)
    weight = smoothstep_weights(lo, hi, a, b, ease) * entry['strength'] * confidence
    changed = weight > 0
    local = quat_matrix(start)
    out = start.copy()
    out[changed, ankle] = matrix_quat(turn(weight[changed, None] * dp[changed], local[changed, ankle]))
    out[changed, toe] = matrix_quat(turn(weight[changed, None] * dq[changed], local[changed, toe]))
    inside = slice(a - lo, b - lo)
    m0, s0, _, _ = model.vectors(side)
    after = quat_matrix(out)
    p_after = ankle_pitch(after[:, ankle], m0); q_after = toe_pitch(after[:, toe], s0)

    def stats(values):
        values = np.asarray(values)[inside]
        return dict(median=float(np.median(values)), min=float(values.min()), max=float(values.max()))
    report.update(window=[int(lo), int(hi)], ease_frames=int(ease), ankle_plantarflexion_before_deg=stats(info['ankle_before']),
                  ankle_plantarflexion_after_deg=stats(p_after), toe_flexion_before_deg=stats(info['toe_before']),
                  toe_flexion_after_deg=stats(q_after), limited_frames=int(info['limited'][inside].sum()), changed_frames=int(changed.sum()))
    if entry['mode'] == 'match':
        report.update(confident_fraction=float((info['confidence'][inside] > .5).mean()),
                      direction_error_before_deg=float(np.nanmedian(info['direction_error_before'][inside])),
                      direction_error_fit_deg=float(np.nanmedian(info['direction_error_fit'][inside])))
    if entry['mode'] == 'pointe':
        met, toes = in_line(model, side, root, out)
        report.update(metatarsal_to_shin_deg=stats(met), toes_to_shin_deg=stats(toes))
    report['seconds'] = time.perf_counter() - started
    blocks = [(lo + s, lo + e, side) for s, e in runs(np.flatnonzero(changed))]
    for s, e, _ in blocks:
        quats[s:e, ankle] = out[s - lo:e - lo, ankle].astype(quats.dtype)
        quats[s:e, toe] = out[s - lo:e - lo, toe].astype(quats.dtype)
    pointe = np.zeros(n)
    if entry['mode'] in ('pointe', 'flexed'):
        pointe[lo:hi] = weight
    return blocks, pointe, report


class Articulation:
    """Result of the articulation pass: report, changed rows and per-foot pointe support weights."""

    passes = FOLLOW_PASSES

    def __init__(self, model, n, report):
        self.model = model; self.report = report; self.changed = []
        self.pointe = np.zeros((n, 2)); self.follow = np.zeros((n, 2)); self.ankles = None

    def follow_shins(self, geometry, roots, quats, probes, out_roots, out_quats):
        """Re-aim shin-relative ankles (pointe, flexed) by the knee rotation the corrected legs got; True if any changed.

        The leg IK keeps ankle world rotations, so without this a moved shin leaves the foot out of line.
        """
        from .correction import fk
        rows = np.flatnonzero(self.follow.max(1) > 0)
        if not len(rows):
            return False
        if self.ankles is None:
            self.ankles = np.array(quats[:, [7, 8]], float, copy=True)
        _, source = fk(self.model.rest, np.asarray(roots[rows], float), np.asarray(quats[rows], float))
        _, corrected = fk(self.model.rest, np.asarray(out_roots[rows], float), np.asarray(out_quats[rows], float))
        change = 0.
        for side, (_, knee, ankle, _) in enumerate(FEET.values()):
            w = self.follow[rows, side]; use = w > 0
            if not use.any():continue
            delta = Rotation.from_matrix(source[use, knee].swapaxes(-1, -2) @ corrected[use, knee]).as_rotvec()
            change = max(change, float(np.rad2deg(np.linalg.norm(delta, axis=1)).max()))
            quats[rows[use], ankle] = matrix_quat(turn(w[use, None] * delta, quat_matrix(self.ankles[rows[use], side]))).astype(quats.dtype)
        for first, last in runs(rows):
            probes[first:last] = geometry.sample(np.asarray(roots[first:last]), np.asarray(quats[first:last]))[0]
        self.support(geometry, roots, quats, probes)
        self.report.setdefault('follow_knee_change_max_deg', []).append(change)
        return True

    def finish(self, geometry, out_roots, out_quats, after):
        """Pointe support on the corrected sole probes (in place) and the final pose measured into the report."""
        self.support(geometry, out_roots, out_quats, after)
        for entry in self.report['entries']:
            if 'window' not in entry:continue
            a, b = entry['start'], entry['end']
            side, _, ankle, toe = FEET[entry['foot']]
            m0, s0, _, _ = self.model.vectors(side)
            local = quat_matrix(np.asarray(out_quats[a:b], float))
            met, toes = in_line(self.model, side, np.asarray(out_roots[a:b], float), np.asarray(out_quats[a:b], float))
            summary = lambda v: dict(median=float(np.median(v)), max=float(np.max(v)))
            entry['corrected'] = dict(ankle_plantarflexion_deg=summary(ankle_pitch(local[:, ankle], m0)),
                                      toe_flexion_deg=summary(toe_pitch(local[:, toe], s0)), metatarsal_to_shin_deg=summary(met),
                                      toes_to_shin_deg=summary(toes))
        return after

    def support(self, geometry, roots, quats, probes):
        """Blend each foot's heel/forefoot probes toward its toe-box tip by the pointe weight (in place)."""
        rows = np.flatnonzero(self.pointe.max(1) > 0)
        if not len(rows) or self.model.tip_index is None:
            return probes
        for first, last in runs(rows):
            _, vertices = geometry.sample(np.asarray(roots[first:last]), np.asarray(quats[first:last]))
            for side in (0, 1):
                w = self.pointe[first:last, side]
                if not (w > 0).any():continue
                box = vertices[:, self.model.tip_index[side]]
                tip = box.mean(1); tip[:, 1] = box[..., 1].min(1)
                for k in (2 * side, 2 * side + 1):
                    probes[first:last, k] = (1 - w[:, None]) * probes[first:last, k] + w[:, None] * tip
        return probes

    def publish(self, output, pose):
        """Write the changed toe rows into output['body_pose']; the ankle rows are part of the legacy leg write-back."""
        import torch
        for s, e, side in self.changed:
            toe = FEET[('LeftFoot', 'RightFoot')[side]][3]
            output['body_pose'][s:e, 3 * (toe - 1):3 * toe] = torch.from_numpy(np.asarray(pose[s:e, toe]).copy())


def runs(rows):
    """Contiguous (first, end) runs of sorted row indices."""
    rows = np.asarray(rows)
    if not len(rows):
        return []
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.r_[rows[0], rows[breaks + 1]]; ends = np.r_[rows[breaks] + 1, rows[-1] + 1]
    return list(zip(starts.tolist(), ends.tolist()))


def apply(model, roots, quats, entries, fps, views=None, emit=lambda *a, **k: None):
    """Articulate enabled entries in order, mutating quats; None when no interval is enabled."""
    work = [(e, r) for e in entries if e.get('enabled', True) for r in e['ranges']]
    if not work:
        return None
    started = time.perf_counter()
    result = Articulation(model, len(roots), None)
    reports, warnings = [], []
    for index, (entry, interval) in enumerate(work):
        blocks, pointe, report = solve_entry(model, roots, quats, entry, interval, fps, views)
        reports.append(report)
        if 'skipped' in report:
            warnings.append('%s %s %d-%d skipped: %s.' % (entry['foot'], entry['mode'], interval[0], interval[1] - 1, report['skipped']))
        if report.get('limited_frames'):
            warnings.append('%s %s %d-%d: the %s profile limited the pose on %d frames.' % (
                entry['foot'], entry['mode'], interval[0], interval[1] - 1, entry['profile'], report['limited_frames']))
        result.changed.extend(blocks)
        if pointe is not None:
            side = FEET[entry['foot']][0]
            weight = result.follow if entry['mode'] == 'flexed' else result.pointe
            weight[:, side] = np.maximum(weight[:, side], pointe)
        emit('progress', stage='correct', current=index + 1, total=len(work), phase='foot_articulation', message='Motion Correction: articulating feet')
    result.follow = np.maximum(result.follow, result.pointe)
    joints = {7: 'LeftFoot', 10: 'LeftToeBase', 8: 'RightFoot', 11: 'RightToeBase'}
    result.report = dict(method=METHOD, entries=reports, seconds=time.perf_counter() - started,
                         changed=[dict(joints=[FEET[f][2], FEET[f][3]], bones=[joints[FEET[f][2]], joints[FEET[f][3]]], start=int(s), end=int(e))
                                  for s, e, side in result.changed for f in [('LeftFoot', 'RightFoot')[side]]],
                         pointe_support_frames=int((result.pointe > 0).sum()),
                         warning=' '.join('Foot articulation: ' + w for w in warnings))
    return result


def from_store(project, base, context, geometry, roots, quats, probes, fps, transform=None, emit=lambda *a, **k: None):
    """correction.run hook before the foot plan; None (nothing touched) without articulation entries."""
    from .self_contact_store import read_articulation
    entries, revision, warning = read_articulation(project, base, context)
    if not entries and not warning:
        return None
    model = FootModel.from_geometry(geometry)
    views = None
    if any(e['mode'] == 'match' and e['enabled'] for e in entries):
        try:
            views = Views.load(project, geometry.rest[0], transform)
            if len(views.keypoints) != len(roots):
                views = None
        except (OSError, KeyError, ValueError, RuntimeError, TypeError):
            views = None
    result = apply(model, roots, quats, entries, fps, views, emit) or Articulation(model, len(roots), dict(method=METHOD, entries=[], changed=[], warning=''))
    for s, e, _ in result.changed:
        probes[s:e] = geometry.sample(np.asarray(roots[s:e]), np.asarray(quats[s:e]))[0]
    result.support(geometry, roots, quats, probes)
    result.report['revision'] = revision
    result.report['warning'] = ' '.join(filter(None, (result.report['warning'], warning)))
    return result
