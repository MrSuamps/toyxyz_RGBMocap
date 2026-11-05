"""Self-contacts: a hand holds another body part, solved on the already corrected pose.

The hand reaches a target that moves with the body; a target leg (or the other arm) may move toward it when the arm is
short. Changes are blended in local-rotation space over ease frames and never touch the root, trunk or toes.
"""
import math
import time
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

METHOD = 'Self-contact moving targets with reciprocal limbs and rotation-space easing'
DISTANCE = {'Foot': .05, 'ToeBase': .04, 'Leg': .06, 'UpLeg': .09, 'Hips': .12, 'Spine': .12, 'Spine1': .12, 'Spine2': .12,
            'Neck': .06, 'Head': .10, 'Hand': .03, 'Shoulder': .06, 'Arm': .06, 'ForeArm': .05}
LEG_PARTS = ('UpLeg', 'Leg', 'Foot', 'ToeBase')
ARMS = ((13, 16, 18, 20), (14, 17, 19, 21))
LEGS = ((1, 4, 7), (2, 5, 8))
ARM_CONFIG = dict(elbow_limit_deg=90., collar_limit_deg=20., iterations=150)
REACH_ELBOW_DEG = 165.
ELBOW_MAX_DEG = 170.
KNEE_MAX_DEG = 170.
RECIPROCAL_MARGIN_M = .01


def quat_matrix(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def matrix_quat(m):
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(m.shape[:-2] + (4,))


def slerp(q0, q1, w):
    """Per-element shortest-path slerp of wxyz quaternions; w broadcasts over the leading axes."""
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1); dot = np.abs(dot)
    w = np.asarray(w, float)[(...,) + (None,) * (q0.ndim - np.ndim(w))]
    theta = np.arccos(np.clip(dot, -1, 1)); s = np.sin(theta)
    small = s < 1e-6
    a = np.where(small, 1 - w, np.sin((1 - w) * theta) / np.where(small, 1, s))
    b = np.where(small, w, np.sin(w * theta) / np.where(small, 1, s))
    out = a * q0 + b * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def interior(a, b, c):
    """Angle at b in degrees; 180 means straight."""
    u = a - b; v = c - b
    cos = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
    return np.rad2deg(np.arccos(np.clip(cos, -1, 1)))


def ease_weights(lo, hi, a, b, ease):
    """Smoothstep up over [a-ease, a), 1 on [a, b), down over [b, b+ease); every ease frame is strictly inside (0, 1)."""
    t = np.arange(lo, hi, dtype=float); x = np.ones(len(t))
    x[t < a] = (t[t < a] - (a - ease - 1)) / (ease + 1)
    x[t >= b] = (b + ease - t[t >= b]) / (ease + 1)
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def parse_target(name):
    from .mixamo_rig import BODY_NAMES
    side = 0 if name.startswith('Left') else 1 if name.startswith('Right') else None
    part = name[4:] if side == 0 else name[5:] if side == 1 else name
    return BODY_NAMES.index(name), side, part


class MovingTracks:
    """Per-frame palm goals for contact_arm_plan.solve; one full-strength range per solved hand."""

    def __init__(self, first, targets):
        self.first = first; self.targets = targets
        m = len(next(iter(targets.values())))
        self.ranges = [[(first, first + m)] if h in targets else [] for h in (0, 1)]
        self.phases = [[dict(start=first, end=first + m, stance=0, kind='self_contact', anchor=targets[h][0].tolist())] if h in targets else []
                       for h in (0, 1)]

    def window(self, h, a, b, stance_weights=None):
        return None, None, self.targets[h][a - self.first:b - self.first]


def palms(rest, root, quat, offsets):
    from .hand_lock import samples
    return samples(rest, root, quat, offsets)


def target_base(j, g, p, tj, part):
    """Surface base point and frame of the target: the other palm for hands, the joint otherwise."""
    return (p[:, tj - 20] if part == 'Hand' else j[:, tj]), g[:, tj]


def local_offset(track, part, base, frame, palm):
    if track['offset'] != 'auto':return np.asarray(track['offset'], float)
    distance = DISTANCE.get(part, .05) if track['distance'] == 'auto' else float(track['distance'])
    direction = np.einsum('tji,tj->ti', frame, palm - base).mean(0)
    norm = np.linalg.norm(direction)
    return (direction / norm if norm > 1e-9 else np.array([0., 0., 1.])) * distance


def arm_reach(rest, collar_deg):
    out = []
    for collar, shoulder, elbow, wrist in ARMS:
        l1 = np.linalg.norm(rest[elbow] - rest[shoulder]); l2 = np.linalg.norm(rest[wrist] - rest[elbow])
        straight = math.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * math.cos(math.radians(REACH_ELBOW_DEG)))
        out.append(straight + 2 * np.linalg.norm(rest[shoulder] - rest[collar]) * math.sin(math.radians(collar_deg) / 2))
    return out


def reach_deficit(rest, j, g, h, goal, offsets, collar_deg):
    """Shoulder-to-wrist-goal distance beyond the 165 deg elbow reach plus collar slack (wrist world frame kept)."""
    collar, shoulder, elbow, wrist = ARMS[h]
    wrist_goal = goal - np.einsum('tij,j->ti', g[:, wrist], offsets[h])
    return np.maximum(np.linalg.norm(wrist_goal - j[:, shoulder], axis=1) - arm_reach(rest, collar_deg)[h], 0)


def leg_limits(rest, g, leg):
    from .anatomical_ik import reach_lengths, minimum_reach_lengths
    hip, knee, ankle = LEGS[leg]
    l1 = np.linalg.norm(rest[knee] - rest[hip]); l2 = np.linalg.norm(rest[ankle] - rest[knee])
    knee_cap = math.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * math.cos(math.radians(KNEE_MAX_DEG)))
    return minimum_reach_lengths(rest, g)[:, leg] * 1.01, np.minimum(reach_lengths(rest, g)[:, leg] * .995, knee_cap)


def move_leg(rest, root, quat, leg, shift, toward):
    """Ankle goal moved toward a point by shift (m) with anatomical_ik; returns new quats and the ankle displacement."""
    from .correction import fk
    from .anatomical_ik import solve_legs
    j, g = fk(rest, root, quat)
    hip, knee, ankle = LEGS[leg]
    direction = toward - j[:, ankle]; direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    targets = j[:, [7, 8]].copy(); goal = j[:, ankle] + direction * shift[:, None]
    low, high = leg_limits(rest, g, leg)
    span = goal - j[:, hip]; length = np.linalg.norm(span, axis=1)
    goal = j[:, hip] + span * (np.clip(length, low, high) / np.maximum(length, 1e-12))[:, None]
    targets[:, leg] = goal
    active = np.zeros((len(root), 2), bool); active[:, leg] = shift > 0
    out = quat.copy()
    if active.any():
        local, _ = solve_legs(rest, j, g, root, targets, active)
        ids = np.flatnonzero(active[:, leg])
        out[np.ix_(ids, [hip, knee, ankle])] = matrix_quat(local[np.ix_(ids, [hip, knee, ankle])])
    moved, _ = fk(rest, root, out)
    return out, np.linalg.norm(moved[:, ankle] - j[:, ankle], axis=1)


def solve_arms(rest, root, quat, prior, offsets, fps, first, goals, config):
    from .contact_arm_plan import solve
    from .palm_contact import Window
    scene = SimpleNamespace(geo=SimpleNamespace(rest=np.asarray(rest, float)), quat=prior, offsets=np.asarray(offsets, float), strength=1.,
                            branch_cache={})
    scene.asset = SimpleNamespace(window=lambda a, b, r, q: Window(scene, r, q))
    tracks = MovingTracks(first, goals)
    out, report = solve(scene, root, quat, tracks.ranges, tracks, fps, first=first, config=config)
    return np.asarray(out, float), report


def clamp_elbows(rest, root, quat, hands, limits=None):
    """Flex elbows straighter than their limit (deg, per hand scalar or per frame) about the hinge; keep wrist world rotation."""
    from .correction import fk
    from .mixamo_rig import align
    from .contact_arm_plan import align_many
    j, g = fk(rest, root, quat); local = quat_matrix(quat); clamped = 0
    for h in hands:
        collar, shoulder, elbow, wrist = ARMS[h]; sign = 1 if h == 0 else -1
        limit = np.broadcast_to(ELBOW_MAX_DEG if limits is None else limits[h], (len(root),))
        ids = np.flatnonzero(interior(j[:, shoulder], j[:, elbow], j[:, wrist]) > limit + 1e-6)
        if not len(ids):continue
        limit = limit[ids, None]
        u = rest[elbow] - rest[shoulder]; v0 = rest[wrist] - rest[elbow]
        axis = align([sign, 0, 0], u) @ np.array([0., -sign, 0.])
        angles = np.deg2rad(np.linspace(-40, 40, 321)); bends = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
        forearm = np.einsum('kij,tjl,l->tki', bends, local[ids, elbow], v0)
        angle = 180 - np.rad2deg(np.arccos(np.clip(forearm @ u / np.linalg.norm(u) / np.linalg.norm(v0), -1, 1)))
        cost = np.where(angle <= limit, np.abs(angles)[None], np.inf)
        best = np.where(np.isfinite(cost.min(1)), cost.argmin(1), angle.argmin(1))
        new_elbow = bends[best] @ local[ids, elbow]
        upper = g[ids, shoulder]
        old_end = np.einsum('tij,tj->ti', upper, u + np.einsum('tij,j->ti', local[ids, elbow], v0))
        new_end = np.einsum('tij,tj->ti', upper, u + np.einsum('tij,j->ti', new_elbow, v0))
        upper = align_many(new_end, old_end) @ upper
        local[ids, shoulder] = g[ids, collar].swapaxes(-1, -2) @ upper
        local[ids, elbow] = new_elbow
        local[ids, wrist] = (upper @ new_elbow).swapaxes(-1, -2) @ g[ids, wrist]
        quat[np.ix_(ids, [shoulder, elbow, wrist])] = matrix_quat(local[np.ix_(ids, [shoulder, elbow, wrist])])
        clamped += len(ids)
    return quat, clamped


def blend(rest, root, start, solved, weights, joints, distal):
    """Slerp local rotations of joints by weights, then restore the distal joints' world rotation from start."""
    from .correction import fk, PARENTS
    out = start.copy()
    out[:, joints] = slerp(start[:, joints], solved[:, joints], weights[:, None])
    _, before_world = fk(rest, root, start); _, world = fk(rest, root, out)
    for d in distal:
        out[:, d] = matrix_quat(world[:, PARENTS[d]].swapaxes(-1, -2) @ before_world[:, d])
    return out


def solve_interval(rest, roots, quats, offsets, fps, track, interval, foot_ranges):
    from .correction import fk
    from .contact_arm_plan import source_reach_gap
    started = time.perf_counter(); n = len(roots); a, b = interval
    h = 0 if track['effector'] == 'LeftHand' else 1
    tj, side, part = parse_target(track['target'])
    entry = dict(id=track['id'], effector=track['effector'], target=track['target'], start=int(a), end=int(b))
    j, g = fk(rest, roots[a:b], quats[a:b]); p, _, _ = palms(rest, roots[a:b], quats[a:b], offsets)
    base, frame = target_base(j, g, p, tj, part)
    offset = local_offset(track, part, base, frame, p[:, h])
    travel = float(np.linalg.norm(p[:, h] - base - np.einsum('tij,j->ti', frame, offset), axis=1).max())
    ease = max(6, math.ceil(travel * 100 / 3))
    lo = max(0, a - ease); hi = min(n, b + ease)
    entry.update(offset_local_m=offset.tolist(), travel_m=travel, ease_frames=ease, window=[lo, hi])
    if hi - lo < 3:return None, dict(entry, skipped='interval shorter than three frames')
    root = np.asarray(roots[lo:hi], float); start = np.asarray(quats[lo:hi], float); work = start.copy()
    weights = ease_weights(lo, hi, a, b, ease); inside = slice(a - lo, b - lo)

    def goals(q):
        jj, gg = fk(rest, root, q); pp, _, _ = palms(rest, root, q, offsets)
        bb, ff = target_base(jj, gg, pp, tj, part)
        return jj, gg, pp, bb + np.einsum('tij,j->ti', ff, offset)

    def deficit_of(q, jj, gg, pp, goal):
        # The arm planner fades out where its own bounded collar/elbow model cannot reach, so take the larger gap.
        gap = source_reach_gap(rest, jj, gg, quat_matrix(q), h, goal, pp[:, h], ARM_CONFIG['collar_limit_deg'], ARM_CONFIG['elbow_limit_deg'])
        return np.maximum(reach_deficit(rest, jj, gg, h, goal, offsets, ARM_CONFIG['collar_limit_deg']), gap)

    def held(values):
        """Interval values, held constant over the ease frames."""
        out = np.empty(hi - lo); out[inside] = values[inside]; out[:a - lo] = values[a - lo]; out[b - lo:] = values[b - lo - 1]
        return out

    jj, gg, pp, goal = goals(work)
    deficit = deficit_of(work, jj, gg, pp, goal)
    entry['reach_deficit_max_m'] = float(deficit[inside].max())
    reciprocal = track['reciprocal'] == 'always' or (track['reciprocal'] == 'auto' and bool((deficit[inside] > 0).any()))
    joints = list(ARMS[h]); distal = [ARMS[h][3]]; hands = [h]; arm_goals = {h: goal}
    if reciprocal and part in LEG_PARTS:
        leg = side; hip, knee, ankle = LEGS[leg]
        locked = np.zeros(hi - lo, bool)
        for s, e in foot_ranges[leg]:locked[max(s, lo) - lo:max(min(e, hi) - lo, 0)] = True
        toward = jj[:, ARMS[h][1]]
        raw = held(np.where((deficit > 0) | (track['reciprocal'] == 'always'), deficit + RECIPROCAL_MARGIN_M, 0.))
        for _ in range(3):
            need = gaussian_filter1d(maximum_filter1d(raw, max(1, round(fps * .3)), mode='nearest'), max(.5, fps * .1), mode='nearest')
            need[need < 1e-4] = 0; conflict = locked & (need > 0); need[locked] = 0
            work, shift = move_leg(rest, root, start, leg, need, toward)
            jj, gg, pp, goal = goals(work); left = deficit_of(work, jj, gg, pp, goal)
            if not (left[inside] > .002).any():break
            raw = raw + held(np.where(left > .002, left + RECIPROCAL_MARGIN_M / 2, 0.))
        joints += [hip, knee, ankle]; distal.append(ankle)
        entry['reciprocal'] = dict(joint=('Left' if leg == 0 else 'Right') + 'Foot', requested_shift_max_m=float(need.max()),
                                   ankle_shift_max_m=float(shift.max()), moved_frames=int((shift > 1e-6).sum()), foot_lock_conflict_frames=int(conflict.sum()))
        arm_goals = {h: goal}
    elif reciprocal and part == 'Hand':
        other = tj - 20; distance = DISTANCE['Hand'] if track['distance'] == 'auto' else float(track['distance'])
        middle = (pp[:, h] + pp[:, other]) / 2; line = pp[:, h] - pp[:, other]
        line /= np.maximum(np.linalg.norm(line, axis=1, keepdims=True), 1e-12)
        arm_goals = {h: middle + line * distance / 2, other: middle - line * distance / 2}
        joints += list(ARMS[other]); distal.append(ARMS[other][3]); hands.append(other)
        entry['reciprocal'] = dict(joint=track['target'], mode='both arms meet at the midpoint')
    elif reciprocal:
        entry['reciprocal'] = dict(joint=None, note='no reciprocal limb for this target; shortfall reported')
    deficit_after = deficit_of(work, jj, gg, pp, arm_goals[h])
    entry['reach_deficit_after_reciprocal_max_m'] = float(deficit_after[inside].max())
    # Ease-frame goals are only blended in, but the planner's reach gate is temporally smoothed: an unreachable
    # ease goal would fade the first and last contact frames too, so pull those goals into reach.
    outside = np.ones(hi - lo, bool); outside[inside] = False
    for _ in range(3):
        gap = np.where(outside, deficit_of(work, jj, gg, pp, arm_goals[h]), 0.)
        if not (gap > .005).any():break
        pull = jj[:, ARMS[h][1]] - arm_goals[h]; pull /= np.maximum(np.linalg.norm(pull, axis=1, keepdims=True), 1e-12)
        arm_goals[h] = arm_goals[h] + pull * np.where(gap > .005, gap + RECIPROCAL_MARGIN_M, 0.)[:, None]
    prior = np.asarray(quats, float).copy(); prior[lo:hi] = work
    solved, arm_report = solve_arms(rest, root, work, prior, offsets, fps, lo, arm_goals, ARM_CONFIG)
    solved, clamped = clamp_elbows(rest, root, solved, hands)
    final = blend(rest, root, start, solved, weights, sorted(set(joints)), distal)
    j0, g0 = fk(rest, root, start); js, _ = fk(rest, root, solved)
    # Slerping twisted elbow frames can straighten past both ends; never end straighter than the limit or the source.
    final, eased_clamped = clamp_elbows(rest, root, final, hands, {k: np.maximum(ELBOW_MAX_DEG, interior(*(j0[:, ARMS[k][i]] for i in (1, 2, 3))))
                                                                  for k in hands})
    changed = weights > 0
    jf, gf = fk(rest, root, final); pf, _, _ = palms(rest, root, final, offsets)
    bf, ff = target_base(jf, gf, pf, tj, part)
    pin = arm_goals[h] if part == 'Hand' and len(hands) == 2 else bf + np.einsum('tij,j->ti', ff, offset)
    residual = np.linalg.norm(pf[:, h] - pin, axis=1)[inside]
    elbows = {}
    for k in hands:
        _, s_, e_, w_ = ARMS[k]
        source, full, eased = (interior(x[:, s_], x[:, e_], x[:, w_]) for x in (j0, js, jf))
        ease_frames = changed & (weights < 1)
        fold = float(np.maximum(np.minimum(source, full) - eased, 0)[ease_frames].max()) if ease_frames.any() else 0.
        name = ('Left' if k == 0 else 'Right') + 'ForeArm'
        elbows[name] = dict(interval_interior_max_deg=float(eased[inside].max()), interior_max_deg=float(eased[changed].max()),
                            interior_min_deg=float(eased[changed].min()), source_interior_max_deg=float(source[changed].max()), ease_fold_deg=fold,
                            ease_overshoot_deg=float(np.maximum(eased - np.maximum(source, full), 0)[ease_frames].max())
                            if ease_frames.any() else 0.)
    wrist_change = max(float(np.rad2deg(Rotation.from_matrix((g0[:, ARMS[k][3]].swapaxes(-1, -2) @ gf[:, ARMS[k][3]])[changed]).magnitude()).max())
                       for k in hands)
    entry.update(residual_max_m=float(residual.max()), residual_median_m=float(np.median(residual)), full_contact_frames=int(b - a),
                 palm_to_target_joint_median_m=float(np.median(np.linalg.norm(pf[:, h] - jf[:, tj], axis=1)[inside])),
                 palm_to_target_joint_max_m=float(np.linalg.norm(pf[:, h] - jf[:, tj], axis=1)[inside].max()),
                 elbows=elbows, elbow_clamped_frames=int(clamped), eased_elbow_clamped_frames=int(eased_clamped), wrist_world_change_max_deg=wrist_change,
                 joints=sorted(set(joints)),
                 arm_solver=[dict(hand=r['hand'], iterations=r['iterations'], success=r['success'], reach_gap_max_m=r['reach_gap_max_m'],
                                  reach_gain_min=r['reach_gain_min'], selected_elbow_limit_deg=r['selected_elbow_limit_deg'])
                             for r in arm_report.get('hands', [])],
                 grip='not implemented' if track['grip'] != 'off' else 'off')
    if 'reciprocal' in entry and part in LEG_PARTS:
        leg = side
        entry['reciprocal']['knee_interior_max_deg'] = float(interior(*(jf[:, LEGS[leg][i]] for i in range(3)))[changed].max())
    ids = np.flatnonzero(changed)
    result = final[ids[0]:ids[-1] + 1]
    entry['seconds'] = time.perf_counter() - started
    return (lo + int(ids[0]), lo + int(ids[-1]) + 1, sorted(set(joints)), result), entry


def apply(rest, out_roots, out_quats, offsets, fps, tracks, foot_ranges, emit=lambda *a, **k: None):
    """Solve enabled tracks in order on the corrected pose, mutating out_quats; None when nothing is enabled."""
    work = [(t, r) for t in tracks if t.get('enabled', True) for r in t['ranges']]
    if not work:return None
    started = time.perf_counter(); rest = np.asarray(rest, float)[:22]; offsets = np.asarray(offsets, float)
    entries = []; changed = []; warnings = []
    for index, (track, interval) in enumerate(work):
        update, entry = solve_interval(rest, out_roots, out_quats, offsets, fps, track, interval, foot_ranges)
        entries.append(entry)
        if update is not None:
            s, e, joints, values = update
            out_quats[s:e, joints] = values[:, joints].astype(out_quats.dtype)
            changed.append(dict(joints=[int(x) for x in joints], start=int(s), end=int(e)))
            if entry['residual_max_m'] > .03:
                warnings.append('%s -> %s %d-%d: palm stays up to %.1f cm from its target.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, 100 * entry['residual_max_m']))
            if entry.get('reciprocal', {}).get('foot_lock_conflict_frames'):
                warnings.append('%s -> %s %d-%d: a foot lock kept the target leg still on %d frames.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, entry['reciprocal']['foot_lock_conflict_frames']))
        emit('progress', stage='correct', current=index + 1, total=len(work), phase='self_contact', message='Motion Correction: solving self-contacts')
    grip = sorted({t['id'] for t, _ in work if t.get('grip', 'off') != 'off'})
    return dict(method=METHOD, tracks=entries, changed=changed, seconds=time.perf_counter() - started,
                grip_not_implemented=grip, warning=' '.join(['Self-contact: ' + w for w in warnings]))


def from_store(project, base, context, rest, out_roots, out_quats, offsets, fps, segments, emit=lambda *a, **k: None):
    """correction.run hook: (report or None, pre-self-contact quats or None). No store file = (None, None)."""
    from .self_contact_store import read
    tracks, revision, warning = read(project, base, context)
    if not tracks and not warning:return None, None
    foot_ranges = [[(s['start'], s['end']) for s in segments if s['foot'] == f] for f in (0, 1)]
    before = np.array(out_quats, copy=True)
    report = apply(rest, out_roots, out_quats, offsets, fps, tracks, foot_ranges, emit) or dict(method=METHOD, tracks=[], changed=[], warning='')
    report['revision'] = revision
    report['warning'] = ' '.join(filter(None, (report['warning'], warning)))
    return report, (before if report['changed'] else None)


def unwritten(changed, written):
    """(joint, start, end) rows a legacy write-back that publishes whole joints in written would miss."""
    return [(j, c['start'], c['end']) for c in changed for j in c['joints'] if j not in written]


def publish_smplx(output, params, pose, before, changed, written):
    """Write changed rows into output['body_pose']; return the body_pose that the correction would have had without them."""
    import torch
    for j, s, e in unwritten(changed, written):
        output['body_pose'][s:e, 3 * (j - 1):3 * j] = torch.from_numpy(np.asarray(pose[s:e, j]).copy())
    prior = output['body_pose'].clone()
    for c in changed:
        s, e = c['start'], c['end']
        for j in c['joints']:
            if j in written:value = torch.from_numpy(Rotation.from_quat(before[s:e, j][:, [1, 2, 3, 0]]).as_rotvec().astype(np.float32))
            else:value = params['body_pose'][s:e, 3 * (j - 1):3 * j]
            prior[s:e, 3 * (j - 1):3 * j] = value
    return prior


def publish_rot6d(output, params, matrices, before, changed, written):
    """SMPL-H rot6d variant of publish_smplx; returns the prior rot6d tensor (1, n, joints, 6)."""
    import torch
    for j, s, e in unwritten(changed, written):
        output['rot6d'][0, s:e, j] = torch.from_numpy(matrices[s:e, j, :, :2].reshape(e - s, 6).astype(np.float32))
    prior = output['rot6d'].clone()
    for c in changed:
        s, e = c['start'], c['end']
        for j in c['joints']:
            if j in written:prior[0, s:e, j] = torch.from_numpy(quat_matrix(before[s:e, j])[:, :, :2].reshape(e - s, 6).astype(np.float32))
            else:prior[0, s:e, j] = params['rot6d'][0, s:e, j]
    return prior
