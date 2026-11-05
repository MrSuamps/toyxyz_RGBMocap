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
# Wrist range of motion (deg) relative to the forearm, normal adult active ranges (AAOS; Norkin & White). Swing about the
# radial-ulnar axis is flexion/extension, about the palm normal radial/ulnar deviation (bounded together by an elliptical
# cone per quadrant); twist about the forearm axis is pronation/supination, zero at the T-pose mid-position (palm down,
# thumb forward). SMPL forearm twist is shared between elbow and wrist; only the wrist-local part is limited here.
WRIST_LIMITS_DEG = dict(flexion=80., extension=70., radial=20., ulnar=30., pronation=80., supination=80.)
# Wrist-local axes in the SMPL rest pose: palms face -Y, thumbs point +Z, the hand continues the rest forearm axis.
PALM_NORMAL = np.array([0., -1., 0.])
THUMB = np.array([0., 0., 1.])
# Grip shape: absolute flexion (deg) of MCP/PIP/DIP per finger in MANO order (index, middle, pinky, ring, thumb), a wrap
# around a ~4 cm limb; the thumb row is CMC opposition, MCP and IP. Scaled per target part (thick or flat targets less).
FINGER_CURL_DEG = ((45., 60., 35.), (50., 65., 35.), (60., 65., 35.), (55., 65., 35.), (35., 25., 30.))
GRIP_CURL = {'Foot': 1., 'ToeBase': .8, 'Leg': 1., 'UpLeg': .6, 'Shoulder': .4, 'Arm': .9, 'ForeArm': 1., 'Hand': .5, 'Neck': .8, 'Head': .3,
             'Hips': .3, 'Spine': .25, 'Spine1': .25, 'Spine2': .25}
CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
# Grip search: coarse wrist-limit grid (flexion, radial, pronation samples), finger-wrap and frame-to-frame weights per
# degree of palm-facing error, and the arm swivel about the shoulder-wrist line it may add (penalised per degree).
GRIP_GRID_STEPS = (11, 6, 9)
GRIP_FINGER_WEIGHT = .25
GRIP_STEP_WEIGHT = 10.
SWIVELS_DEG = np.arange(-45., 46., 9.)
SWIVEL_WEIGHT = .3
SWIVEL_STEP_WEIGHT = 10.


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


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def forearm_axis(rest, h):
    _, _, elbow, wrist = ARMS[h]
    return unit(rest[wrist] - rest[elbow])


def wrist_basis(rest, h):
    """Forearm twist axis t and the swing axes ez (radial-ulnar, thumb side) and ey (palm normal side) orthogonal to it."""
    t = forearm_axis(rest, h)
    ez = unit(THUMB - (THUMB @ t) * t)
    ey = unit(-PALM_NORMAL - (-PALM_NORMAL @ t) * t - (-PALM_NORMAL @ ez) * ez)
    return t, ez, ey


def wrist_angles(local, rest, h):
    """(flexion, radial, pronation) in degrees of wrist-local rotations (..., 3, 3), swing-twist about the rest forearm axis.

    Left forearm runs +X: +Z rotation lifts the fingers (extension), +Y turns them to the pinky (ulnar), +X turns the thumb
    down (pronation). The right forearm runs -X, which mirrors all three signs.
    """
    t, ez, ey = wrist_basis(rest, h); s = 1. if h == 0 else -1.
    q = Rotation.from_matrix(np.asarray(local, float).reshape(-1, 3, 3)).as_quat()
    q = np.where(q[:, 3:] < 0, -q, q)
    projection = q[:, :3] @ t
    twist = 2 * np.arctan2(projection, q[:, 3])
    twist_q = np.c_[projection[:, None] * t, q[:, 3]]
    swing = (Rotation.from_quat(q) * Rotation.from_quat(unit(twist_q)).inv()).as_rotvec()
    shape = np.shape(local)[:-2]
    return tuple(np.rad2deg(x).reshape(shape) for x in (-s * (swing @ ez), -s * (swing @ ey), s * twist))


def swing_cone(flex, radial):
    limits = WRIST_LIMITS_DEG
    return np.hypot(flex / np.where(flex >= 0, limits['flexion'], limits['extension']), radial / np.where(radial >= 0, limits['radial'], limits['ulnar']))


def limit_wrist(flex, radial, pron):
    """Angles (deg) moved into the limits: the swing scaled radially onto its cone, the twist clipped."""
    scale = 1 / np.maximum(swing_cone(flex, radial), 1)
    return flex * scale, radial * scale, np.clip(pron, -WRIST_LIMITS_DEG['supination'], WRIST_LIMITS_DEG['pronation'])


def wrist_local(flex, radial, pron, rest, h):
    """Wrist-local rotation matrices from (flexion, radial, pronation) degrees; inverse of wrist_angles."""
    t, ez, ey = wrist_basis(rest, h); s = 1. if h == 0 else -1.
    f, r, p = (np.deg2rad(np.asarray(x, float)).ravel() for x in (flex, radial, pron))
    swing = Rotation.from_rotvec(-s * f[:, None] * ez - s * r[:, None] * ey)
    return (swing * Rotation.from_rotvec(s * p[:, None] * t)).as_matrix().reshape(np.shape(flex) + (3, 3))


def clamp_wrist(local, rest, h):
    """Wrist-local rotations pulled into WRIST_LIMITS_DEG; frames already inside are returned unchanged."""
    local = np.asarray(local, float)
    angles = wrist_angles(local, rest, h); limited = limit_wrist(*angles)
    outside = (limited[0] != angles[0]) | (limited[2] != angles[2])
    out = local.copy()
    if outside.any():out[outside] = wrist_local(*(x[outside] for x in limited), rest, h)
    return out


def angle_between(a, b):
    return np.rad2deg(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def rotation_angle(a, b):
    """Angle in degrees of a^T b for stacks of rotation matrices."""
    trace = np.einsum('...ji,...ji->...', a, b)
    return np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1, 1)))


def limb_segments(tj):
    """Bone segments (joint, joint) adjacent to the target joint; 'palm' is the target hand's palm point."""
    from .correction import PARENTS
    if tj in (20, 21):return [(ARMS[tj - 20][2], tj), (tj, 'palm')]
    return ([(PARENTS[tj], tj)] if PARENTS[tj] >= 0 else []) + ([(tj, CHILD[tj])] if tj in CHILD else [])


def grip_geometry(rest, root, q, offsets, tj, point, segment=None):
    """Palm-facing direction n (target point toward the limb axis), the limb axis d and the chosen segment on pose q."""
    from .correction import fk
    jj, gg = fk(rest, root, q); pp, _, _ = palms(rest, root, q, offsets)
    at = lambda k: pp[:, tj - 20] if k == 'palm' else jj[:, k]
    segments = limb_segments(tj)
    if segment is None:
        def distance(a, b):
            along = np.clip(((point - a) * (b - a)).sum(1) / np.maximum(((b - a) ** 2).sum(1), 1e-12), 0, 1)
            return np.median(np.linalg.norm(point - a - along[:, None] * (b - a), axis=1))
        segment = int(np.argmin([distance(at(a), at(b)) for a, b in segments]))
    a, b = (at(k) for k in segments[segment])
    d = unit(b - a)
    toward = a + ((point - a) * d).sum(1, keepdims=True) * d - point
    joint = jj[:, tj] - point
    n = np.where(np.linalg.norm(toward, axis=1, keepdims=True) > .005, unit(toward), unit(joint))
    return n, d, segment, jj, gg


def viterbi(unary, transition):
    """Index path minimising the summed per-frame costs (m, k) plus transition costs (k, k) between frames."""
    m, k = unary.shape; cost = unary[0].copy(); back = np.zeros((m, k), int)
    for i in range(1, m):
        total = cost[:, None] + transition
        back[i] = total.argmin(0); cost = total[back[i], np.arange(k)] + unary[i]
    path = np.empty(m, int); path[-1] = int(cost.argmin())
    for i in range(m - 1, 0, -1):path[i - 1] = back[i, path[i]]
    return path


def wrist_grid():
    """(k, 3) coarse (flexion, radial, pronation) samples inside the wrist limits."""
    limits = WRIST_LIMITS_DEG
    grid = np.stack(np.meshgrid(np.linspace(-limits['extension'], limits['flexion'], GRIP_GRID_STEPS[0]),
                                np.linspace(-limits['ulnar'], limits['radial'], GRIP_GRID_STEPS[1]),
                                np.linspace(-limits['supination'], limits['pronation'], GRIP_GRID_STEPS[2]), indexing='ij'), -1).reshape(-1, 3)
    return grid[swing_cone(grid[:, 0], grid[:, 1]) <= 1 + 1e-9]


def grip_cost(forearm, palm_local, finger_local, n, wrap, usable):
    """Palm-facing error plus weighted finger-wrap error (deg) for forearms (m, 3, 3) and wrist candidates (..., k, 3)."""
    palm = np.einsum('tij,t...kj->t...ki', forearm, palm_local)
    fingers = np.einsum('tij,t...kj->t...ki', forearm, finger_local)
    shape = (len(n),) + (1,) * (palm.ndim - 2) + (3,)
    miss = angle_between(fingers, wrap.reshape(shape)) * usable.reshape(shape[:-1])
    return angle_between(palm, n.reshape(shape)) + GRIP_FINGER_WEIGHT * miss


def grip_wrist(rest, h, forearm, n, d, sign, fps):
    """Wrist world rotations inside the limits with the palm normal closest to n and the fingers around the limb axis d.

    Viterbi over a coarse grid of the limit space (penalising wrist change between frames, so the twist never flips),
    then smoothing and a local refinement in angle space, where averaging stays inside the convex limits.
    """
    t = forearm_axis(rest, h); m = len(n)
    around = np.cross(d, n); usable = (np.linalg.norm(around, axis=1) > 1e-6).astype(float); wrap = sign * unit(around)
    grid = wrist_grid(); local = wrist_local(grid[:, 0], grid[:, 1], grid[:, 2], rest, h)
    unary = grip_cost(forearm, np.broadcast_to(local @ PALM_NORMAL, (m,) + grid.shape), np.broadcast_to(local @ t, (m,) + grid.shape), n, wrap, usable)
    path = viterbi(unary, GRIP_STEP_WEIGHT * rotation_angle(local[:, None], local[None]))
    sigma = max(.5, fps * .05)
    angles = gaussian_filter1d(grid[path], sigma, axis=0, mode='nearest')
    spacing = np.array([np.ptp(grid[:, i]) / (GRIP_GRID_STEPS[i] - 1) for i in range(3)]) / 2
    steps = np.stack(np.meshgrid(*[(-1., 0., 1.)] * 3, indexing='ij'), -1).reshape(-1, 3)
    for scale in (1., .5, .25):
        trial = np.stack(limit_wrist(*np.moveaxis(angles[:, None] + steps * spacing * scale, -1, 0)), -1)
        candidates = wrist_local(trial[..., 0], trial[..., 1], trial[..., 2], rest, h)
        cost = grip_cost(forearm, candidates @ PALM_NORMAL, candidates @ t, n, wrap, usable)
        angles = trial[np.arange(m), cost.argmin(1)]
    angles = np.stack(limit_wrist(*gaussian_filter1d(angles, sigma / 2, axis=0, mode='nearest').T), -1)
    return forearm @ wrist_local(angles[:, 0], angles[:, 1], angles[:, 2], rest, h)


def swivel_arm(rest, root, q, h, degrees):
    """Rotate the arm about its shoulder-wrist line by degrees per frame; wrist position and world rotation are kept."""
    from .correction import fk
    j, g = fk(rest, root, q); collar, shoulder, elbow, wrist = ARMS[h]
    spin = Rotation.from_rotvec(unit(j[:, wrist] - j[:, shoulder]) * np.deg2rad(degrees)[:, None]).as_matrix()
    out = q.copy()
    out[:, shoulder] = matrix_quat(g[:, collar].swapaxes(-1, -2) @ spin @ g[:, shoulder])
    out[:, wrist] = matrix_quat((spin @ g[:, elbow]).swapaxes(-1, -2) @ g[:, wrist])
    return out


def grip_swivel(rest, h, j, g, n, d, sign, fps):
    """Per-frame arm swivel (deg) that lets the limited wrist face the palm best, penalised by size and change."""
    _, shoulder, elbow, wrist = ARMS[h]; t = forearm_axis(rest, h)
    around = np.cross(d, n); usable = (np.linalg.norm(around, axis=1) > 1e-6).astype(float); wrap = sign * unit(around)
    grid = wrist_grid(); local = wrist_local(grid[:, 0], grid[:, 1], grid[:, 2], rest, h)
    axis = unit(j[:, wrist] - j[:, shoulder]); unary = np.empty((len(n), len(SWIVELS_DEG)))
    for i, degrees in enumerate(SWIVELS_DEG):
        forearm = Rotation.from_rotvec(axis * np.deg2rad(degrees)).as_matrix() @ g[:, elbow]
        shape = (len(n),) + grid.shape
        cost = grip_cost(forearm, np.broadcast_to(local @ PALM_NORMAL, shape), np.broadcast_to(local @ t, shape), n, wrap, usable)
        unary[:, i] = cost.min(1) + SWIVEL_WEIGHT * abs(degrees)
    path = viterbi(unary, SWIVEL_STEP_WEIGHT * np.abs(SWIVELS_DEG[:, None] - SWIVELS_DEG[None]))
    return gaussian_filter1d(SWIVELS_DEG[path].astype(float), max(.5, fps * .1), mode='nearest')


def set_wrist_world(rest, root, q, h, world):
    from .correction import fk
    _, g = fk(rest, root, q); out = q.copy(); _, _, elbow, wrist = ARMS[h]
    out[:, wrist] = matrix_quat(g[:, elbow].swapaxes(-1, -2) @ world)
    return out


def grip_arms(rest, root, prior, solved, offsets, fps, lo, goals, hands, h, tj, inside):
    """Turn the palm to the target limb, re-solve the arm so the moved palm point reaches the goal, clamp the wrist."""
    from .correction import fk
    _, _, elbow, wrist = ARMS[h]
    n, d, segment, jj, gg = grip_geometry(rest, root, solved, offsets, tj, goals[h])
    around = np.cross(d, n); fingers = gg[:, wrist] @ forearm_axis(rest, h)
    sign = 1. if np.median((fingers * around).sum(1)[inside]) >= 0 else -1.
    swivel = grip_swivel(rest, h, jj, gg, n, d, sign, fps)
    arm = swivel_arm(rest, root, solved, h, swivel)
    for iteration in range(1, 4):
        _, ga = fk(rest, root, arm)
        world = grip_wrist(rest, h, ga[:, elbow], n, d, sign, fps)
        # The planner starts from the gripping arm, so it only corrects the palm point moved by the new wrist rotation.
        graspable = set_wrist_world(rest, root, arm, h, world)
        full = prior.copy(); full[lo:lo + len(root)] = graspable
        solved, report = solve_arms(rest, root, graspable, full, offsets, fps, lo, goals, ARM_CONFIG)
        solved, clamped = clamp_elbows(rest, root, solved, hands)
        n, d, _, _, gg = grip_geometry(rest, root, solved, offsets, tj, goals[h], segment)
        local = gg[:, elbow].swapaxes(-1, -2) @ gg[:, wrist]
        excess = rotation_angle(local, clamp_wrist(local, rest, h))
        if excess.max() < .5:break
        arm = solved
    outside = np.flatnonzero(excess > 1e-4)
    if len(outside):
        solved = solved.copy(); solved[outside, wrist] = matrix_quat(clamp_wrist(local[outside], rest, h))
    return solved, report, clamped, dict(iterations=iteration, finger_side=int(sign), final_wrist_clamp_max_deg=float(excess.max()),
                                         arm_swivel_max_deg=float(np.abs(swivel).max()), arm_swivel_frames=int((np.abs(swivel) > .5).sum()))


def joint_name(k):
    from .mixamo_rig import BODY_NAMES
    return 'palm' if k == 'palm' else BODY_NAMES[k]


def wrist_report(rest, h, start_g, final_g, changed):
    _, _, elbow, wrist = ARMS[h]; limits = WRIST_LIMITS_DEG
    out = {}
    for label, g in (('source', start_g), ('result', final_g)):
        flex, radial, pron = wrist_angles(g[:, elbow].swapaxes(-1, -2) @ g[:, wrist], rest, h)
        f, r, p = flex[changed], radial[changed], pron[changed]
        cone = swing_cone(f, r)
        at_limit = (cone > .99) | (p > limits['pronation'] - .5) | (p < .5 - limits['supination'])
        out[label] = dict(flexion_max_deg=float(max(f.max(), 0)), extension_max_deg=float(max(-f.min(), 0)), radial_max_deg=float(max(r.max(), 0)),
                          ulnar_max_deg=float(max(-r.min(), 0)), pronation_max_deg=float(max(p.max(), 0)), supination_max_deg=float(max(-p.min(), 0)),
                          swing_cone_max=float(cone.max()), frames_at_limit=int(at_limit.sum()), frames_outside_limits=int(((cone > 1.001) |
                          (p > limits['pronation'] + .1) | (p < -limits['supination'] - .1)).sum()))
    out['limits_deg'] = dict(limits)
    return out


def grip_shape(rest, h):
    """(15, 3, 3) local finger rotations of the grip in MANO order; rest holds the full SMPL-X (55) or SMPL-H (52) joints."""
    first = (25 if len(rest) == 55 else 22) + 15 * h
    out = np.empty((15, 3, 3))
    for k in range(15):
        j = first + k; segment = k % 3
        direction = rest[j + 1] - rest[j] if segment < 2 else rest[j] - rest[j - 1]
        axis = unit(np.cross(direction, PALM_NORMAL))
        out[k] = Rotation.from_rotvec(axis * np.deg2rad(FINGER_CURL_DEG[k // 3][segment])).as_matrix()
    return out


def curl_fingers(current, target, weights):
    """Slerp (m, 15, 3, 3) finger rotations toward the grip target by per-frame weights."""
    q0 = Rotation.from_matrix(current.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(len(current), 15, 4)
    q1 = np.broadcast_to(Rotation.from_matrix(target).as_quat()[:, [3, 0, 1, 2]], q0.shape)
    return quat_matrix(slerp(q0, q1, np.repeat(weights[:, None], 15, 1)))


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
    facing, _, segment, _, gs = grip_geometry(rest, root, solved, offsets, tj, arm_goals[h])
    palm_before = angle_between(gs[:, ARMS[h][3]] @ PALM_NORMAL, facing)[inside]
    grip = 'off'
    if track['grip'] == 'auto':
        solved, arm_report, clamped, grip = grip_arms(rest, root, prior, solved, offsets, fps, lo, arm_goals, hands, h, tj, inside)
        # The gripping wrist is slerped in local space like the arm; only other distal joints keep their source world frame.
        distal.remove(ARMS[h][3])
        grip.update(mode='auto', fingers=dict(hand=h, window=[int(lo), int(hi)], contact=[int(a), int(b)], ease=int(ease), curl=GRIP_CURL.get(part, .5)))
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
                 grip=grip)
    facing, _, _, _, _ = grip_geometry(rest, root, final, offsets, tj, arm_goals[h], segment)
    palm_after = angle_between(gf[:, ARMS[h][3]] @ PALM_NORMAL, facing)[inside]
    entry['palm_facing_deg'] = dict(before_median=float(np.median(palm_before)), before_max=float(palm_before.max()),
                                    after_median=float(np.median(palm_after)), after_max=float(palm_after.max()),
                                    limb_axis=[joint_name(k) for k in limb_segments(tj)[segment]])
    entry['wrist'] = wrist_report(rest, h, g0, gf, changed)
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
    entries = []; changed = []; warnings = []; wrists = []; fingers = []
    for index, (track, interval) in enumerate(work):
        update, entry = solve_interval(rest, out_roots, out_quats, offsets, fps, track, interval, foot_ranges)
        entries.append(entry)
        if update is not None:
            s, e, joints, values = update
            out_quats[s:e, joints] = values[:, joints].astype(out_quats.dtype)
            changed.append(dict(joints=[int(x) for x in joints], start=int(s), end=int(e)))
            if isinstance(entry['grip'], dict):
                wrists.append(dict(joint=ARMS[entry['grip']['fingers']['hand']][3], start=int(s), end=int(e)))
                fingers.append(entry['grip']['fingers'])
            if entry['residual_max_m'] > .03:
                warnings.append('%s -> %s %d-%d: palm stays up to %.1f cm from its target.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, 100 * entry['residual_max_m']))
            if entry.get('reciprocal', {}).get('foot_lock_conflict_frames'):
                warnings.append('%s -> %s %d-%d: a foot lock kept the target leg still on %d frames.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, entry['reciprocal']['foot_lock_conflict_frames']))
        emit('progress', stage='correct', current=index + 1, total=len(work), phase='self_contact', message='Motion Correction: solving self-contacts')
    # wrist_changed declares the only rows whose wrist world rotation may differ from the source (grip).
    return dict(method=METHOD, tracks=entries, changed=changed, seconds=time.perf_counter() - started,
                wrist_changed=wrists, fingers=fingers, warning=' '.join(['Self-contact: ' + w for w in warnings]))


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


def audit_exemptions(report, preserved):
    """(start, end, column ids) of the export upper-body audit a self-contact report exempts, columns indexing preserved.

    Changed arm joints are exempt on their rows; a wrist (whose world rotation is otherwise kept) only on wrist_changed rows.
    """
    report = report or {}
    rows = [(c['start'], c['end'], [preserved.index(j) for j in c['joints'] if j in preserved and j not in (20, 21)]) for c in report.get('changed', [])]
    return rows + [(c['start'], c['end'], [preserved.index(c['joint'])]) for c in report.get('wrist_changed', [])
                   if c['joint'] in (20, 21) and c['joint'] in preserved]


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


def finger_rows(spec):
    """(frame ids, weights) of a finger spec; only frames with a positive weight are written."""
    lo, hi = spec['window']
    weights = ease_weights(lo, hi, *spec['contact'], spec['ease']) * spec['curl']
    ids = np.flatnonzero(weights > 0)
    return lo + ids, weights[ids]


def note_fingers(spec, applied, before=None, after=None, note=None):
    spec['applied'] = applied
    if note:spec['note'] = note
    if applied:
        spec['joint_change_max_deg'] = float(rotation_angle(before, after).max())
        spec['frames'] = int(len(before))


def publish_fingers_smplx(output, params, report, rest, mean):
    """Blend the grip into 45-D SMPL-X hand poses (residuals to the hand mean); return the source hand fields replaced.

    rest is the full 55-joint SMPL-X rest and mean the (55, 3) pose mean, whose rows 25-39 / 40-54 are the hand means.
    """
    import torch
    replaced = {}
    for spec in (report or {}).get('fingers', []):
        h = spec['hand']; field = ('left', 'right')[h] + '_hand_pose'; value = params.get(field)
        if value is None or value.shape[-1] != 45 or len(rest) != 55:
            note_fingers(spec, False, note='no 45-D finger rotations in this result; fingers unchanged')
            continue
        if field not in replaced:
            replaced[field] = params[field]; output[field] = params[field].clone()
        ids, weights = finger_rows(spec); ids = torch.from_numpy(ids)
        hand_mean = np.asarray(mean, float)[25 + 15 * h:40 + 15 * h]
        current = Rotation.from_rotvec((output[field][ids].double().numpy().reshape(-1, 15, 3) + hand_mean).reshape(-1, 3)).as_matrix().reshape(-1, 15, 3, 3)
        curled = curl_fingers(current, grip_shape(rest, h), weights)
        values = Rotation.from_matrix(curled.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 15, 3) - hand_mean
        output[field][ids] = torch.from_numpy(values.reshape(-1, 45).astype(np.float32))
        note_fingers(spec, True, current, curled)
    return replaced


def rot6d_matrix(x):
    """FlowHMR rot6d (first two matrix columns, row-major (3, 2)) to rotation matrices."""
    x = np.asarray(x, float).reshape(np.shape(x)[:-1] + (3, 2))
    b1 = unit(x[..., 0]); b2 = unit(x[..., 1] - (b1 * x[..., 1]).sum(-1, keepdims=True) * b1)
    return np.stack((b1, b2, np.cross(b1, b2)), -1)


def publish_fingers_rot6d(output, report, rest):
    """SMPL-H variant: blend the grip into output['rot6d'] finger joints 22-36 (left) / 37-51 (right) in place."""
    import torch
    for spec in (report or {}).get('fingers', []):
        h = spec['hand']; rot6d = output['rot6d']
        if rot6d.shape[2] < 52 or len(rest) != 52:
            note_fingers(spec, False, note='no finger joints in this result; fingers unchanged')
            continue
        ids, weights = finger_rows(spec); ids = torch.from_numpy(ids); joints = slice(22 + 15 * h, 37 + 15 * h)
        current = rot6d_matrix(rot6d[0, ids, joints].double().numpy())
        curled = curl_fingers(current, grip_shape(rest, h), weights)
        rot6d[0, ids, joints] = torch.from_numpy(curled[..., :2].reshape(len(ids), 15, 6).astype(np.float32))
        note_fingers(spec, True, current, curled)
