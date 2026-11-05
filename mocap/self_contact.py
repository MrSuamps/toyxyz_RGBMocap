"""Self-contacts: a hand holds another body part, solved on the already corrected pose.

The hand reaches a point on the target's skin that moves with the body; a target leg (or the other arm) may move toward
it when the arm is short. The posed SMPL-X skin is checked after every arm solve and the goal pushed out until the hand no
longer sinks into the target. Changes are blended in local-rotation space over ease frames and never touch the root, trunk
or toes.
"""
import math
import time
from pathlib import Path
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
# Goals nearer the shoulder than the wrist widen the planner's elbow increment bound up to this, with a margin.
ELBOW_LIMIT_MAX_DEG = 150.
ELBOW_MARGIN_DEG = 30.
# Wrist range of motion (deg) relative to the forearm, normal adult active ranges (AAOS; Norkin & White). Swing about the
# radial-ulnar axis is flexion/extension, about the palm normal radial/ulnar deviation (bounded together by an elliptical
# cone per quadrant); twist about the forearm axis is pronation/supination, zero at the T-pose mid-position (palm down,
# thumb forward). SMPL forearm twist is shared between elbow and wrist; only the wrist-local part is limited here.
WRIST_LIMITS_DEG = dict(flexion=80., extension=70., radial=20., ulnar=30., pronation=80., supination=80.)
# Wrist-local axes in the SMPL rest pose: palms face -Y, thumbs point +Z, the hand continues the rest forearm axis.
PALM_NORMAL = np.array([0., -1., 0.])
THUMB = np.array([0., 0., 1.])
# Grip shape: absolute flexion (deg) of MCP/PIP/DIP per finger in MANO order (index, middle, pinky, ring, thumb), a wrap
# around a ~4 cm limb; the thumb row is CMC opposition, MCP and IP. Fingers curl toward it only until they touch the skin.
FINGER_CURL_DEG = ((45., 60., 35.), (50., 65., 35.), (60., 65., 35.), (55., 65., 35.), (35., 25., 30.))
CURL_STEPS = np.linspace(0., 1., 6)
FINGER_TOUCH_M = .004
CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}
# Grip from the video: coarse wrist-limit grid (flexion, radial, pronation samples) and Viterbi cost per degree of wrist
# change between frames; per-frame cost in degrees of 2D hand-direction error, plus the knuckle-row error (weighted) and
# a pull toward the source wrist (per degree), which also settles what the 2D view leaves free.
GRIP_GRID_STEPS = (11, 6, 9)
GRIP_STEP_WEIGHT = 3.
VIDEO_ACROSS_WEIGHT = .5
VIDEO_SOURCE_WEIGHT = .15
# Arm swivel for the video hand: weight of the 2D shoulder->elbow direction error and cost per degree of swivel.
VIDEO_ELBOW_WEIGHT = .5
VIDEO_SWIVEL_WEIGHT = .1
VIDEO_SWIVEL_STEP_WEIGHT = .5
HAND_SCORE = .6
HAND_SCORE_LOW = .3
HAND_MIN_TORSO = .06
# Skin contact: body segments are SMPL-X vertices by dominant skinning joint (55-joint order); fingers belong to the hand,
# toes to the foot. PART_SEGMENTS lists the segments a holding hand must stay out of; the first carries the contact point.
SEGMENT_JOINTS = {'left_hand': (20,) + tuple(range(25, 40)), 'right_hand': (21,) + tuple(range(40, 55)), 'left_forearm': (18,),
                  'right_forearm': (19,), 'left_upperarm': (16,), 'right_upperarm': (17,), 'left_thigh': (1,), 'right_thigh': (2,),
                  'left_shin': (4,), 'right_shin': (5,), 'left_foot': (7, 10), 'right_foot': (8, 11), 'head': (12, 15, 22),
                  'torso': (0, 3, 6, 9, 13, 14)}
FINGER_JOINTS = {'left_fingers': tuple(range(25, 40)), 'right_fingers': tuple(range(40, 55))}
SIDED = ('hand', 'forearm', 'upperarm', 'thigh', 'shin', 'foot', 'fingers')
SIDES = ('left', 'right')
PART_SEGMENTS = {'Foot': ('foot', 'shin'), 'ToeBase': ('foot', 'shin'), 'Leg': ('shin', 'foot', 'thigh'), 'UpLeg': ('thigh', 'shin', 'torso'),
                 'Hand': ('hand', 'forearm'), 'ForeArm': ('forearm', 'hand', 'upperarm'), 'Arm': ('upperarm', 'forearm', 'torso'),
                 'Shoulder': ('torso', 'upperarm'), 'Neck': ('head', 'torso'), 'Head': ('head', 'torso'), 'Hips': ('torso',),
                 'Spine': ('torso',), 'Spine1': ('torso',), 'Spine2': ('torso',)}
SKIN_STANDOFF_M = .002
PUSH_MARGIN_M = .003
PUSH_ITERATIONS = 4
PENETRATION_TOLERANCE_M = .002
# Elbow swivel that takes the forearm / upper arm out of the target: candidates, cost per degree and per degree of change
# between frames (1 mm of depth costs 1).
SWIVELS_DEG = np.arange(-90., 91., 15.)
SWIVEL_WEIGHT = .02
SWIVEL_STEP_WEIGHT = .2
NEAR_M = .05


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


def elbow_config(rest, j, g, h, goal, offsets, close):
    """ARM_CONFIG, or a copy whose elbow increment bound covers the extra bend that goals nearer than the wrist need."""
    if not close.any():return ARM_CONFIG
    _, shoulder, elbow, wrist = ARMS[h]
    l1 = np.linalg.norm(rest[elbow] - rest[shoulder]); l2 = np.linalg.norm(rest[wrist] - rest[elbow])
    wrist_goal = goal - np.einsum('tij,j->ti', g[:, wrist], offsets[h])
    d = np.clip(np.linalg.norm(wrist_goal - j[:, shoulder], axis=1), abs(l1 - l2), l1 + l2)
    needed = np.rad2deg(np.arccos(np.clip((l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2), -1, 1)))
    need = float((interior(j[:, shoulder], j[:, elbow], j[:, wrist]) - needed)[close].max()) + ELBOW_MARGIN_DEG
    if need <= ARM_CONFIG['elbow_limit_deg']:return ARM_CONFIG
    return dict(ARM_CONFIG, elbow_limit_deg=min(ELBOW_LIMIT_MAX_DEG, need))


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


def set_wrist_world(rest, root, q, h, world):
    from .correction import fk
    _, g = fk(rest, root, q); out = q.copy(); _, _, elbow, wrist = ARMS[h]
    out[:, wrist] = matrix_quat(g[:, elbow].swapaxes(-1, -2) @ world)
    return out


# Skin contact ---------------------------------------------------------------------------------------------------------

def segment_names(part, side):
    prefix = SIDES[side] + '_' if side is not None else ''
    return [(prefix if name in SIDED else '') + name for name in PART_SEGMENTS.get(part, ('torso',))]


def segment_vertices(weights):
    """{segment: vertex ids} by each vertex's dominant skinning joint, plus finger-only groups."""
    dominant = np.asarray(weights).argmax(1)
    return {name: np.flatnonzero(np.isin(dominant, joints)) for name, joints in {**SEGMENT_JOINTS, **FINGER_JOINTS}.items()}


class Segment:
    """One segment's surface, closed for inside tests by a fan cap over each open boundary loop (the cut to the next segment)."""

    def __init__(self, faces, ids):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        self.ids = np.asarray(ids)
        local = np.full(int(np.max(faces)) + 1, -1); local[self.ids] = np.arange(len(self.ids))
        own = local[faces]; self.faces = own[(own >= 0).all(1)]
        edges = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        _, inverse, counts = np.unique(np.sort(edges, 1), axis=0, return_inverse=True, return_counts=True)
        # Boundary edges keep their face's direction, so caps (v, u, centre) continue the surface orientation.
        self.open = edges[counts[inverse.ravel()] == 1]
        self.loop = np.zeros(len(self.open), int); self.loops = []
        if len(self.open):
            graph = coo_matrix((np.ones(len(self.open)), (self.open[:, 0], self.open[:, 1])), shape=(len(self.ids),) * 2)
            label = connected_components(graph, directed=False)[1]
            self.loop = np.unique(label[self.open[:, 0]], return_inverse=True)[1].ravel()
            self.loops = [np.unique(self.open[self.loop == k]) for k in range(self.loop.max() + 1)]

    def closed(self, v):
        """Triangles (F, 3, 3) of the posed segment vertices v (k, 3) plus the caps."""
        tri = v[self.faces]
        if not len(self.open):return tri
        centres = np.stack([v[k].mean(0) for k in self.loops])
        return np.concatenate([tri, np.stack([v[self.open[:, 1]], v[self.open[:, 0]], centres[self.loop]], 1)])


class Segments(dict):
    """Segment per name, built on first use from the mesh faces and segment_vertices groups."""

    def __init__(self, faces, groups):
        super().__init__(); self.faces, self.groups = np.asarray(faces), groups

    def __missing__(self, name):
        self[name] = Segment(self.faces, self.groups[name])
        return self[name]


def winding(points, tri, budget=4_000_000):
    """Generalised winding numbers of points (p, 3) for triangles (f, 3, 3): ~1 inside a closed surface, ~0 outside."""
    out = np.zeros(len(points)); step = max(1, budget // max(len(tri), 1))
    for s in range(0, len(points), step):
        d = tri[None] - points[s:s + step, None, None]
        a, b, c = d[:, :, 0], d[:, :, 1], d[:, :, 2]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = (a * np.cross(b, c)).sum(-1)
        div = la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
        out[s:s + step] = np.arctan2(det, div).sum(1) / (2 * np.pi)
    return out


def closest_points(p, a, b, c):
    """(closest points, region) on triangles (a, b, c) to p, all (..., 3); region 6 is the face interior, 0-5 a vertex or edge.

    Ericson, Real-Time Collision Detection 5.1.5.
    """
    dot = lambda x, y: (x * y).sum(-1, keepdims=True)
    ab, ac = b - a, c - a
    d1, d2 = dot(ab, p - a), dot(ac, p - a); d3, d4 = dot(ab, p - b), dot(ac, p - b); d5, d6 = dot(ab, p - c), dot(ac, p - c)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        edge_ab = a + ab * np.nan_to_num(d1 / (d1 - d3)); edge_ac = a + ac * np.nan_to_num(d2 / (d2 - d6))
        edge_bc = b + (c - b) * np.nan_to_num((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        total = va + vb + vc; total = np.where(np.abs(total) < 1e-30, 1e-30, total)
        face = a + ab * (vb / total) + ac * (vc / total)
    conditions = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0), (d6 >= 0) & (d5 <= d6),
                  (vb <= 0) & (d2 >= 0) & (d6 <= 0), (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)]
    region = np.select([x[..., 0] for x in conditions], range(6), 6)
    choices = np.stack(np.broadcast_arrays(a, b, edge_ab, c, edge_ac, edge_bc, face), -2)
    return np.take_along_axis(choices, region[..., None, None], -2)[..., 0, :], region


def surface_query(points, v, segment, k=8):
    """Distance, closest point, region and face normal of points (p, 3) on the segment's own surface (caps excluded).

    Candidates are the k faces with the nearest centroids; SMPL-X faces are small and even, so the closest face is among them.
    """
    from scipy.spatial import cKDTree
    tri = v[segment.faces]
    tri = tri[cKDTree(tri.mean(1)).query(points, min(k, len(tri)))[1].reshape(len(points), -1)]
    q, region = closest_points(points[:, None], tri[:, :, 0], tri[:, :, 1], tri[:, :, 2])
    d = np.linalg.norm(q - points[:, None], axis=-1)
    best = d.argmin(1); rows = np.arange(len(points)); tri = tri[rows, best]
    return d[rows, best], q[rows, best], region[rows, best], np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def inside_mask(points, v, segment, q, region, normal):
    """Inside the closed segment: by the closest face's outward normal where the closest point is inside that face (exact),
    by the generalised winding number where it lies on an edge or vertex (where the normal sign is unreliable)."""
    inside = ((points - q) * normal).sum(1) < 0
    ambiguous = region != 6
    if ambiguous.any():inside[ambiguous] = np.abs(winding(points[ambiguous], segment.closed(v))) > .5
    return inside


def penetration(points, v, segment, near=NEAR_M):
    """Effector points (p, 3) against one posed segment (v (k, 3)).

    depth: the deepest inside point's distance to the skin (0 when none); inside: inside point count; gap: the closest outside
    point's distance (inf beyond `near` of the segment's box); push: the (3,) move that takes the deepest point to the skin.
    """
    from scipy.spatial import cKDTree
    out = dict(depth=0., inside=0, gap=np.inf, push=np.zeros(3), point=None, surface=None)
    low, high = v.min(0), v.max(0)
    close = np.flatnonzero(((points > low - near) & (points < high + near)).all(1))
    if not len(close):return out
    p = points[close]
    vertex = cKDTree(v).query(p)[0]
    boxed = ((p >= low) & (p <= high)).all(1)
    # Outside the box a point is outside the segment: only those near the closest one can hold the gap.
    exact = np.flatnonzero(boxed | (~boxed & (vertex <= (vertex[~boxed].min() if (~boxed).any() else 0.) + .01)))
    distance, surface, region, normal = surface_query(p[exact], v, segment)
    inside = np.zeros(len(exact), bool); test = boxed[exact]
    if test.any():inside[test] = inside_mask(p[exact][test], v, segment, surface[test], region[test], normal[test])
    if inside.any():
        deepest = np.flatnonzero(inside)[distance[inside].argmax()]
        out.update(depth=float(distance[deepest]), inside=int(inside.sum()), push=surface[deepest] - p[exact][deepest])
    if (~inside).any():
        nearest = np.flatnonzero(~inside)[distance[~inside].argmin()]
        out.update(gap=float(distance[nearest]), point=p[exact][nearest], surface=surface[nearest])
    return out


def penetrations(points, posed, segments, names, near=NEAR_M):
    """penetration over several segments: depth, inside and push of the deepest; gap, point and surface of the first."""
    results = [penetration(points, posed[name], segments[name], near) for name in names]
    worst = max(results, key=lambda r: r['depth'])
    return dict(worst, gap=results[0]['gap'], point=results[0]['point'], surface=results[0]['surface'],
                segment=names[results.index(worst)] if worst['depth'] > 0 else None)


def signed_gap(result):
    """Skin-to-skin distance of a penetration result: negative depth when inside, nan when the hand is far away."""
    return -result['depth'] if result['depth'] > 0 else result['gap'] if np.isfinite(result['gap']) else np.nan


class Skin:
    """SMPL-X skin of one actor on correction arrays (root = pelvis joint position, 22 local body quats).

    LBS with pose correctives like the GUI Mesh view; fingers take the result's hand pose (else the pose mean), jaw and
    eyes the mean. rest holds the 55 shaped rest joints.
    """

    def __init__(self, shaped, weights, posedirs, parents, rest, mean, faces, hands=None):
        self.shaped = np.asarray(shaped, float); self.weights = np.asarray(weights, float)
        self.posedirs = np.asarray(posedirs, np.float32).reshape(486, -1, 3); self.parents = [int(p) for p in parents]
        self.rest = np.asarray(rest, float); self.mean = Rotation.from_rotvec(np.asarray(mean, float).reshape(55, 3)).as_matrix()
        self.hands = hands
        self.groups = segment_vertices(self.weights); self.segments = Segments(faces, self.groups)

    def fingers(self, rows, h):
        """(m, 15, 3, 3) source finger rotations of hand h on result rows."""
        if self.hands is None:return np.repeat(self.mean[None, 25 + 15 * h:40 + 15 * h], len(rows), 0)
        return self.hands[rows, 15 * h:15 * h + 15]

    def vertices(self, root, quat, rows, names, fingers=None):
        """{name: (m, k, 3)} posed vertices of the named groups; fingers optionally overrides {hand: (m, 15, 3, 3)}."""
        ids = [self.groups[name] for name in names]; every = np.concatenate(ids); m = len(root)
        rot = np.repeat(self.mean[None], m, 0)
        rot[:, :22] = quat_matrix(quat)
        if self.hands is not None:rot[:, 25:55] = self.hands[rows]
        for h, value in (fingers or {}).items():rot[:, 25 + 15 * h:40 + 15 * h] = value
        features = (rot[:, 1:] - np.eye(3)).reshape(m, -1).astype(np.float32)
        posed = self.shaped[every] + (features @ self.posedirs[:, every].reshape(486, -1)).reshape(m, -1, 3)
        world = np.empty_like(rot); joints = np.empty((m, 55, 3)); world[:, 0] = rot[:, 0]; joints[:, 0] = root
        for j in range(1, 55):
            p = self.parents[j]; world[:, j] = world[:, p] @ rot[:, j]
            joints[:, j] = joints[:, p] + np.einsum('tij,j->ti', world[:, p], self.rest[j] - self.rest[p])
        shift = joints - np.einsum('tjab,jb->tja', world, self.rest)
        w = self.weights[every]
        blend = (w @ world.reshape(m, 55, 9)).reshape(m, -1, 3, 3)
        out = np.einsum('tvab,tvb->tva', blend, posed) + w @ shift
        return dict(zip(names, np.split(out, np.cumsum([len(i) for i in ids])[:-1], 1)))


def load_skin(project, base, rest, body=None):
    """(Skin, None) of the base result's actor, or (None, reason) when the SMPL-X model does not reproduce rest (22 joints)."""
    import torch
    from .motion_tools import RESULTS
    try:
        if body is None:
            import smplx
            from .mamma_body import body_path
            body = smplx.SMPLX(str(body_path(Path(project))), gender='neutral', num_pca_comps=12, flat_hand_mean=False, batch_size=1).eval()
        filename, key = RESULTS[base]
        params = torch.load(Path(project) / filename, map_location='cpu', weights_only=False, mmap=True)[key]
        betas = params['betas'][0].double().numpy()
        shapedirs = body.shapedirs.double().numpy()[..., :len(betas)]
        shaped = body.v_template.double().numpy() + np.einsum('vci,i->vc', shapedirs, betas)
        joints = body.J_regressor.double().numpy() @ shaped
        if len(joints) != 55 or np.abs(joints[:22] - np.asarray(rest, float)[:22]).max() > 1e-4:
            return None, 'skin contact unavailable: the SMPL-X body does not match this result'
        joints[:22] = np.asarray(rest, float)[:22]
        n = len(params['body_pose']); mean = body.pose_mean.double().numpy().reshape(55, 3)
        hands = np.repeat(mean[None, 25:55], n, 0)
        for k, field in enumerate(('left_hand_pose', 'right_hand_pose')):
            value = params.get(field)
            if value is None:continue
            value = value.double().numpy().reshape(n, -1)
            if value.shape[1] == 12:value = value @ getattr(body, field.replace('_pose', '_components')).double().numpy()
            if value.shape[1] == 45:hands[:, 15 * k:15 * k + 15] += value.reshape(n, 15, 3)
        hands = Rotation.from_rotvec(hands.reshape(-1, 3)).as_matrix().reshape(n, 30, 3, 3)
        return Skin(shaped, body.lbs_weights.double().numpy(), body.posedirs.numpy(), body.parents.numpy(), joints, mean,
                    np.asarray(body.faces, np.int64), hands), None
    except (OSError, KeyError, ValueError, RuntimeError, ImportError, AttributeError, TypeError) as error:
        return None, f'skin contact unavailable ({error})'


def swivel_arm(rest, root, q, h, degrees):
    """Rotate the arm about its shoulder-wrist line by degrees per frame; wrist position and world rotation are kept."""
    from .correction import fk
    j, g = fk(rest, root, q); collar, shoulder, elbow, wrist = ARMS[h]
    spin = Rotation.from_rotvec(unit(j[:, wrist] - j[:, shoulder]) * np.deg2rad(degrees)[:, None]).as_matrix()
    out = q.copy()
    out[:, shoulder] = matrix_quat(g[:, collar].swapaxes(-1, -2) @ spin @ g[:, shoulder])
    out[:, wrist] = matrix_quat((spin @ g[:, elbow]).swapaxes(-1, -2) @ g[:, wrist])
    return out


def clear_arm(rest, skin, root, q, rows, h, names, arms, fps):
    """Swivel the elbow about the shoulder-wrist line where the forearm or upper arm sinks into the target, keeping the hand.

    Candidates are scored on the flagged frames (and a few around them) by depth plus a small cost per degree; a Viterbi
    path over all frames keeps the swivel continuous. Returns (quats, swivel degrees (m,), frames that needed it).
    """
    m = len(root)
    def arm_depth(quat, ids, stride=3):
        posed = skin.vertices(root[ids], quat[ids], rows[ids], list(arms) + list(names))
        out = np.zeros(len(ids))
        for t in range(len(ids)):
            targets = {name: posed[name][t] for name in names}
            out[t] = max(penetrations(posed[a][t, ::stride], targets, skin.segments, names)['depth'] for a in arms)
        return out
    every = np.arange(m)
    depth = arm_depth(q, every)
    bad = depth > PENETRATION_TOLERANCE_M
    if not bad.any():return q, np.zeros(m), 0
    near = np.flatnonzero(maximum_filter1d(bad, 7, mode='nearest'))
    picked = near[::2]
    unary = np.abs(SWIVELS_DEG)[None] * SWIVEL_WEIGHT * np.ones((m, 1))
    for k, degrees in enumerate(SWIVELS_DEG):
        turned = q if degrees == 0 else swivel_arm(rest, root, q, h, np.full(m, degrees))
        unary[near, k] += np.interp(near, picked, 1000 * np.maximum(arm_depth(turned, picked) - PENETRATION_TOLERANCE_M, 0))
    path = viterbi(unary, SWIVEL_STEP_WEIGHT * np.abs(SWIVELS_DEG[:, None] - SWIVELS_DEG[None]))
    degrees = gaussian_filter1d(SWIVELS_DEG[path].astype(float), max(.5, fps * .05), mode='nearest')
    return swivel_arm(rest, root, q, h, degrees), degrees, int(bad.sum())


def limit_wrist_rows(rest, root, q, h):
    """q with hand h's wrist-local rotations pulled into the wrist limits (an elbow swivel keeps the wrist world rotation)."""
    from .correction import fk
    _, g = fk(rest, root, q); _, _, elbow, wrist = ARMS[h]
    local = g[:, elbow].swapaxes(-1, -2) @ g[:, wrist]
    out = q.copy(); out[:, wrist] = matrix_quat(clamp_wrist(local, rest, h))
    return out


def arm_groups(h, part, side):
    """Effector segments that must stay out of the target: the hand, plus forearm and upper arm unless the target is on
    that same arm (they join it) or is the trunk (the upper arm joins it)."""
    if part in ('Hand', 'ForeArm', 'Arm', 'Shoulder') and side == h:return [SIDES[h] + '_hand']
    groups = [SIDES[h] + '_hand', SIDES[h] + '_forearm']
    return groups + ([SIDES[h] + '_upperarm'] if part in LEG_PARTS or part in ('Head', 'Neck', 'Hand', 'ForeArm') else [])


def hand_contact(skin, root, quat, rows, h, names, fingers=None, stride=1, group=None, arm=()):
    """Per frame against the named segments: (depth (m,), signed skin gap (m,), push (m, 3), arm depth (m,)) of hand h
    (or `group`); arm depth is the deepest of the `arm` segments."""
    group = group or SIDES[h] + '_hand'
    posed = skin.vertices(root, quat, rows, [group] + list(arm) + list(names), fingers)
    m = len(root); depth = np.zeros(m); gap = np.full(m, np.nan); push = np.zeros((m, 3)); arm_depth = np.zeros(m)
    for t in range(m):
        targets = {name: posed[name][t] for name in names}
        r = penetrations(posed[group][t, ::stride], targets, skin.segments, names)
        depth[t], gap[t], push[t] = r['depth'], signed_gap(r), r['push']
        for name in arm:
            arm_depth[t] = max(arm_depth[t], penetrations(posed[name][t, ::stride], targets, skin.segments, names)['depth'])
    return depth, gap, push, arm_depth


def skin_offset(skin, root, quat, rows, h, part, side, base, frame, palm):
    """Target-local goal offset on the target's skin (the skin point closest to the palm point, lifted along its normal),
    the target-local outward normal there, and the segment name."""
    name = segment_names(part, side)[0]
    posed = skin.vertices(root, quat, rows, [name])[name]
    local = np.empty((len(root), 3)); normals = np.empty((len(root), 3))
    for t in range(len(root)):
        _, q, _, normal = surface_query(palm[t][None], posed[t], skin.segments[name])
        normals[t] = frame[t].T @ unit(normal[0])
        local[t] = frame[t].T @ (q[0] + unit(normal[0]) * SKIN_STANDOFF_M - base[t])
    return np.median(local, 0), unit(np.median(normals, 0)), name


def push_step(depth, direction, weights, fps):
    """Goal moves (m, 3) along unit directions that lift penetrating frames out by depth + margin, smooth in time; ease
    frames are scaled by 1 / weight (their solved motion is only blended in)."""
    size = np.where(depth > PENETRATION_TOLERANCE_M, (depth + PUSH_MARGIN_M) / np.maximum(weights, .25), 0.)
    size = gaussian_filter1d(maximum_filter1d(size, max(1, round(fps * .15)), mode='nearest'), max(.5, fps * .05), mode='nearest')
    return direction * size[:, None]


def finger_weights(skin, root, quat, rows, h, names, ease):
    """Per-frame curl toward the grip shape that closes the fingers onto the target skin and stops at contact.

    Only frames where the fingers float and touch the skin before penetrating curl; elsewhere they keep the source pose (0).
    """
    group = SIDES[h] + '_fingers'; shape = grip_shape(skin.rest, h)
    picked = np.flatnonzero(ease > 0)[::2]
    if not len(picked):return np.zeros(len(rows)), 0
    curl = np.zeros(len(picked)); done = np.zeros(len(picked), bool); previous = 0.
    for c in CURL_STEPS:
        fingers = {h: curl_fingers(skin.fingers(rows[picked], h), shape, np.full(len(picked), c))}
        depth, gap = hand_contact(skin, root[picked], quat[picked], rows[picked], h, names, fingers, group=group)[:2]
        stop = ~done & (depth > PENETRATION_TOLERANCE_M); touch = ~done & ~stop & (gap <= FINGER_TOUCH_M)
        if c > 0:curl[stop] = previous; curl[touch] = c
        done |= stop | touch; previous = c
    out = np.interp(np.arange(len(rows)), picked, curl)
    return gaussian_filter1d(out, 1., mode='nearest') * ease, int((curl > 0).sum())


# Grip from the video ---------------------------------------------------------------------------------------------------

def angle_2d(u, v):
    """Unsigned angle (deg) between 2D vectors (..., 2)."""
    return np.rad2deg(np.abs(np.arctan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1))))


class HandVideo:
    """COCO-63 hand keypoints and per-frame cameras; correction-frame points (shared floor transform undone) to pixels."""

    def __init__(self, keypoints, rotation, translation, K, transform=None):
        self.keypoints, self.rotation, self.translation, self.K = keypoints, rotation, translation, np.asarray(K, float)
        self.transform = np.eye(4) if transform is None else np.asarray(transform, float)
        kp = keypoints; ok = (kp[:, [5, 6, 11, 12], 2] >= .5).all(1)
        torso = np.linalg.norm((kp[:, 5, :2] + kp[:, 6, :2]) / 2 - (kp[:, 11, :2] + kp[:, 12, :2]) / 2, axis=1)
        self.torso = np.where(ok, torso, np.median(torso[ok]) if ok.any() else 100.)

    def project(self, points, rows):
        """(m, p, 3) points on result rows -> (m, p, 2) pixels."""
        local = (points - self.transform[:3, 3]) @ self.transform[:3, :3]
        cam = np.einsum('tij,tpj->tpi', self.rotation[rows], local) + self.translation[rows][:, None]
        z = np.maximum(cam[..., 2], 1e-6)
        return np.stack([self.K[0, 0] * cam[..., 0] / z + self.K[0, 2], self.K[1, 1] * cam[..., 1] / z + self.K[1, 2]], -1)

    def hand(self, h, rows):
        """(wrist, middle, index, pinky knuckle pixels (m, 4, 2), confidence 0-1 (m,), knuckle-row weight (m,))."""
        base = 23 + 20 * h
        # COCO hand block: thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4; knuckle 1 is the MCP. Body wrists are 9 / 10.
        kp = self.keypoints[rows][:, [9 + h, base + 8, base + 4, base + 16]]
        d, across = kp[:, 1, :2] - kp[:, 0, :2], kp[:, 2, :2] - kp[:, 3, :2]
        length, width = np.linalg.norm(d, axis=1), np.linalg.norm(across, axis=1)
        # A hand holding a limb is half hidden: its scores sit near 0.5, so confidence ramps from 0.3 to 0.6.
        confidence = np.clip((kp[..., 2].min(1) - HAND_SCORE_LOW) / (HAND_SCORE - HAND_SCORE_LOW), 0, 1)
        confidence[length < HAND_MIN_TORSO * self.torso[rows]] = 0
        sine = np.abs(d[:, 0] * across[:, 1] - d[:, 1] * across[:, 0]) / np.maximum(length * width, 1e-9)
        # An edge-on palm projects its knuckle row short and along the fingers; its direction says little then.
        weight = np.clip(width / np.maximum(length, 1e-9) / .5, 0, 1) * np.clip((sine - .2) / .3, 0, 1)
        return kp[..., :2], confidence, weight


def load_video(project, transform=None):
    """HandVideo of an ExMR project (the cameras `review` uses), or None without its keypoints and cameras."""
    import torch
    project = Path(project)
    names = ('exmr_keypoints.pt', 'hmr4d_results.pt', 'initial_joints.npz', 'tracking.npz')
    if not all((project / name).is_file() for name in names):return None
    try:
        keypoints = np.asarray(torch.load(project / names[0], map_location='cpu', weights_only=True), float)
        record = torch.load(project / names[1], map_location='cpu', weights_only=True)
        world, incam = record['smpl_params_global'], record['smpl_params_incam']
        with np.load(project / names[2], allow_pickle=False) as data:initial = np.asarray(data['joints'], float)
        with np.load(project / names[3], allow_pickle=False) as data:K = np.asarray(data['K'], float)
    except (OSError, KeyError, ValueError, RuntimeError):
        return None
    rw = Rotation.from_rotvec(world['global_orient'].numpy()).as_matrix(); ri = Rotation.from_rotvec(incam['global_orient'].numpy()).as_matrix()
    tw, ti = world['transl'].numpy().astype(float), incam['transl'].numpy().astype(float)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (63, 3) or not len(keypoints) == len(tw) == len(initial):return None
    # As review.load_exmr: SMPL translation rotates about the rest pelvis, so both roots pivot there.
    rotation = ri @ rw.transpose(0, 2, 1); pelvis = initial[:, 0] - tw
    return HandVideo(keypoints, rotation, ti + pelvis - np.einsum('fij,fj->fi', rotation, tw + pelvis), K, transform)


def video_cost(rest, h, forearm, wrist_at, source, video, rows, local):
    """(cost, direction error) (m, k) in degrees of wrist-local candidates (m, k, 3, 3) on forearms (m, 3, 3).

    Cost: 2D angle between the model and keypoint wrist->middle-knuckle directions, plus the index->pinky knuckle row (which
    side the thumb is on) where the palm is not edge-on, scaled by the keypoint confidence, plus a pull toward the source
    wrist world rotations; frames without confident keypoints only pull. rest holds the 55 SMPL-X joints.
    """
    _, _, elbow, wrist = ARMS[h]; start = 25 + 15 * h; m = len(rows)
    knuckles = rest[[start + 3, start, start + 6]] - rest[wrist]
    kp, confidence, across = video.hand(h, rows); ok = confidence > 0
    centre = video.project(wrist_at[:, None], rows)[:, 0]
    d_kp, a_kp = kp[:, 1] - kp[:, 0], kp[:, 2] - kp[:, 3]
    world = forearm[:, None] @ local
    uv = video.project((wrist_at[:, None, None] + np.einsum('tkij,pj->tkpi', world, knuckles)).reshape(m, -1, 3), rows).reshape(m, -1, 3, 2)
    direction = angle_2d(uv[:, :, 0] - centre[:, None], d_kp[:, None])
    error = direction + VIDEO_ACROSS_WEIGHT * across[:, None] * angle_2d(uv[:, :, 1] - uv[:, :, 2], a_kp[:, None])
    prior = rotation_angle(world, np.broadcast_to(source[:, None], world.shape))
    return np.where(ok[:, None], confidence[:, None] * error + VIDEO_SOURCE_WEIGHT * prior, prior), direction


def video_swivel(rest, h, j, g, source, video, rows, fps):
    """Per-frame arm swivel (deg) about the shoulder-wrist line that lets the limited wrist match the video hand best and
    puts the projected upper arm along the video's shoulder->elbow, on frames with confident hand keypoints (else 0)."""
    _, shoulder, elbow, wrist = ARMS[h]; m = len(rows)
    _, confidence, _ = video.hand(h, rows)
    kp = video.keypoints[rows][:, [5 + h, 7 + h]]
    arm_score = np.clip((kp[:, :, 2].min(1) - HAND_SCORE_LOW) / (HAND_SCORE - HAND_SCORE_LOW), 0, 1)
    grid = wrist_grid(); local = np.broadcast_to(wrist_local(grid[:, 0], grid[:, 1], grid[:, 2], rest[:22], h), (m, len(grid), 3, 3))
    axis = unit(j[:, wrist] - j[:, shoulder]); unary = np.empty((m, len(SWIVELS_DEG)))
    for k, degrees in enumerate(SWIVELS_DEG):
        spin = Rotation.from_rotvec(axis * np.deg2rad(degrees)).as_matrix()
        hand = video_cost(rest, h, spin @ g[:, elbow], j[:, wrist], source, video, rows, local)[0].min(1)
        elbow_at = j[:, shoulder] + np.einsum('tij,tj->ti', spin, j[:, elbow] - j[:, shoulder])
        uv = video.project(np.stack([j[:, shoulder], elbow_at], 1), rows)
        upper = angle_2d(uv[:, 1] - uv[:, 0], kp[:, 1, :2] - kp[:, 0, :2])
        unary[:, k] = np.where(confidence > 0, hand + VIDEO_ELBOW_WEIGHT * arm_score * upper, 0.) + VIDEO_SWIVEL_WEIGHT * abs(degrees)
    path = viterbi(unary, VIDEO_SWIVEL_STEP_WEIGHT * np.abs(SWIVELS_DEG[:, None] - SWIVELS_DEG[None]))
    return gaussian_filter1d(SWIVELS_DEG[path].astype(float), max(.5, fps * .1), mode='nearest')


def video_wrist(rest, h, forearm, wrist_at, source, video, rows, fps):
    """Wrist world rotations (m, 3, 3) whose projected hand matches the video hand (video_cost), inside the wrist limits.

    Viterbi over the limit grid keeps it continuous, then a local refinement; frames without confident keypoints blend back
    to the source. rest holds the 55 SMPL-X joints; returns (rotations, per-frame video weight, report).
    """
    m = len(rows); ok = video.hand(h, rows)[1] > 0
    cost = lambda local: video_cost(rest, h, forearm, wrist_at, source, video, rows, local)
    grid = wrist_grid(); local = wrist_local(grid[:, 0], grid[:, 1], grid[:, 2], rest[:22], h)
    unary, _ = cost(np.broadcast_to(local, (m,) + local.shape))
    path = viterbi(unary, GRIP_STEP_WEIGHT * rotation_angle(local[:, None], local[None]))
    sigma = max(.5, fps * .05)
    angles = gaussian_filter1d(grid[path], sigma, axis=0, mode='nearest')
    spacing = np.array([np.ptp(grid[:, i]) / (GRIP_GRID_STEPS[i] - 1) for i in range(3)]) / 2
    steps = np.stack(np.meshgrid(*[(-1., 0., 1.)] * 3, indexing='ij'), -1).reshape(-1, 3)
    for scale in (1., .5, .25):
        trial = np.stack(limit_wrist(*np.moveaxis(angles[:, None] + steps * spacing * scale, -1, 0)), -1)
        angles = trial[np.arange(m), cost(wrist_local(trial[..., 0], trial[..., 1], trial[..., 2], rest[:22], h))[0].argmin(1)]
    angles = np.stack(limit_wrist(*gaussian_filter1d(angles, sigma / 2, axis=0, mode='nearest').T), -1)
    fitted = forearm @ wrist_local(angles[:, 0], angles[:, 1], angles[:, 2], rest[:22], h)
    weight = np.clip(2 * gaussian_filter1d(ok.astype(float), max(.5, fps * .1), mode='nearest'), 0, 1)
    out = quat_matrix(slerp(matrix_quat(source), matrix_quat(fitted), weight))
    before = cost((forearm.swapaxes(-1, -2) @ source)[:, None])[1][:, 0]; after = cost((forearm.swapaxes(-1, -2) @ out)[:, None])[1][:, 0]
    report = dict(confident_frames=int(ok.sum()), video_frames=int((weight > 0).sum()),
                  direction_error_source_median_deg=float(np.median(before[ok])) if ok.any() else None,
                  direction_error_median_deg=float(np.median(after[ok])) if ok.any() else None,
                  direction_error_max_deg=float(after[ok].max()) if ok.any() else None)
    return out, weight, report


def grip_video(rest, root, prior, solved, offsets, fps, lo, goals, hands, h, skin, video, config=ARM_CONFIG):
    """Turn the wrist to the video hand, re-solve the arm so the moved palm point reaches the goal, keep the wrist in limits."""
    from .correction import fk
    _, _, elbow, wrist = ARMS[h]; rows = np.arange(lo, lo + len(root))
    source = fk(rest, root, prior[lo:lo + len(root)])[1][:, wrist]
    js, gs = fk(rest, root, solved)
    swivel = video_swivel(skin.rest, h, js, gs, source, video, rows, fps)
    arm = swivel_arm(rest, root, solved, h, swivel)
    for iteration in range(1, 4):
        ja, ga = fk(rest, root, arm)
        world, weight, fit = video_wrist(skin.rest, h, ga[:, elbow], ja[:, wrist], source, video, rows, fps)
        # The planner starts from the turned arm, so it only corrects the palm point moved by the new wrist rotation.
        graspable = set_wrist_world(rest, root, arm, h, world)
        full = prior.copy(); full[lo:lo + len(root)] = graspable
        solved, report = solve_arms(rest, root, graspable, full, offsets, fps, lo, goals, config)
        solved, clamped = clamp_elbows(rest, root, solved, hands)
        _, gs = fk(rest, root, solved)
        local = gs[:, elbow].swapaxes(-1, -2) @ gs[:, wrist]
        excess = np.where(weight > 0, rotation_angle(local, clamp_wrist(local, rest, h)), 0.)
        if excess.max() < .5:break
        arm = solved
    outside = np.flatnonzero(excess > 1e-4)
    if len(outside):
        solved = solved.copy(); solved[outside, wrist] = matrix_quat(clamp_wrist(local[outside], rest, h))
    return solved, report, clamped, dict(fit, iterations=iteration, final_wrist_clamp_max_deg=float(excess.max()),
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


def solve_interval(rest, roots, quats, offsets, fps, track, interval, foot_ranges, skin=None, video=None):
    from .correction import fk
    from .contact_arm_plan import source_reach_gap
    started = time.perf_counter(); n = len(roots); a, b = interval
    h = 0 if track['effector'] == 'LeftHand' else 1
    tj, side, part = parse_target(track['target'])
    entry = dict(id=track['id'], effector=track['effector'], target=track['target'], start=int(a), end=int(b))
    j, g = fk(rest, roots[a:b], quats[a:b]); p, _, _ = palms(rest, roots[a:b], quats[a:b], offsets)
    base, frame = target_base(j, g, p, tj, part)
    offset = local_offset(track, part, base, frame, p[:, h]); entry['goal'] = 'joint offset'
    normal = unit(offset if np.linalg.norm(offset) > 1e-9 else np.array([0., 0., 1.]))
    if skin is not None and track['offset'] == 'auto' and track['distance'] == 'auto':
        offset, normal, entry['goal'] = skin_offset(skin, np.asarray(roots[a:b], float), np.asarray(quats[a:b], float), np.arange(a, b), h, part, side,
                                                    base, frame, p[:, h])
        entry['goal'] = 'skin of ' + entry['goal']
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

    def too_close(jj, gg, goal):
        """Frames whose wrist goal is nearer the shoulder than the current wrist: the elbow must bend, nothing can move closer."""
        _, shoulder, _, wrist = ARMS[h]
        wrist_goal = goal - np.einsum('tij,j->ti', gg[:, wrist], offsets[h])
        return np.linalg.norm(wrist_goal - jj[:, shoulder], axis=1) < np.linalg.norm(jj[:, wrist] - jj[:, shoulder], axis=1)

    def gap_of(q, jj, gg, pp, goal):
        return source_reach_gap(rest, jj, gg, quat_matrix(q), h, goal, pp[:, h], ARM_CONFIG['collar_limit_deg'], ARM_CONFIG['elbow_limit_deg'])

    def deficit_of(q, jj, gg, pp, goal):
        # The arm planner fades out where its own bounded collar/elbow model cannot reach, so take the larger gap.
        # A goal too close needs more elbow bend (elbow_config), never a reciprocal move toward the shoulder.
        gap = gap_of(q, jj, gg, pp, goal)
        return np.maximum(reach_deficit(rest, jj, gg, h, goal, offsets, ARM_CONFIG['collar_limit_deg']), np.where(too_close(jj, gg, goal), 0., gap))

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
    config = elbow_config(rest, jj, gg, h, arm_goals[h], offsets, too_close(jj, gg, arm_goals[h]) & (gap_of(work, jj, gg, pp, arm_goals[h]) > 0))
    if config is not ARM_CONFIG:entry['elbow_limit_deg'] = config['elbow_limit_deg']
    prior = np.asarray(quats, float).copy(); prior[lo:hi] = work
    rows = np.arange(lo, hi); names = segment_names(part, side) if skin is not None else []; arms = arm_groups(h, part, side)[1:] if names else []
    fit = track['grip'] == 'auto' and bool(names) and video is not None
    if fit:distal.remove(ARMS[h][3])
    # Penetration loop: after each solve the posed skin is checked and the goal pushed out of the target by the depth.
    push = np.zeros((hi - lo, 3)); share = .5 if len(hands) == 2 else 1.; history = []
    for attempt in range(PUSH_ITERATIONS + 1 if names else 1):
        moved = dict(arm_goals); moved[h] = arm_goals[h] + push * share
        if len(hands) == 2:moved[hands[1]] = arm_goals[hands[1]] - push * share
        solved, arm_report = solve_arms(rest, root, work, prior, offsets, fps, lo, moved, config)
        solved, clamped = clamp_elbows(rest, root, solved, hands)
        if not attempt:
            facing, _, segment, _, gs = grip_geometry(rest, root, solved, offsets, tj, arm_goals[h])
            palm_before = angle_between(gs[:, ARMS[h][3]] @ PALM_NORMAL, facing)[inside]
        grip = 'off'
        if track['grip'] == 'auto':
            grip = dict(mode='auto', wrist='source (no hand keypoints or skin)')
            if fit:
                # The fitted wrist is slerped in local space like the arm; other distal joints keep their source world frame.
                solved, arm_report, clamped, fitted = grip_video(rest, root, prior, solved, offsets, fps, lo, moved, hands, h, skin, video, config)
                grip.update(fitted, wrist='video')
        if arms:
            solved, swivel, swivel_frames = clear_arm(rest, skin, root, solved, rows, h, names, arms, fps)
            if fit:solved = limit_wrist_rows(rest, root, solved, h)
        final = blend(rest, root, start, solved, weights, sorted(set(joints)), distal)
        j0, g0 = fk(rest, root, start); js, _ = fk(rest, root, solved)
        # Slerping twisted elbow frames can straighten past both ends; never end straighter than the limit or the source.
        final, eased_clamped = clamp_elbows(rest, root, final, hands, {k: np.maximum(ELBOW_MAX_DEG, interior(*(j0[:, ARMS[k][i]] for i in (1, 2, 3))))
                                                                      for k in hands})
        if not names:break
        depth = hand_contact(skin, root, final, rows, h, names, stride=2)[0]
        history.append(float(depth.max()))
        if depth.max() <= PENETRATION_TOLERANCE_M or attempt == PUSH_ITERATIONS:break
        # Out along the contact point's skin normal, which moves with the target.
        push = push + push_step(depth, fk(rest, root, final)[1][:, tj] @ normal, weights, fps)
    changed = weights > 0
    if names:
        depth, gap, _, arm_depth = hand_contact(skin, root, final, rows, h, names, arm=arms)
        entry['skin'] = dict(segments=names, effector=[SIDES[h] + '_hand'] + arms, iterations=len(history), depth_max_by_iteration_m=history,
                             push_max_m=float(np.linalg.norm(push, axis=1).max()),
                             arm_penetration_max_m=float(arm_depth[changed].max()), arm_penetrating_frames=int((arm_depth[changed] > .005).sum()),
                             elbow_swivel_max_deg=float(np.abs(swivel).max()) if arms else 0., elbow_swivel_needed_frames=swivel_frames if arms else 0,
                             penetration_max_m=float(depth[changed].max()), penetration_interval_max_m=float(depth[inside].max()),
                             penetration_median_m=float(np.median(depth[inside])),
                             penetrating_frames=int((depth[changed] > PENETRATION_TOLERANCE_M).sum()),
                             skin_gap_median_m=float(np.nanmedian(gap[inside])) if np.isfinite(gap[inside]).any() else None,
                             skin_gap_max_m=float(np.nanmax(gap[inside])) if np.isfinite(gap[inside]).any() else None,
                             far_frames=int(np.isnan(gap[inside]).sum()))
        if track['grip'] == 'auto':
            curl, curled = finger_weights(skin, root, final, rows, h, names, weights)
            grip['fingers'] = dict(hand=h, window=[int(lo), int(hi)], weights=curl.tolist(), curled_frames=curled)
    jf, gf = fk(rest, root, final); pf, _, _ = palms(rest, root, final, offsets)
    bf, ff = target_base(jf, gf, pf, tj, part)
    pin = (arm_goals[h] if part == 'Hand' and len(hands) == 2 else bf + np.einsum('tij,j->ti', ff, offset)) + push * share
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


def apply(rest, out_roots, out_quats, offsets, fps, tracks, foot_ranges, emit=lambda *a, **k: None, skin=None, video=None):
    """Solve enabled tracks in order on the corrected pose, mutating out_quats; None when nothing is enabled.

    skin (Skin) enables skin goals and the penetration loop; video (HandVideo) lets grip 'auto' turn the wrist to the video hand.
    """
    work = [(t, r) for t in tracks if t.get('enabled', True) for r in t['ranges']]
    if not work:return None
    started = time.perf_counter(); rest = np.asarray(rest, float)[:22]; offsets = np.asarray(offsets, float)
    entries = []; changed = []; warnings = []; wrists = []; fingers = []
    for index, (track, interval) in enumerate(work):
        update, entry = solve_interval(rest, out_roots, out_quats, offsets, fps, track, interval, foot_ranges, skin, video)
        entries.append(entry)
        if update is not None:
            s, e, joints, values = update
            out_quats[s:e, joints] = values[:, joints].astype(out_quats.dtype)
            changed.append(dict(joints=[int(x) for x in joints], start=int(s), end=int(e)))
            if isinstance(entry['grip'], dict) and entry['grip'].get('wrist') == 'video':
                wrists.append(dict(joint=ARMS[0 if track['effector'] == 'LeftHand' else 1][3], start=int(s), end=int(e)))
            if isinstance(entry['grip'], dict) and 'fingers' in entry['grip']:
                fingers.append(entry['grip']['fingers'])
            if entry['residual_max_m'] > .03:
                warnings.append('%s -> %s %d-%d: palm stays up to %.1f cm from its target.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, 100 * entry['residual_max_m']))
            if entry.get('skin', {}).get('penetration_interval_max_m', 0) > .01:
                warnings.append('%s -> %s %d-%d: the hand stays up to %.1f cm inside the %s.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, 100 * entry['skin']['penetration_interval_max_m'],
                    entry['skin']['segments'][0].replace('_', ' ')))
            if entry.get('reciprocal', {}).get('foot_lock_conflict_frames'):
                warnings.append('%s -> %s %d-%d: a foot lock kept the target leg still on %d frames.' % (
                    track['effector'], track['target'], interval[0], interval[1] - 1, entry['reciprocal']['foot_lock_conflict_frames']))
        emit('progress', stage='correct', current=index + 1, total=len(work), phase='self_contact', message='Motion Correction: solving self-contacts')
    # wrist_changed declares the only rows whose wrist world rotation may differ from the source (grip).
    return dict(method=METHOD, tracks=entries, changed=changed, seconds=time.perf_counter() - started,
                wrist_changed=wrists, fingers=fingers, warning=' '.join(['Self-contact: ' + w for w in warnings]))


def from_store(project, base, context, rest, out_roots, out_quats, offsets, fps, segments, emit=lambda *a, **k: None, body=None, transform=None):
    """correction.run hook: (report or None, pre-self-contact quats or None). No store file = (None, None).

    body is the correction's SMPL-X model (loaded here when None); transform maps the result frame to the correction frame.
    """
    from .self_contact_store import read
    tracks, revision, warning = read(project, base, context)
    if not tracks and not warning:return None, None
    foot_ranges = [[(s['start'], s['end']) for s in segments if s['foot'] == f] for f in (0, 1)]
    before = np.array(out_quats, copy=True)
    enabled = [t for t in tracks if t.get('enabled', True) and t['ranges']]
    skin, note = load_skin(project, base, rest, body) if enabled else (None, None)
    video = load_video(project, transform) if skin is not None and any(t['grip'] == 'auto' for t in enabled) else None
    report = apply(rest, out_roots, out_quats, offsets, fps, tracks, foot_ranges, emit, skin, video) or dict(method=METHOD, tracks=[], changed=[], warning='')
    report['revision'] = revision
    report['skin'] = dict(available=skin is not None, video=video is not None, note=note)
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
    weights = np.asarray(spec['weights'], float) if 'weights' in spec else ease_weights(lo, hi, *spec['contact'], spec['ease']) * spec['curl']
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
        ids, weights = finger_rows(spec)
        if not len(ids):
            note_fingers(spec, False, note='the fingers never had to close onto the skin; fingers unchanged')
            continue
        if field not in replaced:
            replaced[field] = params[field]; output[field] = params[field].clone()
        ids = torch.from_numpy(ids)
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
        ids, weights = finger_rows(spec)
        if not len(ids):
            note_fingers(spec, False, note='the fingers never had to close onto the skin; fingers unchanged')
            continue
        ids = torch.from_numpy(ids); joints = slice(22 + 15 * h, 37 + 15 * h)
        current = rot6d_matrix(rot6d[0, ids, joints].double().numpy())
        curled = curl_fingers(current, grip_shape(rest, h), weights)
        rot6d[0, ids, joints] = torch.from_numpy(curled[..., :2].reshape(len(ids), 15, 6).astype(np.float32))
        note_fingers(spec, True, current, curled)
