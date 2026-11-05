"""Anatomy checks for `review`: knee/elbow extension, wrist limits, joint twist, sudden folds, finger pose and hand orientation.

Issues use review's format (check, severity, score, frames inclusive, worst_frame, joints, metrics, message, _draw).
Rotations are rest-relative: `world` maps the rest pose to the frame, `local` is parent-relative in rest-aligned axes
(SMPL-X locals, or for an FBX the bone rotation relative to its bind, so export surface roll is measured too).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .smplx_body import FINGER_OFFSETS, FINGERS, HAND_START

BLENDER_SCRIPT = Path(__file__).resolve().with_name('blender_pose.py')
SIDES = ('left', 'right')
WRISTS, ELBOWS, SHOULDERS = (20, 21), (18, 19), (16, 17)

# Thresholds (degrees unless noted). Interior angle: 180 = straight. Bend: signed flexion about the HINGES axis.
HYPEREXTENSION_DEG = 5.0
LOCK_DEG = {'elbow': 170.0, 'knee': 178.0}
LOCK_HIGH_DEG = {'elbow': 177.0, 'knee': 179.5}
LOCK_FRAMES = 6
# Wrist range relative to the forearm (swing split into palm flexion / extension and radial / ulnar deviation).
WRIST_LIMITS = dict(flexion=80.0, extension=70.0, radial=25.0, ulnar=40.0)
# Twist about the bone's own rest axis; LBS skin collapses ("candy-wrap") well before 90 deg on a single joint. Shoulder and
# hip: upper-arm / thigh twist (the joint's local rotation about its rest bone), the humerus and femur anatomical roll range.
TWIST_LIMITS = {'knee': 40.0, 'elbow': 80.0, 'wrist': 80.0, 'shoulder': 90.0, 'hip': 60.0}
TWIST_HINTS = {'shoulder': 'the upper arm is rolled about its own bone: re-run `correct` (the self-contact arm keeps its twist in range)',
               'hip': 'the thigh is rolled about its own bone: check the knee direction against the video'}
FOLD_DEG = 35.0                 # dip below both neighbouring maxima inside the window
FOLD_SECONDS = 0.35
# Joint flip: an elbow / knee swinging round its root-end line (swivel, measured while bent below FLIP_BENT_DEG interior),
# the end twisting about the lower bone, or the upper / lower bone turning against the torso faster than FLIP_STEP_DEG per
# 1/30 s, or swivel / twist netting FLIP_WINDOW_DEG within FLIP_WINDOW_SECONDS. Fast real punches and dance swings stay
# under ~30 deg per frame and ~80 deg of swivel or twist per window; a branch flip of the IK sweeps 140-180.
FLIP_STEP_DEG = 45.0
FLIP_WINDOW_DEG = 120.0
FLIP_WINDOW_SECONDS = 5 / 30
FLIP_BENT_DEG = 150.0
# Finger limits relative to the flat hand: (flexion min, flexion max, |abduction|, |twist|) for knuckles 1-3.
FINGER_LIMITS = ((-30.0, 100.0, 30.0, 30.0), (-10.0, 115.0, 15.0, 25.0), (-15.0, 95.0, 15.0, 25.0))
THUMB_LIMITS = (100.0, 45.0)    # swing, |twist|
FINGER_MARGIN = 10.0            # degrees past a limit before a frame counts
HAND_VISIBLE = 0.5
HAND_DIRECTION_DEG = 50.0
HAND_EDGE_ON = 0.35             # |sin| of the wrist->knuckle vs knuckle-row angle below which the palm order is unreliable
HAND_MIN_TORSO = 0.06
PREFIX = dict(hyperextension='HX', locked_limb='LL', wrist_limit='WL', joint_twist='TW', joint_flip='JF', sudden_fold='SF', finger_pose='FG',
              hand_orientation='HO', self_penetration='SP', contact_gap='CG')
# Skin interpenetration (posed SMPL-X mesh, segments by skinning weight): depth of one segment's vertices inside another.
PENETRATION_M = 0.01
PENETRATION_HIGH_M = 0.025
# A declared self-contact whose hand skin stays this far from the target skin floats.
CONTACT_GAP_M = 0.015
CONTACT_GAP_HIGH_M = 0.04
SEGMENT_JOINT = dict(hand=(20, 21), forearm=(18, 19), upperarm=(16, 17), thigh=(1, 2), shin=(4, 5), foot=(7, 8))
REVIEW_PREFIX = dict(missed_contact='MC', pose_error='PE', foot_slide='FS', floor_penetration='FP', jitter='JT', placement_drift='PD', body_offset='BO')
SEVERITY = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}


@dataclass
class Pose:
    """Per-frame rotations and positions; row i is video frame first + i."""
    joints: np.ndarray            # (N, 22, 3) world Y-up metres, SMPL-X body order
    world: np.ndarray             # (N, 22, 3, 3) rest -> frame rotation
    local: np.ndarray             # (N, 22, 3, 3) parent-relative, rest-aligned axes
    rest: np.ndarray              # (22, 3) rest joint positions
    knuckles: np.ndarray | None   # (N, 2, 5, 3, 3) finger joints [side][thumb..pinky][1..3] world positions
    fingers: np.ndarray | None    # (N, 2, 5, 3, 3, 3) finger locals relative to the flat hand (SMPL-X only)
    finger_rest: np.ndarray | None  # (2, 5, 4, 3) rest finger joints 1-3 plus the tip direction end
    first: int
    source: str
    mesh: object = None           # Mesh of the same rows (SMPL-X results only)

    def rows(self, frames):
        index = np.asarray(frames) - self.first
        if index.min() < 0 or index.max() >= len(self.joints):
            raise ValueError(f'pose covers video frames {self.first}-{self.first + len(self.joints) - 1}, not {frames[0]}-{frames[-1]}')
        pick = lambda value: None if value is None else value[index]
        return Pose(self.joints[index], self.world[index], self.local[index], self.rest, pick(self.knuckles), pick(self.fingers),
                    self.finger_rest, int(frames[0]), self.source, None if self.mesh is None else self.mesh.rows(index))


class Mesh:
    """Posed SMPL-X vertices of a Body state on demand (rows index the state), with skin-contact measurements cached."""

    def __init__(self, body, state, index=None, project=None, kind=None, cache=None):
        from ..self_contact import Segments, segment_vertices
        self.body, self.state, self.project, self.kind = body, state, project, kind
        self.index = np.arange(len(state['joints'])) if index is None else np.asarray(index)
        self.groups = segment_vertices(body.weights)
        self.segments = Segments(body.faces, self.groups)
        self.cache = {} if cache is None else cache

    def rows(self, index):
        out = Mesh.__new__(Mesh)
        out.__dict__.update(self.__dict__, index=self.index[np.asarray(index)])
        return out

    def vertices(self, i):
        return self.body.vertices(self.state, int(self.index[i]))

    def measure(self, pairs):
        """{pair: (depth (n,), inside count (n,), signed skin gap (n,))} for (effector segment, target segment names) pairs."""
        from ..self_contact import penetrations, signed_gap
        key = (tuple(self.index.tolist()), tuple(pairs))
        if key in self.cache:return self.cache[key]
        n = len(self.index)
        out = {pair: (np.zeros(n), np.zeros(n, int), np.full(n, np.nan)) for pair in pairs}
        for i in range(n):
            v = self.vertices(i)
            boxes = {}
            for effector, targets in pairs:
                for name in (effector,) + tuple(targets):
                    if name not in boxes:
                        part = v[self.groups[name]]; boxes[name] = (part, part.min(0) - .05, part.max(0) + .05)
                points, low, high = boxes[effector]
                near = [t for t in targets if (boxes[t][2] >= low).all() and (boxes[t][1] <= high).all()]
                if not near:continue
                r = penetrations(points, {t: boxes[t][0] for t in near}, self.segments, list(near))
                depth, inside, gap = out[(effector, targets)]
                depth[i], inside[i] = r['depth'], r['inside']
                gap[i] = signed_gap(r)
        self.cache[key] = out
        return out


# Maths ---------------------------------------------------------------------------------------------------------------

def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def orthogonal(v, axis):
    """Unit component of v perpendicular to unit axis (both (..., 3))."""
    v, axis = np.broadcast_arrays(np.asarray(v, np.float64), np.asarray(axis, np.float64))
    return unit(v - axis * np.sum(v * axis, -1, keepdims=True))


def quaternions(matrices):
    """(..., 3, 3) -> (..., 4) wxyz with w >= 0."""
    from scipy.spatial.transform import Rotation
    m = np.asarray(matrices, np.float64)
    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(m.shape[:-2] + (4,))


def swing_twist(matrices, axis):
    """Swing-twist split about a unit axis: (swing rotation vector (..., 3) perpendicular to it, signed twist in degrees)."""
    q = quaternions(matrices)
    axis = unit(np.broadcast_to(axis, q.shape[:-1] + (3,)))
    w, v = q[..., 0], q[..., 1:]
    along = np.sum(v * axis, -1)
    twist = np.degrees(2 * np.arctan2(along, w))
    twist = (twist + 180.0) % 360.0 - 180.0
    norm = np.sqrt(w * w + along * along)
    tw = np.where(norm[..., None] > 1e-12, np.concatenate([w[..., None], along[..., None] * axis], -1) / np.maximum(norm, 1e-12)[..., None],
                  np.array([1.0, 0, 0, 0]))
    # swing = q * conj(twist)
    a0, av = q[..., 0], q[..., 1:]
    b0, bv = tw[..., 0], -tw[..., 1:]
    s0 = a0 * b0 - np.sum(av * bv, -1)
    sv = a0[..., None] * bv + b0[..., None] * av + np.cross(av, bv)
    sv *= np.where(s0 < 0, -1.0, 1.0)[..., None]
    s0 = np.abs(s0)
    length = np.linalg.norm(sv, axis=-1)
    angle = 2 * np.arctan2(length, s0)
    swing = sv / np.maximum(length, 1e-12)[..., None] * angle[..., None]
    return swing, twist


def interior(a, b, c):
    """Interior angle at b in degrees (180 = straight)."""
    u, v = unit(np.asarray(a) - b), unit(np.asarray(c) - b)
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1.0, 1.0)))


def hinge_bend(joints, world, upper, middle, end, axis):
    """(bend, interior, off-plane) degrees. Bend is signed flexion about the limb's anatomical hinge, as in
    `mixamo_rig.HINGES` / `flow_export.validate_axes`: 0 straight, positive flexed, negative hyperextended."""
    u = unit(joints[:, middle] - joints[:, upper])
    v = unit(joints[:, end] - joints[:, middle])
    hinge = orthogonal(np.einsum('nij,j->ni', world[:, upper], np.asarray(axis, np.float64)), u)
    cross = np.cross(u, v)
    bend = np.degrees(np.arctan2(np.sum(cross * hinge, -1), np.sum(u * v, -1)))
    off = np.degrees(np.arcsin(np.clip(np.sum(v * hinge, -1), -1.0, 1.0)))
    return bend, interior(joints[:, upper], joints[:, middle], joints[:, end]), off


def wrist_frame(rest, side):
    """Rest forearm axis, palm normal and thumb direction (T-pose: palms down, thumbs forward)."""
    s = SIDES.index(side)
    axis = unit(rest[WRISTS[s]] - rest[ELBOWS[s]])
    palm = orthogonal(np.array([0.0, -1.0, 0.0]), axis)
    thumb = unit(np.cross(axis, palm)) * (-1.0 if s == 0 else 1.0)
    return axis, palm, thumb


def wrist_angles(local, rest, side):
    """Wrist local rotation vs the forearm: flexion (+ palm-ward), radial deviation (+ thumb-ward), twist, swing (degrees)."""
    axis, palm, thumb = wrist_frame(rest, side)
    swing, twist = swing_twist(local, axis)
    flex = np.degrees(swing @ unit(np.cross(axis, palm)))
    radial = np.degrees(swing @ unit(np.cross(axis, thumb)))
    return flex, radial, twist, np.degrees(np.linalg.norm(swing, axis=-1))


def wrist_excess(flex, radial):
    """Elliptic limit ratio (> 1 outside the WRIST_LIMITS range) and the degrees past it along the swing direction."""
    f = np.where(flex >= 0, WRIST_LIMITS['flexion'], WRIST_LIMITS['extension'])
    d = np.where(radial >= 0, WRIST_LIMITS['radial'], WRIST_LIMITS['ulnar'])
    ratio = np.sqrt((flex / f) ** 2 + (radial / d) ** 2)
    swing = np.hypot(flex, radial)
    return ratio, np.where(ratio > 1, swing - swing / np.maximum(ratio, 1e-9), 0.0)


def fold_depth(values, window):
    """Per frame: how far the value dips below the smaller of the maxima just before and just after it."""
    values = np.asarray(values, np.float64)
    n = len(values)
    depth = np.zeros(n)
    for t in range(n):
        before = values[max(0, t - window):t + 1].max()
        after = values[t:min(n, t + window + 1)].max()
        depth[t] = min(before, after) - values[t]
    return depth


def finger_angles(local, rest_axis, palm):
    """Finger knuckle local rotation -> (flexion, abduction, twist) degrees; flexion positive toward the palm."""
    swing, twist = swing_twist(local, rest_axis)
    flex_axis = unit(np.cross(rest_axis, palm))
    side_axis = unit(np.cross(rest_axis, flex_axis))
    return np.degrees(np.sum(swing * flex_axis, -1)), np.degrees(np.sum(swing * side_axis, -1)), twist


def runs(mask, gap=2, minimum=3):
    """Inclusive (start, end) runs of True, bridging gaps of up to `gap` frames, with at least `minimum` True frames."""
    spans = []
    for i in np.flatnonzero(mask):
        if spans and i - spans[-1][1] <= gap + 1:
            spans[-1][1] = int(i)
        else:
            spans.append([int(i), int(i)])
    return [(a, b) for a, b in spans if int(np.count_nonzero(mask[a:b + 1])) >= minimum]


def limbs():
    from ..mixamo_rig import HINGES
    out = []
    for upper, middle, end, axis in HINGES:
        kind = 'knee' if upper < 3 else 'elbow'
        side = 'left' if upper in (1, 16) else 'right'
        out.append((f'{side}_{kind}', kind, upper, middle, end, axis))
    return out


# Issues -------------------------------------------------------------------------------------------------------------

def make_issue(check, severity, score, frames, a, b, worst, joints, metrics, message, hint, draw):
    first, last = int(frames[a]), int(frames[b])
    return dict(check=check, severity=severity, score=float(score), frames=[first, last], blender_frames=[first + 1, last + 1],
                length=last - first + 1, worst_frame=int(frames[worst]), joints=list(joints), metrics=metrics, message=message, fix=None,
                hint=hint, _draw=draw)


def check_limbs(pose, frames, fps):
    issues = []
    window = max(int(round(FOLD_SECONDS * fps)), 3)
    for name, kind, upper, middle, end, axis in limbs():
        bend, angle, off = hinge_bend(pose.joints, pose.world, upper, middle, end, axis)
        label = name.replace('_', ' ')
        draw = dict(smpl=(upper, middle, end), series={f'{label} interior deg': angle, f'{label} bend deg': bend})
        hyper = bend < -HYPEREXTENSION_DEG
        for a, b in runs(hyper):
            span = slice(a, b + 1)
            worst = a + int(np.argmin(bend[span]))
            peak, count = float(-bend[worst]), int(hyper[span].sum())
            severity = 'high' if peak >= 25 and count >= 5 else 'medium' if peak >= 12 else 'low'
            issues.append(make_issue('hyperextension', severity, count * peak, frames, a, b, worst, [name],
                                     dict(frames_flagged=count, hyperextension_max_deg=peak, interior_at_worst_deg=float(angle[worst]),
                                          off_plane_at_worst_deg=float(off[worst])),
                                     f'{label} bends {peak:.0f} deg the wrong way (hyperextended)', 'reduce the reach or move the target limb', draw))
        straight = angle >= LOCK_DEG[kind]
        for a, b in runs(straight, 1, LOCK_FRAMES):
            span = slice(a, b + 1)
            count = int(straight[span].sum())
            median = float(np.median(angle[span][straight[span]]))
            worst = a + int(np.argmax(angle[span]))
            spread = float(np.ptp(angle[span]))
            locked = median >= LOCK_HIGH_DEG[kind]
            if kind == 'elbow':
                severity = 'high' if locked and count >= 10 else 'medium' if locked else 'low'
            else:
                severity = 'medium' if locked and count >= 15 else 'low'
            issues.append(make_issue('locked_limb', severity, count * (median - LOCK_DEG[kind] + 1), frames, a, b, worst, [name],
                                     dict(frames_flagged=count, interior_median_deg=median, interior_max_deg=float(angle[worst]),
                                          interior_range_deg=spread),
                                     f'{label} held straight (interior {median:.0f} deg median, max {angle[worst]:.0f}) for {count} frames'
                                     + (' - looks locked by IK reach' if locked else ''),
                                     'arm out of reach: move the target limb (reciprocal) or let the shoulder help', draw))
        depth = fold_depth(angle, window)
        fold = depth > FOLD_DEG
        for a, b in runs(fold, 1, 1):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(depth[span]))
            peak = float(depth[worst])
            near = slice(max(0, a - window), min(len(angle), b + window + 1))
            speed = float(np.abs(np.gradient(angle[near])).max() * fps) if len(angle[near]) > 1 else 0.0
            # Fast real knee kicks exist (dance); an elbow folding and reopening inside the window is an easing artifact.
            severity = ('high' if peak >= 60 else 'medium' if peak >= 45 else 'low') if kind == 'elbow' else ('medium' if peak >= 60 else 'low')
            issues.append(make_issue('sudden_fold', severity, peak, frames, a, b, worst, [name],
                                     dict(fold_depth_deg=peak, interior_min_deg=float(angle[worst]), window_frames=window,
                                          peak_speed_deg_s=speed),
                                     f'{label} folds to {angle[worst]:.0f} deg interior and opens again ({peak:.0f} deg dip within '
                                     f'{window} frames)', 'ease in rotation space, not by moving the IK goal', draw))
    return issues


def joint_twists(pose):
    """{name: (kind, joint, signed twist degrees)} for hips, knees, shoulders, elbows (about their child segment) and wrists
    (about the forearm)."""
    out = {}
    for side, (hip, knee, ankle), (shoulder, elbow, wrist) in zip(SIDES, ((1, 4, 7), (2, 5, 8)), ((16, 18, 20), (17, 19, 21))):
        out[f'{side}_hip'] = ('hip', hip, swing_twist(pose.local[:, hip], unit(pose.rest[knee] - pose.rest[hip]))[1])
        out[f'{side}_knee'] = ('knee', knee, swing_twist(pose.local[:, knee], unit(pose.rest[ankle] - pose.rest[knee]))[1])
        out[f'{side}_shoulder'] = ('shoulder', shoulder, swing_twist(pose.local[:, shoulder], unit(pose.rest[elbow] - pose.rest[shoulder]))[1])
        forearm = unit(pose.rest[wrist] - pose.rest[elbow])
        out[f'{side}_elbow'] = ('elbow', elbow, swing_twist(pose.local[:, elbow], forearm)[1])
        out[f'{side}_wrist'] = ('wrist', wrist, swing_twist(pose.local[:, wrist], forearm)[1])
    return out


def check_twist(pose, frames):
    issues = []
    for name, (kind, joint, twist) in joint_twists(pose).items():
        limit = TWIST_LIMITS[kind]
        flag = np.abs(twist) > limit
        for a, b in runs(flag):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(np.abs(twist[span])))
            peak, count = float(abs(twist[worst])), int(flag[span].sum())
            severity = 'high' if peak >= limit + 30 and count >= 5 else 'medium' if peak >= limit + 15 or count >= 10 else 'low'
            issues.append(make_issue('joint_twist', severity, count * (peak - limit), frames, a, b, worst, [name],
                                     dict(frames_flagged=count, twist_max_deg=peak, twist_signed_at_worst_deg=float(twist[worst]), limit_deg=limit),
                                     f'{name.replace("_", " ")} twists {peak:.0f} deg about its bone (limit {limit:.0f}): skin may candy-wrap',
                                     TWIST_HINTS.get(kind, 'check the limb roll; on an FBX, export surface roll adds forearm twist'), dict(smpl=(joint,),
                                     series={f'{name} twist deg': twist})))
    return issues


def limb_steps(pose):
    """{limb: (joints, {measure: signed or unsigned degrees per frame step (N - 1,)})} for both arms and legs: the upper and
    lower bone's rotation against the torso (spine3 for arms, pelvis for legs), the elbow / knee swivel about the root-end
    line in the torso frame (0 while the limb is nearly straight) and the end's twist against the upper bone about the lower
    bone. Swivel and twist are per-step increments, so they need no reference direction and never wrap."""
    from scipy.spatial.transform import Rotation
    out = {}
    for side, arm, leg in zip(SIDES, ((16, 18, 20), (17, 19, 21)), ((1, 4, 7), (2, 5, 8))):
        for kind, (a, b, c), base in (('arm', arm, 9), ('leg', leg, 0)):
            torso = pose.world[:, base].transpose(0, 2, 1)
            steps = {}
            for name, joint in (('upper', a), ('lower', b)):
                rel = torso @ pose.world[:, joint]
                cosine = (np.einsum('nij,nij->n', rel[:-1], rel[1:]) - 1) / 2
                steps[name] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            p, e, w = (np.einsum('nij,nj->ni', torso, pose.joints[:, k]) for k in (a, b, c))
            axis = unit(w - p)
            perp = orthogonal(e - p, axis)
            # Carry each frame's elbow direction onto the next frame's axis by the minimal rotation, then measure the turn about it.
            cross = np.cross(axis[:-1], axis[1:])
            turn = unit(cross) * np.arctan2(np.linalg.norm(cross, axis=-1), np.sum(axis[:-1] * axis[1:], -1))[:, None]
            carried = Rotation.from_rotvec(turn).apply(perp[:-1])
            swivel = np.degrees(np.arctan2(np.sum(np.cross(carried, perp[1:]) * axis[1:], -1), np.sum(carried * perp[1:], -1)))
            bent = interior(pose.joints[:, a], pose.joints[:, b], pose.joints[:, c]) < FLIP_BENT_DEG
            steps['swivel'] = np.where(bent[1:] & bent[:-1], swivel, 0.0)
            # End vs upper bone angular step (upper frame), read in the lower bone's frame: elbow / knee flex and wrist / ankle
            # swing are about axes across the lower bone, so only the twist lands on its rest axis.
            upper = pose.world[:, a].transpose(0, 2, 1)
            rel = upper @ pose.world[:, c]
            step = Rotation.from_matrix(rel[1:] @ rel[:-1].transpose(0, 2, 1)).as_rotvec()
            lower = (upper @ pose.world[:, b])[:-1]
            steps['twist'] = np.degrees(np.einsum('nji,nj->ni', lower, step) @ unit(pose.rest[c] - pose.rest[b]))
            out[f'{side}_{kind}'] = ((a, b, c), steps)
    return out


def check_joint_flip(pose, frames, fps):
    """Arm / leg rotations that flip: per-step speed spikes and swivel / twist sweeps (FLIP_* thresholds, scaled to fps)."""
    if len(pose.joints) < 3:
        return []
    step_limit = FLIP_STEP_DEG * 30.0 / fps
    span = max(2, int(round(FLIP_WINDOW_SECONDS * fps)))
    issues = []
    for limb, (joints, steps) in limb_steps(pose).items():
        n = len(pose.joints)
        flag = np.zeros(n, bool)
        per_step = np.zeros(n)
        windowed = np.zeros(n)
        for name, values in steps.items():
            speed = np.abs(values)
            flag[1:] |= speed > step_limit
            per_step[1:] = np.maximum(per_step[1:], speed / step_limit)
            if name in ('swivel', 'twist'):
                total = np.r_[0.0, np.cumsum(values)]
                net = np.abs(total[span:] - total[:-span])
                for i in np.flatnonzero(net > FLIP_WINDOW_DEG):
                    flag[i + 1:i + span + 1] = True
                for i, value in enumerate(net):
                    windowed[i + 1:i + span + 1] = np.maximum(windowed[i + 1:i + span + 1], value / FLIP_WINDOW_DEG)
        score = np.maximum(per_step, windowed)
        for a, b in runs(flag, 2, 1):
            span_rows = slice(a, b + 1)
            worst = a + int(np.argmax(score[span_rows]))
            lo, hi = max(a - span, 1), min(b + span, n - 1)
            metrics = dict(frames_flagged=int(flag[span_rows].sum()), window_frames=span, step_limit_deg=step_limit, window_limit_deg=FLIP_WINDOW_DEG)
            for name, values in steps.items():
                metrics[f'{name}_step_max_deg'] = float(np.abs(values[lo - 1:hi]).max())
                if name in ('swivel', 'twist'):
                    total = np.r_[0.0, np.cumsum(values[lo - 1:hi])]
                    metrics[f'{name}_window_max_deg'] = float(np.abs(total[span:] - total[:-span]).max()) if len(total) > span else 0.0
            peak = max(metrics['swivel_window_max_deg'], metrics['twist_window_max_deg'])
            fastest = max(metrics[f'{name}_step_max_deg'] for name in steps)
            severity = 'high' if peak >= 150 or fastest >= 1.5 * step_limit else 'medium'
            side, kind = limb.split('_')
            middle = 'elbow' if kind == 'arm' else 'knee'
            seconds = span / fps
            issues.append(make_issue('joint_flip', severity, float(score[worst]) * 100, frames, a, b, worst, [f'{side}_{middle}'], metrics,
                                     f'{side} {kind} flips: the {middle} swings {metrics["swivel_window_max_deg"]:.0f} deg round its line and '
                                     f'the {"hand" if kind == "arm" else "foot"} twists {metrics["twist_window_max_deg"]:.0f} deg within '
                                     f'{seconds:.2f} s (fastest bone step {fastest:.0f} deg per frame)',
                                     'an IK / contact solve changed elbow or knee branch; keep the branch continuous (re-run `correct`)',
                                     dict(smpl=joints, series={f'{limb} swivel step deg': np.r_[0.0, steps['swivel']],
                                                               f'{limb} twist step deg': np.r_[0.0, steps['twist']]})))
    return issues


def check_wrists(pose, frames):
    issues = []
    for s, side in enumerate(SIDES):
        flex, radial, twist, swing = wrist_angles(pose.local[:, WRISTS[s]], pose.rest, side)
        ratio, excess = wrist_excess(flex, radial)
        flag = ratio > 1
        for a, b in runs(flag):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(excess[span]))
            peak, count = float(excess[worst]), int(flag[span].sum())
            severity = 'high' if peak >= 15 and count >= 10 else 'medium' if peak >= 8 or count >= 10 else 'low'
            direction = ('flexed' if flex[worst] >= 0 else 'extended') if abs(flex[worst]) >= abs(radial[worst]) else \
                ('radially deviated' if radial[worst] >= 0 else 'ulnarly deviated')
            issues.append(make_issue('wrist_limit', severity, count * peak, frames, a, b, worst, [f'{side}_wrist'],
                                     dict(frames_flagged=count, swing_max_deg=float(swing[span].max()), swing_at_worst_deg=float(swing[worst]),
                                          flexion_at_worst_deg=float(flex[worst]), radial_at_worst_deg=float(radial[worst]),
                                          twist_at_worst_deg=float(twist[worst]), past_limit_deg=peak, limits_deg=WRIST_LIMITS),
                                     f'{side} wrist bent {swing[worst]:.0f} deg vs the forearm ({direction}, {peak:.0f} deg past its range)',
                                     'turn the palm with the arm (grip) instead of keeping the source hand rotation',
                                     dict(smpl=(ELBOWS[s], WRISTS[s]), series={f'{side} wrist swing deg': swing})))
    return issues


def finger_excess(pose):
    """(N, 2, 5, 3) degrees past the finger limits, or None without flat-relative finger rotations."""
    if pose.fingers is None or pose.finger_rest is None:
        return None
    n = len(pose.fingers)
    out = np.zeros((n, 2, 5, 3))
    for s, side in enumerate(SIDES):
        _, palm, _ = wrist_frame(pose.rest, side)
        for f, finger in enumerate(FINGERS):
            for k in range(3):
                axis = unit(pose.finger_rest[s, f, k + 1] - pose.finger_rest[s, f, k])
                local = pose.fingers[:, s, f, k]
                if finger == 'thumb':
                    swing, twist = swing_twist(local, axis)
                    over = np.maximum(np.degrees(np.linalg.norm(swing, axis=-1)) - THUMB_LIMITS[0], np.abs(twist) - THUMB_LIMITS[1])
                else:
                    low, high, side_limit, twist_limit = FINGER_LIMITS[k]
                    flex, abduction, twist = finger_angles(local, axis, orthogonal(palm, axis))
                    over = np.max(np.stack([low - flex, flex - high, np.abs(abduction) - side_limit, np.abs(twist) - twist_limit]), 0)
                out[:, s, f, k] = np.maximum(over, 0.0)
    return out


def check_fingers(pose, frames):
    excess = finger_excess(pose)
    if excess is None:
        return []
    issues = []
    for s, side in enumerate(SIDES):
        worst_per_frame = excess[:, s].reshape(len(excess), -1).max(1)
        flag = worst_per_frame > FINGER_MARGIN
        for a, b in runs(flag):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(worst_per_frame[span]))
            peak, count = float(worst_per_frame[worst]), int(flag[span].sum())
            bad = sorted({f'{side}_{FINGERS[f]}{k + 1}' for f in range(5) for k in range(3) if excess[span, s, f, k].max() > FINGER_MARGIN})
            severity = 'medium' if peak >= 25 and count >= 5 else 'low'
            issues.append(make_issue('finger_pose', severity, count * peak, frames, a, b, worst, bad,
                                     dict(frames_flagged=count, past_limit_max_deg=peak, joints_flagged=len(bad)),
                                     f'{side} hand fingers past their range by up to {peak:.0f} deg ({", ".join(bad[:4])}'
                                     + (' ...' if len(bad) > 4 else '') + ')', 'finger pose from the model; check the hand keypoints',
                                     dict(smpl=(WRISTS[s],), series={f'{side} finger excess deg': worst_per_frame})))
    return issues


def check_anatomy(pose, frames, fps):
    """All rotation checks on a pose sliced to `frames` (video frame per row)."""
    frames = np.asarray(frames)
    return (check_limbs(pose, frames, fps) + check_wrists(pose, frames) + check_twist(pose, frames) + check_joint_flip(pose, frames, fps) +
            check_fingers(pose, frames))


# Hand orientation vs the video ---------------------------------------------------------------------------------------

def hand_series(ctx, pose):
    """Per side: 2D direction error (deg), keypoint and model palm order, and the confident-frame mask."""
    if pose.knuckles is None:
        return {}
    kp = ctx.keypoints
    out = {}
    for s, side in enumerate(SIDES):
        base = 23 if s == 0 else 43
        wrist_c = 9 if s == 0 else 10
        # COCO hand block: thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4; knuckle 1 of index/middle/pinky is the MCP.
        coco = [wrist_c, base + 8, base + 4, base + 16]
        points = np.stack([pose.joints[:, WRISTS[s]], pose.knuckles[:, s, 2, 0], pose.knuckles[:, s, 1, 0], pose.knuckles[:, s, 4, 0]], 1)
        uv = ctx.project(points)
        k2 = kp[:, coco, :2]
        score = kp[:, coco, 2]
        d_kp, d_md = k2[:, 1] - k2[:, 0], uv[:, 1] - uv[:, 0]
        across_kp, across_md = k2[:, 2] - k2[:, 3], uv[:, 2] - uv[:, 3]
        cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        norm = lambda a: np.maximum(np.linalg.norm(a, axis=1), 1e-9)
        error = np.degrees(np.abs(np.arctan2(cross(d_md, d_kp), np.sum(d_md * d_kp, 1))))
        order_kp = cross(d_kp, across_kp) / (norm(d_kp) * norm(across_kp))
        order_md = cross(d_md, across_md) / (norm(d_md) * norm(across_md))
        hand_score = kp[:, base:base + 20, 2].mean(1)
        ok = ((score >= HAND_VISIBLE).all(1) & (hand_score >= HAND_VISIBLE) & np.isfinite(uv).all((1, 2)) &
              (norm(d_kp) >= HAND_MIN_TORSO * ctx.torso) & (norm(d_md) >= HAND_MIN_TORSO * ctx.torso))
        flipped = ok & (np.abs(order_kp) > HAND_EDGE_ON) & (np.abs(order_md) > HAND_EDGE_ON) & (np.sign(order_kp) != np.sign(order_md))
        out[side] = dict(error=np.where(ok, error, np.nan), order_kp=order_kp, order_model=order_md, ok=ok, flipped=flipped)
    return out


def check_hand_orientation(ctx, pose):
    issues = []
    frames = np.asarray(ctx.frames)
    for side, data in hand_series(ctx, pose).items():
        s = SIDES.index(side)
        error, ok, flipped = data['error'], data['ok'], data['flipped']
        flag = flipped | (ok & (np.nan_to_num(error) > HAND_DIRECTION_DEG))
        for a, b in runs(flag):
            span = slice(a, b + 1)
            hit = flag[span]
            worst = a + int(np.argmax(np.where(hit, np.nan_to_num(error[span]) + 180.0 * flipped[span], -1)))
            count, flips = int(hit.sum()), int(flipped[span].sum())
            measured = np.isfinite(error[span][hit]).any()
            median = float(np.nanmedian(error[span][hit])) if measured else 0.0
            peak = float(np.nanmax(error[span][hit])) if measured else None
            severity = 'high' if count >= 10 and (median >= 70 or flips >= 10) else 'medium' if count >= 5 else 'low'
            what = 'palm faces the other way (thumb and pinky swapped vs the video)' if flips >= count / 2 else \
                f'hand points {median:.0f} deg away from the video hand'
            issues.append(make_issue('hand_orientation', severity, count * (median + 90.0 * flips / max(count, 1)), frames, a, b, worst,
                                     [f'{side}_wrist', f'{side}_hand'],
                                     dict(frames_flagged=count, flipped_frames=flips, direction_error_median_deg=median,
                                          direction_error_max_deg=peak,
                                          confident_frames=int(ok[span].sum())),
                                     f'{side} {what} on {count} confident frames', 'needs a hand orientation (grip) correction or re-run on this range',
                                     dict(smpl=(ELBOWS[s], WRISTS[s]), series={f'{side} hand direction err deg': error})))
    return issues


# Skin contact (mesh) ------------------------------------------------------------------------------------------------

def penetration_pairs():
    """(effector segment, (target segment,)) pairs checked on every frame: hands against legs, head, torso and the other
    arm, forearms against legs and head, and the legs against each other. Segments that share a joint are skipped."""
    legs = [f'{side}_{part}' for side in SIDES for part in ('foot', 'shin', 'thigh')]
    pairs = []
    for side, other in zip(SIDES, SIDES[::-1]):
        pairs += [(f'{side}_hand', (t,)) for t in legs + ['head', 'torso', f'{other}_forearm', f'{other}_upperarm']]
        pairs += [(f'{side}_forearm', (t,)) for t in legs + ['head']]
    pairs += [('left_hand', ('right_hand',)), ('left_shin', ('right_shin',)), ('left_foot', ('right_shin',)), ('right_foot', ('left_shin',)),
              ('left_foot', ('right_foot',))]
    return pairs


def segment_joint(name):
    side, _, part = name.partition('_')
    return {'head': 15, 'torso': 3}[name] if not part else SEGMENT_JOINT[part][SIDES.index(side)]


def contact_tracks(mesh):
    """Enabled self-contact tracks of the mesh's project result: [(track, (hand, target segments))]."""
    if mesh is None or mesh.project is None or mesh.kind is None:
        return []
    from ..motion_tools import base_result
    from .. import self_contact_store as store
    from ..self_contact import parse_target, segment_names
    try:
        tracks, _, _ = store.read(mesh.project, base_result(mesh.kind), {'frames': len(mesh.state['joints']), 'fps': 30.0})
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for track in tracks:
        if not track.get('enabled', True):
            continue
        _, side, part = parse_target(track['target'])
        hand = ('left' if track['effector'] == 'LeftHand' else 'right') + '_hand'
        out.append((track, (hand, tuple(segment_names(part, side)))))
    return out


def check_self_penetration(pose, frames):
    if pose.mesh is None:
        return []
    issues = []
    for (effector, (target,)), (depth, inside, _) in pose.mesh.measure(penetration_pairs()).items():
        flag = depth > PENETRATION_M
        for a, b in runs(flag, 2, 2):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(depth[span]))
            peak, count = float(depth[worst]), int(flag[span].sum())
            median = float(np.median(depth[span][flag[span]]))
            severity = 'high' if peak >= PENETRATION_HIGH_M else 'medium'
            what, into = effector.replace('_', ' '), target.replace('_', ' ')
            issues.append(make_issue('self_penetration', severity, count * peak * 100, frames, a, b, worst, [effector, target],
                                     dict(frames_flagged=count, depth_max_m=peak, depth_median_m=median, inside_vertices_max=int(inside[span].max())),
                                     f'{what} is {peak * 100:.1f} cm inside the {into} (median {median * 100:.1f} cm) on {count} frames',
                                     'push the contact out of the target skin (self-contact solver) or fix the limb placement',
                                     dict(smpl=(segment_joint(effector), segment_joint(target)), series={f'{what} in {into} cm': depth * 100})))
    return issues


def check_contact_gap(pose, frames):
    """Declared self-contacts whose hand skin floats away from the target skin (the real surface gap, not a palm point)."""
    tracks = contact_tracks(pose.mesh)
    if not tracks:
        return []
    measured = pose.mesh.measure([pair for _, pair in tracks])
    issues = []
    for track, pair in tracks:
        _, _, gap = measured[pair]
        active = np.zeros(len(frames), bool)
        for a, b in track['ranges']:
            active[max(a - frames[0], 0):max(b - frames[0], 0)] = True
        far = np.nan_to_num(gap, nan=1.0)
        flag = active & (far > CONTACT_GAP_M)
        for a, b in runs(flag, 2, 3):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(far[span]))
            count, median = int(flag[span].sum()), float(np.median(far[span][flag[span]]))
            severity = 'high' if median >= CONTACT_GAP_HIGH_M else 'medium'
            measured_gap = np.isfinite(gap[span][flag[span]]).all()
            label = f"{median * 100:.1f} cm" if measured_gap else 'more than 5 cm'
            issues.append(make_issue('contact_gap', severity, count * min(median, .1) * 100, frames, a, b, worst, [pair[0], pair[1][0]],
                                     dict(frames_flagged=count, skin_gap_median_m=median if measured_gap else None,
                                          skin_gap_max_m=float(gap[span][flag[span]].max()) if measured_gap else None, track=track.get('id')),
                                     f"{track['effector']} -> {track['target']} ({track.get('id')}): the hand skin floats {label} from the "
                                     f"{pair[1][0].replace('_', ' ')} on {count} contact frames", 'run `correct` with the track enabled',
                                     dict(smpl=(segment_joint(pair[0]), segment_joint(pair[1][0])), series={'skin gap cm': gap * 100})))
    return issues


def skin_summary(pose, frames):
    """Per declared self-contact: skin gap (negative = penetration) and penetration over its contact frames; None without a mesh."""
    if pose is None or pose.mesh is None:
        return None
    sliced = pose.rows(frames)
    tracks = contact_tracks(sliced.mesh)
    measured = sliced.mesh.measure([pair for _, pair in tracks]) if tracks else {}
    out = dict(geometry='SMPL-X mesh, segments by dominant skinning weight', penetration_m=PENETRATION_M, high_m=PENETRATION_HIGH_M, contacts=[])
    for track, pair in tracks:
        depth, _, gap = measured[pair]
        active = np.zeros(len(frames), bool)
        for a, b in track['ranges']:
            active[max(a - frames[0], 0):max(b - frames[0], 0)] = True
        if not active.any():
            continue
        g = gap[active]
        out['contacts'].append(dict(id=track.get('id'), effector=pair[0], target=list(pair[1]), frames=int(active.sum()),
                                    penetration_max_m=float(depth[active].max()), penetration_median_m=float(np.median(depth[active])),
                                    penetrating_frames=int((depth[active] > PENETRATION_M).sum()),
                                    skin_gap_median_m=float(np.nanmedian(g)) if np.isfinite(g).any() else None,
                                    skin_gap_max_m=float(np.nanmax(g)) if np.isfinite(g).any() else None, far_frames=int(np.isnan(g).sum())))
    return out


def rank(issues):
    """review.rank with the anatomy prefixes: sort by severity then score, assign ids such as WL1."""
    issues.sort(key=lambda item: (SEVERITY[item['severity']], -item['score']))
    prefix, counters = {**REVIEW_PREFIX, **PREFIX}, {}
    for issue in issues:
        code = prefix.get(issue['check'], issue['check'][:2].upper())
        counters[code] = counters.get(code, 0) + 1
        issue['id'] = f'{code}{counters[code]}'
    return issues


def checks(ctx, pose):
    """Every anatomy check for review's Context (`ctx.frames` video frames) and a Pose covering them."""
    sliced = pose.rows(ctx.frames)
    return (check_anatomy(sliced, ctx.frames, ctx.fps) + check_hand_orientation(ctx, sliced) + check_self_penetration(sliced, ctx.frames) +
            check_contact_gap(sliced, ctx.frames))


def hand_coverage(ctx, pose):
    sliced = pose.rows(ctx.frames)
    return {side: dict(confident_frames=int(data['ok'].sum()), flipped_frames=int(data['flipped'].sum()),
                       direction_error_median_deg=float(np.nanmedian(data['error'])) if data['ok'].any() else None)
            for side, data in hand_series(ctx, sliced).items()}


# Loaders ------------------------------------------------------------------------------------------------------------

def pose_from_state(state, first=0, source='smplx', body=None, project=None, kind=None):
    """Pose from `smplx_body.Body.pose` output (55 SMPL-X joints); with the Body it carries the mesh for skin checks
    (project and kind name the result whose self-contact tracks the gap check reads)."""
    n = len(state['joints'])
    knuckles = np.empty((n, 2, 5, 3, 3))
    fingers = np.empty((n, 2, 5, 3, 3, 3))
    finger_rest = np.empty((2, 5, 4, 3))
    rest = state['rest'][0]
    for s, side in enumerate(SIDES):
        start = HAND_START[side]
        for f, finger in enumerate(FINGERS):
            ids = [start + k for k in FINGER_OFFSETS[finger]]
            knuckles[:, s, f] = state['joints'][:, ids]
            fingers[:, s, f] = state['local'][:, ids]
            finger_rest[s, f, :3] = rest[ids]
            finger_rest[s, f, 3] = 2 * rest[ids[2]] - rest[ids[1]]
    mesh = None if body is None else Mesh(body, state, project=None if project is None else Path(project), kind=kind)
    return Pose(state['joints'][:, :22], state['world'][:, :22], state['local'][:, :22], rest[:22], knuckles, fingers, finger_rest,
                int(first), source, mesh)


def load_project_pose(project, kind, body=None, path=None):
    """Pose of a project result (or of `path`, a file in the same params format, e.g. `<kind>_before_selfcontact.pt`)."""
    from ..motion_tools import RESULTS
    from .smplx_body import Body, load_params, model_path
    project = Path(project)
    if path is None:
        filename, key = RESULTS[kind]
        path, key = project / filename, key
    else:
        key = 'smpl_params_global'
    body = body or Body(model_path(project))
    return pose_from_state(body.pose(load_params(path, key)), 0, 'smplx', body, project, kind), body


def pose_from_dump(data, frame_base=1):
    """Pose from a `blender_pose.py` npz (dict-like). Finger rotations stay None: the FBX bind bakes the hand mean."""
    from .blender_joints import BONES
    names = [str(name) for name in data['names']]
    index = {name.casefold(): i for i, name in enumerate(names)}
    missing = [bone for bone in BONES if bone.casefold() not in index]
    if missing:
        raise ValueError('FBX lacks body bones: ' + ', '.join(missing))
    body = [index[bone.casefold()] for bone in BONES]
    heads, rot = np.asarray(data['heads'], np.float64), np.asarray(data['rotations'], np.float64)
    rest_heads, rest_rot = np.asarray(data['rest_heads'], np.float64), np.asarray(data['rest_rotations'], np.float64)
    world = rot[:, body] @ rest_rot[body].transpose(0, 2, 1)[None]
    from ..mixamo_rig import BODY_PARENTS
    local = world.copy()
    for j, p in enumerate(BODY_PARENTS):
        if p >= 0:
            local[:, j] = world[:, p].transpose(0, 2, 1) @ world[:, j]
    knuckles = None
    names_needed = [f'{side.capitalize()}Hand{finger.capitalize()}{k}' for side in SIDES for finger in FINGERS for k in (1, 2, 3)]
    if all(name.casefold() in index for name in names_needed):
        ids = [index[name.casefold()] for name in names_needed]
        knuckles = heads[:, ids].reshape(len(heads), 2, 5, 3, 3)
    frames = np.asarray(data['frames'])
    return Pose(heads[:, body], world, local, rest_heads[body], knuckles, None, None, int(frames[0]) - frame_base, 'fbx')


def load_fbx_pose(path, session=None):
    """Bone rotations of an FBX via the bundled Blender (frame 1 of a default import = video frame 0)."""
    from ..retarget.runtime import blender
    from .pipeline import CliError
    try:
        executable = blender()
    except RuntimeError as error:
        raise CliError(str(error)) from None
    path = Path(path)
    if session is not None:
        session.info(f'Reading {path.name} bone rotations with Blender ...')
    with tempfile.TemporaryDirectory(prefix='rgbmocap-anatomy-') as folder:
        out = Path(folder) / 'pose.npz'
        command = [str(executable), '-b', '--factory-startup', '--disable-autoexec', '--threads', '3', '--python-exit-code', '1',
                   '--python', str(BLENDER_SCRIPT), '--', str(path), str(out)]
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        done = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        failure = out.with_suffix('.json')
        if failure.is_file():
            raise CliError(f'{path.name}: ' + json.loads(failure.read_text(encoding='utf-8'))['error'])
        if done.returncode or not out.is_file():
            tail = '\n'.join((done.stdout + done.stderr).strip().splitlines()[-15:])
            raise CliError(f'Blender could not read {path.name} (exit {done.returncode}):\n{tail}')
        with np.load(out, allow_pickle=False) as data:
            try:
                return pose_from_dump(data)
            except ValueError as error:
                raise CliError(f'{path.name}: {error}') from None


def load_pose(project, kind, fbx=None, session=None):
    """Pose for review: the FBX when given, else the project result."""
    if fbx is not None:
        return load_fbx_pose(fbx, session)
    return load_project_pose(project, kind)[0]


def series(pose):
    """Per-frame anatomy numbers (for reports and baselines): interiors, bends, wrist angles, twists."""
    out = {}
    for name, kind, upper, middle, end, axis in limbs():
        bend, angle, off = hinge_bend(pose.joints, pose.world, upper, middle, end, axis)
        out[f'{name}_interior_deg'], out[f'{name}_bend_deg'], out[f'{name}_off_plane_deg'] = angle, bend, off
    for s, side in enumerate(SIDES):
        flex, radial, twist, swing = wrist_angles(pose.local[:, WRISTS[s]], pose.rest, side)
        out.update({f'{side}_wrist_flexion_deg': flex, f'{side}_wrist_radial_deg': radial, f'{side}_wrist_swing_deg': swing})
    for name, (_, _, twist) in joint_twists(pose).items():
        out[f'{name}_twist_deg'] = twist
    return out
