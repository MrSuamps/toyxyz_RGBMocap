"""`review` command: measure how well a reconstructed (or edited FBX) motion matches the source video and flag fixable problems.

Writes report.json (machine-readable, with a ready `fix` spec), summary.md and overlay images. numpy/torch/cv2 load lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from .blender_joints import BONES
from .pipeline import MODELS, CliError, current_kind, read_json, resolve_project

EXIT_USAGE = 2
BLENDER_SCRIPT = Path(__file__).resolve().with_name('blender_joints.py')
# Spec / Blender frame numbering after a default FBX import: Blender frame = video frame + 1.
FRAME_BASE = 1
# Sapiens scores are raw heatmap maxima (~0.04-0.85); below this a keypoint is treated as not visible.
VISIBLE = 0.5
GAP = 2
MIN_RUN = 3
SEVERITY = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}

# COCO-63 = COCO-WholeBody without face: 0 nose, 1/2 L/R eye, 3/4 L/R ear, 5/6 L/R shoulder, 7/8 L/R elbow, 9/10 L/R wrist, 11/12 L/R hip,
# 13/14 L/R knee, 15/16 L/R ankle, 17/18/19 L big toe/small toe/heel, 20/21/22 R big toe/small toe/heel, 23-42 L hand, 43-62 R hand
# (thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4). SMPL-X body joints follow BONES: 0 pelvis, 1/2 L/R hip, 4/5 knee, 7/8 ankle,
# 10/11 toe, 12 neck, 15 head, 16/17 shoulder, 18/19 elbow, 20/21 wrist.
POSE = (('left_shoulder', 5, 16), ('right_shoulder', 6, 17), ('left_elbow', 7, 18), ('right_elbow', 8, 19), ('left_wrist', 9, 20),
        ('right_wrist', 10, 21), ('left_hip', 11, 1), ('right_hip', 12, 2), ('left_knee', 13, 4), ('right_knee', 14, 5), ('left_ankle', 15, 7),
        ('right_ankle', 16, 8), ('left_toe', 17, 10), ('right_toe', 20, 11))
POSE_INDEX = {name: i for i, (name, _, _) in enumerate(POSE)}
# SMPL-X hip centres sit ~30 px off the COCO hips, so they steer the 2D alignment only weakly.
ALIGN = {0: 1.0, 1: 1.0, 6: 0.3, 7: 0.3}
LIMBS = {'left_arm': (2, 4), 'right_arm': (3, 5), 'left_leg': (8, 10), 'right_leg': (9, 11)}
GROUPS = {'torso': (0, 1, 6, 7), 'arms': (2, 3, 4, 5), 'legs': (8, 9, 10, 11), 'feet': (12, 13)}
HAND = {'Left': range(23, 43), 'Right': range(43, 63)}
# Left/right keypoint pairs Sapiens can swap, and the POSE pairs that decide a swap.
SWAPS = {'legs': ([(13, 14), (15, 16), (17, 20), (18, 21), (19, 22)], [(8, 9), (10, 11), (12, 13)]),
         'arms': ([(7, 8), (9, 10)] + [(23 + i, 43 + i) for i in range(20)], [(2, 3), (4, 5)])}
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
COCO_EDGES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19),
              (17, 18), (16, 20), (16, 22), (20, 21), (0, 1), (0, 2), (1, 3), (2, 4))

POSE_THRESHOLD = 0.35          # aligned limb error, torso lengths
CONTACT_2D = {'foot': 0.25, 'knee': 0.20, 'hip': 0.15, 'head': 0.25, 'hand': 0.25}       # torso lengths
CONTACT_3D = {'foot': 0.15, 'knee': 0.15, 'hip': 0.22, 'head': 0.25, 'hand': 0.15}       # meters, joint to joint
PIN_DISTANCE = {'foot': 0.08, 'knee': 0.08, 'hip': 0.10, 'head': 0.12, 'hand': 0.06}     # meters
HAND_REACH = 0.08              # wrist -> palm centre along the forearm, meters
SLIDE_SPEED = 0.15             # m/s
PENETRATION = 0.01             # m
JITTER_FACTOR, JITTER_MIN, JITTER_LOCAL = 8.0, 20.0, 3.0   # x median, m/s^2, x local median
DRIFT = 0.25                   # torso lengths
JITTER_GROUPS = {'torso': ((0, 3, 6, 9, 12, 15), ('Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head')),
                 'left_arm': ((16, 18, 20), ('LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand')),
                 'right_arm': ((17, 19, 21), ('RightShoulder', 'RightArm', 'RightForeArm', 'RightHand')),
                 'left_leg': ((4, 7, 10), ('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase')),
                 'right_leg': ((5, 8, 11), ('RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'))}
NOT_YET = {'gvhmr': 'its 2D keypoints are COCO-17 (keypoints.npz)', 'flowhmr': 'it has no saved 2D keypoints'}

DESCRIPTION = '''Compare a motion with the source video's 2D keypoints and flag fixable problems: missed hand contacts,
limb pose errors, foot sliding, floor penetration, jitter and placement drift, plus anatomy (locked or hyperextended
elbows/knees, sudden folds, wrist limits, joint twist, finger pose, hand orientation vs the video). Writes report.json
(with a ready `fix` spec), summary.md and overlay images (green: 2D keypoints, magenta: 3D motion projected per frame).'''
EPILOG = '''frames: report and --frames use video frame numbers (0-based); fix suggestions use Blender numbering
(video frame + 1), as the `fix` command expects. Supports ExMR projects.'''


def register(commands, common):
    from ..motion_tools import RESULTS
    p = commands.add_parser('review', parents=[common], help='measure a motion against the video; report issues and fixes',
                            description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', required=True, metavar='P', help='workspace root, model project or person take')
    p.add_argument('--model', choices=MODELS, help='model inside a workspace root (default: its active model)')
    p.add_argument('--result', choices=sorted(RESULTS), metavar='KIND', help='project result to review (default: the project selection)')
    p.add_argument('--motion', metavar='EDITED.fbx', help='review this FBX (e.g. a `fix` output) against the project video instead')
    p.add_argument('--out', metavar='DIR', help='report folder (default: <model project>/review/<result or FBX name>-<time>)')
    p.add_argument('--frames', metavar='A-B', help='video frames to evaluate, inclusive (default: all)')
    p.add_argument('--max-images', dest='max_images', type=int, default=24, metavar='N',
                   help='overlay images to write including the contact sheet (default 24, 0 = none)')
    p.add_argument('--compare', choices=('original', 'before-selfcontact'),
                   help='also review the reference layer and report fixed / new issues, untouched-frame identity and rotation deltas')
    p.add_argument('--no-anatomy', dest='anatomy', action='store_false', help='skip the anatomy checks (no SMPL-X / bone rotations needed)')
    p.set_defaults(handler=cmd_review)


# Inputs -------------------------------------------------------------------------------------------------------------

@dataclass
class Observation:
    """2D evidence and per-frame cameras for one take; index = video frame."""
    keypoints: object       # (F, 63, 3) COCO-63 [x_px, y_px, score] in full-resolution video pixels
    rotation: object        # (F, 3, 3) world -> camera (OpenCV: +X right, +Y down, +Z forward)
    translation: object     # (F, 3)
    K: object
    video: Path
    fps: float


@dataclass
class Motion:
    joints: object          # (N, 22, 3) world Y-up meters, BONES order
    first: int              # video frame of joints[0]
    label: str
    source: str


def exmr_inputs(project):
    return [project / name for name in ('exmr_keypoints.pt', 'hmr4d_results.pt', 'initial_joints.npz', 'tracking.npz')]


def load_exmr(project):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    keypoints = torch.load(project / 'exmr_keypoints.pt', map_location='cpu', weights_only=True)
    keypoints = np.asarray(keypoints, np.float64)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (63, 3):
        raise CliError(f'exmr_keypoints.pt has shape {keypoints.shape}; expected (frames, 63, 3).')
    record = torch.load(project / 'hmr4d_results.pt', map_location='cpu', weights_only=True)
    world, incam = record['smpl_params_global'], record['smpl_params_incam']
    with np.load(project / 'initial_joints.npz', allow_pickle=False) as data:
        initial = np.asarray(data['joints'], np.float64)
    with np.load(project / 'tracking.npz', allow_pickle=False) as data:
        K = np.asarray(data['K'], np.float64)
    rotation_world = Rotation.from_rotvec(world['global_orient'].numpy()).as_matrix()
    rotation_incam = Rotation.from_rotvec(incam['global_orient'].numpy()).as_matrix()
    world_t, incam_t = world['transl'].numpy().astype(np.float64), incam['transl'].numpy().astype(np.float64)
    if not len(keypoints) == len(world_t) == len(initial):
        raise CliError(f'Keypoints ({len(keypoints)}), camera ({len(world_t)}) and joints ({len(initial)}) cover different frame counts.')
    # scene_camera.paired_camera per frame: SMPL translation rotates about the rest pelvis, so pivot both roots there.
    rotation = rotation_incam @ rotation_world.transpose(0, 2, 1)
    pelvis = initial[:, 0] - world_t
    translation = incam_t + pelvis - np.einsum('fij,fj->fi', rotation, world_t + pelvis)
    video = project / 'input.mp4'
    if not video.is_file():
        from ..project_storage import source_video
        video = Path(source_video(project))
    fps = float(read_json(project / 'project.json').get('fps', 30))
    return Observation(keypoints, rotation, translation, K, video, fps)


SOURCES = {'exmr': (exmr_inputs, load_exmr)}


def load_project_motion(project, kind):
    import numpy as np
    with np.load(project / f'{kind}_joints.npz', allow_pickle=False) as data:
        joints = np.asarray(data['joints'], np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise CliError(f'{kind}_joints.npz has shape {joints.shape}; expected (frames, 22, 3).')
    return Motion(joints, 0, kind, f'{kind}_joints.npz')


def load_fbx_motion(path, session):
    """Bone head positions from the bundled Blender; frame 1 of a default import is video frame 0."""
    import numpy as np
    from ..retarget.runtime import blender
    try:
        executable = blender()
    except RuntimeError as error:
        raise CliError(str(error)) from None
    session.info(f'Reading {path.name} with Blender ...')
    with tempfile.TemporaryDirectory(prefix='rgbmocap-review-') as folder:
        out = Path(folder) / 'joints.npz'
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
            joints, frames, fps = np.asarray(data['joints'], np.float64), np.asarray(data['frames']), float(data['fps'])
    if len(frames) > 1 and np.any(np.diff(frames) != 1):
        raise CliError(f'{path.name}: the animation frames are not contiguous.')
    return Motion(joints, int(frames[0]) - FRAME_BASE, path.stem, str(path)), fps


def parse_frames(value):
    if value is None:
        return None
    try:
        first, last = (int(part) for part in str(value).split('-', 1))
    except ValueError:
        raise CliError('--frames expects A-B video frame numbers, e.g. 300-420', EXIT_USAGE) from None
    if first < 0 or last < first:
        raise CliError(f'--frames {value}: expected 0 <= A <= B', EXIT_USAGE)
    return first, last


# Geometry -----------------------------------------------------------------------------------------------------------

def project_points(points, rotation, translation, K):
    """World points (N, ..., 3) -> pixels (N, ..., 2) with one camera per frame; behind-camera points become NaN."""
    import numpy as np
    shape = points.shape
    flat = points.reshape(shape[0], -1, 3)
    cam = np.einsum('fij,fkj->fki', rotation, flat) + translation[:, None]
    z = cam[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.stack([K[0, 0] * cam[..., 0] / z + K[0, 2], K[1, 1] * cam[..., 1] / z + K[1, 2]], -1)
    uv[z <= 1e-6] = np.nan
    return uv.reshape(shape[:-1] + (2,))


def weighted_median(values, weights):
    import numpy as np
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return values[order][np.searchsorted(cumulative, cumulative[-1] / 2)]


def runs(mask, gap=GAP, minimum=MIN_RUN):
    """Inclusive (start, end) index runs of True, bridging gaps of up to `gap` frames, with at least `minimum` True frames."""
    import numpy as np
    spans = []
    for i in np.flatnonzero(mask):
        if spans and i - spans[-1][1] <= gap + 1:
            spans[-1][1] = int(i)
        else:
            spans.append([int(i), int(i)])
    return [(a, b) for a, b in spans if int(np.count_nonzero(mask[a:b + 1])) >= minimum]


def fill(values):
    """Linear interpolation over NaNs (edge values held)."""
    import numpy as np
    good = np.isfinite(values)
    if not good.any():
        return values
    index = np.arange(len(values))
    return np.interp(index, index[good], values[good])


def rounded(value, digits=3):
    import numpy as np
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, np.ndarray):
        return rounded(value.tolist(), digits)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), digits)
    return value


class Context:
    """Per-frame arrays for the evaluated range; local index i = video frame frames[i]."""

    def __init__(self, obs, motion, first, last, reference=None):
        import numpy as np
        from scipy.ndimage import median_filter
        self.frames = np.arange(first, last + 1)
        self.fps = obs.fps
        local = self.frames - motion.first
        self.raw_keypoints = obs.keypoints[self.frames].copy()
        self.joints = motion.joints[local]
        self.rotation, self.translation, self.K = obs.rotation[self.frames], obs.translation[self.frames], obs.K
        self.uv = self.project(self.joints)
        # An edited FBX is aligned with the offsets of its source motion: fixes that move the shoulders (chain 3 pins)
        # must not shift the whole skeleton and show up as errors on untouched limbs.
        anchor = self.uv
        if reference is not None:
            ref_local = self.frames - reference.first
            if ref_local.min() >= 0 and ref_local.max() < len(reference.joints):
                anchor = self.project(reference.joints[ref_local])
        kp = self.raw_keypoints
        n = len(self.frames)
        # Global 2D offset per frame: weighted median of confident torso keypoint - projection, so root shifts do not dominate pose error.
        offset = np.full((n, 2), np.nan)
        for f in range(n):
            rows = [(POSE[p][1], POSE[p][2], w) for p, w in ALIGN.items() if kp[f, POSE[p][1], 2] >= VISIBLE]
            rows = [(c, s, w) for c, s, w in rows if np.isfinite(anchor[f, s]).all()]
            if rows:
                delta = np.array([kp[f, c, :2] - anchor[f, s] for c, s, _ in rows])
                weights = np.array([w for _, _, w in rows])
                offset[f] = [weighted_median(delta[:, 0], weights), weighted_median(delta[:, 1], weights)]
        self.aligned = np.isfinite(offset[:, 0])
        self.offset = np.stack([fill(offset[:, 0]), fill(offset[:, 1])], 1) if self.aligned.any() else np.zeros((n, 2))
        ok = (kp[:, [5, 6, 11, 12], 2] >= VISIBLE).all(1)
        torso = np.where(ok, np.linalg.norm((kp[:, 5, :2] + kp[:, 6, :2]) / 2 - (kp[:, 11, :2] + kp[:, 12, :2]) / 2, axis=1), np.nan)
        if np.isfinite(torso).any():
            torso = median_filter(fill(torso), size=9, mode='nearest')
        else:
            torso = np.linalg.norm((self.uv[:, 16] + self.uv[:, 17]) / 2 - (self.uv[:, 1] + self.uv[:, 2]) / 2, axis=1)
        self.torso = np.maximum(fill(torso), 1.0)
        self.keypoints, self.swaps = self.unswap(kp)
        self.err, self.raw = self.pose_errors(self.keypoints)

    def project(self, points):
        return project_points(points, self.rotation, self.translation, self.K)

    def project_at(self, i, points):
        """(K, 3) world points on local frame i -> (K, 2) pixels."""
        return project_points(points[None], self.rotation[i:i + 1], self.translation[i:i + 1], self.K)[0]

    def unswap(self, kp):
        """Relabel left/right limb keypoints on frames where the swapped assignment matches the 3D clearly better."""
        import numpy as np
        fixed = kp.copy()
        swaps = {}
        for group, (pairs, tests) in SWAPS.items():
            direct = np.zeros(len(kp))
            cross = np.zeros(len(kp))
            count = np.zeros(len(kp))
            for a, b in tests:
                (_, ca, sa), (_, cb, sb) = POSE[a], POSE[b]
                ok = (kp[:, ca, 2] >= VISIBLE) & (kp[:, cb, 2] >= VISIBLE)
                ka, kb = kp[:, ca, :2] - self.offset, kp[:, cb, :2] - self.offset
                d = np.linalg.norm(ka - self.uv[:, sa], axis=1) + np.linalg.norm(kb - self.uv[:, sb], axis=1)
                c = np.linalg.norm(ka - self.uv[:, sb], axis=1) + np.linalg.norm(kb - self.uv[:, sa], axis=1)
                ok &= np.isfinite(d) & np.isfinite(c)
                direct += np.where(ok, d, 0)
                cross += np.where(ok, c, 0)
                count += ok
            per_joint = 2 * np.maximum(count, 1) * self.torso
            direct, cross = direct / per_joint, cross / per_joint
            flag = (count > 0) & (cross < 0.5 * direct) & (direct - cross > 0.15)
            for a, b in pairs:
                fixed[flag, a], fixed[flag, b] = kp[flag, b], kp[flag, a]
            swaps[group] = dict(mask=flag, direct=direct, swapped=cross)
        return fixed, swaps

    def pose_errors(self, kp):
        import numpy as np
        coco = [c for _, c, _ in POSE]
        smpl = [s for _, _, s in POSE]
        visible = kp[:, coco, 2] >= VISIBLE
        aligned = np.linalg.norm(kp[:, coco, :2] - self.offset[:, None] - self.uv[:, smpl], axis=2) / self.torso[:, None]
        raw = np.linalg.norm(kp[:, coco, :2] - self.uv[:, smpl], axis=2) / self.torso[:, None]
        return np.where(visible, aligned, np.nan), np.where(visible, raw, np.nan)

    def video(self, i):
        return int(self.frames[i])


def interval(ctx, a, b):
    first, last = ctx.video(a), ctx.video(b)
    return dict(frames=[first, last], blender_frames=[first + FRAME_BASE, last + FRAME_BASE], length=last - first + 1)


def grade(value, high, medium):
    return 'high' if value >= high else 'medium' if value >= medium else 'low'


def lower(severity):
    return {'high': 'medium', 'medium': 'low'}.get(severity, severity)


# Checks -------------------------------------------------------------------------------------------------------------

def effector_points(ctx, side):
    """(name, 2D [x, y, score], 3D, bone) for a hand: the wrist and the hand keypoint centroid (palm point in 3D)."""
    import numpy as np
    kp, joints = ctx.keypoints, ctx.joints
    wrist_c, wrist_s, elbow_s = (9, 20, 18) if side == 'Left' else (10, 21, 19)
    block = kp[:, list(HAND[side])]
    ok = block[..., 2] >= VISIBLE
    count = ok.sum(1)
    centroid = (block[..., :2] * ok[..., None]).sum(1) / np.maximum(count, 1)[:, None]
    centroid = np.concatenate([centroid, np.where(count >= 5, 1.0, 0.0)[:, None]], 1)
    forearm = joints[:, wrist_s] - joints[:, elbow_s]
    palm = joints[:, wrist_s] + HAND_REACH * forearm / np.maximum(np.linalg.norm(forearm, axis=1, keepdims=True), 1e-6)
    bone = side + 'Hand'
    return [(f'{side.lower()}_wrist', kp[:, wrist_c], joints[:, wrist_s], bone), (f'{side.lower()}_hand', centroid, palm, bone)]


def target_groups(ctx):
    kp, joints = ctx.keypoints, ctx.joints
    groups = []
    for side, (ankle, heel, big, small, knee, hip, a3, t3, k3, h3) in (('Left', (15, 19, 17, 18, 13, 11, 7, 10, 4, 1)),
                                                                     ('Right', (16, 22, 20, 21, 14, 12, 8, 11, 5, 2))):
        low = side.lower()
        groups.append((side + 'Foot', 'foot', [(low + '_ankle', kp[:, ankle], joints[:, a3], side + 'Foot'),
                                                (low + '_heel', kp[:, heel], joints[:, a3], side + 'Foot'),
                                                (low + '_big_toe', kp[:, big], joints[:, t3], side + 'ToeBase'),
                                                (low + '_small_toe', kp[:, small], joints[:, t3], side + 'ToeBase')]))
        groups.append((side + 'Knee', 'knee', [(low + '_knee', kp[:, knee], joints[:, k3], side + 'Leg')]))
        groups.append((side + 'Hip', 'hip', [(low + '_hip', kp[:, hip], joints[:, h3], side + 'UpLeg')]))
    groups.append(('Head', 'head', [(name, kp[:, c], joints[:, 15], 'Head') for name, c in (('nose', 0), ('left_ear', 3), ('right_ear', 4))]))
    return groups


def contact_series(ctx, effectors, targets, kind):
    """Per frame: 2D-touching flag, 3D gap over the touching pairs, projected gap, and the closest pair."""
    import numpy as np
    n = len(ctx.frames)
    touching = np.zeros(n, bool)
    gap3 = np.full(n, np.inf)
    gap_proj = np.full(n, np.inf)
    gap2 = np.full(n, np.inf)
    best = np.full(n, -1)
    pairs = [(e, t) for e in effectors for t in targets]
    for index, ((_, e2, e3, _), (_, t2, t3, _)) in enumerate(pairs):
        visible = (e2[:, 2] >= VISIBLE) & (t2[:, 2] >= VISIBLE)
        d2 = np.where(visible, np.linalg.norm(e2[:, :2] - t2[:, :2], axis=1) / ctx.torso, np.inf)
        close = d2 < CONTACT_2D[kind]
        d3 = np.linalg.norm(e3 - t3, axis=1)
        dp = np.linalg.norm(ctx.project(e3[:, None])[:, 0] - ctx.project(t3[:, None])[:, 0], axis=1) / ctx.torso
        touching |= close
        gap3 = np.where(close, np.minimum(gap3, d3), gap3)
        gap_proj = np.where(close, np.minimum(gap_proj, dp), gap_proj)
        better = d2 < gap2
        best = np.where(better, index, best)
        gap2 = np.minimum(gap2, d2)
    return touching, gap3, gap_proj, gap2, best, pairs


def bone_joint(bone):
    """Joint index at the head of a body bone (BONES order)."""
    return BONES.index(bone)


def contact_pins(ctx, a, b, effector, target, kind, frames):
    """Pin(s) for a missed hand contact. When the arm cannot reach, add chain 3 and the reciprocal pin so the target limb meets it."""
    import numpy as np
    shoulder, elbow, wrist = (16, 18, 20) if effector.startswith('Left') else (17, 19, 21)
    joints = ctx.joints
    reach = float(np.max(np.linalg.norm(joints[:, shoulder] - joints[:, elbow], axis=1) + np.linalg.norm(joints[:, elbow] - joints[:, wrist], axis=1)))
    distance = PIN_DISTANCE[kind]
    # `fix` holds the effector `distance` from the target along their direction at the closest frame; approximate that
    # direction as constant in world space over the range.
    target_path, wrist_path = joints[a:b + 1, bone_joint(target)], joints[a:b + 1, wrist]
    gaps = np.linalg.norm(wrist_path - target_path, axis=1)
    closest = int(np.argmin(gaps))
    direction = (wrist_path[closest] - target_path[closest]) / max(float(gaps[closest]), 1e-6)
    need = np.linalg.norm(joints[a:b + 1, shoulder] - (target_path + distance * direction), axis=1)
    shortfall = float(need.max() - reach)
    # Large corrections need a longer blend or the eased frames themselves jitter (about 1 frame per 7 cm moved).
    ease = int(np.clip(round(float(np.median(gaps)) / 0.07), 6, 15))
    pins = [dict(effector=effector, target=target, frames=frames, ease=ease, distance=distance)]
    note = None
    if shortfall > 0.01:
        pins[0]['chain'] = 3
        if target.endswith(('Hand', 'Foot', 'ToeBase')):
            pins.append(dict(effector=target, target=effector, frames=frames, ease=ease, distance=distance))
            limb = 'other arm' if target.endswith('Hand') else ('left' if target.startswith('Left') else 'right') + ' leg'
            note = f'unreachable by the arm alone (short by up to {shortfall * 100:.0f} cm): the second pin also moves the {limb}'
        else:
            note = f'unreachable by the arm alone (short by up to {shortfall * 100:.0f} cm); chain 3 turns the clavicle, a residual gap may remain'
    return pins, dict(arm_reach_m=reach, reach_shortfall_m=max(shortfall, 0.0)), note


def check_missed_contacts(ctx):
    import numpy as np
    issues = []
    hands = {side: effector_points(ctx, side) for side in ('Left', 'Right')}
    jobs = [(side, name, kind, targets) for side in ('Left', 'Right') for name, kind, targets in target_groups(ctx)]
    jobs.append(('Left', 'RightHand', 'hand', hands['Right']))
    for side, name, kind, targets in jobs:
        touching, gap3, gap_proj, gap2, best, pairs = contact_series(ctx, hands[side], targets, kind)
        flag = touching & (gap3 > CONTACT_3D[kind])
        for a, b in runs(flag):
            span = slice(a, b + 1)
            hit = flag[span]
            worst = a + int(np.argmax(np.where(hit, gap3[span], -1)))
            bones = [pairs[i][1][3] for i in best[span][hit] if i >= 0]
            target_bone = max(set(bones), key=bones.count) if bones else targets[0][3]
            effector = side + 'Hand'
            if kind == 'hand':
                # Pin the hand whose 2D wrist error is larger; the other one is presumably right.
                left, right = (np.nanmean(ctx.err[span, POSE_INDEX[f'{s}_wrist']]) for s in ('left', 'right'))
                effector, target_bone = ('RightHand', 'LeftHand') if np.nan_to_num(right) > np.nan_to_num(left) else ('LeftHand', 'RightHand')
            gaps = gap3[span][hit]
            evidence = 'image' if float(np.median(gap_proj[span][hit])) > 0.4 else 'depth'
            count = int(hit.sum())
            severity = 'high' if count >= 10 and np.median(gaps) >= 0.2 else 'medium' if count >= 5 else 'low'
            if evidence == 'depth' and kind in ('knee', 'hip'):
                severity = lower(severity)
            span_info = interval(ctx, a, b)
            pins, reach, note = contact_pins(ctx, a, b, effector, target_bone, kind, span_info['blender_frames'])
            pair = pairs[best[worst]] if best[worst] >= 0 else (hands[side][0], targets[0])
            issues.append(dict(
                check='missed_contact', severity=severity, score=count * float(np.median(gaps)), **span_info,
                worst_frame=ctx.video(worst), joints=[pair[0][0], pair[1][0]], effector=effector, target=target_bone, evidence=evidence,
                metrics=dict(frames_flagged=count, distance_2d_torso_median=float(np.median(gap2[span][hit])),
                             distance_3d_m_median=float(np.median(gaps)), distance_3d_m_max=float(gaps.max()),
                             projected_distance_torso_median=float(np.median(gap_proj[span][hit])), **reach),
                message=(f'{effector} touches {name} in the video ({np.median(gap2[span][hit]):.2f} torso apart in 2D) but is '
                         f'{np.median(gaps):.2f} m away in 3D' + (' (separated mostly in depth)' if evidence == 'depth' else '')),
                fix=dict(pins=pins), **({'fix_reach_note': note} if note else {}),
                _draw=dict(pair=(pair[0], pair[1]), series={'2D torso': gap2, '3D m': gap3})))
    return issues


def check_pose(ctx):
    import numpy as np
    issues = []
    for limb, members in LIMBS.items():
        values = np.fmax.reduce(ctx.err[:, list(members)], axis=1)
        flag = np.nan_to_num(values) > POSE_THRESHOLD
        for a, b in runs(flag):
            span = slice(a, b + 1)
            means = {POSE[m][0]: float(np.nanmean(ctx.err[span, m])) if np.isfinite(ctx.err[span, m]).any() else None for m in members}
            worst_joint = max((name for name in means if means[name] is not None), key=lambda name: means[name])
            worst = a + int(np.nanargmax(np.nan_to_num(values[span], nan=-1)))
            peak = float(values[worst])
            count = int(flag[span].sum())
            severity = 'high' if peak >= 0.7 and count >= 10 else 'medium' if peak >= 0.5 or count >= 10 else 'low'
            issues.append(dict(
                check='pose_error', severity=severity, score=count * float(np.nanmean(values[span])), **interval(ctx, a, b),
                worst_frame=ctx.video(worst), joints=list(means), limb=limb,
                metrics=dict(frames_flagged=count, peak_torso=peak, mean_torso=float(np.nanmean(values[span])), worst_joint=worst_joint,
                             joint_mean_torso=means, peak_px=peak * float(ctx.torso[worst])),
                message=f'{limb.replace("_", " ")} pose differs from the video by up to {peak:.2f} torso ({peak * ctx.torso[worst]:.0f} px), '
                        f'worst at {worst_joint.replace("_", " ")}',
                fix=None, hint='not pin-fixable from 2D alone; re-run reconstruction on this range or edit the limb by hand',
                _draw=dict(members=members, series={'limb err torso': values})))
    return issues


def contact_mask(ctx, project, kind, floor):
    """(N, 2) foot contacts for L/R: the project's contact segments when present, else low-and-slow feet."""
    import numpy as np
    path = project / f'{kind}_contacts.npz'
    n = len(ctx.frames)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            segments = np.asarray(data['segments'])
        mask = np.zeros((int(max(ctx.frames[-1] + 1, segments[:, 2].max() if len(segments) else 0)), 2), bool)
        for foot, start, end, *_ in segments:
            mask[int(start):int(end), int(foot)] = True
        return mask[ctx.frames], path.name
    speed = foot_speeds(ctx)
    mask = np.zeros((n, 2), bool)
    for foot, (ankle, toe) in enumerate(((7, 10), (8, 11))):
        heights = [ctx.joints[:, j, 1] - floor for j in (ankle, toe)]
        low = np.zeros(n, bool)
        for h in heights:
            low |= h - np.percentile(h, 5) < 0.03
        mask[:, foot] = low & (speed[:, foot].min(1) < 0.4)
        clean = np.zeros(n, bool)
        for a, b in runs(mask[:, foot], 0, 3):
            clean[a:b + 1] = True
        mask[:, foot] = clean
    return mask, 'derived (foot within 3 cm of its lowest height and slower than 0.4 m/s)'


def foot_speeds(ctx):
    """(N, 2 feet, 2 [ankle, toe]) horizontal speed in m/s."""
    import numpy as np
    xz = ctx.joints[:, [7, 10, 8, 11]][..., [0, 2]]
    if len(xz) < 2:
        return np.zeros((len(xz), 2, 2))
    speed = np.linalg.norm(np.gradient(xz, axis=0), axis=-1) * ctx.fps
    return speed.reshape(len(xz), 2, 2)


def check_foot_slide(ctx, contacts):
    import numpy as np
    issues = []
    speed = foot_speeds(ctx)
    for foot, side in enumerate(('Left', 'Right')):
        slowest = speed[:, foot].min(1)
        flag = contacts[:, foot] & (slowest > SLIDE_SPEED)
        for a, b in runs(flag):
            span = slice(a, b + 1)
            pivot = int(np.argmin(speed[span, foot].mean(0)))
            joint = (7, 10)[pivot] if foot == 0 else (8, 11)[pivot]
            path = ctx.joints[span, joint][:, [0, 2]]
            drift = float(np.linalg.norm(path[-1] - path[0]))
            travel = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            peak = float(slowest[span].max())
            worst = a + int(np.argmax(slowest[span]))
            severity = 'high' if travel > 0.1 or peak > 0.5 else 'medium' if travel > 0.04 else 'low'
            bone = side + ('Foot', 'ToeBase')[pivot]
            info = interval(ctx, a, b)
            issues.append(dict(
                check='foot_slide', severity=severity, score=travel * 100, **info, worst_frame=ctx.video(worst),
                joints=[f'{side.lower()}_ankle', f'{side.lower()}_toe'], effector=bone,
                metrics=dict(frames_flagged=int(flag[span].sum()), peak_speed_mps=peak, mean_speed_mps=float(slowest[span].mean()),
                             travel_m=travel, net_drift_m=drift, held_point=bone),
                message=f'{side.lower()} foot slides {travel * 100:.1f} cm during ground contact (up to {peak:.2f} m/s)',
                fix=dict(pins=[dict(effector=bone, target='world', frames=info['blender_frames'], ease=4)]),
                _draw=dict(smpl=((7, 10) if foot == 0 else (8, 11)), series={'speed m/s': slowest})))
    return issues


def check_penetration(ctx, contacts, floor):
    import numpy as np
    issues = []
    for foot, side in enumerate(('Left', 'Right')):
        joints = (7, 10) if foot == 0 else (8, 11)
        depth = np.full(len(ctx.frames), -np.inf)
        which = np.zeros(len(ctx.frames), int)
        rests = {}
        for k, j in enumerate(joints):
            height = ctx.joints[:, j, 1] - floor
            # Joints sit above the sole; their resting height during contact is the zero level.
            ground = contacts[:, foot] if contacts[:, foot].sum() >= 10 else np.ones(len(height), bool)
            rest = float(np.clip(np.median(height[ground]) if floor.source != 'estimated' else 0.0, 0.0, 0.15))
            rests[('ankle', 'toe')[k]] = rest
            d = rest - height
            which = np.where(d > depth, k, which)
            depth = np.maximum(depth, d)
        flag = depth > PENETRATION
        for a, b in runs(flag, GAP, 2):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(depth[span]))
            peak = float(depth[worst])
            name = ('ankle', 'toe')[int(which[worst])]
            issues.append(dict(
                check='floor_penetration', severity=grade(peak, 0.05, 0.025), score=peak * 100 * int(flag[span].sum()), **interval(ctx, a, b),
                worst_frame=ctx.video(worst), joints=[f'{side.lower()}_{name}'],
                metrics=dict(frames_flagged=int(flag[span].sum()), max_depth_m=peak, joint=name, resting_height_m=rests, floor_y=float(floor)),
                message=f'{side.lower()} {name} goes {peak * 100:.1f} cm below its resting height above the floor',
                fix=None, hint='raise the motion with `correct --pelvis-offset` or check the floor height (`correct --floor`)',
                _draw=dict(smpl=joints, series={'depth m': depth})))
    return issues


def check_jitter(ctx):
    import numpy as np
    from scipy.ndimage import median_filter
    issues = []
    if len(ctx.frames) < 5:
        return issues
    local = ctx.joints.copy()
    local[:, 1:] -= ctx.joints[:, :1]
    accel = np.zeros(local.shape[:2])
    accel[1:-1] = np.linalg.norm(local[2:] - 2 * local[1:-1] + local[:-2], axis=-1) * ctx.fps ** 2
    for group, (members, bones) in JITTER_GROUPS.items():
        values = accel[:, list(members)]
        limit = np.maximum(JITTER_FACTOR * np.median(values[1:-1], axis=0), JITTER_MIN)
        context = median_filter(values, size=(15, 1), mode='nearest')
        ratio = values / limit
        flag_joint = (ratio > 1) & (values > JITTER_LOCAL * context)
        flag = flag_joint.any(1)
        for a, b in runs(flag, 3, 1):
            span = slice(a, b + 1)
            worst = a + int(np.argmax(ratio[span].max(1)))
            column = int(np.argmax(ratio[worst]))
            peak = float(ratio[worst, column])
            info = interval(ctx, a, b)
            first, last = info['blender_frames']
            frames = [max(first - 2, ctx.video(0) + FRAME_BASE), min(last + 2, ctx.video(len(ctx.frames) - 1) + FRAME_BASE)]
            issues.append(dict(
                check='jitter', severity='medium' if peak > 2 or int(flag[span].sum()) >= 4 else 'low', score=peak, **info,
                worst_frame=ctx.video(worst), joints=[BONES[m] for m in members], group=group,
                metrics=dict(frames_flagged=int(flag[span].sum()), peak_accel_mps2=float(values[worst, column]),
                             threshold_mps2=float(limit[column]), peak_ratio=peak, worst_joint=BONES[members[column]]),
                message=f'{group.replace("_", " ")} jitters: {BONES[members[column]]} acceleration {values[worst, column]:.0f} m/s2 '
                        f'({peak:.1f}x the threshold)',
                fix=dict(smooth=dict(bones=list(bones), frames=frames, window=5)),
                _draw=dict(smpl=members, series={'accel m/s2': values.max(1)})))
    return issues


def check_drift(ctx):
    import numpy as np
    offset = np.linalg.norm(ctx.offset, axis=1) / ctx.torso
    flag = offset > DRIFT
    issues = []
    for a, b in runs(flag, 5, 5):
        span = slice(a, b + 1)
        worst = a + int(np.argmax(offset[span]))
        issues.append(dict(
            check='placement_drift', severity='info', score=float(offset[worst]), **interval(ctx, a, b), worst_frame=ctx.video(worst),
            joints=['pelvis'],
            metrics=dict(peak_offset_torso=float(offset[worst]), peak_offset_px=float(offset[worst] * ctx.torso[worst]),
                         mean_offset_torso=float(offset[span].mean())),
            message=f'the whole body sits {offset[worst]:.2f} torso ({offset[worst] * ctx.torso[worst]:.0f} px) off the video person '
                    '(root placement / camera drift, removed before the pose checks)',
            fix=None, hint='informational: global placement, not a pose error',
            _draw=dict(unaligned=True, series={'offset torso': offset})))
    return issues


class Floor(float):
    def __new__(cls, value, source):
        item = super().__new__(cls, value)
        item.source = source
        return item


def floor_level(ctx, project, kind):
    import numpy as np
    path = project / f'{kind}_correction.json'
    if path.is_file():
        try:
            return Floor(float(read_json(path)['floor']), path.name)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return Floor(float(np.percentile(ctx.joints[:, [10, 11], 1].min(1), 2)), 'estimated')


def collect_issues(ctx, project, kind, pose=None):
    """All checks on one Context, ranked: (issues, floor, contacts, contacts_source). `pose` adds the anatomy checks."""
    floor = floor_level(ctx, project, kind)
    contacts, contacts_source = contact_mask(ctx, project, kind, floor)
    issues = (check_missed_contacts(ctx) + check_pose(ctx) + check_foot_slide(ctx, contacts) + check_penetration(ctx, contacts, floor) +
              check_jitter(ctx) + check_drift(ctx))
    if pose is not None:
        from . import anatomy
        issues += anatomy.checks(ctx, pose)
    return rank(issues), floor, contacts, contacts_source


# Report -------------------------------------------------------------------------------------------------------------

def statistics(ctx):
    import numpy as np

    def describe(values):
        values = values[np.isfinite(values)]
        if not len(values):
            return dict(median=None, mean=None, p90=None, samples=0)
        return dict(median=float(np.median(values)), mean=float(values.mean()), p90=float(np.percentile(values, 90)), samples=int(len(values)))

    kp = ctx.raw_keypoints
    visible = kp[..., 2] >= VISIBLE
    offset = np.linalg.norm(ctx.offset, axis=1) / ctx.torso
    return dict(
        frames_evaluated=len(ctx.frames), frames_aligned=int(ctx.aligned.sum()), torso_px_median=float(np.median(ctx.torso)),
        aligned_error_torso={group: describe(ctx.err[:, list(members)].ravel()) for group, members in GROUPS.items()},
        raw_error_torso={group: describe(ctx.raw[:, list(members)].ravel()) for group, members in GROUPS.items()},
        joint_error_torso={name: describe(ctx.err[:, i]) for i, (name, _, _) in enumerate(POSE)},
        keypoint_coverage=dict(body=float(visible[:, :17].mean()), feet=float(visible[:, 17:23].mean()), left_hand=float(visible[:, 23:43].mean()),
                               right_hand=float(visible[:, 43:63].mean())),
        placement_offset_torso=dict(median=float(np.median(offset)), max=float(offset.max()),
                                    per_second=[float(offset[i]) for i in range(0, len(offset), max(int(round(ctx.fps)), 1))]))


def anatomy_block(ctx, pose):
    if pose is None:
        return dict(checked=False)
    from . import anatomy
    return dict(checked=True, rotations=pose.source, fingers_checked=pose.fingers is not None, hand_keypoints=anatomy.hand_coverage(ctx, pose),
                thresholds=dict(hyperextension_deg=anatomy.HYPEREXTENSION_DEG, locked_deg=anatomy.LOCK_DEG, locked_frames=anatomy.LOCK_FRAMES,
                                wrist_deg=anatomy.WRIST_LIMITS, twist_deg=anatomy.TWIST_LIMITS, fold_deg=anatomy.FOLD_DEG,
                                fold_seconds=anatomy.FOLD_SECONDS, hand_direction_deg=anatomy.HAND_DIRECTION_DEG))


def swap_report(ctx):
    rows = {}
    for group, data in ctx.swaps.items():
        rows[group] = [dict(frames=[ctx.video(a), ctx.video(b)], blender_frames=[ctx.video(a) + FRAME_BASE, ctx.video(b) + FRAME_BASE])
                       for a, b in runs(data['mask'], 1, 1)]
    return rows


def changed_window(frames, ease):
    margin = max(ease - 1, 0)
    return frames[0] - margin, frames[1] + margin


def suggested_fixes(issues):
    """Pins and smoothing from medium/high issues in rank order; later pins whose ease windows meet an earlier one on the same limb are skipped."""
    pins, smooth, taken = [], [], []
    for issue in issues:
        fix = issue.get('fix') or {}
        if SEVERITY[issue['severity']] > SEVERITY['medium']:
            continue
        if 'pins' in fix:
            # A reciprocal pin may clash with an earlier one; the primary pin alone is still worth suggesting.
            for chosen in ([fix['pins'], fix['pins'][:1]] if len(fix['pins']) > 1 else [fix['pins']]):
                claims = [(('arm' if pin['effector'].endswith('Hand') else 'leg', 'Left' if pin['effector'].startswith('Left') else 'Right'),
                           changed_window(pin['frames'], pin['ease'])) for pin in chosen]
                if not any(limb == other and window[0] <= span[1] and span[0] <= window[1] for limb, window in claims for other, span in taken):
                    break
            else:
                issue['fix_note'] = 'not in suggested_fixes: overlaps a higher-ranked pin on the same limb'
                continue
            taken.extend(claims)
            pins.extend(dict(pin) for pin in chosen)
            issue['in_suggested_fixes'] = True
            if len(chosen) < len(fix['pins']):
                issue['fix_note'] = 'only the first pin is in suggested_fixes: the second overlaps a higher-ranked pin on the same limb'
        elif 'smooth' in fix:
            smooth.append(dict(fix['smooth']))
            issue['in_suggested_fixes'] = True
    spec = dict(version=1, pins=pins, smooth=smooth)
    try:
        from . import fix_spec
        fix_spec.validate(spec)
        status = 'valid'
    except ImportError:
        status = 'not checked (fix command unavailable)'
    except ValueError as error:
        status = f'invalid: {error}'
    return spec, status


def rank(issues):
    from .anatomy import PREFIX
    issues.sort(key=lambda item: (SEVERITY[item['severity']], -item['score']))
    counters = {}
    prefix = dict(missed_contact='MC', pose_error='PE', foot_slide='FS', floor_penetration='FP', jitter='JT', placement_drift='PD', **PREFIX)
    for issue in issues:
        code = prefix[issue['check']]
        counters[code] = counters.get(code, 0) + 1
        issue['id'] = f'{code}{counters[code]}'
    return issues


# Images -------------------------------------------------------------------------------------------------------------

LONG_SIDE = 900
GREEN, MAGENTA, YELLOW, RED, GREY = (60, 220, 60), (255, 0, 255), (0, 230, 255), (40, 40, 255), (150, 90, 150)


def read_frames(path, wanted):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise CliError(f'Cannot open the project video {path}')
    frames, index, last = {}, 0, max(wanted)
    try:
        while index <= last:
            ok, image = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = image
            index += 1
    finally:
        capture.release()
    return frames


def text(image, lines, origin=(8, 20), scale=0.5):
    import cv2
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(22 * scale / 0.5)


def draw(ctx, image, i, issue, lines):
    import cv2
    import numpy as np
    s = LONG_SIDE / max(image.shape[:2])
    image = cv2.resize(image, (int(round(image.shape[1] * s)), int(round(image.shape[0] * s))), interpolation=cv2.INTER_AREA)
    point = lambda xy: (int(round(xy[0] * s)), int(round(xy[1] * s))) if np.isfinite(xy).all() else None
    kp = ctx.raw_keypoints[i]
    shown = ctx.uv[i] + ctx.offset[i]
    draw_info = (issue or {}).get('_draw', {})
    if draw_info.get('unaligned'):
        for j, parent in enumerate(PARENTS):
            a, b = point(ctx.uv[i, j]), point(ctx.uv[i, parent]) if parent >= 0 else None
            if a and b:
                cv2.line(image, a, b, GREY, 1, cv2.LINE_AA)
    for a, b in COCO_EDGES:
        if kp[a, 2] >= VISIBLE and kp[b, 2] >= VISIBLE:
            cv2.line(image, point(kp[a, :2]), point(kp[b, :2]), GREEN, 2, cv2.LINE_AA)
    for k in range(63):
        if kp[k, 2] >= VISIBLE:
            cv2.circle(image, point(kp[k, :2]), 3 if k < 23 else 1, GREEN, -1, cv2.LINE_AA)
    for j, parent in enumerate(PARENTS):
        a, b = point(shown[j]), point(shown[parent]) if parent >= 0 else None
        if a and b:
            cv2.line(image, a, b, MAGENTA, 2, cv2.LINE_AA)
    for j in range(22):
        if point(shown[j]):
            cv2.circle(image, point(shown[j]), 3, MAGENTA, -1, cv2.LINE_AA)
    labels = []
    if 'pair' in draw_info:
        (en, e2, e3, _), (tn, t2, t3, _) = draw_info['pair']
        e3p, t3p = ctx.project_at(i, np.stack([e3[i], t3[i]])) + ctx.offset[i]
        if e2[i, 2] >= VISIBLE and t2[i, 2] >= VISIBLE:
            cv2.line(image, point(e2[i, :2]), point(t2[i, :2]), YELLOW, 2, cv2.LINE_AA)
        if point(e3p) and point(t3p):
            cv2.line(image, point(e3p), point(t3p), RED, 2, cv2.LINE_AA)
            labels += [(e3p, en), (t3p, tn)]
    for m in draw_info.get('members', ()):
        name, c, j = POSE[m]
        ka = ctx.keypoints[i, c]
        if ka[2] >= VISIBLE and point(shown[j]):
            cv2.line(image, point(ka[:2]), point(shown[j]), RED, 2, cv2.LINE_AA)
        labels.append((shown[j], f'{name} {ctx.err[i, m]:.2f}' if np.isfinite(ctx.err[i, m]) else name))
    for j in draw_info.get('smpl', ()):
        labels.append((shown[j], BONES[j]))
    for xy, label in labels:
        p = point(xy)
        if p:
            cv2.circle(image, p, 7, YELLOW, 2, cv2.LINE_AA)
            text(image, [label], (p[0] + 8, p[1] - 6), 0.42)
    width = max(int(image.shape[1] / 9.5), 30)
    header = [f'video frame {ctx.video(i)}  |  Blender frame {ctx.video(i) + FRAME_BASE}']
    header += [part for line in lines for part in textwrap.wrap(line, width)]
    for name, values in draw_info.get('series', {}).items():
        if np.isfinite(values[i]):
            header.append(f'{name}: {values[i]:.2f}')
    text(image, header)
    text(image, ['green: 2D keypoints   magenta: 3D aligned' + ('   grey: 3D unaligned' if draw_info.get('unaligned') else '')],
         (8, image.shape[0] - 10), 0.42)
    return image


def contact_sheet(tiles, path):
    import cv2
    import numpy as np
    columns, rows = 4, 3
    width = LONG_SIDE // columns
    height = max(int(round(tile.shape[0] * width / tile.shape[1])) for tile, _ in tiles)
    sheet = np.zeros((rows * height, columns * width, 3), np.uint8)
    for k, (tile, label) in enumerate(tiles[:columns * rows]):
        small = cv2.resize(tile, (width, int(round(tile.shape[0] * width / tile.shape[1]))), interpolation=cv2.INTER_AREA)
        y, x = (k // columns) * height, (k % columns) * width
        sheet[y:y + small.shape[0], x:x + width] = small
        cv2.rectangle(sheet, (x, y), (x + width - 1, y + 26), (0, 0, 0), -1)
        text(sheet, [label], (x + 4, y + 18), 0.45)
    used = (len(tiles[:columns * rows]) + columns - 1) // columns
    cv2.imwrite(str(path), sheet[:max(used, 1) * height], [cv2.IMWRITE_JPEG_QUALITY, 85])


def render(ctx, obs, issues, folder, limit):
    import cv2
    import numpy as np
    if limit <= 0:
        return [], None
    flagged = [issue for issue in issues if issue['severity'] != 'info']
    sheet = 1 if flagged and limit >= 3 else 0
    samples = min(4, limit // 6)
    budget = limit - sheet - samples
    local = {int(v): k for k, v in enumerate(ctx.frames)}
    plan = []
    for issue in issues:
        if len(plan) < budget:
            plan.append((issue, 'worst', issue['worst_frame']))
    for issue in issues:
        for role, frame in (('start', issue['frames'][0]), ('end', issue['frames'][1])):
            if len(plan) < budget and frame != issue['worst_frame'] and issue['severity'] != 'info':
                plan.append((issue, role, frame))
    for frame in np.linspace(ctx.frames[0], ctx.frames[-1], max(min(limit - sheet - len(plan), 8), 0) + 2)[1:-1]:
        plan.append((None, 'sample', int(round(frame))))
    images = read_frames(obs.video, {frame for _, _, frame in plan})
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob('*.jpg'):
        old.unlink()
    written, tiles = [], []
    for issue, role, frame in plan:
        if frame not in images or frame not in local:
            continue
        i = local[frame]
        if issue is None:
            err = ctx.err[i]
            lines = [f'sample  median aligned error {np.nanmedian(err):.2f} torso' if np.isfinite(err).any() else 'sample']
            name = f'sample_f{frame:04d}.jpg'
        else:
            lines = [f"{issue['id']} {issue['check']} [{issue['severity']}] ({role})", issue['message']]
            name = f"{issue['id']}_{role}_f{frame:04d}.jpg"
        image = draw(ctx, images[frame], i, issue, lines)
        cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(name)
        if issue is not None:
            issue.setdefault('images', []).append('images/' + name)
            if role == 'worst' and issue['severity'] != 'info':
                tiles.append((image, f"{issue['id']} f{frame} b{frame + FRAME_BASE}"))
    sheet_name = None
    if sheet and tiles:
        sheet_name = 'contact_sheet.jpg'
        contact_sheet(tiles, folder / sheet_name)
        written.append(sheet_name)
    return written, sheet_name


def summary_markdown(report):
    lines = [f"# Motion review: {report['motion']['label']}", '',
             f"- Project: `{report['project']}` ({report['model']}, result `{report['result']}`)",
             f"- Motion: `{report['motion']['source']}`",
             f"- Frames: video {report['frames'][0]}-{report['frames'][1]} (Blender {report['blender_frames'][0]}-{report['blender_frames'][1]}), "
             f"{report['stats']['frames_evaluated']} evaluated; floor y = {report['floor']['y']:.3f} m ({report['floor']['source']}); "
             f"contacts: {report['contacts_source']}",
             f"- Errors are in torso lengths (median torso {report['stats']['torso_px_median']:.0f} px), after removing the per-frame 2D offset.",
             '']
    issues = report['issues']
    counts = {level: sum(issue['severity'] == level for issue in issues) for level in SEVERITY}
    lines += [f"## Issues ({', '.join(f'{count} {level}' for level, count in counts.items() if count) or 'none'})", '']
    for number, issue in enumerate(issues, 1):
        f0, f1 = issue['frames']
        b0, b1 = issue['blender_frames']
        lines.append(f"{number}. **{issue['id']} [{issue['severity']}] {issue['check']}**: video {f0}-{f1} (Blender {b0}-{b1}), "
                     f"worst {issue['worst_frame']}. {issue['message']}.")
        if issue.get('images'):
            lines.append('   Images: ' + ', '.join(f'`{name}`' for name in issue['images']))
        if issue.get('fix'):
            note = issue.get('fix_note') or ('' if issue.get('in_suggested_fixes') else 'not in suggested_fixes: low severity')
            note = f' ({note})' if note else ''
            lines.append(f"   Fix{note}: `{json.dumps(next(iter(issue['fix'].values())))}`")
            if issue.get('fix_reach_note'):
                lines.append(f"   Reach: {issue['fix_reach_note']}")
        elif issue.get('hint'):
            lines.append(f"   Hint: {issue['hint']}")
    swaps = report['keypoint_swaps']
    lines += ['', '## Keypoint left/right swaps (2D detector, informational)', '']
    for group, spans in swaps.items():
        lines.append(f"- {group}: " + (', '.join(f"{a}-{b}" for a, b in (s['frames'] for s in spans)) if spans else 'none'))
    stats = report['stats']
    lines += ['', '## Global stats', '', '| group | aligned median | aligned p90 | raw median |', '|---|---|---|---|']
    for group in GROUPS:
        a, r = stats['aligned_error_torso'][group], stats['raw_error_torso'][group]
        fmt = lambda value: '-' if value is None else f'{value:.3f}'
        lines.append(f"| {group} | {fmt(a['median'])} | {fmt(a['p90'])} | {fmt(r['median'])} |")
    cover = stats['keypoint_coverage']
    lines += ['', 'Keypoint coverage (score >= 0.5): ' + ', '.join(f'{key} {value:.0%}' for key, value in cover.items()),
              f"Placement offset (torso): median {stats['placement_offset_torso']['median']:.2f}, max {stats['placement_offset_torso']['max']:.2f}", '']
    fixes = report['suggested_fixes']
    lines += ['## Suggested fix spec', '',
              f"{len(fixes['pins'])} pin(s), {len(fixes['smooth'])} smoothing pass(es); {report['suggested_fixes_status']}. "
              f"Saved as `suggested_fix.json`:", '',
              f"    rgbmocap-cli.cmd fix {report['fix_input']} --spec suggested_fix.json --out fixed.fbx", '',
              '```json', json.dumps(fixes, indent=1), '```']
    if report.get('contact_sheet'):
        lines += ['', f"Contact sheet: `images/{report['contact_sheet']}`"]
    if report.get('compare'):
        from .compare import summary_lines
        lines += summary_lines(report['compare'])
    return '\n'.join(lines) + '\n'


# Command ------------------------------------------------------------------------------------------------------------

def cmd_review(args, session):
    started = time.monotonic()
    target = resolve_project(args.project, args.model)
    model = target.model
    if model not in SOURCES:
        reason = NOT_YET.get(model, 'no 2D keypoint reader exists for it')
        raise CliError(f'review supports {", ".join(SOURCES)} projects for now; {model} is not supported yet ({reason}).')
    if target.project is None:
        raise CliError(f'This workspace has no {model} motion yet. Reconstruct it first.')
    project = target.project
    inputs, loader = SOURCES[model]
    missing = [path.name for path in inputs(project) if not path.is_file()]
    if missing:
        where = ' Pass a person take, not the people root.' if target.people_root is not None and project == target.people_root else ''
        raise CliError(f'{project} lacks {", ".join(missing)}; reconstruct it first.{where}')
    kind = args.result or current_kind(project, model)
    frames = parse_frames(args.frames)
    if args.max_images < 0:
        raise CliError('--max-images must be 0 or more', EXIT_USAGE)
    fbx = None
    if args.motion:
        fbx = Path(args.motion).expanduser().resolve()
        if fbx.suffix.casefold() != '.fbx':
            raise CliError(f'--motion must be an .fbx file: {fbx}', EXIT_USAGE)
        if not fbx.is_file():
            raise CliError(f'Motion file not found: {fbx}')
    elif not (project / f'{kind}_joints.npz').is_file():
        raise CliError(f'The {kind} motion has no {kind}_joints.npz in {project}; run `reconstruct` or `correct` first.')
    if args.compare and fbx:
        raise CliError('--compare reviews project results; it cannot be combined with --motion.', EXIT_USAGE)
    if args.compare:
        from .compare import reference
        reference(project, kind, args.compare)
    stem = fbx.stem if fbx else kind
    out = Path(args.out).expanduser().resolve() if args.out else project / 'review' / f'{stem}-{time.strftime("%Y%m%d-%H%M%S")}'
    if out.exists() and not out.is_dir():
        raise CliError(f'--out must be a folder: {out}', EXIT_USAGE)
    summary = dict(command='review', project=str(project), model=model, result=kind, motion=str(fbx) if fbx else f'{kind}_joints.npz',
                   out=str(out))
    if session.dry_run:
        summary.update(dry_run=True, frames=list(frames) if frames else 'all', writes=[str(out / name) for name in
                       ('report.json', 'summary.md', 'suggested_fix.json', 'images/')], blender=bool(fbx), anatomy=args.anatomy,
                       compare=args.compare)
        if not session.json:
            print(f'Would review {summary["motion"]} against {project / "input.mp4"} ({model}, result {kind})')
            if fbx:
                print('  reads the FBX bone positions' + (' and rotations' if args.anatomy else '') + ' with the bundled Blender first')
            if args.compare:
                print(f'  also reviews the {args.compare} layer and diffs the issues')
            print(f'  frames: {args.frames or "all"}; up to {args.max_images} images')
            print(f'  writes: {out}\\report.json, summary.md, suggested_fix.json, images\\*.jpg')
        return 0, summary
    import numpy as np
    session.info(f'Loading {model} keypoints and cameras ...')
    obs = loader(project)
    blender_seconds = None
    if fbx:
        clock = time.monotonic()
        motion, fps = load_fbx_motion(fbx, session)
        blender_seconds = round(time.monotonic() - clock, 2)
        if abs(fps - obs.fps) > 0.01:
            session.info(f'Note: {fbx.name} plays at {fps:g} fps, the project at {obs.fps:g} fps; frames are compared one to one.')
    else:
        motion = load_project_motion(project, kind)
    count = len(obs.keypoints)
    first, last = max(0, motion.first), min(count, motion.first + len(motion.joints)) - 1
    if last < first:
        raise CliError(f'The motion (video frames {motion.first}-{motion.first + len(motion.joints) - 1}) does not overlap the video (0-{count - 1}).')
    if frames:
        if frames[0] < first or frames[1] > last:
            raise CliError(f'--frames {args.frames} is outside the reviewable frames {first}-{last}', EXIT_USAGE)
        first, last = frames
    if last - first + 1 < 3:
        raise CliError('Review needs at least 3 frames.', EXIT_USAGE)
    pose = None
    if args.anatomy:
        from . import anatomy
        clock = time.monotonic()
        pose = anatomy.load_pose(project, kind, fbx, session)
        blender_seconds = round((blender_seconds or 0) + time.monotonic() - clock, 2) if fbx else blender_seconds
    clock = time.monotonic()
    reference = load_project_motion(project, kind) if fbx and (project / f'{kind}_joints.npz').is_file() else None
    ctx = Context(obs, motion, first, last, reference)
    issues, floor, contacts, contacts_source = collect_issues(ctx, project, kind, pose)
    comparison = None
    if args.compare:
        from .compare import review_against
        session.info(f'Reviewing the {args.compare} layer for the comparison ...')
        comparison = review_against(obs, project, kind, args.compare, first, last, issues, pose, out)
    spec, status = suggested_fixes(issues)
    analysis_seconds = round(time.monotonic() - clock, 2)
    clock = time.monotonic()
    written, sheet = render(ctx, obs, issues, out / 'images', args.max_images)
    image_seconds = round(time.monotonic() - clock, 2)
    stats = statistics(ctx)
    per_frame = dict(frames=ctx.frames.tolist(), torso_px=ctx.torso, offset_px=ctx.offset,
                     limb_error_torso={limb: np.fmax.reduce(ctx.err[:, list(m)], axis=1) for limb, m in LIMBS.items()},
                     contacts=dict(left=contacts[:, 0].astype(int).tolist(), right=contacts[:, 1].astype(int).tolist()))
    report = dict(
        version=1, command='review', project=str(project), model=model, result=kind,
        motion=dict(label=motion.label, source=motion.source, kind='fbx' if fbx else 'project', first_video_frame=motion.first,
                    frames=len(motion.joints)),
        video=str(obs.video), fps=obs.fps, frames=[first, last], blender_frames=[first + FRAME_BASE, last + FRAME_BASE],
        frame_numbering='frames = video frames (0-based); blender_frames and fix frames = video frame + 1',
        floor=dict(y=float(floor), source=floor.source), contacts_source=contacts_source,
        thresholds=dict(visible_score=VISIBLE, pose_torso=POSE_THRESHOLD, contact_2d_torso=CONTACT_2D, contact_3d_m=CONTACT_3D,
                        slide_mps=SLIDE_SPEED, penetration_m=PENETRATION, jitter=dict(factor=JITTER_FACTOR, min_mps2=JITTER_MIN,
                        local_factor=JITTER_LOCAL), drift_torso=DRIFT, merge_gap=GAP, min_frames=MIN_RUN),
        issues=[{key: value for key, value in issue.items() if not key.startswith('_')} for issue in issues],
        keypoint_swaps=swap_report(ctx), stats=stats, suggested_fixes=spec, suggested_fixes_status=status,
        fix_input=str(fbx) if fbx else '<exported FBX of this result>', images=written, contact_sheet=sheet, per_frame=per_frame,
        anatomy=anatomy_block(ctx, pose), compare=comparison,
        seconds=dict(blender=blender_seconds, analysis=analysis_seconds, images=image_seconds, total=round(time.monotonic() - started, 2)))
    report = rounded(report)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'suggested_fix.json').write_text(json.dumps(spec, indent=1), encoding='utf-8')
    (out / 'summary.md').write_text(summary_markdown(report), encoding='utf-8')
    for issue in report['issues'][:5]:
        session.info(f"  {issue['id']} [{issue['severity']}] video {issue['frames'][0]}-{issue['frames'][1]}: {issue['message']}")
    session.say(f"Review: {len(report['issues'])} issue(s); summary: {out / 'summary.md'}")
    if comparison:
        counts, identity = report['compare']['issues']['counts'], report['compare']['identity'].get('ok')
        session.say(f"Compare with {report['compare']['reference']}: {counts['fixed']} fixed, {counts['new']} new, {counts['unchanged']} unchanged"
                    + ('' if identity is None else f"; untouched frames {'identical' if identity else 'CHANGED'}"))
    summary.update(summary=str(out / 'summary.md'), report_path=str(out / 'report.json'), report=report)
    return 0, summary
