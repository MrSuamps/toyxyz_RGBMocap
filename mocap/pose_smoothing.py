"""Pose smoothing: per-interval temporal smoothing of local joint rotations, for jitter flagged by review.

Each frame's rotation moves to the Gaussian mean of its neighbours, taken in the tangent space of the frame itself
(log of q_t^-1 q_t+k, hemisphere-aligned), then blended in with smoothstep ease frames outside the interval. Runs last
in correction.run, after self-contacts, so it also reaches jitter their ease adds; a row over a contact can loosen it
by the smoothing change. The root (joint 0) is never touched, and leg joints are held still while their foot is locked.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

METHOD = 'Pose smoothing: log-space Gaussian of local rotations with smoothstep ease'
LEG_JOINTS = ((1, 4, 7, 10), (2, 5, 8, 11))
LEGS = (1, 2, 4, 5, 7, 8)
ARMS = (13, 14, 16, 17, 18, 19, 20, 21)
LOCK_EASE = 4


def joint_ids(joints, hand_written=False):
    """SMPL-X joint ids of a smoothing row; 'all-written' is what correction.run writes back (legs, arms with the hand lock)."""
    from .mixamo_rig import BODY_NAMES
    from .self_contact_store import smoothing_bones
    if joints == 'all-written':
        return list(LEGS + (ARMS if hand_written else ()))
    return sorted(BODY_NAMES.index(name) for name in smoothing_bones(joints) if name != 'Hips')


def as_rotation(q):
    return Rotation.from_quat(np.asarray(q, float)[..., [1, 2, 3, 0]].reshape(-1, 4))


def as_wxyz(r, shape):
    return r.as_quat()[:, [3, 0, 1, 2]].reshape(shape)


def smooth_series(q, window):
    """Smoothed (m, 4) wxyz rotations; samples beyond the ends repeat the first / last frame."""
    q = np.asarray(q, float); m = len(q)
    radius, sigma = window // 2, window / 4.
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-offsets ** 2 / (2 * sigma * sigma)); weights /= weights.sum()
    centre = as_rotation(q)
    tangent = np.zeros((m, 3))
    for k, w in zip(offsets, weights):
        neighbour = as_rotation(q[np.clip(np.arange(m) + k, 0, m - 1)])
        # Rotation.as_rotvec is the shortest log, so q and -q (hemispheres) give the same tangent vector.
        tangent += w * (centre.inv() * neighbour).as_rotvec()
    return as_wxyz(centre * Rotation.from_rotvec(tangent), (m, 4))


def angular_acceleration(q, fps):
    """(m,) magnitude of the second difference of a rotation series (deg/s^2); 0 at the ends."""
    r = as_rotation(q); m = len(q)
    out = np.zeros(m)
    if m < 3:
        return out
    velocity = (r[:-1].inv() * r[1:]).as_rotvec()
    out[1:-1] = np.rad2deg(np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)) * fps * fps
    return out


def solve_interval(out_quats, fps, entry, interval, joints, foot_ranges):
    """Smooth one interval; returns ([(joint, start, end, values)], report entry)."""
    from .limb_placement import ease_weights, lock_gate, slerp
    from .foot_articulation import runs
    started = time.perf_counter(); n = len(out_quats); a, b = interval
    window = int(entry['window_frames']); ease = window
    lo, hi = max(0, a - ease), min(n, b + ease)
    report = dict(id=entry['id'], joints=entry['joints'], joint_ids=list(joints), window_frames=window, start=int(a), end=int(b),
                  window=[int(lo), int(hi)], ease_frames=int(ease))
    pad = window // 2
    s0, s1 = max(0, lo - pad), min(n, hi + pad)
    source = np.asarray(out_quats[s0:s1], float)
    base = ease_weights(lo, hi, a, b, ease)
    updates, change, before_acc, after_acc, held = [], {}, {}, {}, 0
    from .mixamo_rig import BODY_NAMES
    for j in joints:
        weights = base.copy()
        side = next((s for s, ids in enumerate(LEG_JOINTS) if j in ids), None)
        if side is not None:
            locked = np.zeros(hi - lo, bool)
            for s, e in foot_ranges[side]:
                locked[max(s, lo) - lo:max(min(e, hi) - lo, 0)] = True
            held += int(locked[a - lo:b - lo].sum())
            weights *= lock_gate(locked, LOCK_EASE)
        weights[weights < 1e-6] = 0.
        if not (weights > 0).any():
            continue
        smoothed = smooth_series(source[:, j], window)[lo - s0:hi - s0]
        current = source[lo - s0:hi - s0, j]
        values = slerp(current, smoothed, weights)
        after = source.copy()
        # Only rows with a positive weight are written: a zero-weight slerp still renormalises and would change bits.
        for first, last in runs(np.flatnonzero(weights > 0)):
            updates.append((j, lo + first, lo + last, values[first:last]))
            after[lo - s0 + first:lo - s0 + last, j] = values[first:last]
        name = BODY_NAMES[j]
        change[name] = float(np.rad2deg((as_rotation(current).inv() * as_rotation(values)).magnitude()).max())
        span = slice(a - s0, b - s0)
        before_acc[name] = float(angular_acceleration(source[:, j], fps)[span].max())
        after_acc[name] = float(angular_acceleration(after[:, j], fps)[span].max())
    report.update(rotation_change_max_deg=change, angular_accel_max_deg_s2=dict(before=before_acc, after=after_acc),
                  leg_frames_held_by_foot_locks=held, seconds=time.perf_counter() - started)
    return updates, report


def apply(out_quats, fps, entries, foot_ranges, hand_written=False, emit=lambda *a, **k: None):
    """Smooth enabled rows in order, mutating out_quats; None when nothing is enabled."""
    work = [(e, r) for e in entries if e.get('enabled', True) for r in e['ranges']]
    if not work:
        return None
    started = time.perf_counter()
    reports, changed = [], []
    for index, (entry, interval) in enumerate(work):
        joints = joint_ids(entry['joints'], hand_written)
        updates, report = solve_interval(out_quats, fps, entry, interval, joints, foot_ranges)
        reports.append(report)
        spans = {}
        for j, s, e, values in updates:
            out_quats[s:e, j] = values.astype(out_quats.dtype)
            spans.setdefault((s, e), []).append(int(j))
        changed.extend(dict(joints=sorted(js), start=int(s), end=int(e)) for (s, e), js in sorted(spans.items()))
        emit('progress', stage='correct', current=index + 1, total=len(work), phase='smoothing', message='Motion Correction: smoothing')
    return dict(method=METHOD, entries=reports, changed=changed, seconds=time.perf_counter() - started, warning='')


def from_store(project, base, context, out_quats, fps, segments, hand_written=False, emit=lambda *a, **k: None):
    """correction.run hook after self-contacts; None (nothing touched) without smoothing rows."""
    from .self_contact_store import read_smoothing
    entries, revision, warning = read_smoothing(project, base, context)
    if not entries and not warning:
        return None
    foot_ranges = [[(s['start'], s['end']) for s in segments if s['foot'] == f] for f in (0, 1)]
    report = apply(out_quats, fps, entries, foot_ranges, hand_written, emit) or dict(method=METHOD, entries=[], changed=[], warning='')
    report['revision'] = revision
    report['warning'] = ' '.join(filter(None, (report['warning'], 'Pose smoothing: ' + warning if warning else '')))
    return report
