"""Corrected vs reference motion: issue diff, untouched-frame identity and a per-frame joint-rotation delta strip.

Reference layers: `original` (the uncorrected base result) and `before-selfcontact` (`<kind>_before_selfcontact.pt`,
written by `correct` when self-contacts changed the pose). numpy loads lazily so `review` stays cheap to import.
"""
from __future__ import annotations

from pathlib import Path

from .pipeline import CliError, read_json

AGAINST = ('original', 'before-selfcontact')
SEVERITY = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}
IDENTITY_TOLERANCE = 0.0
# Arm joints per side, for deciding which hand-pose changes a `changed` entry allows.
ARM_JOINTS = {'left_hand_pose': {13, 16, 18, 20}, 'right_hand_pose': {14, 17, 19, 21}}
DELTA_EPSILON_DEG = 1e-3
STALE_BEFORE = '{name} is outdated (the corrected result or its report changed since `correct` wrote it).'


# Inputs -------------------------------------------------------------------------------------------------------------

def reference(project, kind, against):
    """(path, key, label, reference kind or None) of the reference layer for a corrected result."""
    from ..motion_tools import RESULTS, base_result
    project = Path(project)
    if against not in AGAINST:
        raise CliError(f'--compare expects one of {", ".join(AGAINST)}', 2)
    if not kind.startswith('corrected_'):
        raise CliError(f'--compare needs a corrected result; {kind} is the original motion itself.')
    if against == 'original':
        base = base_result(kind)
        filename, key = RESULTS[base]
        return project / filename, key, base, base
    from ..self_contact_store import before_current, before_path
    path = before_path(project, kind)
    if not path.is_file():
        raise CliError(f'{path.name} does not exist: `correct` writes it only when self-contacts changed the motion. '
                       'Use --compare original, or add contacts and run `correct`.')
    if not before_current(project, kind):
        raise CliError(STALE_BEFORE.format(name=path.name) + ' Use --compare original, or run `correct` to refresh it.')
    return path, 'smpl_params_global', f'{kind} before self-contacts', None


def self_contact_report(project, kind):
    """The `self_contact` block of `<kind>_correction.json`, or None."""
    path = Path(project) / f'{kind}_correction.json'
    if not path.is_file():
        return None
    try:
        return read_json(path).get('self_contact')
    except (OSError, ValueError):
        return None


# Issue diff -----------------------------------------------------------------------------------------------------------

def brief(issue):
    keys = ('id', 'check', 'severity', 'frames', 'worst_frame', 'joints', 'message')
    return {key: issue.get(key) for key in keys if key in issue}


def overlap(a, b):
    """Inclusive frame overlap length of two issues (0 when disjoint)."""
    return max(0, min(a['frames'][1], b['frames'][1]) - max(a['frames'][0], b['frames'][0]) + 1)


def same_kind(a, b):
    return a['check'] == b['check'] and sorted(a.get('joints') or []) == sorted(b.get('joints') or [])


def issue_diff(before, after):
    """Match issues by check + joints + overlapping frames (one to one, largest overlap first).

    Returns fixed (only before), new (only after) and unchanged pairs with their severity change."""
    pairs = sorted(((overlap(a, b), i, k) for i, a in enumerate(before) for k, b in enumerate(after) if same_kind(a, b) and overlap(a, b)),
                   key=lambda item: -item[0])
    used_before, used_after, matched = set(), set(), []
    for _, i, k in pairs:
        if i in used_before or k in used_after:
            continue
        used_before.add(i)
        used_after.add(k)
        matched.append((i, k))
    unchanged = []
    for i, k in sorted(matched, key=lambda item: item[1]):
        a, b = before[i], after[k]
        change = SEVERITY[a['severity']] - SEVERITY[b['severity']]
        unchanged.append(dict(before=brief(a), after=brief(b), severity='worse' if change > 0 else 'better' if change < 0 else 'same'))
    fixed = [brief(a) for i, a in enumerate(before) if i not in used_before]
    new = [brief(b) for k, b in enumerate(after) if k not in used_after]

    def count(items):
        return {level: sum(item['severity'] == level for item in items) for level in SEVERITY}
    return dict(fixed=fixed, new=new, unchanged=unchanged,
                counts=dict(fixed=len(fixed), new=len(new), unchanged=len(unchanged), worse=sum(u['severity'] == 'worse' for u in unchanged),
                            better=sum(u['severity'] == 'better' for u in unchanged)),
                severity=dict(before=count(before), after=count(after)),
                new_high_or_medium=[item for item in new if SEVERITY[item['severity']] <= SEVERITY['medium']])


# Identity and delta -------------------------------------------------------------------------------------------------

def allowed_mask(changed, frames):
    """(frames, 22) True where a `changed` entry allows the pose to differ; entries are end-exclusive."""
    import numpy as np
    mask = np.zeros((frames, 22), bool)
    for entry in changed or ():
        start, end = max(0, int(entry['start'])), min(frames, int(entry['end']))
        for joint in entry.get('joints', ()):
            if 0 <= int(joint) < 22:
                mask[start:end, int(joint)] = True
    return mask


def frame_runs(mask):
    import numpy as np
    spans = []
    for i in np.flatnonzero(mask):
        if spans and i == spans[-1][1] + 1:
            spans[-1][1] = int(i)
        else:
            spans.append([int(i), int(i)])
    return spans


def untouched_identity(after, before, changed, tolerance=IDENTITY_TOLERANCE):
    """Check that frames (and joints) outside `changed` are identical to the before layer.

    `after` / `before` are param dicts of numpy arrays. Reports violations as inclusive frame runs per param/joint."""
    import numpy as np
    n = len(after['body_pose'])
    if len(before['body_pose']) != n:
        return dict(ok=False, error=f'frame counts differ ({len(before["body_pose"])} vs {n})')
    allowed = allowed_mask(changed, n)
    touched_frames = allowed.any(1)
    violations = []
    outside_max = 0.0
    unlisted_max = 0.0

    def report(name, joint, diff, mask):
        nonlocal outside_max, unlisted_max
        bad = (diff > tolerance) & ~mask
        if not bad.any():
            return
        out = bad & ~touched_frames
        inside = bad & touched_frames
        if out.any():
            outside_max = max(outside_max, float(diff[out].max()))
        if inside.any():
            unlisted_max = max(unlisted_max, float(diff[inside].max()))
        violations.append(dict(param=name, joint=joint, frames=frame_runs(bad), max_abs=float(diff[bad].max())))

    body = np.abs(after['body_pose'].reshape(n, 21, 3) - before['body_pose'].reshape(n, 21, 3)).max(-1)
    for j in range(1, 22):
        report('body_pose', j, body[:, j - 1], allowed[:, j])
    report('global_orient', 0, np.abs(after['global_orient'] - before['global_orient']).reshape(n, -1).max(-1), allowed[:, 0])
    report('transl', None, np.abs(after['transl'] - before['transl']).reshape(n, -1).max(-1), allowed[:, 0])
    for name, joints in ARM_JOINTS.items():
        if name in after and name in before:
            report(name, None, np.abs(after[name] - before[name]).reshape(n, -1).max(-1), allowed[:, sorted(joints)].any(1))
    for name in sorted(set(after) & set(before) - {'body_pose', 'global_orient', 'transl', *ARM_JOINTS}):
        if getattr(after[name], 'shape', ())[:1] == (n,) and after[name].shape == before[name].shape:
            report(name, None, np.abs(after[name] - before[name]).reshape(n, -1).max(-1), np.zeros(n, bool))
    return dict(ok=not violations, tolerance=tolerance, frames=n, changed_frames=int(touched_frames.sum()),
                untouched_frames=int((~touched_frames).sum()), outside_frames_identical=outside_max <= tolerance,
                unlisted_joints_identical=unlisted_max <= tolerance, outside_max_abs=outside_max, unlisted_max_abs=unlisted_max,
                violations=violations[:40], violation_count=len(violations))


def rotation_delta(after, before):
    """Per-frame max body joint rotation change (deg), the joint, root translation change (m) and hand-pose change (deg)."""
    import numpy as np
    from .smplx_body import rodrigues
    from .blender_joints import BONES
    n = len(after['body_pose'])

    def rotations(params):
        pose = np.concatenate([params['global_orient'].reshape(n, 1, 3), params['body_pose'].reshape(n, 21, 3)], 1)
        return rodrigues(pose)
    ra, rb = rotations(after), rotations(before)
    relative = np.einsum('njba,njbc->njac', rb, ra)
    sine = 0.5 * np.linalg.norm(np.stack([relative[..., 2, 1] - relative[..., 1, 2], relative[..., 0, 2] - relative[..., 2, 0],
                                          relative[..., 1, 0] - relative[..., 0, 1]], -1), axis=-1)
    angle = np.degrees(np.arctan2(sine, (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2))
    joint = angle.argmax(1)
    hands = np.zeros(n)
    for name in ARM_JOINTS:
        if name in after and name in before:
            delta = (after[name] - before[name]).reshape(n, -1, 3)
            hands = np.maximum(hands, np.degrees(np.linalg.norm(delta, axis=-1)).max(1))
    transl = np.linalg.norm((after['transl'] - before['transl']).reshape(n, 3), axis=1)
    peak = int(angle.max(1).argmax()) if n else 0
    return dict(max_deg=angle.max(1), joint=[BONES[j] for j in joint], per_joint_max_deg={BONES[j]: float(angle[:, j].max()) for j in range(22)},
                transl_m=transl, hands_deg=hands, changed=frame_runs(angle.max(1) > DELTA_EPSILON_DEG),
                peak=dict(frame=peak, deg=float(angle.max(1)[peak]) if n else 0.0, joint=BONES[int(joint[peak])] if n else None))


def strip_image(delta, path, changed=None, scale_deg=30.0, height=48):
    """PNG strip: one column per frame, brightness = max joint rotation change (saturates at scale_deg); `changed` ticks on top."""
    import cv2
    import numpy as np
    values = np.asarray(delta['max_deg'], np.float64)
    n = len(values)
    width = max(n, 1)
    level = np.clip(values / scale_deg, 0, 1)
    strip = np.zeros((height, width, 3), np.uint8)
    strip[8:, :, 2] = (255 * level)[None]
    strip[8:, :, 1] = (180 * level ** 2)[None]
    strip[8:, :, 0] = 40
    for entry in changed or ():
        strip[:6, max(0, int(entry['start'])):min(n, int(entry['end']))] = (255, 255, 255)
    zoom = max(1, 900 // width)
    image = cv2.resize(strip, (width * zoom, height * 2), interpolation=cv2.INTER_NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def compare_params(project, kind, against):
    """Identity + delta (+ self-contact residuals) for a corrected result against its reference layer."""
    import numpy as np
    from ..motion_tools import RESULTS
    from .smplx_body import load_params
    project = Path(project)
    path, key, label, ref_kind = reference(project, kind, against)
    after = load_params(project / RESULTS[kind][0], RESULTS[kind][1])
    before = load_params(path, key)
    contact = self_contact_report(project, kind) or {}
    changed = contact.get('changed') or []
    delta = rotation_delta(after, before)
    result = dict(against=against, reference=label, reference_file=path.name, result=kind, changed=changed, delta=delta)
    if against == 'before-selfcontact':
        result['identity'] = untouched_identity(after, before, changed)
    else:
        result['identity'] = dict(ok=None, note='not checked against the original: foot and hand locks change every corrected frame')
    tracks = contact.get('tracks') or []
    result['self_contact_tracks'] = [{k: t.get(k) for k in ('id', 'effector', 'target', 'interval', 'residual_max_m', 'residual_median_m',
                                                            'palm_to_target_joint_median_m', 'elbows', 'reciprocal', 'goal', 'skin') if k in t}
                                      for t in tracks]
    result['delta_summary'] = dict(changed_runs=delta['changed'][:40], peak=delta['peak'],
                                   frames_changed=int((np.asarray(delta['max_deg']) > DELTA_EPSILON_DEG).sum()),
                                   transl_max_m=float(np.max(delta['transl_m'])) if len(delta['transl_m']) else 0.0,
                                   hands_max_deg=float(np.max(delta['hands_deg'])) if len(delta['hands_deg']) else 0.0)
    return result, after, before, ref_kind


# Review integration ---------------------------------------------------------------------------------------------------

def reference_motion(project, before_params, label, with_pose=True, kind=None):
    """(review.Motion, anatomy Pose or None) of the reference layer, both from FK of its params (= its joints file).

    kind names the reviewed result, whose self-contact tracks the reference's skin-gap check measures too."""
    from . import review
    from .anatomy import pose_from_state
    from .smplx_body import Body, model_path
    body = Body(model_path(project))
    pose = pose_from_state(body.pose(before_params), 0, 'smplx', body, project, kind)
    return review.Motion(pose.joints, 0, label, label), (pose if with_pose else None)


def review_against(obs, project, kind, against, first, last, issues_after, pose_after=None, out=None):
    """Run review's checks on the reference layer over the same frames and diff them with `issues_after`.

    Returns the `compare` block for report.json (issue diff, identity, delta strip, self-contact residuals)."""
    import numpy as np
    from . import anatomy, review
    project = Path(project)
    result, after, before, ref_kind = compare_params(project, kind, against)
    motion, pose_ref = reference_motion(project, before, result['reference'], pose_after is not None, kind)
    ctx = review.Context(obs, motion, first, last)
    floor_kind = ref_kind or kind
    if hasattr(review, 'collect_issues'):
        issues_before = review.collect_issues(ctx, project, floor_kind, pose_ref)[0]
    else:
        floor = review.floor_level(ctx, project, floor_kind)
        contacts, _ = review.contact_mask(ctx, project, floor_kind, floor)
        issues_before = (review.check_missed_contacts(ctx) + review.check_pose(ctx) + review.check_foot_slide(ctx, contacts) +
                         review.check_penetration(ctx, contacts, floor) + review.check_jitter(ctx) + review.check_drift(ctx))
        if pose_ref is not None:
            issues_before += anatomy.checks(ctx, pose_ref)
        anatomy.rank(issues_before)
    result['issues'] = issue_diff(issues_before, issues_after)
    delta = result.pop('delta')
    span = slice(first, last + 1)
    result['per_frame'] = dict(frames=list(range(first, last + 1)), max_rotation_delta_deg=np.asarray(delta['max_deg'])[span],
                               joint=delta['joint'][span], transl_delta_m=np.asarray(delta['transl_m'])[span],
                               hands_delta_deg=np.asarray(delta['hands_deg'])[span])
    result['per_joint_max_deg'] = delta['per_joint_max_deg']
    if out is not None:
        result['strip_image'] = strip_image(delta, Path(out) / 'images' / f'compare_delta_{against}.png', result['changed'])
    return result


def summary_lines(block):
    """Markdown lines for summary.md."""
    if not block:
        return []
    issues, identity, delta = block['issues'], block['identity'], block['delta_summary']
    c = issues['counts']
    lines = ['', f"## Compare with {block['reference']} (`{block['reference_file']}`)", '',
             f"- Issues: {c['fixed']} fixed, {c['new']} new, {c['unchanged']} unchanged ({c['worse']} worse, {c['better']} better).",
             f"- Severity before: {fmt_counts(issues['severity']['before'])}; after: {fmt_counts(issues['severity']['after'])}."]
    if identity.get('ok') is None:
        lines.append(f"- Untouched-frame identity: {identity.get('note')}.")
    else:
        lines.append(f"- Untouched-frame identity: {'OK' if identity['ok'] else 'FAILED'} ({identity.get('untouched_frames')} untouched frames"
                     f", outside max |d| {identity.get('outside_max_abs', 0):.2e}, unlisted joints max |d| {identity.get('unlisted_max_abs', 0):.2e}).")
    peak = delta['peak']
    lines.append(f"- Rotation delta: {delta['frames_changed']} frames changed, peak {peak['deg']:.1f} deg at frame {peak['frame']} "
                 f"({peak['joint']}); root moves up to {delta['transl_max_m'] * 100:.1f} cm; hands up to {delta['hands_max_deg']:.1f} deg.")
    if block.get('strip_image'):
        lines.append(f"- Delta strip: `{Path(block['strip_image']).name}` (brightness = max joint rotation change, white ticks = changed ranges).")
    for track in block.get('self_contact_tracks', []):
        residual, skin = track.get('residual_max_m'), track.get('skin') or {}
        lines.append(f"- Self-contact {track.get('id')}: {track.get('effector')} -> {track.get('target')}, residual max "
                     + (f'{residual * 100:.1f} cm' if isinstance(residual, (int, float)) else 'n/a')
                     + (f", hand inside the target up to {skin['penetration_max_m'] * 100:.1f} cm, skin gap median "
                        f"{skin['skin_gap_median_m'] * 100:.1f} cm" if skin.get('skin_gap_median_m') is not None else ''))
    for title, items in (('New', issues['new']), ('Fixed', issues['fixed'])):
        for item in items[:12]:
            lines.append(f"  - {title}: {item.get('id')} [{item['severity']}] {item['check']} {item['frames'][0]}-{item['frames'][1]}: {item.get('message')}")
    return lines


def fmt_counts(counts):
    return ', '.join(f'{count} {level}' for level, count in counts.items() if count) or 'none'
