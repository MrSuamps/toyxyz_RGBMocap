"""Per-limb 2D placement error for `review`, and per limb placement interval the error before and after placement.

Before / solved come from the `placement` block of `<kind>_correction.json` (the solver's own measure on the pose it
placed); final is review's aligned error on the reviewed motion, so later passes (self-contacts, smoothing) that pull a
placed limb away from the video show up as final > solved. Torso units, as review's pose_error. numpy loads lazily.
"""
from __future__ import annotations

from pathlib import Path

LIMB_OF_CHAIN = {'LeftArm': 'left_arm', 'RightArm': 'right_arm', 'LeftLeg': 'left_leg', 'RightLeg': 'right_leg'}
CHAIN_JOINTS = {'LeftArm': (16, 18, 20), 'RightArm': (17, 19, 21), 'LeftLeg': (1, 4, 7), 'RightLeg': (2, 5, 8)}   # SMPL
KEPT_TORSO = 0.1               # final median may exceed the solved median by this before the interval counts as undone


def describe(values):
    import numpy as np
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(median=None, p90=None, max=None, samples=0)
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)), max=float(values.max()), samples=int(len(values)))


def limb_series(ctx):
    """{limb: (frames,) worst aligned joint error of the limb (torso units)}, as review's pose_error uses."""
    import numpy as np
    from .review import LIMBS
    return {limb: np.fmax.reduce(ctx.err[:, list(members)], axis=1) for limb, members in LIMBS.items()}


def correction_block(project, kind, name):
    path = Path(project) / f'{kind}_correction.json'
    if not path.is_file():
        return None
    try:
        from .pipeline import read_json
        return read_json(path).get(name)
    except (OSError, ValueError, AttributeError):
        return None


def placement_report(project, kind):
    return correction_block(project, kind, 'placement')


def contact_moved(changed, chain, start, end):
    """True when a self-contact changed joints of the placed chain inside [start, end) (it is solved on the placed limb)."""
    joints = set(CHAIN_JOINTS.get(chain, ()))
    return any(joints & set(row.get('joints', ())) and row.get('start', 0) < end and row.get('end', 0) > start for row in changed or ())


def statistics(ctx, project, kind):
    """{'limbs': {limb: stats}, 'intervals': [...]} for the reviewed frames; intervals only when the result has placement rows."""
    series = limb_series(ctx)
    out = dict(limbs={limb: describe(values) for limb, values in series.items()}, intervals=[])
    changed = (correction_block(project, kind, 'self_contact') or {}).get('changed')
    for entry in (placement_report(project, kind) or {}).get('entries', []):
        row = dict(id=entry.get('id'), chain=entry.get('chain'), frames=[entry.get('start'), (entry.get('end') or 1) - 1])
        if 'skipped' in entry:
            out['intervals'].append(dict(row, skipped=entry['skipped']))
            continue
        mask = (ctx.frames >= entry['start']) & (ctx.frames < entry['end'])
        if not mask.any() or entry.get('chain') not in LIMB_OF_CHAIN:
            continue
        final = describe(series[LIMB_OF_CHAIN[entry['chain']]][mask])
        before = (entry.get('error_before_torso') or {}).get('limb', {})
        solved = (entry.get('error_after_torso') or {}).get('limb', {})
        kept = None
        if final['median'] is not None and solved.get('median') is not None:
            kept = final['median'] <= solved['median'] + KEPT_TORSO
        moved = kept is False and contact_moved(changed, entry['chain'], entry['start'], entry['end'])
        out['intervals'].append(dict(row, before=before, solved=solved, final=final, kept=kept, moved_by_contact=moved))
    return out


def summary_lines(block):
    """Markdown lines for summary.md (nothing without placement intervals)."""
    if not block or not block.get('intervals'):
        return []
    fmt = lambda stats: '-' if not stats or stats.get('median') is None else f"{stats['median']:.3f} / {stats['p90']:.3f}"
    lines = ['', '## Limb placement (2D error, torso; median / p90)', '', '| row | chain | frames | before | solved | final | kept |',
             '|---|---|---|---|---|---|---|']
    for row in block['intervals']:
        frames = f"{row['frames'][0]}-{row['frames'][1]}"
        if 'skipped' in row:
            lines.append(f"| {row['id']} | {row['chain']} | {frames} | skipped: {row['skipped']} | | | |")
            continue
        kept = ('-' if row['kept'] is None else 'yes' if row['kept'] else 'moved by a self-contact on this limb (check the sheet)'
                if row.get('moved_by_contact') else 'NO: a later pass moved the limb off the video')
        lines.append(f"| {row['id']} | {row['chain']} | {frames} | {fmt(row['before'])} | {fmt(row['solved'])} | {fmt(row['final'])} | {kept} |")
    return lines
