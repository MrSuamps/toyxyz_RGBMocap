"""`contacts` command group: edit the project's self-contact tracks (a hand held on a body part) and foot articulation, the same file the GUI edits.

Command-line frames are inclusive A-B result frames (as in `review`); the store keeps zero-based end-exclusive intervals.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .pipeline import MODELS, CliError, current_kind, read_json, resolve_project

EXIT_USAGE = 2
SEVERITIES = ('high', 'medium', 'low')
STALE = 'The corrected result is now stale: run `correct --project {project}` to apply the self-contacts.'
STALE_CLEARED = 'The corrected result is now stale: run `correct --project {project}` to update it.'

DESCRIPTION = '''Self-contacts hold a hand (effector) on another body part (target) over frame intervals; `correct` applies them.
Tracks live in <model project>/selfcontact_<base>_*.json, shared with the GUI. Frames on the command line are inclusive
result frames A-B (the same numbers `review` reports), several separated by commas: 345-420,500-510.
`contacts feet` edits foot articulation (pointe, flat, flexed, match per foot interval), `contacts place` limb placement
(fit an arm or leg to the video's 2D keypoints) and `contacts smooth` pose smoothing, all stored in the same files.'''
FEET_DESCRIPTION = '''Foot articulation poses one foot's ankle and toes over frame intervals; `correct` applies it before the foot locks.
Modes: pointe (foot in line with the shin, support on the toe box), flat (sole level), flexed (dorsiflexed), match
(fit the ankle to the video's 2D foot direction; ExMR only). One enabled mode per foot and frame. Frames are inclusive A-B.'''
FOOT_MODES = ('pointe', 'flat', 'flexed', 'match')
FEET_STALE = 'The corrected result is now stale: run `correct --project {project}` to apply the foot articulation.'
PLACE_DESCRIPTION = '''Limb placement ("match 2D") fits one arm or leg chain to the video's 2D keypoints over frame intervals (ExMR);
`correct` applies it after the hand lock and before the self-contacts, which are solved on the placed limb. The hand or
foot keeps its world rotation; depth stays near the source. One enabled row per chain and frame. Frames are inclusive A-B.'''
SMOOTH_DESCRIPTION = '''Pose smoothing calms rotation jitter of a joint group or bone list over frame intervals; `correct` applies it last,
after the self-contacts. Groups: arms, legs, spine, all-written (what correction writes: the legs, plus the arms when the
hand lock moved them). --window is the Gaussian window in frames (3-61, default 5; keep it small). Leg joints stay put
while their foot is locked. Enabled rows sharing a bone may not overlap. Frames are inclusive A-B.'''
# Pose row sections edited like `feet`: subcommand, singular noun, plural noun, stale message tail.
ROWS = {'articulation': ('feet', 'foot articulation entry', 'foot articulation entries', 'the foot articulation'),
        'placement': ('place', 'limb placement row', 'limb placement rows', 'the limb placement'),
        'smoothing': ('smooth', 'pose smoothing row', 'pose smoothing rows', 'the pose smoothing')}


def register(commands, common):
    parser = commands.add_parser('contacts', parents=[common], help='list, edit and suggest hand-to-body self-contacts',
                                 description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest='contacts_command', metavar='ACTION', required=True)
    target = argparse.ArgumentParser(add_help=False)
    target.add_argument('--project', required=True, metavar='P', help='workspace root, model project or person take')
    target.add_argument('--model', choices=MODELS, help='model inside a workspace root (default: its active model)')
    settings = argparse.ArgumentParser(add_help=False)
    group = settings.add_argument_group('track settings')
    group.add_argument('--distance', metavar='M', help='palm-to-target distance in metres, or auto')
    group.add_argument('--offset', metavar='X,Y,Z', help='contact point in the target joint frame (metres), or auto')
    group.add_argument('--reciprocal', choices=('off', 'auto', 'always'), help='move the target limb too: off, auto (when out of reach), always')
    group.add_argument('--grip', choices=('off', 'auto'), help='orient the palm and wrap the fingers around the target')
    parents = [common, target]

    p = actions.add_parser('list', parents=parents, help='show tracks, ranges, settings and last residuals')
    p.set_defaults(handler=cmd_list)

    p = actions.add_parser('add', parents=parents + [settings], help='add a track')
    p.add_argument('--effector', required=True, type=bone, metavar='HAND', help='LeftHand or RightHand')
    p.add_argument('--target', required=True, type=bone, metavar='BONE', help='body bone to touch, e.g. RightFoot, LeftLeg, Head')
    p.add_argument('--frames', required=True, metavar='A-B[,C-D]', help='inclusive frame intervals')
    p.add_argument('--id', metavar='ID', help='track id (default from the pair, e.g. rh-rf)')
    p.add_argument('--disabled', action='store_true', help='add the track switched off')
    p.set_defaults(handler=cmd_add)

    p = actions.add_parser('edit', parents=parents + [settings], help='change a track')
    p.add_argument('--id', required=True, metavar='ID')
    p.add_argument('--frames', metavar='A-B[,C-D]', help='replace the intervals')
    p.add_argument('--add-frames', dest='add_frames', metavar='A-B[,C-D]', help='add intervals')
    p.add_argument('--remove-frames', dest='remove_frames', metavar='A-B[,C-D]', help='remove these frames from the intervals')
    p.add_argument('--accept', action='store_true', help='add the track\'s suggested (automatic) intervals to its ranges')
    p.add_argument('--target', type=bone, metavar='BONE', help='new target bone')
    switch = p.add_mutually_exclusive_group()
    switch.add_argument('--enable', dest='enabled', action='store_const', const=True, help='switch the track on')
    switch.add_argument('--disable', dest='enabled', action='store_const', const=False, help='switch the track off')
    p.set_defaults(handler=cmd_edit)

    p = actions.add_parser('remove', parents=parents, help='remove a track, or all self-contacts')
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument('--id', metavar='ID')
    which.add_argument('--all', action='store_true',
                       help='remove every self-contact track (foot, placement and smoothing rows are kept unless the files are for other motion)')
    p.set_defaults(handler=cmd_remove)

    p = actions.add_parser('suggest', parents=parents, help='find missed and sunk hand contacts and store suggested intervals',
                           description='Run the `review` missed-contact check (hands touching in the video but apart in 3D) and find '
                                       'hands resting inside the body (sustained, slow penetration of the SMPL-X mesh), and store '
                                       'each as a track\'s suggested (automatic) intervals. Suggestions alone do not make the corrected '
                                       'result stale. Every run replaces all suggestions; --accept also adds them to the tracks\' ranges, '
                                       'skipping frames another enabled track already holds with the same hand.')
    p.add_argument('--accept', action='store_true', help='also copy the suggestions into the ranges that `correct` applies')
    p.add_argument('--min-severity', dest='min_severity', choices=SEVERITIES, default='low', help='ignore weaker issues (default low = all)')
    p.add_argument('--result', metavar='KIND', help='result to review (default: the uncorrected base, before any self-contact)')
    p.set_defaults(handler=cmd_suggest)

    feet = actions.add_parser('feet', parents=[common], help='list and edit foot articulation (pointe, flat, flexed, match)',
                              description=FEET_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    foot_actions = feet.add_subparsers(dest='feet_command', metavar='ACTION', required=True)
    shape = argparse.ArgumentParser(add_help=False)
    group = shape.add_argument_group('articulation settings')
    group.add_argument('--strength', type=float, metavar='S', help='blend from the source foot (0) to the full mode (1); default 1')
    group.add_argument('--profile', choices=('default', 'dancer'), help='joint limits: default, or dancer (plantarflexion up to 90 deg)')

    p = foot_actions.add_parser('list', parents=parents, help='show foot articulation entries')
    p.set_defaults(handler=cmd_feet_list)

    p = foot_actions.add_parser('add', parents=parents + [shape], help='add an entry')
    p.add_argument('--foot', required=True, type=foot, metavar='FOOT', help='LeftFoot or RightFoot')
    p.add_argument('--mode', required=True, choices=FOOT_MODES, help='pointe, flat, flexed, or match (fit the video\'s 2D foot)')
    p.add_argument('--frames', required=True, metavar='A-B[,C-D]', help='inclusive frame intervals')
    p.add_argument('--id', metavar='ID', help='entry id (default from foot and mode, e.g. lf-pointe)')
    p.add_argument('--disabled', action='store_true', help='add the entry switched off')
    p.set_defaults(handler=cmd_feet_add)

    p = foot_actions.add_parser('edit', parents=parents + [shape], help='change an entry')
    p.add_argument('--id', required=True, metavar='ID')
    p.add_argument('--foot', type=foot, metavar='FOOT', help='LeftFoot or RightFoot')
    p.add_argument('--mode', choices=FOOT_MODES)
    p.add_argument('--frames', metavar='A-B[,C-D]', help='replace the intervals')
    p.add_argument('--add-frames', dest='add_frames', metavar='A-B[,C-D]', help='add intervals')
    p.add_argument('--remove-frames', dest='remove_frames', metavar='A-B[,C-D]', help='remove these frames from the intervals')
    switch = p.add_mutually_exclusive_group()
    switch.add_argument('--enable', dest='enabled', action='store_const', const=True, help='switch the entry on')
    switch.add_argument('--disable', dest='enabled', action='store_const', const=False, help='switch the entry off')
    p.set_defaults(handler=cmd_feet_edit)

    p = foot_actions.add_parser('remove', parents=parents, help='remove an entry, or all foot articulation')
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument('--id', metavar='ID')
    which.add_argument('--all', action='store_true', help='remove every foot articulation entry (self-contact tracks are kept)')
    p.set_defaults(handler=cmd_feet_remove)

    def row_parsers(name, key, help_text, description, example):
        """list/add/edit/remove parsers of one pose row section; returns the add and edit parsers for the section's settings."""
        section = actions.add_parser(name, parents=[common], help=help_text, description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        sub = section.add_subparsers(dest=name + '_command', metavar='ACTION', required=True)
        _, singular, plural, _ = ROWS[key]
        sub.add_parser('list', parents=parents, help=f'show {plural}').set_defaults(handler=cmd_rows_list, rows=key)
        add = sub.add_parser('add', parents=parents, help='add a row')
        add.add_argument('--frames', required=True, metavar='A-B[,C-D]', help='inclusive frame intervals')
        add.add_argument('--id', metavar='ID', help=f'row id (default from the row, e.g. {example})')
        add.add_argument('--disabled', action='store_true', help='add the row switched off')
        add.set_defaults(handler=cmd_rows_add, rows=key)
        edit = sub.add_parser('edit', parents=parents, help='change a row')
        edit.add_argument('--id', required=True, metavar='ID')
        edit.add_argument('--frames', metavar='A-B[,C-D]', help='replace the intervals')
        edit.add_argument('--add-frames', dest='add_frames', metavar='A-B[,C-D]', help='add intervals')
        edit.add_argument('--remove-frames', dest='remove_frames', metavar='A-B[,C-D]', help='remove these frames from the intervals')
        switch = edit.add_mutually_exclusive_group()
        switch.add_argument('--enable', dest='enabled', action='store_const', const=True, help='switch the row on')
        switch.add_argument('--disable', dest='enabled', action='store_const', const=False, help='switch the row off')
        edit.set_defaults(handler=cmd_rows_edit, rows=key)
        remove = sub.add_parser('remove', parents=parents, help=f'remove a row, or all {plural}')
        which = remove.add_mutually_exclusive_group(required=True)
        which.add_argument('--id', metavar='ID')
        which.add_argument('--all', action='store_true', help=f'remove every {singular} (everything else is kept)')
        remove.set_defaults(handler=cmd_rows_remove, rows=key)
        return add, edit

    add, edit = row_parsers('place', 'placement', 'list and edit limb placement (fit an arm or leg to the video\'s 2D keypoints)',
                            PLACE_DESCRIPTION, 'rl-place')
    for p in (add, edit):
        group = p.add_argument_group('placement settings')
        group.add_argument('--chain', required=p is add, type=chain, metavar='CHAIN', help='LeftArm, RightArm, LeftLeg or RightLeg')
        group.add_argument('--strength', type=float, metavar='S', help='blend from the source limb (0) to the fitted one (1); default 1')

    add, edit = row_parsers('smooth', 'smoothing', 'list and edit pose smoothing (calm rotation jitter)', SMOOTH_DESCRIPTION, 'smooth-arms')
    for p in (add, edit):
        p.add_argument('--window', type=int, metavar='N', help='Gaussian window in frames, 3-61; default 5')
        joints = p.add_mutually_exclusive_group()
        joints.add_argument('--joints', type=bone_list, metavar='BONE[,BONE]', help='bones to smooth, e.g. LeftArm,LeftForeArm (not Hips)')
        joints.add_argument('--group', choices=('arms', 'legs', 'spine', 'all-written'),
                            help='bone group to smooth (add default all-written: what correction writes)')


# Parsing ------------------------------------------------------------------------------------------------------------

def bone(value):
    from ..mixamo_rig import BODY_NAMES
    names = {name.casefold(): name for name in BODY_NAMES}
    try:
        return names[str(value).casefold()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'unknown bone {value!r}; use one of {", ".join(BODY_NAMES)}') from None


def foot(value):
    from ..self_contact_store import FEET
    names = {name.casefold(): name for name in FEET}
    try:
        return names[str(value).casefold()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'unknown foot {value!r}; use {" or ".join(FEET)}') from None


def chain(value):
    from ..self_contact_store import CHAINS
    names = {name.casefold(): name for name in CHAINS}
    try:
        return names[str(value).casefold()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'unknown chain {value!r}; use one of {", ".join(CHAINS)}') from None


def bone_list(value):
    return [bone(name) for name in str(value).replace(' ', '').split(',') if name]


def parse_frames(value, frames, option='--frames'):
    """'345-420,500' (inclusive) -> [(345, 421), (500, 501)]."""
    intervals = []
    for part in str(value).replace(' ', '').split(','):
        if not part:
            continue
        try:
            first, last = (int(x) for x in part.split('-', 1)) if '-' in part[1:] else (int(part), int(part))
        except ValueError:
            raise CliError(f'{option} expects inclusive frame intervals such as 345-420,500-510', EXIT_USAGE) from None
        if not 0 <= first <= last < frames:
            raise CliError(f'{option} {part}: frames must satisfy 0 <= A <= B <= {frames - 1}', EXIT_USAGE)
        intervals.append((first, last + 1))
    if not intervals:
        raise CliError(f'{option} expects at least one interval', EXIT_USAGE)
    return intervals


def inclusive(ranges):
    return ','.join(f'{start}-{end - 1}' if end - start > 1 else str(start) for start, end in ranges) or '-'


def parse_distance(value):
    if value is None or str(value).lower() == 'auto':
        return 'auto'
    try:
        return float(value)
    except ValueError:
        raise CliError('--distance expects metres or auto', EXIT_USAGE) from None


def parse_offset(value):
    if value is None or str(value).lower() == 'auto':
        return 'auto'
    try:
        values = [float(x) for x in str(value).split(',')]
    except ValueError:
        values = []
    if len(values) != 3:
        raise CliError('--offset expects x,y,z in metres or auto', EXIT_USAGE)
    return values


def subtract(ranges, removed):
    result = []
    for start, end in ranges:
        pieces = [(start, end)]
        for a, b in removed:
            pieces = [p for s, e in pieces for p in ((s, min(e, a)), (max(s, b), e)) if p[0] < p[1]]
        result += pieces
    return result


def merge(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


# Store access -------------------------------------------------------------------------------------------------------

class Store:
    """The resolved project, its result base and the context self-contacts are saved with (as the GUI saves foot locks)."""

    def __init__(self, args):
        from ..foot_contacts import BASES
        from ..motion_tools import base_result
        target = resolve_project(args.project, args.model)
        if target.project is None:
            raise CliError(f'This workspace has no {target.model} motion yet. Reconstruct it first.')
        self.project, self.model = target.project, target.model
        self.kind = current_kind(self.project, self.model)
        self.base = base_result(self.kind)
        if self.base not in BASES:
            raise CliError(f'Self-contacts are not supported for the {self.base} result.')
        stem = 'corrected_' + self.base
        self.report_path = self.project / f'{stem}_correction.json'
        sources = self.project / f'{stem}_sources.json'
        if not self.report_path.is_file() or not sources.is_file():
            where = ' Pass a person take, not the people root.' if target.people_root is not None and self.project == target.people_root else ''
            raise CliError(f'{self.project} has no corrected {self.base} result yet; run `correct --project {self.project}` first.{where}')
        try:
            self.report = read_json(self.report_path)
            self.context = {'frames': int(self.report['frames']), 'fps': float(self.report['fps']), 'sources': read_json(sources)}
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CliError(f'Cannot read {self.report_path.name}: {error}') from None
        self.frames = self.context['frames']

    def read(self, writing=False):
        from .. import self_contact_store as store
        try:
            tracks, revision, warning = store.read(self.project, self.base, self.context)
        except (OSError, ValueError) as error:
            raise CliError(f'Cannot read the self-contacts: {error}') from None
        if writing and warning:
            raise CliError(warning + ' Run `contacts remove --all` to start over.')
        return tracks, revision, warning

    def save(self, tracks, session):
        from .. import self_contact_store as store
        try:
            tracks = store.normalize_tracks(tracks, self.frames)
        except ValueError as error:
            raise CliError(str(error), EXIT_USAGE) from None
        if session.dry_run:
            return tracks, None
        if not tracks:
            store.clear(self.project, self.base)
            return tracks, None
        return tracks, store.save(self.project, self.base, tracks, self.context)

    def read_rows(self, key, writing=False):
        """(entries, revision, warning) of one pose row section: articulation, placement or smoothing."""
        from .. import self_contact_store as store
        try:
            entries, revision, warning = getattr(store, 'read_' + key)(self.project, self.base, self.context)
            _, _, timing = store.read(self.project, self.base, self.context)
        except (OSError, ValueError) as error:
            raise CliError(f'Cannot read {ROWS[key][3]}: {error}') from None
        if writing and timing:
            raise CliError((warning or timing) + ' Run `contacts remove --all` to start over (removes all saved tracks and pose rows).')
        return entries, revision, warning

    def save_rows(self, key, entries, session):
        from .. import self_contact_store as store
        try:
            entries = getattr(store, 'normalize_' + key)(entries, self.frames)
        except ValueError as error:
            raise CliError(str(error), EXIT_USAGE) from None
        if session.dry_run:
            return entries, None
        try:
            return entries, getattr(store, 'save_' + key)(self.project, self.base, entries, self.context)
        except (OSError, ValueError) as error:
            raise CliError(f'Cannot save {ROWS[key][3]}: {error}') from None

    def read_feet(self, writing=False):
        return self.read_rows('articulation', writing)

    def save_feet(self, entries, session):
        return self.save_rows('articulation', entries, session)

    def applied(self):
        from ..foot_contacts import contact_revision
        return self.report.get('contact_revision') == contact_revision(self.project, self.base)

    def summary(self, action, tracks, **extra):
        return dict(command='contacts', action=action, project=str(self.project), model=self.model, result='corrected_' + self.base,
                    base=self.base, frames=self.frames, fps=self.context['fps'],
                    frame_numbering='ranges/automatic: zero-based end-exclusive; *_inclusive: inclusive A-B as on the command line',
                    tracks=[track_json(t) for t in tracks], **extra)


def track_json(track):
    return {**track, 'ranges': [list(p) for p in track['ranges']], 'automatic': [list(p) for p in track['automatic']],
            'ranges_inclusive': inclusive(track['ranges']), 'automatic_inclusive': inclusive(track['automatic'])}


def describe(track):
    state = '' if track['enabled'] else '  (disabled)'
    offset = 'auto' if track['offset'] == 'auto' else ','.join(f'{v:g}' for v in track['offset'])
    distance = track['distance'] if track['distance'] == 'auto' else f'{track["distance"]:g} m'
    return [f'{track["id"]}: {track["effector"]} -> {track["target"]}{state}',
            f'    frames {inclusive(track["ranges"])}   suggested {inclusive(track["automatic"])}',
            f'    distance {distance}, offset {offset}, reciprocal {track["reciprocal"]}, grip {track["grip"]}, source {track["source"]}']


def describe_foot(entry):
    state = '' if entry['enabled'] else '  (disabled)'
    return [f'{entry["id"]}: {entry["foot"]} {entry["mode"]}{state}',
            f'    frames {inclusive(entry["ranges"])}   suggested {inclusive(entry["automatic"])}',
            f'    strength {entry["strength"]:g}, profile {entry["profile"]}, source {entry["source"]}']


def describe_row(key, entry):
    if key == 'articulation':
        return describe_foot(entry)
    state = '' if entry['enabled'] else '  (disabled)'
    settings = f'strength {entry["strength"]:g}' if key == 'placement' else f'window {entry["window_frames"]} frames'
    return [f'{entry["id"]}: {row_what(key, entry)}{state}', f'    frames {inclusive(entry["ranges"])}   suggested {inclusive(entry["automatic"])}',
            f'    {settings}, source {entry["source"]}']


def row_what(key, entry):
    if key == 'placement':
        return entry['chain']
    return entry['joints'] if isinstance(entry['joints'], str) else ','.join(entry['joints'])


def row_note(key, row):
    """One interval of the last correction report of a placement or smoothing row."""
    note = f'{row["start"]}-{row["end"] - 1}'
    if row.get('skipped'):
        return note + f' skipped: {row["skipped"]}'
    if key == 'placement':
        before, after = (((row.get(k) or {}).get('limb') or {}).get('median') for k in ('error_before_torso', 'error_after_torso'))
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            note += f' 2D error {before:.2f} -> {after:.2f} torso (median)'
        if row.get('foot_lock_conflict_frames'):
            note += f', foot lock held {row["foot_lock_conflict_frames"]} frame(s)'
        return note
    change = row.get('rotation_change_max_deg')
    if isinstance(change, dict) and change:
        note += f' max change {max(change.values()):.1f} deg'
    if row.get('leg_frames_held_by_foot_locks'):
        note += f', {row["leg_frames_held_by_foot_locks"]} leg frame(s) held by foot locks'
    return note


def finish(store, session, action, tracks, revision, **extra):
    stale = not session.dry_run and not store.applied()
    if session.dry_run:
        session.say(f'Would save {len(tracks)} self-contact track(s) (dry run, nothing written):')
    for track in tracks:
        for line in describe(track):
            session.info(line)
    if stale:
        session.say(STALE.format(project=store.project))
    return 0, store.summary(action, tracks, revision=revision, stale=stale, dry_run=session.dry_run, **extra)


def find(tracks, ident):
    for track in tracks:
        if track['id'] == ident:
            return track
    raise CliError(f'No self-contact track {ident!r}. Tracks: {", ".join(t["id"] for t in tracks) or "none"}')


# Commands -----------------------------------------------------------------------------------------------------------

def cmd_list(args, session):
    store = Store(args)
    tracks, revision, warning = store.read()
    residuals = store.report.get('self_contact')
    by_id = {}
    if isinstance(residuals, dict):
        for row in residuals.get('tracks') or []:
            if isinstance(row, dict) and 'id' in row:
                by_id[row['id']] = row
    if warning:
        session.say('warning: ' + warning)
    session.say(f'{len(tracks)} self-contact track(s) on corrected_{store.base} ({store.frames} frames, frames inclusive):')
    for track in tracks:
        for line in describe(track):
            session.say(line)
        row = by_id.get(track['id'])
        if row:
            values = ', '.join(f'{k} {v:.3g}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()
                               if k != 'id' and isinstance(v, (int, float, str, bool)))
            session.say('    last residuals: ' + values)
    entries, articulated = list_feet(store, session)
    placement, placed = list_rows(store, session, 'placement')
    smoothing, smoothed = list_rows(store, session, 'smoothing')
    applied = store.applied()
    if revision is not None or not applied:
        session.say('Applied to the corrected result.' if applied else STALE.format(project=store.project))
    return 0, store.summary('list', tracks, revision=revision, applied=applied, warning=warning, residuals=residuals,
                            articulation=[track_json(e) for e in entries], articulation_report=articulated,
                            placement=[track_json(e) for e in placement], placement_report=placed,
                            smoothing=[track_json(e) for e in smoothing], smoothing_report=smoothed)


def list_rows(store, session, key):
    """Print one placement or smoothing section with its last correction notes; returns (entries, report block or None)."""
    entries, _, warning = store.read_rows(key)
    block = store.report.get(key)
    by_id = {}
    if isinstance(block, dict):
        for row in block.get('entries') or []:
            if isinstance(row, dict) and 'id' in row and isinstance(row.get('start'), int) and isinstance(row.get('end'), int):
                by_id.setdefault(row['id'], []).append(row)
    if warning:
        session.say('warning: ' + warning)
    if not entries:
        return entries, block
    _, singular, plural, _ = ROWS[key]
    session.say(f'{len(entries)} {singular if len(entries) == 1 else plural} on corrected_{store.base} (frames inclusive):')
    for entry in entries:
        for line in describe_row(key, entry):
            session.say(line)
        notes = [row_note(key, row) for row in by_id.get(entry['id'], [])]
        if notes:
            session.say('    last correction: ' + '; '.join(notes))
    return entries, block


def list_feet(store, session):
    """Print the foot articulation entries with their last correction notes; returns (entries, report block or None)."""
    entries, _, warning = store.read_feet()
    block = store.report.get('foot_articulation')
    by_id = {}
    if isinstance(block, dict):
        for row in block.get('entries') or []:
            if isinstance(row, dict) and 'id' in row:
                by_id.setdefault(row['id'], []).append(row)
    if warning:
        session.say('warning: ' + warning)
    if not entries and not warning:
        return entries, block
    session.say(f'{len(entries)} foot articulation entr{"y" if len(entries) == 1 else "ies"} on corrected_{store.base} (frames inclusive):')
    for entry in entries:
        for line in describe_foot(entry):
            session.say(line)
        notes = [f'{row["start"]}-{row["end"] - 1}' + (f' skipped: {row["skipped"]}' if row.get('skipped') else '')
                 + (f' joint limit on {row["limited_frames"]} frame(s)' if row.get('limited_frames') else '')
                 + (f', eased over {row["window"][0]}-{row["window"][1] - 1}' if row.get('window') else '')
                 for row in by_id.get(entry['id'], []) if isinstance(row.get('start'), int) and isinstance(row.get('end'), int)]
        if notes:
            session.say('    last correction: ' + '; '.join(notes))
    return entries, block


def track_settings(args, track):
    if args.distance is not None:
        track['distance'] = parse_distance(args.distance)
    if args.offset is not None:
        track['offset'] = parse_offset(args.offset)
    for key in ('reciprocal', 'grip'):
        if getattr(args, key) is not None:
            track[key] = getattr(args, key)


def cmd_add(args, session):
    store = Store(args)
    tracks, _, _ = store.read(True)
    track = dict(effector=args.effector, target=args.target, ranges=parse_frames(args.frames, store.frames),
                 enabled=not args.disabled, source='user')
    if args.id:
        track['id'] = args.id
    track_settings(args, track)
    before = {t['id'] for t in tracks}
    tracks, revision = store.save(tracks + [track], session)
    added = next(t for t in tracks if t['id'] not in before)
    verb = 'Would add' if session.dry_run else 'Added'
    session.say(f'{verb} self-contact {added["id"]}: {added["effector"]} -> {added["target"]}, frames {inclusive(added["ranges"])}')
    return finish(store, session, 'add', tracks, revision, id=added['id'])


def cmd_edit(args, session):
    store = Store(args)
    tracks, _, _ = store.read(True)
    track = find(tracks, args.id)
    if args.frames is not None:
        track['ranges'] = parse_frames(args.frames, store.frames)
    if args.add_frames is not None:
        track['ranges'] = merge(track['ranges'] + parse_frames(args.add_frames, store.frames, '--add-frames'))
    if args.accept:
        track['ranges'] = merge(track['ranges'] + track['automatic'])
    if args.remove_frames is not None:
        track['ranges'] = subtract(track['ranges'], parse_frames(args.remove_frames, store.frames, '--remove-frames'))
    if args.target is not None:
        track['target'] = args.target
    if args.enabled is not None:
        track['enabled'] = args.enabled
    track_settings(args, track)
    tracks, revision = store.save(tracks, session)
    session.say(f'{"Would update" if session.dry_run else "Updated"} self-contact {args.id}.')
    return finish(store, session, 'edit', tracks, revision, id=args.id)


def cmd_remove(args, session):
    from .. import self_contact_store
    store = Store(args)
    if args.all:
        if session.dry_run:
            session.say('Would delete every self-contact file (dry run).')
            return 0, store.summary('remove', [], dry_run=True, removed='all')
        try:
            timing = store.read()[2]
        except CliError as error:
            timing = str(error)
        # Saved data for another time base can be neither applied nor edited, so starting over removes every section.
        removed = (self_contact_store.remove_all if timing else self_contact_store.clear)(store.project, store.base)
        session.say('Removed all self-contacts.' if removed else 'There were no self-contacts.')
        for key, (command, singular, plural, _) in ROWS.items():
            kept = [] if timing else store.read_rows(key)[0]
            if kept:
                session.say(f'Kept {len(kept)} {singular if len(kept) == 1 else plural}; `contacts {command} remove --all` removes them.')
        stale = not store.applied()
        if stale:
            session.say(STALE_CLEARED.format(project=store.project))
        return 0, store.summary('remove', [], removed='all', revision=None, stale=stale, dry_run=False)
    tracks, _, _ = store.read(True)
    find(tracks, args.id)
    tracks, revision = store.save([t for t in tracks if t['id'] != args.id], session)
    session.say(f'{"Would remove" if session.dry_run else "Removed"} self-contact {args.id}.')
    return finish(store, session, 'remove', tracks, revision, removed=args.id)


def review_issues(store, kind, session, rejected=None):
    """Ranked contacts to suggest: review's missed contacts plus hands resting sunk into the body (anatomy.check_sunk_hands)."""
    from . import review
    if store.model not in review.SOURCES:
        raise CliError(f'contacts suggest uses review, which supports {", ".join(review.SOURCES)} projects; this one is {store.model}.')
    inputs, loader = review.SOURCES[store.model]
    missing = [path.name for path in inputs(store.project) if not path.is_file()]
    if missing:
        raise CliError(f'{store.project} lacks {", ".join(missing)}; reconstruct it first.')
    if not (store.project / f'{kind}_joints.npz').is_file():
        raise CliError(f'The {kind} motion has no {kind}_joints.npz; run `reconstruct` or `correct` first.')
    session.info(f'Reviewing {kind} for missed hand contacts ...')
    obs = loader(store.project, kind)
    motion = review.load_project_motion(store.project, kind)
    if len(motion.joints) != store.frames:
        raise CliError(f'{kind} has {len(motion.joints)} frames but the corrected result has {store.frames}; run `correct` first.')
    last = min(len(obs.keypoints), len(motion.joints)) - 1
    if last < 2:
        raise CliError('Review needs at least 3 frames.')
    issues = review.check_missed_contacts(review.Context(obs, motion, 0, last))
    return review.rank(issues + sunk_issues(store, kind, last, session, rejected))


def sunk_issues(store, kind, last, session, rejected=None):
    """anatomy.check_sunk_hands on the SMPL-X mesh of `kind` over frames 0-last; [] with a note when the mesh cannot be posed."""
    import numpy as np
    from . import anatomy
    session.info(f'Measuring hands inside the body on {kind} ...')
    try:
        pose, _ = anatomy.load_project_pose(store.project, kind)
    except (CliError, OSError, ValueError, KeyError, RuntimeError) as error:
        session.say(f'note: hands resting inside the body were not checked ({error}).')
        return []
    frames = np.arange(last + 1)
    return anatomy.check_sunk_hands(pose.rows(frames), frames, store.context['fps'], rejected)


def cmd_suggest(args, session):
    from ..motion_tools import RESULTS
    store = Store(args)
    kind = args.result or store.base
    if kind not in RESULTS:
        raise CliError(f'Unknown result {kind!r}', EXIT_USAGE)
    tracks, _, _ = store.read(True)
    limit = SEVERITIES.index(args.min_severity)
    rejected = []
    issues = [i for i in review_issues(store, kind, session, rejected) if SEVERITIES.index(i['severity']) <= limit]
    source = 'review:' + time.strftime('%Y-%m-%dT%H:%M')
    pairs = {}
    for issue in issues:
        key = (issue['effector'], issue['target'])
        pairs.setdefault(key, []).append(issue)
    for track in tracks:
        track['automatic'] = []
    touched = []
    for (effector, target), group in pairs.items():
        track = next((t for t in tracks if (t['effector'], t['target']) == (effector, target)), None)
        if track is None:
            track = dict(effector=effector, target=target, ranges=[], source=source)
            tracks.append(track)
        track['automatic'] = merge([(i['frames'][0], i['frames'][1] + 1) for i in group])
        touched.append((max(i['score'] for i in group), track))
    # Suggestion-only tracks from an earlier run that no longer match any issue carry nothing.
    tracks = [t for t in tracks if t['ranges'] or t['automatic'] or not str(t.get('source', '')).startswith('review:')]
    skipped = []
    if args.accept:
        for _, track in sorted(touched, key=lambda item: -item[0]):
            held = [r for t in tracks if t is not track and t.get('enabled', True) and t['effector'] == track['effector'] for r in t['ranges']]
            free = subtract(track['automatic'], held)
            lost = subtract(track['automatic'], free)
            if lost:
                skipped.append(dict(effector=track['effector'], target=track['target'], frames_inclusive=inclusive(merge(lost))))
            track['ranges'] = merge(list(track['ranges']) + free)
    tracks, revision = store.save(tracks, session)
    for issue in issues:
        session.info(f"  {issue['id']} [{issue['severity']}] {issue['effector']} -> {issue['target']} frames "
                     f"{issue['frames'][0]}-{issue['frames'][1]} ({evidence(issue)})")
    for row in rejected:
        session.info(f"  not suggested: {row['effector']} inside the {row['segment'].replace('_', ' ')} frames "
                     f"{row['frames'][0]}-{row['frames'][1]}: {row['reason']}")
    for row in skipped:
        session.say(f"  not accepted for {row['effector']} -> {row['target']}: frames {row['frames_inclusive']} (hand already held by another track)")
    verb = 'Accepted' if args.accept else 'Suggested'
    sunk = sum(i['check'] == 'sunk_hand' for i in issues)
    session.say(f'{verb} {len(issues) - sunk} missed contact(s) and {sunk} hand(s) resting inside the body from {kind} into {len(touched)} track(s).'
                + ('' if args.accept or not issues else ' Accept with `contacts suggest --accept` or `contacts edit --id ID --accept`.'))
    used = [dict(id=i['id'], check=i['check'], severity=i['severity'], effector=i['effector'], target=i['target'], frames_inclusive=i['frames'],
                 **({'depth_max_m': round(float(i['metrics']['depth_max_m']), 3)} if i['check'] == 'sunk_hand' else
                    {'distance_3d_m_median': round(float(i['metrics']['distance_3d_m_median']), 3)})) for i in issues]
    return finish(store, session, 'suggest', tracks, revision, reviewed=kind, issues=used, accepted=args.accept, skipped=skipped,
                  not_suggested=rejected)


def evidence(issue):
    m = issue['metrics']
    if issue['check'] == 'sunk_hand':
        from .anatomy import SUNK_SPEED_M_S
        how = 'during a fast turn' if m['palm_speed_m_s_median'] > SUNK_SPEED_M_S else 'resting'
        return f"{m['depth_max_m'] * 100:.1f} cm inside the {issue['joints'][1].replace('_', ' ')}, {how}"
    return f"{m['distance_3d_m_median']:.2f} m apart in 3D"


# Foot articulation --------------------------------------------------------------------------------------------------

def find_foot(entries, ident):
    for entry in entries:
        if entry['id'] == ident:
            return entry
    raise CliError(f'No foot articulation entry {ident!r}. Entries: {", ".join(e["id"] for e in entries) or "none"}')


def foot_settings(args, entry):
    for key in ('strength', 'profile'):
        if getattr(args, key) is not None:
            entry[key] = getattr(args, key)


def finish_feet(store, session, action, entries, revision, **extra):
    stale = not session.dry_run and not store.applied()
    if session.dry_run:
        session.say(f'Would save {len(entries)} foot articulation entr{"y" if len(entries) == 1 else "ies"} (dry run, nothing written):')
    for entry in entries:
        for line in describe_foot(entry):
            session.info(line)
    if store.model == 'flowhmr' and entries:
        session.say('note: FlowHMR correction does not apply foot articulation yet; the entries are kept for SMPL-X results.')
    if stale:
        session.say((STALE_CLEARED if action == 'remove' else FEET_STALE).format(project=store.project))
    return 0, store.summary('feet ' + action, store.read()[0], articulation=[track_json(e) for e in entries], revision=revision, stale=stale,
                            dry_run=session.dry_run, **extra)


def cmd_feet_list(args, session):
    store = Store(args)
    entries, block = list_feet(store, session)
    if not entries:
        session.say(f'No foot articulation on corrected_{store.base}.')
    applied = store.applied()
    if entries and not applied:
        session.say(FEET_STALE.format(project=store.project))
    return 0, store.summary('feet list', store.read()[0], articulation=[track_json(e) for e in entries], applied=applied,
                            articulation_report=block)


def cmd_feet_add(args, session):
    store = Store(args)
    entries, _, _ = store.read_feet(True)
    entry = dict(foot=args.foot, mode=args.mode, ranges=parse_frames(args.frames, store.frames), enabled=not args.disabled, source='user')
    if args.id:
        entry['id'] = args.id
    foot_settings(args, entry)
    before = {e['id'] for e in entries}
    entries, revision = store.save_feet(entries + [entry], session)
    added = next(e for e in entries if e['id'] not in before)
    session.say(f'{"Would add" if session.dry_run else "Added"} foot articulation {added["id"]}: {added["foot"]} {added["mode"]}, '
                f'frames {inclusive(added["ranges"])}')
    return finish_feet(store, session, 'add', entries, revision, id=added['id'])


def cmd_feet_edit(args, session):
    store = Store(args)
    entries, _, _ = store.read_feet(True)
    entry = find_foot(entries, args.id)
    if args.frames is not None:
        entry['ranges'] = parse_frames(args.frames, store.frames)
    if args.add_frames is not None:
        entry['ranges'] = merge(entry['ranges'] + parse_frames(args.add_frames, store.frames, '--add-frames'))
    if args.remove_frames is not None:
        entry['ranges'] = subtract(entry['ranges'], parse_frames(args.remove_frames, store.frames, '--remove-frames'))
    for key in ('foot', 'mode', 'enabled'):
        if getattr(args, key) is not None:
            entry[key] = getattr(args, key)
    foot_settings(args, entry)
    entries, revision = store.save_feet(entries, session)
    session.say(f'{"Would update" if session.dry_run else "Updated"} foot articulation {args.id}.')
    return finish_feet(store, session, 'edit', entries, revision, id=args.id)


def cmd_feet_remove(args, session):
    store = Store(args)
    entries, _, _ = store.read_feet(True)
    if args.all:
        kept = []
    else:
        find_foot(entries, args.id)
        kept = [e for e in entries if e['id'] != args.id]
    if args.all and not entries:
        session.say('There was no foot articulation.')
        return finish_feet(store, session, 'remove', [], None, removed='all')
    entries, revision = store.save_feet(kept, session)
    session.say(f'{"Would remove" if session.dry_run else "Removed"} ' + ('all foot articulation.' if args.all else f'foot articulation {args.id}.'))
    return finish_feet(store, session, 'remove', entries, revision, removed='all' if args.all else args.id)


# Limb placement and pose smoothing ----------------------------------------------------------------------------------

def find_row(key, entries, ident):
    for entry in entries:
        if entry['id'] == ident:
            return entry
    raise CliError(f'No {ROWS[key][1]} {ident!r}. Rows: {", ".join(e["id"] for e in entries) or "none"}')


def row_settings(args, entry):
    """Copy the given section settings into an entry: chain/strength (placement), joints/window (smoothing)."""
    for key in ('chain', 'strength'):
        if getattr(args, key, None) is not None:
            entry[key] = getattr(args, key)
    joints = getattr(args, 'joints', None) or getattr(args, 'group', None)
    if joints is not None:
        entry['joints'] = joints
    if getattr(args, 'window', None) is not None:
        entry['window_frames'] = args.window


def finish_rows(store, session, key, action, entries, revision, **extra):
    command, singular, plural, what = ROWS[key]
    stale = not session.dry_run and not store.applied()
    if session.dry_run:
        session.say(f'Would save {len(entries)} {singular if len(entries) == 1 else plural} (dry run, nothing written):')
    for entry in entries:
        for line in describe_row(key, entry):
            session.info(line)
    if store.model == 'flowhmr' and entries:
        session.say(f'note: FlowHMR correction does not apply {what} yet; the rows are kept for SMPL-X results.')
    if stale:
        session.say(STALE_CLEARED.format(project=store.project) if action == 'remove' else
                    f'The corrected result is now stale: run `correct --project {store.project}` to apply {what}.')
    return 0, store.summary(f'{command} {action}', store.read()[0], **{key: [track_json(e) for e in entries]}, revision=revision,
                            stale=stale, dry_run=session.dry_run, **extra)


def cmd_rows_list(args, session):
    store = Store(args)
    entries, block = list_rows(store, session, args.rows)
    command, _, plural, what = ROWS[args.rows]
    if not entries:
        session.say(f'No {plural} on corrected_{store.base}.')
    applied = store.applied()
    if entries and not applied:
        session.say(f'The corrected result is stale: run `correct --project {store.project}` to apply {what}.')
    return 0, store.summary(f'{command} list', store.read()[0], **{args.rows: [track_json(e) for e in entries]}, applied=applied,
                            **{args.rows + '_report': block})


def cmd_rows_add(args, session):
    store, key = Store(args), args.rows
    entries, _, _ = store.read_rows(key, True)
    entry = dict(ranges=parse_frames(args.frames, store.frames), enabled=not args.disabled, source='user')
    if args.id:
        entry['id'] = args.id
    row_settings(args, entry)
    before = {e['id'] for e in entries}
    entries, revision = store.save_rows(key, entries + [entry], session)
    added = next(e for e in entries if e['id'] not in before)
    session.say(f'{"Would add" if session.dry_run else "Added"} {ROWS[key][1]} {added["id"]}: {row_what(key, added)}, frames {inclusive(added["ranges"])}')
    return finish_rows(store, session, key, 'add', entries, revision, id=added['id'])


def cmd_rows_edit(args, session):
    store, key = Store(args), args.rows
    entries, _, _ = store.read_rows(key, True)
    entry = find_row(key, entries, args.id)
    if args.frames is not None:
        entry['ranges'] = parse_frames(args.frames, store.frames)
    if args.add_frames is not None:
        entry['ranges'] = merge(entry['ranges'] + parse_frames(args.add_frames, store.frames, '--add-frames'))
    if args.remove_frames is not None:
        entry['ranges'] = subtract(entry['ranges'], parse_frames(args.remove_frames, store.frames, '--remove-frames'))
    if args.enabled is not None:
        entry['enabled'] = args.enabled
    row_settings(args, entry)
    entries, revision = store.save_rows(key, entries, session)
    session.say(f'{"Would update" if session.dry_run else "Updated"} {ROWS[key][1]} {args.id}.')
    return finish_rows(store, session, key, 'edit', entries, revision, id=args.id)


def cmd_rows_remove(args, session):
    store, key = Store(args), args.rows
    _, singular, plural, _ = ROWS[key]
    entries, _, _ = store.read_rows(key, True)
    if args.all and not entries:
        session.say(f'There were no {plural}.')
        return finish_rows(store, session, key, 'remove', [], None, removed='all')
    if not args.all:
        find_row(key, entries, args.id)
    entries, revision = store.save_rows(key, [] if args.all else [e for e in entries if e['id'] != args.id], session)
    session.say(f'{"Would remove" if session.dry_run else "Removed"} ' + (f'all {plural}.' if args.all else f'{singular} {args.id}.'))
    return finish_rows(store, session, key, 'remove', entries, revision, removed='all' if args.all else args.id)
