"""`contacts` command group: edit the project's self-contact tracks (a hand held on a body part), the same file the GUI edits.

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
result frames A-B (the same numbers `review` reports), several separated by commas: 345-420,500-510.'''


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
    which.add_argument('--all', action='store_true', help='delete every self-contact file of the result')
    p.set_defaults(handler=cmd_remove)

    p = actions.add_parser('suggest', parents=parents, help='run review\'s missed-contact check and store suggested intervals',
                           description='Run the `review` missed-contact check and store each issue as a track\'s suggested '
                                       '(automatic) intervals. Every run replaces all suggestions; --accept also adds them to the '
                                       'tracks\' ranges, skipping frames another enabled track already holds with the same hand.')
    p.add_argument('--accept', action='store_true', help='also copy the suggestions into the ranges that `correct` applies')
    p.add_argument('--min-severity', dest='min_severity', choices=SEVERITIES, default='low', help='ignore weaker issues (default low = all)')
    p.add_argument('--result', metavar='KIND', help='result to review (default: the uncorrected base, before any self-contact)')
    p.set_defaults(handler=cmd_suggest)


# Parsing ------------------------------------------------------------------------------------------------------------

def bone(value):
    from ..mixamo_rig import BODY_NAMES
    names = {name.casefold(): name for name in BODY_NAMES}
    try:
        return names[str(value).casefold()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'unknown bone {value!r}; use one of {", ".join(BODY_NAMES)}') from None


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
    applied = store.applied()
    if revision is not None or not applied:
        session.say('Applied to the corrected result.' if applied else STALE.format(project=store.project))
    return 0, store.summary('list', tracks, revision=revision, applied=applied, warning=warning, residuals=residuals)


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
        removed = self_contact_store.clear(store.project, store.base)
        session.say('Removed all self-contacts.' if removed else 'There were no self-contacts.')
        stale = not store.applied()
        if stale:
            session.say(STALE_CLEARED.format(project=store.project))
        return 0, store.summary('remove', [], removed='all', revision=None, stale=stale, dry_run=False)
    tracks, _, _ = store.read(True)
    find(tracks, args.id)
    tracks, revision = store.save([t for t in tracks if t['id'] != args.id], session)
    session.say(f'{"Would remove" if session.dry_run else "Removed"} self-contact {args.id}.')
    return finish(store, session, 'remove', tracks, revision, removed=args.id)


def review_issues(store, kind, session):
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
    obs = loader(store.project)
    motion = review.load_project_motion(store.project, kind)
    if len(motion.joints) != store.frames:
        raise CliError(f'{kind} has {len(motion.joints)} frames but the corrected result has {store.frames}; run `correct` first.')
    last = min(len(obs.keypoints), len(motion.joints)) - 1
    if last < 2:
        raise CliError('Review needs at least 3 frames.')
    issues = review.check_missed_contacts(review.Context(obs, motion, 0, last))
    return review.rank(issues)


def cmd_suggest(args, session):
    from ..motion_tools import RESULTS
    store = Store(args)
    kind = args.result or store.base
    if kind not in RESULTS:
        raise CliError(f'Unknown result {kind!r}', EXIT_USAGE)
    tracks, _, _ = store.read(True)
    limit = SEVERITIES.index(args.min_severity)
    issues = [i for i in review_issues(store, kind, session) if SEVERITIES.index(i['severity']) <= limit]
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
                     f"{issue['frames'][0]}-{issue['frames'][1]} ({issue['metrics']['distance_3d_m_median']:.2f} m apart in 3D)")
    for row in skipped:
        session.say(f"  not accepted for {row['effector']} -> {row['target']}: frames {row['frames_inclusive']} (hand already held by another track)")
    verb = 'Accepted' if args.accept else 'Suggested'
    session.say(f'{verb} {len(issues)} missed contact(s) from {kind} into {len(touched)} track(s).'
                + ('' if args.accept or not issues else ' Accept with `contacts suggest --accept` or `contacts edit --id ID --accept`.'))
    used = [dict(id=i['id'], severity=i['severity'], effector=i['effector'], target=i['target'], frames_inclusive=i['frames'],
                 distance_3d_m_median=round(float(i['metrics']['distance_3d_m_median']), 3)) for i in issues]
    return finish(store, session, 'suggest', tracks, revision, reviewed=kind, issues=used, accepted=args.accept, skipped=skipped)
