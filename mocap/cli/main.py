"""rgbmocap-cli: headless capture, people, reconstruction, correction and export using the GUI's worker jobs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import runner
from .exports import options_from_args, plan_export, resolve_format, run_export, single_target
from .pipeline import (DEFAULT_MODEL, MODELS, VIDEO_SUFFIXES, CliError, MissingModels, PeopleQueue, build_settings,
                       correction_kind, current_kind, find_people_jobs, load_group, missing_models, parse_people, person_frames,
                       person_jobs, project_range, read_json, require_models, resolve_project, save_checked, save_correction,
                       save_group_options, saved_settings, scene_correction_jobs, select_people, simulate_people,
                       single_correction_jobs, single_jobs, tool_stages, video_range)
from .runner import WorkerError

EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_MODELS = 3
EXIT_CANCELLED = 130

# Project to open in the GUI for --view; cli.py launches it after releasing the app lock the GUI needs.
VIEW_PROJECT = None


class Session:
    def __init__(self, args):
        self.json = bool(args.json)
        self.quiet = bool(args.quiet)
        self.dry_run = bool(args.dry_run)

    @property
    def stream(self):
        return sys.stderr if self.json else sys.stdout

    def info(self, text):
        if not self.quiet:
            print(text, file=self.stream, flush=True)

    def say(self, text):
        """Results the user asked for; still printed with --quiet (stderr with --json)."""
        print(text, file=self.stream, flush=True)


_CHARACTERS = None


def characters_module():
    """Import the optional character commands once; a broken module must not break the other commands."""
    global _CHARACTERS
    if _CHARACTERS is None:
        try:
            from . import characters
            _CHARACTERS = (characters, None)
        except Exception as error:
            _CHARACTERS = (None, error)
    return _CHARACTERS


def make_env(args, settings=None):
    from ..resource_safety import load
    safety = False if args.no_resource_safety else load()
    return runner.worker_env('low_memory' if args.low_memory else 'auto', safety,
                             settings['flow'] if settings else None, settings['exmr'] if settings else None)


def existing_video(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise CliError(f'Video not found: {path}')
    return path


def prepared(events):
    projects = [event for event in events if event.get('type') == 'prepared']
    if not projects:
        raise CliError('The prepare stage finished without reporting a project.')
    return runner.prepared_project(projects[-1])


def show_plan(session, rows, env, summary, notes=()):
    summary.update(dry_run=True, jobs=rows, env=env)
    if not session.json:
        runner.format_plan(rows, sys.stdout)
        print('Worker environment: ' + ' '.join(f'{key}={value}' for key, value in env.items() if key.startswith('RGB_MOCAP_')))
        for note in notes:
            print('Note: ' + note)
    summary.setdefault('notes', []).extend(notes)
    return 0, summary


def missing_note(models):
    return 'missing models (a real run exits with code 3): ' + ', '.join(model.name for model in models) if models else None


def placeholder_person_jobs(selection, kind, options):
    people = [dict(id=ident) for ident in selection] if isinstance(selection, list) else [dict(id='<person>')]
    return person_jobs('<project>', people, 0, '<frames>', kind, options)


def placeholder_takes(ident):
    return ['<take>' if str(ident).startswith('<') else f'<take:{ident}>']


# Shared flows ------------------------------------------------------------------------------------------------------

def run_single(session, video, first, end, settings, env, workspace=None):
    jobs, kind = single_jobs(video, first, end, settings, workspace)
    events = runner.run_jobs(jobs, env, quiet=session.quiet)
    project = prepared(events)
    session.info(f'Project: {project}')
    return project, kind


def reconstruct_people(session, target, settings, state, selected, env, start=None, end=None, summary=None):
    """people_ui.input_person_jobs + people_event for an existing people root."""
    root = target.people_root
    manifest = read_json(root / 'project.json')
    first, last = project_range(manifest, start, end)
    begin, finish = person_frames(manifest, first, last)
    if not 0 <= begin < finish <= manifest['frames']:
        raise CliError('The processing range is outside the people scene.', 2)
    stages, kind = tool_stages(settings)
    jobs = person_jobs(root, selected, begin, finish, kind, settings['correction_options'])
    summary.update(project=str(root), workspace=str(target.workspace), model=settings['model'], result=kind,
                   people=[dict(id=p['id'], name=p['name']) for p in selected])
    if session.dry_run:
        found = missing_models(settings, stages)
        planned = simulate_people(root, stages, kind, jobs, placeholder_takes)
        notes = ['one placeholder take per person; real takes come from each person_prepared event']
        if missing_note(found):
            notes.append(missing_note(found))
        return show_plan(session, runner.plan(planned, root), env, summary, notes)
    require_models(settings, stages)
    save_checked(root, state, selected)
    save_group_options(root, settings)
    queue = PeopleQueue(root, stages, kind, jobs)
    runner.run_jobs(queue.jobs, env, queue.on_event, session.quiet, project=root)
    names = {p['id']: p['name'] for p in selected}
    summary['takes'] = [dict(id=ident, name=names.get(ident, str(ident)), projects=paths) for ident, paths in queue.takes.items()]
    session.info(f'People root: {root}')
    for row in summary['takes']:
        session.info(f"  {row['name']}: {len(row['projects'])} take(s)")
    return 0, summary


def export_after(session, args, target, kind, env, summary, stem=None):
    options = options_from_args(args, args.out, target.model)
    plan = plan_export(target, args.out, options, parse_people(getattr(args, 'people', None)), kind, stem)
    outputs = run_export(plan, env, session.quiet)
    summary['outputs'] = outputs
    for path in outputs:
        session.say('Exported: ' + path)
    return outputs


# Commands ----------------------------------------------------------------------------------------------------------

def cmd_models(args, session):
    from ..models import MODELS as MODEL_FOLDER
    report = []
    for model in [args.model] if args.model else MODELS:
        settings = build_settings(args, model)
        stages, kind = tool_stages(settings)
        found = missing_models(settings, stages, args.people)
        report.append(dict(model=model, result=kind, stages=stages,
                           missing=[dict(key=m.key, name=m.name, path=str(m.path), url=m.url) for m in found]))
        if not session.json:
            print(f"{model}: {'ready' if not found else str(len(found)) + ' missing'}  (stages: {', '.join(stages)})")
            for m in found:
                print(f'  missing  {m.name}  ->  {m.path}')
                print(f'           {m.url}')
    code = EXIT_MODELS if any(row['missing'] for row in report) else 0
    if code and not session.json:
        print(f'Install missing models with the app\'s Model Manager (files go under {MODEL_FOLDER}).')
    return code, dict(models=report)


def cmd_capture(args, session):
    video = existing_video(args.video)
    model = args.model or DEFAULT_MODEL
    settings = build_settings(args, model)
    stages, kind = tool_stages(settings)
    selection = parse_people(args.people)
    if selection == 'checked':
        raise CliError('capture finds people itself; use --people all or person ids.', 2)
    (first, end), _ = video_range(video, args.start, args.end)
    env = make_env(args, settings)
    summary = dict(command='capture', video=str(video), model=model, result=kind, start=first, end=end)
    if args.out:
        options = options_from_args(args, args.out, model)
        if Path(args.out).suffix.lower() == '.' + options.fmt:
            if isinstance(selection, list) and len(selection) > 1:
                raise CliError('Several people export several files; --out must be a folder.', 2)
            if selection is None:
                single_target(args.out, 'Person 1', options)
    found = missing_models(settings, stages, people=selection is not None)
    if session.dry_run:
        notes = [note for note in [missing_note(found)] if note]
        if selection is None:
            jobs, _ = single_jobs(video, first, end, settings)
            rows = runner.plan(jobs)
        else:
            rows = runner.plan(find_people_jobs(video, first, end, model))
            planned = simulate_people('<project>', stages, kind, placeholder_person_jobs(selection, kind, settings['correction_options']),
                                      placeholder_takes)
            rows += runner.plan(planned, '<project>')
            notes.append('people ids, frame counts and takes are known after the prepare/people/person stages run')
        if args.out:
            notes.append(f'then export {resolve_format(args.out, args.format)} to {Path(args.out).resolve()}')
        return show_plan(session, rows, env, summary, notes)
    if found:
        raise MissingModels(found)
    if selection is None:
        project, kind = run_single(session, video, first, end, settings, env)
        target = resolve_project(project, model)
        summary.update(project=str(project), workspace=str(target.workspace))
    else:
        events = runner.run_jobs(find_people_jobs(video, first, end, model), env, quiet=session.quiet)
        project = prepared(events)
        found_root = [event for event in events if event.get('type') == 'people_found']
        root = Path(found_root[-1]['project']).resolve() if found_root else project
        target = resolve_project(root, model)
        if target.people_root is None:
            raise CliError(f'Person detection did not produce a people list in {root}.')
        state = load_group(target.people_root)
        print_people(session, state)
        selected = select_people(state, selection)
        code, summary = reconstruct_people(session, target, settings, state, selected, env, summary=summary)
    if args.out:
        export_after(session, args, target, kind, env, summary)
    session.say(f"Project: {summary['project']}")
    return 0, summary


def print_people(session, state):
    if session.json:
        return
    print(f"{'ID':>4}  {'Name':<20} {'Checked':<8} {'First':>6} {'Last':>6} {'Observed':>8}  Takes", file=session.stream)
    for p in state['people']:
        print(f"{p['id']:>4}  {p['name']:<20} {('yes' if p.get('checked') else 'no'):<8} {p.get('first', ''):>6} {p.get('last', ''):>6} "
              f"{p.get('observations', ''):>8}  {len(p.get('takes', []))}", file=session.stream)


def people_rows(state):
    return [dict(id=p['id'], name=p['name'], checked=bool(p.get('checked')), first=p.get('first'), last=p.get('last'),
                 observations=p.get('observations'), takes=[t['project'] for t in p.get('takes', [])]) for p in state['people']]


def cmd_people(args, session):
    from ..models import missing
    if bool(args.video) == bool(args.project):
        raise CliError('Pass either VIDEO or --project.', 2)
    summary = dict(command='people')
    if args.video:
        video = existing_video(args.video)
        model = args.model or DEFAULT_MODEL
        (first, end), _ = video_range(video, args.start, args.end)
        jobs = find_people_jobs(video, first, end, model)
    else:
        target = resolve_project(args.project, args.model)
        model = target.model
        if target.project is None:
            from .. import project_workspace as workspace
            state = workspace.read(target.workspace)
            video = workspace.media_path(target.workspace, state)
            (first, end), _ = video_range(video, args.start, args.end, float(state.get('time_offset', 0.)))
            jobs = find_people_jobs(video, first, end, model, target.workspace)
        else:
            from ..project_storage import source_video
            base = target.people_root or target.project
            manifest = read_json(base / 'project.json')
            first, end = project_range(manifest, args.start, args.end)
            full = project_range(manifest)
            # people_ui.find_people reuses the group's discovery when the range is the group's complete input.
            if target.people_root is not None and abs(first - full[0]) < 1e-6 and abs(end - full[1]) < 1e-6:
                jobs = [('people', ['--project', str(target.people_root)])]
            else:
                jobs = find_people_jobs(source_video(base), first, end, model, target.workspace)
    settings = build_settings(args, model)
    env = make_env(args, settings)
    summary.update(model=model)
    unavailable = missing('people')
    if session.dry_run:
        return show_plan(session, runner.plan(jobs), env, summary, [note for note in [missing_note(unavailable)] if note])
    if unavailable:
        raise MissingModels(unavailable)
    events = runner.run_jobs(jobs, env, quiet=session.quiet)
    found = [event for event in events if event.get('type') == 'people_found']
    if not found:
        raise CliError('Person detection finished without reporting a people list.')
    root = Path(found[-1]['project']).resolve()
    state = load_group(root)
    summary.update(project=str(root), people=people_rows(state))
    print_people(session, state)
    session.say(f'People root: {root}')
    return 0, summary


def cmd_reconstruct(args, session):
    target = resolve_project(args.project, args.model)
    model = target.model
    selection = parse_people(args.people)
    summary = dict(command='reconstruct', model=model)
    if target.people_root is not None:
        state = load_group(target.people_root)
        selected = select_people(state, selection)
        takes = [t for t in selected[0].get('takes', [])]
        from ..people_store import safe_child
        base = safe_child(target.people_root, takes[-1]['project']) if takes else target.people_root
        settings = build_settings(args, model, saved_settings(base, model))
        env = make_env(args, settings)
        code, summary = reconstruct_people(session, target, settings, state, selected, env, args.start, args.end, summary)
        kind = summary['result']
    else:
        if selection is not None:
            raise CliError(f'This project has no people list. Run `people --project {args.project}` first.', 2)
        from .. import project_workspace as workspace
        if target.project is None:
            state = workspace.read(target.workspace)
            saved = state.get('models', {}).get(model, {}).get('settings', {})
            values = {}
            if saved.get('result'):
                values = saved_settings_from_result(saved['result'], model)
            values.update(correction_options=saved.get('correction'), exmr=saved.get('exmr_options') or None)
            values = {key: value for key, value in values.items() if value is not None}
            if saved.get('flow_options'):
                values['flow'] = saved['flow_options']
            settings = build_settings(args, model, values)
            video = workspace.media_path(target.workspace, state)
            (first, end), _ = video_range(video, args.start, args.end, float(state.get('time_offset', 0.)))
        else:
            from ..project_storage import source_video
            settings = build_settings(args, model, saved_settings(target.project, model))
            video = source_video(target.project)
            first, end = project_range(read_json(target.project / 'project.json'), args.start, args.end)
        stages, kind = tool_stages(settings)
        env = make_env(args, settings)
        summary.update(result=kind, workspace=str(target.workspace), start=first, end=end)
        found = missing_models(settings, stages)
        if session.dry_run:
            jobs, _ = single_jobs(video, first, end, settings, target.workspace)
            return show_plan(session, runner.plan(jobs), env, summary, [note for note in [missing_note(found)] if note])
        if found:
            raise MissingModels(found)
        project, kind = run_single(session, video, first, end, settings, env, target.workspace)
        target = resolve_project(project, model)
        summary['project'] = str(project)
    if session.dry_run:
        return code, summary
    if args.out:
        export_after(session, args, target, kind, env, summary)
    session.say(f"Project: {summary['project']}")
    return 0, summary


def saved_settings_from_result(kind, model):
    from ..motion_tools import RESULTS, has_handflow, result_options
    if kind not in RESULTS:
        return {}
    htd, foot = result_options(kind)
    gvhmr = model == 'gvhmr'
    return dict(refine=htd and gvhmr, footmr=foot and gvhmr, handflow=has_handflow(kind) and gvhmr, correction=kind.startswith('corrected_'))


def cmd_correct(args, session):
    from ..motion_tools import RESULTS, base_result
    target = resolve_project(args.project, args.model)
    if target.project is None:
        raise CliError(f'This project has no saved {target.model} motion. Reconstruct it first.')
    model = target.model
    selection = parse_people(args.people)
    summary = dict(command='correct', model=model)
    if target.people_root is None:
        if selection is not None:
            raise CliError('This project has no people list; --people applies to projects processed with `people`.', 2)
        project = target.project
        settings = build_settings(args, model, saved_settings(project, model))
        kind = correction_kind(args.result or current_kind(project, model))
        if not (project / RESULTS[base_result(kind)][0]).is_file():
            raise CliError(f'The {base_result(kind)} motion has not been reconstructed yet. Run `reconstruct` first.')
        jobs = single_correction_jobs(project, kind)
        writes = [project]
        anchor = project
    else:
        from ..people_store import safe_child
        root = target.people_root
        state = load_group(root)
        if selection is None:
            selected = [p for p in state['people'] if p.get('checked')]
        else:
            selected = select_people(state, selection)
        candidates = [p for p in selected if p.get('takes')] or [p for p in state['people'] if p.get('takes')]
        if not candidates:
            raise CliError('Reconstruct a person before applying the shared scene floor.')
        base = safe_child(root, candidates[0]['takes'][-1]['project'])
        settings = build_settings(args, model, saved_settings(base, model))
        kind = correction_kind(args.result or current_kind(base, model))
        jobs, takes = scene_correction_jobs(root, state, kind, {p['id'] for p in selected})
        chosen = {p['id'] for p in selected}
        writes = [project for person, project in takes if selection is None or person['id'] in chosen]
        anchor = root
    options = settings['correction_options']
    env = make_env(args, settings)
    summary.update(project=str(anchor), result=kind, correction=options, updated=[str(p) for p in writes])
    characters, error = characters_module()
    if session.dry_run:
        notes = [f'writes correction settings to {len(writes)} project(s)' + (' and the shared scene floor' if target.people_root else '')]
        if characters is None:
            notes.append(f'character refresh unavailable: {error}')
        code, summary = show_plan(session, runner.plan(jobs), env, summary, notes)
        if characters is not None and not args.no_character_refresh:
            try:
                characters.refresh_after_correction(anchor, jobs, None, quiet=session.quiet, dry_run=True)
            except (characters.CharacterError, OSError, ValueError, KeyError, WorkerError) as exc:
                session.info(f'Note: cannot plan linked character refresh: {exc}')
        return code, summary
    for project in writes:
        save_correction(project, options)
    if target.people_root is not None:
        from ..scene_floor import save_config
        save_config(target.people_root, options)
    plans = None
    if characters is not None:
        try:
            plans = characters.prepare_correction_refresh(anchor, jobs)
        except (characters.CharacterError, OSError, ValueError, KeyError) as exc:
            raise CliError(f'Cannot update linked characters: {exc}') from None
    else:
        session.info(f'Warning: character refresh unavailable ({error}); linked characters are not updated.')
    runner.run_jobs(list(jobs), env, quiet=session.quiet, project=anchor)
    if characters is not None and plans:
        if args.no_character_refresh:
            session.info(f'{len(plans)} linked character(s) need retargeting (`character retarget`).')
        else:
            summary['characters'] = characters.refresh_after_correction(anchor, jobs, plans, quiet=session.quiet)
    session.say(f'Corrected {len(jobs) // 2} take(s) with {kind}. Project: {anchor}')
    return 0, summary


def cmd_export(args, session):
    target = resolve_project(args.project, args.model)
    options = options_from_args(args, args.out, target.model)
    plan = plan_export(target, args.out, options, parse_people(args.people), args.result)
    env = make_env(args)
    summary = dict(command='export', project=str(target.root), stage=plan.stage, outputs=[str(p) for p in plan.targets])
    if session.dry_run:
        summary.update(dry_run=True, request=plan.rows, args=['--project', str(plan.request_project), '--output', '<request>'])
        if not session.json:
            print(f'Planned job: {plan.stage} --project {plan.request_project} --output <request>')
            print('Request: ' + json.dumps(plan.rows, ensure_ascii=False, indent=2))
        return 0, summary
    outputs = run_export(plan, env, session.quiet)
    for path in outputs:
        session.say('Exported: ' + path)
    return 0, summary


def cmd_batch(args, session):
    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        raise CliError(f'Not a folder: {folder}')
    clips = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES), key=lambda p: p.name.casefold())
    if not clips:
        raise CliError(f'No videos ({", ".join(VIDEO_SUFFIXES)}) in {folder}')
    model = args.model or DEFAULT_MODEL
    settings = build_settings(args, model)
    stages, kind = tool_stages(settings)
    env = make_env(args, settings)
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not out.is_dir():
        raise CliError(f'--out must be a folder: {out}')
    fmt = args.format or 'fbx'
    stems = {}
    for clip in clips:
        stems.setdefault(clip.stem.casefold(), []).append(clip.name)
    duplicates = [names for names in stems.values() if len(names) > 1]
    if duplicates:
        raise CliError('Clips would share an output name: ' + '; '.join(', '.join(names) for names in duplicates))
    targets = {clip: out / (clip.stem + '.' + fmt) for clip in clips}
    options = options_from_args(args, targets[clips[0]], model)
    if fmt != 'blend' and not args.overwrite:
        existing = [str(path) for path in targets.values() if path.exists()]
        if existing:
            raise CliError('Outputs already exist (pass --overwrite): ' + ', '.join(existing))
    found = missing_models(settings, stages)
    summary = dict(command='batch', model=model, result=kind, folder=str(folder), out=str(out), clips=[])
    if session.dry_run:
        rows = []
        for clip in clips:
            try:
                (first, end), _ = video_range(clip, args.start, args.end)
            except CliError as error:
                summary['clips'].append(dict(clip=clip.name, error=error.message))
                if not session.json:
                    print(f'{clip.name}: {error.message}')
                continue
            jobs, _ = single_jobs(clip, first, end, settings)
            planned = runner.plan(jobs)
            rows.append(dict(clip=clip.name, output=str(targets[clip]), jobs=planned))
            if not session.json:
                print(f'{clip.name} -> {targets[clip]}')
                runner.format_plan(planned, sys.stdout)
        summary.update(dry_run=True, plans=rows, env=env)
        if not session.json:
            print('Worker environment: ' + ' '.join(f'{k}={v}' for k, v in env.items() if k.startswith('RGB_MOCAP_')))
            if missing_note(found):
                print('Note: ' + missing_note(found))
        return 0, summary
    if found:
        raise MissingModels(found)
    failures = 0
    stop = False
    for number, clip in enumerate(clips, 1):
        row = dict(clip=clip.name, output=str(targets[clip]), status='skipped')
        summary['clips'].append(row)
        if stop:
            continue
        session.info(f'=== {number}/{len(clips)} {clip.name}')
        started = time.monotonic()
        try:
            (first, end), _ = video_range(clip, args.start, args.end)
            project, result = run_single(session, clip, first, end, settings, env)
            plan = plan_export(resolve_project(project, model), targets[clip], options, None, result)
            row.update(project=str(project), outputs=run_export(plan, env, session.quiet), status='ok')
        except (CliError, WorkerError, OSError) as error:
            failures += 1
            row.update(status='failed', error=getattr(error, 'message', None) or str(error))
            session.say(f'{clip.name}: failed: {row["error"]}')
            stop = not args.continue_on_error
        row['seconds'] = round(time.monotonic() - started, 1)
    if not session.json:
        print()
        width = max(len(row['clip']) for row in summary['clips'])
        for row in summary['clips']:
            detail = ', '.join(row.get('outputs', [])) if row['status'] == 'ok' else row.get('error', '')
            print(f"{row['clip']:<{width}}  {row['status']:<7}  {row.get('seconds', 0):>7.1f} s  {detail}")
    pending = sum(row['status'] != 'ok' for row in summary['clips'])
    return (EXIT_ERROR if failures or pending else 0), summary


# Parser ------------------------------------------------------------------------------------------------------------

def add_globals(parser, nested=False):
    # Subcommands accept the global flags too; SUPPRESS keeps a value given before the command.
    default = argparse.SUPPRESS if nested else False
    group = parser.add_argument_group('global options')
    group.add_argument('--json', action='store_true', default=default, help='machine-readable summary on stdout; progress goes to stderr')
    group.add_argument('--quiet', action='store_true', default=default, help='print only results and errors')
    group.add_argument('--dry-run', dest='dry_run', action='store_true', default=default,
                       help='print the job list / request that would run; run nothing (no models needed)')
    group.add_argument('--low-memory', dest='low_memory', action='store_true', default=default, help='Low Memory resource mode (RGB_MOCAP_COMPUTE_MODE)')
    group.add_argument('--no-resource-safety', dest='no_resource_safety', action='store_true', default=default,
                       help='disable RAM headroom stopping and VRAM limits for this run')
    group.add_argument('--no-lock', dest='no_lock', action='store_true', default=default, help='do not take the app lock (allows running with the GUI open)')
    group.add_argument('--view', action='store_true', default=default, help='open the result in the RGB MoCap GUI when the command succeeds')


def motion_parent():
    on_off = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('motion model')
    group.add_argument('--model', choices=MODELS, help=f'motion model (new videos default to {DEFAULT_MODEL}; projects use their own)')
    group.add_argument('--refine', action=on_off, help='GVHMR: HTD temporal refinement (default on)')
    group.add_argument('--footmr', action=on_off, help='GVHMR: FootMR ankle refinement (default on)')
    group.add_argument('--handflow', action=on_off, help='GVHMR: HandFlow finger motion (default on)')
    group.add_argument('--correction', action=on_off, help='contact motion correction (default on)')
    group = parser.add_argument_group('ExMR options')
    group.add_argument('--camera', choices=('static', 'moving'), help='static camera (default) or moving camera / SimpleVO')
    group.add_argument('--smoothing', action=on_off, help='smooth motion (default on)')
    group = parser.add_argument_group('FlowHMR options')
    group.add_argument('--flow-camera', dest='flow_camera', metavar='auto|static|N',
                       help='camera sampling: auto (default), static, or every N frames (1, 2, 5, 10, 30)')
    group.add_argument('--fast-sam', dest='fast_sam', action=on_off, help='fast SAM 3D Body features (default on)')
    group.add_argument('--sam-batch4', dest='sam_batch4', action=on_off, help='SAM batch of 4 (default on)')
    group.add_argument('--sam-gpu-crop', dest='sam_gpu_crop', action=on_off, help='SAM GPU cropping (default on)')
    group.add_argument('--hand-stride4', dest='sam_hand_stride4', action=on_off, help='hand features every 4 frames (default on)')
    group.add_argument('--hamer', action=on_off, help='HaMeR finger refinement (default off)')
    return parser


def correction_parent():
    from ..correction_settings import PRESETS
    on_off = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('motion correction')
    group.add_argument('--preset', choices=list(PRESETS), help='lock preset (default normal)')
    group.add_argument('--floor-mode', dest='floor_mode', choices=('flat', 'multi_level'), help='flat ground (default) or multi level')
    group.add_argument('--floor', metavar='M|auto', help='manual floor height in meters (world Y), or auto (default)')
    group.add_argument('--pelvis-offset', dest='pelvis_offset', type=float, metavar='M', help='pelvis height offset, -0.25..0.25 m (default 0)')
    group.add_argument('--foot-lock', dest='foot_lock', action=on_off, help='foot contact lock (default on)')
    group.add_argument('--hand-lock', dest='hand_lock', action=on_off, help='hand contact lock (default off)')
    return parser


def range_parent():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('range')
    group.add_argument('--start', type=float, metavar='S', help='start time in seconds on the source video (default: beginning)')
    group.add_argument('--end', type=float, metavar='S', help='end time in seconds (default: end of the video)')
    return parser


def export_parent():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('export')
    group.add_argument('--format', choices=('fbx', 'bvh', 'blend'), help='output format (default: from --out suffix, else fbx)')
    group.add_argument('--bvh-profile', dest='bvh_profile', choices=('generic', 'biped'), help='BVH naming: generic / Blender (default) or 3ds Max biped')
    group.add_argument('--tpose', action='store_true', help='add a T-pose before the motion')
    group.add_argument('--no-camera-animation', dest='no_camera_animation', action='store_true', help='Blender: do not export saved camera tracking')
    group.add_argument('--camera-scale', dest='camera_scale', type=float, metavar='X', help='Blender: camera movement scale 0..100 (default 1)')
    group.add_argument('--refine-camera-rotation', dest='refine_camera_rotation', action='store_true',
                       help='Blender, FlowHMR/ExMR: refine camera rotation against the background')
    group.add_argument('--overwrite', action='store_true', help='replace existing output files')
    return parser


PEOPLE_HELP = 'people to process: all, checked (people.json selection), or ids such as 1,3 or 1 3'


def build_parser():
    from ..motion_tools import RESULTS
    parser = argparse.ArgumentParser(prog='rgbmocap-cli', description='Headless RGB MoCap: runs the same worker jobs as the GUI.',
                                     epilog='Exit codes: 0 ok, 1 failed, 2 usage, 3 missing models, 4 app busy (GUI open), 130 cancelled.')
    add_globals(parser)
    common = argparse.ArgumentParser(add_help=False)
    add_globals(common, True)
    motion, correction, ranges, export = motion_parent(), correction_parent(), range_parent(), export_parent()
    commands = parser.add_subparsers(dest='command', metavar='COMMAND', required=True)
    results = sorted(RESULTS)

    p = commands.add_parser('models', parents=[common, motion], help='report missing model files',
                            description='Report model files missing for the selected pipeline (all models when --model is omitted). Exit 3 if any are missing.')
    p.add_argument('--people', action='store_true', help='include person detection models (multi-person)')
    p.set_defaults(handler=cmd_models)

    p = commands.add_parser('capture', parents=[common, motion, correction, ranges, export], help='video -> motion (single person or selected people)',
                            description='Prepare a project from VIDEO and reconstruct motion like Reconstruct 3D. With --people, find people first.')
    p.add_argument('video', metavar='VIDEO')
    p.add_argument('--people', nargs='+', metavar='ID', help='find people, then reconstruct: all or ids such as 1,3')
    p.add_argument('--out', metavar='PATH', help='export the result to this file or folder')
    p.set_defaults(handler=cmd_capture)

    p = commands.add_parser('people', parents=[common, ranges], help='find people in a video or project',
                            description='Run Find person and print the discovered people.')
    p.add_argument('video', metavar='VIDEO', nargs='?')
    p.add_argument('--project', metavar='P', help='workspace root, model project or people root')
    p.add_argument('--model', choices=MODELS, help=f'motion model for a new video (default {DEFAULT_MODEL})')
    p.set_defaults(handler=cmd_people)

    p = commands.add_parser('reconstruct', parents=[common, motion, correction, ranges, export], help='reconstruct an existing project or people root',
                            description='Re-run reconstruction on a project. Saved settings are used unless overridden.')
    p.add_argument('--project', required=True, metavar='P', help='workspace root, model project, people root or person take')
    p.add_argument('--people', nargs='+', metavar='ID', help=PEOPLE_HELP + ' (default: checked)')
    p.add_argument('--out', metavar='PATH', help='export the result to this file or folder')
    p.set_defaults(handler=cmd_reconstruct)

    p = commands.add_parser('correct', parents=[common, correction], help='re-apply motion correction',
                            description='Apply Correction: write the correction settings, then correct and export every take.')
    p.add_argument('--project', required=True, metavar='P', help='workspace root, model project, people root or person take')
    p.add_argument('--people', nargs='+', metavar='ID', help='people whose takes receive the settings (default: all corrected takes)')
    p.add_argument('--result', choices=results, metavar='KIND', help='base result to correct (default: the project selection)')
    p.add_argument('--no-character-refresh', dest='no_character_refresh', action='store_true',
                   help='only mark linked characters as needing retarget')
    p.add_argument('--model', choices=MODELS, help='model inside a workspace root (default: its active model)')
    p.set_defaults(handler=cmd_correct)

    p = commands.add_parser('export', parents=[common, export], help='export source motion (FBX, BVH, Blender)',
                            description='Export the committed source motion of a project or of selected people.')
    p.add_argument('--project', required=True, metavar='P', help='workspace root, model project, people root or person take')
    p.add_argument('--out', required=True, metavar='PATH', help='output file, or folder for several takes')
    p.add_argument('--people', nargs='+', metavar='ID', help=PEOPLE_HELP + ' (default: checked)')
    p.add_argument('--result', choices=results, metavar='KIND', help='result to export (default: the project selection)')
    p.add_argument('--model', choices=MODELS, help='model inside a workspace root (default: its active model)')
    p.set_defaults(handler=cmd_export)

    p = commands.add_parser('batch', parents=[common, motion, correction, ranges, export], help='single-person capture for every video in a folder',
                            description='Capture every video (mp4, mov, mkv, avi, webm) in DIR; outputs are named after each clip.')
    p.add_argument('dir', metavar='DIR')
    p.add_argument('--out', required=True, metavar='OUTDIR', help='output folder')
    p.add_argument('--continue-on-error', dest='continue_on_error', action='store_true', help='keep going after a failed clip')
    p.set_defaults(handler=cmd_batch)

    from . import fixes
    fixes.register(commands, common)

    from . import review
    review.register(commands, common)

    characters, error = characters_module()
    if characters is not None:
        try:
            characters.register(commands)
        except Exception as exc:
            error = exc
    if 'character' not in commands.choices:
        p = commands.add_parser('character', help='character commands (unavailable)', add_help=False)
        p.add_argument('rest', nargs=argparse.REMAINDER)
        p.set_defaults(handler=lambda args, session, reason=error: _unavailable(reason))
    return parser


def _unavailable(reason):
    raise CliError(f'Character commands are unavailable: {reason}')


def _model_lines(keys):
    from ..models import BY_KEY
    return [f'  {BY_KEY[key].name}  ->  {BY_KEY[key].path}' if key in BY_KEY else '  ' + key for key in keys]


def main(argv=None) -> int:
    global VIEW_PROJECT
    from .viewer import view_target
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, 'handler', None) is None and getattr(args, 'func', None) is not None:
        if args.json:
            runner.progress_stream = sys.stderr
        code = int(args.func(args) or 0)
        VIEW_PROJECT = view_target(args, None, code)
        return code
    session = Session(args)
    if session.json:
        runner.progress_stream = sys.stderr
    summary = dict(command=args.command)
    try:
        code, summary = args.handler(args, session)
        summary.setdefault('command', args.command)
        summary['ok'] = code == 0
    except MissingModels as error:
        code = error.code
        summary.update(ok=False, error=error.message, missing=[dict(key=m.key, name=m.name, path=str(m.path)) for m in error.models])
        print('error: ' + error.message, file=sys.stderr)
        for model in error.models:
            print(f'  {model.name}  ->  {model.path}', file=sys.stderr)
        print('Install them with the app\'s Model Manager, then run again. `rgbmocap-cli models` lists them.', file=sys.stderr)
    except CliError as error:
        code = error.code
        summary.update(ok=False, error=error.message)
        print('error: ' + error.message, file=sys.stderr)
    except WorkerError as error:
        blocked = error.blocked_models is not None or error.code == EXIT_MODELS
        code = EXIT_MODELS if blocked else EXIT_ERROR
        summary.update(ok=False, error=error.message, stage=error.stage, missing=error.blocked_models)
        print(f'error: {error.stage} failed: {error.message}', file=sys.stderr)
        for line in _model_lines(error.blocked_models or []):
            print(line, file=sys.stderr)
    except KeyboardInterrupt:
        code = EXIT_CANCELLED
        summary.update(ok=False, error='Cancelled. Completed stages will be reused next time.')
        print('\nCancelled. Completed stages will be reused next time.', file=sys.stderr)
    except OSError as error:
        code = EXIT_ERROR
        summary.update(ok=False, error=str(error))
        print('error: ' + str(error), file=sys.stderr)
    VIEW_PROJECT = view_target(args, summary, code)
    if session.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return code
