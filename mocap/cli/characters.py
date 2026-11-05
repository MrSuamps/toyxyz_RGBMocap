"""Headless character import, retargeting and export through the `character` worker stage.

Mirrors the GUI character workspace (mocap/character_ui.py) without Qt: the same job requests, the same
characters/index.json updates and the same limits. Character data lives in the project's workspace root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..paths import ROOT
from ..storage import atomic_json
from .runner import run_stage, WorkerError

MAX_CHARACTERS = 8
CHARACTER_SUFFIXES = ('.fbx', '.vrm', '.pmx')
# The GUI starts character jobs from the system environment plus these two variables.
CHARACTER_ENV = {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}


class CharacterError(ValueError):
    """A user-facing failure; the message is printed as-is."""


@dataclass
class Context:
    root: Path              # character root: the model-folder workspace, or the project itself for older layouts
    project: Path | None    # model project whose people/takes provide source motion (None: no saved motion yet)


def _json_mode(args):
    return bool(getattr(args, 'json', False))


def _info(args, text):
    if getattr(args, 'quiet', False):
        return
    machine = _json_mode(args) or getattr(args, 'dry_run', False)
    print(text, file=sys.stderr if machine else sys.stdout, flush=True)


def _print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def _public(character):
    return {key: value for key, value in character.items() if key != '_starts'}


def open_project(value) -> Context:
    """Accept a model-folder workspace root, a model project or a person take."""
    from .. import project_workspace as workspace
    path = Path(value).expanduser().resolve()
    if not (path / 'project.json').is_file():
        raise CharacterError(f'Not a project folder (no project.json): {path}')
    try:
        state = json.loads((path / 'project.json').read_text(encoding='utf-8'))
        if state.get('workspace_version') == workspace.VERSION:
            state = workspace.read(path)
            model = workspace.model_project(path, state['active_model'], state)
            return Context(path, model if (model / 'project.json').is_file() else None)
        return Context(workspace.root_for(path, state) or path, path)
    except (OSError, ValueError, KeyError) as exc:
        raise CharacterError(f'Cannot open project: {exc}') from exc


def character_path(root, relative):
    base = (Path(root) / 'characters').resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise CharacterError('Invalid character project path')
    return path


def load_characters(root):
    path = Path(root) / 'characters/index.json'
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf8'))
        if len(data) > MAX_CHARACTERS:
            raise ValueError(f'Too many imported characters (maximum {MAX_CHARACTERS}).')
        for c in data:
            character_path(root, c['asset'])
            if c.get('animation'):
                character_path(root, c['animation'])
                for take in c.get('takes', []):
                    character_path(root, c['animation'] + '/' + take['folder'])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise CharacterError('Cannot load characters: ' + str(exc)) from exc
    return data


def save_characters(root, characters):
    atomic_json(Path(root) / 'characters/index.json', characters)


def find_character(characters, key):
    """CHAR may be an exact id, a name (case-insensitive) or a unique id prefix."""
    for c in characters:
        if c['id'] == key:
            return c
    named = [c for c in characters if c['name'] == key] or [c for c in characters if c['name'].casefold() == key.casefold()]
    if len(named) == 1:
        return named[0]
    if named:
        raise CharacterError(f'Several characters are named {key!r}; use an id: ' + ', '.join(c['id'] for c in named))
    prefixed = [c for c in characters if key and c['id'].startswith(key.lower())]
    if len(prefixed) == 1:
        return prefixed[0]
    if prefixed:
        raise CharacterError(f'Ambiguous character id prefix {key!r}: ' + ', '.join(c['id'] for c in prefixed))
    raise CharacterError(f'Unknown character {key!r}. Use `character list` to see ids and names.')


def default_result(project):
    """The project's saved result selection (pipeline.result), or the first take's for a people group."""
    from ..people_store import load_people, root_for, safe_child
    if project is None:
        raise CharacterError('This project has no saved motion for its active model.')
    manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    kind = manifest.get('pipeline', {}).get('result')
    if kind:
        return kind
    people_root = root_for(project, manifest)
    state = load_people(people_root)
    for person in (state or {}).get('people', []):
        for take in person.get('takes', []):
            path = safe_child(people_root, take['project']) / 'project.json'
            if path.is_file():
                kind = json.loads(path.read_text(encoding='utf-8')).get('pipeline', {}).get('result')
                if kind:
                    return kind
    raise CharacterError('No saved motion result in this project; pass --result KIND.')


def prepare_scene_cameras(root):
    """Start the same CPU helper the GUI scene loader starts when shared people-scene cameras are missing or stale."""
    if getattr(sys, 'frozen', False) or os.environ.get('RGB_MOCAP_MANAGED') == '1':
        command = [str(ROOT / 'runtime/python.exe'), '-I', '-X', 'utf8', str(ROOT / 'scene_boot.py'), str(root)]
    else:
        command = [sys.executable, '-X', 'utf8', '-m', 'mocap.scene_camera', str(root)]
    log = Path(root) / '.scene_camera.log'
    with log.open('wb') as output:
        try:
            child = subprocess.run(command, stdout=output, stderr=output, cwd=str(ROOT), timeout=90,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired:
            raise CharacterError('Shared scene camera preparation timed out; retry after freeing RAM.') from None
    if child.returncode:
        message = log.read_bytes()[-4096:].decode('utf-8', errors='replace')
        raise CharacterError('Cannot prepare shared cameras: ' + message)


def scene_transforms(root, state, kind, prepare=True):
    """Per-take take-to-shared-scene transforms, as scene_data.SceneDataLoader.load computes them for the preview.

    Returns a function project -> 4x4 list, or None when the GUI scene would not place that take.
    """
    import numpy as np
    from ..people_store import media_receipt, safe_child
    from ..scene_camera import VERSION, read_camera, take_projects, validate_transform
    root = Path(root)
    manifest = json.loads((root / 'project.json').read_text(encoding='utf-8'))
    if manifest.get('pipeline', {}).get('model') == 'flowhmr':
        # FlowHMR takes are generated in one shared world; flow_scene.load_scene uses identity.
        return lambda project: np.eye(4).tolist()
    projects = take_projects(root, state)
    if not projects:
        return lambda project: None
    path = root / 'people_scene.json'

    def current():
        if not path.exists() or any(read_camera(project) is None for _, project in projects):
            return False
        reference = json.loads(path.read_text(encoding='utf-8'))
        if reference.get('version') != VERSION or reference.get('input') != media_receipt(root):
            return False
        anchor = read_camera(safe_child(root, reference['anchor']))
        return anchor is not None and anchor['sources'] == reference.get('sources')

    if not current() and prepare:
        prepare_scene_cameras(root)
    if not path.exists():
        return lambda project: None
    reference = json.loads(path.read_text(encoding='utf-8'))
    camera_to_scene = np.linalg.inv(validate_transform(reference['world_to_camera']))

    def transform(project):
        try:
            camera = read_camera(Path(project))
            if camera is None or not np.allclose(camera['K'], reference['K'], atol=1e-4):
                return None
            if 'refined' in kind and not camera.get('htd_compatible'):
                return None
            return validate_transform(camera_to_scene @ validate_transform(camera['world_to_camera'])).tolist()
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return transform


def character_sources(project, kind, prepare=True):
    """People with retargetable takes of `kind`, as CharacterWorkspace.character_sources lists them in the Retarget menu."""
    from ..motion_tools import result_is_current
    from ..people_store import load_people, root_for, safe_child
    if project is None:
        return []
    project = Path(project).resolve()
    people_root = root_for(project)
    state = load_people(people_root)
    transforms = None
    if state:
        people = state['people']
        transforms = scene_transforms(people_root, state, kind, prepare)
    else:
        name = 'Person 1'
        if (project / 'tracking.npz').is_file():
            from ..single_actor import preferences
            name = preferences(project)['name']
        people = [dict(id=1, name=name, takes=[dict(project=str(project))])]
    result = []
    for person in people:
        takes = []
        for take in person.get('takes', []):
            path = safe_child(people_root, take['project']) if state else Path(take['project'])
            if not (path / (kind + '_mixamo.json')).is_file() or not result_is_current(path, kind):
                continue
            manifest = json.loads((path / 'project.json').read_text(encoding='utf8'))
            start = float(manifest.get('start', 0.))
            takes.append(dict(project=str(path), start=start, transform=transforms(path) if transforms else None))
        if takes:
            result.append(dict(id=person['id'], name=person['name'], takes=sorted(takes, key=lambda t: t['start'])))
    return result


def select_source(sources, key):
    """--person accepts a person name (case-insensitive) or a numeric person id."""
    named = [s for s in sources if s['name'] == key] or [s for s in sources if s['name'].casefold() == key.casefold()]
    if len(named) == 1:
        return named[0]
    if named:
        raise CharacterError(f'Several people are named {key!r}; use the person id: ' + ', '.join(str(s['id']) for s in named))
    if key.strip().isdigit():
        for s in sources:
            if s['id'] == int(key):
                return s
    available = ', '.join(f"{s['id']} {s['name']}" for s in sources)
    raise CharacterError(f'No prepared source motion for person {key!r}. Available: {available}')


def stale_exports(source, kind):
    """Corrected takes whose exportable result is missing or outdated (GUI: current_export() is None)."""
    from ..motion_tools import selected_export
    if not kind.startswith('corrected_'):
        return []
    return [t['project'] for t in source['takes'] if selected_export(Path(t['project']), kind) is None]


def start_character_job(root, job, quiet=False, dry_run=False):
    """Write the request and run the `character` stage exactly as CharacterWorkspace.start_character_job does.

    Returns the worker's complete.json, or None for a dry run (the request is printed, nothing is written).
    """
    request = Path(root) / 'characters' / ('job-' + uuid.uuid4().hex + '.json')
    stage_args = ['--request', str(request)]
    if job['mode'] == 'retarget':
        # Selects the processing runtime from the first take's model.
        stage_args += ['--project', job['takes'][0]['project']]
    if dry_run:
        _print_json(dict(stage='character', args=stage_args, request=job))
        return None
    from ..retarget.runtime import blender
    if job['mode'] != 'adopt':
        blender()
    atomic_json(request, job)
    run_stage('character', stage_args, env=dict(CHARACTER_ENV), quiet=quiet)
    complete = Path(job['output']) / 'complete.json'
    if not complete.is_file():
        raise CharacterError('Character processing failed: the worker wrote no complete.json.')
    return json.loads(complete.read_text(encoding='utf8'))


def apply_retarget(root, job, result):
    characters = load_characters(root)
    c = next((c for c in characters if c['id'] == job['id']), None)
    if c is None:
        raise CharacterError('The character was removed while it was being retargeted.')
    c.pop('_starts', None)
    c.pop('needs_retarget', None)
    c.update(animation=job['animation'], takes=result['takes'], source_name=result['source_name'], result=result['result'])
    save_characters(root, characters)
    return c


def prepare_correction_refresh(project, jobs):
    """Mark characters baked from the projects corrected by `jobs` as needing retarget and return their refresh plans.

    Call before running the correction, as gui.start_jobs(refresh_characters=True) does. Raises CharacterError or
    ValueError (ambiguous character source); the GUI then refuses to start the correction.
    """
    from ..retarget.refresh import correction_refreshes
    context = open_project(project)
    characters = load_characters(context.root)
    plans = correction_refreshes(characters, jobs, context.project or Path(project).resolve())
    ids = {plan['id'] for plan in plans}
    for c in characters:
        if c['id'] in ids:
            c['needs_retarget'] = True
    if ids:
        save_characters(context.root, characters)
    return plans


def refresh_after_correction(project, correct_jobs, plans=None, quiet=False, dry_run=False):
    """Re-retarget characters linked to the projects corrected by `correct_jobs` ([(stage, args)]; only 'correct' counts).

    Pass the `plans` returned by prepare_correction_refresh (called before the correction) to mirror the GUI exactly;
    when omitted they are prepared here. Stops at the first failure and raises WorkerError/CharacterError; characters not
    yet refreshed keep needs_retarget. Returns [{'id', 'name', 'takes'}] for the refreshed characters.
    """
    context = open_project(project)
    if plans is None:
        if dry_run:
            from ..retarget.refresh import correction_refreshes
            plans = correction_refreshes(load_characters(context.root), correct_jobs, context.project or Path(project).resolve())
        else:
            plans = prepare_correction_refresh(project, correct_jobs)
    refreshed = []
    for index, plan in enumerate(plans, 1):
        c = next((c for c in load_characters(context.root) if c['id'] == plan['id']), None)
        if c is None:
            continue
        folder = c['id'] + '/animation-' + uuid.uuid4().hex
        job = dict(plan, mode='retarget', output=str(character_path(context.root, folder)),
                   asset=str(character_path(context.root, c['asset'])), animation=folder,
                   refresh_label=f"Updating {c['name']} · {index}/{len(plans)}")
        if not quiet:
            print(job['refresh_label'], file=sys.stderr, flush=True)
        result = start_character_job(context.root, job, quiet, dry_run)
        if result is not None:
            c = apply_retarget(context.root, job, result)
            refreshed.append(dict(id=c['id'], name=c['name'], takes=len(c['takes'])))
    return refreshed


def cmd_import(args):
    context = open_project(args.project)
    characters = load_characters(context.root)
    if len(characters) >= MAX_CHARACTERS:
        raise CharacterError(f'Up to {MAX_CHARACTERS} characters are supported per project.')
    source = Path(args.file).expanduser().resolve()
    if source.suffix.lower() not in CHARACTER_SUFFIXES:
        raise CharacterError('Humanoid characters must be .fbx (Mixamo rig), .vrm or .pmx files.')
    if not source.is_file():
        raise CharacterError(f'Character file not found: {source}')
    ident = uuid.uuid4().hex
    job = dict(mode='import', file=str(source), output=str(character_path(context.root, ident + '/asset')), id=ident)
    result = start_character_job(context.root, job, getattr(args, 'quiet', False), getattr(args, 'dry_run', False))
    if result is None:
        return 0
    characters = load_characters(context.root)
    entry = dict(id=ident, name=source.stem, kind=result['kind'], asset=ident + '/asset', visible=True)
    characters.append(entry)
    save_characters(context.root, characters)
    if _json_mode(args):
        _print_json(dict(ok=True, root=str(context.root), character=entry, seconds=result.get('seconds')))
    else:
        print(f"Imported {entry['name']} ({entry['kind']}) as {ident} in {result.get('seconds', 0):.1f}s")
    return 0


def _motion_text(c):
    if not c.get('takes'):
        return 'rest pose'
    count = len(c['takes'])
    text = f"{c.get('source_name', 'source')} ({c.get('result', '')}), {count} take{'s' if count != 1 else ''}"
    return text + (' [needs retarget]' if c.get('needs_retarget') else '')


def cmd_list(args):
    context = open_project(args.project)
    characters = load_characters(context.root)
    if _json_mode(args):
        _print_json(dict(ok=True, root=str(context.root), characters=[_public(c) for c in characters]))
        return 0
    _info(args, f'Characters in {context.root}')
    if not characters:
        print('No imported characters.')
        return 0
    rows = [('ID', 'NAME', 'KIND', 'VISIBLE', 'MOTION')]
    rows += [(c['id'], c['name'], c['kind'], 'yes' if c.get('visible', True) else 'no', _motion_text(c)) for c in characters]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    for row in rows:
        print('  '.join(value.ljust(width) for value, width in zip(row[:4], widths)) + '  ' + row[4])
    return 0


def cmd_sources(args):
    context = open_project(args.project)
    kind = args.result or default_result(context.project)
    sources = character_sources(context.project, kind)
    if _json_mode(args):
        _print_json(dict(ok=bool(sources), project=str(context.project) if context.project else None, result=kind, sources=sources))
    elif not sources:
        print(f'No prepared source motion for result {kind}.')
    else:
        print(f'Source motion for result {kind}:')
        for s in sources:
            count = len(s['takes'])
            print(f"  {s['id']:>3}  {s['name']}  ({count} take{'s' if count != 1 else ''})")
            if not getattr(args, 'quiet', False):
                for take in s['takes']:
                    placed = 'shared scene' if take['transform'] is not None else 'take world'
                    print(f"         start {take['start']:.2f}s  {placed}  {take['project']}")
    return 0 if sources else 1


def cmd_retarget(args):
    context = open_project(args.project)
    c = find_character(load_characters(context.root), args.character)
    kind = args.result or default_result(context.project)
    dry_run = getattr(args, 'dry_run', False)
    sources = character_sources(context.project, kind, not dry_run)
    if not sources:
        raise CharacterError(f'No prepared source motion for result {kind}. Reconstruct or export the source first.')
    source = select_source(sources, args.person)
    stale = stale_exports(source, kind)
    if stale:
        raise CharacterError('Apply contact changes before retargeting: the corrected result is not exported or is out of date '
                             '(run `correct` first) in ' + ', '.join(stale))
    folder = c['id'] + '/animation-' + uuid.uuid4().hex
    job = dict(mode='retarget', output=str(character_path(context.root, folder)), asset=str(character_path(context.root, c['asset'])),
               id=c['id'], animation=folder, result=kind, source_name=source['name'], takes=source['takes'])
    _info(args, f"Retargeting {c['name']} to {source['name']} ({kind}, {len(source['takes'])} take(s))")
    result = start_character_job(context.root, job, getattr(args, 'quiet', False), dry_run)
    if result is None:
        return 0
    c = apply_retarget(context.root, job, result)
    if _json_mode(args):
        _print_json(dict(ok=True, root=str(context.root), character=_public(c), seconds=result.get('seconds')))
    else:
        print(f"Retargeted {c['name']} to {c['source_name']} ({c['result']}), {len(c['takes'])} take(s) in {result.get('seconds', 0):.1f}s")
    return 0


def export_files(c, out, extension):
    """One take: a file (extension appended when missing; a folder gets <name>.<ext>). Several: <name>_take_<n>.<ext> in a folder."""
    from ..person_names import export_stem
    stem = export_stem(c['name'])
    takes = c['takes']
    folder_hint = str(out).endswith(('/', '\\'))
    out = Path(out).expanduser().resolve()
    if len(takes) == 1:
        if out.is_dir() or folder_hint:
            return [str(out / (stem + '.' + extension))]
        if out.suffix.lower() != '.' + extension:
            out = Path(str(out) + '.' + extension)
        return [str(out)]
    if out.exists() and not out.is_dir():
        raise CharacterError(f"{c['name']} has {len(takes)} takes; --out must be a folder.")
    return [str(out / (stem + f'_take_{i + 1}.{extension}')) for i in range(len(takes))]


def cmd_export(args):
    context = open_project(args.project)
    c = find_character(load_characters(context.root), args.character)
    if not c.get('takes'):
        raise CharacterError(f"{c['name']} has no retargeted motion. Run `character retarget` first.")
    if c.get('needs_retarget'):
        raise CharacterError('Character motion is out of date. Apply Correction or retarget it again.')
    extension = args.format
    if extension == 'vmd' and c['kind'] != 'pmx':
        raise CharacterError('VMD export requires a PMX character.')
    options = dict(format=extension, include_mesh=extension == 'fbx' and not args.no_mesh)
    files = export_files(c, args.out, extension)
    existing = [name for name in files if Path(name).exists()]
    if existing and not args.overwrite:
        raise CharacterError('Refusing to replace existing files (pass --overwrite): ' + ', '.join(existing))
    dry_run = getattr(args, 'dry_run', False)
    if not dry_run:
        for name in files:
            Path(name).parent.mkdir(parents=True, exist_ok=True)
    job = dict(mode='export', output=str(character_path(context.root, 'export-' + uuid.uuid4().hex)),
               animation=str(character_path(context.root, c['animation'])), takes=c['takes'], files=files, source_name=c['name'], **options)
    result = start_character_job(context.root, job, getattr(args, 'quiet', False), dry_run)
    if result is None:
        return 0
    if _json_mode(args):
        _print_json(dict(ok=True, character=c['id'], format=extension, files=result['files'], seconds=result.get('seconds')))
    else:
        for name in result['files']:
            print(name)
    return 0


def cmd_remove(args):
    context = open_project(args.project)
    characters = load_characters(context.root)
    c = find_character(characters, args.character)
    if getattr(args, 'dry_run', False):
        _print_json(dict(remove=_public(c), index=str(context.root / 'characters/index.json')))
        return 0
    # Only the registry entry is removed; asset and animation folders stay on disk.
    save_characters(context.root, [x for x in characters if x['id'] != c['id']])
    if _json_mode(args):
        _print_json(dict(ok=True, removed=_public(c)))
    else:
        print(f"Removed {c['name']} ({c['id']}) from the project. Its files were kept.")
    return 0


def _guarded(handler):
    def run(args):
        try:
            return handler(args)
        except WorkerError as exc:
            message = 'Character processing failed: ' + str(getattr(exc, 'message', None) or exc)
            code = getattr(exc, 'code', None)
            status = code if isinstance(code, int) and code > 0 else 1
        except (CharacterError, OSError, ValueError, KeyError, RuntimeError) as exc:
            message, status = str(exc), 1
        except KeyboardInterrupt:
            message, status = 'Cancelled.', 130
        if _json_mode(args):
            _print_json(dict(ok=False, error=message))
        print('error: ' + message, file=sys.stderr, flush=True)
        return status
    run.__name__ = handler.__name__
    return run


def register(subparsers):
    """Add the `character` command group. Handlers are stored as args.func(args) -> int exit code."""
    import argparse
    from ..motion_tools import RESULTS
    # Accept the global flags after the action too; SUPPRESS keeps values given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--json', action='store_true', default=argparse.SUPPRESS, help='machine-readable summary on stdout')
    common.add_argument('--quiet', action='store_true', default=argparse.SUPPRESS, help='less progress output')
    common.add_argument('--dry-run', dest='dry_run', action='store_true', default=argparse.SUPPRESS,
                        help='print the character worker request instead of running it')
    common.add_argument('--project', required=True, metavar='P', help='project folder (workspace root, model project or person take)')

    parser = subparsers.add_parser('character', help='import, retarget and export humanoid characters',
                                   description=f'Characters (Mixamo FBX, VRM, PMX; up to {MAX_CHARACTERS} per project) are stored in '
                                               '<project>/characters/. CHAR is an id, a unique id prefix or a name.')
    actions = parser.add_subparsers(dest='character_command', metavar='ACTION', required=True)
    results = sorted(RESULTS)

    p = actions.add_parser('import', parents=[common], help='import a humanoid character file')
    p.add_argument('file', metavar='FILE', help='.fbx (Mixamo rig), .vrm or .pmx')
    p.set_defaults(func=_guarded(cmd_import))

    p = actions.add_parser('list', parents=[common], help='list imported characters')
    p.set_defaults(func=_guarded(cmd_list))

    p = actions.add_parser('sources', parents=[common], help='list people with motion that can be retargeted')
    p.add_argument('--result', choices=results, metavar='KIND', help='motion result (default: the project selection)')
    p.set_defaults(func=_guarded(cmd_sources))

    p = actions.add_parser('retarget', parents=[common], help='bake a source person\'s motion onto a character')
    p.add_argument('character', metavar='CHAR')
    p.add_argument('--person', required=True, metavar='NAME|ID', help='source person name or id (see `character sources`)')
    p.add_argument('--result', choices=results, metavar='KIND', help='motion result (default: the project selection)')
    p.set_defaults(func=_guarded(cmd_retarget))

    p = actions.add_parser('export', parents=[common], help='export a retargeted character')
    p.add_argument('character', metavar='CHAR')
    p.add_argument('--out', required=True, metavar='PATH', help='output file (one take) or folder (several takes)')
    p.add_argument('--format', choices=('fbx', 'vmd'), default='fbx', help='vmd is available for PMX characters only')
    p.add_argument('--no-mesh', dest='no_mesh', action='store_true', help='FBX: skeleton and animation only')
    p.add_argument('--overwrite', action='store_true', help='replace existing output files')
    p.set_defaults(func=_guarded(cmd_export))

    p = actions.add_parser('remove', parents=[common], help='remove a character from the project (files are kept)')
    p.add_argument('character', metavar='CHAR')
    p.set_defaults(func=_guarded(cmd_remove))
    return parser
