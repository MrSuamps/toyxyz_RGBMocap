"""Source motion export (FBX, BVH, Blender) through the GUI's request files and save workers."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import CliError, current_kind, load_group, select_people
from .runner import run_stage

FORMATS = ('fbx', 'bvh', 'blend')
STAGES = {'fbx': 'save_fbx', 'bvh': 'copy_bvh', 'blend': 'save_blend'}


class StaleExport(CliError):
    pass


@dataclass
class ExportOptions:
    fmt: str = 'fbx'
    bvh_profile: str = 'generic'
    prepend_tpose: bool = False
    animate_camera: bool = True
    background_refine: bool = False
    camera_translation_scale: float = 1.
    overwrite: bool = False


@dataclass
class ExportPlan:
    stage: str
    request_project: Path
    rows: list = field(default_factory=list)

    @property
    def targets(self):
        return [Path(row['target']) for row in self.rows]


def resolve_format(out, fmt=None):
    suffix = Path(out).suffix.lower().lstrip('.')
    if suffix in FORMATS:
        if fmt and fmt != suffix:
            raise CliError(f'--out ends in .{suffix} but --format is {fmt}.', 2)
        return suffix
    return fmt or 'fbx'


def options_from_args(args, out, model):
    fmt = resolve_format(out, getattr(args, 'format', None))
    if fmt != 'bvh' and getattr(args, 'bvh_profile', None) not in (None, 'generic'):
        raise CliError('--bvh-profile applies to BVH export only.', 2)
    scale = getattr(args, 'camera_scale', None)
    scale = 1. if scale is None else float(scale)
    if not 0 <= scale <= 100:
        raise CliError('--camera-scale must be between 0 and 100.', 2)
    animate = not getattr(args, 'no_camera_animation', False)
    # source_export_dialog: Refine Camera Rotation only with an animated camera, for FlowHMR/ExMR.
    refine = bool(getattr(args, 'refine_camera_rotation', False)) and animate and model in ('flowhmr', 'exmr')
    options = ExportOptions(fmt=fmt, bvh_profile=getattr(args, 'bvh_profile', None) or 'generic', prepend_tpose=bool(getattr(args, 'tpose', False)),
                            overwrite=bool(getattr(args, 'overwrite', False)))
    if fmt == 'blend':
        options.animate_camera, options.background_refine, options.camera_translation_scale = animate, refine, round(scale, 3)
    return options


def blend_targets(files):
    """blend_export_ui.targets: X/Name.blend becomes X/Name/Name.blend in a new, numbered folder."""
    used = set()
    result = []
    for project, kind, source, target, name in files:
        target = Path(target)
        folder = target.with_suffix('')
        number = 2
        while folder.exists() or str(folder.resolve()).casefold() in used:
            folder = target.parent / (target.stem + '_' + str(number))
            number += 1
        used.add(str(folder.resolve()).casefold())
        result.append((project, kind, source, folder / target.name, name))
    return result


def single_target(out, stem, options):
    out = Path(out).expanduser().resolve()
    suffix = '.' + options.fmt
    target = out if out.suffix.lower() == suffix else out / (stem + suffix)
    if options.fmt != 'blend' and target.exists() and not options.overwrite:
        raise CliError(f'{target} already exists. Pass --overwrite to replace it.')
    if target.is_dir():
        raise CliError(f'{target} is a folder.')
    return target


def _source(project, kind, multi):
    from ..motion_tools import result_is_current, selected_export
    if multi and not result_is_current(project, kind):
        return None
    return selected_export(project, kind)


def _stale(name, kind):
    return StaleExport(f'{name}: the {kind} motion is unavailable, out of date or has pending correction. '
                       'Run `correct` (or `reconstruct`) and export again.')


def plan_export(target, out, options, people=None, kind=None, stem=None):
    """gui.save_fbx / people_ui.export_checked_people: one request row per exported take."""
    from ..person_names import export_path, export_stem
    from ..people_store import safe_child
    if target.project is None:
        raise CliError(f'This project has no saved {target.model} motion. Reconstruct it first.')
    files = []
    scene_root = target.people_root
    if scene_root is None:
        if people is not None:
            raise CliError('This project has no people list; --people applies to projects processed with `people`.', 2)
        from ..single_actor import preferences
        project = target.project
        kind = kind or current_kind(project, target.model)
        name = preferences(project)['name']
        source = _source(project, kind, False)
        if source is None:
            raise _stale(name, kind)
        files.append((project, kind, source, single_target(out, stem or export_stem(name), options), name))
    else:
        state = load_group(scene_root)
        selected = select_people(state, people)
        takes = []
        for person in selected:
            if not person.get('takes'):
                raise CliError(f"No motion for {person['name']}. Reconstruct it first.")
            for number, take in enumerate(person['takes'], 1):
                takes.append((person, number, len(person['takes']), safe_child(scene_root, take['project'])))
        if kind is None:
            kind = current_kind(takes[0][3], target.model)
        if len(takes) == 1:
            person, _, _, project = takes[0]
            source = _source(project, kind, False)
            if source is None:
                raise _stale(person['name'], kind)
            files.append((project, kind, source, single_target(out, stem or export_stem(person['name']), options), person['name']))
        else:
            if options.fmt == 'blend':
                raise CliError('Blender export supports one take at a time. Select one person with one take (--people ID).', 2)
            folder = Path(out).expanduser().resolve()
            if folder.suffix.lower() in ('.fbx', '.bvh', '.blend') and not folder.is_dir():
                raise CliError(f'{len(takes)} takes are selected; --out must be a folder.', 2)
            if folder.exists() and not folder.is_dir():
                raise CliError(f'{folder} is not a folder.')
            used = {path.name.casefold() for path in folder.iterdir()} if folder.is_dir() else set()
            batch = set()
            suffix = '.' + options.fmt
            for person, number, count, project in takes:
                source = _source(project, kind, True)
                if source is None:
                    raise _stale(f"{person['name']}, take {number}", kind)
                name = export_stem(person['name']) + (f'_take_{number:02d}' if count > 1 else '')
                if options.overwrite:
                    path, copy = folder / (name + suffix), 2
                    while path.name.casefold() in batch:
                        path, copy = folder / f'{name}_{copy}{suffix}', copy + 1
                    batch.add(path.name.casefold())
                else:
                    path = export_path(folder, name, used, suffix)
                files.append((project, kind, source, path, person['name']))
    if options.fmt == 'blend':
        files = blend_targets(files)
    rows = []
    for project, kind, source, path, name in files:
        row = dict(project=str(project), kind=kind, source=str(source), target=str(path), rig_name=name,
                   prepend_tpose=options.prepend_tpose, bvh_profile=options.bvh_profile if options.fmt == 'bvh' else 'generic',
                   animate_camera=options.animate_camera, background_refine=options.background_refine,
                   camera_translation_scale=options.camera_translation_scale)
        if scene_root is not None:
            from ..scene_export import context
            try:
                row['scene'] = context(scene_root, project)
            except (OSError, ValueError, KeyError) as error:
                raise CliError('Cannot export shared scene: ' + str(error)) from None
        rows.append(row)
    return ExportPlan(STAGES[options.fmt], Path(files[0][0]), rows)


def run_export(plan, env=None, quiet=False):
    """gui.submit_fbx_exports + cleanup_fbx_export. The writer deletes the request; leftovers are removed here."""
    from ..storage import atomic_json
    request = plan.request_project / ('.fbx-export-' + uuid.uuid4().hex + '.json')
    for path in plan.targets:
        # A Blender bundle folder must not exist yet; the writer creates it.
        (path.parent.parent if plan.stage == 'save_blend' else path.parent).mkdir(parents=True, exist_ok=True)
    if plan.stage == 'save_blend':
        from ..blend_support import partial
        leftovers = [partial(path.parent, request) for path in plan.targets]
    else:
        leftovers = [path.with_name('.' + path.name + '.rgbmocap-' + request.stem + '.partial') for path in plan.targets]
    atomic_json(request, plan.rows)
    try:
        run_stage(plan.stage, ['--project', str(plan.request_project), '--output', str(request)], env, quiet=quiet)
    finally:
        for path in [request, *leftovers]:
            try:
                if path.is_dir():
                    from ..blend_support import cleanup
                    cleanup(path, request)
                else:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
    return [str(path) for path in plan.targets]
