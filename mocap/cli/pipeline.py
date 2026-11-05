"""Job-list builders reproducing gui.input_jobs/run_full, people_ui person jobs and both correction paths."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..motion_tools import MOTION_TOOLS, RESULTS, base_result, has_handflow, result_kind, result_options, result_sources
from .runner import EXMR_DEFAULTS, FLOW_DEFAULTS

MODELS = ('exmr', 'flowhmr', 'gvhmr')
DEFAULT_MODEL = 'exmr'  # the GUI's initial model selection
RESULT_STAGES = {'export', 'flow_export', 'footmr', 'handflow', 'correct'}
POST_STAGES = ('correct', 'export', 'flow_export')
VIDEO_SUFFIXES = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
FLOW_INTERVALS = (0, -1, 1, 2, 5, 10, 30)


class CliError(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.message = message
        self.code = code


class MissingModels(CliError):
    def __init__(self, models):
        self.models = list(models)
        super().__init__('Prepare these models first: ' + ', '.join(model.name for model in self.models), 3)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@dataclass
class Target:
    """A resolved --project: the workspace root, its model project and the people group, when present."""
    workspace: Path
    model: str
    project: Path | None
    people_root: Path | None

    @property
    def root(self):
        return self.people_root or self.project


def resolve_project(value, model=None) -> Target:
    """Accept a model-folder workspace root, a model project (single or people root) or a person take."""
    from .. import project_workspace as workspace
    from ..motion_tools import UnsupportedMotionTool, require_supported_project
    from ..people_store import root_for
    from ..project_storage import project_to_open
    path = Path(value).expanduser().resolve()
    if not (path / 'project.json').is_file():
        raise CliError(f'Not a project folder (no project.json): {path}')
    try:
        if workspace.is_workspace(path):
            state = workspace.read(path)
            model = model or state['active_model']
            if model not in MODELS:
                raise CliError(f'The CLI supports {", ".join(MODELS)}; this project is set to {model}. Pass --model.')
            project = workspace.model_project(path, model, state)
            if not (project / 'project.json').is_file():
                return Target(path, model, None, None)
        else:
            project, _ = project_to_open(path)
        saved = require_supported_project(project).key
        if saved not in MODELS:
            raise CliError(f'The CLI supports {", ".join(MODELS)} projects; this one uses {saved}.')
        if model is not None and model != saved:
            raise CliError(f'This model project uses {saved}, not {model}. Pass the workspace root to switch models.')
        root = workspace.root_for(project)
        if root is None:
            raise CliError('This project uses the previous folder format. Open its video to create a new model-folder project.')
        group = root_for(project)
        return Target(root, saved, project, group if (group / 'people.json').is_file() else None)
    except UnsupportedMotionTool as error:
        raise CliError(str(error)) from None
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, CliError):
            raise
        raise CliError(f'Cannot open project {path}: {error}') from None


# Settings ---------------------------------------------------------------------------------------------------------

def saved_settings(project, model):
    """What gui.load_project restores from a model project: result toggles, correction, ExMR and FlowHMR options."""
    from ..correction_settings import effective_options
    from ..exmr_options import saved as saved_exmr
    from ..flow_sam import saved_options
    from ..flow_camera import saved_interval
    state = read_json(Path(project) / 'project.json')
    result = state.get('pipeline', {}).get('result')
    values = {}
    if result in RESULTS:
        htd, foot = result_options(result)
        values.update(refine=htd, footmr=foot, handflow=has_handflow(result), correction=result.startswith('corrected_'))
    values['correction_options'] = effective_options(Path(project), state)
    values['exmr'] = saved_exmr(Path(project), state)
    values['flow'] = dict(saved_options(Path(project), state), camera_interval=saved_interval(Path(project), state))
    if model != 'gvhmr':
        values.update(refine=False, footmr=False, handflow=False)
    return values


def build_settings(args, model, saved=None):
    """GUI control state: defaults (or a loaded project's values) with explicit command-line overrides."""
    from ..correction_settings import correction_options
    from ..exmr_options import options as exmr_options
    saved = saved or {}
    gvhmr = model == 'gvhmr'
    settings = dict(model=model, refine=saved.get('refine', gvhmr), footmr=saved.get('footmr', gvhmr),
                    handflow=saved.get('handflow', gvhmr), correction=saved.get('correction', True))
    for key in ('refine', 'footmr', 'handflow', 'correction'):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    options = dict(correction_options(saved.get('correction_options')))
    if getattr(args, 'preset', None):
        options['preset'] = args.preset
    if getattr(args, 'floor_mode', None):
        options['mode'] = args.floor_mode
    floor = getattr(args, 'floor', None)
    if floor is not None:
        if str(floor).lower() == 'auto':
            options.update(auto_floor=True, floor=0.)
        else:
            try:
                options.update(auto_floor=False, floor=round(float(floor), 3))
            except ValueError:
                raise CliError('--floor expects a height in meters or "auto"', 2) from None
    if getattr(args, 'pelvis_offset', None) is not None:
        options['pelvis_offset'] = round(float(args.pelvis_offset), 3)
    for key in ('foot_lock', 'hand_lock'):
        if getattr(args, key, None) is not None:
            options[key] = getattr(args, key)
    try:
        settings['correction_options'] = correction_options(options)
    except ValueError as error:
        raise CliError(str(error), 2) from None
    exmr = dict(exmr_options(saved.get('exmr', EXMR_DEFAULTS)))
    if getattr(args, 'camera', None):
        exmr['static_camera'] = args.camera == 'static'
    if getattr(args, 'smoothing', None) is not None:
        exmr['smoothing'] = 'global' if args.smoothing else 'none'
    settings['exmr'] = exmr
    flow = {**FLOW_DEFAULTS, **saved.get('flow', {})}
    camera = getattr(args, 'flow_camera', None)
    if camera is not None:
        interval = {'auto': 0, 'static': -1}.get(str(camera).lower())
        if interval is None:
            try:
                interval = int(camera)
            except ValueError:
                interval = None
        if interval not in FLOW_INTERVALS:
            raise CliError('--flow-camera expects auto, static or a frame interval of 1, 2, 5, 10 or 30', 2)
        flow['camera_interval'] = interval
    for key in ('fast_sam', 'sam_batch4', 'sam_gpu_crop', 'sam_hand_stride4', 'hamer'):
        if getattr(args, key, None) is not None:
            flow[key] = getattr(args, key)
    settings['flow'] = flow
    return settings


def tool_stages(settings):
    """gui.run_full: the selected tool's stage list and result kind."""
    tool = MOTION_TOOLS[settings['model']]
    refine = bool(tool.refinement) and settings['refine']
    foot = tool.key == 'gvhmr' and settings['footmr']
    hand = tool.key == 'gvhmr' and settings['handflow']
    correction = settings['correction']
    return tool.stages(refine, foot, correction, handflow=hand), result_kind(refine, foot, correction, handflow=hand)


def missing_models(settings, stages, people=False):
    """gui.input_jobs model check (plus person detection when people are found in the same run)."""
    from ..models import BY_KEY, missing, ready
    model = settings['model']
    found = []
    for stage in (['people'] if people else []) + list(stages):
        if model == 'flowhmr' and stage == 'flow_camera' and settings['flow']['camera_interval'] == -1:
            continue
        found += missing('flow_export' if model == 'flowhmr' and stage in ('correct', 'export') else stage)
    if model == 'flowhmr' and settings['flow']['hamer']:
        from ..flow_hamer import MODEL_KEYS
        found += [BY_KEY[key] for key in MODEL_KEYS if not ready(BY_KEY[key])]
    return list(dict.fromkeys(found))


def require_models(settings, stages, people=False):
    found = missing_models(settings, stages, people)
    if found:
        raise MissingModels(found)


# Ranges -----------------------------------------------------------------------------------------------------------

def video_range(video, start=None, end=None, offset=0.):
    """gui.processing_range for a freshly opened video: seconds on the original source timeline."""
    from ..video import probe
    try:
        info = probe(Path(video))
    except FileNotFoundError:
        raise CliError(f'Video not found: {video}') from None
    except (OSError, ValueError, KeyError) as error:
        raise CliError(f'Cannot read video {video}: {error}') from None
    if not info['fps'] or info['fps'] <= 0 or info['frames'] <= 0:
        raise CliError(f'Cannot determine the length and frame rate of {video}')
    source_end = offset + info['frames'] / info['fps']
    return _clip(offset, source_end, start, end), info


def project_range(manifest, start=None, end=None):
    """gui.processing_range for a loaded project: its own start, clamped to its frames."""
    first = float(manifest['start'])
    last = min(float(manifest['end']), first + manifest['frames'] / manifest.get('fps', 30))
    return _clip(first, last, start, end)


def _clip(first, last, start, end):
    begin = first if start is None else float(start)
    finish = last if end is None else min(last, float(end))
    if begin < first or finish <= begin:
        raise CliError(f'Check the start and end of the processing range (available {first:.3f}-{last:.3f} s).', 2)
    return begin, finish


def person_frames(root_manifest, first, end):
    """people_ui.input_person_jobs: seconds to frames on the people root's grid."""
    fps = root_manifest['fps']
    return round((first - root_manifest['start']) * fps), round((end - root_manifest['start']) * fps)


# Single person ----------------------------------------------------------------------------------------------------

def prepare_args(video, first, end, model, kind=None, options=None, workspace=None):
    args = ['--video', str(video), '--start', str(float(first)), '--end', str(float(end))]
    if workspace is not None:
        args += ['--project', str(workspace)]
    if kind or model != 'gvhmr':
        args += ['--motion-tool', model]
    if kind:
        args += ['--result', kind]
        if kind.startswith('corrected_') and not options['auto_floor']:
            args += ['--floor', str(options['floor'])]
        if kind.startswith('corrected_'):
            args += ['--correction-preset', options['preset'], '--correction-mode', options['mode'],
                     '--foot-lock', str(int(options['foot_lock'])), '--hand-lock', str(int(options['hand_lock'])),
                     '--pelvis-offset', str(options['pelvis_offset'])]
    return args


def single_jobs(video, first, end, settings, workspace=None):
    """gui.run_full -> input_jobs. Stage arguments stay None/['--result', kind] until the queue knows the project."""
    stages, kind = tool_stages(settings)
    jobs = [('prepare', prepare_args(video, first, end, settings['model'], kind, settings['correction_options'], workspace))]
    jobs += [(stage, ['--result', kind] if stage in RESULT_STAGES else None) for stage in stages]
    return jobs, kind


def find_people_jobs(video, first, end, model, workspace=None):
    """people_ui.find_people -> input_jobs(['people']): prepare without a result, then person discovery."""
    return [('prepare', prepare_args(video, first, end, model, None, None, workspace)), ('people', None)]


# Multi person -----------------------------------------------------------------------------------------------------

def load_group(root):
    from ..people_store import load_people
    try:
        state = load_people(root)
    except (OSError, ValueError, KeyError) as error:
        raise CliError(f'Cannot read the people list: {error}') from None
    if state is None:
        raise CliError(f'No people were found in this project yet. Run `people --project {root}` first.')
    return state


def parse_people(values):
    """'all', 'checked', or ids separated by commas and/or spaces."""
    if not values:
        return None
    tokens = [token for value in values for token in str(value).replace(',', ' ').split()]
    if len(tokens) == 1 and tokens[0].lower() in ('all', 'checked'):
        return tokens[0].lower()
    try:
        ids = [int(token) for token in tokens]
    except ValueError:
        raise CliError('--people expects "all", "checked" or person ids such as 1,3', 2) from None
    if not ids:
        raise CliError('--people expects "all", "checked" or person ids such as 1,3', 2)
    return list(dict.fromkeys(ids))


def select_people(state, selection):
    people = state['people']
    if selection in (None, 'checked'):
        chosen = [person for person in people if person.get('checked')]
        if not chosen:
            raise CliError('No person is checked in this project. Pass --people all or person ids.')
        return chosen
    if selection == 'all':
        return list(people)
    known = {person['id']: person for person in people}
    unknown = [ident for ident in selection if ident not in known]
    if unknown:
        listing = ', '.join(f"{p['id']} ({p['name']})" for p in people)
        raise CliError(f"Unknown person id(s): {', '.join(map(str, unknown))}. Available: {listing}")
    return [known[ident] for ident in selection]


def save_checked(root, state, selected):
    """The GUI persists People-list checkboxes in people.json; mirror checking exactly the selected people."""
    from ..people_store import save_people
    ids = {person['id'] for person in selected}
    changed = False
    for person in state['people']:
        checked = person['id'] in ids
        if bool(person.get('checked')) != checked:
            person['checked'] = checked
            changed = True
    if changed:
        save_people(root, state)


def save_group_options(root, settings):
    """gui.run_full writes the group's ExMR/FlowHMR options and the shared floor before person jobs start."""
    from ..storage import atomic_json
    path = Path(root) / 'project.json'
    state = read_json(path)
    model = settings['model']
    if model == 'exmr':
        state['exmr_options'] = dict(settings['exmr'])
        atomic_json(path, state)
    elif model == 'flowhmr':
        state['flow_options'] = {**state.get('flow_options', {}), **settings['flow']}
        atomic_json(path, state)
    if settings['correction']:
        from ..scene_floor import save_config
        save_config(root, settings['correction_options'])


def person_jobs(root, selected, begin, finish, kind, options):
    jobs = []
    for person in selected:
        args = ['--project', str(root), '--person-id', str(person['id']), '--first-frame', str(begin), '--end-frame', str(finish),
                '--result', kind, '--correction-preset', options['preset'], '--correction-mode', options['mode'],
                '--foot-lock', str(int(options['foot_lock'])), '--hand-lock', str(int(options['hand_lock'])),
                '--pelvis-offset', str(options['pelvis_offset'])]
        if not options['auto_floor']:
            args += ['--floor', str(options['floor'])]
        jobs.append(('person', args))
    return jobs


class PeopleQueue:
    """people_ui.people_event: takes announced by `person` jobs become grouped inference jobs, then deferred post stages."""

    def __init__(self, root, stages, kind, jobs):
        people_stages = [stage for stage in stages if stage != 'track']
        self.inference = [stage for stage in people_stages if stage not in POST_STAGES]
        self.post = [stage for stage in people_stages if stage in POST_STAGES]
        self.root = Path(root)
        self.kind = kind
        self.jobs = jobs
        self.pending_inference = []
        self.pending_post = []
        self.takes = {}

    def add_takes(self, person_id, projects):
        from ..queue_groups import grouped_jobs
        self.takes.setdefault(person_id, []).extend(projects)
        self.pending_inference += [(stage, ['--project', path, '--result', self.kind]) for path in projects for stage in self.inference]
        self.pending_post += [(stage, ['--project', path, '--result', self.kind]) for path in projects for stage in self.post]
        # Prepare every actor's camera/motion before the one shared floor resolves and any corrected export is committed.
        if not any(stage == 'person' for stage, _ in self.jobs):
            self.jobs[:] = grouped_jobs(self.pending_inference, self.root) + self.jobs + self.pending_post
            self.pending_inference, self.pending_post = [], []

    def on_event(self, event):
        if event.get('type') == 'person_prepared':
            self.add_takes(event.get('person_id'), [str(path) for path in event.get('projects') or []])


def simulate_people(root, stages, kind, jobs, takes):
    """Dry-run expansion with placeholder takes; real take folders come from each person_prepared event."""
    queue = PeopleQueue(root, stages, kind, list(jobs))
    planned = []
    while queue.jobs:
        stage, args = queue.jobs.pop(0)
        planned.append((stage, args))
        if stage == 'person':
            ident = args[args.index('--person-id') + 1]
            queue.add_takes(ident, takes(ident))
    return planned


# Correction -------------------------------------------------------------------------------------------------------

def correction_kind(kind):
    """gui.apply_correction uses selected_result(True): the same toggles with correction forced on."""
    return 'corrected_' + base_result(kind)


def single_correction_jobs(project, kind):
    return [('correct', ['--project', str(project), '--result', kind]), ('export', ['--project', str(project), '--result', kind])]


def scene_correction_jobs(root, state, kind, selected_ids):
    """people_ui.apply_scene_correction: every available take is rebuilt against the one shared plane."""
    from ..motion_tools import result_is_current
    from ..people_store import safe_child
    jobs = []
    takes = []
    for person in state['people']:
        for take in person.get('takes', []):
            project = safe_child(root, take['project'])
            bases = result_sources(base_result(kind))
            available = all((project / RESULTS[base][0]).is_file() and result_is_current(project, base) for base in bases)
            if not available:
                if person['id'] in selected_ids:
                    raise CliError(f"{person['name']}: selected base motion is unavailable. Reconstruct it first.")
                continue
            takes.append((person, project))
            jobs.extend((stage, ['--project', str(project), '--result', kind]) for stage in ('correct', 'export'))
    if not jobs:
        raise CliError('Reconstruct a person before applying the shared scene floor.')
    return jobs, takes


def save_correction(project, options):
    from ..storage import atomic_json
    path = Path(project) / 'project.json'
    state = read_json(path)
    state['correction'] = dict(options)
    atomic_json(path, state)


def current_kind(project, model):
    """The result gui.load_project selects: pipeline.result, export.json, or the legacy initial/refined default."""
    project = Path(project)
    state = read_json(project / 'project.json')
    kind = state.get('pipeline', {}).get('result')
    if kind not in RESULTS:
        try:
            kind = read_json(project / 'export.json').get('result_kind')
        except (OSError, ValueError):
            kind = None
    if kind not in RESULTS:
        has_refined = any((project / name).exists() for name in ('opt_results.pt', 'refined_joints.npz', 'refined_mesh.json'))
        kind = 'refined' if has_refined else 'initial'
    htd, foot = result_options(kind)
    hand = has_handflow(kind)
    if model != 'gvhmr':
        htd = foot = hand = False
    return result_kind(htd, foot, kind.startswith('corrected_'), handflow=hand)
