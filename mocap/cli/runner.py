"""Headless worker execution: the GUI's worker environment, event protocol and sequential queue rules."""
from __future__ import annotations

import json
import os
import queue as queues
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..paths import ROOT, worker_command

PREFIX = '@@MOCAP@@'
CHARACTER_PREFIX = 'CHARACTER:'
EXIT_BLOCKED = 3
BATCH_STAGES = ('flow_batch', 'exmr_batch', 'gvhmr_batch')
# gui.next_job prepends the current project only to these when their arguments lack one.
PROJECT_STAGES = {'export', 'footmr', 'handflow', 'correct', 'flow_export'}
FLOW_DEFAULTS = dict(camera_interval=0, fast_sam=True, sam_batch4=True, sam_gpu_crop=True, sam_hand_stride4=True, hamer=False)
EXMR_DEFAULTS = dict(static_camera=True, smoothing='global')

# Progress and worker log lines go here; main switches it to stderr for --json.
progress_stream = None


class WorkerError(RuntimeError):
    def __init__(self, stage, message, code=None, blocked_models=None):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.blocked_models = blocked_models


def _out():
    return progress_stream or sys.stdout


def _flag(value):
    return '1' if value else '0'


def worker_env(compute_mode='auto', resource_safety=True, flow=None, exmr=None) -> dict:
    """Variables gui.next_job adds to the system environment. FLOW_* and EXMR_OPTIONS are read only by `prepare`."""
    if compute_mode not in ('auto', 'low_memory'):
        raise ValueError('Unknown compute mode: ' + str(compute_mode))
    flow = {**FLOW_DEFAULTS, **(flow or {})}
    exmr = {**EXMR_DEFAULTS, **(exmr or {})}
    return {
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUNBUFFERED': '1',
        'RGB_MOCAP_COMPUTE_MODE': compute_mode,
        'RGB_MOCAP_RESOURCE_SAFETY': _flag(resource_safety),
        'RGB_MOCAP_FLOW_INTERVAL': str(int(flow['camera_interval'])),
        'RGB_MOCAP_FLOW_FAST_SAM': _flag(flow['fast_sam']),
        'RGB_MOCAP_FLOW_BATCH4': _flag(flow['sam_batch4']),
        'RGB_MOCAP_FLOW_GPU_CROP': _flag(flow['sam_gpu_crop']),
        'RGB_MOCAP_FLOW_HAND_STRIDE4': _flag(flow['sam_hand_stride4']),
        'RGB_MOCAP_FLOW_HAMER': _flag(flow['hamer']),
        'RGB_MOCAP_EXMR_OPTIONS': json.dumps({'static_camera': exmr['static_camera'], 'smoothing': exmr['smoothing']}),
    }


class _Reporter:
    """One concise line per event; repeated progress messages are limited to one per second."""

    def __init__(self, stage, quiet):
        self.stage = stage
        self.quiet = quiet
        self.last_message = None
        self.last_time = 0.

    def write(self, text):
        if not self.quiet:
            print(f'[{self.stage}] {text}', file=_out(), flush=True)

    def event(self, event):
        kind = event.get('type')
        if kind == 'progress':
            message = str(event.get('message') or 'Processing...')
            now = time.monotonic()
            if message == self.last_message and now - self.last_time < 1.:
                return
            self.last_message, self.last_time = message, now
            current, total = event.get('current'), event.get('total')
            if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                message += f' {current:g}/{total:g} ({100 * current / total:.0f}%)'
            self.write(message)
        elif kind == 'complete' and not event.get('batch_inner'):
            elapsed = event.get('elapsed')
            text = 'done' + (f' in {elapsed:.1f} s' if isinstance(elapsed, (int, float)) else '') + (' (cached)' if event.get('cached') else '')
            self.write(text)
        elif kind == 'batch_item_start':
            self.write(f"actor/take {event.get('index')}/{event.get('count')}: {event.get('stage')} {event.get('project', '')}")
        elif kind in ('info', 'resource_mode'):
            self.write(str(event.get('message', '')))
        elif kind == 'prepared':
            self.write('project ' + str(event.get('project')))
        elif kind == 'people_found':
            self.write(f"{event.get('count')} people found" + (' (cached)' if event.get('cached') else ''))
        elif kind == 'person_prepared':
            self.write(f"person {event.get('person_id')}: {len(event.get('projects') or [])} take(s)")


def _pump(stream, sink):
    try:
        for raw in iter(stream.readline, b''):
            sink.put(raw)
    finally:
        sink.put(None)


def _forward_stderr(stream):
    target = sys.stderr
    for raw in iter(stream.readline, b''):
        try:
            target.write(raw.decode('utf-8', 'replace'))
            target.flush()
        except (OSError, ValueError):
            pass


def _kill_tree(process):
    if process.poll() is not None:
        return
    # Stop FFmpeg and other descendants before losing their parent PID, as gui.cancel_job does.
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_stage(stage: str, args: list[str], env: dict | None = None, on_event=None, quiet=False) -> list[dict]:
    """Run one worker stage to completion and return its parsed events; raise WorkerError on failure."""
    args = [str(value) for value in args]
    try:
        command = worker_command(stage, *args)
    except RuntimeError as error:
        raise WorkerError(stage, str(error)) from None
    environment = dict(os.environ)
    environment.update(env or {})
    # A separate group keeps Ctrl+C away from the worker; we stop the whole tree ourselves.
    flags = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    try:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    except OSError as error:
        raise WorkerError(stage, 'Cannot start the processing engine: ' + str(error)) from None
    reporter = _Reporter(stage, quiet)
    lines = queues.Queue()
    reader = threading.Thread(target=_pump, args=(process.stdout, lines), daemon=True)
    errors = threading.Thread(target=_forward_stderr, args=(process.stderr,), daemon=True)
    reader.start()
    errors.start()
    events = []
    failure = None
    blocked = None
    try:
        while True:
            # A timed wait keeps Ctrl+C responsive while the worker is silent.
            try:
                raw = lines.get(timeout=.25)
            except queues.Empty:
                continue
            if raw is None:
                break
            line = raw.decode('utf-8', 'replace').rstrip('\r\n')
            if line.startswith(PREFIX):
                try:
                    event = json.loads(line[len(PREFIX):])
                except ValueError:
                    reporter.write(line)
                    continue
                if not isinstance(event, dict):
                    continue
                events.append(event)
                kind = event.get('type')
                if kind == 'failed':
                    failure = str(event.get('message') or 'Processing failed. See the messages above.')
                elif kind == 'blocked':
                    blocked = [str(key) for key in event.get('models') or []]
                    failure = str(event.get('message') or 'Prepare the required models first.')
                reporter.event(event)
                if on_event is not None:
                    on_event(event)
            elif line.startswith(CHARACTER_PREFIX):
                reporter.write(line.partition(':')[2].strip())
            elif line.strip():
                reporter.write(line)
        code = process.wait()
    except BaseException:
        _kill_tree(process)
        raise
    finally:
        reader.join(timeout=5)
        errors.join(timeout=5)
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass
    if code != 0 or failure is not None or blocked is not None:
        if code == 0:
            code = EXIT_BLOCKED if blocked is not None else 1
        message = failure or f'{stage} stopped with exit code {code}. See the messages above.'
        raise WorkerError(stage, message, code, blocked)
    return events


def _argument(args, name):
    return args[args.index(name) + 1]


def _complete(stage, arguments, project):
    """gui.next_job argument completion: None means ['--project', current]; result stages get it prepended."""
    if arguments is None:
        if project is None:
            raise WorkerError(stage, 'No project is available for this stage.')
        return ['--project', str(project)]
    arguments = [str(value) for value in arguments]
    if stage in PROJECT_STAGES and '--project' not in arguments:
        if project is None:
            raise WorkerError(stage, 'No project is available for this stage.')
        return ['--project', str(project), *arguments]
    return arguments


def prepared_project(event):
    """Resolve and record the model project announced by `prepare`, as gui.handle_event does."""
    project = Path(event['project']).resolve()
    if not (project / 'project.json').is_file():
        raise WorkerError('prepare', f'The prepared project has no project.json: {project}')
    from ..project_storage import storage_root
    from ..project_workspace import is_workspace
    saved_root = storage_root(project)
    if saved_root is not None and not is_workspace(saved_root):
        from ..storage import atomic_json
        atomic_json(saved_root / 'workspace_state.json', dict(version=1, name=saved_root.name,
                    active=project.relative_to(saved_root).as_posix(), frame=0, trim=None))
    return project


def run_jobs(jobs: list[tuple[str, list[str]]], env=None, on_event=None, quiet=False, project=None) -> list[dict]:
    """Sequential queue mirroring gui.next_job. `jobs` is consumed in place, so on_event may extend it."""
    from ..queue_groups import join_single_pipeline, write_request, remove_request
    queue = jobs
    events = []
    state = {'project': Path(project) if project is not None else None}

    def handle(event):
        if event.get('type') == 'prepared':
            state['project'] = prepared_project(event)
        if on_event is not None:
            on_event(event)

    while queue:
        queue[:] = join_single_pipeline(queue, state['project'])
        stage, arguments = queue.pop(0)
        arguments = _complete(stage, arguments, state['project'])
        request = None
        try:
            if stage in BATCH_STAGES:
                try:
                    request = write_request(stage, arguments)
                except (OSError, ValueError) as error:
                    raise WorkerError(stage, 'Cannot prepare people processing: ' + str(error)) from None
                arguments = ['--request', str(request), '--project', _argument(arguments, '--project')]
            events += run_stage(stage, arguments, env, handle, quiet)
        finally:
            if request is not None:
                try:
                    remove_request(request)
                except OSError as error:
                    print('Could not remove worker request: ' + str(error), file=sys.stderr)
    return events


def plan(jobs, project=None, placeholder='<project>'):
    """Static run_jobs simulation for --dry-run: the launched arguments plus each batch request body."""
    from ..queue_groups import join_single_pipeline
    queue = list(jobs)
    current = project
    rows = []
    while queue:
        queue = join_single_pipeline(queue, current)
        stage, arguments = queue.pop(0)
        if current is None and (arguments is None or (stage in PROJECT_STAGES and '--project' not in arguments)):
            current = placeholder
        row = dict(stage=stage, args=_complete(stage, arguments, current))
        if stage in BATCH_STAGES:
            args = row['args']
            row['request'] = dict(version=1, root=_argument(args, '--batch-root'), phase=_argument(args, '--batch-phase'),
                                  jobs=json.loads(_argument(args, '--batch-items')))
            row['args'] = ['--request', '<request>', '--project', _argument(args, '--project')]
        rows.append(row)
        if stage == 'prepare' and current is None:
            current = placeholder
    return rows


def format_plan(rows, stream=None):
    stream = stream or _out()
    print(f'Planned jobs ({len(rows)}):', file=stream)
    for number, row in enumerate(rows, 1):
        print(f'  {number:>2}. ' + subprocess.list2cmdline([row['stage'], *row['args']]), file=stream)
        if 'request' in row:
            print('      request: ' + json.dumps(row['request'], ensure_ascii=False), file=stream)
    stream.flush()
