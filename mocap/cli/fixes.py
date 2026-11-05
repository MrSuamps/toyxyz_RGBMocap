"""`fix` command: apply IK pins and smoothing from a JSON spec to an FBX animation with the bundled Blender, headless.

Spec checks and bone resolution live in fix_spec.py (shared with the Blender side, blender_fix.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import deque
from pathlib import Path

from . import fix_spec
from .pipeline import CliError

EXIT_USAGE = 2
BLENDER_SCRIPT = Path(__file__).resolve().with_name('blender_fix.py')
PROGRESS = 'FIX: '

DESCRIPTION = '''Apply animation corrections from a JSON spec to an FBX: IK pins (a hand or foot held to another bone or to a
fixed point) and temporal smoothing. Runs the bundled Blender headless. The output keeps the input's skeleton, mesh,
bone names, frame range and fps; only the animation inside the pinned / smoothed frames changes.'''

EPILOG = '''spec (JSON):
  {"version": 1,
   "pins": [{"effector": "RightHand", "target": "RightFoot", "frames": [381, 421], "ease": 6,
             "distance": 0.08, "offset": [0, 0, 0], "chain": 2, "rotation": "keep"}],
   "smooth": [{"bones": ["RightHand", "RightForeArm"], "frames": [100, 140], "window": 5}]}

  effector  LeftHand/RightHand/LeftFoot/RightFoot/LeftToeBase/RightToeBase (a "prefix:" namespace is ignored)
  target    bone to stick to, or "world" to hold the effector on a fixed point
  frames    inclusive range, numbered as Blender shows them after a default FBX import (first key = frame 1)
  ease      frames to blend in before and out after the range (smoothstep)
  distance  effector-target distance in meters along the direction at the closest frame in the range
  offset    effector position in the target bone's local space, meters (world pins: added to the held point)
  chain     2 (only value supported)    rotation  keep (default) or follow
  See docs/CLI.md for details.'''


def register(commands, common):
    p = commands.add_parser('fix', parents=[common], help='apply IK pins / smoothing from a JSON spec to an FBX animation',
                            description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', metavar='INPUT.fbx', help='FBX with one animated armature')
    p.add_argument('--spec', required=True, metavar='FILE|-', help='fix spec JSON file, or - to read it from stdin')
    p.add_argument('--out', required=True, metavar='OUTPUT.fbx', help='corrected FBX to write')
    p.add_argument('--report', metavar='REPORT.json', help='also write the per-pin report as JSON')
    p.add_argument('--overwrite', action='store_true', help='replace existing output / report files')
    p.set_defaults(handler=cmd_fix)


def _fbx_path(value, what):
    path = Path(value).expanduser().resolve()
    if path.suffix.casefold() != '.fbx':
        raise CliError(f'{what} must be an .fbx file: {path}', EXIT_USAGE)
    return path


def _read_spec(value):
    name = 'stdin' if value == '-' else str(Path(value).expanduser().resolve())
    try:
        text = sys.stdin.read() if value == '-' else Path(name).read_text(encoding='utf-8-sig')
    except OSError as exc:
        raise CliError(f'Cannot read the fix spec {name}: {exc}', EXIT_USAGE) from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CliError(f'Fix spec {name} is not valid JSON: {exc}', EXIT_USAGE) from None


def _blender_command(blender, request):
    return [str(blender), '-b', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1', '--python', str(BLENDER_SCRIPT), '--', str(request)]


def _blender():
    from ..retarget.runtime import blender
    try:
        return blender()
    except RuntimeError as exc:
        raise CliError(str(exc)) from None


def cmd_fix(args, session):
    source = _fbx_path(args.input, 'INPUT')
    if not source.is_file():
        raise CliError(f'Input not found: {source}', EXIT_USAGE)
    target = _fbx_path(args.out, '--out')
    if os.path.normcase(target) == os.path.normcase(source):
        raise CliError('--out must differ from the input file', EXIT_USAGE)
    report = Path(args.report).expanduser().resolve() if args.report else None
    for path in (target, report):
        if path is not None and path.exists() and not args.overwrite:
            raise CliError(f'{path} already exists. Pass --overwrite to replace it.')
    try:
        spec = fix_spec.validate(_read_spec(args.spec))
        try:
            skeleton, clip = fix_spec.read_fbx(source)
            unresolved = None
        except fix_spec.FbxError as exc:
            skeleton = clip = None
            unresolved = f'bones and frames are checked in Blender ({exc})'
        plan = fix_spec.plan(spec, skeleton, clip)
    except fix_spec.SpecError as exc:
        raise CliError(f'Invalid fix spec: {exc}', EXIT_USAGE) from None
    if unresolved:
        plan['note'] = unresolved
    summary = dict(command='fix', input=str(source), output=str(target), report=str(report) if report else None)
    if session.dry_run:
        try:
            command = _blender_command(_blender(), '<request.json>')
        except CliError as exc:
            command = 'unavailable: ' + exc.message
        summary.update(dry_run=True, spec=spec, plan=plan, blender=command)
        if not session.json:
            print(json.dumps(dict(spec=spec, plan=plan, blender=command), indent=2, ensure_ascii=False), flush=True)
        return 0, summary
    result = _run_blender(source, target, spec, session)
    if report is not None:
        from ..storage import atomic_json
        atomic_json(report, dict(result, output=str(target)))
    summary.update({key: value for key, value in result.items() if key != 'ok'})
    _print_result(session, source, target, report, result)
    return 0, summary


def _run_blender(source, target, spec, session):
    blender = _blender()
    work = Path(tempfile.mkdtemp(prefix='rgbmocap-fix-'))
    partial = target.with_name('.' + target.stem + '-' + uuid.uuid4().hex + '.partial.fbx')
    result_path = work / 'result.json'
    request = work / 'request.json'
    request.write_text(json.dumps(dict(input=str(source), output=str(partial), result=str(result_path), spec=spec)), encoding='utf-8')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    tail = deque(maxlen=30)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(_blender_command(blender, request), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8',
                                   errors='replace', env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            for line in process.stdout:
                tail.append(line)
                if line.startswith(PROGRESS):
                    session.info(line[len(PROGRESS):].rstrip())
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else None
        if result is not None and not result.get('ok'):
            raise CliError(result['error'], EXIT_USAGE if result.get('usage') else 1)
        if code or result is None or not partial.is_file():
            raise CliError(f'Blender fix failed (exit {code}):\n' + ''.join(tail)[-2500:])
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    return result


def _print_result(session, source, target, report, result):
    session.say(f'Fixed {source.name} -> {target} ({len(result["pins"])} pin(s), {len(result["smooth"])} smooth range(s), '
                f'{result["blender_seconds"]} s)')
    for i, pin in enumerate(result['pins'], 1):
        before, after = pin['distance_before'], pin['distance_after']
        session.say(f'  pin {i} {pin["effector"]} -> {pin["target"]}, frames {pin["frames"][0]}-{pin["frames"][1]}: '
                    f'distance {before["start"]:.3f}/{before["mid"]:.3f}/{before["end"]:.3f} -> {after["start"]:.3f}/{after["mid"]:.3f}/'
                    f'{after["end"]:.3f} m (start/mid/end, wanted {pin["desired_distance"]:.3f}, max error {pin["max_error"]:.4f}); '
                    f'{pin["mid_joint"]} moved up to {pin["mid_joint_max_shift"]:.3f} m; {pin["frames_changed_count"]} frames changed')
        failures = pin['reach_failures']
        if failures['frames']:
            session.say(f'    warning: target out of reach on {failures["frames"]} frame(s) (from {failures["first_frames"][0]}), '
                        f'residual up to {failures["max_residual"]:.3f} m')
    for i, item in enumerate(result['smooth'], 1):
        largest = max(item['max_change_deg'].values(), default=0.0)
        session.say(f'  smooth {i} {", ".join(item["bones"])}, frames {item["frames"][0]}-{item["frames"][1]}, window {item["window"]}: '
                    f'rotations changed up to {largest:.2f} deg')
    if report is not None:
        session.say(f'Report: {report}')
