"""--view: open a finished result in the RGB MoCap GUI, the same way RGBMoCap.cmd starts it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def view_target(args, summary, code) -> Path | None:
    """Project the GUI should open after this command, or None when there is nothing to show."""
    if not getattr(args, 'view', False) or getattr(args, 'dry_run', False) or code != 0:
        return None
    summary = summary or {}
    candidates = [summary.get('workspace'), summary.get('project')]
    if summary.get('command') == 'batch':
        # Batch produces one project per clip; open the last one that succeeded.
        done = [row.get('project') for row in summary.get('clips', []) if row.get('status') == 'ok']
        candidates = done[-1:]
    candidates.append(getattr(args, 'project', None))
    for value in candidates:
        if value and Path(value).is_dir() and (Path(value) / 'project.json').is_file():
            return Path(value).resolve()
    return None


def gui_running() -> bool:
    sys.path.insert(0, str(ROOT))
    from tools.session_lock import app_lock
    try:
        with app_lock(ROOT):
            return False
    except RuntimeError:
        return True


def launch(project: Path, environment: dict) -> int:
    """Start the GUI in its own console window and return immediately. Call only after the CLI released the app lock."""
    if gui_running():
        print(f'RGB MoCap is already open. Use File > Open Project... and select:\n  {project}', file=sys.stderr)
        return 0
    command = ['cmd.exe', '/c', str(ROOT / 'RGBMoCap.cmd'), '--project', str(project)]
    subprocess.Popen(command, cwd=str(ROOT), env=environment, creationflags=subprocess.CREATE_NEW_CONSOLE, close_fds=True)
    print(f'Opening RGB MoCap: {project}', file=sys.stderr)
    return 0
