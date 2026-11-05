"""Headless RGB MoCap command line using the private common runtime (see docs/CLI.md)."""
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
# --view starts the GUI with the caller's own environment, before runtime activation changes it.
launch_environment = dict(os.environ)
os.environ.pop('PYTHONHOME', None)
os.environ.pop('PYTHONPATH', None)
os.environ.update(RGB_MOCAP_MANAGED='1', RGB_MOCAP_DATA=str(root),
                  PYTHONDONTWRITEBYTECODE='1', HF_HOME=str(root / 'cache/huggingface'),
                  TORCH_HOME=str(root / 'cache/torch'), XDG_CACHE_HOME=str(root / 'cache'))
sys.dont_write_bytecode = True
EXIT_BUSY = 4


def lock_needed(argv):
    # These only read projects (review/fix/render write to their own output folders), so they also work while the GUI is open.
    if '--no-lock' in argv or '-h' in argv or '--help' in argv or '--dry-run' in argv:
        return False
    words = [value for value in argv if not value.startswith('-')]
    if words[:1] == ['contacts']:
        # Only listing and plain suggestions (review is read-only, `automatic` is advisory) run beside the GUI.
        action = words[1] if len(words) > 1 else None
        return not (action == 'list' or (action == 'suggest' and '--accept' not in argv))
    return (words[0] if words else None) not in ('models', 'review', 'fix', 'render')


def check_install():
    if (root / '.setup/git-sync.pending').exists():
        raise SystemExit('Update was not completed. Run install.cmd to repair before launching.')
    try:
        state = json.loads((root / '.setup/git-runtime-state.json').read_text(encoding='utf8'))
        for key, path in [('lock_sha256', 'installer/runtime-lock.json'), ('registry_sha256', 'mocap/runtime_registry.py'),
                          ('support_sha256', 'installer/git-support-lock.json'), ('assets_sha256', 'installer/external-assets.json')]:
            if state[key] != hashlib.sha256((root / path).read_bytes()).hexdigest():
                raise ValueError('Installed dependencies do not match this source version.')
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit('Run install.cmd before launching: ' + str(exc)) from None


def run():
    from tools.session_lock import app_lock
    argv = sys.argv[1:]
    with ExitStack() as stack:
        if lock_needed(argv):
            try:
                stack.enter_context(app_lock(root))
            except RuntimeError as exc:
                print('error: RGB MoCap, an installation or an update is running. ' + str(exc) +
                      ' (Read-only --dry-run works while it is open; --no-lock skips this check.)', file=sys.stderr)
                return EXIT_BUSY
        check_install()
        from mocap.runtime_registry import activate
        activate(root, 'gui')
        from mocap.cli import main as cli_main
        code = cli_main.main(argv)
    if cli_main.VIEW_PROJECT is not None:
        from mocap.cli.viewer import launch
        launch(cli_main.VIEW_PROJECT, launch_environment)
    return code


if __name__ == '__main__':
    raise SystemExit(run())
