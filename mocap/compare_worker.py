"""Display arrays for the original-vs-corrected comparison, built in an isolated CPU process.

Reads the pre-self-contact pose and ExMR's 2D keypoints (Torch files the editor must not load) and writes mesh frames,
joints and keypoints into one cache folder; the project is never written.
"""
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np

SCHEMA = 1
SHAPES = slice(1366, 1386)
TRANSL = slice(1386, 1389)


def sources(project, kind):
    """Files the comparison depends on, with their stats; absent optional inputs are omitted."""
    from .result_artifacts import file_stat
    from .motion_tools import RESULTS
    from .self_contact_store import before_current, before_path
    project = Path(project)
    names = [f'{kind}_joints.npz', f'{kind}_mesh.json', RESULTS[kind][0]]
    if before_current(project, kind):
        names.append(before_path(project, kind).name)
        mesh = project / f'{kind}_mesh.json'
        if mesh.is_file():
            names.append(json.loads(mesh.read_text(encoding='utf-8'))['frames_file'])
    if (project / 'exmr_keypoints.pt').is_file():
        names.append('exmr_keypoints.pt')
    return {name: file_stat(project / name) for name in names if (project / name).is_file()}


def rest_joints(model_path, shapes):
    """Shape-dependent rest joints (frames, 55, 3) of the SMPL-X body used by the mesh preview."""
    import smplx
    body = smplx.SMPLX(str(model_path), gender='neutral', num_pca_comps=12, flat_hand_mean=False, batch_size=1).eval()
    regressor = body.J_regressor.numpy().astype(np.float64)
    dirs = np.concatenate((body.shapedirs.numpy(), body.expr_dirs.numpy()), axis=-1).astype(np.float64)
    base = regressor @ body.v_template.numpy().astype(np.float64)
    directions = np.einsum('jv,vci->jci', regressor, dirs)
    return base[None] + np.einsum('jci,bi->bjc', directions, np.asarray(shapes, np.float64))


def posed_joints(frames, rest):
    """Mesh-preview frames store relative LBS transforms A_j = G_j * T(-J_j); the posed joint is A_j J_j + transl."""
    frames = np.asarray(frames, np.float64)
    transforms = frames[:, :880].reshape(-1, 55, 4, 4)
    return np.einsum('bjik,bjk->bji', transforms[:, :, :3, :3], rest) + transforms[:, :, :3, 3] + frames[:, None, TRANSL]


def build(project, kind, out):
    from .motion_tools import RESULTS
    from .mamma_body import body_path
    project, out = Path(project).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamps = sources(project, kind)
    record = dict(schema=SCHEMA, project=str(project), kind=kind, sources=stamps)
    tag = uuid.uuid4().hex[:12]
    written = []
    from .self_contact_store import before_path
    before = before_path(project, kind)
    if before.name in stamps:
        mesh = json.loads((project / f'{kind}_mesh.json').read_text(encoding='utf-8'))
        if mesh.get('representation') == 'smplh':
            raise ValueError('The original-layer mesh is available for SMPL-X results only.')
        source = project / RESULTS[kind][0]
        if [source.stat().st_size, source.stat().st_mtime_ns] != [mesh.get('source_size'), mesh.get('source_mtime_ns')]:
            raise ValueError('The corrected mesh preview is older than the corrected motion. Use Prepare Preview.')
        from .mesh_data import prepare_mesh_preview, FRAME_FLOATS
        staging = out / ('build-' + tag)
        staging.mkdir()
        try:
            shutil.copy2(before, staging / RESULTS[kind][0])
            model = body_path(project)
            prepare_mesh_preview(staging, model_path=model, kinds=[kind])
            info = json.loads((staging / f'{kind}_mesh.json').read_text(encoding='utf-8'))
            ghost = np.load(staging / info['frames_file'], mmap_mode='r')
            corrected = np.load(project / mesh['frames_file'], mmap_mode='r')
            if ghost.shape != corrected.shape or ghost.shape[1:] != (FRAME_FLOATS,):
                raise ValueError('Original and corrected motions have different frame counts.')
            with np.load(project / f'{kind}_joints.npz', allow_pickle=False) as data:
                joints = np.asarray(data['joints'], np.float64)
            if len(joints) != len(ghost):
                raise ValueError('Corrected joints and mesh frames differ in length.')
            rest = rest_joints(model, np.asarray(ghost[:, SHAPES]))
            # Offsets from the same LBS keep untouched frames bit-identical to the corrected skeleton.
            delta = posed_joints(ghost, rest)[:, :22] - posed_joints(corrected, rest)[:, :22]
            mesh_name, joints_name = f'before-mesh-{tag}.npy', f'before-joints-{tag}.npy'
            del ghost, corrected  # Windows cannot move a file that is still memory-mapped.
            os.replace(staging / info['frames_file'], out / mesh_name)
            np.save(out / joints_name, (joints + delta).astype(np.float32))
            written += [mesh_name, joints_name]
            record.update(mesh=mesh_name, joints=joints_name, changed_max_m=float(np.linalg.norm(delta, axis=-1).max()))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    keypoints = project / 'exmr_keypoints.pt'
    if keypoints.is_file():
        import torch
        values = np.asarray(torch.load(keypoints, map_location='cpu', weights_only=True), np.float32)
        if values.ndim != 3 or values.shape[1:] != (63, 3):
            raise ValueError(f'exmr_keypoints.pt has shape {values.shape}; expected (frames, 63, 3).')
        name = f'keypoints-{tag}.npy'
        np.save(out / name, values)
        written.append(name)
        record['keypoints'] = name
    if sources(project, kind) != stamps:
        raise ValueError('The motion changed while the comparison was prepared. Try again.')
    temporary = out / f'compare-{tag}.partial'
    temporary.write_text(json.dumps(record), encoding='utf-8')
    os.replace(temporary, out / 'compare.json')
    for old in out.glob('*.npy'):
        if old.name not in written:
            try:
                old.unlink()
            except OSError:
                pass  # Still mapped by the editor; removed on a later build.
    return record


def main():
    from .compute_policy import configure_worker_cpu
    import torch
    torch.set_num_threads(configure_worker_cpu('selfcontact_compare'))
    args = sys.argv
    try:
        value = build(args[args.index('--project') + 1], args[args.index('--kind') + 1], args[args.index('--out') + 1])
        print(json.dumps(value), flush=True)
        return 0
    except Exception as error:
        print('Comparison data could not be prepared: ' + str(error), file=sys.stderr, flush=True)
        return 1
