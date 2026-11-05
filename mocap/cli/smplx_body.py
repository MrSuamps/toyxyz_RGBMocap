"""SMPL-X body exactly as the GUI Mesh view poses it (neutral, flat_hand_mean=False, shape + pose correctives).

Mirrors `mesh_data.prepare_mesh_preview`: full pose = model pose_mean (hand_mean on the finger joints) + result params.
Never use the bound `*-display-mesh.npy` skin for judging motion: it carries an unrelated shin roll (candy-wrap report).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Offsets of each finger's three joints inside a 15-joint SMPL-X hand block (left 25-39, right 40-54).
# SMPL-X order per hand: index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3.
FINGER_OFFSETS = {'thumb': (12, 13, 14), 'index': (0, 1, 2), 'middle': (3, 4, 5), 'ring': (9, 10, 11), 'pinky': (6, 7, 8)}
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
HAND_START = {'left': 25, 'right': 40}
HAND_FIELDS = (('left_hand_pose', 25), ('right_hand_pose', 40))
EYE_FIELDS = (('jaw_pose', 22), ('leye_pose', 23), ('reye_pose', 24))


def rodrigues(rotvec):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    rotvec = np.asarray(rotvec, np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = rotvec / np.maximum(angle, 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(rotvec.shape[:-1] + (3, 3))
    s, c = np.sin(angle)[..., None], np.cos(angle)[..., None]
    return np.eye(3) + s * skew + (1 - c) * skew @ skew


def model_path(project=None):
    from ..paths import MODELS
    if project is not None:
        try:
            from ..mamma_body import body_path
            return Path(body_path(Path(project)))
        except Exception:
            pass
    return Path(MODELS) / 'body_models' / 'smplx' / 'SMPLX_NEUTRAL.npz'


def load_params(path, key='smpl_params_global'):
    """Result params as float64 numpy arrays."""
    import torch
    record = torch.load(Path(path), map_location='cpu', weights_only=False)
    params = record[key] if key in record else record
    return {name: np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value, np.float64) for name, value in params.items()}


class Body:
    """Neutral SMPL-X buffers as numpy; one instance serves every frame and result."""

    def __init__(self, path):
        import smplx
        import torch
        model = smplx.SMPLX(str(path), gender='neutral', num_pca_comps=12, flat_hand_mean=False, batch_size=1).eval()
        self.path = Path(path)
        self.template = model.v_template.numpy().astype(np.float64)
        self.shapedirs = torch.cat((model.shapedirs, model.expr_dirs), -1).numpy().astype(np.float64)
        self.posedirs = model.posedirs.numpy().astype(np.float64)
        self.regressor = model.J_regressor.numpy().astype(np.float64)
        self.weights = model.lbs_weights.numpy().astype(np.float64)
        parents = model.parents.numpy().astype(np.int64).copy()
        parents[0] = -1
        self.parents = parents
        self.faces = np.asarray(model.faces, np.int64)
        self.pose_mean = model.pose_mean.numpy().reshape(55, 3).astype(np.float64)
        self.components = {'left_hand_pose': model.left_hand_components.numpy().astype(np.float64),
                           'right_hand_pose': model.right_hand_components.numpy().astype(np.float64)}

    def full_pose(self, params):
        n = len(params['body_pose'])
        full = np.repeat(self.pose_mean[None], n, 0)
        full[:, 0] += params['global_orient'].reshape(n, 3)
        full[:, 1:22] += params['body_pose'].reshape(n, 21, 3)
        for field, index in EYE_FIELDS:
            if field in params:
                full[:, index] += params[field].reshape(n, 3)
        for field, begin in HAND_FIELDS:
            if field in params:
                hand = params[field].reshape(n, -1)
                if hand.shape[-1] == 12:
                    hand = hand @ self.components[field]
                full[:, begin:begin + 15] += hand.reshape(n, 15, 3)
        return full

    def shapes(self, params):
        n = len(params['body_pose'])
        betas = params['betas'].reshape(n, -1)[:, :10]
        expression = params.get('expression')
        expression = np.zeros((n, 10)) if expression is None else expression.reshape(n, -1)[:, :10]
        return np.concatenate([betas, expression], 1)

    def rest_joints(self, shapes):
        """(N, 55, 3) rest joints of the shaped template; computed once when the shape is constant."""
        if np.allclose(shapes, shapes[:1]):
            shaped = self.template + self.shapedirs @ shapes[0]
            return np.repeat((self.regressor @ shaped)[None], len(shapes), 0)
        shaped = self.template[None] + np.einsum('vci,ni->nvc', self.shapedirs, shapes)
        return np.einsum('jv,nvc->njc', self.regressor, shaped)

    def pose(self, params):
        """Per-frame state: local/world rotations, rest and posed joints (with transl) for all 55 joints."""
        full = self.full_pose(params)
        local = rodrigues(full)
        shapes = self.shapes(params)
        rest = self.rest_joints(shapes)
        n = len(full)
        world = np.empty_like(local)
        joints = np.empty((n, 55, 3))
        for j, p in enumerate(self.parents):
            if p < 0:
                world[:, j] = local[:, j]
                joints[:, j] = rest[:, j]
            else:
                world[:, j] = world[:, p] @ local[:, j]
                joints[:, j] = joints[:, p] + np.einsum('nij,nj->ni', world[:, p], rest[:, j] - rest[:, p])
        transl = params['transl'].reshape(n, 3)
        return dict(local=local, world=world, rest=rest, joints=joints + transl[:, None], shapes=shapes, transl=transl)

    def vertices(self, state, i):
        """(V, 3) posed vertices of frame i, the same maths as `mesh_data.reference_vertices`."""
        shaped = self.template + self.shapedirs @ state['shapes'][i]
        features = (state['local'][i, 1:] - np.eye(3)).reshape(-1)
        posed = shaped + (features @ self.posedirs).reshape(-1, 3)
        world = state['world'][i]
        offset = state['joints'][i] - state['transl'][i] - np.einsum('jab,jb->ja', world, state['rest'][i])
        rotation = np.einsum('vj,jab->vab', self.weights, world)
        return np.einsum('vab,vb->va', rotation, posed) + self.weights @ offset + state['transl'][i]


def gui_vertices(project, kind, frame):
    """Vertices from the project's own `<kind>_mesh` preview frames (the GUI Mesh view input), or None when absent."""
    import json
    from ..mesh_data import reference_vertices
    project = Path(project)
    meta = project / f'{kind}_mesh.json'
    if not meta.is_file():
        return None
    info = json.loads(meta.read_text(encoding='utf-8'))
    frames_file = project / info['frames_file']
    asset = Path(info['asset'])
    if not frames_file.is_file() or not asset.is_file():
        return None
    with np.load(asset, allow_pickle=False) as data:
        packed, count = data['packed'], int(data['vertex_count'])
    frames = np.load(frames_file, mmap_mode='r')
    return reference_vertices(packed.astype(np.float64), np.asarray(frames[frame], np.float64), count)
