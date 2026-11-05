"""SMPL-X body exactly as the GUI Mesh view poses it (neutral, flat_hand_mean=False, shape + pose correctives).

Mirrors `mesh_data.prepare_mesh_preview`: full pose = model pose_mean (hand_mean on the finger joints) + result params.
Never use the bound `*-display-mesh.npy` skin for judging motion: it carries an unrelated shin roll (candy-wrap report).
FlowHMR results are SMPL-H: `SmplhBody` poses them as FlowHMR's own SMPLMesh does (the GUI's native SMPL-H surface).
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
# SMPL-H joint j is SMPL-X joint SMPLH_TO_SMPLX[j] (same body prefix and finger order); SMPL-X joint i takes SMPL-H joint
# SMPLX_FROM_SMPLH[i], so the jaw and eyes (22-24, absent in SMPL-H) sit on the head.
SMPLH_TO_SMPLX = np.r_[np.arange(22), np.arange(25, 55)]
SMPLX_FROM_SMPLH = np.r_[np.arange(22), [15, 15, 15], np.arange(22, 52)]
# Fields of a FlowHMR result (one flat record; the rest is metadata such as `source_representation`).
SMPLH_FIELDS = ('rot6d', 'shapes', 'trans')


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


def rot6d_matrix(rot6d):
    """(..., 6) FlowHMR 6D rotation -> (..., 3, 3): both columns Gram-Schmidt orthonormalised, as flowhmr rot6d_to_rotation_matrix."""
    x = np.asarray(rot6d, np.float64)
    x = x.reshape(x.shape[:-1] + (3, 2))
    a, b = x[..., 0], x[..., 1]
    b1 = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b2 = b - np.sum(b1 * b, -1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def model_path(project=None):
    from ..paths import MODELS
    if project is not None:
        try:
            from ..mamma_body import body_path
            return Path(body_path(Path(project)))
        except Exception:
            pass
    return Path(MODELS) / 'body_models' / 'smplx' / 'SMPLX_NEUTRAL.npz'


def smplh_path():
    """The AMASS SMPL-H neutral body FlowHMR is registered with (Model Manager row `smplh`)."""
    from ..models import BY_KEY
    return Path(BY_KEY['smplh'].path)


def body_for(project, model):
    """The body a model's results are posed on: SMPL-H for FlowHMR, else SMPL-X (MAMMA's own body when the project has one)."""
    if model == 'flowhmr':
        path = smplh_path()
        if not path.is_file():
            raise FileNotFoundError(f'FlowHMR results need the SMPL-H neutral body ({path}); import it in FlowHMR Model Manager.')
        return SmplhBody(path)
    return Body(model_path(project))


def _array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value, np.float64)


def load_params(path, key='smpl_params_global'):
    """Result params as float64 numpy arrays; a FlowHMR record (no `key`, SMPL-H rot6d) gives its SMPLH_FIELDS."""
    import torch
    record = torch.load(Path(path), map_location='cpu', weights_only=False)
    if key not in record and 'rot6d' in record:
        return {name: _array(record[name]) for name in SMPLH_FIELDS if name in record}
    params = record[key] if key in record else record
    return {name: _array(value) for name, value in params.items()}


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


class SmplhBody:
    """Neutral AMASS SMPL-H as FlowHMR's SMPLMesh poses it (`flow_export` writes the GUI surface with it): shape blend shapes, no pose
    correctives. Weights and pose state use Body's 55-joint SMPL-X layout (SMPLH_TO_SMPLX), so joint indices mean the same."""
    representation = 'smplh'

    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as data:
            self.template = data['v_template'].astype(np.float64)
            self.shapedirs = data['shapedirs'].astype(np.float64)
            self.regressor = data['J_regressor'].astype(np.float64)
            weights = data['weights'].astype(np.float64)
            parents = data['kintree_table'][0].astype(np.int64)
            self.faces = data['f'].astype(np.int64)
        self.path = Path(path)
        if self.regressor.shape[0] != 52 or weights.shape[1] != 52:
            raise ValueError(f'{self.path.name} is not a 52-joint SMPL-H model.')
        parents[0] = -1
        self.weights = np.zeros((len(weights), 55))
        self.weights[:, SMPLH_TO_SMPLX] = weights
        self.parents = np.full(55, 15, np.int64)
        self.parents[SMPLH_TO_SMPLX] = np.where(parents < 0, -1, SMPLH_TO_SMPLX[np.maximum(parents, 0)])

    def shapes(self, params, n):
        betas = np.asarray(params['shapes'], np.float64).reshape(-1)
        count = min(len(betas), self.shapedirs.shape[-1])
        return np.repeat(betas[None, :count], n, 0)

    def pose(self, params):
        """Per-frame state like Body.pose (55 SMPL-X-layout joints, with transl) from FlowHMR rot6d / shapes / trans."""
        rotations = rot6d_matrix(params['rot6d']).reshape(-1, 52, 3, 3)
        n = len(rotations)
        shapes = self.shapes(params, n)
        shaped = self.template + self.shapedirs[..., :shapes.shape[1]] @ shapes[0]
        rest = np.repeat((self.regressor @ shaped)[SMPLX_FROM_SMPLH][None], n, 0)
        local = np.repeat(np.eye(3)[None, None], n, 0).repeat(55, 1)
        local[:, SMPLH_TO_SMPLX] = rotations
        world = np.empty_like(local)
        joints = np.empty((n, 55, 3))
        for j, p in enumerate(self.parents):
            if p < 0:
                world[:, j] = local[:, j]
                joints[:, j] = rest[:, j]
            else:
                world[:, j] = world[:, p] @ local[:, j]
                joints[:, j] = joints[:, p] + np.einsum('nij,nj->ni', world[:, p], rest[:, j] - rest[:, p])
        transl = np.asarray(params['trans'], np.float64).reshape(n, 3)
        return dict(local=local, world=world, rest=rest, joints=joints + transl[:, None], shapes=shapes, transl=transl)

    def vertices(self, state, i):
        """(V, 3) posed vertices of frame i: linear blend skinning of the shaped template, as SMPLMesh(use_pose_blending=False)."""
        shaped = self.template + self.shapedirs[..., :state['shapes'].shape[1]] @ state['shapes'][i]
        world = state['world'][i]
        offset = state['joints'][i] - state['transl'][i] - np.einsum('jab,jb->ja', world, state['rest'][i])
        rotation = np.einsum('vj,jab->vab', self.weights, world)
        return np.einsum('vab,vb->va', rotation, shaped) + self.weights @ offset + state['transl'][i]


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
    if info.get('representation') == 'smplh':
        # FlowHMR's preview frames are the native SMPL-H surface itself, (frames, 6890, 3).
        return np.asarray(np.load(frames_file, mmap_mode='r')[frame], np.float64) if frames_file.is_file() else None
    asset = Path(info['asset'])
    if not frames_file.is_file() or not asset.is_file():
        return None
    with np.load(asset, allow_pickle=False) as data:
        packed, count = data['packed'], int(data['vertex_count'])
    frames = np.load(frames_file, mmap_mode='r')
    return reference_vertices(packed.astype(np.float64), np.asarray(frames[frame], np.float64), count)
