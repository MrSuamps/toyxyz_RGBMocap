"""Read an original paired SMPL-X camera in an isolated CPU process; no writes."""
from pathlib import Path
import json,zipfile
import numpy as np


def root_change(project,kind,count):
    """(T,3) pelvis change Motion Correction made to `kind` over its source result, or None."""
    from .motion_tools import base_result
    roots=[]
    for name in (kind,base_result(kind)):
        with zipfile.ZipFile(project/(name+'_joints.npz')) as archive:
            if archive.getinfo('joints.npy').file_size>32*1024**2:raise ValueError('Corrected joints exceed the preview memory limit.')
        with np.load(project/(name+'_joints.npz'),allow_pickle=False) as data:roots.append(np.asarray(data['joints'][:,0],float))
    if not len(roots[0])==len(roots[1])==count:return None
    delta=roots[0]-roots[1]
    return delta if np.isfinite(delta).all() else None


def source_pairing_inputs(motion):
    """Per-frame (world rotation, in-camera rotation, world translation, in-camera translation) of an ORIGINAL source result."""
    from scipy.spatial.transform import Rotation
    world,incam=motion['smpl_params_global'],motion['smpl_params_incam']
    rotation=lambda params:Rotation.from_rotvec(np.asarray(params['global_orient'],float)).as_matrix()
    return rotation(world),rotation(incam),np.asarray(world['transl'],float),np.asarray(incam['transl'],float)


def result_root_change(project,kind):
    """(T,3) pelvis change correction made to result `kind` over its source result; None for an uncorrected or mismatched one."""
    from .motion_tools import base_result
    if not kind or not kind.startswith('corrected_'):return None
    try:
        roots=[]
        for name in (kind,base_result(kind)):
            with np.load(Path(project)/(name+'_joints.npz'),allow_pickle=False) as data:roots.append(np.asarray(data['joints'][:,0],float))
    except (OSError,KeyError,ValueError):return None
    if len(roots[0])!=len(roots[1]):return None
    delta=roots[0]-roots[1]
    return delta if np.isfinite(delta).all() else None


def static_source_camera(project,shift=None):
    """The one fixed world-to-camera (4,4) of a static take in its own frame, from its original motion: see
    scene_camera.static_paired_camera (`shift` as there). Uses the committed original pelvis, as reference()."""
    import torch
    from .scene_camera import static_paired_camera
    project=Path(project)
    motion=torch.load(project/'hmr4d_results.pt',map_location='cpu',weights_only=True)
    with np.load(project/'initial_joints.npz',allow_pickle=False) as data:root=np.asarray(data['joints'][0,0],float)
    inputs=source_pairing_inputs(motion)
    if shift is not None and np.shape(shift)!=np.shape(inputs[2]):shift=None
    return static_paired_camera(*inputs,root-np.asarray(motion['smpl_params_global']['transl'][0],float),shift)


def exmr_view(project,kind=None,shift=None):
    """'moving', the static (R, t) world-to-camera every consumer projects a static ExMR take with (GUI camera view, review,
    render, placement, Blender export), or None without a saved ExMR camera policy. The camera of corrected result `kind`
    follows its root change (or `shift`, (F,3) in the take frame): see scene_camera.static_centre."""
    from .scene_camera import exmr_static
    project=Path(project)
    if shift is None:shift=result_root_change(project,kind)
    static=exmr_static(project)
    if static is None:return None
    if not static:return 'moving'
    try:
        sources=json.loads((project/'exmr_camera_sources.json').read_text(encoding='utf-8'))
        r=np.asarray(np.load((project/sources['root']).resolve()/'exmr_camera.npy',allow_pickle=False)[int(sources['first'])],float)
        reference=np.asarray(sources['reference_rotation'],float)
        align=reference@np.asarray(sources['first_rotation'],float).T
    except (OSError,ValueError,KeyError,IndexError,TypeError):return None
    # As camera_tracks.load: the pairing turned into the scene camera frame, at a fixed centre, with the (constant) ExMR rotation.
    base=align@static_source_camera(project,shift)[:3]
    centre=-base[:,:3].T@base[:,3]
    rotation=r@reference.T@base[:,:3]
    return rotation,-rotation@centre


def reference(project,kind=None):
    """`kind`: the camera follows the root change Motion Correction made to this displayed/exported result (moving: per-frame
    centres; static: the one centre, scene_camera.static_centre)."""
    from .result_artifacts import artifacts_current,file_stat
    from .scene_camera import paired_camera,static_paired_camera,validate_transform
    project=Path(project).resolve()
    state=json.loads((project/'project.json').read_text(encoding='utf-8'))
    model=state.get('pipeline',{}).get('model')
    if model not in ('exmr','gvhmr'):raise ValueError('Paired source camera is unsupported for this model.')
    if not artifacts_current(project,'initial'):raise ValueError('Original motion preview is stale. Use Prepare Preview.')
    names=['hmr4d_results.pt','initial_joints.npz','initial_mesh.json','tracking.npz']
    if (project/'initial_artifacts.json').exists():names.append('initial_artifacts.json')
    follow=model=='exmr' and bool(kind) and kind.startswith('corrected_') and artifacts_current(project,kind)
    if follow:
        from .motion_tools import base_result
        for name in (kind+'_joints.npz',kind+'_artifacts.json',base_result(kind)+'_joints.npz'):
            if name not in names and (project/name).exists():names.append(name)
    saved=cam_root=None
    if model=='exmr':
        names.append('exmr_camera_sources.json')
        saved=json.loads((project/'exmr_camera_sources.json').read_text(encoding='utf-8'))
        cam_root=(project/saved['root']).resolve()
        if not project.is_relative_to(cam_root):raise ValueError('Invalid ExMR camera root.')
        import os
        names.extend(os.path.relpath(cam_root/name,project) for name in ('exmr_camera.json','exmr_camera.npy'))
    sources={name:file_stat(project/name) for name in names}
    import psutil
    if psutil.virtual_memory().available<512*1024**2:raise MemoryError('Free at least 512 MiB RAM to read the saved camera.')
    import torch
    torch.set_num_threads(1)
    from scipy.spatial.transform import Rotation
    motion=torch.load(project/'hmr4d_results.pt',map_location='cpu',weights_only=True,mmap=True)
    world,incam=motion['smpl_params_global'],motion['smpl_params_incam']
    # Read only one joint directly from the archive; do not extract a preview cache.
    with zipfile.ZipFile(project/'initial_joints.npz') as archive:
        with archive.open('joints.npy') as stream:
            version=np.lib.format.read_magic(stream)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(stream)
            if fortran or dtype.kind!='f' or len(shape)!=3 or shape[0]!=len(world['transl']) or shape[1:]!=(22,3):
                raise ValueError('Original source joints do not match the saved motion.')
            root=np.frombuffer(stream.read(3*dtype.itemsize),dtype=dtype).astype(float)
    pelvis=root-world['transl'][0].numpy()
    if pelvis.shape!=(3,) or not np.isfinite(pelvis).all():raise ValueError('Invalid original pelvis pivot.')
    policy=json.loads((cam_root/'exmr_camera.json').read_text(encoding='utf-8')) if model=='exmr' else None
    centers=origin=follows=paired=None
    if policy is not None and policy['identity']['static']:
        if len(world['transl'])*9*8>32*1024**2:raise ValueError('Camera pairing exceeds the preview memory limit.')
        shift=root_change(project,kind,len(world['transl'])) if follow else None
        matrix=static_paired_camera(*source_pairing_inputs(motion),pelvis,shift)
        if shift is not None:follows=kind
    else:
        rotation=lambda params:Rotation.from_rotvec(params['global_orient'][0].numpy()).as_matrix()
        matrix=paired_camera(rotation(world),rotation(incam),world['transl'][0].numpy(),incam['transl'][0].numpy(),pelvis)
    if model=='exmr':
        align=np.asarray(saved['reference_rotation'])@np.asarray(saved['first_rotation']).T
        matrix[:3,:3]=align@matrix[:3,:3];matrix[:3,3]=align@matrix[:3,3]
        if not policy['identity']['static']:
            rotations=np.load(cam_root/'exmr_camera.npy',mmap_mode='r',allow_pickle=False)
            try:
                count=len(world['transl']);first=saved['first']
                if count*3*8>32*1024**2:raise ValueError('Camera path exceeds the preview memory limit.')
                r=np.asarray(rotations[first:first+count],float)@np.asarray(saved['reference_rotation']).T@matrix[:3,:3]
                if len(r)!=count:raise ValueError('ExMR camera does not cover the original take.')
                # Relative camera-center estimates from ORIGINAL paired roots.
                # Body/inference errors remain visible; no pose refitting occurs.
                centers=world['transl'].numpy()+pelvis-np.einsum('tji,tj->ti',r,incam['transl'].numpy()+pelvis)
                if not np.isfinite(centers).all():raise ValueError('Invalid estimated camera path.')
                origin=centers[0].copy()
                # The overlay camera is paired per frame (review, render, Blender export); the VO rotation differs by up to ~12 deg.
                world_rotation,cam_rotation,world_translation,cam_translation=source_pairing_inputs(motion)
                paired=cam_rotation@world_rotation.swapaxes(-1,-2)
                paired_centers=world_translation+pelvis-np.einsum('tji,tj->ti',paired,cam_translation+pelvis)
                # SimpleVO observes rotation only, so the camera may follow the corrected root: the body keeps its place in view.
                delta=root_change(project,kind,count) if follow else None
                if delta is not None:centers=centers+delta;paired_centers=paired_centers+delta;follows=kind
                if not np.isfinite(paired).all() or not np.isfinite(paired_centers).all():raise ValueError('Invalid paired camera path.')
            finally:rotations._mmap.close()
    with np.load(project/'tracking.npz',allow_pickle=False) as data:k=data['K'].tolist()
    if not artifacts_current(project,'initial') or any(file_stat(project/name)!=stamp for name,stamp in sources.items()):
        raise ValueError('Original camera inputs changed while loading.')
    return dict(schema=3,world_to_camera=validate_transform(matrix).tolist(),K=k,sources=sources,
        centers=None if centers is None else centers.tolist(),origin=None if origin is None else origin.tolist(),follows=follows,
        paired_rotations=None if paired is None else paired.tolist(),paired_centers=None if paired is None else paired_centers.tolist(),
        start=state['start'],fps=state['fps'])


def main():
    import sys
    from .compute_policy import configure_worker_cpu
    configure_worker_cpu('camera_reference')
    try:
        kind=sys.argv[sys.argv.index('--kind')+1] if '--kind' in sys.argv else None
        print(json.dumps(reference(Path(sys.argv[sys.argv.index('--project')+1]),kind)),flush=True)
        return 0
    except Exception as error:
        print('Saved original camera could not be read: '+str(error),file=sys.stderr,flush=True)
        return 1
