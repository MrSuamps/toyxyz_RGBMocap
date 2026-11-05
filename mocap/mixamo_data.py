"""Transactional target motion preparation, outside the editor process."""
from __future__ import annotations
import json,os,uuid
from pathlib import Path
import numpy as np
from .mixamo_rig import VERSION,skeleton,fk,mannequin,write_bvh,BODY_PARENTS,unit,align,TRUNK_POSE_JOINTS,HINGES
from .result_artifacts import file_stat
from .storage import atomic_json


def code_identity():
    from .paths import ROOT
    from .storage import sha256
    return {name:sha256(ROOT/'mocap'/name) for name in ('mixamo_rig.py','mixamo_data.py','bvh_bind.py','source_pose_fit.py','rig_mesh.py','correction.py','correction_settings.py','anatomical_ik.py','hand_pose.py','source_animation.py','source_bvh.py','surface_roll.py','surface_landmarks.py','source_surface.py','fbx_bind.py')}


def current(project,kind):
    try:
        project=Path(project)
        if (project/(kind+'_mixamo_pending.json')).exists():return False
        record=json.loads((project/(kind+'_mixamo.json')).read_text(encoding='utf-8'))
        if record['version']!=VERSION or record['kind']!=kind:return False
        # Small source identities are checked by the loader, never by painting.
        # Current export/GUI calls only stat their immutable code files.
        from .cache_compat import code_stats_current
        if not code_stats_current(record):return False
        from .motion_tools import result_is_current
        if not result_is_current(project,kind,ignore_contacts=True):return False
        state=json.loads((project/'project.json').read_text(encoding='utf-8')) if (project/'project.json').exists() else {}
        from .motion_tools import require_supported_project
        require_supported_project(project)
        if float(state.get('fps',30))!=record['fps']:return False
        from .models import BY_KEY
        if state.get('pipeline',{}).get('model')=='mamma':
            from .mamma_body import body_path
            model=body_path(project)
        else:model=BY_KEY[record.get('model_key','smplx')].path
        if model.is_file() and record.get('model_stat') is not None and file_stat(model)!=record['model_stat']:return False
        for name,stamp in record['sources'].items():
            from .project_storage import resolve_reference
            path=resolve_reference(project,name)
            if name==record.get('model_path') and not path.exists():continue
            if file_stat(path)!=stamp:return False
        return all(Path(name).name==name and file_stat(project/name)==stamp for name,stamp in record['outputs'].items())
    except (OSError,ValueError,KeyError,TypeError):return False


def open_preview(project,kind,cancelled=lambda:False):
    """Called only by the coalescing loader. Never imports Torch or SciPy."""
    from .preview_data import FrameArray,LoadCancelled
    if not current(project,kind):return None
    info=json.loads((project/(kind+'_mixamo.json')).read_text(encoding='utf-8'))
    arrays={}
    try:
        for key,name in (('rig',info['frames_file']),('rig_joints',info['joints_file'])):
            arrays[key]=FrameArray(project/name,max_blocks=1)
        if info.get('reference_joints_file'):
            arrays['raw_joints']=FrameArray(project/info['reference_joints_file'],max_blocks=1)
            if arrays['raw_joints'].shape!=(info['frames'],22,3):raise ValueError('Invalid source joint reference')
        if info.get('display_mesh_file'):
            from .mesh_data import FRAME_FLOATS
            arrays['display_mesh']=FrameArray(project/info['display_mesh_file'],max_blocks=1)
            if arrays['display_mesh'].shape!=(info['frames'],FRAME_FLOATS) or arrays['display_mesh'].dtype!=np.float32:raise ValueError('Invalid export mesh preview')
        if arrays['rig'].shape!=(info['frames'],52,4,4) or arrays['rig_joints'].shape!=(info['frames'],22,3):raise ValueError('Invalid Mixamo preview dimensions')
        with np.load(project/info['asset_file'],allow_pickle=False) as data:asset={k:data[k] for k in ('packed','faces','vertex_count')}
        asset['mode']='rig';asset['key']=(str(project/info['asset_file']),*file_stat(project/info['asset_file']))
        if info.get('display_binding_file'):
            path=project/info['display_binding_file']
            if path.stat().st_size>1024**2:raise ValueError('Canonical binding exceeds size limit')
            with np.load(path,allow_pickle=False) as data:
                binding={k:data[k] for k in ('vertices','frames','local','rest','parents','betas')}
            if binding['vertices'].shape!=(10475,3) or binding['frames'].shape!=(55,3,3) or binding['rest'].shape!=(55,3):raise ValueError('Invalid canonical binding dimensions')
            if not all(np.isfinite(v).all() for v in binding.values()):raise ValueError('Invalid canonical binding values')
            binding['key']=(str(path),*file_stat(path));asset['binding']=binding
        if info.get('raw_file'):
            arrays['native']=FrameArray(project/info['raw_file'],max_blocks=1)
            if arrays['native'].shape!=(info['frames'],info.get('raw_vertex_count',595),3):raise ValueError('Invalid native preview dimensions')
            with np.load(project/info['raw_asset'],allow_pickle=False) as data:raw={k:data[k] for k in ('faces','offsets','neighbors','vertex_count')}
            raw['mode']='raw';raw['key']=(str(project/info['raw_asset']),*file_stat(project/info['raw_asset']))
            if info.get('representation')=='smplh':raw['topology_key']=('smplh-topology-v1',*info['model_stat'])
            asset['raw_asset']=raw
        if cancelled():raise LoadCancelled()
        if not current(project,kind):raise ValueError('Target motion changed while loading')
        return arrays,asset,info
    except Exception:
        for value in arrays.values():value.close()
        raise


def prepare(project, result, emit=lambda *a, **k: None, body=None):
    """Fit the selected SMPL-X pose after optional source correction."""
    from .motion_tools import require_supported_project
    require_supported_project(project)
    import psutil
    if psutil.virtual_memory().available < 1536 * 1024 ** 2:
        raise MemoryError('Free at least 1.5 GiB RAM to prepare the target preview safely.')
    import torch, smplx
    from scipy.spatial.transform import Rotation
    from .source_animation import fit,temporal_halo,FIT_VERSION,source_rig,validate,policy
    from .source_bvh import write_bvh
    from .motion_tools import RESULTS, result_sources, result_is_current, base_result
    from .correction import contact_strength
    from .models import BY_KEY
    project = Path(project)
    from .mamma_body import body_path
    model = body_path(project)
    model_stamp = file_stat(model)
    if body is None:
        body = smplx.SMPLX(str(model), gender='neutral', num_pca_comps=12, flat_hand_mean=False, batch_size=1).eval()
    state = json.loads((project / 'project.json').read_text(encoding='utf-8')) if (project / 'project.json').exists() else {}
    fps = float(state.get('fps', 30))
    for kind in result_sources(result):
        if current(project, kind):
            continue
        if not result_is_current(project, kind):
            raise ValueError('Target source unavailable or correction pending. Apply Correction first.')
        base = base_result(kind)
        (filename, key) = RESULTS[kind]
        sources = {filename: file_stat(project / filename)}
        sources[str(model)] = model_stamp
        params = torch.load(project / filename, map_location='cpu', weights_only=False, mmap=True)[key]
        n = len(params['body_pose'])
        if n * 10000 > max(64 * 1024 ** 2, psutil.virtual_memory().available * 0.35):
            raise MemoryError('Free more RAM or trim the clip to prepare the target skeleton safely.')
        from .source_surface import smplx_endpoints
        emit('progress', stage='rig', current=0, total=0, message='Read selected surface directions')
        surface_endpoints, surface_sources = smplx_endpoints(project, kind, n)
        sources.update(surface_sources)
        with torch.no_grad():
            template = body(betas=params['betas'][:1])
        body_data = getattr(body, 'bm', body)
        anatomy = dict(vertices=template.vertices[0].numpy(), joints=template.joints[0].numpy(), skin=body_data.lbs_weights.numpy())
        rest_template = (body_data.J_regressor[:22] @ body_data.v_template).detach().numpy()
        rest_shapes = torch.einsum('jv,vcd->jcd', body_data.J_regressor[:22], body_data.shapedirs).detach().numpy()
        rest = template.joints[0, :22].numpy().copy()
        rig = source_rig(skeleton(rest, anatomy['joints'], source_anchors=True), rest)
        asset = mannequin(rig, anatomy=anatomy)
        from .fbx_bind import prepare as prepare_fbx_bind
        fbx_binding=prepare_fbx_bind(rig,rest,anatomy['vertices'],'smplx',fps)
        token = uuid.uuid4().hex[:12]
        stem = f'{kind}_rig-{token}'
        files = {'frames_file': stem + '.npy', 'joints_file': stem + '-joints.npy', 'reference_joints_file': stem + '-reference.npy', 'asset_file': stem + '-asset.npz', 'bvh_file': stem + '.bvh'}
        pending = project / (kind + '_mixamo_pending.json')
        atomic_json(pending, dict(generation=token))
        arrays = []
        created = []
        finished = False
        try:

            def array(suffix, shape):
                path = project / (stem + suffix + '.npy')
                created.append(path)
                value = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
                arrays.append(value)
                return value
            roots = array('-roots', (n, 3))
            quats = array('-quats', (n, 22, 4))
            frames = array('', (n, 52, 4, 4))
            points_out = array('-joints', (n, 22, 3))
            reference_out = array('-reference', (n, 22, 3))
            fit_errors = array('-fit-errors', (n, 22))
            distal_reference = array('-distal-reference', (n, 4, 3, 3))
            distal_errors = array('-distal-errors', (n, 4))
            facing_errors = array('-anatomical-facing-errors', (n, 4))
            display_output = None
            display_error = 0.0
            arm_pole_error = 0.0
            source_axis_error = 0.0
            hinge_error = 0.0
            orthogonality_error = 0.0
            upper_joints = (*TRUNK_POSE_JOINTS, 16, 17, 18, 19, 20, 21)
            from .mesh_data import FRAME_FLOATS
            from .rig_mesh import make_binding
            display_binding = make_binding(body, rig, params['betas'][:1])
            files['display_binding_file'] = stem + '-display-binding.npz'
            np.savez(project / files['display_binding_file'], **display_binding)
            created.append(project / files['display_binding_file'])
            files['display_mesh_file'] = stem + '-display-mesh.npy'
            display_output = array('-display-mesh', (n, FRAME_FLOATS))
            distals = [7, 8, 20, 21]
            for first in range(0, n, 128):
                end = min(n, first + 128)
                halo = temporal_halo(fps)
                begin = max(0, first - halo)
                stop = min(n, end + halo)
                # Temporal roll support overlaps each bounded batch. All goals
                # come from the selected source, including committed correction.
                local_source = Rotation.from_rotvec(torch.cat((params['global_orient'][begin:stop, None], params['body_pose'][begin:stop].reshape(-1, 21, 3)), 1).numpy().reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
                g = local_source.copy()
                for (j, p) in enumerate(rig['parents'][:22]):
                    if p >= 0:
                        g[:, j] = g[:, p] @ g[:, j]
                betas = params['betas'][begin:stop].numpy() if len(params['betas']) > 1 else np.repeat(params['betas'].numpy(), stop - begin, axis=0)
                shaped_rest = rest_template + np.einsum('jcd,td->tjc', rest_shapes[..., :betas.shape[1]], betas)
                root = params['transl'][begin:stop].numpy() + shaped_rest[:, 0]
                goals = np.empty((stop - begin, 22, 3))
                goals[:, 0] = root
                for (j, p) in enumerate(BODY_PARENTS):
                    if p >= 0:
                        goals[:, j] = goals[:, p] + np.einsum('tij,tj->ti', g[:, p], shaped_rest[:, j] - shaped_rest[:, p])
                (local, positions, world) = fit(rig, root, g, goals, surface_endpoints=surface_endpoints[begin:stop],fps=fps)
                core = slice(first - begin, end - begin)
                (local, positions, world) = (local[core], positions[core], world[core])
                (root, g, goals) = (root[core], g[core], goals[core])
                reference_out[first:end] = goals
                distal_reference[first:end] = g[:, distals]
                for index,(upper, middle, distal, axis) in enumerate(HINGES):
                    axis = np.asarray(axis, dtype=float)
                    anatomical = world[:,upper] @ axis
                    source_anatomical = g[:,upper] @ axis
                    facing_errors[first:end,index] = np.rad2deg(np.arccos(np.clip(np.sum(anatomical*source_anatomical,axis=-1),-1,1)))
                    calibrated = g[:, upper] @ rig['alignment'][upper]
                    old_direction = unit(rig['rest'][middle] - rig['rest'][upper])
                    direction = world[:, upper] @ old_direction
                    expected = np.stack([align(frame @ old_direction, goal) @ (frame @ axis) for (frame, goal) in zip(calibrated, direction)])
                    actual = world[:, upper] @ axis
                    error = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(actual, expected), axis=-1), np.sum(actual * expected, axis=-1)))
                    source_axis_error = max(source_axis_error, float(error.max()))
                    if upper >= 16:
                        arm_pole_error = max(arm_pole_error, float(error.max()))
                fit_errors[first:end] = np.linalg.norm(positions[:, :22] - goals, axis=-1)
                roots[first:end] = root
                quats[first:end] = Rotation.from_matrix(local[:, :22].reshape(-1, 3, 3)).as_quat().reshape(-1, 22, 4)[..., [3, 0, 1, 2]]
            report = {}
            if kind.startswith('corrected_'):
                report_source = project / (kind + '_correction.json')
                contact_file = project / (kind + '_contacts.npz')
                saved = json.loads(report_source.read_text(encoding='utf-8'))
                sources[report_source.name] = file_stat(report_source)
                sources[contact_file.name] = file_stat(contact_file)
                report = dict(source_correction_applied_once=True, second_target_correction=False, floor=saved['floor'], preset=saved.get('preset', 'normal'), source_warning=saved.get('warning', ''), policy='Fit selected corrected source directly; never correct target a second time')
            local_all = np.empty((n, 52, 3, 3), np.float32)
            for first in range(0, n, 128):
                end = min(n, first + 128)
                local = np.tile(np.eye(3), (end - first, 52, 1, 1))
                local[:, :22] = Rotation.from_quat(quats[first:end, ..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(-1, 22, 3, 3)
                from .hand_pose import apply_fingers
                apply_fingers(local,params,body,display_binding,first,end)
                (positions, world) = fk(rig, roots[first:end], local)
                fit_errors[first:end] = np.linalg.norm(positions[:, :22] - reference_out[first:end], axis=-1)
                orthogonality_error = max(orthogonality_error, float(np.abs(world.transpose(0, 1, 3, 2) @ world - np.eye(3)).max()))
                if orthogonality_error > 0.0001 or np.linalg.det(world).min() < 0.9999:
                    raise ValueError('Invalid target rotation frame. Preview not committed.')
                hinge_error = max(hinge_error, validate(local))
                relative = world[:, distals] @ distal_reference[first:end].transpose(0, 1, 3, 2)
                distal_errors[first:end] = np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(-1, 4))
                local_all[first:end] = local
                if display_output is not None:
                    from .rig_mesh import mesh_frames
                    beta = params['betas'][first:end] if len(params['betas']) > 1 else params['betas']
                    (display_output[first:end], error) = mesh_frames(body, rig, positions, world, beta, display_binding)
                    display_error = max(display_error, error)
                frames[first:end] = np.eye(4)
                frames[first:end, :, :3, :3] = world
                frames[first:end, :, :3, 3] = positions
                points_out[first:end] = positions[:, :22]
                emit('progress', stage='rig', current=end, total=n, message='Prepare export skeleton preview')
            export_root = np.asarray(roots)
            bvh_encoding = write_bvh(project / files['bvh_file'], rig, export_root, local_all, fps)
            created.append(project / files['bvh_file'])
            from .bvh_bind import bind_pose, asset_frames
            export_binding = bind_pose(rig)
            export_probes = np.einsum('vij,vj->vi', export_binding['basis'][asset['probe_bones']].swapaxes(-1, -2), asset['probe_local'])
            np.savez(project / files['asset_file'], packed=asset_frames(asset['packed'], export_binding), faces=asset['faces'], vertex_count=np.array(asset['vertex_count']), rest=export_binding['rest'], parents=rig['parents'], ends=export_binding['ends'], export_basis=export_binding['basis'], anatomical_calibration=rig['alignment'], probe_local=export_probes, probe_bones=asset['probe_bones'], sole_ids=asset['sole_ids'], sole_groups=asset['sole_groups'], source_regressor=np.empty((0, 0), np.float32), **fbx_binding)
            created.append(project / files['asset_file'])
            fit_summary = dict(mean_m=float(fit_errors.mean()), p95_m=float(np.quantile(fit_errors, 0.95)), max_m=float(fit_errors.max()), joints={name: dict(mean_m=float(fit_errors[:, j].mean()), p95_m=float(np.quantile(fit_errors[:, j], 0.95)), max_m=float(fit_errors[:, j].max())) for (j, name) in enumerate(rig['names'][:22])}, distal_orientation_max_deg={rig['names'][j]: float(distal_errors[:, i].max()) for (i, j) in enumerate(distals)}, comparison='Actually selected source pose vs anatomical fixed-length target')
            fit_summary['anatomical_facing_deg'] = {rig['names'][middle]:dict(mean=float(facing_errors[:,index].mean()),p95=float(np.quantile(facing_errors[:,index],.95)),max=float(facing_errors[:,index].max())) for index,(_,middle,_,_) in enumerate(HINGES)}
            archive = project / f'{kind}_joints.npz'
            # Compare against the actually displayed source, not rebound skin.
            if archive.exists():
                sources[archive.name] = file_stat(archive)
                with np.load(archive, allow_pickle=False) as source_joints:
                    selected_reference = source_joints['joints']
            else:
                (selected_file, selected_key) = RESULTS[kind]
                selected_params = torch.load(project / selected_file, map_location='cpu', weights_only=False, mmap=True)[selected_key]
                selected_reference = np.empty((n, 22, 3), np.float32)
                with torch.no_grad():
                    for start in range(0, n, 32):
                        batch = {k: v[start:start + 32] if len(v) > 1 else v for (k, v) in selected_params.items()}
                        from .hand_pose import evaluate
                        selected_reference[start:start + len(batch['body_pose'])] = evaluate(body,batch).joints[:, :22].numpy()
            if selected_reference.shape != (n, 22, 3):
                raise ValueError('Selected source pose dimensions changed during target fitting')
            selected_errors = np.linalg.norm(points_out[:] - selected_reference, axis=-1)
            selected_summary = dict(mean_m=float(selected_errors.mean()), p95_m=float(np.quantile(selected_errors, 0.95)), max_m=float(selected_errors.max()), joints={name: dict(mean_m=float(selected_errors[:, j].mean()), max_m=float(selected_errors[:, j].max())) for (j, name) in enumerate(rig['names'][:22])}, comparison='Selected original source pose versus exported skeleton; no rebound display surface')
            selected_summary['anatomical_facing_deg']=fit_summary['anatomical_facing_deg']
            if kind.startswith('corrected_'):
                # Source correction already solved the contacts. Independently
                # report target/source ankle residuals without a second solve.
                with np.load(contact_file, allow_pickle=False) as contacts:
                    source_segments = [dict(foot=int(f), start=int(s), end=int(e)) for (f, s, e, *_) in contacts['segments']]
                full = contact_strength(source_segments, n, fps, report['preset']) >= 0.999
                ankle_error = selected_errors[:, [7, 8]]
                report['full_lock_samples'] = int(full.sum())
                report['export_ankle_source_max_m'] = float(ankle_error[full].max()) if full.any() else None
                report['source_surface_height_max_m'] = saved.get('full_lock_height_error_max_m')
                if full.any() and report['export_ankle_source_max_m'] > 0.01:
                    report['warning'] = 'Export foot position differs from the source by over 1 cm at some contacts; inspect the overlay.'
                base_info = json.loads((project / (base + '_mixamo.json')).read_text(encoding='utf-8'))
                sources[base + '_mixamo.json'] = file_stat(project / (base + '_mixamo.json'))
                reference_frames = np.load(project / base_info['frames_file'], mmap_mode='r')
                sources[base_info['frames_file']] = file_stat(project / base_info['frames_file'])
                upper_delta = 0.0
                for start in range(0, n, 128):
                    reference_world = reference_frames[start:start + 128, upper_joints, :3, :3].astype(float) @ export_binding['basis'][list(upper_joints)].swapaxes(-1, -2)
                    rel = frames[start:start + 128, upper_joints, :3, :3].astype(float) @ reference_world.transpose(0, 1, 3, 2)
                    upper_delta = max(upper_delta, float(np.rad2deg(Rotation.from_matrix(rel.reshape(-1, 3, 3)).magnitude()).max()))
                reference_frames._mmap.close()
                # Surface-driven outgoing roll can differ when pose correctives
                # deform the selected skin, even though correction preserves
                # the inferred upper body. Audit these separately: do not hide
                # a source change behind an export-frame comparison.
                report['export_upper_body_rotation_change_deg'] = upper_delta
                base_file,base_key=RESULTS[base]
                base_params=torch.load(project/base_file,map_location='cpu',weights_only=False,mmap=True)[base_key]
                sources[base_file]=file_stat(project/base_file)
                source_upper_delta=0.
                # Explicit palm locking may adjust collar/shoulder/elbow. The
                # torso and wrist world orientations still must match source.
                hand_enabled=bool(saved.get('options',{}).get('hand_lock',False))
                preserved_joints=tuple(j for j in upper_joints if not hand_enabled or j not in (13,14,16,17,18,19))
                # Declared self-contact arm rows are exempt; wrists only on rows a grip declares.
                from .self_contact import audit_exemptions
                exempt=audit_exemptions(saved.get('self_contact'),preserved_joints)
                for start in range(0,n,128):
                    stop=min(n,start+128)
                    pair=[]
                    for values in (base_params,params):
                        angles=torch.cat((values['global_orient'][start:stop,None],values['body_pose'][start:stop].reshape(-1,21,3)),1).numpy()
                        matrices=Rotation.from_rotvec(angles.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
                        for j,p in enumerate(BODY_PARENTS):
                            if p>=0:matrices[:,j]=matrices[:,p]@matrices[:,j]
                        pair.append(matrices[:,preserved_joints])
                    relative=pair[0].swapaxes(-1,-2)@pair[1]
                    angles=np.rad2deg(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).reshape(stop-start,-1)
                    for s,e,ids in exempt:angles[max(s-start,0):max(e-start,0),ids]=0
                    source_upper_delta=max(source_upper_delta,float(angles.max()))
                report['upper_body_rotation_max_deg'] = source_upper_delta
                report['source_preserved_rotation_joints']=list(preserved_joints)
                if source_upper_delta > 0.01:
                    raise ValueError('Source correction changed inferred upper-body rotations. Preview not committed.')
            del selected_reference, selected_errors
            if distal_errors.max() > 0.01:
                raise ValueError('Wrist/ankle world rotation changed during retargeting. Preview not committed: ' + str(fit_summary['distal_orientation_max_deg']))
            for start in range(0, n, 128):
                # Overlay and BVH share the same constant reference basis.
                frames[start:start + 128, :, :3, :3] = frames[start:start + 128, :, :3, :3] @ export_binding['basis']
            for value in arrays:
                value.flush()
                value._mmap.close()
            arrays = []
            for (name, stamp) in sources.items():
                from .project_storage import resolve_reference
                path = resolve_reference(project, name)
                if file_stat(path) != stamp:
                    raise ValueError('Inputs changed while preparing export motion. Retry.')
            if not result_is_current(project, kind):
                raise ValueError('Motion changed while preparing export bones. Apply Correction again.')
            latest = json.loads((project / 'project.json').read_text(encoding='utf-8')) if (project / 'project.json').exists() else {}
            if float(latest.get('fps', 30)) != fps:
                raise ValueError('Frame rate changed while preparing export bones. Retry.')
            from .paths import ROOT
            identity=code_identity()
            payload = dict(version=VERSION,kind=kind,frames=n,fps=fps,names=rig['names'],sources=sources,**files,
                representation='smplx',source_model=state.get('pipeline',{}).get('model','gvhmr'),
                model_path=str(model),model_stat=model_stamp,code=identity,
                code_stats={name:file_stat(ROOT/'mocap'/name) for name in identity},
                correction=report,source_fit=fit_summary,selected_source_fit=selected_summary,
                display_joint_max_m=display_error,display_pose='Selected SMPL-X source surface / separately fitted export bones',
                anatomical_arm_pole_max_deg=arm_pole_error,source_shared_axis_adjustment_max_deg=source_axis_error,
                **policy(fps),
                hinge_axis_max_error=hinge_error,rotation_orthogonality_max_error=orthogonality_error,
                geometry='52-bone humanoid hierarchy including 30 finger channels',
                export_bind=dict(version=export_binding['bind_version'],length_unit='meter',up_axis='Y',
                    rotation_order_default='Per joint',joint_rotation_orders=dict(zip(rig['names'],bvh_encoding['orders'])),minimum_singularity_margin=bvh_encoding['minimum_singularity_margin'],
                    root_offset=export_binding['rest'][0].tolist(),root_translation='Absolute world positions for Blender BVH import',
                    reference='Standing surface-aligned limbs; source-neutral torso/feet; Taunt open hands; actor lengths retained',
                    animation='Constant reference-basis conversion; all joint and tip positions preserved'),
                outputs={name:file_stat(project/name) for name in files.values()})
            atomic_json(project / (kind + '_mixamo.json'), payload)
            pending.unlink(missing_ok=True)
            finished = True
            keep = set(files.values())
            # Only disposable generations of this exact derived target kind.
            for old in (*project.glob(kind + '_mixamo-*'), *project.glob(kind + '_rig-*')):
                if old.is_file() and old.name not in keep:
                    try:
                        old.unlink()
                    except OSError:
                        pass
        finally:
            for value in arrays:
                try:
                    value._mmap.close()
                except (ValueError, OSError):
                    pass
            if not finished:
                for path in created:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
    return project / (result + '_mixamo.json')
