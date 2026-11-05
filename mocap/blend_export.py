"""PR #18-derived portable source animation, source video and camera bundles."""
import json
from pathlib import Path
import numpy as np
from .blend_support import Inputs,Bundle,read,copy_video,build,disk_check


def person_camera(project,kind,count,scale,inputs):
    """(T,4,4) world-to-camera of a moving ExMR take as review/render project it: paired per frame with the ORIGINAL world and
    in-camera roots, centres following the root change Motion Correction made. `scale` scales the centre path from the original
    frame-0 centre. Returns (track, followed)."""
    import torch
    from scipy.spatial.transform import Rotation
    from .camera_reference import root_change
    from .motion_tools import base_result
    from .result_artifacts import artifacts_current
    motion=torch.load(inputs.track(project/'hmr4d_results.pt'),map_location='cpu',weights_only=True,mmap=True)
    world,incam=motion['smpl_params_global'],motion['smpl_params_incam']
    with np.load(inputs.track(project/'initial_joints.npz'),allow_pickle=False) as data:root=np.asarray(data['joints'][:,0],float)
    rotation=(Rotation.from_rotvec(incam['global_orient'].numpy()).as_matrix()
              @Rotation.from_rotvec(world['global_orient'].numpy()).as_matrix().transpose(0,2,1))
    tw,tc=world['transl'].numpy().astype(float),incam['transl'].numpy().astype(float)
    if not len(rotation)==len(tw)==len(tc)==len(root)==count:raise ValueError('Saved camera does not cover the selected motion')
    # scene_camera.paired_camera per frame: SMPL translation rotates about the rest pelvis, so both roots pivot there.
    pelvis=root-tw;centre=tw+pelvis-np.einsum('tji,tj->ti',rotation,tc+pelvis);origin=centre[0].copy()
    delta=None;roots=[project/(name+'_joints.npz') for name in (kind,base_result(kind))] if kind.startswith('corrected_') else []
    if roots and all(p.is_file() for p in roots) and artifacts_current(project,kind):
        for p in roots:inputs.track(p)
        delta=root_change(project,kind,count)
    if delta is not None:centre=centre+delta
    centre=origin+scale*(centre-origin)
    if not np.isfinite(rotation).all() or not np.isfinite(centre).all():raise ValueError('Invalid saved camera path')
    track=np.tile(np.eye(4),(count,1,1));track[:,:3,:3]=rotation;track[:,:3,3]=-np.einsum('tij,tj->ti',rotation,centre)
    return track,delta is not None


def run(request,emit=lambda *a,**k:None):
    from .motion_tools import selected_export
    from .scene_export import context
    from .fbx_export import run as export_fbx
    from .camera_tracks import load,at
    from .camera_background_worker import refined,current
    from .video import probe
    request=Path(request);bundles=[];checks=[];cache={}
    try:
        rows=read(request)
        if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('Invalid Blender export request')
        folders=set()
        for row in rows:
            folder=Path(row['target']).resolve().parent
            if str(folder).casefold() in folders:raise ValueError('Duplicate Blender bundle destination')
            folders.add(str(folder).casefold())
        for row in rows:
            project=Path(row['project']).resolve();kind=row['kind'];source=Path(row['source']).resolve();inputs=Inputs()
            if selected_export(project,kind)!=source:raise ValueError('Selected motion changed. Prepare Preview or Apply Correction first.')
            state=read(inputs.track(project/'project.json'));info=read(inputs.track(project/(kind+'_mixamo.json')))
            for key in ('asset_file','frames_file'):
                if Path(info[key]).name!=info[key]:raise ValueError('Invalid committed animation path')
                inputs.track(project/info[key])
            inputs.track(source);scene=row.get('scene')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Shared scene placement changed')
            group=scene['root'] if scene is not None else None
            cameras=load(project,group,kind=kind)
            if row.get('background_refine',False) and row.get('animate_camera',True):
                cameras=refined(cameras,project,lambda:False,
                    lambda done,total:emit('progress',stage='save_blend',current=done,total=total,message='Refining camera rotation'),cache)
            for item in cache.values():
                if not current(item['sources']):raise ValueError('Background camera inputs changed during export')
                for p,_ in item['sources']:inputs.track(p)
            if not current(list(cameras['camera_sources'].items())):raise ValueError('Camera sources changed during export')
            for p in cameras['camera_sources']:inputs.track(p)
            camera=cameras['cameras'][0];n=int(info['frames']);fps=float(info['fps'])
            if abs(camera['K'][0,1])>1e-6:raise ValueError('Blender export requires camera intrinsics without skew')
            if n!=int(state['frames']) or abs(fps-state['fps'])>1e-6:raise ValueError('Video and selected animation timing differ')
            scale=float(row.get('camera_translation_scale',1.))
            if not np.isfinite(scale) or not 0<=scale<=100:raise ValueError('Invalid camera movement scale')
            animate=row.get('animate_camera',True)
            times=state['start']+np.arange(n)/fps
            track=[at(camera,t,scale if animate else 0.) for t in times]
            if any(m is None for m in track):raise ValueError('Saved camera does not cover the selected motion')
            track=np.asarray(track);mode=cameras['note']
            if not animate:track[:]=track[0]
            elif cameras['model']=='exmr' and scene is None and not cameras['static_camera'] and not row.get('background_refine',False):
                # The visual-odometry rotation track differs from the body's paired rotation by up to ~12 deg, tilting the body off the person.
                track,follows=person_camera(project,kind,n,scale,inputs)
                mode=('ExMR camera paired per frame with the original world/camera roots, as review and render'
                      +(', following the root change of Motion Correction' if follows else '')+'. Approximate, not measured motion.')
            video=inputs.track(project/'input.mp4');media=probe(video)
            if (media['frames']!=n or abs(media['fps']-fps)>.001
                    or [media['width'],media['height']]!=list(camera['size'])):
                raise ValueError('Source video timing or resolution differs from the saved camera')
            disk_check(Path(row['target']).parent.parent,video.stat().st_size+source.stat().st_size*4)
            bundle=Bundle(row['target'],request);bundles.append(bundle);work=bundle.work
            copy_video(video,work/'video.mp4',emit,'save_blend')
            nested=work/'fbx-request.json';nested.write_text(json.dumps([{**row,'target':str(work/'animation.fbx')}]),encoding='utf-8')
            export_fbx(nested,emit)
            np.save(work/'camera.npy',track,allow_pickle=False)
            job=dict(target=bundle.target.name,frames=n,fps=fps,width=media['width'],height=media['height'],
                world_to_camera=track[0].tolist(),K=camera['K'].tolist(),prepend_tpose=bool(row.get('prepend_tpose')),
                camera_track='camera.npy' if animate and not np.array_equal(track,np.broadcast_to(track[0],track.shape)) else None,
                camera_mode=mode)
            (work/'blend-job.json').write_text(json.dumps(job),encoding='utf-8')
            emit('progress',stage='save_blend',current=0,total=0,message='Building Blender project')
            build(work,'blend_export_scene.py',emit,'save_blend')
            report=dict(model=cameras['model'],result=kind,frames=n,fps=fps,video_start_frame=int(job['prepend_tpose']),
                camera_mode=mode,camera_warning=cameras.get('warning',''),animate_camera=bool(animate),
                camera_movement_scale=scale,background_refinement=cameras.get('background_report'),
                camera_translation='Approximate display estimate, not measured metric camera motion')
            (work/'camera_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            (work/'README.txt').write_text('Open '+bundle.target.name+' in Blender 4.5 or newer.\n'
                'Keep video.mp4 beside the .blend file when moving this folder.\n'
                'Numpad 0: camera view. Space: playback. The video is a viewport background, not a render background.\n'
                'The source armature is editable. An optional T-pose is one extra animation frame; the video starts after it.\n'
                'Camera movement may include reconstruction error and arbitrary scale; it is not measured calibration.\n'
                'See camera_report.json. Source motion and original camera files were not changed.\n',encoding='utf-8')
            for name in ('animation.fbx','blend-job.json','blender.log','camera.npy'):(work/name).unlink(missing_ok=True)
            checks.append((inputs,project,kind,source,scene))
        for inputs,project,kind,source,scene in checks:
            inputs.verify()
            if selected_export(project,kind)!=source:raise ValueError('Motion changed during export')
            if scene is not None and context(scene['root'],project)!=scene:raise ValueError('Scene placement changed during export')
        for bundle in bundles:bundle.publish()
        return len(bundles)
    finally:
        for bundle in bundles:bundle.close()
        request.unlink(missing_ok=True)
