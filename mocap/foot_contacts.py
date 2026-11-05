"""Sparse foot-lock timing records. Frames are zero-based, end-exclusive."""
import json
import math
from pathlib import Path

BASES=('initial','refined','footmr_initial','footmr_refined')
BASES=(*BASES,*('handflow_'+name for name in BASES))
FOOT_COLORS=('#56b6a7','#c78f5f')


def normalize_ranges(ranges, frames):
    if len(ranges) not in (2,4):raise ValueError('Expected two foot or four foot/hand tracks')
    result=[]
    for track in ranges:
        normalized=[]
        for pair in sorted(track):
            if len(pair)!=2:raise ValueError('Invalid foot-lock interval')
            start,end=pair
            if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=frames:
                raise ValueError('Foot-lock intervals must be within the motion frame range')
            if normalized and start<normalized[-1][1]:
                normalized[-1]=(normalized[-1][0],max(end,normalized[-1][1]))
            else:normalized.append((start,end))
        result.append(normalized)
    return result


def segment_ranges(segments):
    return [[(int(s['start']),int(s['end'])) for s in segments if s['foot']==foot] for foot in range(2)]


def state_path(project, base):
    if base not in BASES:raise ValueError('Unknown foot-lock base')
    return Path(project)/f'footlock_{base}_state.json'


def contact_state(project, base):
    path=state_path(project,base)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def footlock_revision(project, base):
    return contact_state(project,base).get('revision')


def with_self_contacts(project, base, revision):
    # Without a self-contact file the value is exactly the foot-lock revision, so existing stale checks are unchanged.
    from .self_contact_store import revision as self_contact_revision
    extra=self_contact_revision(project,base)
    return revision if extra is None else f'{revision or ""}|sc:{extra}'


def contact_revision(project, base):
    return with_self_contacts(project,base,footlock_revision(project,base))


def read_edit(project, base, context):
    ranges,revision,warning=read_footlock_edit(project,base,context)
    return ranges,with_self_contacts(project,base,revision),warning


def read_footlock_edit(project, base, context):
    state=contact_state(project,base)
    if not state.get('manual'):return None,state.get('revision'),''
    if any(state.get(key)!=context.get(key) for key in ('frames','fps','sources')):
        return None,state.get('revision'),'Saved foot-lock edits belong to older motion; automatic contacts are used. Review the tracks.'
    filename=state.get('file')
    if filename not in (f'footlock_{base}_edit0.json',f'footlock_{base}_edit1.json'):
        raise ValueError('Invalid foot-lock edit file')
    data=json.loads((Path(project)/filename).read_text(encoding='utf-8'))
    if data.get('revision')!=state['revision']:raise ValueError('Incomplete foot-lock save; edit the tracks again')
    return normalize_ranges(data['ranges'],context['frames']),state['revision'],''


def save_edit(project, base, ranges, context):
    """Two bounded payload slots plus an atomic small commit record."""
    import uuid
    from .storage import atomic_json
    frames=int(context['frames']);fps=float(context['fps'])
    if frames<1 or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid foot-lock time base')
    previous=contact_state(project,base)
    revision=uuid.uuid4().hex
    state={**context,'revision':revision,'manual':ranges is not None}
    if ranges is not None:
        ranges=normalize_ranges(ranges,frames)
        filename=f'footlock_{base}_edit{0 if previous.get("file","").endswith("edit1.json") else 1}.json'
        atomic_json(Path(project)/filename,{'revision':revision,'ranges':ranges})
        state['file']=filename
    atomic_json(state_path(project,base),state)
    return state
def contact_levels(track,frame):
    """Sparse per-stance heights; None means the legacy common floor."""
    import bisect
    if not track or track.get('mode','flat')!='multi_level':return None
    heights=[]
    for foot in (0,1):
        rows=track['levels'][foot];slot=bisect.bisect_right(track['level_starts'][foot],frame)-1
        heights.append(rows[slot][2] if slot>=0 and frame<rows[slot][1] else float('nan'))
    return heights

