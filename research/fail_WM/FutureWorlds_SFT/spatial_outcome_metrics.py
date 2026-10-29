"""Image-plane diagnostics, not a simulator success classifier."""
import numpy as np

VERSION = 'spatial_v1'

def point(frame, color='red', minimum=None):
    x = np.asarray(frame, dtype=np.float32)
    if color == 'red':
        mask = (x[..., 0] > 75) & (x[..., 0] > 1.5*x[..., 1]) & (x[..., 0] > 1.4*x[..., 2])
        minimum = 8 if minimum is None else minimum
    else:
        spread = x.max(-1)-x.min(-1)
        mask = (x[...,1]>63.75)&(x[...,1]>1.18*x[...,0])&(x[...,1]>1.08*x[...,2])&(spread>20.4)
        minimum = 20 if minimum is None else minimum
    y, xx = np.where(mask)
    return {'xy': [float(xx.mean()), float(y.mean())] if len(xx)>=minimum else None,
            'area': int(len(xx))}

def distance(a, b):
    return None if a is None or b is None else float(np.linalg.norm(np.asarray(a)-np.asarray(b)))

def references(real):
    start = point(real[0])['xy']
    end = point(real[-1])['xy']
    goal = point(real[-1], 'green')['xy']
    if goal is None:
        goal = point(real[0], 'green')['xy']
    return {'real':end, 'goal':goal, 'start':start,
            'real_to_start':distance(end,start), 'real_to_goal':distance(end,goal),
            'start_to_goal':distance(start,goal),
            'real_last_area':point(real[-1])['area']}

def endpoint(p, refs):
    ds = {k:distance(p,refs[k]) for k in ('real','goal','start')}
    available = sorted((v,k) for k,v in ds.items() if v is not None)
    if p is None:
        return {'class':'undetected','distances':ds,'ambiguous':True,'far_from_all':None,'nearest_margin':None}
    if not available:
        return {'class':'unreferenced','distances':ds,'ambiguous':True,'far_from_all':None,'nearest_margin':None}
    best, name = available[0]
    margin = available[1][0]-best if len(available)>1 else None
    overlap = len(available)>1 and distance(refs[name], refs[available[1][1]])<5
    return {'class':name+'_like','distances':ds,
            'ambiguous':len(available)<3 or overlap or margin<2,
            'far_from_all':best>10, 'nearest_margin':margin}

def analyze(frames, real):
    if len(frames) != len(real)-2:
        raise ValueError('Predicted horizon differs from ground truth')
    refs = references(real)
    track = [point(f) for f in frames]
    actual = [point(f) for f in real[2:]]
    result = endpoint(track[-1]['xy'], refs)
    errors=[distance(a['xy'],b['xy']) for a,b in zip(track,actual)]
    visible_errors=[x for x in errors if x is not None]
    movement=[distance(p['xy'], refs['start']) for p in track]
    movement=[x for x in movement if x is not None]
    pred_origin = next((p['xy'] for p in track if p['xy'] is not None),None)
    own_motion=[distance(p['xy'], pred_origin) for p in track if p['xy'] is not None]
    own_motion=[x for x in own_motion if x is not None]
    elig = refs['real_to_goal'] is not None and refs['real_to_goal']>=8 and refs['start_to_goal'] is not None and refs['start_to_goal']>=8
    ds=result['distances']
    goalward={}
    for margin in (0,2,4):
        goalward[str(margin)] = (ds['goal']+margin < min(ds['real'],ds['start'])) if elig and track[-1]['xy'] is not None else None
    return {**result, 'references':refs, 'last':track[-1],
            'tail_classes':[endpoint(p['xy'],refs)['class'] for p in track[-5:]],
            'tail_detected':sum(p['xy'] is not None for p in track[-5:]),
            'last_missing':track[-1]['xy'] is None,
            'all_frame_missing':sum(p['xy'] is None for p in track),
            'real_visible_pred_missing':sum(a['xy'] is None and b['xy'] is not None for a,b in zip(track,actual)),
            'compared_frames':len(visible_errors),
            'mean_track_error':float(np.mean(visible_errors)) if visible_errors else None,
            'max_distance_from_real_start':max(movement) if movement else None,
            'max_motion_from_first_prediction':max(own_motion) if own_motion else None,
            'stationary_prediction_proxy':max(own_motion)<6 if len(own_motion)>=0.8*len(track) else None,
            'goalward_eligible':bool(elig), 'goalward_by_margin':goalward,
            'tracking':track, 'frames':len(frames)}
