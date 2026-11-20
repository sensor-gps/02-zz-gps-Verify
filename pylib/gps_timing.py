import numpy as np


def interpolate_with_gap(t, values, query, max_gap_ms):
    """线性插值；不外推，也不跨越超过 max_gap_ms 的空洞。"""
    t=np.asarray(t,dtype=float)
    values=np.asarray(values,dtype=float)
    query=np.asarray(query,dtype=float)
    one_dim=values.ndim==1
    if one_dim: values=values[:,None]
    out=np.full((len(query),values.shape[1]),np.nan)
    if not len(t):
        return out[:,0] if one_dim else out
    right=np.searchsorted(t,query,side='left')
    exact=(right<len(t))&(t[np.minimum(right,len(t)-1)]==query)
    out[exact]=values[right[exact]]
    inside=(~exact)&(right>0)&(right<len(t))
    indexes=np.flatnonzero(inside)
    left,right=right[indexes]-1,right[indexes]
    good=(t[right]-t[left])<=max_gap_ms+1e-9
    indexes,left,right=indexes[good],left[good],right[good]
    weight=((query[indexes]-t[left])/(t[right]-t[left]))[:,None]
    out[indexes]=values[left]*(1-weight)+values[right]*weight
    return out[:,0] if one_dim else out


def _uniform_grid(reference_times, gps_times, sample_interval_ms):
    """在两路有效时间范围的交集内生成均匀采样网格。"""
    if not len(reference_times) or not len(gps_times): return None
    start=max(float(reference_times[0]),float(gps_times[0]))
    end=min(float(reference_times[-1]),float(gps_times[-1]))
    if end<start: return None
    count=int(np.floor((end-start)/sample_interval_ms+1e-9))+1
    if count<1: return None
    return start+np.arange(count,dtype=float)*sample_interval_ms


def estimate_delay(reference, gps, max_ms=2000, step_ms=10,
                   uniform_grid_resample=False, sample_interval_ms=20.0, max_gap_ms=1000.0):
    """Estimate N/E joint delay by centered normalized correlation.

    uniform_grid_resample=True 时，在两路有效范围交集内按 sample_interval_ms
    生成均匀网格，并限制插值跨度不超过 max_gap_ms。正值表示 gps 序列滞后
    reference 序列。
    """
    rt=np.array(sorted(reference),float); rv=np.array([[reference[t].get('vel_n',np.nan),reference[t].get('vel_e',np.nan)] for t in rt])
    gt=np.array(sorted(gps),float); gv=np.array([[gps[t].get('vel_n',np.nan),gps[t].get('vel_e',np.nan)] for t in gt])
    rvok=np.isfinite(rv).all(1); gvok=np.isfinite(gv).all(1)
    rt,rv=rt[rvok],rv[rvok]; gt,gv=gt[gvok],gv[gvok]
    if uniform_grid_resample:
        grid=_uniform_grid(rt,gt,sample_interval_ms)
        if grid is None: return -float(max_ms),-np.inf
        query_t=grid
        query_v=interpolate_with_gap(rt,rv,query_t,max_gap_ms)
        query_ok=np.isfinite(query_v).all(1)
        if query_ok.sum()<20: return -float(max_ms),-np.inf
    else:
        query_t=rt; query_v=rv; query_ok=np.ones(len(rt),dtype=bool)
    scores=[]
    for delay in range(-max_ms,max_ms+1,step_ms):
        if uniform_grid_resample:
            pred=interpolate_with_gap(gt,gv,query_t+delay,max_gap_ms)
        else:
            pred=np.column_stack([np.interp(query_t+delay,gt,gv[:,k],left=np.nan,right=np.nan) for k in range(2)])
        ok=query_ok&np.isfinite(pred).all(1)
        if ok.sum()<20: scores.append((np.inf,delay)); continue
        a=query_v[ok]-query_v[ok].mean(0); b=pred[ok]-pred[ok].mean(0)
        scores.append((-float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)),delay))
    score,delay=min(scores); return float(delay),float(-score)


def estimate_channel_delay(reference, gps, channel, max_ms=2000, step_ms=10,
                           uniform_grid_resample=False, sample_interval_ms=20.0, max_gap_ms=1000.0):
    """Estimate one scalar channel delay by centered normalized correlation."""
    rt=np.array(sorted(reference),float); rv=np.array([reference[t].get(channel,np.nan) for t in rt],float)
    gt=np.array(sorted(gps),float); gv=np.array([gps[t].get(channel,np.nan) for t in gt],float)
    goodr=np.isfinite(rv); goodg=np.isfinite(gv)
    rt,rv=rt[goodr],rv[goodr]; gt,gv=gt[goodg],gv[goodg]
    if len(rt)<20 or len(gt)<20: return np.nan,np.nan,np.nan
    if uniform_grid_resample:
        grid=_uniform_grid(rt,gt,sample_interval_ms)
        if grid is None: return np.nan,np.nan,np.nan
        query_t=grid
        query_v=interpolate_with_gap(rt,rv,query_t,max_gap_ms)
        query_ok=np.isfinite(query_v)
        if query_ok.sum()<20: return np.nan,np.nan,np.nan
    else:
        query_t=rt; query_v=rv; query_ok=np.ones(len(rt),dtype=bool)
    best=(-np.inf,np.nan,np.nan)
    for delay in range(-max_ms,max_ms+1,step_ms):
        if uniform_grid_resample:
            q=interpolate_with_gap(gt,gv,query_t+delay,max_gap_ms)
        else:
            q=np.interp(query_t+delay,gt,gv,left=np.nan,right=np.nan)
        ok=query_ok&np.isfinite(q)
        if ok.sum()<20: continue
        a=query_v[ok]-query_v[ok].mean(); b=q[ok]-q[ok].mean()
        den=np.linalg.norm(a)*np.linalg.norm(b)
        if den<1e-12: continue
        corr=float(np.sum(a*b)/den)
        rmse=float(np.sqrt(np.mean((q[ok]-query_v[ok])**2)))
        if corr>best[0]: best=(corr,float(delay),rmse)
    return best[1],best[0],best[2]


def estimate_clock_offset(reference, gps, max_s=120.0, step_s=0.25):
    """Use horizontal speed signatures to find the large FFC/GPS clock offset."""
    rt=np.array(sorted(reference),float); rs=np.array([np.hypot(reference[t].get('vel_n',np.nan),reference[t].get('vel_e',np.nan)) for t in rt])
    gt=np.array(sorted(gps),float); gs=np.array([np.hypot(gps[t].get('vel_n',np.nan),gps[t].get('vel_e',np.nan)) for t in gt])
    goodr=np.isfinite(rs); goodg=np.isfinite(gs)
    rt,rs=rt[goodr],rs[goodr]; gt,gs=gt[goodg],gs[goodg]
    if len(rt)<20 or len(gt)<20: return float(min(gt)-min(rt))
    grid=np.arange(max(min(rt),min(gt)-max_s*1000),min(max(rt),max(gt)+max_s*1000)+1,100.0)
    best=(-np.inf,0.0)
    for off in np.arange(-max_s,max_s+step_s/2,step_s):
        # FFC time + off seconds is compared with GPS time.
        q=np.interp(grid-off*1000,rt,rs,left=np.nan,right=np.nan); g=np.interp(grid,gt,gs,left=np.nan,right=np.nan)
        ok=np.isfinite(q)&np.isfinite(g)
        if ok.sum()<30: continue
        a=q[ok]-q[ok].mean(); b=g[ok]-g[ok].mean(); score=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        if score>best[0]: best=(score,off)
    return float(min(gt)-min(rt)+best[1]*1000)
