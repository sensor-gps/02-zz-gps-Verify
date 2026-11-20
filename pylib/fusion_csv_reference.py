"""Estimate body-to-NED yaw from ffc.csv using least squares."""
import csv, math
import numpy as np

def load(path, min_speed=0.5):
    csv.field_size_limit(10_000_000)
    with open(path,encoding='utf-8-sig',newline='') as f:
        r=csv.reader(f); header=next(r); time_key=next((k for k in ('iTOW','itow','gps_iTOW','sensor.gps.iTOW','t') if k in header),None)
        if not time_key: raise ValueError('CSV ?? iTOW/itow/t ???')
        need=[time_key,'sensor.gps.velE','sensor.gps.velN','preint.vel.x','preint.vel.y']
        true_yaw_key='est.trueNorth.yaw(deg)' if 'est.trueNorth.yaw(deg)' in header else None
        pre_yaw_key='preint.euler.yaw(deg)' if 'preint.euler.yaw(deg)' in header else None
        if true_yaw_key and pre_yaw_key: need += [true_yaw_key,pre_yaw_key]
        ix={k:header.index(k) for k in need}; ts=[]; ge=[]; gn=[]; vx=[]; vy=[]; true_yaws=[]; pre_yaws=[]
        for row in r:
            if len(row)<=max(ix.values()): continue
            try:
                vals=[float(row[ix[k]]) for k in need]
                if not np.isfinite(vals).all(): continue
                t,e,n,x,y=vals[:5]
                if time_key != 't' and t < 1000000: t *= 1000
                # 保留全部有限行用于最终真值输出；是否运动仅在 yaw/延迟拟合时筛选。
                ts.append(t); ge.append(e); gn.append(n); vx.append(x); vy.append(y)
                if true_yaw_key and pre_yaw_key:
                    true_yaws.append(vals[5]); pre_yaws.append(vals[6])
            except (ValueError,TypeError): continue
    if len(ts)<20: raise ValueError('ffc.csv 中有效 GPS/VIO 速度样本不足')
    ts=np.asarray(ts); ge=np.asarray(ge); gn=np.asarray(gn); vx=np.asarray(vx); vy=np.asarray(vy)
    if true_yaw_key and pre_yaw_key:
        true_yaws=np.asarray(true_yaws); pre_yaws=np.asarray(pre_yaws)
    fitmask=(np.hypot(ge,gn)>=min_speed)&(np.hypot(vx,vy)>=min_speed)
    if fitmask.sum()<20: raise ValueError('ffc.csv 中有效运动速度样本不足')
    fit_t,fit_ge,fit_gn,fit_vx,fit_vy=ts[fitmask],ge[fitmask],gn[fitmask],vx[fitmask],vy[fitmask]
    # 先搜索时间偏移，再在每个偏移下用二维正交 Procrustes 最小二乘估计 yaw。
    # delay>0 表示 GPS 相对 preint 滞后。
    best=None
    for delay in np.arange(-2000,2001,10):
        qx=np.interp(fit_t-delay/1000.0,ts,vx,left=np.nan,right=np.nan)
        qy=np.interp(fit_t-delay/1000.0,ts,vy,left=np.nan,right=np.nan)
        ok=np.isfinite(qx+qy)
        if ok.sum()<20: continue
        dot=np.sum(fit_ge[ok]*qx[ok]+fit_gn[ok]*qy[ok]); cross=np.sum(fit_ge[ok]*qy[ok]-fit_gn[ok]*qx[ok])
        yaw0=math.atan2(cross,dot); c0,s0=math.cos(yaw0),math.sin(yaw0)
        ee=c0*qx[ok]-s0*qy[ok]; nn=s0*qx[ok]+c0*qy[ok]
        rms=float(np.sqrt(np.mean((fit_ge[ok]-ee)**2+(fit_gn[ok]-nn)**2)))
        if best is None or rms<best[0]: best=(rms,float(delay),yaw0)
    if best is None: raise ValueError('无法从 FFC 速度估计 yaw 和延迟')
    rms,delay,yaw=best
    yaw_source='least_squares'
    if true_yaw_key and pre_yaw_key:
        # 真北航向与 preint 航向的差是随时间记录的，使用圆周中位数作为安装偏角。
        # 角度差必须逐点环绕到 [-pi, pi]；对差值直接 unwrap 会在跨越
        # ±180° 时累积整圈，产生 579° 这类不可能的安装偏角。
        delta=np.arctan2(np.sin(np.deg2rad(true_yaws-pre_yaws)),
                         np.cos(np.deg2rad(true_yaws-pre_yaws)))
        valid=np.isfinite(delta)&fitmask
        if valid.sum()>=20:
            yaw=float(np.arctan2(np.mean(np.sin(delta[valid])),np.mean(np.cos(delta[valid])))); yaw_source='est.trueNorth.yaw - preint.euler.yaw'
            c,s=math.cos(yaw),math.sin(yaw)
            qx=np.interp(ts-delay/1000.0,ts,vx); qy=np.interp(ts-delay/1000.0,ts,vy)
            ee=c*qx-s*qy; nn=s*qx+c*qy
            ok=fitmask&np.isfinite(ee+nn)
            rms=float(np.sqrt(np.mean((ge[ok]-ee[ok])**2+(gn[ok]-nn[ok])**2)))
        else: c,s=math.cos(yaw),math.sin(yaw)
    else: c,s=math.cos(yaw),math.sin(yaw)
    qx=np.interp(ts-delay/1000.0,ts,vx); qy=np.interp(ts-delay/1000.0,ts,vy)
    east=c*qx-s*qy; north=s*qx+c*qy
    records={int(round(t*1000)):{'vel_n':float(north[i]),'vel_e':float(east[i]),'vel_d':np.nan,
             'speed':float(math.hypot(north[i],east[i]))} for i,t in enumerate(ts)}
    return records,((math.degrees(yaw)+180)%360)-180,len(ts),rms,delay,yaw_source


def estimate_delay(reference, gps, max_ms=2000, step_ms=10):
    """???? GPS ???????"""
    rt=np.array(sorted(reference),float); rv=np.array([[reference[t].get('vel_n',np.nan),reference[t].get('vel_e',np.nan)] for t in rt])
    gt=np.array(sorted(gps),float); gv=np.array([[gps[t].get('vel_n',np.nan),gps[t].get('vel_e',np.nan)] for t in gt])
    gvok=np.isfinite(gv).all(1); rvok=np.isfinite(rv).all(1)
    rt,rv=rt[rvok],rv[rvok]; gt,gv=gt[gvok],gv[gvok]
    scores=[]
    for delay in range(-max_ms,max_ms+1,step_ms):
        pred=np.column_stack([np.interp(rt+delay,gt,gv[:,k],left=np.nan,right=np.nan) for k in range(2)])
        ok=np.isfinite(pred).all(1)
        if ok.sum()<20: scores.append((np.inf,delay)); continue
        a=rv[ok]-rv[ok].mean(0); b=pred[ok]-pred[ok].mean(0)
        scores.append((-float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)),delay))
    score,delay=min(scores); return float(delay),float(-score)

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
