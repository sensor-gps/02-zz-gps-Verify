"""无真值局部去趋势残差比较；不把残差解释为可辨识的绝对噪声。"""
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
from .accuracy_analysis import static_errors

HEADERS=('指标','单位','GPS1 残差 RMS','GPS2 残差 RMS','改善比例','结论','差分残差 RMS','有效点数','多尺度改善范围')
SINGLE_HEADERS=('指标','单位','残差 RMS','残差 STD','绝对残差 P50','绝对残差 P95','最大绝对残差','有效点数','多尺度 RMS 范围')


def analyze_single(data,window=3.):
    """单路自身去趋势；没有对齐、差分或优劣判定。"""
    if not np.isfinite(window) or not 1<=window<=30:raise ValueError('滑窗长度必须在 1～30 秒之间')
    if len(data)<30:raise ValueError('有效唯一 PVT 不足 30 个')
    keys=sorted(data);times=np.array(keys,dtype=float)/1000
    rows={i:data[t] for i,t in enumerate(keys)};first=rows[0]
    position=static_errors(rows,first['lat'],first['lon'],first['alt'])
    values=np.column_stack([[r[k] for r in rows.values()] for k in ('vel_n','vel_e','vel_d')]+[position[k] for k in ('位置北向','位置东向','位置地向')])
    windows=sorted(set((max(1.,window*2/3),window,min(30.,window*5/3))))
    residuals={w:detrend(times,values,w) for w in windows}
    good=np.logical_and.reduce([np.isfinite(r).all(1) for r in residuals.values()])
    if good.sum()<20:raise ValueError('完整多尺度滑窗有效点不足 20 个')
    residual=residuals[window];out=[];fields=[];scale_rows=[]
    for k,title in enumerate(('速度 N','速度 E','速度 D','位置 N','位置 E','位置 D')):
        unit='m/s' if k<3 else 'm';x=residual[good,k]
        rms=[float(np.sqrt(np.mean(r[good,k]**2))) for r in residuals.values()]
        out.append(((title,unit,f'{np.sqrt(np.mean(x*x)):.6g}',f'{np.std(x):.6g}',
                     f'{np.percentile(abs(x),50):.6g}',f'{np.percentile(abs(x),95):.6g}',
                     f'{max(abs(x)):.6g}',int(good.sum()),f'{min(rms):.6g}～{max(rms):.6g}'),'neutral'))
        scale_rows.extend((title,w,int(good.sum()),v) for w,v in zip(windows,rms))
        fields.extend([(f'raw{k}',title,unit),(f'residual{k}',title+' 去趋势残差',unit)])
    plot_rows=[]
    for i in np.flatnonzero(good):
        row={f'raw{k}':values[i,k] for k in range(6)}
        row.update({f'residual{k}':residual[i,k] for k in range(6)})
        plot_rows.append((keys[i],row))
    directory=Path(__file__).resolve().parent.parent/'analysis_output';directory.mkdir(exist_ok=True)
    stem='relative_accuracy_single_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path=directory/(stem+'.csv')
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(SINGLE_HEADERS);writer.writerows(row for row,_ in out)
    detail=directory/(stem+'_residuals.csv')
    with detail.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['itow_ms']+[f'{prefix}{k}' for prefix in ('raw','residual') for k in range(6)])
        for t,row in plot_rows:writer.writerow([t,*[row[f'{prefix}{k}'] for prefix in ('raw','residual') for k in range(6)]])
    with (directory/(stem+'_scales.csv')).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(('指标','窗口秒','有效点数','残差RMS'));writer.writerows(scale_rows)
    summary=f'单 GPS 独立统计，不进行接收机对比或红绿判定。\nfix=3 唯一 PVT {len(data)} 点；多尺度共同有效 {good.sum()} 点。\niTOW：{plot_rows[0][0]}～{plot_rows[-1][0]} ms；主窗口 {window:g} 秒；尺度 {windows}。\n汇总：{path}\n逐点数据：{detail}\n'
    return out,summary,({t:{} for t,_ in plot_rows},[plot_rows],fields)


def detrend(times, values, window):
    """完整居中时间窗二次拟合，至少11点，不跨超过0.5秒空洞。"""
    result=np.full_like(values,np.nan,dtype=float)
    for i,t in enumerate(times):
        lo=np.searchsorted(times,t-window/2);hi=np.searchsorted(times,t+window/2,side='right')
        if hi-lo<11 or t-times[lo]<window*.4 or times[hi-1]-t<window*.4:continue
        if np.max(np.diff(times[lo:hi]))>.5:continue
        x=(times[lo:hi]-t)/(window/2)
        design=np.column_stack((np.ones(len(x)),x,x*x))
        coef=np.linalg.lstsq(design,values[lo:hi],rcond=None)[0]
        result[i]=values[i]-coef[0]
    return result


def analyze(first,second,window=3.,tolerance_ms=60.):
    if not np.isfinite(window) or not 1<=window<=30:raise ValueError('滑窗长度必须在 1～30 秒之间')
    if not np.isfinite(tolerance_ms) or not 0<=tolerance_ms<=500:raise ValueError('匹配容差必须在 0～500 ms 之间')
    a=np.array(sorted(first));b=np.array(sorted(second));pairs=[];used=set()
    # 一对一最近邻，避免低频接收机的同一记录重复参与统计。
    for t in a:
        j=int(np.searchsorted(b,t));options=[k for k in (j-1,j) if 0<=k<len(b) and k not in used]
        if not options:continue
        k=min(options,key=lambda k:abs(int(b[k])-int(t)))
        if abs(int(b[k])-int(t))<=tolerance_ms:pairs.append((int(t),int(b[k])));used.add(k)
    if len(pairs)<30:raise ValueError('一对一共同匹配点不足 30 个，请检查日志或容差')
    ta=np.array([p[0] for p in pairs])/1000.;tb=np.array([p[1] for p in pairs])/1000.
    rows_a={i:first[t] for i,(t,_) in enumerate(pairs)};rows_b={i:second[t] for i,(_,t) in enumerate(pairs)}
    origin=rows_a[0]
    def values(rows):
        position=static_errors(rows,origin['lat'],origin['lon'],origin['alt'])
        return np.column_stack(([ [r[k] for r in rows.values()] for k in ('vel_n','vel_e','vel_d')]+[position[k] for k in ('位置北向','位置东向','位置地向')]))
    va,vb=values(rows_a),values(rows_b)
    windows=sorted(set((max(1.,window*2/3),window,min(30.,window*5/3))))
    residuals={w:(detrend(ta,va,w),detrend(tb,vb,w)) for w in windows}
    ra,rb=residuals[window]
    # 各尺度使用完全相同的共同样本，避免时间覆盖变化造成排序变化。
    good=np.logical_and.reduce([np.isfinite(a).all(1)&np.isfinite(b).all(1) for a,b in residuals.values()])
    if good.sum()<20:raise ValueError('完整滑窗内有效点不足 20 个，请缩短窗口或检查数据空洞')
    ra,rb=ra[good],rb[good];out=[];scale_rows=[]
    for k,title in enumerate(('速度 N','速度 E','速度 D','位置 N','位置 E','位置 D')):
        x,y=np.sqrt(np.mean(ra[:,k]**2)),np.sqrt(np.mean(rb[:,k]**2))
        equal=np.isclose(x,y,rtol=1e-6,atol=1e-9)
        improvements=[]
        for w,(a_res,b_res) in residuals.items():
            ar=a_res[good,k];br=b_res[good,k]
            a_rms=float(np.sqrt(np.mean(ar**2)));b_rms=float(np.sqrt(np.mean(br**2)))
            improvement=(a_rms-b_rms)/a_rms if a_rms>1e-12 else np.nan
            improvements.append(improvement)
            corr=float(np.corrcoef(ar,br)[0,1]) if np.std(ar)>1e-12 and np.std(br)>1e-12 else np.nan
            scale_rows.append((title,w,int(good.sum()),a_rms,b_rms,improvement*100,corr))
        stable_better=all(np.isfinite(v) and v>.05 for v in improvements)
        stable_worse=all(np.isfinite(v) and v<-.05 for v in improvements)
        tag='better' if stable_better else ('worse' if stable_worse else 'neutral')
        conclusion='GPS2 多尺度更平滑' if stable_better else ('GPS2 多尺度波动更大' if stable_worse else '差异小或尺度敏感')
        finite=[v*100 for v in improvements if np.isfinite(v)]
        scale_range=f'{min(finite):+.2f}%～{max(finite):+.2f}%' if finite else '无法比较'
        ratio=f'{(x-y)/x*100:+.2f}%' if x>1e-12 else '—'
        out.append(((title,'m/s' if k<3 else 'm',f'{x:.6g}',f'{y:.6g}',ratio,
                     conclusion,f'{np.sqrt(np.mean((rb[:,k]-ra[:,k])**2)):.6g}',int(good.sum()),scale_range),tag))
    directory=Path(__file__).resolve().parent.parent/'analysis_output';directory.mkdir(exist_ok=True)
    stem='relative_accuracy_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path=directory/(stem+'.csv')
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(HEADERS);writer.writerows(row for row,_ in out)
    detail=directory/(stem+'_residuals.csv')
    scale_path=directory/(stem+'_scales.csv')
    with scale_path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(('指标','窗口秒','共同样本数','GPS1_RMS','GPS2_RMS','改善百分比','两路残差相关系数'));writer.writerows(scale_rows)
    with detail.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['gps1_itow_ms','gps2_itow_ms']+[f'g{g}_{k}' for g in (1,2) for k in ('vn','ve','vd','pn','pe','pd')])
        for i,index in enumerate(np.flatnonzero(good)):writer.writerow([*pairs[index],*ra[i],*rb[i]])
    summary=f'fix=3 唯一 PVT：GPS1 {len(first)}，GPS2 {len(second)}；一对一匹配 {len(pairs)}，完整窗口统计 {good.sum()} 点。\nGPS1 统计 iTOW：{ta[good][0]*1000:.0f}～{ta[good][-1]*1000:.0f} ms；GPS2：{tb[good][0]*1000:.0f}～{tb[good][-1]*1000:.0f} ms。\n窗口 {window:g} 秒，二次趋势，容差 {tolerance_ms:g} ms。未自动补偿相对时延。\n汇总：{path}\n逐点残差：{detail}\n'
    summary+=f'多尺度窗口：{windows} 秒，共同样本一致；仅全部尺度同向且差异超过 5% 时红绿着色（工程判别阈值，非统计显著性）。\n多尺度及残差相关性：{scale_path}\n'
    fields=[]
    for k,title in enumerate(('速度 N','速度 E','速度 D','位置 N','位置 E','位置 D')):
        unit='m/s' if k<3 else 'm'
        fields.extend([(f'raw{k}',title,unit),(f'residual{k}',title+' 去趋势残差',unit)])
    aligned=[]
    for times,values,residual in ((ta,va,residuals[window][0]),(tb,vb,residuals[window][1])):
        rows=[]
        for index in np.flatnonzero(good):
            row={f'raw{k}':values[index,k] for k in range(6)}
            row.update({f'residual{k}':residual[index,k] for k in range(6)})
            rows.append((int(round(times[index]*1000)),row))
        aligned.append(rows)
    return out,summary,({t:{} for t,_ in aligned[0]},aligned,fields)
