"""精度指标计算：静态已知坐标，或相对 ULog 的速度误差与去偏轨迹误差。"""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np


def metrics(errors):
    x = np.asarray(errors, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return (len(x), float(np.mean(x)), float(np.std(x)),
            float(np.sqrt(np.mean(x*x))), float(np.mean(np.abs(x))),
            float(np.percentile(np.abs(x), 50)), float(np.percentile(np.abs(x), 95)),
            float(np.max(np.abs(x))))


def ecef(lat, lon, altitude):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    a, e2 = 6378137.0, 6.69437999014e-3
    n = a / np.sqrt(1-e2*np.sin(lat)**2)
    return np.array([(n+altitude)*np.cos(lat)*np.cos(lon),
                     (n+altitude)*np.cos(lat)*np.sin(lon),
                     (n*(1-e2)+altitude)*np.sin(lat)])


def static_errors(data, lat, lon, alt):
    if not np.isfinite([lat,lon,alt]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError('参考经纬度或海拔不合法')
    rows = list(data.values())
    # 水平投影统一使用零椭球高；垂直误差单独采用 NAV-PVT 海拔高程。
    delta = ecef(np.array([r['lat'] for r in rows]), np.array([r['lon'] for r in rows]), 0)-ecef(lat,lon,0)[:,None]
    p,l = np.deg2rad([lat,lon])
    north = np.array([-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)]) @ delta
    east = np.array([-np.sin(l),np.cos(l),0]) @ delta
    down = alt-np.array([r['alt'] for r in rows])
    return {'位置北向':north, '位置东向':east, '位置地向':down,
            '位置水平':np.hypot(north,east), '位置三维':np.sqrt(north*north+east*east+down*down)}


def velocity_errors(reference, data, tolerance):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('对齐容差必须是非负有限值')
    rt = np.array(sorted(reference),dtype=float)
    gt = np.array(sorted(data),dtype=float)
    rv = np.array([[reference[t].get(k,np.nan) for k in ('vel_n','vel_e','vel_d')] for t in rt])
    gv = np.array([[data[t].get(k,np.nan) for k in ('vel_n','vel_e','vel_d')] for t in gt])
    good=np.isfinite(rv).all(1);rt,rv=rt[good],rv[good]
    if not len(rt):raise ValueError('真值中没有有效的三轴速度')
    j=np.searchsorted(rt,gt);left=np.clip(j-1,0,len(rt)-1);right=np.clip(j,0,len(rt)-1)
    nearest=np.minimum(np.abs(rt[left]-gt),np.abs(rt[right]-gt))
    # 只在 GPS 独立历元计数，不重复扩增 GPS 样本；不跨大空洞插值。
    bracket=(right==left)|((rt[right]-rt[left])<=2*tolerance)
    ok=(gt>=rt[0])&(gt<=rt[-1])&(nearest<=tolerance)&bracket&np.isfinite(gv).all(1)
    if not ok.any():raise ValueError('GPS 与真值没有满足容差的有效速度样本')
    q=np.column_stack([np.interp(gt[ok],rt,rv[:,k]) for k in range(3)])
    err=gv[ok]-q
    return {'速度北向':err[:,0], '速度东向':err[:,1], '速度地向':err[:,2],
            '速度水平矢量':np.linalg.norm(err[:,:2],axis=1), '速度三维矢量':np.linalg.norm(err,axis=1),
            '水平速率差':np.linalg.norm(gv[ok,:2],axis=1)-np.linalg.norm(q[:,:2],axis=1)}


def compensate_series(data, delay_ms):
    """NE 时延补偿：GPS 滞后 delay_ms 时，把其时间戳整体前移。"""
    shift=int(round(delay_ms))
    return {int(round(t))-shift: row for t,row in data.items()}


def compensate_delays(reference, series):
    """逐路估计 GPS 相对真值的 NE 时延，并返回补偿后的序列。

    返回 (补偿后的 series, [(GPS 编号, 补偿时延 ms, 相关系数), ...])。
    时延为正表示 GPS 滞后真值，补偿时把 GPS 时间戳前移同样毫秒数。
    """
    from .gps_timing import estimate_delay
    adjusted=[]; metadata=[]
    for number,data in series:
        delay,correlation=estimate_delay(reference,data)
        if not np.isfinite(delay) or not np.isfinite(correlation):
            raise ValueError(f'GPS {number} 无法估计 NE 时延（有效运动或重叠样本不足）；请取消勾选“时延补偿(NE)”')
        shift=int(round(delay))
        adjusted.append((number,compensate_series(data,shift)))
        metadata.append((number,shift,float(correlation)))
    return adjusted,metadata


def evaluate(series, mode, reference=None, coordinates=None, tolerance=200):
    result=[]
    for number,data in series:
        errors=(static_errors(data,*coordinates) if mode=='static'
                else velocity_errors(reference,data,tolerance))
        if mode!='static':
            errors.update(trajectory_errors(reference,data,tolerance))
        for channel,values in errors.items():
            stats=metrics(values)
            if stats:result.append((f'GPS {number}',channel,'m' if mode=='static' or channel.startswith('轨迹') else 'm/s',*stats))
    if not result:raise ValueError('没有可用于评估的有效样本')
    return result


def trajectory_errors(reference,data,tolerance):
    rt=np.array(sorted(reference),dtype=float)
    rv=np.array([[reference[t].get(k,np.nan) for k in ('pos_n','pos_e','pos_d')] for t in rt])
    good=np.isfinite(rv).all(1);rt,rv=rt[good],rv[good]
    if len(rt)<2:raise ValueError('真值缺少有效局部位置，无法评估相对轨迹')
    gt=np.array(sorted(data),dtype=float)
    j=np.searchsorted(rt,gt);lo=np.clip(j-1,0,len(rt)-1);hi=np.clip(j,0,len(rt)-1)
    ok=(gt>=rt[0])&(gt<=rt[-1])&(np.minimum(abs(rt[lo]-gt),abs(rt[hi]-gt))<=tolerance)&((rt[hi]-rt[lo])<=2*tolerance)
    times=gt[ok]
    if len(times)<3:raise ValueError('轨迹有效匹配样本不足三个')
    rows={int(t):data[t] for t in times}
    first=rows[int(times[0])]
    local=static_errors(rows,first['lat'],first['lon'],first['alt'])
    gps=np.column_stack([local[k] for k in ('位置北向','位置东向','位置地向')])
    ref=np.column_stack([np.interp(times,rt,rv[:,k]) for k in range(3)])
    err=(gps-gps[0])-(ref-ref[0])
    # 首个样本用于确定原点，不将其必然为零的误差计入统计。
    err=err[1:]
    # 相对轨迹误差的均值可能不为零（例如两路轨迹的漂移趋势不同）：
    # 逐轴求平均偏差并扣除，剩余部分再做精度统计。
    err=err-np.nanmean(err,axis=0,keepdims=True)
    return {'轨迹北向':err[:,0],'轨迹东向':err[:,1],'轨迹地向':err[:,2],
            '轨迹水平':np.linalg.norm(err[:,:2],axis=1),'轨迹三维':np.linalg.norm(err,axis=1)}


def curve_data(reference,series,tolerance):
    """以实际统计误差生成曲线；位置误差沿用去初始偏置及去均值口径。"""
    aligned=[];truth={}
    for number,data in series:
        rt=np.array(sorted(reference),float)
        rv=np.array([[reference[t].get(k,np.nan) for k in ('vel_n','vel_e','vel_d','pos_n','pos_e','pos_d')] for t in rt])
        finite=np.isfinite(rv).all(1);rt,rv=rt[finite],rv[finite]
        gt=np.array(sorted(data),float);j=np.searchsorted(rt,gt)
        lo=np.clip(j-1,0,len(rt)-1);hi=np.clip(j,0,len(rt)-1)
        ok=(gt>=rt[0])&(gt<=rt[-1])&(np.minimum(abs(rt[lo]-gt),abs(rt[hi]-gt))<=tolerance)&((rt[hi]-rt[lo])<=2*tolerance)
        times=gt[ok]
        if len(times)<2:raise ValueError(f'GPS {number} 的曲线匹配样本不足')
        rows={int(t):data[t] for t in times};first=rows[int(times[0])]
        local=static_errors(rows,first['lat'],first['lon'],first['alt'])
        position=np.column_stack([local[k] for k in ('位置北向','位置东向','位置地向')])
        q=np.column_stack([np.interp(times,rt,rv[:,k]) for k in range(6)])
        relative=q[:,3:]-q[0,3:]
        bias=np.mean((position-relative)[1:],axis=0)
        output=[]
        for i,t in enumerate(times):
            row={k:rows[int(t)][k] for k in ('vel_n','vel_e','vel_d')}
            row.update({f'position{k}':position[i,k]-bias[k] for k in range(3)})
            row.update({f'error{k}':row[key]-q[i,k] for k,key in enumerate(('vel_n','vel_e','vel_d'))})
            row.update({f'position_error{k}':position[i,k]-relative[i,k]-bias[k] if i else np.nan for k in range(3)})
            # 每路位置原点不同，单独携带对应参考，绘图显示误差页以避免混用原点。
            output.append((int(t),row))
            truth[int(t)]={k:q[i,kidx] for kidx,k in enumerate(('vel_n','vel_e','vel_d'))}
        aligned.append(output)
    fields=[(k,title,'m/s') for k,title in zip(('vel_n','vel_e','vel_d'),('北向速度','东向速度','地向速度'))]
    fields += [(f'error{k}',f'{direction} 速度误差','m/s') for k,direction in enumerate(('N','E','D'))]
    fields += [(f'position_error{k}',f'{direction} 相对位置误差','m') for k,direction in enumerate(('N','E','D'))]
    return truth,aligned,fields


HEADERS=('GPS','指标','单位','样本数','均值/偏差','标准差','RMSE','MAE','绝对误差P50','绝对误差P95','最大绝对误差')

COMPARISON_HEADERS=('指标','单位','GPS1','对比对象','对比值','误差变化','改善比例','结论')

# 原始统计行索引：标准差=5，RMSE=6，P50=8，P95=9。
COMPARISON_METRICS=(
    ('水平位置 RMSE','轨迹水平',6,'m'),
    ('水平位置 CEP50','轨迹水平',8,'m'),
    ('水平位置 CEP95','轨迹水平',9,'m'),
    ('垂直位置 CEP50','轨迹地向',8,'m'),
    ('垂直位置 CEP95','轨迹地向',9,'m'),
    ('三维位置 RMSE','轨迹三维',6,'m'),
    ('三维位置模长 STD','轨迹三维',5,'m'),
    ('三维速度 RMSE','速度三维矢量',6,'m/s'),
    ('水平速度 CEP50','速度水平矢量',8,'m/s'),
    ('水平速度 CEP95','速度水平矢量',9,'m/s'),
    ('垂直速度 CEP50','速度地向',8,'m/s'),
    ('垂直速度 CEP95','速度地向',9,'m/s'),
)


def compare(result):
    """固定十二项指标；所有值均为非负误差，越小越好。"""
    lookup={(r[0],r[1]):r for r in result}
    receivers=list(dict.fromkeys(r[0] for r in result if r[0]!='GPS 1')) or ['GPS 1']
    output=[]
    for title,channel,index,unit in COMPARISON_METRICS:
        reference=lookup.get(('GPS 1',channel))
        for receiver in receivers:
            row=lookup.get((receiver,channel))
            a=float(reference[index]) if reference else np.nan
            b=float(row[index]) if row else np.nan
            if receiver=='GPS 1' or not np.isfinite([a,b]).all():
                output.append(((title,unit,f'{a:.6g}' if np.isfinite(a) else '无数据',
                                receiver,f'{b:.6g}' if np.isfinite(b) else '无数据',
                                '—','—','仅展示，无对比'),'neutral'))
                continue
            delta=b-a
            equal=np.isclose(a,b,rtol=1e-9,atol=1e-12)
            tag='neutral' if equal else ('better' if delta<0 else 'worse')
            conclusion='相同' if equal else ('优于 GPS1' if delta<0 else '劣于 GPS1')
            ratio=f'{(a-b)/a*100:+.2f}%' if a>1e-12 else ('0.00%' if equal else '基准为零')
            output.append(((title,unit,f'{a:.6g}',receiver,f'{b:.6g}',
                            f'{delta:+.6g}',ratio,conclusion),tag))
    return output


def export_comparison(result):
    directory=Path(__file__).resolve().parent.parent/'analysis_output'
    directory.mkdir(exist_ok=True)
    path=directory/('accuracy_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(COMPARISON_HEADERS)
        writer.writerows(values for values,_ in compare(result))
    return path


def export(result, mode):
    directory=Path(__file__).resolve().parent.parent/'analysis_output'
    directory.mkdir(exist_ok=True)
    path=directory/('accuracy_'+mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(HEADERS);writer.writerows(result)
    return path
