"""由机体系与 NED 水平速度序列估计固定 yaw_diff（安装偏角）。

约定与 pylib.fusion_reference 一致：

    yaw_diff = drone_fusion.yaw - gps.yaw
    实际旋转角 psi = -yaw_diff

对每个有效时刻，寻找使 R(psi) @ [vel_x, vel_y] 最接近 [vel_n, vel_e] 的 psi，
再返回 yaw_diff = -psi。只使用水平分量，并过滤低速样本。

直接调用：

    from pylib.yaw_estimation import estimate, estimate_from_series

命令行（两个 CSV，分别含机体系速度和 GPS 水平速度）：

    python -m pylib.yaw_estimation body.csv gps.csv

或一个合并 CSV：

    python -m pylib.yaw_estimation combined.csv

时间列单位为毫秒。表头别名见 TIME_KEYS / BODY_X_KEYS 等常量。
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
import numpy as np

TIME_KEYS=('time','t','itow','iTOW','timestamp')
BODY_X_KEYS=('vel_x','velocity_x','vx','x')
BODY_Y_KEYS=('vel_y','velocity_y','vy','y')
GPS_N_KEYS=('vel_n','veln','vn','n')
GPS_E_KEYS=('vel_e','vele','ve','e')

DEFAULT_MIN_SPEED=0.5
DEFAULT_MIN_SAMPLES=20
DEFAULT_TOLERANCE=200.0


def estimate(body_x, body_y, gps_n, gps_e, min_speed=DEFAULT_MIN_SPEED, min_samples=DEFAULT_MIN_SAMPLES):
    """成对输入机体系与 NED 水平速度，返回固定 yaw_diff。

    返回 dict：
        yaw_diff_deg  固定安装偏角（drone_fusion.yaw - gps.yaw），单位度
        rotation_deg  实际旋转角 -yaw_diff_deg
        rms           旋转后水平速度残差均方根，m/s
        samples       参与估计的运动样本数
    """
    bx,by,gn,ge=(np.asarray(x,dtype=float).ravel() for x in (body_x,body_y,gps_n,gps_e))
    if not len(bx)==len(by)==len(gn)==len(ge):
        raise ValueError('四组速度序列长度必须一致')
    keep=(np.isfinite(bx)&np.isfinite(by)&np.isfinite(gn)&np.isfinite(ge)
          &(np.hypot(bx,by)>=min_speed)&(np.hypot(gn,ge)>=min_speed))
    bx,by,gn,ge=bx[keep],by[keep],gn[keep],ge[keep]
    if len(bx)<min_samples:
        raise ValueError(f'有效运动样本不足：{len(bx)} < {min_samples}')
    dot=float(np.sum(bx*gn+by*ge))
    cross=float(np.sum(bx*ge-by*gn))
    if abs(dot)<1e-12 and abs(cross)<1e-12:
        raise ValueError('水平速度模长过小，无法确定旋转角')
    psi=math.atan2(cross,dot)
    c,s=math.cos(psi),math.sin(psi)
    rotated_n=c*bx-s*by
    rotated_e=s*bx+c*by
    rms=float(np.sqrt(np.mean((rotated_n-gn)**2+(rotated_e-ge)**2)))
    yaw_diff=-math.degrees(psi)
    return {'yaw_diff_deg':yaw_diff, 'rotation_deg':-yaw_diff,
            'rms':rms, 'samples':int(len(bx))}


def estimate_from_series(body_times, body_x, body_y, gps_times, gps_n, gps_e,
                         tolerance=DEFAULT_TOLERANCE, min_speed=DEFAULT_MIN_SPEED,
                         min_samples=DEFAULT_MIN_SAMPLES):
    """按时间对齐两组水平速度序列后估计 yaw_diff。

    body_times / gps_times 单位必须一致（通常为 ms）。GPS 速度在机体系时刻
    线性插值，只在最近 GPS 样本不超过 tolerance、且插值两侧间隔不超过
    2*tolerance 时使用，避免跨越较大空洞。
    """
    bt=np.asarray(body_times,dtype=float).ravel()
    bx=np.asarray(body_x,dtype=float).ravel()
    by=np.asarray(body_y,dtype=float).ravel()
    gt=np.asarray(gps_times,dtype=float).ravel()
    gn=np.asarray(gps_n,dtype=float).ravel()
    ge=np.asarray(gps_e,dtype=float).ravel()
    if not len(bt)==len(bx)==len(by):
        raise ValueError('机体系时间与速度序列长度必须一致')
    if not len(gt)==len(gn)==len(ge):
        raise ValueError('GPS 时间与速度序列长度必须一致')
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('对齐容差必须是非负有限值')
    order=np.argsort(gt)
    gt,gn,ge=gt[order],gn[order],ge[order]
    good=np.isfinite(gt)&np.isfinite(gn)&np.isfinite(ge)
    gt,gn,ge=gt[good],gn[good],ge[good]
    if len(gt)<2: raise ValueError('GPS 水平速度有效样本不足')
    body_ok=np.isfinite(bt)&np.isfinite(bx)&np.isfinite(by)
    bt,bx,by=bt[body_ok],bx[body_ok],by[body_ok]
    if not len(bt): raise ValueError('机体系水平速度没有有效样本')
    j=np.searchsorted(gt,bt)
    lo=np.clip(j-1,0,len(gt)-1)
    hi=np.clip(j,0,len(gt)-1)
    nearest=np.minimum(np.abs(gt[lo]-bt),np.abs(gt[hi]-bt))
    ok=((bt>=gt[0])&(bt<=gt[-1])&(nearest<=tolerance)
        &((gt[hi]-gt[lo])<=2*tolerance))
    if not ok.any(): raise ValueError('机体系与 GPS 时间没有满足容差的重叠样本')
    times=bt[ok]
    qn=np.interp(times,gt,gn)
    qe=np.interp(times,gt,ge)
    result=estimate(bx[ok],by[ok],qn,qe,min_speed=min_speed,min_samples=min_samples)
    result['matched']=int(len(times))
    return result


def _pick(fieldnames, keys, label, path):
    for key in keys:
        if key in fieldnames: return key
    raise ValueError(f'{path} 缺少{label}列（可用别名：{", ".join(keys)}）')


def _read_rows(path):
    csv.field_size_limit(10_000_000)
    with open(path,encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames: raise ValueError(f'{path} 没有表头')
        return reader.fieldnames,list(reader)


def _body_series(path):
    fields,rows=_read_rows(path)
    tk=_pick(fields,TIME_KEYS,'时间',path)
    xk=_pick(fields,BODY_X_KEYS,'机体系 vel_x',path)
    yk=_pick(fields,BODY_Y_KEYS,'机体系 vel_y',path)
    times=[];xs=[];ys=[]
    for row in rows:
        try:
            values=[float(row[k]) for k in (tk,xk,yk)]
        except (TypeError,ValueError): continue
        if not np.isfinite(values).all(): continue
        times.append(values[0]);xs.append(values[1]);ys.append(values[2])
    if not times: raise ValueError(f'{path} 没有有效数据行')
    return times,xs,ys


def _gps_series(path):
    fields,rows=_read_rows(path)
    tk=_pick(fields,TIME_KEYS,'时间',path)
    nk=_pick(fields,GPS_N_KEYS,'GPS vel_n',path)
    ek=_pick(fields,GPS_E_KEYS,'GPS vel_e',path)
    times=[];ns=[];es=[]
    for row in rows:
        try:
            values=[float(row[k]) for k in (tk,nk,ek)]
        except (TypeError,ValueError): continue
        if not np.isfinite(values).all(): continue
        times.append(values[0]);ns.append(values[1]);es.append(values[2])
    if not times: raise ValueError(f'{path} 没有有效数据行')
    return times,ns,es


def _combined_series(path):
    fields,rows=_read_rows(path)
    tk=_pick(fields,TIME_KEYS,'时间',path)
    xk=_pick(fields,BODY_X_KEYS,'机体系 vel_x',path)
    yk=_pick(fields,BODY_Y_KEYS,'机体系 vel_y',path)
    nk=_pick(fields,GPS_N_KEYS,'GPS vel_n',path)
    ek=_pick(fields,GPS_E_KEYS,'GPS vel_e',path)
    times=[];xs=[];ys=[];ns=[];es=[]
    for row in rows:
        try:
            values=[float(row[k]) for k in (tk,xk,yk,nk,ek)]
        except (TypeError,ValueError): continue
        if not np.isfinite(values).all(): continue
        times.append(values[0]);xs.append(values[1]);ys.append(values[2]);ns.append(values[3]);es.append(values[4])
    if not times: raise ValueError(f'{path} 没有有效数据行')
    return times,xs,ys,ns,es


def main(argv=None):
    parser=argparse.ArgumentParser(description='由机体系与 GPS 水平速度序列估计固定 yaw_diff')
    parser.add_argument('files',nargs='+',help='body.csv gps.csv，或一个包含全部列的 combined.csv')
    parser.add_argument('--tolerance',type=float,default=DEFAULT_TOLERANCE,help='时间对齐容差(ms)，默认 200')
    parser.add_argument('--min-speed',type=float,default=DEFAULT_MIN_SPEED,help='参与估计的最小水平速度(m/s)，默认 0.5')
    parser.add_argument('--min-samples',type=int,default=DEFAULT_MIN_SAMPLES,help='最少运动样本数，默认 20')
    args=parser.parse_args(argv)
    try:
        if len(args.files)==1:
            times,xs,ys,ns,es=_combined_series(args.files[0])
            result=estimate_from_series(times,xs,ys,times,ns,es,tolerance=args.tolerance,
                                        min_speed=args.min_speed,min_samples=args.min_samples)
        elif len(args.files)==2:
            body_times,bxs,bys=_body_series(args.files[0])
            gps_times,gns,ges=_gps_series(args.files[1])
            result=estimate_from_series(body_times,bxs,bys,gps_times,gns,ges,tolerance=args.tolerance,
                                        min_speed=args.min_speed,min_samples=args.min_samples)
        else:
            raise ValueError('请提供两个 CSV（机体系、GPS）或一个合并 CSV')
    except (OSError,ValueError) as exc:
        print(f'估计失败：{exc}',file=sys.stderr);return 1
    print(f'yaw_diff = {result["yaw_diff_deg"]:.6f}°')
    print(f'rotation = {result["rotation_deg"]:.6f}°')
    print(f'RMS      = {result["rms"]:.6f} m/s')
    print(f'samples  = {result["samples"]}')
    if 'matched' in result: print(f'matched  = {result["matched"]}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
