"""GPS iTOW/PVT 对齐与延迟分析 GUI。

依赖: Python 3, matplotlib；读取 ULog 真值还需要 pyulog (pip install pyulog)。
运行: python gps_delay_analyzer.py
"""
from __future__ import annotations
import os, re, struct, threading, csv, json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

FIELDS = ("lat", "lon", "alt", "vel_n", "vel_e", "vel_d", "speed", "heading", "hacc", "vacc", "sacc", "num_sv")
DELAY_CHANNELS = (('vel_n','速度 N'),('vel_e','速度 E'),('vel_d','速度 D'),('speed','水平速度'),
                  ('pos_n','位置 N'),('pos_e','位置 E'),('pos_d','位置 D'))
# 两路 GPS 互比不输出水平地速通道；该通道只用于 GPS 相对 ULog 真值的比较。
TWO_GPS_DELAY_CHANNELS = tuple(channel for channel in DELAY_CHANNELS if channel[0] != 'speed')

def _add(d, tow, **kw):
    if tow is not None and tow >= 0:
        d.setdefault(int(tow), {}).update(kw)

def parse_optional_float(text, name):
    """留空返回 None；否则必须是有限数值。"""
    text=str(text).strip()
    if not text: return None
    try: value=float(text)
    except ValueError: raise ValueError(f'{name}必须是数字，或留空使用逐时刻自动旋转') from None
    if not np.isfinite(value): raise ValueError(f'{name}必须是有限值')
    return value

def rotation_description(reference, yaw_diff, estimate):
    """描述本次真值使用的 yaw_diff 来源。"""
    if yaw_diff is not None:
        return f'固定 yaw_diff={yaw_diff:g}°（实际旋转角 {-yaw_diff:g}°）'
    if estimate:
        try: value=next(iter(reference.values()))['yaw_diff_deg']
        except (StopIteration,KeyError,TypeError): return '速度序列估计'
        return f'速度序列估计 yaw_diff={value:.4g}°（实际旋转角 {-value:.4g}°）'
    return '逐时刻自动（drone_fusion.yaw − fusion_gps.yaw）'

def parse_ubx(path, exact_fix=None):
    data = open(path, "rb").read(); out = {}
    i = 0
    while i + 8 <= len(data):
        j = data.find(b"\xb5\x62", i)
        if j < 0 or j + 8 > len(data): break
        cls, mid, n = data[j+2], data[j+3], struct.unpack_from("<H", data, j+4)[0]
        end = j + 8 + n
        if end > len(data):
            i=j+2
            continue
        p = data[j+6:j+6+n]
        ca=cb=0
        for byte in data[j+2:j+6+n]:
            ca=(ca+byte)&255; cb=(cb+ca)&255
        if data[j+6+n:j+8+n] != bytes((ca,cb)):
            i=j+2
            continue
        if cls == 1 and mid == 7 and n >= 92: # NAV-PVT
            tow = struct.unpack_from("<I", p, 0)[0]
            lon, lat = struct.unpack_from("<ii", p, 24)
            hmsl = struct.unpack_from("<i", p, 36)[0]
            hacc, vacc = struct.unpack_from("<II", p, 40)
            vel_n, vel_e, vel_d, speed, head = struct.unpack_from("<iiiii", p, 48)
            # sAcc：速度精度估计（mm/s），位于 NAV-PVT 偏移 68。
            sacc = struct.unpack_from("<I", p, 68)[0]
            num_sv = p[23]
            if exact_fix is not None and p[20]!=exact_fix:
                i=end
                continue
            # 排除尚无有效定位的启动数据；保留原始 GPS 周内毫秒。
            if not (0 < tow < 604800000 and p[20]>=3 and p[21]&1
                    and -1800000000<=lon<=1800000000 and -900000000<=lat<=900000000):
                i=end
                continue
            _add(out, tow, lat=lat/1e7, lon=lon/1e7, alt=hmsl/1000, vel_n=vel_n/1000,
                 vel_e=vel_e/1000, vel_d=vel_d/1000, speed=speed/1000, heading=head/1e5,
                 hacc=hacc/1000, vacc=vacc/1000, sacc=sacc/1000, num_sv=num_sv)
        i = end
    return out

def parse_text(path):
    out = {}; raw = open(path, "rb").read()
    for m in re.finditer(rb"\$(G[PN]RMC|G[PN]GGA),([^*\r\n]*)", raw):
        typ, s = m.group(1).decode(), m.group(2).decode(errors="ignore"); a = s.split(',')
        try:
            if typ.endswith("RMC") and len(a) >= 9 and a[1] and a[2] == 'A':
                tow = int(float(a[1])*1000); lat = float(a[2][:2])+float(a[2][2:])/60; lon = float(a[4][:3])+float(a[4][3:])/60
                if a[3]=='S': lat=-lat
                if a[5]=='W': lon=-lon
                _add(out,tow,lat=lat,lon=lon,speed=float(a[6])*0.514444,heading=float(a[7] or 0))
            elif typ.endswith("GGA") and len(a) >= 9 and a[0]:
                tow=int(float(a[0])*1000); lat=float(a[1][:2])+float(a[1][2:])/60; lon=float(a[3][:3])+float(a[3][3:])/60
                if a[2]=='S': lat=-lat
                if a[4]=='W': lon=-lon
                _add(out,tow,lat=lat,lon=lon,alt=float(a[8] or 0),num_sv=int(a[6] or 0))
        except (ValueError, IndexError): pass
    return out

def parse_gps(path):
    # .log 也可能是混合 UBX 二进制日志，不能仅按扩展名判断。
    records=parse_ubx(path)
    if not records:
        raise RuntimeError(f'未提取到校验通过且定位有效的 UBX NAV-PVT：{path}')
    export_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_output')
    os.makedirs(export_dir,exist_ok=True)
    # 接收机内部可能以 50 Hz 重复输出同一 PVT；原始导航解算频率按 10 Hz
    # 统计，因此每 100 ms 时间格只保留一条最接近格点的记录。
    records=decimate_pvt_10hz(records)
    export_path=os.path.join(export_dir,os.path.basename(path)+'.pvt.csv')
    with open(export_path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=('itow_ms','_source_itow_ms')+FIELDS)
        writer.writeheader()
        for t,row in sorted(records.items()): writer.writerow(dict(itow_ms=t,**row))
    return records

def decimate_pvt_10hz(records):
    if not records: return records
    grouped={}
    for tow,row in records.items():
        bucket=int(round(tow/100.0))*100
        old=grouped.get(bucket)
        if old is None or abs(tow-bucket)<abs(old[0]-bucket): grouped[bucket]=(tow,row)
    out={int(bucket):dict(item[1],_source_itow_ms=int(item[0])) for bucket,item in grouped.items()}
    return dict(sorted(out.items()))

def parse_ulog(path):
    try:
        from pyulog import ULog
    except ImportError as e: raise RuntimeError('读取 ULog 需要安装 pyulog：pip install pyulog') from e
    u=ULog(path); topics={x.name:x for x in u.data_list}; out={}
    # drone_fusion 是飞控融合输出：position/velocity 为机体系，yaw 为相对真北航向。
    # 先将机体系 x/y 旋转到 N/E，z 保持 Down 方向。
    for name in ('drone_fusion','vehicle_gps_position','gps_position','vehicle_global_position'):
        if name not in topics: continue
        d=topics[name].data; keys=d.keys(); tk=next((k for k in ('time','timestamp','timestamp_sample') if k in keys),None)
        if not tk: continue
        for i,t in enumerate(d[tk]):
            # 若主题带 iTOW/itow，直接使用；否则用时间戳毫秒作为统一时间轴。
            if 'iTOW' in keys: tow = d['iTOW'][i]
            elif 'itow' in keys: tow = d['itow'][i]
            else: tow = d.get('time_utc_usec', d.get('timestamp', d[tk]))[i] / 1000.0
            kw={}
            if name == 'drone_fusion':
                def get(*ks):
                    k0=next((x for x in ks if x in keys),None)
                    return float(d[k0][i]) if k0 else None
                px,py,pz=get('position_x','pos_x'),get('position_y','pos_y'),get('position_z','pos_z')
                vx,vy,vz=get('velocity_x','vel_x'),get('velocity_y','vel_y'),get('velocity_z','vel_z')
                yaw=get('yaw','heading')
                if yaw is not None:
                    # 常见日志 yaw 为度；弧度值通常绝对值不超过 2*pi。
                    import math
                    yr = yaw if abs(yaw) <= 2*math.pi+0.1 else math.radians(yaw)
                    c,s=math.cos(yr),math.sin(yr)
                    if px is not None: kw['pos_n'],kw['pos_e'],kw['pos_d'] = c*px-s*py, s*px+c*py, pz if pz is not None else 0
                    if vx is not None: kw['vel_n'],kw['vel_e'],kw['vel_d'] = c*vx-s*vy, s*vx+c*vy, vz if vz is not None else 0
                    kw['heading']=math.degrees(yr)%360
                if not kw: continue
                _add(out,tow,**kw)
                continue
            for k, aliases, scale in [('lat',('lat','latitude'),1e-7),('lon',('lon','longitude'),1e-7),('alt',('alt','altitude_msl'),1e-3),('vel_n',('vel_n','vel_n_m_s'),1),('vel_e',('vel_e','vel_e_m_s'),1),('vel_d',('vel_d','vel_d_m_s'),1),('hacc',('eph','hacc'),1e-2),('vacc',('epv','vacc'),1e-2),('num_sv',('satellites_used','num_sv'),1)]:
                k0=next((x for x in aliases if x in keys),None)
                if k0: kw[k]=float(d[k0][i])*scale
            _add(out, tow, **kw)
        if out: return out
    raise RuntimeError('ULog 中未找到 vehicle_gps_position/gps_position 主题')

def align(ref, series, tolerance=200):
    rk=np.array(sorted(ref)); sk=np.array(sorted(series)); rows=[]
    for t in rk:
        j=np.searchsorted(sk,t); cand=[x for x in (j-1,j) if 0<=x<len(sk)]
        if not cand: continue
        k=min(cand,key=lambda q:abs(sk[q]-t))
        if abs(sk[k]-t)<=tolerance:
            row=dict(series[int(sk[k])])
            row['_source_itow_ms']=int(sk[k])
            rows.append((t,row))
    return rows

def show_comparison(parent, reference, ref, selected, aligned, labels=None, fields=None, window_title=None, interval_callback=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    window=tk.Toplevel(parent)
    window.title(window_title or 'GPS PVT 对比 — 分页查看')
    window.geometry('1200x800')
    toolbar_text='滚轮：以鼠标位置缩放｜左键：放置垂直光标｜←/→：移动光标｜+/-：缩放｜Home：恢复｜工具栏：平移、框选、保存'
    ttk.Label(window,text=toolbar_text).pack(fill='x',padx=10,pady=4)
    window._interval_states=[]
    if interval_callback is not None:
        ttk.Button(window,text='按 A-B 光标区间重算指标',command=lambda:interval_callback(window._interval_states[notebook.index(notebook.select())] if window._interval_states else None)).pack(anchor='e',padx=10,pady=(0,4))
    notebook=ttk.Notebook(window); notebook.pack(fill='both',expand=True)
    fields=fields or [('lat','纬度','°'),('lon','经度','°'),('alt','海拔高度','m'),
            ('vel_n','北向速度','m/s'),('vel_e','东向速度','m/s'),('vel_d','地向速度','m/s'),
            ('speed','地速','m/s'),('heading','航向角','°'),('hacc','水平精度','m'),
            ('vacc','垂直精度','m'),('num_sv','卫星数','颗')]
    origin=min(reference)
    for field,title,unit in fields:
        page=ttk.Frame(notebook); notebook.add(page,text=title)
        fig=Figure(figsize=(10,6)); ax=fig.add_subplot(111)
        canvas=FigureCanvasTkAgg(fig,master=page)
        toolbar=NavigationToolbar2Tk(canvas,page,pack_toolbar=False); toolbar.pack(side='bottom',fill='x')
        readout=tk.StringVar(value='点击曲线区域放置垂直对比光标')
        ttk.Label(page,textvariable=readout,anchor='w',justify='left').pack(side='bottom',fill='x',padx=8,pady=6)
        canvas.get_tk_widget().pack(fill='both',expand=True)
        curves=[]
        source_times=[]
        if ref is not None:
            times=np.array(sorted(ref)); curves.append(('ULog 真值',(times-origin)/1000,np.array([ref[t].get(field,np.nan) for t in times])))
            source_times.append(times)
        for (number,path),rows in zip(selected,aligned):
            curves.append(((labels or {}).get(number) or os.path.basename(path),
                           np.array([(t-origin)/1000 for t,_ in rows]),np.array([row.get(field,np.nan) for _,row in rows])))
            source_times.append(np.array([row.get('_source_itow_ms',t) for t,row in rows]))
        plot_lines=[]
        for name,x,y in curves:
            line,=ax.plot(x,y,label=name,linewidth=1.1,marker='o',markersize=5,
                          markeredgecolor='white',markeredgewidth=0.6)
            plot_lines.append(line)
        def adapt_markers(event=None,ax=ax,fig=fig,canvas=canvas,curves=curves,plot_lines=plot_lines):
            lo,hi=sorted(ax.get_xlim())
            for line,(_,x,y) in zip(plot_lines,curves):
                left=int(np.searchsorted(x,lo)); right=int(np.searchsorted(x,hi,side='right'))
                count=right-left
                spacing=ax.bbox.width/max(count,1) # 可见采样点的平均像素间距
                size=float(np.clip(spacing*.55*72/fig.dpi,2.2,6))
                line.set_markersize(size)
                line.set_markeredgewidth(.6 if spacing>=7 else 0)
                # 每个真实采样点始终显示，只调整尺寸，避免缩放后标记被跳过。
                line.set_markevery(None)
            canvas.draw_idle()
        ax.callbacks.connect('xlim_changed',adapt_markers)
        canvas.mpl_connect('resize_event',adapt_markers)
        ax.set(title=f'{title}对比',xlabel='相对时间 (s)',ylabel=f'{title} ({unit})'); ax.grid(True,alpha=.35); ax.legend()
        fig.tight_layout(); adapt_markers(); canvas.draw()
        initial=(ax.get_xlim(),ax.get_ylim())
        cursors=[ax.axvline(0,color=color,ls='--',lw=1,visible=False) for color in ('purple','brown')]
        from matplotlib.patches import Rectangle
        box=Rectangle((0,0),0,0,fill=False,edgecolor='gray',linestyle='--',visible=False)
        ax.add_patch(box)
        state={'times':[None,None],'active':0,'annotations':[],'drag':None}
        window._interval_states.append(state)
        def place(t,slot=None,ax=ax,canvas=canvas,cursors=cursors,state=state,curves=curves,readout=readout,unit=unit,source_times=source_times,has_ref=ref is not None):
            if slot is None: slot=state['active']
            state['times'][slot]=t
            for annotation in state['annotations']: annotation.remove()
            state['annotations'].clear()
            for k,time in enumerate(state['times']):
                cursors[k].set_visible(time is not None)
                if time is not None: cursors[k].set_xdata([time,time])
            ta,tb=state['times']; lines=[]
            if ta is not None and tb is not None: lines.append(f'A 相对时间={ta:.6f} s    B 相对时间={tb:.6f} s    区间长度={abs(tb-ta):.6f} s')
            else: lines.append('右键依次放置 A、B 光标；左键按住光标线可拖动；按 A/B 选择光标')
            for row,(name,x,y) in enumerate(curves):
                vals=[]
                for k,time in enumerate(state['times']):
                    value=None
                    if time is not None and len(x) and x[0]<=time<=x[-1]:
                        j=int(np.searchsorted(x,time))
                        if j<len(x) and x[j]==time: value=y[j]
                        elif 0<j<len(x) and np.isfinite(y[j-1:j+1]).all(): value=float(np.interp(time,x[j-1:j+1],y[j-1:j+1]))
                    vals.append(value)
                    if value is not None and np.isfinite(value):
                        label=ax.annotate(f'{"AB"[k]} {name}: {value:.8g}',(time,value),xytext=(6 if k==0 else -6,10+row*14),textcoords='offset points',ha='left' if k==0 else 'right',fontsize=8,bbox=dict(boxstyle='round',fc='white',alpha=.8))
                        state['annotations'].append(label)
                va,vb=vals
                fmt=lambda v: '无数据' if v is None else f'{v:.8g} {unit}'
                line=f'{name}: A={fmt(va)}    B={fmt(vb)}'
                if va is not None and vb is not None: line+=f'    ΔY (B−A)={vb-va:.8g} {unit}'
                lines.append(line)
            # 插值交点没有唯一的原始 iTOW；单独列出最近的对齐样本及其真实采样时间。
            for slot,time in enumerate(state['times']):
                if time is None: continue
                samples=[]
                for (name,x,y),original in zip(curves,source_times):
                    if not len(x) or not x[0]<=time<=x[-1]: continue
                    j=int(np.argmin(np.abs(x-time)))
                    samples.append((name,int(original[j])))
                lines.append(f'{"AB"[slot]} 最近采样点原始时间（ms，非插值交点时间）：'+ '； '.join(f'{name}={stamp}' for name,stamp in samples))
                gps_samples=samples[1:] if has_ref and samples and samples[0][0]=='FFC 真值' else samples
                for j,(name,stamp) in enumerate(gps_samples):
                    for other,other_stamp in gps_samples[j+1:]:
                        lines.append(f'{"AB"[slot]} ΔiTOW [{other} − {name}] = {other_stamp-stamp:+d} ms')
            readout.set('\n'.join(lines)); canvas.draw_idle()
        def zoom(factor,cx=None,cy=None,ax=ax,canvas=canvas):
            a,b=ax.get_xlim(); c,d=ax.get_ylim()
            cx=(a+b)/2 if cx is None else cx; cy=(c+d)/2 if cy is None else cy
            ax.set_xlim(cx+(a-cx)*factor,cx+(b-cx)*factor)
            ax.set_ylim(cy+(c-cy)*factor,cy+(d-cy)*factor); canvas.draw_idle()
        def click(event,ax=ax,canvas=canvas,toolbar=toolbar,place=place,state=state,box=box):
            canvas.get_tk_widget().focus_set()
            if event.inaxes!=ax or toolbar.mode: return
            if event.button==3:
                place(event.xdata); state['active']=1-state['active']
            elif event.button in (1,2):
                if event.button==1:
                    # 使用屏幕像素判定命中，缩放后仍保持一致的选择宽度。
                    candidates=[(abs(ax.transData.transform((t,0))[0]-event.x),k)
                                for k,t in enumerate(state['times']) if t is not None]
                    if candidates and min(candidates)[0]<=9:
                        state['active']=min(candidates)[1]
                        state['drag']=('cursor',)
                        place(event.xdata)
                        return
                state['drag']=(event.button,event.x,event.y,event.xdata,event.ydata,ax.get_xlim(),ax.get_ylim())
                if event.button==1:
                    box.set_xy((event.xdata,event.ydata)); box.set_width(0); box.set_height(0); box.set_visible(True)
        def motion(event,ax=ax,canvas=canvas,state=state,box=box,place=place):
            drag=state['drag']
            if drag is None: return
            if drag[0]=='cursor':
                if event.inaxes==ax: place(event.xdata)
                return
            button,px,py,x,y,xlim,ylim=drag
            if button==2:
                dx=(event.x-px)/ax.bbox.width*(xlim[1]-xlim[0]); dy=(event.y-py)/ax.bbox.height*(ylim[1]-ylim[0])
                ax.set_xlim(xlim[0]-dx,xlim[1]-dx); ax.set_ylim(ylim[0]-dy,ylim[1]-dy)
            elif event.inaxes==ax:
                box.set_width(event.xdata-x); box.set_height(event.ydata-y)
            canvas.draw_idle()
        def release(event,ax=ax,canvas=canvas,state=state,box=box):
            drag=state['drag']; state['drag']=None; box.set_visible(False)
            if drag and drag[0]==1 and event.inaxes==ax and abs(event.x-drag[1])>5 and abs(event.y-drag[2])>5:
                ax.set_xlim(sorted((drag[3],event.xdata))); ax.set_ylim(sorted((drag[4],event.ydata)))
            canvas.draw_idle()
        def scroll(event,ax=ax,zoom=zoom):
            if event.inaxes==ax: zoom(.8 if event.button=='up' else 1.25,event.xdata,event.ydata)
        def key(event,ax=ax,canvas=canvas,initial=initial,state=state,place=place,zoom=zoom):
            if event.key in ('+','=','-'): zoom(1.25 if event.key=='-' else .8)
            elif event.key in ('home',' ','space'):
                ax.set_xlim(initial[0]); ax.set_ylim(initial[1]); canvas.draw_idle()
            elif event.key in ('a','b'): state['active']=0 if event.key=='a' else 1
            elif event.key in ('left','right'):
                a,b=ax.get_xlim(); t=state['times'][state['active']]
                place(((a+b)/2 if t is None else t)+(b-a)/100*(1 if event.key=='right' else -1))
        for event,handler in [('button_press_event',click),('motion_notify_event',motion),('button_release_event',release),('scroll_event',scroll),('key_press_event',key)]: canvas.mpl_connect(event,handler)
        page.plot_objects=(fig,canvas,toolbar,click,motion,release,scroll,key,adapt_markers)

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title('GPS 延迟分析（iTOW/PVT）'); self.geometry('1450x850'); self.minsize(1100,700); self.items=[]
        self.config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'.gps_delay_history.json')
        left=ttk.Frame(self); left.pack(side='left',fill='y',padx=8,pady=8); right=ttk.Frame(self); right.pack(side='right',fill='both',expand=True,padx=8,pady=8)
        ttk.Label(left,text='分析项目').pack(anchor='w'); self.list=tk.Listbox(left,width=24,height=12); self.list.pack(fill='y',expand=True); self.list.insert(0,'GPS 延迟分析（iTOW/PVT）'); self.list.selection_set(0)
        ttk.Label(right,text='VIO 真值 ffc.csv').grid(row=0,column=0,sticky='w'); self.ref=tk.StringVar(); ttk.Entry(right,textvariable=self.ref,width=75).grid(row=0,column=1); ttk.Button(right,text='选择',command=lambda:self.pick(self.ref,'ulog')).grid(row=0,column=2)
        self.gps=[]; self.gps_labels=[]
        for i in range(3):
            v=tk.StringVar(); self.gps.append(v); ttk.Label(right,text=f'GPS {i+1} 原始文件').grid(row=i+1,column=0,sticky='w'); ttk.Entry(right,textvariable=v,width=75).grid(row=i+1,column=1); ttk.Button(right,text='选择',command=lambda x=v:self.pick(x)).grid(row=i+1,column=2)
            name=tk.StringVar(); self.gps_labels.append(name)
            ttk.Label(right,text='图例名称').grid(row=i+1,column=3,padx=(8,2))
            ttk.Entry(right,textvariable=name,width=18).grid(row=i+1,column=4,sticky='ew')
        ttk.Label(right,text='对齐容差(ms)').grid(row=4,column=0,sticky='w'); self.tol=tk.StringVar(value='200'); ttk.Entry(right,textvariable=self.tol,width=10).grid(row=4,column=1,sticky='w')
        self.yaw_diff=tk.StringVar();self.estimate_yaw=tk.BooleanVar(value=True)
        ttk.Checkbutton(right,text='估计 yaw_diff',variable=self.estimate_yaw).grid(row=4,column=2,sticky='w',padx=(8,2))
        ttk.Label(right,text='固定 yaw_diff(°)').grid(row=4,column=3,sticky='w');ttk.Entry(right,textvariable=self.yaw_diff,width=10).grid(row=4,column=4,sticky='ew')
        ttk.Label(right,text='固定值优先；否则勾选=速度序列估计，取消=逐时刻自动').grid(row=5,column=3,columnspan=2,sticky='w')
        self.log=tk.Text(right,height=18); self.log.grid(row=6,column=0,columnspan=3,sticky='nsew'); right.rowconfigure(6,weight=1); right.columnconfigure(1,weight=1); ttk.Button(right,text='开始分析并绘图',command=self.run).grid(row=5,column=1,sticky='e',pady=8)
        self.list.insert('end','RTCM 星历检测')
        scrollbar=ttk.Scrollbar(left,orient='vertical',command=self.list.yview)
        scrollbar.pack(side='right',fill='y'); self.list.configure(yscrollcommand=scrollbar.set)
        self.project_area=right
        self.eph_area=ttk.Frame(self)
        ttk.Label(self.eph_area,text='RTCM 星历检测文件').grid(row=0,column=0,padx=8,pady=12)
        self.eph_path=tk.StringVar()
        ttk.Entry(self.eph_area,textvariable=self.eph_path,width=70).grid(row=0,column=1,sticky='ew')
        ttk.Button(self.eph_area,text='选择',command=lambda:self.pick(self.eph_path)).grid(row=0,column=2,padx=8)
        ttk.Button(self.eph_area,text='解析并绘制星历缺失比例',command=self.run_ephemeris).grid(row=1,column=1,pady=12)
        ttk.Label(self.eph_area,text='支持含 RTCM3 二进制的日志；按 MSM 时间每 10 秒统计。\n左轴：星历接收缺失比例；右轴：该时间窗观测到的卫星数。\n缺失指本文件中尚未收到该卫星的星历，不代表接收机实际无星历。').grid(row=2,column=0,columnspan=3,sticky='w',padx=8)
        self.eph_area.columnconfigure(1,weight=1)
        self.list.insert('end','C/N0 载噪比对比')
        self.cno_area=ttk.Frame(self)
        self.cno_gps=[]; self.cno_labels=[]
        for i in range(3):
            v=tk.StringVar(); self.cno_gps.append(v)
            ttk.Label(self.cno_area,text=f'GPS {i+1} 原始文件').grid(row=i,column=0,sticky='w',padx=8,pady=6)
            ttk.Entry(self.cno_area,textvariable=v,width=75).grid(row=i,column=1,sticky='ew')
            ttk.Button(self.cno_area,text='选择',command=lambda v=v:self.pick(v)).grid(row=i,column=2,padx=8)
            name=tk.StringVar(); self.cno_labels.append(name)
            ttk.Label(self.cno_area,text='图例名称').grid(row=i,column=3,padx=(8,2))
            ttk.Entry(self.cno_area,textvariable=name,width=18).grid(row=i,column=4,sticky='ew')
        ttk.Label(self.cno_area,text='自动解析 NAV-SAT / NAV-SVINFO；同一 iTOW 优先使用 NAV-SAT。\n只统计参与定位卫星数大于 0 的历元，比较参与定位卫星数及有效 C/N0 的算术平均值；不需要真值文件。\n多路时取所有文件公共 iTOW 时间范围，各路按自身历元绘制；不做对齐、插值或外推。').grid(row=4,column=0,columnspan=5,sticky='w',padx=8,pady=8)
        self.cno_button=ttk.Button(self.cno_area,text='开始分析并导出图片',command=self.run_cno)
        self.cno_button.grid(row=5,column=1,sticky='e',pady=8)
        self.cno_log=tk.Text(self.cno_area,height=16)
        self.cno_log.grid(row=6,column=0,columnspan=3,sticky='nsew',padx=8)
        self.cno_area.columnconfigure(1,weight=1); self.cno_area.rowconfigure(6,weight=1)
        ttk.Label(right,text='图例名称留空时使用文件名').grid(row=4,column=3,columnspan=2,sticky='w',padx=8)
        ttk.Label(self.cno_area,text='图例名称留空时使用文件名').grid(row=3,column=3,columnspan=2,sticky='w',padx=8)
        self.log.grid_configure(columnspan=5)
        self.cno_log.grid_configure(columnspan=5)
        self.list.insert('end','精度评估（动态指标）')
        self.accuracy_area=ttk.Frame(self)
        self.accuracy_ref=tk.StringVar();self.accuracy_gps=[];self.accuracy_tol=tk.StringVar(value='200');self.accuracy_yaw_diff=tk.StringVar();self.accuracy_estimate_yaw=tk.BooleanVar(value=True);self.accuracy_compensate=tk.BooleanVar(value=True)
        ttk.Label(self.accuracy_area,text='ULog 真值文件').grid(row=0,column=0,sticky='w',padx=8)
        ttk.Entry(self.accuracy_area,textvariable=self.accuracy_ref,width=75).grid(row=0,column=1,sticky='ew')
        ttk.Button(self.accuracy_area,text='选择',command=lambda:self.pick(self.accuracy_ref,'ulog')).grid(row=0,column=2)
        for i in range(3):
            v=tk.StringVar();self.accuracy_gps.append(v)
            ttk.Label(self.accuracy_area,text=f'GPS {i+1} 原始文件').grid(row=i+1,column=0,sticky='w',padx=8,pady=6)
            ttk.Entry(self.accuracy_area,textvariable=v,width=75).grid(row=i+1,column=1,sticky='ew')
            ttk.Button(self.accuracy_area,text='选择',command=lambda v=v:self.pick(v)).grid(row=i+1,column=2)
        ttk.Label(self.accuracy_area,text='对齐容差(ms)').grid(row=4,column=0,sticky='w',padx=8)
        ttk.Entry(self.accuracy_area,textvariable=self.accuracy_tol,width=10).grid(row=4,column=1,sticky='w')
        ttk.Label(self.accuracy_area,text='固定 yaw_diff(°)').grid(row=4,column=2,sticky='w',padx=(8,2))
        ttk.Entry(self.accuracy_area,textvariable=self.accuracy_yaw_diff,width=10).grid(row=4,column=3,sticky='w')
        ttk.Checkbutton(self.accuracy_area,text='估计 yaw_diff',variable=self.accuracy_estimate_yaw).grid(row=4,column=4,sticky='w',padx=(8,2))
        ttk.Checkbutton(self.accuracy_area,text='时延补偿(NE)',variable=self.accuracy_compensate).grid(row=4,column=5,sticky='w',padx=(8,2))
        accuracy_help=ttk.Label(self.accuracy_area,text='GPS1 必选，用于建立 ULog 时间映射。按 GPS 独立历元统计。\n默认先逐路估计 GPS 相对真值的 NE 时延并补偿时间戳，再统计精度；取消“时延补偿(NE)”则保留原始时延影响。\n轨迹去除首个共同有效样本的位置偏置，再逐轴扣除平均偏差；结果是相对轨迹误差，不是绝对定位精度。\n固定 yaw_diff 有值时全时段使用该值；留空且勾选“估计 yaw_diff”时由 GPS1 与 drone_fusion 水平速度序列估计固定值；两者都不用时逐时刻自动计算 drone_fusion.yaw − fusion_gps.yaw。实际旋转角均为 −yaw_diff。\n按指定的 12 项位置、速度 RMSE / CEP / STD 指标对比，同时绘制补偿后的速度和位置曲线。',wraplength=1250,justify='left')
        accuracy_help.grid(row=5,column=0,columnspan=6,sticky='ew',padx=8,pady=8)
        self.accuracy_button=ttk.Button(self.accuracy_area,text='开始精度评估并导出指标',command=self.run_accuracy)
        self.accuracy_button.grid(row=6,column=1,sticky='e')
        accuracy_log_frame=ttk.Frame(self.accuracy_area)
        accuracy_log_frame.grid(row=7,column=0,columnspan=6,sticky='nsew',padx=8,pady=(4,6))
        self.accuracy_log=tk.Text(accuracy_log_frame,height=8,wrap='word',font=('Consolas',10))
        accuracy_log_scroll=ttk.Scrollbar(accuracy_log_frame,orient='vertical',command=self.accuracy_log.yview)
        self.accuracy_log.configure(yscrollcommand=accuracy_log_scroll.set)
        self.accuracy_log.pack(side='left',fill='both',expand=True);accuracy_log_scroll.pack(side='right',fill='y')
        from pylib.accuracy_analysis import COMPARISON_HEADERS
        self.accuracy_table=ttk.Treeview(self.accuracy_area,columns=COMPARISON_HEADERS,show='headings',height=15)
        for key in COMPARISON_HEADERS:
            self.accuracy_table.heading(key,text=key);self.accuracy_table.column(key,width=95,anchor='center',stretch=True)
        self.accuracy_table.tag_configure('better',foreground='#137333',background='#e6f4ea')
        self.accuracy_table.tag_configure('worse',foreground='#b3261e',background='#fce8e6')
        self.accuracy_table.tag_configure('neutral',foreground='#555555')
        self.accuracy_table.grid(row=8,column=0,columnspan=5,sticky='nsew',padx=(8,0))
        scroll=ttk.Scrollbar(self.accuracy_area,orient='vertical',command=self.accuracy_table.yview)
        scroll.grid(row=8,column=5,sticky='ns');self.accuracy_table.configure(yscrollcommand=scroll.set)
        table_hscroll=ttk.Scrollbar(self.accuracy_area,orient='horizontal',command=self.accuracy_table.xview)
        table_hscroll.grid(row=9,column=0,columnspan=5,sticky='ew',padx=(8,0));self.accuracy_table.configure(xscrollcommand=table_hscroll.set)
        self.accuracy_area.columnconfigure(1,weight=1);self.accuracy_area.rowconfigure(8,weight=1);self.accuracy_area.rowconfigure(7,weight=0)
        self.list.insert('end','ULog 脱敏')
        self.sanitize_area=ttk.Frame(self)
        self.sanitize_source=tk.StringVar();self.sanitize_destination=tk.StringVar()
        for row,label,var in ((0,'原始 ULog 文件',self.sanitize_source),(1,'输出 ULog 文件',self.sanitize_destination)):
            ttk.Label(self.sanitize_area,text=label).grid(row=row,column=0,sticky='w',padx=8,pady=8)
            ttk.Entry(self.sanitize_area,textvariable=var,width=80).grid(row=row,column=1,sticky='ew')
            ttk.Button(self.sanitize_area,text='选择',command=lambda row=row:self.pick_sanitize(row)).grid(row=row,column=2,padx=8)
        ttk.Label(self.sanitize_area,text='只保留延迟与精度评估需要的三个主题及字段；不复制参数、设备信息和日志文本。\n注意：GPS1 身份匹配需要 sensor_gps 经纬度，所以新文件仍含位置和相对轨迹数据。\n不改变时间戳、采样率或保留字段的数值；原文件不会被修改。').grid(row=2,column=0,columnspan=3,sticky='w',padx=8,pady=8)
        self.sanitize_button=ttk.Button(self.sanitize_area,text='生成并验证精简 ULog',command=self.run_sanitize)
        self.sanitize_button.grid(row=3,column=1,sticky='e',pady=8)
        self.sanitize_log=tk.Text(self.sanitize_area,height=20)
        self.sanitize_log.grid(row=4,column=0,columnspan=3,sticky='nsew')
        self.sanitize_area.columnconfigure(1,weight=1);self.sanitize_area.rowconfigure(4,weight=1)
        self.list.insert('end','GPS 相对精度对比')
        self.relative_area=ttk.Frame(self);self.relative_paths=[tk.StringVar(),tk.StringVar()]
        self.relative_window=tk.StringVar(value='3');self.relative_tolerance=tk.StringVar(value='60')
        for i,v in enumerate(self.relative_paths):
            ttk.Label(self.relative_area,text=f'GPS {i+1} 原始文件').grid(row=i,column=0,padx=8,pady=8)
            ttk.Entry(self.relative_area,textvariable=v,width=80).grid(row=i,column=1,sticky='ew')
            ttk.Button(self.relative_area,text='选择',command=lambda v=v:self.pick(v)).grid(row=i,column=2)
        options=ttk.Frame(self.relative_area);options.grid(row=2,column=0,columnspan=3,sticky='w',padx=8)
        ttk.Label(options,text='滑窗长度(s)').pack(side='left');ttk.Entry(options,textvariable=self.relative_window,width=8).pack(side='left',padx=8)
        ttk.Label(options,text='一对一匹配容差(ms)').pack(side='left');ttk.Entry(options,textvariable=self.relative_tolerance,width=8).pack(side='left',padx=8)
        ttk.Label(self.relative_area,text='同运动条件下，以居中滑窗二次拟合去除局部运动趋势；默认 3 秒。只使用 fixType=3 有效 PVT。\n绿色表示 GPS2 残差更小，红色表示更大；这是短时去趋势波动指标，不是可分离的绝对测量噪声。\n真实急转、加速度变化、接收机滤波和时延均会影响结果；自动检查多个尺度；只有各尺度同向且差异均超过 5% 才红绿着色，其余灰色。',wraplength=1000).grid(row=3,column=0,columnspan=3,sticky='ew',padx=8,pady=8)
        self.relative_button=ttk.Button(self.relative_area,text='开始相对精度对比',command=self.run_relative)
        self.relative_button.grid(row=4,column=1,sticky='e',pady=8)
        from pylib.relative_accuracy import HEADERS as RELATIVE_HEADERS
        self.relative_table=ttk.Treeview(self.relative_area,columns=RELATIVE_HEADERS,show='headings',height=8)
        for key in RELATIVE_HEADERS:
            self.relative_table.heading(key,text=key);self.relative_table.column(key,width=125,anchor='center')
        for tag,color,bg in (('better','#137333','#e6f4ea'),('worse','#b3261e','#fce8e6'),('neutral','#555555','#ffffff')):
            self.relative_table.tag_configure(tag,foreground=color,background=bg)
        self.relative_table.grid(row=5,column=0,columnspan=3,sticky='nsew')
        scrollbar=ttk.Scrollbar(self.relative_area,orient='horizontal',command=self.relative_table.xview)
        scrollbar.grid(row=6,column=0,columnspan=3,sticky='ew');self.relative_table.configure(xscrollcommand=scrollbar.set)
        self.relative_log=tk.Text(self.relative_area,height=8,wrap='word');self.relative_log.grid(row=7,column=0,columnspan=3,sticky='nsew')
        self.relative_area.columnconfigure(1,weight=1);self.relative_area.rowconfigure(7,weight=1)
        self.list.insert('end','sAcc 一致性评估')
        self.sacc_area=ttk.Frame(self)
        self.sacc_ref=tk.StringVar();self.sacc_gps=[tk.StringVar() for _ in range(3)]
        self.sacc_tol=tk.StringVar(value='200');self.sacc_yaw_diff=tk.StringVar()
        self.sacc_estimate_yaw=tk.BooleanVar(value=True);self.sacc_compensate=tk.BooleanVar(value=True)
        ttk.Label(self.sacc_area,text='ULog 真值文件').grid(row=0,column=0,sticky='w',padx=8)
        ttk.Entry(self.sacc_area,textvariable=self.sacc_ref,width=75).grid(row=0,column=1,sticky='ew')
        ttk.Button(self.sacc_area,text='选择',command=lambda:self.pick(self.sacc_ref,'ulog')).grid(row=0,column=2)
        for i,v in enumerate(self.sacc_gps):
            ttk.Label(self.sacc_area,text=f'GPS {i+1} 原始文件').grid(row=i+1,column=0,sticky='w',padx=8,pady=6)
            ttk.Entry(self.sacc_area,textvariable=v,width=75).grid(row=i+1,column=1,sticky='ew')
            ttk.Button(self.sacc_area,text='选择',command=lambda v=v:self.pick(v)).grid(row=i+1,column=2)
        ttk.Label(self.sacc_area,text='对齐容差(ms)').grid(row=4,column=0,sticky='w',padx=8)
        ttk.Entry(self.sacc_area,textvariable=self.sacc_tol,width=10).grid(row=4,column=1,sticky='w')
        ttk.Label(self.sacc_area,text='固定 yaw_diff(°)').grid(row=4,column=2,sticky='w',padx=(8,2))
        ttk.Entry(self.sacc_area,textvariable=self.sacc_yaw_diff,width=10).grid(row=4,column=3,sticky='w')
        ttk.Checkbutton(self.sacc_area,text='估计 yaw_diff',variable=self.sacc_estimate_yaw).grid(row=4,column=4,sticky='w',padx=(8,2))
        ttk.Checkbutton(self.sacc_area,text='时延补偿(NE)',variable=self.sacc_compensate).grid(row=4,column=5,sticky='w',padx=(8,2))
        ttk.Label(self.sacc_area,text='GPS1 必选，用于建立 ULog 到 iTOW 的时间映射；可再选 GPS2、GPS3。\n'
                  '默认先估计并补偿各路相对真值的 NE 时延，再用真值速度与 GPS 速度计算三维速度残差 r=‖v_gps−v_truth‖。\n'
                  'sAcc 取 UBX NAV-PVT 的速度精度估计，视为一倍速度中误差；样本需 sAcc>0 且残差三分量有限，少于 10 个时指标为 NaN。\n'
                  '秩相关系数：sAcc 与 r 分别取平均秩（并列共享平均秩）后求 Pearson；全部相等导致分母为 0 时返回 NaN。\n'
                  '余弦相似度：Σ(sAcc·r)/sqrt(ΣsAcc²·Σr²)，越接近 1 表示两者越接近同比例变化。\n'
                  '中位数：sAcc 中位数与速度残差中位数仅作描述输出，不参与对比好坏。\n'
                  '占比：r 小于 1/2/3 倍 sAcc 的历元比例；对比时与正态分布 ±1/2/3σ 目标 68.27%/95.45%/99.73% 比较，越接近越好。\n'
                  '输入两路及以上时以第一路 GPS1 为基准；ρ、余弦相似度越大越好。',
                  wraplength=1250,justify='left').grid(row=5,column=0,columnspan=6,sticky='ew',padx=8,pady=8)
        self.sacc_button=ttk.Button(self.sacc_area,text='开始 sAcc 一致性评估',command=self.run_sacc)
        self.sacc_button.grid(row=6,column=1,sticky='e')
        self.sacc_log=tk.Text(self.sacc_area,height=6,wrap='word',font=('Consolas',10))
        self.sacc_log.grid(row=7,column=0,columnspan=6,sticky='nsew',padx=8,pady=(4,6))
        from pylib.sacc_reliability import HEADERS as SACC_HEADERS,COMPARISON_HEADERS as SACC_COMPARISON_HEADERS
        self.sacc_table=ttk.Treeview(self.sacc_area,columns=SACC_HEADERS,show='headings',height=5)
        for key in SACC_HEADERS:
            self.sacc_table.heading(key,text=key);self.sacc_table.column(key,width=120,anchor='center',stretch=True)
        self.sacc_table.grid(row=8,column=0,columnspan=5,sticky='nsew',padx=(8,0))
        table_scroll=ttk.Scrollbar(self.sacc_area,orient='horizontal',command=self.sacc_table.xview)
        table_scroll.grid(row=9,column=0,columnspan=5,sticky='ew',padx=(8,0));self.sacc_table.configure(xscrollcommand=table_scroll.set)
        ttk.Label(self.sacc_area,text='相对 GPS1 对比（ρ/余弦越大越好；覆盖率越接近正态目标越好；绿色为更优）').grid(row=10,column=0,columnspan=6,sticky='w',padx=8,pady=(6,2))
        self.sacc_comparison=ttk.Treeview(self.sacc_area,columns=SACC_COMPARISON_HEADERS,show='headings',height=6)
        for key in SACC_COMPARISON_HEADERS:
            self.sacc_comparison.heading(key,text=key);self.sacc_comparison.column(key,width=110,anchor='center',stretch=True)
        for tag,color,bg in (('better','#137333','#e6f4ea'),('worse','#b3261e','#fce8e6'),('neutral','#555555','#ffffff')):
            self.sacc_comparison.tag_configure(tag,foreground=color,background=bg)
        self.sacc_comparison.grid(row=11,column=0,columnspan=5,sticky='nsew',padx=(8,0))
        comparison_scroll=ttk.Scrollbar(self.sacc_area,orient='horizontal',command=self.sacc_comparison.xview)
        comparison_scroll.grid(row=12,column=0,columnspan=5,sticky='ew',padx=(8,0));self.sacc_comparison.configure(xscrollcommand=comparison_scroll.set)
        self.sacc_area.columnconfigure(1,weight=1);self.sacc_area.rowconfigure(8,weight=1);self.sacc_area.rowconfigure(11,weight=1)
        self.list.bind('<<ListboxSelect>>',self.select_project)
        self.fusion_frame=tk.StringVar(value='local')
        self.north_confirmed=tk.BooleanVar(value=True)
        self.anchor_boot=tk.StringVar(); self.anchor_tow=tk.StringVar()
        self.load_history(); self.protocol('WM_DELETE_WINDOW',self.on_close)
    def select_project(self,event=None):
        selection=self.list.curselection()
        if not selection: return
        for area in (self.project_area,self.eph_area,self.cno_area,self.accuracy_area,self.sanitize_area,self.relative_area,self.sacc_area): area.pack_forget()
        area=(self.project_area,self.eph_area,self.cno_area,self.accuracy_area,self.sanitize_area,self.relative_area,self.sacc_area)[selection[0]]
        area.pack(side='right',fill='both',expand=True,padx=8,pady=8)
    def clear_previous_results(self):
        """新任务开始前释放旧图窗并清空结果；不删除磁盘导出文件。"""
        if any(getattr(self,name,False) for name in ('_running','_cno_running','_accuracy_running','_relative_running','_sanitize_running','_sacc_running')):
            messagebox.showinfo('任务进行中','请等待当前计算完成后再开始新计算。')
            return False
        for child in list(self.winfo_children()):
            if isinstance(child,tk.Toplevel):
                child.destroy()
        for name in ('log','cno_log','accuracy_log','relative_log','sanitize_log','sacc_log'):
            widget=getattr(self,name,None)
            if widget is not None:widget.delete('1.0','end')
        for name in ('accuracy_table','relative_table','sacc_table','sacc_comparison'):
            widget=getattr(self,name,None)
            if widget is not None:widget.delete(*widget.get_children())
        return True

    def run_relative(self, interval=None):
        if getattr(self,'_relative_running',False):return
        if not self.clear_previous_results():return
        try:
            paths=[v.get().strip() for v in self.relative_paths]
            selected=[(i+1,path) for i,path in enumerate(paths) if path]
            if not selected:raise ValueError('请至少选择一路 GPS 文件')
            window=float(self.relative_window.get());tolerance=float(self.relative_tolerance.get())
        except Exception as exc:messagebox.showerror('相对精度对比失败',str(exc));return
        self.save_history();self._relative_running=True;self.relative_button.configure(state='disabled')
        from pylib.relative_accuracy import HEADERS,SINGLE_HEADERS
        headers=SINGLE_HEADERS if len(selected)==1 else HEADERS
        self.relative_table.configure(columns=headers)
        for key in headers:
            self.relative_table.heading(key,text=key);self.relative_table.column(key,width=125,anchor='center')
        self.relative_table.delete(*self.relative_table.get_children())
        self.relative_log.delete('1.0','end');self.relative_log.insert('end','正在后台解析和滑窗拟合……\n')
        def finish(result=None,error=None):
            try:
                if error:raise ValueError(error)
                for values,tag in result[0]:self.relative_table.insert('','end',values=values,tags=(tag,))
                self.relative_log.insert('end',result[1]);self.relative_log.see('end')
                plot=result[2]
                plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
                plt.rcParams['axes.unicode_minus']=False
                def recalc(state):
                    if state is None or None in state['times']:
                        messagebox.showinfo('区间重算','请先在同一曲线页放置 A、B 两个光标。');return
                    lo,hi=sorted(state['times']);self.run_relative(interval=(lo,hi))
                show_comparison(self,plot[0],None,selected,plot[1],
                                labels={n:f'GPS{n}' for n,_ in selected},fields=plot[2],window_title='GPS 独立指标：原始曲线与去趋势残差' if len(selected)==1 else 'GPS 相对精度：原始曲线与去趋势残差',interval_callback=recalc)
            except Exception as exc:messagebox.showerror('相对精度对比失败',str(exc))
            finally:self._relative_running=False;self.relative_button.configure(state='normal')
        def worker():
            try:
                from pylib.relative_accuracy import analyze,analyze_single
                data=[parse_ubx(path,exact_fix=3) for _,path in selected]
                if interval is not None:
                    origin=min(min(d) for d in data)
                    data=[{t:r for t,r in d.items() if interval[0]*1000 <= t-origin <= interval[1]*1000} for d in data]
                result=analyze_single(data[0],window=window) if len(selected)==1 else analyze(*data,window=window,tolerance_ms=tolerance)
                self.after(0,lambda:finish(result=result))
            except Exception as exc:self.after(0,lambda error=str(exc):finish(error=error))
        threading.Thread(target=worker,daemon=True).start()
    def run_sacc(self):
        if getattr(self,'_sacc_running',False):return
        if not self.clear_previous_results():return
        try:
            path=self.sacc_ref.get().strip()
            if not path.lower().endswith(('.ulg','.ulog')):raise ValueError('请选择 ULog 真值文件')
            selected=[(i+1,v.get().strip()) for i,v in enumerate(self.sacc_gps) if v.get().strip()]
            if not selected or selected[0][0]!=1:raise ValueError('必须选择 GPS1 作为 ULog 时间映射锚点')
            tolerance=float(self.sacc_tol.get())
            if not np.isfinite(tolerance) or tolerance<0:raise ValueError('对齐容差必须是非负有限值')
            yaw_diff=parse_optional_float(self.sacc_yaw_diff.get(),'固定 yaw_diff(°)')
            estimate_yaw=bool(self.sacc_estimate_yaw.get())
            compensate=bool(self.sacc_compensate.get())
        except Exception as exc:messagebox.showerror('sAcc 一致性评估失败',str(exc));return
        self.save_history();self._sacc_running=True;self.sacc_button.configure(state='disabled')
        self.sacc_log.delete('1.0','end')
        self.sacc_log.insert('end',f'正在后台解析并计算 sAcc 一致性……\n时延补偿：{"开启（NE）" if compensate else "关闭"}\n')
        for widget in (self.sacc_table,self.sacc_comparison):widget.delete(*widget.get_children())
        def fmt(value,spec):
            return 'NaN' if not np.isfinite(value) else format(float(value),spec)
        def finish(result=None,error=None):
            try:
                if error:raise ValueError(error)
                results,details,paths=result
                for label,info in results:
                    self.sacc_table.insert('','end',values=(label,info['samples'],
                        fmt(info['median_sacc'],'.4f'),fmt(info['median_residual'],'.4f'),
                        fmt(info['rho'],'.5f'),fmt(info['cosine'],'.5f'),fmt(info['coverage'][0],'.2f'),
                        fmt(info['coverage'][1],'.2f'),fmt(info['coverage'][2],'.2f')))
                from pylib.sacc_reliability import compare
                if len(results)>=2:
                    for values,tag in compare(results):self.sacc_comparison.insert('','end',values=values,tags=(tag,))
                    self.sacc_log.insert('end','对比基准：GPS1。ρ、余弦相似度越大越好；覆盖率与正态目标 68.27%/95.45%/99.73% 比较，越接近越好。\n')
                else:
                    self.sacc_log.insert('end','仅输入一路 GPS，不进行对比。\n')
                for label,info in results:
                    if info['samples']<10:
                        self.sacc_log.insert('end',f'{label}：有效样本 {info["samples"]} < 10，指标返回 NaN。\n')
                    else:
                        self.sacc_log.insert('end',f'{label}：样本 {info["samples"]}，'
                            f'sAcc中位={fmt(info["median_sacc"],".4f")} m/s，残差中位={fmt(info["median_residual"],".4f")} m/s，'
                            f'ρ={fmt(info["rho"],".5f")}，'
                            f'cos={fmt(info["cosine"],".5f")}，'
                            f'<1σ={fmt(info["coverage"][0],".2f")}%，<2σ={fmt(info["coverage"][1],".2f")}%，'
                            f'<3σ={fmt(info["coverage"][2],".2f")}%\n')
                self.sacc_log.insert('end',f'汇总 CSV：{paths["summary"]}\n逐历元数据：{paths["detail"]}\n对比 CSV：{paths["comparison"] or "无"}\n')
                self.sacc_log.see('end')
            except Exception as exc:messagebox.showerror('sAcc 一致性评估失败',str(exc))
            finally:self._sacc_running=False;self.sacc_button.configure(state='normal')
        def worker():
            try:
                from pylib.fusion_reference import load
                from pylib.accuracy_analysis import compensate_delays
                from pylib.sacc_reliability import evaluate,export
                series=[(number,parse_gps(p)) for number,p in selected]
                reference=load(path,gps1=series[0][1],yaw_diff_deg=yaw_diff,estimate_yaw_diff=estimate_yaw)
                note=rotation_description(reference,yaw_diff,estimate_yaw)
                self.after(0,lambda note=note:self.sacc_log.insert('end',f'旋转方式：{note}\n'))
                if compensate:
                    series,delays=compensate_delays(reference,series)
                    for number,delay,correlation in delays:
                        self.after(0,lambda number=number,delay=delay,correlation=correlation:
                                   self.sacc_log.insert('end',f'GPS {number}: NE 时延={delay:+d} ms（corr={correlation:.5f}），时间戳偏移 {-delay:+d} ms\n'))
                results,details=evaluate(reference,series,tolerance)
                paths=export(results,details)
                self.after(0,lambda:finish(result=(results,details,paths)))
            except Exception as exc:self.after(0,lambda error=str(exc):finish(error=error))
        threading.Thread(target=worker,daemon=True).start()
    def pick_sanitize(self,row):
        if row==0:
            path=filedialog.askopenfilename(filetypes=[('ULog','*.ulg *.ulog'),('所有文件','*.*')])
            if path:
                self.sanitize_source.set(path)
                self.sanitize_destination.set(os.path.splitext(path)[0]+'_sanitized.ulg')
        else:
            path=filedialog.asksaveasfilename(defaultextension='.ulg',filetypes=[('ULog','*.ulg')])
            if path:self.sanitize_destination.set(path)
    def run_sanitize(self):
        if getattr(self,'_sanitize_running',False):return
        if not self.clear_previous_results():return
        source=self.sanitize_source.get().strip();destination=self.sanitize_destination.get().strip()
        if not source or not destination:
            messagebox.showerror('ULog 脱敏失败','请选择原始文件和输出文件');return
        self.save_history();self._sanitize_running=True;self.sanitize_button.configure(state='disabled')
        self.sanitize_log.delete('1.0','end');self.sanitize_log.insert('end','正在后台提取白名单字段并验证……\n')
        def finish(report=None,error=None):
            try:
                if error:raise ValueError(error)
                for name,count,fields in report:
                    self.sanitize_log.insert('end',f'{name}：{count} 条\n  保留字段：{", ".join(fields)}\n')
                self.sanitize_log.insert('end',f'完成，已逐字段验证数值及真值加载兼容性。\n输出：{destination}\n大小：{os.path.getsize(source):,} → {os.path.getsize(destination):,} 字节\n可在延迟或精度项目中选择该新文件，并搭配原来同一份 GPS1 日志。\n')
            except Exception as exc:messagebox.showerror('ULog 脱敏失败',str(exc))
            finally:self._sanitize_running=False;self.sanitize_button.configure(state='normal')
        def worker():
            try:
                from pylib.ulog_sanitize import sanitize
                report=sanitize(source,destination)
                self.after(0,lambda:finish(report=report))
            except Exception as exc:self.after(0,lambda error=str(exc):finish(error=error))
        threading.Thread(target=worker,daemon=True).start()
    def run_accuracy(self):
        if getattr(self,'_accuracy_running',False):return
        if not self.clear_previous_results():return
        try:
            path=self.accuracy_ref.get().strip()
            if not path.lower().endswith(('.ulg','.ulog')):raise ValueError('请选择 ULog 真值文件')
            selected=[(i+1,v.get().strip()) for i,v in enumerate(self.accuracy_gps) if v.get().strip()]
            if not selected or selected[0][0]!=1:raise ValueError('必须选择 GPS1 作为时间映射锚点')
            tolerance=float(self.accuracy_tol.get())
            if not np.isfinite(tolerance) or tolerance<0:raise ValueError('对齐容差必须是非负有限值')
            yaw_diff=parse_optional_float(self.accuracy_yaw_diff.get(),'固定 yaw_diff(°)')
            estimate_yaw=bool(self.accuracy_estimate_yaw.get())
            compensate=bool(self.accuracy_compensate.get())
        except Exception as exc:messagebox.showerror('精度评估失败',str(exc));return
        self.save_history();self._accuracy_running=True;self.accuracy_button.configure(state='disabled')
        self.accuracy_log.delete('1.0','end')
        self.accuracy_log.insert('end',f'正在后台解析并计算动态精度指标……\n时延补偿：{"开启（NE）" if compensate else "关闭"}\n')
        self.accuracy_table.delete(*self.accuracy_table.get_children())
        def finish(result=None,output=None,plot=None,error=None):
            try:
                if error:raise ValueError(error)
                from pylib.accuracy_analysis import compare,export_comparison
                for values,tag in compare(result):
                    self.accuracy_table.insert('', 'end',values=values,tags=(tag,))
                comparison_path=export_comparison(result)
                compensation='已按各自 NE 时延补偿' if compensate else '未补偿延迟'
                self.accuracy_log.insert('end',f'绿色：优于 GPS1；红色：劣于 GPS1；灰色：相同或未比较。12 项指标均为误差值，越小越好。\n{compensation}；各 GPS 按自身有效样本统计，样本数详见原始指标 CSV。\n指标 CSV：{output}\n对比 CSV：{comparison_path}\n')
                self.accuracy_log.see('end')
                plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
                plt.rcParams['axes.unicode_minus']=False
                show_comparison(self,plot[0],plot[0],selected,plot[1],labels={n:f'GPS {n}' for n,_ in selected},
                                fields=plot[2],window_title='动态精度：统计时刻的数据曲线对比')
            except Exception as exc:messagebox.showerror('精度评估失败',str(exc))
            finally:self._accuracy_running=False;self.accuracy_button.configure(state='normal')
        def worker():
            try:
                from pylib.fusion_reference import load
                from pylib.accuracy_analysis import evaluate,export,compensate_delays
                series=[(number,parse_gps(p)) for number,p in selected]
                reference=load(path,gps1=series[0][1],require_position=True,yaw_diff_deg=yaw_diff,estimate_yaw_diff=estimate_yaw)
                note=rotation_description(reference,yaw_diff,estimate_yaw)
                self.after(0,lambda note=note:self.accuracy_log.insert('end',f'旋转方式：{note}\n'))
                if compensate:
                    series,delays=compensate_delays(reference,series)
                    for number,delay,correlation in delays:
                        self.after(0,lambda number=number,delay=delay,correlation=correlation:
                                   self.accuracy_log.insert('end',f'GPS {number}: NE 时延={delay:+d} ms（corr={correlation:.5f}），时间戳偏移 {-delay:+d} ms\n'))
                result=evaluate(series,'dynamic',reference=reference,tolerance=tolerance)
                output=export(result,'dynamic')
                from pylib.accuracy_analysis import curve_data
                plot=curve_data(reference,series,tolerance)
                self.after(0,lambda:finish(result,output,plot))
            except Exception as exc:self.after(0,lambda error=str(exc):finish(error=error))
        threading.Thread(target=worker,daemon=True).start()
    def run_cno(self):
        if getattr(self,'_cno_running',False): return
        if not self.clear_previous_results():return
        try:
            if plt is None: raise ValueError('请先安装 matplotlib')
            selected=[(i+1,v.get().strip()) for i,v in enumerate(self.cno_gps) if v.get().strip()]
            labels={number:self.cno_labels[number-1].get().strip() or os.path.basename(path) for number,path in selected}
            if not selected: raise ValueError('请至少选择一路 GPS 原始文件')
        except Exception as exc:
            messagebox.showerror('载噪比分析失败',str(exc)); return
        self.save_history(); self._cno_running=True
        self.cno_button.configure(state='disabled')
        self.cno_log.delete('1.0','end'); self.cno_log.insert('end','正在后台解析卫星消息……\n')
        def finish(result=None,error=None):
            try:
                if error: raise ValueError(error)
                from pylib import cno_analysis
                plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
                plt.rcParams['axes.unicode_minus']=False
                lower,upper=result['lower'],result['upper']
                self.cno_log.insert('end',f'公共时间范围：{lower}～{upper} ms，跨度 {(upper-lower)/1000:.1f} s\n')
                for (number,count,stats),(times,rows) in zip(result['summaries'],result['series']):
                    values=[r['mean_cno'] for r in rows if np.isfinite(r['mean_cno'])]
                    self.cno_log.insert('end',f'GPS {number}：参与定位历元 {count} 个，公共范围内 {len(times)} 个；SAT={stats["sat_frames"]} 帧，SVINFO={stats["svinfo_frames"]} 帧，校验失败={stats["checksum_errors"]} 帧\n')
                    if values: self.cno_log.insert('end',f'  公共范围内平均 C/N0={np.mean(values):.2f} dB-Hz；平均使用卫星数={np.mean([r["used_count"] for r in rows]):.2f}\n')
                    else: self.cno_log.insert('end','  参与定位但没有 C/N0>0 的有效卫星数据。\n')
                image_path,csv_path=cno_analysis.show(self,result,labels=labels)
                self.cno_log.insert('end',f'图片已保存：{image_path}\n数据已保存：{csv_path}\n')
            except Exception as exc: messagebox.showerror('载噪比分析失败',str(exc))
            finally:
                self._cno_running=False; self.cno_button.configure(state='normal')
        def worker():
            try:
                from pylib.cno_analysis import analyze
                result=analyze(selected)
                self.after(0,lambda:finish(result=result))
            except Exception as exc:
                self.after(0,lambda error=str(exc):finish(error=error))
        threading.Thread(target=worker,daemon=True).start()
    def run_ephemeris(self):
        if not self.clear_previous_results():return
        try:
            path=self.eph_path.get().strip()
            if not path: raise ValueError('请选择需要分析的 RTCM 日志文件')
            from pylib import rtcm_ephemeris
            if plt is not None:
                plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
                plt.rcParams['axes.unicode_minus']=False
            rtcm_ephemeris.show(self,path)
        except Exception as e: messagebox.showerror('星历检测失败',str(e))
    def pick(self,v,kind='gps'):
        filetypes=[('所有文件','*.*')] if kind=='ffc' else [('GPS files','*.ubx *.log *.bin *.*'),('所有文件','*.*')]
        f=filedialog.askopenfilename(filetypes=filetypes);
        if f:v.set(f)
    def load_history(self):
        try:
            with open(self.config_path,encoding='utf-8') as f: d=json.load(f)
            saved_ref=d.get('ref','')
            self.ref.set(saved_ref if str(saved_ref).lower().endswith(('.ulg','.ulog')) else '')
            for v,p in zip(self.gps,d.get('gps',[])): v.set(p)
            for v,name in zip(self.gps_labels,d.get('gps_labels',[])): v.set(name)
            self.tol.set(str(d.get('tol','200')))
            # 固定 yaw_diff 与估计开关都不写入历史：启动时固定值为空、估计默认打开。
            self.fusion_frame.set(d.get('fusion_frame','local'))
            self.north_confirmed.set(bool(d.get('north_confirmed',False)))
            self.anchor_boot.set(str(d.get('anchor_boot',''))); self.anchor_tow.set(str(d.get('anchor_tow','')))
            self.eph_path.set(d.get('eph_path',''))
            for v,p in zip(self.cno_gps,d.get('cno_gps',d.get('gps',[]))): v.set(p)
            for v,name in zip(self.cno_labels,d.get('cno_labels',[])): v.set(name)
            self.accuracy_ref.set(d.get('accuracy_ref',self.ref.get()))
            for v,p in zip(self.accuracy_gps,d.get('accuracy_gps',d.get('gps',[]))):v.set(p)
            self.accuracy_tol.set(str(d.get('accuracy_tol','200')))
            self.sanitize_source.set(d.get('sanitize_source',''))
            self.sanitize_destination.set(d.get('sanitize_destination',''))
            for v,p in zip(self.relative_paths,d.get('relative_paths',d.get('gps',[]))):v.set(p)
            self.relative_window.set(str(d.get('relative_window','3')))
            self.relative_tolerance.set(str(d.get('relative_tolerance','60')))
            self.sacc_ref.set(d.get('sacc_ref',d.get('accuracy_ref',self.ref.get())))
            for v,p in zip(self.sacc_gps,d.get('sacc_gps',d.get('accuracy_gps',d.get('gps',[])))):v.set(p)
            self.sacc_tol.set(str(d.get('sacc_tol','200')))
            # 固定 yaw_diff 与估计开关都不写入历史：启动时固定值为空、估计默认打开。
        except (OSError,ValueError,TypeError,json.JSONDecodeError): pass
    def save_history(self):
        d={'ref':self.ref.get(),'gps':[v.get() for v in self.gps],'gps_labels':[v.get() for v in self.gps_labels],'tol':self.tol.get(),
           'fusion_frame':self.fusion_frame.get(),'north_confirmed':self.north_confirmed.get(),
           'anchor_boot':self.anchor_boot.get(),'anchor_tow':self.anchor_tow.get(),'eph_path':self.eph_path.get(),
           'cno_gps':[v.get() for v in self.cno_gps],'cno_labels':[v.get() for v in self.cno_labels],
           'accuracy_ref':self.accuracy_ref.get(),'accuracy_gps':[v.get() for v in self.accuracy_gps],'accuracy_tol':self.accuracy_tol.get(),
           'sanitize_source':self.sanitize_source.get(),'sanitize_destination':self.sanitize_destination.get(),
           'relative_paths':[v.get() for v in self.relative_paths],'relative_window':self.relative_window.get(),'relative_tolerance':self.relative_tolerance.get(),
           'sacc_ref':self.sacc_ref.get(),'sacc_gps':[v.get() for v in self.sacc_gps],'sacc_tol':self.sacc_tol.get()}
        try:
            with open(self.config_path,'w',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2)
        except OSError: pass
    def on_close(self): self.save_history(); self.destroy()
    def run(self, interval=None):
        if getattr(self,'_running',False): return
        if not self.clear_previous_results():return
        self.log.delete('1.0','end')
        self.save_history()
        self._running=True
        self.log.insert('end','正在后台解析文件，请稍候，界面仍可响应……\n')
        threading.Thread(target=lambda:self._run(interval),daemon=True).start()
    def _run(self, interval=None):
        try:
            if plt is None: raise RuntimeError('请先安装 matplotlib：pip install matplotlib')
            ref_path=self.ref.get().strip()
            if ref_path and not ref_path.lower().endswith(('.ulg','.ulog')):
                raise ValueError('?? GPS ??????? ULog ?????? .ulg ????? ffc.csv ??????')
            from pylib.fusion_reference import load
            ref=None
            selected=[(i+1,v.get().strip()) for i,v in enumerate(self.gps) if v.get().strip()]
            labels={number:self.gps_labels[number-1].get().strip() or os.path.basename(path) for number,path in selected}
            gps=[parse_gps(path) for _,path in selected]
            if not gps: raise RuntimeError('至少选择一个 GPS 文件')
            for (number,path),data in zip(selected,gps):
                if not data: raise RuntimeError(f'GPS {number} 未解析到有效数据：{path}')
            yaw_diff=parse_optional_float(self.yaw_diff.get(),'固定 yaw_diff(°)')
            estimate_yaw=bool(self.estimate_yaw.get())
            if ref_path:
                if not self.gps[0].get().strip(): raise ValueError('GPS1 is required as the sensor_gps clock anchor')
                ref=load(ref_path,gps1=gps[0],yaw_diff_deg=yaw_diff,estimate_yaw_diff=estimate_yaw)
                rotation=rotation_description(ref,yaw_diff,estimate_yaw)
                self.after(0,lambda rotation=rotation:self.log.insert('end',f'ULog sensor_gps -> GPS1 PVT identity clock mapping; GPS2/3 share iTOW. 旋转方式：{rotation}\n'))
            reference=ref if ref is not None else gps[0]
            if not reference: raise RuntimeError('参考文件中没有可分析的数据')
            if interval is not None:
                origin=min(reference)
                lo,hi=interval
                reference={t:r for t,r in reference.items() if lo*1000 <= t-origin <= hi*1000}
                gps=[{t:r for t,r in data.items() if lo*1000 <= t-origin <= hi*1000} for data in gps]
                if not reference or any(not data for data in gps):raise ValueError('所选 A-B 区间内没有足够数据')
                if ref is not None: ref=reference
            tol=float(self.tol.get())
            if not np.isfinite(tol) or tol<0: raise ValueError('对齐容差必须是非负有限数值')
            aligned=[align(reference,x,tol) for x in gps]
            from pylib.gps_timing import estimate_delay, estimate_channel_delay

            def report_delay(reference_data, data, title, uniform_grid=False):
                # 真值比较与 GPS 间比较共用同一算法；两路 GPS 比较按参考脚本
                # 使用 20 ms 均匀网格、限制插值间隙，且不输出水平地速通道。
                delay,score=estimate_delay(reference_data,data,uniform_grid_resample=uniform_grid)
                text=f'{title}: NE lag={delay:.0f} ms, correlation={score:.5f}\n'
                channels=TWO_GPS_DELAY_CHANNELS if uniform_grid else DELAY_CHANNELS
                for channel,label in channels:
                    ch_delay,ch_corr,ch_rmse=estimate_channel_delay(reference_data,data,channel,uniform_grid_resample=uniform_grid)
                    if np.isfinite(ch_delay):
                        text+=f'  {label}: lag={ch_delay:.0f} ms, corr={ch_corr:.5f}, RMSE={ch_rmse:.6g}\n'
                self.after(0,lambda text=text:self.log.insert('end',text))

            if ref is not None:
                for (number,path),data in zip(selected,gps):
                    report_delay(ref,data,f'GPS {number}')
            gps_by_number={number:data for (number,_),data in zip(selected,gps)}
            if 1 in gps_by_number and 2 in gps_by_number:
                report_delay(gps_by_number[1],gps_by_number[2],'GPS 2 相对 GPS 1',uniform_grid=True)
                self.after(0,lambda:self.log.insert('end','GPS 2 相对 GPS 1：正延迟表示 GPS2 滞后 GPS1，负延迟表示 GPS2 超前 GPS1。\n'))
            if ref is not None and not any(aligned):
                raise ValueError(f'GPS 与 fusion 的 iTOW 无重叠，请检查时间锚点。fusion 范围 {min(ref)}～{max(ref)} ms')
            mode='ULog drone_fusion NED' if ref is not None else 'GPS only'
            self.after(0,lambda: self.log.insert('end',f'参考：{mode}\n'))
            for (number,_),data,rows in zip(selected,gps,aligned):
                self.after(0,lambda number=number,data=data,rows=rows: self.log.insert('end',f'GPS {number}：{len(data)} 点，匹配 {len(rows)} 点\n'))
            plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
            plt.rcParams['axes.unicode_minus']=False
            def recalc(state):
                if state is None or None in state['times']:
                    messagebox.showinfo('区间重算','请先在同一曲线页放置 A、B 两个光标。');return
                lo,hi=sorted(state['times']);self.run(interval=(lo,hi))
            self.after(0,lambda: show_comparison(self,reference,ref,selected,aligned,labels=labels,interval_callback=recalc))
        except Exception as e:
            # Python 会在离开 except 块时清理异常变量 e，不能在稍后执行的
            # Tkinter lambda 中直接引用它；先复制成普通字符串。
            error_text = str(e)
            self.after(0, lambda msg=error_text: messagebox.showerror('分析失败', msg))
        finally:
            self._running=False

if __name__=='__main__': App().mainloop()
