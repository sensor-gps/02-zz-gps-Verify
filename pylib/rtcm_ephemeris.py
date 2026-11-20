"""RTCM3 星历接收覆盖率：文件内已收到星历，不代表星历有效期/健康状态。"""
import math
import csv
from pathlib import Path
import numpy as np

EPH={1019:'GPS',1020:'GLO',1042:'BDS',1044:'QZSS',1045:'GAL',1046:'GAL'}
MSM={107:'GPS',108:'GLO',109:'GAL',110:'SBAS',111:'QZSS',112:'BDS',113:'NAVIC'}

def bits(data,start,length):
    value=int.from_bytes(data,'big')
    return (value >> (len(data)*8-start-length)) & ((1<<length)-1)

def crc24(data):
    crc=0
    for byte in data:
        crc^=byte<<16
        for _ in range(8):
            crc<<=1
            if crc&0x1000000: crc^=0x1864cfb
    return crc&0xffffff

def analyze(path):
    raw=open(path,'rb').read(); i=0; received=set(); windows={}; stats={'frames':0,'crc_errors':0,'ephemerides':0,'msm':0}
    clocks={}; origin=None; current=None; duration=0
    def window(k):
        return windows.setdefault(k,{'sat':set(),'covered':set(received),'sent':set()})
    while i+6<=len(raw):
        if raw[i]!=0xd3 or raw[i+1]&0xfc:
            i+=1; continue
        n=((raw[i+1]&3)<<8)|raw[i+2]; end=i+n+6
        if end>len(raw): i+=1; continue
        frame=raw[i:end]
        if crc24(frame[:-3])!=int.from_bytes(frame[-3:],'big'):
            stats['crc_errors']+=1; i+=1; continue
        i=end; p=frame[3:-3]
        if len(p)<3: continue
        msg=bits(p,0,12); stats['frames']+=1
        if msg in EPH:
            width=4 if msg==1044 else 6
            satellite=(EPH[msg],bits(p,12,width))
            received.add(satellite); stats['ephemerides']+=1
            if current is not None:
                w=window(current); w['covered'].update(received); w['sent'].add(satellite)
        family=msg//10
        if family not in MSM or msg%10 not in range(1,8) or len(p)*8<169: continue
        system=MSM[family]
        if system in ('SBAS','NAVIC'): continue # 本工具尚无相应星历解析，不能算作缺失
        # GPS MSM 给出统一时间锚；其他星座及无时间戳星历按文件顺序
        # 归属到最近的前一条 GPS 历元，避免混用 GLO/BDT 时间尺度。
        if system=='GPS':
            epoch=bits(p,24,30)
            previous=clocks.get(system,epoch)
            epoch+=round((previous-epoch)/604800000)*604800000
            if epoch<previous: continue
            clocks[system]=epoch
            if origin is None: origin=epoch
            elapsed=(epoch-origin)/1000
            if elapsed<0: continue
            duration=max(duration,elapsed); current=int(elapsed//10)
            if current>100000: raise ValueError('GPS 时间跨度异常')
        if current is None: continue
        w=window(current)
        satellites={(system,k+1) for k in range(64) if bits(p,73+k,1)}
        w['sat'].update(satellites)
        # 窗口结束时重新按累计收到的星历计算覆盖。
        w['covered'].update(received)
        stats['msm']+=1
    if not windows: raise ValueError('未找到可用的 GPS MSM 时间锚，无法建立可靠时间轴。')
    x=[]; missing=[]; counts=[]; segment_missing=[]; table=[]
    for k in range(max(windows)+1):
        if k>100000: raise ValueError('观测时间跨度异常，请检查日志时间或星座时间格式')
        w=windows.get(k); x.append(k*10)
        counts.append(len(w['sat']) if w else 0)
        missing.append(100*len(w['sat']-w['covered'])/len(w['sat']) if w and w['sat'] else np.nan)
        segment_missing.append(100*len(w['sat']-w['sent'])/len(w['sat']) if w and w['sat'] else np.nan)
        table.append((k*10,min((k+1)*10,duration),counts[-1],len(w['sat']-w['covered']) if w else 0,missing[-1],segment_missing[-1]))
    stats.update(duration=duration,segment_missing=np.array(segment_missing),table=table)
    return np.array(x),np.array(missing),np.array(counts),stats

def show(parent,path):
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
    from matplotlib.ticker import MultipleLocator
    x,y,n,stats=analyze(path)
    window=tk.Toplevel(parent); window.title('RTCM 星历检测'); window.geometry('1200x760')
    fig=Figure(figsize=(14,8)); ax=fig.add_subplot(111); right=ax.twinx()
    width=np.array([max(.1,end-start) for start,end,*_ in stats['table']]); centers=x+width/2
    bars=right.bar(centers,n,width=width*.82,color='#dbe1e9',label='每段观测卫星数',zorder=0)
    ax.set_zorder(right.get_zorder()+1); ax.patch.set_visible(False)
    a,=ax.plot(centers,y,'o-',ms=3,lw=2,color='#d94d3d',label='截至段末仍未收到星历（主指标）')
    b,=ax.plot(centers,stats['segment_missing'],'--',lw=1.6,color='#b58b31',label='该10秒段未输出星历（发送情况）')
    fig.suptitle(f'{Path(path).stem}：星历缺失比例与观测卫星数随时间变化',fontsize=18,fontweight='bold',y=.97)
    ax.set_title(f'观测首末跨度 {stats["duration"]:.1f} 秒',pad=16)
    ax.set(xlabel='距首个 GPS 观测历元的时间（秒）；每段10秒，末段不足10秒',ylabel='缺少对应星历的观测卫星比例（%）',ylim=(0,100),xlim=(0,max(10,x[-1]+width[-1])))
    right.set(ylabel='每段观测卫星数（颗，按卫星去重）',ylim=(0,max(5,math.ceil(max(n)/5)*5+5)))
    ax.xaxis.set_major_locator(MultipleLocator(10)); right.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='x',labelrotation=90); ax.grid(axis='y',alpha=.3); ax.legend(handles=[a,b,bars],loc='upper left')
    fig.subplots_adjust(left=.075,right=.92,top=.82,bottom=.24)
    fig.text(.075,.075,'主指标：本段有观测、截至段末仍未收到对应星历的卫星数 / 本段观测卫星数；之前收到的星历计为已存在。',fontsize=10)
    fig.text(.075,.04,'虚线仅表示本段未输出，不等同于星历无效；未检验有效期和健康状态。时间按文件顺序关联前一 GPS 历元。',fontsize=10)
    output=Path(__file__).resolve().parent.parent/'analysis_output'; output.mkdir(exist_ok=True)
    csv_path=output/(Path(path).name+'.ephemeris.csv')
    with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['start_s','end_s','observed_satellites','missing_satellites','cumulative_missing_percent','segment_missing_percent']); writer.writerows(stats['table'])
    canvas=FigureCanvasTkAgg(fig,master=window); canvas.get_tk_widget().pack(fill='both',expand=True)
    NavigationToolbar2Tk(canvas,window)
    ttk.Label(window,text=f'有效 RTCM 帧：{stats["frames"]}；星历帧：{stats["ephemerides"]}；MSM 帧：{stats["msm"]}；CRC 错误候选帧：{stats["crc_errors"]}').pack(anchor='w')
    canvas.draw(); window.plot_objects=(fig,canvas)
