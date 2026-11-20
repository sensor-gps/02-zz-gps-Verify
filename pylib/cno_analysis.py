"""解析 NAV-SAT / NAV-SVINFO，并按 iTOW 对比参与定位卫星的载噪比。"""
import csv
import struct
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'analysis_output'


def parse(path):
    """同一历元优先使用 NAV-SAT；不混合两类消息的卫星编号；只保留参与定位卫星数大于 0 的历元。"""
    raw = Path(path).read_bytes()
    records = {}
    stats = dict(sat_frames=0, svinfo_frames=0, checksum_errors=0)
    offset = 0
    while offset + 8 <= len(raw):
        start = raw.find(b'\xb5\x62', offset)
        if start < 0 or start + 8 > len(raw):
            break
        cls, mid, length = struct.unpack_from('<BBH', raw, start + 2)
        end = start + 8 + length
        if end > len(raw):
            offset = start + 2
            continue
        offset = end
        if cls != 1 or mid not in (0x30, 0x35):
            continue
        a = b = 0
        for value in raw[start + 2:end - 2]:
            a = (a + value) & 255
            b = (b + a) & 255
        if raw[end - 2:end] != bytes((a, b)):
            stats['checksum_errors'] += 1
            offset = start + 2
            continue
        payload = raw[start + 6:end - 2]
        if length < 8:
            continue
        tow = struct.unpack_from('<I', payload)[0]
        count = payload[5] if mid == 0x35 else payload[4]
        if not 0 <= tow < 604800000 or length < 8 + 12 * count:
            continue
        if mid == 0x35 and payload[4] != 1:
            continue
        source = 'NAV-SAT' if mid == 0x35 else 'NAV-SVINFO'
        stats['sat_frames' if mid == 0x35 else 'svinfo_frames'] += 1
        satellites = {}
        for index in range(count):
            block = 8 + index * 12
            if mid == 0x35:
                key = (payload[block], payload[block + 1])
                cno = payload[block + 2]
                used = bool(struct.unpack_from('<I', payload, block + 8)[0] & 8)
            else:
                # SVINFO 的 svid 是协议定义的全局编号；chn=255 表示未分配通道。
                key = payload[block + 1]
                if payload[block] == 255 or key == 0:
                    continue
                cno = payload[block + 4]
                used = bool(payload[block + 2] & 1)
            satellites[key] = (used, cno)
        used_values = [cno for used, cno in satellites.values() if used]
        positive = [cno for cno in used_values if cno > 0]
        row = dict(source=source, tracked_count=len(satellites),
                   used_count=len(used_values), cno_count=len(positive),
                   mean_cno=float(np.mean(positive)) if positive else np.nan)
        if tow not in records or source == 'NAV-SAT':
            records[tow] = row
    if not records:
        raise ValueError('未找到校验有效的 NAV-SAT / NAV-SVINFO 数据；仅含 NAV-PVT 的文件无法计算卫星载噪比。')
    # 只记录参与定位卫星数大于 0 的历元。
    records = {tow: row for tow, row in records.items() if row['used_count'] > 0}
    if not records:
        raise ValueError('日志中没有参与定位的卫星历元，无法计算载噪比对比。')
    return dict(sorted(records.items())), stats


def common_series(series):
    """取所有文件的公共 iTOW 时间范围；每路保留自身历元，不做对齐、插值或外推。"""
    if any(not data for data in series):
        raise ValueError('至少一路日志没有参与定位的卫星，无法建立有效对比时间范围。')
    # 交集：起点取各文件最晚的首个历元，终点取各文件最早的末个历元。
    lower = max(min(data) for data in series)
    upper = min(max(data) for data in series)
    if lower > upper:
        raise ValueError('所选日志没有共同的 iTOW 时间范围，请检查是否来自同一次记录且未跨 GPS 周。')
    clipped = []
    for data in series:
        times = np.array([t for t in data if lower <= t <= upper], dtype=np.int64)
        clipped.append((times, [data[int(t)] for t in times]))
    return int(lower), int(upper), clipped


def analyze(selected):
    series, summaries = [], []
    for number, path in selected:
        try:
            data, stats = parse(path)
        except ValueError as exc:
            raise ValueError(f'GPS {number}：{exc}') from exc
        series.append(data)
        summaries.append((number, len(data), stats))
    lower, upper, clipped = common_series(series)
    return dict(selected=selected, summaries=summaries,
                lower=lower, upper=upper, series=clipped)


def make_figure(result, labels=None):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(14, 5), dpi=120)
    axes = fig.subplots(1, 2, sharex=True)
    origin = result['lower']
    for (number, path), (times, rows) in zip(result['selected'], result['series']):
        x = (times - origin) / 1000
        for ax, field in zip(axes, ('used_count', 'mean_cno')):
            values = [row[field] for row in rows]
            ax.plot(x, values, label=(labels or {}).get(number) or Path(path).name, linewidth=1.1)
    axes[0].set(title='参与定位卫星数', ylabel='卫星数（颗）')
    axes[1].set(title='参与定位卫星平均载噪比', ylabel='平均 C/N0（dB-Hz）')
    for ax in axes:
        ax.set_xlabel('相对公共时间起点 iTOW（秒）')
        ax.grid(True, alpha=.3)
        ax.legend()
    fig.suptitle('GPS 载噪比对比（公共时间范围）')
    fig.tight_layout()
    return fig


def show(parent, result, labels=None):
    import tkinter as tk
    from tkinter import ttk
    from datetime import datetime
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    fig = make_figure(result, labels=labels)
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = 'cno_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    image_path = OUTPUT_DIR / (stem + '.png')
    csv_path = OUTPUT_DIR / (stem + '.csv')
    fig.savefig(image_path, dpi=160)
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['gps', 'legend_name', 'itow_ms', 'source_message',
                         'used_satellites', 'valid_cno_satellites', 'mean_cno_dbhz'])
        for (number, path), (times, rows) in zip(result['selected'], result['series']):
            name=(labels or {}).get(number) or Path(path).name
            for t, row in zip(times, rows):
                writer.writerow([number, name, int(t), row['source'],
                                 row['used_count'], row['cno_count'], row['mean_cno']])
    window = tk.Toplevel(parent)
    window.title('C/N0 载噪比对比')
    window.geometry('1400x650')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    NavigationToolbar2Tk(canvas, window)
    ttk.Label(window, text=f'图片：{image_path}\n数据：{csv_path}').pack(anchor='w')
    canvas.draw()
    window.plot_objects = (fig, canvas)
    return image_path, csv_path
