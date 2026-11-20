"""sAcc 一致性与可靠性评估。

流程：

1. 以 ULog 真值（drone_fusion 的 NED 速度）为参考，先按 NE 速度时延估计并
   补偿各路 GPS 的时间戳。
2. 每个 GPS PVT 历元在容差内匹配真值速度，得到三维速度残差
   ``rr_i = v_gps - v_truth``，``r_i = ||rr_i||``。
3. ``sAcc_i`` 取自 UBX NAV-PVT 的速度精度估计（m/s）。
4. 仅保留 sAcc 有限且 > 0、残差三个分量全部有限的历元；样本数 < 10 时
   指标直接返回 NaN。
5. sAcc 与 r 分别稳定排序并取平均秩（并列值共享平均秩，如
   [1, 2, 2, 5] -> [1, 2.5, 2.5, 4]），中心化后求 Pearson 相关，即
   Spearman 秩相关系数；分母为 0（全部 sAcc 或全部残差相等）时返回 NaN。
   同时计算未中心化的余弦相似度，衡量两者是否同比例变化；并输出 sAcc 与 r
   的中位数，仅作样本分布描述，不参与多路对比。
6. 把 sAcc 视为一倍速度中误差，统计 r 小于 1/2/3 倍 sAcc 的历元占比。
7. 输入不少于两路时，以第一路为基准输出对比。ρ 与余弦相似度越大越好；
   覆盖率则与正态分布 ±1/2/3σ 的理论占比（68.27% / 95.45% / 99.73%）
   比较，越接近目标越好，而不是占比越大越好。

结论边界：秩相关、余弦相似度与覆盖率只说明接收机自报速度精度与实测速度
残差在等级顺序、同比例变化和覆盖比例上是否自洽，不代表绝对定位精度，也
不代表 sAcc 是真实测量噪声。
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

SIGMA_MULTIPLIERS = (1, 2, 3)
MIN_SAMPLES = 10
VELOCITY_KEYS = ('vel_n', 'vel_e', 'vel_d')

HEADERS = ('GPS', '样本数', 'sAcc 中位数(m/s)', '残差中位数(m/s)', '秩相关系数 ρ',
           '余弦相似度', '<1σ 占比(%)', '<2σ 占比(%)', '<3σ 占比(%)')
COMPARISON_HEADERS = ('指标', '单位', '目标', 'GPS1', '对比对象', '对比值', '差值', '改善比例', '结论')
DETAIL_HEADERS = ('gps', 'itow_ms', 'sacc_m_s', 'r_m_s', 'residual_n', 'residual_e', 'residual_d')
# 正态分布 ±1/2/3σ 的理论覆盖率（erf(k/sqrt(2))）。
NORMAL_COVERAGE = (68.26894921370858, 95.44997361036416, 99.73002039367398)
# 对比指标索引：0 = ρ，1 = 余弦相似度，2/3/4 = <1σ/<2σ/<3σ 占比。
# 中位数为描述性输出，不进入对比。target 为 None 时按越大越好比较；
# 覆盖率按与正态目标的绝对偏离比较，越接近目标越好。
COMPARISON_METRICS = (
    ('Spearman 秩相关系数 ρ', '', 0, None),
    ('余弦相似度', '', 1, None),
    ('r<1σ 占比', '%', 2, NORMAL_COVERAGE[0]),
    ('r<2σ 占比', '%', 3, NORMAL_COVERAGE[1]),
    ('r<3σ 占比', '%', 4, NORMAL_COVERAGE[2]),
)


def average_ranks(values):
    """稳定排序并返回 1 起的平均秩；相等值共享平均秩。"""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def rank_correlation(a, b):
    """平均秩中心化后的 Pearson 相关；任一序列全相等时返回 NaN。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return float('nan')
    u = average_ranks(a)
    w = average_ranks(b)
    u = u - u.mean()
    w = w - w.mean()
    denominator = float(np.sqrt(np.sum(u * u) * np.sum(w * w)))
    if not np.isfinite(denominator) or denominator <= 0:
        return float('nan')
    return float(np.sum(u * w) / denominator)


def cosine_similarity(a, b):
    """未中心化余弦相似度 Σ(a·b)/sqrt(Σa²·Σb²)；分母为 0 时返回 NaN。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or not a.size:
        return float('nan')
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if not np.isfinite(denominator) or denominator <= 0:
        return float('nan')
    return float(np.sum(a * b) / denominator)


def velocity_samples(reference, data, tolerance):
    """按 GPS 独立历元返回 (itow, sAcc, 三维速度残差向量)。

    只在真值覆盖范围内、最近真值样本不超过 tolerance 且插值两侧间隔不超过
    2*tolerance 时匹配，不外推、不跨越较大真值空洞。残差为 GPS 减真值。
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('对齐容差必须是非负有限值')
    rt = np.array(sorted(reference), dtype=float)
    rv = np.array([[reference[t].get(k, np.nan) for k in VELOCITY_KEYS] for t in rt], dtype=float)
    truth_ok = np.isfinite(rv).all(1)
    rt, rv = rt[truth_ok], rv[truth_ok]
    if not len(rt):
        raise ValueError('真值中没有有效的三轴速度')
    gt = np.array(sorted(data), dtype=float)
    if not len(gt):
        return np.empty(0), np.empty(0), np.empty((0, 3))
    gv = np.array([[data[t].get(k, np.nan) for k in VELOCITY_KEYS] for t in gt], dtype=float)
    sacc = np.array([data[t].get('sacc', np.nan) for t in gt], dtype=float)
    index = np.searchsorted(rt, gt)
    left = np.clip(index - 1, 0, len(rt) - 1)
    right = np.clip(index, 0, len(rt) - 1)
    # 时间戳恰好等于真值样本时直接使用该样本，不要求两侧插值间隔。
    exact = (index < len(rt)) & (rt[np.minimum(index, len(rt) - 1)] == gt)
    nearest = np.minimum(np.abs(rt[left] - gt), np.abs(rt[right] - gt))
    nearest[exact] = 0.0
    bracket = (right == left) | ((rt[right] - rt[left]) <= 2 * tolerance) | exact
    keep = ((gt >= rt[0]) & (gt <= rt[-1]) & (nearest <= tolerance) & bracket
            & np.isfinite(gv).all(1) & np.isfinite(sacc) & (sacc > 0))
    if not keep.any():
        return np.empty(0), np.empty(0), np.empty((0, 3))
    truth_velocity = np.column_stack([np.interp(gt[keep], rt, rv[:, k]) for k in range(3)])
    residual = gv[keep] - truth_velocity
    return gt[keep], sacc[keep], residual


def summary(sacc, residual):
    """单个接收机的汇总指标；样本不足或退化时对应值为 NaN。"""
    count = int(len(sacc))
    result = {'samples': count, 'rho': float('nan'), 'cosine': float('nan'),
              'median_sacc': float('nan'), 'median_residual': float('nan'),
              'coverage': [float('nan')] * len(SIGMA_MULTIPLIERS)}
    if count < MIN_SAMPLES:
        return result
    sacc = np.asarray(sacc, dtype=float)
    r = np.linalg.norm(np.asarray(residual, dtype=float), axis=1)
    result['rho'] = rank_correlation(sacc, r)
    result['cosine'] = cosine_similarity(sacc, r)
    result['median_sacc'] = float(np.median(sacc))
    result['median_residual'] = float(np.median(r))
    result['coverage'] = [float(np.mean(r < multiplier * sacc) * 100) for multiplier in SIGMA_MULTIPLIERS]
    return result


def evaluate(reference, series, tolerance):
    """对 (GPS 编号, 数据) 列表逐路计算；返回 (汇总, 逐历元明细)。"""
    results = []
    details = []
    for number, data in series:
        times, sacc, residual = velocity_samples(reference, data, tolerance)
        results.append((f'GPS {number}', summary(sacc, residual)))
        details.append((number, times, sacc, residual))
    return results, details


def _values(info):
    """把汇总字典展平为 [ρ, 余弦相似度, <1σ, <2σ, <3σ]，缺失项为 NaN。"""
    if info is None:
        return [float('nan')] * (2 + len(SIGMA_MULTIPLIERS))
    return [float(info['rho']), float(info['cosine']),
            *[float(value) for value in info['coverage']]]


def compare(results):
    """以第一路为基准：ρ、余弦越大越好；覆盖率按与正态目标的偏离比较。"""
    if len(results) < 2:
        return []
    _, base = results[0]
    base_values = _values(base)
    output = []
    for title, unit, index, target in COMPARISON_METRICS:
        a = base_values[index]
        target_text = '—' if target is None else f'{target:.2f}%'
        for label, info in results[1:]:
            b = _values(info)[index]
            if not np.isfinite([a, b]).all():
                output.append(((title, unit, target_text,
                                '无数据' if not np.isfinite(a) else f'{a:.6g}', label,
                                '无数据' if not np.isfinite(b) else f'{b:.6g}',
                                '—', '—', '无数据，无法比较'), 'neutral'))
                continue
            if target is None:
                # ρ 与余弦相似度：越大越好；两者均可能为负，只给绝对差。
                delta = b - a
                equal = np.isclose(a, b, rtol=1e-9, atol=1e-12)
                tag = 'neutral' if equal else ('better' if delta > 0 else 'worse')
                conclusion = '相同' if equal else ('优于 GPS1' if delta > 0 else '劣于 GPS1')
                ratio = '—'
            else:
                # 覆盖率：比较与正态目标的理论占比的绝对偏离。
                deviation_a = abs(a - target)
                deviation_b = abs(b - target)
                delta = b - a
                equal = np.isclose(deviation_a, deviation_b, rtol=1e-9, atol=1e-12)
                closer = deviation_b < deviation_a
                tag = 'neutral' if equal else ('better' if closer else 'worse')
                conclusion = '相同' if equal else ('更接近正态' if closer else '更偏离正态')
                ratio = (f'{(deviation_a - deviation_b) / deviation_a * 100:+.2f}%'
                         if deviation_a > 1e-12 else '—')
            output.append(((title, unit, target_text, f'{a:.6g}', label, f'{b:.6g}',
                            f'{delta:+.6g}', ratio, conclusion), tag))
    return output


def export(results, details):
    """导出汇总、逐历元对；两路及以上时另导出对比 CSV。"""
    directory = Path(__file__).resolve().parent.parent / 'analysis_output'
    directory.mkdir(exist_ok=True)
    stem = 'sacc_reliability_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    summary_path = directory / (stem + '.csv')
    detail_path = directory / (stem + '_pairs.csv')
    with summary_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        for label, info in results:
            writer.writerow((label, info['samples'], info['median_sacc'], info['median_residual'],
                             *_values(info)))
    with detail_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(DETAIL_HEADERS)
        for number, times, sacc, residual in details:
            r = np.linalg.norm(residual, axis=1) if len(residual) else np.empty(0)
            for i, time in enumerate(times):
                writer.writerow((f'GPS {number}', int(round(time)), float(sacc[i]), float(r[i]),
                                 *[float(value) for value in residual[i]]))
    paths = {'summary': summary_path, 'detail': detail_path, 'comparison': None}
    if len(results) >= 2:
        comparison_path = directory / (stem + '_comparison.csv')
        with comparison_path.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(COMPARISON_HEADERS)
            writer.writerows(values for values, _ in compare(results))
        paths['comparison'] = comparison_path
    return paths
