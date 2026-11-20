# yaw_diff 速度序列估计说明

编写日期：2026-11-18。

[返回文档目录](01-工具使用说明.md)

## 用途

用 `drone_fusion` 的机体系水平速度与 GPS 的 NED 水平速度，估计一个全时段固定的 `yaw_diff`，用于把真值速度、位置旋转到北东地坐标系。核心实现在 `pylib/yaw_estimation.py`。

## 符号约定

与 `pylib/fusion_reference.py` 完全一致：

```text
yaw_diff = drone_fusion.yaw − gps.yaw
实际旋转角 psi = −yaw_diff
```

估计器先求出使机体系速度最接近 GPS 速度的旋转角 `psi`，再返回 `yaw_diff = −psi`。因此界面显示的 yaw_diff 与“固定 yaw_diff”输入框使用同一符号。

## 估计方法

对每个有效时刻，机体系水平速度为 \(\mathbf b=(v_x,v_y)\)，GPS 水平速度为 \(\mathbf g=(v_N,v_E)\)。寻找旋转矩阵 \(R(\psi)\) 使

$$
R(\psi)\mathbf b \approx \mathbf g
$$

二维正交 Procrustes 的闭式解为

$$
\psi=\operatorname{atan2}\left(\sum_i (b_{x,i}g_{E,i}-b_{y,i}g_{N,i}),\ \sum_i (b_{x,i}g_{N,i}+b_{y,i}g_{E,i})\right)
$$

再输出

$$
\text{yaw\_diff}=-\psi
$$

残差均方根：

$$
\text{RMS}=\sqrt{\frac1n\sum_i \left(\left[R(\psi)\mathbf b\right]_{N,i}-g_{N,i}\right)^2
+\left(\left[R(\psi)\mathbf b\right]_{E,i}-g_{E,i}\right)^2}
$$

仅使用水平分量，不使用高度或垂直速度。

## 样本筛选与时间对齐

- 四组速度必须有限；
- 机体系水平速率与 GPS 水平速率都不小于 `min_speed`，默认 `0.5 m/s`，避免低速时航向噪声主导估计；
- 有效运动样本不少于 `min_samples`，默认 `20`；
- 按时间对齐时，GPS 速度在机体系时刻线性插值；只使用最近 GPS 样本不超过 `tolerance`（默认 `200 ms`）、且插值两侧间隔不超过 `2×tolerance` 的点，不外推、不跨越较大空洞。

`estimate()` 接收已经配对的四组数组；`estimate_from_series()` 接收两组时间序列并完成上述对齐。

## 在真值加载中的使用

`pylib/fusion_reference.load()` 新增参数：

```python
load(..., yaw_diff_deg=None, estimate_yaw_diff=True)
```

旋转角度来源优先级：

1. `yaw_diff_deg` 有值：全时段使用该固定角度，忽略估计开关；
2. 否则 `estimate_yaw_diff=True`（界面勾选“估计 yaw_diff”，默认打开）：由 GPS1 的 NED 速度与 `drone_fusion` 机体系速度估计一个固定 `yaw_diff`；
3. 两者都不用：退回逐时刻 `drone_fusion.yaw − fusion_gps.yaw`，并要求 `fusion_gps` 航向覆盖。

估计失败（例如静止悬停、有效运动样本不足）时会报错，并提示取消勾选“估计 yaw_diff”或填写固定值；不会静默切换到其他模式。

界面每次启动时固定 `yaw_diff` 为空、“估计 yaw_diff”默认勾选，因此默认直接使用速度序列估计。固定值与估计开关都不写入历史配置，只在当前会话生效。估计成功后，运行日志会显示实际的 `yaw_diff` 与旋转角。

## 单独运行

模块可直接作为命令使用：

```sh
python -m pylib.yaw_estimation body.csv gps.csv
```

或一个包含全部列的合并 CSV：

```sh
python -m pylib.yaw_estimation combined.csv
```

可选参数：

```text
--tolerance 200      时间对齐容差 (ms)
--min-speed 0.5      参与估计的最小水平速度 (m/s)
--min-samples 20     最少运动样本数
```

CSV 需要表头，时间列单位为毫秒。列名可用别名：

| 数据 | 可用列名 |
| --- | --- |
| 时间 | `time`、`t`、`itow`、`iTOW`、`timestamp` |
| 机体系 X 速度 | `vel_x`、`velocity_x`、`vx`、`x` |
| 机体系 Y 速度 | `vel_y`、`velocity_y`、`vy`、`y` |
| GPS 北向速度 | `vel_n`、`veln`、`vn`、`n` |
| GPS 东向速度 | `vel_e`、`vele`、`ve`、`e` |

两个文件模式分别读取机体系列与 GPS 列；合并模式要求两组列在同一行对齐。输出示例：

```text
yaw_diff = -25.000000°
rotation = 25.000000°
RMS      = 0.000000 m/s
samples  = 40
matched  = 40
```

Python 调用示例：

```python
from pylib.yaw_estimation import estimate, estimate_from_series

result = estimate(vel_x, vel_y, vel_n, vel_e)
print(result['yaw_diff_deg'], result['rms'], result['samples'])
```

## 局限

- 只估计水平安装偏角；滚转、俯仰耦合与安装杆臂不在模型中。
- 时间轴必须先对齐；残余时间滞后会被折算成角度偏差，速度越大、转弯越剧烈时影响越明显。
- 低动态或长时间静止的日志无法提供足够方向信息，应改用固定值或逐时刻自动模式。
- 估计结果是全时段常量，不跟踪航向随时间的变化；如需逐时刻航向，请使用 `fusion_gps` 自动模式。
