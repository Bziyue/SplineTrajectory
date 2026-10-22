# PPolyND 的 Bezier / MINVO 凸包表示与梯度

本文说明 `include/SplineConvexHull.hpp` 使用的数学变换、Bezier 细分、
反向梯度以及它和 MINCO/SplineOptimizer 的连接方式。

## 1. 输入约定

`PPolyND` 的第 $s$ 段在局部物理时间
$\tau\in[0,T_s]$ 上按升幂保存：

$$
p_s(\tau)=\sum_{k=0}^{n}c_{s,k}\tau^k,\qquad
c_{s,k}\in\mathbb R^D.
$$

转换不采样曲线。位置、速度、加速度等都使用同一个解析变换。对
$r$ 阶物理时间导数，令 $m=n-r$，再用
$u=\tau/T_s\in[0,1]$ 归一化：

$$
\frac{d^r p_s}{d\tau^r}(T_su)
=\sum_{\ell=0}^{m}\hat c_{s,\ell}u^\ell,
\qquad
\hat c_{s,\ell}
=\frac{(\ell+r)!}{\ell!}c_{s,\ell+r}T_s^\ell.
$$

这一步是稀疏对角变换，记作
$\hat C_s=N_r(T_s)C_s$。

## 2. 幂基到凸包基

设 $B_m$ 是“归一化升幂系数到控制点”的矩阵，则

$$
V_s=B_m\hat C_s=B_mN_r(T_s)C_s=M_sC_s.
$$

`V_s` 每一行是一个 $D$ 维控制点。

### 2.1 Bezier

对 $0\leq \ell\leq i\leq m$，

$$
(B_m^\mathrm{bez})_{i\ell}
=\frac{\binom{i}{\ell}}{\binom{m}{\ell}},
$$

其余元素为零。这是升幂到 Bernstein 控制点的闭式变换，不需要求逆。
Bernstein 基在 $[0,1]$ 上非负且和为一，因此曲线完全位于控制点凸包内。

### 2.2 MINVO

若官方 MINVO 基矩阵 $A_m$ 的每一行按降幂给出一个基函数，则在转换到
$[0,1]$ 后，本实现使用

$$
B_m^\mathrm{mv}=A_m^{-T}J,
$$

其中 $J$ 负责把升幂顺序反转为降幂顺序。头文件中的常数由
`minvo/src/solutions/solutionDeg0.mat` 到 `solutionDeg7.mat` 和官方
`getA_MV.m` 的行顺序一次性生成。因而 MINVO 的固定双精度矩阵支持 0 到
7 次多项式，
覆盖本库的 cubic、quintic、septic 轨迹及它们的所有导数。

运行时只做小型稠密矩阵乘法，不读取 MAT 文件，也不依赖 MATLAB、SciPy
或动态矩阵求逆。

## 3. 任意层细分与共享转换核

`subdivision_depth=s` 令每个源段产生 $L=2^s$ 个等长叶片。该操作现在同时
支持 Bezier 和 MINVO。第 $\ell$ 个叶片对应

$$
u=a_\ell+h v,\qquad
a_\ell=\frac{\ell}{L},\quad h=\frac{1}{L},\quad v\in[0,1].
$$

若归一化导数幂系数为 $\hat C$，仿射限制后的局部升幂系数
$Y_\ell=H_\ell\hat C$ 满足

$$
(H_\ell)_{jk}=
\begin{cases}
\binom{k}{j}a_\ell^{k-j}h^j,&k\ge j,\\
0,&k<j.
\end{cases}
$$

叶片控制点是

$$
Q_\ell=B_mH_\ell\hat C.
$$

把所有叶片按行堆叠，得到一次预计算的算子

$$
K=
\begin{bmatrix}
B_mH_0\\
B_mH_1\\
\vdots\\
B_mH_{L-1}
\end{bmatrix},
\qquad
Q=K\hat C.
$$

这与 Bezier 的重复二分/de Casteljau 在数学上等价，但运行时不再建立
细分树。由于 $B_m$ 可以是 Bezier 或 MINVO，MINVO 细分无需另一套算法。

内部转换核是不可变共享对象，按“基、源系数数、导数阶、细分深度”缓存
$K$、$K^T$、下降阶乘和叶片起始比例。相同拓扑的多个表示或多个优化上下文
复用同一份核。

`prepare()` 准备输出、临时工作区和共享转换核；调用者负责可行的内存及索引范围。
固定拓扑的 `update(poly)` 只更新系数、时长幂、控制点和 piece 元数据；
便捷形式 `update(poly, options)` 仅在拓扑或选项不匹配时重新准备。
时长幂通过

$$
T^0=1,\qquad T^{j+1}=T^jT
$$

递推，因此热路径不调用 `pow()`。

## 4. 控制点梯度反传

设损失对堆叠控制点的梯度为 $G_Q=\partial L/\partial Q$。先执行共享核
的伴随：

$$
G_{\hat C}=K^TG_Q.
$$

再利用归一化导数系数中的对角结构。系数梯度为

$$
\frac{\partial L}{\partial c_{s,j+r}}
=\frac{(j+r)!}{j!}T_s^j
\frac{\partial L}{\partial\hat c_{s,j}}.
$$

时长偏导是在局部物理时间幂系数固定时求得：

$$
\frac{\partial L}{\partial T_s}
=\sum_{j=1}^{m}
\left\langle
\frac{\partial L}{\partial\hat c_{s,j}},
\frac{j}{T_s}\hat c_{s,j}
\right\rangle.
$$

`backwardAdd()` 使用预分配的临时矩阵，并把结果加到调用方已有的
`grad_coeffs`、`grad_times`，不清空、不调整尺寸。这样位置、速度、加速度
等多个凸包代价可先相加，最后只调用一次 spline `backward()`。

`backward()` 返回：

- `coefficients`：和输入 `PPolyND::coefficients()` 同形状；
- `durations`：每个原始段一个 $\partial L/\partial T_s$。

对于 MINCO spline，可直接继续传播：

```cpp
auto hull = SplineTrajectory::toBezier(spline.polynomial(), 1, 2);
auto partial = hull.backward(control_gradients);

auto parameters = spline.backward(partial.coefficients, partial.durations);
```

这样控制点代价最终会传播到内部航点、边界状态和时间变量。

### 4.1 piece 时间梯度

若轨迹公共起始时间为 $t_0$，第 $s$ 段之前的时长前缀为
$\sum_{i<s}T_i$，第 $\ell$ 个叶片的起始时间和时长为

$$
t_{s,\ell}=t_0+\sum_{i<s}T_i+\frac{\ell}{L}T_s,
\qquad
\Delta t_{s,\ell}=\frac{T_s}{L}.
$$

`backwardPieceTimesAdd()` 接收每个 piece 对 $t_{s,\ell}$ 和
$\Delta t_{s,\ell}$ 的梯度，用一次逆序后缀累加得到所有源时长梯度和
$t_0$ 梯度。因此全局时间相关代价不需要 $O(N^2)$ 的前缀求导。

## 5. API

```cpp
#include "SplineConvexHull.hpp"

const auto& ppoly = spline.polynomial();

// 位置，每个原段二分 3 层，得到 8 个局部 Bezier 凸包。
auto position = SplineTrajectory::toBezier(ppoly, 0, 3);

// 速度的 MINVO 表示。导数次数可以取 0 到原多项式次数。
// 第三个参数同样是任意细分深度。
auto velocity = SplineTrajectory::toMINVO(ppoly, 1, 2);

for (int piece = 0; piece < position.numPieces(); ++piece)
{
    auto controls = position.pieceControls(piece);
    const auto& info = position.pieceInfo(piece);
    // info.source_segment / start_time / duration 可用于走廊分配。
}

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> grad_controls;
grad_controls.setZero(position.controls().rows(), 3);
// ...写入避障或动力学约束对控制点的梯度...
auto partial = position.backward(grad_controls);
```

`controls()` 的行按 piece-major 排列；每个 piece 连续占用
`degree()+1` 行。表示对象只用于读取控制点和反传，不承担轨迹采样。

优化循环应复用 Workspace：

```cpp
SplineTrajectory::ConvexHullWorkspace<3> hull;
hull.prepare(ppoly, {SplineTrajectory::ConvexHullBasis::MINVO, 1, 2});

// 每次迭代：不改变 topology，不重新分配。
hull.update(spline.polynomial());
hull.backwardAdd(grad_controls, grad_coeffs, grad_times);
```

`SplineOptimizer` 通过普通 Objective 的 `coefficient` 成员绑定凸包代价。
每个代价结构体实现 `operator()`，可直接返回 `double`；可能失败时返回 `EvaluationResult`。
线性总时长代价通过 `options.duration_weight` 配置，额外时长代价放在 `parameter` 成员中。
凸包代价只读取多项式，在统一样条伴随之前累加系数和独立时间偏导：

```cpp
struct ProblemObjective {
    IntegralCost integral;
    HullCost coefficient;
};

// HullCost's callable interface:
SplineTrajectory::EvaluationResult operator()(const SplineTrajectory::ParameterView<Spline>& state,
                  SplineTrajectory::CoefficientGradientView<Spline>& gradient);
```

代价工作区在准备阶段配置，回调内调用 `update()` 和 `backwardAdd()`；
后者为非 const 操作，需要独占工作区。`gradient.coefficients/durations/start_time`
都使用加法语义，优化器随后只执行一次 `backward()`。
`isPrepared()` 与 `isValid()` 分别表示已配置拓扑和本次数值有效；
`matches(ppoly, HullOptions)` 可判断是否需要重新配置。
共享转换核不依赖 DIM，相同拓扑的不同维度工作区复用同一个不可变核。
完整用法见[协议文档](optimizer.md)和[优化示例](../examples/convex_hull_optimizer.cpp)。

## 6. 优化示例

[凸包优化示例](../examples/convex_hull_optimizer.cpp) 使用 GCOPTER L-BFGS，组合分段凸走廊、
Bezier 速度与加速度控制点边界、minimum-jerk 能量和总时长。运行方法见[示例目录](../examples/README.md)。

同一 piece 的全部控制点满足凸集约束时，整段曲线满足该约束；ESDF 自由空间通常非凸，不能直接套用此结论。

[Bezier 细分图](images/bezier_subdivision_depths.png)与[MINVO 细分图](images/minvo_subdivision_depths.png)
展示不同细分深度下的凸包收缩。
