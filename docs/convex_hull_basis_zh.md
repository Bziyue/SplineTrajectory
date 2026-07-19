# PPolyND 的 Bezier / MINVO 凸包表示与梯度

本文说明 `include/ConvexHullBasis.hpp` 使用的数学变换、Bezier 细分、
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

`Kernel` 是不可变共享对象，按“基、源系数数、导数阶、细分深度”缓存
$K$、$K^T$、下降阶乘和叶片起始比例。相同拓扑的多个表示或多个优化上下文
复用同一份核。

`resetTopology()` 只在拓扑改变时分配输出和临时工作区，并按真实字节数
检查内存预算；`update()` 只更新系数、时长幂、控制点和 piece 元数据。
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
等多个凸包代价可先相加，最后只调用一次 spline `propagateGrad()`。

`backward()` 返回：

- `coefficients`：和输入 `PPolyND::getCoefficients()` 同形状；
- `durations`：每个原始段一个 $\partial L/\partial T_s$。

对于 MINCO spline，可直接继续传播：

```cpp
auto hull = SplineTrajectory::toBezier(spline.getPPoly(), 1, 2);
auto partial = hull.backward(control_gradients);

QuinticSplineND<3>::Gradients parameters;
spline.propagateGrad(partial.coefficients, partial.durations, parameters);
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
#include "ConvexHullBasis.hpp"

const auto& ppoly = spline.getPPoly();

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
SplineTrajectory::ConvexHullRepresentation<3> hull;
hull.resetTopology(ppoly, SplineTrajectory::ConvexHullBasis::MINVO, 1, 2);

// 每次迭代：不改变 topology，不重新分配。
hull.update(spline.getPPoly());
hull.backwardAdd(grad_controls, grad_coeffs, grad_times);
```

`SplineOptimizer::EvaluateSpec::withCoefficientCost()` 提供了
allocation-free 的非拥有型回调。它在统一 spline 伴随传播之前运行：

```cpp
auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost)
    .withCoefficientCost(hull_cost);
```

回调签名为：

```cpp
double operator()(Spline& spline,
                  const std::vector<double>& times,
                  double start_time,
                  Matrix& grad_coeffs,
                  Eigen::VectorXd& grad_times,
                  double& grad_start_time) const;
```

回调必须使用加法语义。优化器随后只执行一次 `propagateGrad()`。

## 6. 统一的 3/5/7 次样条名

三个实际使用的最小导数样条现在可统一写成：

```cpp
SplineTrajectory::MinDerivativeSplineND<DIM, S>
```

- `S=2`：cubic / minimum acceleration；
- `S=3`：quintic / minimum jerk；
- `S=4`：septic / minimum snap。

这是编译期选择器，内部仍使用已经稳定的三个专用线性求解与梯度实现，
没有运行时分支；旧的 `CubicSplineND`、`QuinticSplineND`、
`SepticSplineND` API 完全保留。三个专用实现的系数求解均已改成原位写入
持久化矩阵，`Gradients` 提供固定拓扑的 `resetTopology()`/`setZero()`，
因此统一名字不会牺牲现有专用求解器的热路径性能。

## 7. 优化示例与验证

`examples/convex_hull_optimization_demo.cpp` 使用 `SplineOptimizer` 和一个
紧凑 L-BFGS 驱动器，同时优化：

- minimum-jerk 能量；
- ESDF 圆障碍或分段凸走廊；
- Bezier 速度与加速度控制点上界；
- 总时间。

ESDF 的自由空间是非凸集合，因此“每个控制点 ESDF 为正”本身不是严格的
整段安全证明；示例采用二层细分和小的控制点安全缓冲，并额外密集采样
验证。凸走廊不同：同一 Bezier piece 的全部控制点位于凸走廊内时，该
piece 的整条曲线必定位于走廊内。速度和加速度的欧氏球也是凸集，因此
对应导数控制点全部满足范数上界即可界定整段导数。

核心测试 `test_convex_hull_basis.cpp` 覆盖：

- 0 到 7 次 Bezier/MINVO 基；
- 每个合法导数阶；
- 三层 Bezier 与 MINVO 细分后的曲线重构；
- 基函数非负性和单位分解；
- 系数梯度、时长梯度的随机中心差分；
- `backwardAdd()` 的加法语义和共享 Kernel；
- piece 起始时间、piece 时长到源时长/公共起始时间的有限差分；
- 优化器决策变量层的 `CoefficientCost` 梯度检查；
- 禁止 Eigen 堆分配后的样条更新、凸包更新、反向与 MINCO 伴随传播。

运行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_convex_hull_basis convex_hull_optimization_demo
./build/test_convex_hull_basis
./build/convex_hull_optimization_demo convex_hull_demo_output
/home/beiyue/venvs/mate/bin/python \
  examples/plot_convex_hull_optimization.py --skip-run
```

可视化输出位于：

- `docs/images/convex_hull_optimization.png`
- `docs/images/convex_hull_basis_comparison.png`

转换、导数、反向传播和细分的微基准见
[`convex_hull_performance_zh.md`](convex_hull_performance_zh.md)。
