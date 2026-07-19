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
`getA_MV.m` 的行顺序一次性生成。因而精确 MINVO 支持 0 到 7 次多项式，
覆盖本库的 cubic、quintic、septic 轨迹及它们的所有导数。

运行时只做小型稠密矩阵乘法，不读取 MAT 文件，也不依赖 MATLAB、SciPy
或动态矩阵求逆。

## 3. Bezier 任意层二分

`subdivision_depth=s` 表示连续执行 $s$ 层二分，每个原始多项式段产生
$2^s$ 个 Bezier 段。二分采用 $u=1/2$ 的 de Casteljau 算法，它不改变
曲线，只缩小每个局部控制凸包。

左、右子段控制点分别写成

$$
V^L=LV,\qquad V^R=RV,
$$

其中

$$
L_{ij}=
\begin{cases}
\binom{i}{j}2^{-i},&j\le i,\\
0,&j>i,
\end{cases}
$$

$$
R_{ij}=
\begin{cases}
\binom{m-i}{j-i}2^{-(m-i)},&j\ge i,\\
0,&j<i.
\end{cases}
$$

代码逐层应用这两个固定矩阵。相较于对子区间重新展开幂多项式，这种方法
数值更稳定，并且天然提供高效的伴随反传。

## 4. 控制点梯度反传

设损失对控制点的梯度为 $G_V=\partial L/\partial V$。未细分时，

$$
G_C=M_s^TG_V.
$$

时长的偏导要求把局部幂系数固定。由于第 $\ell$ 个归一化导数系数仅含
$T_s^\ell$，

$$
\frac{\partial \hat c_{s,\ell}}{\partial T_s}
=\frac{\ell}{T_s}\hat c_{s,\ell},
$$

所以

$$
\frac{\partial L}{\partial T_s}
=\left\langle
G_V,\,
B_m\frac{\partial N_r(T_s)}{\partial T_s}C_s
\right\rangle_F.
$$

实现只复用一份基矩阵，并利用稀疏归一化结构逐行反传，不构造每段完整
变换矩阵、三阶张量，也不做有限差分。

对细分 Bezier，叶子控制点梯度按树反向合并：

$$
G_{\mathrm{parent}}=L^TG_{\mathrm{left}}+R^TG_{\mathrm{right}}.
$$

合并到原段控制点后再执行上述 $M_s^T$ 反传。计算量和生成的控制点数
线性相关，且不保存每个叶子到原幂系数的完整矩阵。

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

## 5. API

```cpp
#include "ConvexHullBasis.hpp"

const auto& ppoly = spline.getPPoly();

// 位置，每个原段二分 3 层，得到 8 个局部 Bezier 凸包。
auto position = SplineTrajectory::toBezier(ppoly, 0, 3);

// 速度的 MINVO 表示。导数次数可以取 0 到原多项式次数。
auto velocity = SplineTrajectory::toMINVO(ppoly, 1);

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

## 6. 优化示例与验证

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
- 三层 Bezier 二分后的曲线重构；
- 基函数非负性和单位分解；
- 系数梯度、时长梯度的随机中心差分。

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
