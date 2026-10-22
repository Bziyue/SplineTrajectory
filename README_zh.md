# SplineTrajectory

基于 C++17 / Eigen 的纯头文件 N 维最小控制量轨迹库。
与 [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) 求解相同的夹持边界样条问题，
提供 O(N) 块三对角求解器、解析能量梯度、可组合的优化器接口，以及可微的凸包转换。

[English](README.md) | **中文**

[**在线 API 文档 · 中文 / English**](https://bziyue.github.io/SplineTrajectory/api/?lang=zh) — 直接查阅函数签名、数据维度、双语搜索和交互图示，无需构建。

## ✨ 概览

- 任意维度的三次、五次和七次最小控制量样条。
- O(N) 构造与伴随反传，覆盖时长、插值点和边界状态。
- 基于 Hamiltonian 守恒与变分原理的解析能量梯度。
- 通用 `SplineOptimizer`，支持复用变量映射与目标组件。
- 顺序采样复用导数系数。
- Bezier / MINVO 凸包转换、细分与梯度传播。

| 样条类型 | 多项式次数 | 最小化的平方积分 | MINCO 对应形式 |
| --- | ---: | --- | --- |
| 三次样条 | 3 | 加速度 | MINCO S2 |
| 五次样条 | 5 | Jerk | MINCO S3 |
| 七次样条 | 7 | Snap | MINCO S4 |

库由三个公共头文件组成：

| 头文件 | 功能 |
| --- | --- |
| `SplineTrajectory.hpp` | 样条、能量梯度、分段多项式与采样 |
| `SplineOptimizer.hpp` | 决策变量、目标组合与梯度传播 |
| `SplineConvexHull.hpp` | Bezier / MINVO 控制点、细分与伴随反传 |

## 🔁 梯度传播

`spline.backward()` 将系数与独立时长偏导映射为分段时长、全部途经点（含端点）及端点导数的梯度，
伴随过程复用构造阶段的矩阵分解。

对于采样状态代价，`SplineOptimizer` 负责数值积分、系数与时间偏导的组装，以及变量映射的反传。
目标组件提供位置、速度、加速度及更高阶导数的局部梯度，也可表达显式时间依赖。
详见[采样与梯度](docs/trajectory.md)。

## ⚡ 解析能量梯度

`spline.energyGradient()` 直接从样条解计算最小控制量能量梯度。
时长梯度由段边界信息以每段 O(1) 的代价得到；插值点梯度由连接处高阶导数的跳变得到。
能量项因此无需先组装系数偏导，再单独执行一次伴随反传。

## 🧩 SplineOptimizer

`SplineOptimizer<Spline>` 持有准备好的问题与可复用的评价工作区。
目标对象是用户定义的普通结构体，按需持有 `integral`、`coefficient` 等可调用成员。
每个代价结构体实现 `operator()`，仅启用实际提供的成员。

- 将决策变量映射为正时长、空间点与可选的辅助状态。
- 组合能量、时间、采样状态与控制点代价。
- 通过 `evaluate()` 向 L-BFGS 或其他数值求解器提供代价与梯度。

下方快速开始给出最小用法，自定义映射与目标协议见[优化器参考](docs/optimizer.md)。

## 📐 样条凸包转换

`SplineConvexHull.hpp` 将每个多项式段或其导数转换为 Bezier / MINVO 控制点，
保持所表示的曲线不变。

- `toBezier()`、`toMINVO()` 创建控制点工作区；内置 MINVO 基覆盖 0–7 次多项式。
- 二分细分收紧局部包络，同时保持轨迹不变。
- `prepare()` 准备转换数据，`update()` 与 `backwardAdd()` 复用工作区存储。
- 控制点梯度同时反传至多项式系数与分段时长。

若全部控制点位于同一凸走廊内，对应的整段曲线也位于其中；速度和加速度控制点同样可以提供
整段导数的保守界。

![Bezier 细分与 MINVO 凸包](docs/images/convex_hull_basis_comparison.png)

[转换指南](docs/convex_hull.md) · [数学推导](docs/convex_hull_basis_zh.md) ·
[优化器示例](examples/convex_hull_optimizer.cpp)

## 📈 性能基准

性能对比覆盖样条构造、梯度计算，以及 GCOPTER 和 SUPER 中的轨迹优化。
详见[测试方法与结果](docs/benchmarks.md)。

### GCOPTER 与 SUPER

每个项目在 72 个走廊与 ESDF 问题上，对比原始 MINCO 后端与 `SplineOptimizer`，
使用相同输入、代价和 L-BFGS 设置。GCOPTER 采用五次最小 jerk，SUPER 采用七次最小 snap。

| 项目 | 问题 | 段数 | 代价与梯度，MINCO / Optimizer（µs） | 求解，MINCO / Optimizer（ms） |
| --- | --- | ---: | ---: | ---: |
| GCOPTER | 走廊 | 4 | 7.204 / 5.893 | 0.336 / 0.274 |
| GCOPTER | 走廊 | 12 | 21.487 / 17.602 | 4.804 / 3.963 |
| GCOPTER | 走廊 | 32 | 57.016 / 46.872 | 30.261 / 25.056 |
| GCOPTER | ESDF | 4 | 7.218 / 6.331 | 0.829 / 0.711 |
| GCOPTER | ESDF | 12 | 21.518 / 19.003 | 3.375 / 2.949 |
| GCOPTER | ESDF | 32 | 57.131 / 49.500 | 32.517 / 29.390 |
| SUPER | 走廊 | 4 | 7.887 / 5.754 | 0.308 / 0.211 |
| SUPER | 走廊 | 12 | 23.512 / 17.029 | 8.125 / 5.412 |
| SUPER | 走廊 | 32 | 62.772 / 44.880 | 60.795 / 44.637 |
| SUPER | ESDF | 4 | 7.919 / 5.694 | 2.857 / 2.170 |
| SUPER | ESDF | 12 | 23.748 / 16.967 | 5.416 / 3.990 |
| SUPER | ESDF | 32 | 63.220 / 45.159 | 49.105 / 38.576 |

每行数据为 12 个问题的测量中位数。MINCO 为项目原始后端，Optimizer 使用
SplineTrajectory 与 `SplineOptimizer`。

### 样条内核：MINCO 与 large_scale_traj_optimizer

三张图分别比较构造、梯度反传和解析能量梯度，使用相同三维输入，覆盖 **4–10,000 段的 25 个规模**。
蓝色为 SplineTrajectory，橙色为 MINCO，绿色为 `large_scale_traj_optimizer` 的 MinJerk/MinSnap；
实线、虚线和点划线分别对应三次、五次和七次样条。

![轨迹构造速度](docs/images/construction_speed.png)

![梯度反向传播速度](docs/images/backprop_speed.png)

![解析能量梯度速度](docs/images/analytic_grad_speed.png)

## 🛠️ 重构项目

以下规划器实现使用 SplineTrajectory：

| 项目 | 原始仓库 | 重构版本 |
| --- | --- | --- |
| **EGO-Planner-v2** | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/EGO-Planner-v2) | [重构版](https://github.com/Bziyue/EGO-Planner-v2) |
| **GCOPTER** | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/GCOPTER) | [重构版](https://github.com/Bziyue/GCOPTER) |
| **SUPER** | [HKU-MaRS](https://github.com/hku-mars/SUPER) | [重构版](https://github.com/Bziyue/SUPER) |
| **DDR-opt** | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/DDR-opt) | [重构版](https://github.com/Bziyue/DDR-opt) |

## 🚀 快速开始

库本身需要 C++17 和 Eigen ≥ 3.3。仓库构建使用 CMake ≥ 3.12；测试工具还需要 Python 3。

```sh
git clone https://github.com/Bziyue/SplineTrajectory.git
cd SplineTrajectory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
(cd build && ctest --output-on-failure)
./build/examples/corridor_optimizer
./build/examples/esdf_optimizer
```

默认构建测试与示例。性能基准通过 `-DSPLINE_BUILD_BENCHMARKS=ON` 启用。
仅使用库时，可关闭 `BUILD_TESTING`、`SPLINE_BUILD_EXAMPLES` 和 `SPLINE_BUILD_BENCHMARKS`，或直接包含头文件。
添加 `-DSPLINE_NATIVE_ARCH=ON` 可为仓库内可执行程序启用 CPU 专属优化。
CMake 接入与编译选项见[构建指南](docs/building.md)。

### 物理输入与优化选择

`SplineInput<DIM>` 包含 N 个段时长、N+1 行途经点、端点导数和调用方轨迹时间轴上的
`start_time`。在 `spline.update(input)` 中，**所有途经点都是插值约束**；
在 `optimizer.prepare(input, options, objective)` 中，`options.mask` 选中的值是**优化初值**。
默认优化时长和内部途经点，端点位置与端点导数固定。若希望固定所有途经点，设置
`options.mask.waypoints.assign(input.waypoints.rows(), 0)`。

下面的完整程序构造五次样条，并评价优化目标。端点速度和加速度默认为零。
设置 `energy_weight` 和 `duration_weight` 可启用内置代价，两者默认均为零。
`NoObjective` 表示仅使用内置代价。

```cpp
#include "SplineOptimizer.hpp"
#include <cmath>
#include <iostream>

int main()
{
    using Spline = SplineTrajectory::QuinticSplineND<3>;
    SplineTrajectory::SplineInput<3> input;
    input.durations = {1.2, 1.5};
    input.waypoints.resize(3, 3);
    input.waypoints << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
    input.start_time = 2.0;
    input.boundary.start.v << 0.2, 0, 0;

    Spline spline;
    if (!spline.update(input)) return 1;
    const auto state = spline.polynomial().evaluateDerivatives<2>(2.5);
    // 等价形式：使用轨迹时间轴上的 N+1 个时间节点构建。
    Spline from_breakpoints;
    if (!from_breakpoints.updateFromBreakpoints(
            {2.0, 3.2, 4.7}, input.waypoints, input.boundary)) return 1;

    SplineTrajectory::SplineOptimizer<Spline> optimizer;
    SplineTrajectory::OptimizerOptions options;
    options.energy_weight = 1.0;   // 五次样条对应最小 jerk。
    options.duration_weight = 0.5; // 线性总时长代价。
    SplineTrajectory::NoObjective objective; // 没有自定义代价组件。
    if (!optimizer.prepare(input, options, objective)) return 1;
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient; // 首次调整尺寸，同尺寸时复用。
    const auto result = optimizer.evaluate(x, gradient, objective);
    if (!result) return 1;

    // 本例只评价初值。数值求解器会重复调用 evaluate()；
    // 求解结束后，此处应使用求解器最终返回的 x。
    if (!optimizer.update(x)) return 1;
    auto trajectory = optimizer.copyPolynomial();
    auto sampler = trajectory.makeSampler<2>();
    const auto command = sampler.evaluateDerivatives(trajectory.startTime());
    std::cout << "Initial cost: " << result.cost << '\n';
    return std::isfinite(result.cost) && gradient.allFinite() &&
           state[0].allFinite() && command[0].allFinite() ? 0 : 1;
}
```

`prepare(input, objective)` 使用默认配置；`prepare(input, options, objective)` 显式指定权重和变量选择。
准备时复制输入并分配可复用的工作区。代价持有的凸包、映射缓存和梯度缓冲在求解循环前准备，
参见[凸包示例](examples/convex_hull_optimizer.cpp)。

### 求解与轨迹交付

使用求解器最终返回的决策向量，按以下顺序交付：

1. 调用 `optimizer.update(x_final)` 并检查成功。
2. 按规划任务验收约束，决定是否接受候选。
3. 用 `auto trajectory = optimizer.copyPolynomial()` 保存独立轨迹。
4. 从保存的轨迹创建 sampler，供各控制线程采样。

`polynomial()` 提供当前候选的视图，`copyPolynomial()` 保存独立轨迹。
采样期间保持保存的轨迹存活且不变。完整优化流程见 [L-BFGS 示例](examples/README.md)。

按规划需求选择代价入口即可。自定义代价结构体返回 `double`，可能拒绝候选时返回 `EvaluationResult`；
梯度通过传入的引用视图累加。

| 规划需求 | 入口 |
| --- | --- |
| 平滑与总时长 | `energy_weight`、`duration_weight` |
| 逐点避障、速度和加速度惩罚 | `integral` |
| Bezier / MINVO 控制点约束 | `coefficient` |
| 途经点、边界和额外时长代价 | `parameter` |
| 跨多个采样点的耦合代价 | `sample` |
| 决策坐标上的特殊正则项 | `decision` |

## 🧰 示例与文档

| 示例 | 优化目标 |
| --- | --- |
| [基础优化](examples/basic_optimizer.cpp) | 最小 jerk 与总时长 |
| [走廊优化](examples/corridor_optimizer.cpp) | 分段走廊、速度与加速度 |
| [ESDF 避障](examples/esdf_optimizer.cpp) | 距离场避障与导数约束 |
| [动力学约束](examples/dynamics_optimizer.cpp) | 速度与加速度惩罚 |
| [凸包优化](examples/convex_hull_optimizer.cpp) | Bezier 走廊与导数控制点边界 |

[在线 API 文档](https://bziyue.github.io/SplineTrajectory/api/?lang=zh) ·
[采样与梯度](docs/trajectory.md) · [优化器参考](docs/optimizer.md) ·
[凸包](docs/convex_hull.md) · [性能基准](docs/benchmarks.md) · [数值说明](docs/numerical_notes.md)

## 🔭 未来计划

- [x] 与 MINCO 等价的梯度传播
- [x] 夹持边界七次样条支持
- [x] 样条凸包转换支持
- [ ] N 维非均匀 B 样条支持

## 📄 许可证

公共头文件采用 [MIT 许可证](LICENSE)，随附依赖保留各自的[第三方许可证](THIRD_PARTY_NOTICES.md)。

## 🙏 致谢

- [Eigen](https://eigen.tuxfamily.org/) 提供线性代数基础。
- [MINCO / GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER) 与 [SUPER](https://github.com/hku-mars/SUPER) 提供参考实现。
- 经典样条插值理论与最小范数方法。
