# SplineTrajectory

[English](README.md) · [文档](docs/README.md) · [示例](examples/README.md)

基于 C++17 / Eigen 的纯头文件 N 维最小控制量轨迹库。
三次、五次和七次样条分别最小化加速度、jerk 和 snap 的平方积分，支持线性复杂度的构造与解析梯度。
通过组合目标函数，接入数值求解器完成轨迹优化。

| 头文件 | 功能 |
| --- | --- |
| `SplineTrajectory.hpp` | 样条、能量梯度、分段多项式与顺序采样 |
| `SplineOptimizer.hpp` | 决策变量、目标组合与梯度传播 |
| `SplineConvexHull.hpp` | Bezier / MINVO 控制点、细分与反传 |

## 构建与测试

需要支持 C++17 的编译器、CMake ≥ 3.12、Eigen ≥ 3.3；测试工具还需要 Python 3。

```sh
git clone https://github.com/Bziyue/SplineTrajectory.git
cd SplineTrajectory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 3
(cd build && ctest --output-on-failure)
./build/examples/corridor_optimizer
./build/examples/esdf_optimizer
```

默认构建测试与示例。性能基准通过 `-DSPLINE_BUILD_BENCHMARKS=ON` 启用。
仅使用库时，可关闭 `BUILD_TESTING`、`SPLINE_BUILD_EXAMPLES` 和 `SPLINE_BUILD_BENCHMARKS`，或直接包含头文件。

## 创建样条

```cpp
#include "SplineTrajectory.hpp"

using Spline = SplineTrajectory::QuinticSplineND<3>;
Spline::CoefficientMatrix points(3, 3);
points << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
std::vector<double> durations{1.2, 1.5};
Spline spline(durations, points, 0.0);
if (!spline.isValid()) throw std::runtime_error("Invalid spline");
auto state = spline.polynomial().evaluateDerivatives<2>(0.5); // position, velocity, acceleration
```

## 创建优化器

沿用上面的插值点和时长：

```cpp
#include "SplineOptimizer.hpp"

SplineTrajectory::SplineOptimizer<Spline> optimizer;
SplineTrajectory::SplineProblem<Spline> problem;
problem.waypoints = points;
problem.durations = durations;
if (!optimizer.prepare(problem)) throw std::runtime_error("Invalid problem");
auto x = optimizer.initialGuess();
Eigen::VectorXd gradient(x.size());
SplineTrajectory::NoObjective objective; // minimum-jerk energy only
const double cost = optimizer.costAndGradient(x, gradient, objective);
```

在目标结构体中按需声明 `duration`、`integral` 或 `coefficient` 成员，即可组合时间、状态与控制点代价。
外部求解器调用 `costAndGradient`；求解结束后，先调用 `updateAccepted(x)` 再读取轨迹。
完整接入见 [GCOPTER L-BFGS 示例](examples/README.md)。

## 示例与指南

| 示例 | 优化目标 |
| --- | --- |
| [基础优化](examples/basic_optimizer.cpp) | 最小 jerk 与总时长 |
| [走廊优化](examples/corridor_optimizer.cpp) | 分段走廊、速度与加速度 |
| [ESDF 避障](examples/esdf_optimizer.cpp) | 距离场避障与导数约束 |
| [动力学约束](examples/dynamics_optimizer.cpp) | 速度与加速度惩罚 |
| [凸包优化](examples/convex_hull_optimizer.cpp) | Bezier 走廊与导数控制点边界 |

[采样与梯度](docs/trajectory.md) · [优化器参考](docs/optimizer.md) ·
[凸包](docs/convex_hull.md) · [性能基准](docs/benchmarks.md) · [数值说明](docs/numerical_notes.md)

公共头文件采用 [MIT 许可证](LICENSE)，随附依赖保留各自的[第三方许可证](THIRD_PARTY_NOTICES.md)。
