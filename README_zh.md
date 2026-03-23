# SplineTrajectory

一个纯头文件的 C++ 库，用于生成平滑的 N 维最小控制量样条轨迹。它与 [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) 在数学上等价，并在实现上采用 O(N) 的块三对角求解结构，同时提供全控制参数梯度传播、能量解析梯度以及面向组件化优化的接口。

[English](README.md) | **中文**

## ✨ 概览

核心内容包括：

- 三次、五次、七次最小控制量样条
- 基于块三对角求解器的 O(N) 轨迹构造
- 面向时间段、空间点和边界状态的伴随法梯度传播
- 基于哈密顿量守恒和变分法推导的能量解析梯度
- 一个通用的 `SplineOptimizer`，支持时间映射、空间映射、代价接口和附加优化变量的组件化组织

样条类型与 MINCO 的对应关系如下：

| 样条类型 | 多项式阶次 | 最小化目标 | MINCO 对应 |
| --- | --- | --- | --- |
| 三次样条 | 3 阶 | 加速度 | MINCO S2 |
| 五次样条 | 5 阶 | Jerk | MINCO S3 |
| 七次样条 | 7 阶 | Snap | MINCO S4 |

## 🔁 梯度传播

🔁 提供了面向主要控制参数的伴随法梯度传播机制，覆盖：

- 时间段 `T`
- 空间点 `P`
- 起终点边界状态，包括速度、加速度、jerk 等导数条件

在底层接口上，`propagateGrad(gdC, gdT)` 可以将定义在多项式系数和时间上的梯度直接反传回这些控制参数。

在实际优化任务中，更常用的是 `SplineOptimizer` 这一层。对于大多数积分型代价，可以直接在采样状态上定义代价，由优化器内部完成：

- 数值积分
- 对系数 `C` 的梯度组装
- 对时间 `T` 的梯度计算
- 向时间变量、空间点、边界状态和附加变量继续反传

很多情况下，用户只需要提供采样点上 `p / v / a / j` 的局部梯度，必要时再补充 `s` 和显式全局时间项。

## ⚡ 能量解析梯度

⚡ 对于最小加速度、最小 jerk、最小 snap 这类纯能量项，还提供了一条独立的解析梯度路径。它基于哈密顿量守恒和变分法推导，可直接得到控制参数上的全导数。

主要结果包括：

- 时间梯度可以通过分段边界信息在 O(1) 时间内直接获得
- 空间点梯度可以通过连接点处最高阶导数的跳变量直接构造

这条路径不需要先计算能量对 `C` 和 `T` 的偏导，再额外走一遍通用伴随传播。因此在纯能量优化场景下，它比通用传播路径更直接，也更快。

## 🧩 SplineOptimizer

🧩 `SplineOptimizer` 是建立在样条与梯度传播机制之上的通用优化器，采用组件化的接口组织方式，主要包括：

- `TimeMap`，用于将无约束变量映射到正时间段
- `SpatialMap`，用于将无约束变量映射到物理空间点、走廊或 polytope 约束状态
- 积分型、离散路点型、整条轨迹型代价接口
- `AuxiliaryStateMap`，用于表达共享时间变量、拼接状态或其他低维附加优化变量
- 面向 L-BFGS 等优化器的统一目标函数组织方式

这套结构的重点在于让代价逻辑尽量停留在物理状态空间内，而将时间映射、空间映射、积分和梯度组装收进优化器内部处理。

当前推荐的最小调用方式是“显式 workspace + 显式状态返回”：

```cpp
#include "SplineOptimizer.hpp"

using Optimizer = SplineTrajectory::SplineOptimizer<3>;
using Waypoints = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct TimeCost
{
    double operator()(const std::vector<double>& Ts, Eigen::VectorXd& grad) const
    {
        grad = Eigen::Map<const Eigen::VectorXd>(Ts.data(), static_cast<Eigen::Index>(Ts.size()));
        return 0.5 * grad.squaredNorm();
    }
};

struct IntegralCost
{
    double operator()(double,
                      double,
                      int,
                      int,
                      const Eigen::Vector3d&,
                      const Eigen::Vector3d& v,
                      const Eigen::Vector3d&,
                      const Eigen::Vector3d&,
                      const Eigen::Vector3d&,
                      Eigen::Vector3d& gp,
                      Eigen::Vector3d& gv,
                      Eigen::Vector3d& ga,
                      Eigen::Vector3d& gj,
                      Eigen::Vector3d& gs,
                      double& gt) const
    {
        gp.setZero();
        gv = v;
        ga.setZero();
        gj.setZero();
        gs.setZero();
        gt = 0.0;
        return 0.5 * v.squaredNorm();
    }
};

Optimizer optimizer;
Optimizer::Workspace workspace;

std::vector<double> durations{1.0, 1.2};
Waypoints waypoints(3, 3);
waypoints << 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0,
             2.0, 1.0, 0.0;

SplineTrajectory::BoundaryConditions<3> bc;

auto init_status = optimizer.setInitState(durations, waypoints, 0.0, bc);
if (!init_status)
{
    std::cerr << init_status.message << std::endl;
    return;
}

TimeCost time_cost;
IntegralCost integral_cost;
auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost, workspace);

Eigen::VectorXd x = optimizer.generateInitialGuess();
Eigen::VectorXd grad;

auto eval_result = optimizer.evaluate(x, grad, spec);
if (!eval_result)
{
    std::cerr << eval_result.message << std::endl;
    return;
}

std::cout << "cost = " << eval_result.cost << std::endl;
const auto& spline = optimizer.getWorkingSpline(workspace);
```

## 📌 与 MINCO 的对比

SplineTrajectory 与 MINCO 处理的是同一类样条问题，差异主要体现在实现效率和接口组织上：

| 特性 | SplineTrajectory | MINCO |
| --- | --- | --- |
| 算法 | 块三对角求解器（Thomas） | LU 分解 |
| 构造速度 | 更快 | 基准 |
| 梯度传播 | 更快（复用分解缓存） | 基准 |
| 能量梯度 | 解析 + 伴随 | 仅伴随 |
| 边界状态梯度 | 包含 | 未提供 |
| 优化器代价接口 | 局部状态梯度 -> 自动组装 `dL/dC`、`dL/dT` | 未内置 |
| 空间维度 | 任意（模板化） | 固定 3D |
| 求值方式 | 分段批量 + 系数缓存 | 标准求值 |

SplineTrajectory 在轨迹生成和求值方面也优于 [large_scale_traj_optimizer](https://github.com/ZJU-FAST-Lab/large_scale_traj_optimizer)。

## 📈 性能基准

📈 轨迹构造速度（2 至 10^5 段）

![轨迹构造速度](docs/images/construction_speed.png)

📈 梯度反向传播速度（2 至 10^5 段）

![梯度反向传播速度](docs/images/backprop_speed.png)

📈 能量解析梯度计算速度（2 至 10^5 段）

![能量解析梯度计算速度](docs/images/analytic_grad_speed.png)

## 🛠️ 重构项目

🛠️ 以下规划器已经基于 SplineTrajectory 与 SplineOptimizer 完成重构：

| 项目 | 简介 | 原始仓库 | 重构版 |
| --- | --- | --- | --- |
| **EGO-Planner-v2** | 无人机集群规划（Science Robotics） | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/EGO-Planner-v2) | [重构版](https://github.com/Bziyue/EGO-Planner-v2) |
| **GCOPTER** | 几何约束多旋翼轨迹优化（IEEE T-RO） | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/GCOPTER) | [重构版](https://github.com/Bziyue/GCOPTER) |
| **SUPER** | 安全高速 MAV 导航（Science Robotics） | [HKU-MaRS](https://github.com/hku-mars/SUPER) | [重构版](https://github.com/Bziyue/SUPER) |
| **DDR-opt** | 差速驱动机器人通用轨迹优化（IEEE T-ASE） | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/DDR-opt) | [重构版](https://github.com/Bziyue/DDR-opt) |

这些重构版采用的是同一套思路：代价项主要写在物理状态空间里，而时间映射、空间映射、积分组装和梯度传播交给共享的优化框架处理。DDR-opt 还进一步使用了整条轨迹代价接口和附加状态变量接口来表达耦合优化问题。

## 📦 环境要求

- C++17 或更高版本
- Eigen 3.3 或更高版本
- CMake 3.10+，用于示例和测试

## 🚀 快速开始

🚀

```bash
git clone https://github.com/Bziyue/SplineTrajectory.git
cd SplineTrajectory

# 如有需要，先安装 Eigen3
sudo apt install libeigen3-dev

# 编译
mkdir build && cd build
cmake ..
make

# 性能测试
./test_cubic_spline_vs_minco_nd
./test_quintic_spline_vs_minco_nd
./test_septic_spline_vs_minco_nd

# 梯度测试
./test_Grad
./test_cost_grad
./test_bc_grad
```

如果需要完整的运动规划工具链，可以参考 [ST-opt-tools](https://github.com/MarineRock10/ST-opt-tools)，其中集成了 ESDF 建图、A* 路径规划和 L-BFGS 轨迹优化。

## 🧰 主要功能

### 🛤️ 轨迹构造

- 通过时间点或时间段配合边界条件构造样条
- 使用默认零导数边界，或完全自定义的夹持边界
- 通过 `update()` 高效更新已有样条

### 📍 求值

- 单点、批量和区间求值
- 分段批量求值与系数缓存
- 导数轨迹提取

### 🔁 梯度

- `propagateGrad(gdC, gdT)`，用于定义在系数和时间上的梯度
- `SplineOptimizer::evaluate(...)`，用于采样轨迹状态上的代价，返回 `EvaluationResult { ok, code, cost, message }`
- `getEnergyGradInnerP()` 和 `getEnergyGradTimes()`，用于能量解析梯度
- `getEnergyPartialGradByCoeffs()` 和 `getEnergyPartialGradByTimes()`，用于能量偏导
- 起终点导数条件对应的边界状态梯度

### 📐 轨迹分析

- 全程或区间轨迹长度
- 任意时刻的累积弧长
- 能量计算

## ✅ 梯度验证

✅ 仓库中包含多组梯度验证测试：

- `test_Grad`，验证伴随传播与解析能量梯度的一致性
- `test_cost_grad`，验证自动积分与传播后的任意代价梯度与有限差分一致
- `test_bc_grad`，验证边界状态梯度

这些测试用于确认伴随路径、解析路径与数值检查之间的一致性。

## 📚 文档

`examples/` 目录中包含构造、求值、梯度传播和优化器接入示例。
优化器接口协议和接入约定见 [`include/SplineOptimizerProtocols.md`](include/SplineOptimizerProtocols.md)。

## 🔭 未来计划

- [x] 与 MINCO 等价的梯度传播
- [x] 夹持七次样条支持
- [ ] N 维非均匀 B 样条支持
- [ ] 夹持样条到非均匀 B 样条的精确转换

## 📄 许可证

MIT License，详见 [LICENSE](LICENSE)。

## 🙏 致谢

- [Eigen](http://eigen.tuxfamily.org/) 提供线性代数支持
- [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) 提供问题背景与对照
- 经典样条插值理论和最小范数视角
