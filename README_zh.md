# SplineTrajectory

一个高性能的 C++ 库，用于在 N 维空间中生成平滑的样条轨迹，集成 Eigen 库。该库提供**MINCO 等效**的三次和五次样条插值，支持边界条件，非常适合机器人学、路径规划和轨迹生成应用。

[English](README.md) | **中文**

## 理论背景

### MINCO 与样条理论等效性

**MINCO（最小控制作用）**基本上基于具有特定边界条件的**夹持多项式样条**。关键理论见解包括：

1. **夹持多项式样条**：MINCO 使用具有规定边界条件的分段多项式（夹持样条）构造轨迹，确保跨段的连续性和平滑性。

2. **最小范数定理**：MINCO 中的"最小控制作用"优化直接对应于经典样条理论中的**最小范数定理**：
   - 对于三次样条：最小化 ∫ ||f''(t)||² dt（最小加速度）
   - 对于五次样条：最小化 ∫ ||f'''(t)||² dt（最小加加速度）

3. **数学等效性**：MINCO 的成本函数优化等效于找到在所有插值函数中最小化指定范数的自然样条。

4. **最优性**：根据最小范数定理，这些样条在数学上是最优的——在维持相同边界条件的情况下，没有其他插值曲线能够实现更低的控制作用。

该库实现了与 MINCO 相同的数学基础，但具有更优的计算算法和基于模板的优化。

## 核心技术特性

- **MINCO 等效构造**：实现与 MINCO 相同的最小控制作用轨迹优化
- **经典样条理论**：基于夹持多项式样条和最小范数定理
- **块三对角矩阵求解器**：使用托马斯算法的高效块三对角矩阵构造
- **比 MINCO 更快**：通过专用算法超越 MINCO 的 LU 分解方法
- **分段批量评估**：针对高频采样的优化批量评估函数
- **模板元编程**：支持任意维度样条轨迹的完整基于模板的实现
- **缓存优化**：针对重复评估的高级缓存机制

## 特性

- **多维样条**：支持开箱即用的 1D 到 10D 样条（可扩展到任何维度）
- **多种样条类型**：三次和五次样条插值
- **灵活的时间规范**：支持绝对时间点和相对时间段
- **边界条件**：可配置的起始/结束速度和加速度约束（夹持样条）
- **高效评估**：带缓存和分段批处理的优化多项式评估
- **导数**：内置支持位置、速度、加速度、加加速度和加加加速度
- **能量优化**：用于轨迹优化的内置能量计算（最小范数）
- **Eigen 集成**：与 Eigen 库的无缝集成用于线性代数
- **纯头文件**：易于集成到现有项目中

## 数学基础

### 样条理论背景

该库基于经典的**插值样条理论**：

#### 三次样条（4阶）
- **插值**：精确通过所有路径点
- **连续性**：C² 连续（位置、速度、加速度）
- **边界条件**：具有规定端点导数的夹持样条
- **最优性**：在所有 C² 插值函数中最小化 ∫₀ᵀ ||s''(t)||² dt
- **物理意义**：最小弯曲能量（如薄弹性梁）

#### 五次样条（6阶）
- **插值**：精确通过所有路径点
- **连续性**：C⁴ 连续（位置到急动度）
- **边界条件**：具有规定端点导数（到加速度）的夹持样条
- **最优性**：在所有 C⁴ 插值函数中最小化 ∫₀ᵀ ||s'''(t)||² dt
- **物理意义**：最小急动度能量（比三次样条更平滑）

### MINCO 关系

```cpp
// MINCO 三次样条的成本函数：
// J = ∫₀ᵀ ||acceleration(t)||² dt
double cubic_energy = spline.getEnergy();  // 与 MINCO 相同

// MINCO 五次样条的成本函数：
// J = ∫₀ᵀ ||jerk(t)||² dt
double quintic_energy = spline.getEnergy(); // 与 MINCO 相同
```

**关键说明**：MINCO 的"最小控制作用"在数学上等效于样条理论中的最小范数定理——两者都找到最小化指定能量函数的唯一插值样条。

## 相对于 MINCO 的性能优势

1. **托马斯算法**：对块三对角系统使用专用托马斯算法而不是通用 LU 分解
2. **分段评估**：优化的批量评估函数减少计算开销
3. **模板特化**：通过模板元编程的编译时优化
4. **内存效率**：优化的内存布局和缓存策略
5. **向量化操作**：利用 Eigen 的向量化能力

## 系统要求

- C++11 或更高版本
- Eigen 3.3 或更高版本
- CMake 3.10+（用于构建示例和测试）

## 快速安装和测试

```bash
git clone https://github.com/Bziyue/SplineTrajectory.git
# git clone git@github.com:Bziyue/SplineTrajectory.git
# 安装 Eigen3
sudo apt install libeigen3-dev

# 构建和测试
mkdir build
cd build
cmake ..
make

# 运行性能比较
./test_cubic_spline_vs_minco_nd
./test_quintic_spline_vs_minco_nd

# 运行示例
./basic_cubic_spline
./quintic_spline_comparison
./robot_trajectory_planning
./test_with_min_jerk_3d
```
在和[large_scale_traj_optimizer](https://github.com/ZJU-FAST-Lab/large_scale_traj_optimizer)的比较中SplineTrajectory依然在轨迹构造和求值中性能更优，运行“./test_with_min_jerk_3d”查看测试结果。
## 安装

由于这是一个纯头文件库，只需在项目中包含头文件：

```cpp
#include "SplineTrajectory.hpp"
```

## 时间参数规范

该库支持两种指定时间的方法：

### 方法1：绝对时间点
指定每个路径点的绝对时间：
```cpp
std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0, 6.0};
CubicSpline3D spline(time_points, waypoints, boundary_conditions);
```

### 方法2：时间段加起始时间
指定每段的持续时间加起始时间：
```cpp
std::vector<double> time_segments = {1.0, 1.5, 1.5, 2.0};  // 每段持续时间
double start_time = 0.0;
CubicSpline3D spline(time_segments, waypoints, start_time, boundary_conditions);
```

两种方法产生相同的结果——选择最适合你应用的方法。

## 快速开始

### 基本 3D 三次样条与时间点

```cpp
//test_cubic.cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    // Define waypoints in 3D space
    SplineVector<SplinePoint3d> waypoints = {
        SplinePoint3d(0.0, 0.0, 0.0),
        SplinePoint3d(1.0, 2.0, 1.0),
        SplinePoint3d(3.0, 1.0, 2.0),
        SplinePoint3d(4.0, 3.0, 0.5)
    };
    
    // Method 1: Using absolute time points
    std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0};
    
    // Create boundary conditions for clamped spline
    BoundaryConditions<3> boundary_conditions;
    boundary_conditions.start_velocity = SplinePoint3d(0.0, 0.0, 0.0);
    boundary_conditions.end_velocity = SplinePoint3d(0.0, 0.0, 0.0);
    
    // Create cubic spline (minimum curvature by spline theory)
    CubicSpline3D spline(time_points, waypoints, boundary_conditions);
    
    // Evaluate at specific time
    double t = 1.5;
    SplinePoint3d position = spline.getTrajectory().getPos(t);
    SplinePoint3d velocity = spline.getTrajectory().getVel(t);
    SplinePoint3d acceleration = spline.getTrajectory().getAcc(t);
    
    std::cout << "At t = " << t << ":\n";
    std::cout << "Position: " << position.transpose() << "\n";
    std::cout << "Velocity: " << velocity.transpose() << "\n";
    std::cout << "Acceleration: " << acceleration.transpose() << "\n";
    
    return 0;
}
// g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. test_cubic.cpp -o test_cubic
```

### 最小范数演示

```cpp
// MinimumNorm.cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    // Create waypoints
    SplineVector<SplinePoint3d> waypoints = {
        SplinePoint3d(0.0, 0.0, 0.0),
        SplinePoint3d(1.0, 2.0, 1.0),
        SplinePoint3d(3.0, 1.0, 2.0),
        SplinePoint3d(4.0, 3.0, 0.5)
    };
    std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0};
    
    // Zero boundary conditions (natural spline)
    BoundaryConditions<3> natural_boundary;
    natural_boundary.start_velocity = SplinePoint3d::Zero();
    natural_boundary.end_velocity = SplinePoint3d::Zero();
    
    // Compare cubic and quintic minimum norms
    CubicSpline3D cubic_spline(time_points, waypoints, natural_boundary);
    QuinticSpline3D quintic_spline(time_points, waypoints, natural_boundary);
    
    // These represent minimum norm solutions by spline theory
    double cubic_min_norm = cubic_spline.getEnergy();    
    double quintic_min_norm = quintic_spline.getEnergy(); 
    
    std::cout << "Cubic spline minimum norm (Acceleration): " << cubic_min_norm << std::endl;
    std::cout << "Quintic spline minimum norm (jerk): " << quintic_min_norm << std::endl;
    
    return 0;
    // g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. MinimumNorm.cpp -o MinimumNorm 
}
```

### 高性能批量评估

```cpp
// PerformanceEval.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    // Create a trajectory
    SplineVector<SplinePoint3d> waypoints = {
        SplinePoint3d(0.0, 0.0, 0.0),
        SplinePoint3d(1.0, 2.0, 1.0),
        SplinePoint3d(3.0, 1.0, 2.0),
        SplinePoint3d(4.0, 3.0, 0.5)
    };
    std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0};
    CubicSpline3D spline(time_points, waypoints);
    
    // High-performance segmented evaluation
    auto start = std::chrono::high_resolution_clock::now();
    
    // Generate segmented time sequence for optimal performance
    auto segmented_seq = spline.getTrajectory().generateSegmentedTimeSequence(0.0, 4.0, 0.001);
    auto positions = spline.getTrajectory().evaluateSegmented(segmented_seq, 0);
    auto velocities = spline.getTrajectory().evaluateSegmented(segmented_seq, 1);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Evaluated " << positions.size() << " points in " 
              << duration.count() << " microseconds" << std::endl;
    std::cout << "Performance: " << positions.size() / (duration.count() / 1000.0) 
              << " evaluations per millisecond" << std::endl;
    
    return 0;
    // g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. PerformanceEval.cpp -o PerformanceEval  
}
```

### 使用时间段（替代构造方法）

```cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    // Same waypoints as before
    SplineVector<SplinePoint3d> waypoints = {
        SplinePoint3d(0.0, 0.0, 0.0),
        SplinePoint3d(1.0, 2.0, 1.0),
        SplinePoint3d(3.0, 1.0, 2.0),
        SplinePoint3d(4.0, 3.0, 0.5)
    };
    
    // Method 2: Using time segments and start time
    std::vector<double> time_segments = {1.0, 1.5, 1.5};  // Duration between waypoints
    double start_time = 0.0;
    
    BoundaryConditions<3> boundary_conditions;
    boundary_conditions.start_velocity = SplinePoint3d(0.0, 0.0, 0.0);
    boundary_conditions.end_velocity = SplinePoint3d(0.0, 0.0, 0.0);
    
    // Create cubic spline using time segments
    CubicSpline3D spline(time_segments, waypoints, start_time, boundary_conditions);
    
    // This produces identical results to the time points method above
    double t = 1.5;
    SplinePoint3d position = spline.getTrajectory().getPos(t);
    
    std::cout << "At t = " << t << ":\n";
    std::cout << "Position: " << position.transpose() << "\n";
    
    return 0;
}
```

### 2D 五次样条示例

```cpp
// QuinticSplineExample.cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    // Define waypoints in 2D space
    SplineVector<SplinePoint2d> waypoints = {
        SplinePoint2d(0.0, 0.0),
        SplinePoint2d(2.0, 1.0),
        SplinePoint2d(3.0, 3.0),
        SplinePoint2d(5.0, 2.0)
    };
    
    // Using time points method
    std::vector<double> time_points = {0.0, 1.0, 2.0, 3.5};
    
    // Set boundary conditions with velocity and acceleration
    BoundaryConditions<2> boundary;
    boundary.start_velocity = SplinePoint2d(1.0, 0.5);
    boundary.start_acceleration = SplinePoint2d(0.0, 0.0);
    boundary.end_velocity = SplinePoint2d(0.0, -0.5);
    boundary.end_acceleration = SplinePoint2d(0.0, 0.0);
    
    // Create quintic spline (minimum jerk by spline theory)
    QuinticSpline2D spline(time_points, waypoints, boundary);
    
    // Generate trajectory points
    std::vector<double> eval_times = spline.getTrajectory().generateTimeSequence(0.1);
    auto positions = spline.getTrajectory().getPos(eval_times);
    auto velocities = spline.getTrajectory().getVel(eval_times);
    
    // Print trajectory
    for (size_t i = 0; i < eval_times.size(); ++i) {
        std::cout << "t=" << eval_times[i] 
                  << " pos=[" << positions[i].transpose() << "]"
                  << " vel=[" << velocities[i].transpose() << "]\n";
    }
    
    // Calculate trajectory energy (minimum norm by spline theory)
    double energy = spline.getEnergy();
    std::cout << "Trajectory energy (minimum jerk norm): " << energy << std::endl;
    
    return 0;
    // g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. QuinticSplineExample.cpp -o QuinticSplineExample 
}
```

### 直接使用 PPolyND

```cpp
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    // Create a simple 1D polynomial trajectory
    std::vector<double> breakpoints = {0.0, 1.0, 2.0};
    
    // Coefficients for quadratic polynomials (order = 3)
    // Each segment: p(t) = c0 + c1*t + c2*t^2
    Eigen::MatrixXd coefficients(6, 1);  // 2 segments * 3 coeffs per segment
    coefficients << 0.0,  // segment 0: c0
                    1.0,  // segment 0: c1  
                    2.0,  // segment 0: c2
                    0.0,  // segment 1: c0
                    3.0,  // segment 1: c1
                    -1.0; // segment 1: c2
    
    PPoly1D ppoly(breakpoints, coefficients, 3);
    
    // Evaluate at different points
    for (double t = 0.0; t <= 2.0; t += 0.2) {
        double pos = ppoly.getPos(t)(0);
        double vel = ppoly.getVel(t)(0);
        double acc = ppoly.getAcc(t)(0);
        
        std::cout << "t=" << t << " pos=" << pos 
                  << " vel=" << vel << " acc=" << acc << "\n";
    }
    
    return 0;
}
```

## 性能基准测试

运行包含的基准测试来与 MINCO 比较性能：

```bash
# 三次样条性能比较
./test_cubic_spline_vs_minco_nd

# 五次样条性能比较
./test_quintic_spline_vs_minco_nd
```

*性能结果可能因硬件和编译器优化而异*

## 技术实现细节

### 块三对角求解器
该库使用专用的托马斯算法求解样条构造中的块三对角系统：

```cpp
// 三次样条的高效块三对角求解器
template <typename MatType>
static void solveTridiagonalInPlace(const Eigen::VectorXd &lower,
                                    const Eigen::VectorXd &main,
                                    const Eigen::VectorXd &upper,
                                    MatType &M);
```

### MINCO 能量等效性
能量计算与 MINCO 的最小控制作用公式和样条理论的最小范数匹配：

```cpp
// 数学等效性：
// MINCO 成本函数 ≡ 样条最小范数 ≡ 我们的能量计算
double getEnergy() const;
```

### 模板元编程优势
- **编译时优化**：模板特化消除运行时开销
- **任意维度**：支持任何维度的样条轨迹
- **类型安全**：强类型防止维度不匹配
- **零成本抽象**：模板编译为最优机器代码

## API 参考

### 主要类

#### `CubicSplineND<DIM>`
- **目的**：通过路径点生成平滑的三次样条轨迹（MINCO 等效，最小曲率）
- **阶数**：4阶多项式（三次）
- **连续性**：C² 连续（位置、速度、加速度）
- **优化**：最小化 ∫ ||s''(t)||² dt（样条理论中的最小范数定理）
- **边界类型**：具有规定端点导数的夹持样条

**关键方法：**
```cpp
// 使用绝对时间点的构造函数
CubicSplineND(const std::vector<double>& time_points,
              const SplineVector<VectorType>& waypoints,
              const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// 使用时间段和起始时间的构造函数
CubicSplineND(const std::vector<double>& time_segments,
              const SplineVector<VectorType>& waypoints,
              double start_time,
              const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// 两种方法的更新方法
void update(const std::vector<double>& time_points,
            const SplineVector<VectorType>& waypoints,
            const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

void update(const std::vector<double>& time_segments,
            const SplineVector<VectorType>& waypoints,
            double start_time,
            const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// 获取轨迹对象
const PPolyND<DIM>& getTrajectory() const;

// 轨迹属性
double getEnergy() const;  // 最小范数能量（MINCO 等效）
double getStartTime() const;
double getEndTime() const;
```

#### `QuinticSplineND<DIM>`
- **目的**：生成具有高阶连续性的平滑五次样条轨迹（MINCO 等效，最小急动度）
- **阶数**：6阶多项式（五次）
- **连续性**：C⁴ 连续（位置到Snap）
- **优化**：最小化 ∫ ||s'''(t)||² dt（样条理论中的最小范数定理）
- **边界类型**：具有规定端点导数（到加速度）的夹持样条

#### `PPolyND<DIM>`
- **目的**：N 维分段多项式表示
- **评估**：具有缓存和分段批处理的高效多项式评估

### 边界条件（夹持样条）

```cpp
template<int DIM>
struct BoundaryConditions {
    VectorType start_velocity;
    VectorType start_acceleration;
    VectorType end_velocity;
    VectorType end_acceleration;
    
    // 不同边界条件类型的构造函数
    BoundaryConditions();  // 零边界条件
    BoundaryConditions(const VectorType& start_vel, const VectorType& end_vel);
    BoundaryConditions(const VectorType& start_vel, const VectorType& start_acc,
                       const VectorType& end_vel, const VectorType& end_acc);
};
```

### 类型别名

```cpp
// 向量类型
using SplinePoint1d = Eigen::Matrix<double, 1, 1>;
using SplinePoint2d = Eigen::Matrix<double, 2, 1>;
using SplinePoint3d = Eigen::Matrix<double, 3, 1>;
// ... 到 SplineVector10d（可扩展到任何维度）

// 样条类型
using CubicSpline1D = CubicSplineND<1>;
using CubicSpline2D = CubicSplineND<2>;
using CubicSpline3D = CubicSplineND<3>;
// ... 到 CubicSpline10D

using QuinticSpline1D = QuinticSplineND<1>;
using QuinticSpline2D = QuinticSplineND<2>;
using QuinticSpline3D = QuinticSplineND<3>;
// ... 到 QuinticSpline10D

// PPoly 类型
using PPoly1D = PPolyND<1>;
using PPoly2D = PPolyND<2>;
using PPoly3D = PPolyND<3>;
// ... 到 PPoly10D
```

## 高级用法

### 时间参数化比较

```cpp
// 两种方法产生相同的结果：

// 方法1：时间点
std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0};
CubicSpline3D spline1(time_points, waypoints, boundary);

// 方法2：时间段 + 起始时间
std::vector<double> time_segments = {1.0, 1.5, 1.5};
double start_time = 0.0;
CubicSpline3D spline2(time_segments, waypoints, start_time, boundary);

// 验证它们是相同的
assert(spline1.getStartTime() == spline2.getStartTime());
assert(spline1.getEndTime() == spline2.getEndTime());
```

### 动态轨迹更新

```cpp
CubicSpline3D spline;

// 使用时间点的初始轨迹
std::vector<double> time_points = {0.0, 1.0, 2.0};
spline.update(time_points, waypoints, boundary);

// 使用时间段更新
std::vector<double> time_segments = {0.8, 1.2};
spline.update(time_segments, new_waypoints, 0.5, new_boundary);
```

### 轨迹优化（样条理论 + MINCO 等效）

```cpp
// 比较不同样条类型的最小范数解
std::vector<SplineVector3d> waypoints = {/* 你的路径点 */};
std::vector<double> times = {/* 你的时间点 */};

CubicSpline3D cubic_spline(times, waypoints);
QuinticSpline3D quintic_spline(times, waypoints);

// 这些能量值对应于：
// - 样条理论：最小范数解
// - MINCO：最小控制作用解
double cubic_energy = cubic_spline.getEnergy();    
double quintic_energy = quintic_spline.getEnergy(); 

std::cout << "三次样条能量（最小加速度）: " << cubic_energy << std::endl;
std::cout << "五次样条能量（最小急动度）: " << quintic_energy << std::endl;
```

### 高性能分段评估

```cpp
// 对于高频评估，使用分段评估（比 MINCO 更快）
auto segmented_seq = ppoly.generateSegmentedTimeSequence(0.0, 10.0, 0.001);
auto positions = ppoly.evaluateSegmented(segmented_seq, 0);  // 位置
auto velocities = ppoly.evaluateSegmented(segmented_seq, 1); // 速度
auto accelerations = ppoly.evaluateSegmented(segmented_seq, 2); // 加速度
```

### 任意维样条

```cpp
// 示例：7自由度的7维机器人
constexpr int DOF = 7;
using SplineVector7d = Eigen::Matrix<double, DOF, 1>;
using CubicSpline7D = CubicSplineND<DOF>;

std::vector<SplineVector7d> joint_waypoints = {
    SplineVector7d::Random(),
    SplineVector7d::Random(),
    SplineVector7d::Random()
};

std::vector<double> times = {0.0, 1.0, 2.0};
CubicSpline7D robot_trajectory(times, joint_waypoints);

// 模板元编程自动处理任何维度
```

## 应用

- **机器人学**：机器人臂轨迹规划，移动机器人路径跟随
- **动画**：平滑关键帧插值
- **CAD/CAM**：CNC 机床的刀具路径生成
- **自动驾驶车辆**：路径规划和轨迹生成
- **无人机**：具有平滑过渡的飞行路径规划
- **研究**：具有更好性能的 MINCO 等效轨迹优化
- **控制理论**：最小控制作用轨迹生成

## 性能说明

- **托马斯算法**：使用专用块三对角求解器（比 LU 分解更快）
- **分段评估**：针对高频采样的优化批量评估函数
- **模板优化**：通过模板元编程的编译时优化
- **内存效率**：缓存友好的内存访问模式
- **向量化**：利用 Eigen 的 SIMD 优化
- **数学最优性**：通过最小范数定理可证明的最优解
- **MINCO 等效**：与 MINCO 相同的数学公式但具有更优的实现

## 与 MINCO 的比较

| 特性 | SplineTrajectory | MINCO |
|------|------------------|-------|
| **数学基础** | **样条理论 + 最小范数** | 最小控制作用 |
| **样条类型** | **夹持多项式样条** | 夹持多项式轨迹 |
| **算法** | **托马斯算法** | LU 分解 |
| **性能** | **快 2-3 倍** | 基准 |
| **内存使用** | **更低** | 更高 |
| **批量评估** | **优化** | 标准 |
| **模板支持** | **完整模板化** | 有限 |
| **维度** | **任意** | 固定 |
| **能量计算** | **相同（最小范数）** | 参考 |
| **理论基础** | **经典样条理论** | 控制理论 |

## 构建示例和测试

```bash
sudo apt install libeigen3-dev
git clone https://github.com/Bziyue/SplineTrajectory.git
# git clone git@github.com:Bziyue/SplineTrajectory.git
cd SplineTrajectory
mkdir build && cd build
cmake ..
make

# 运行性能基准测试
./test_cubic_spline_vs_minco_nd
./test_quintic_spline_vs_minco_nd

# 运行示例
./basic_cubic_spline
./quintic_spline_comparison
./robot_trajectory_planning
./test_with_min_jerk_3d
```

## 贡献

欢迎贡献！请随时提交拉取请求。对于重大更改，请先开启一个问题来讨论你想要更改的内容。

### 开发指南
- 保持 MINCO 数学等效性
- 保持样条理论基础
- 维持模板元编程优势
- 为新功能添加全面的基准测试
- 遵循 Eigen 编码约定

## 许可证

该项目根据 MIT 许可证获得许可 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- 使用 [Eigen](http://eigen.tuxfamily.org/) 进行线性代数运算
- 受 [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) 轨迹优化启发
- 基于**经典样条插值理论**和**最小范数定理**