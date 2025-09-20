# SplineTrajectory

SplineTrajectory 是一个高性能、纯头文件的 C++ 库，用于生成平滑的N维样条轨迹。该库提供了与 **MINCO 等效** 的三次、五次和七次样条插值，并支持边界条件，是机器人学、路径规划和轨迹生成应用的理想选择。

[English](README.md) | **中文**

## 核心特性

- **与MINCO等效**: 实现与MINCO相同的最小化加速度、加加速度(Jerk)和加加加速度(Snap)轨迹。   
- **高性能**: 使用专门的 **块三对角矩阵求解器** (托马斯算法) 代替通用的LU分解，性能超越传统方法。
- **基于模板**: 完全模板化，支持 **任意维度** (1D到ND)，并进行编译时优化。
- **灵活高效**: 支持多种时间规格，优化的分段批量求值，并提供导数 (速度、加速度、Jerk、Snap)。
- **集成Eigen**: 无缝使用 Eigen 库进行所有线性代数运算。
- **纯头文件**: 只需包含头文件即可轻松集成到任何项目中。

## 环境要求

- C++11 或更高版本
- Eigen 3.3 或更高版本
- CMake 3.10+ (用于编译示例和测试)

## 快速开始

```bash
git clone https://github.com/Bziyue/SplineTrajectory.git
# git clone git@github.com:Bziyue/SplineTrajectory.git

cd SplineTrajectory

# Install Eigen3 (if not installed)
sudo apt install libeigen3-dev

# Build and test
mkdir build && cd build
cmake ..
make

# Run performance comparisons
./test_cubic_spline_vs_minco_nd
./test_quintic_spline_vs_minco_nd
./test_septic_spline_vs_minco_nd

# Run examples
./basic_cubic_spline
./quintic_spline_comparison
./robot_trajectory_planning
./test_with_min_jerk_3d
./test_with_min_snap_3d
```
SplineTrajectory 在轨迹生成和求值方面也优于 [large_scale_traj_optimizer](https://github.com/ZJU-FAST-Lab/large_scale_traj_optimizer) 。查看测试结果，请运行 `./test_with_min_jerk_3d`。

如果需要一个集成了此库的完整运动规划工具包，请查看 [ST-opt-tools](https://github.com/MarineRock10/ST-opt-tools)。这是一个运动规划工具包，具有 ESDF 建图、A* 路径规划和与 SplineTrajectory 库集成的 L-BFGS 轨迹优化功能。

## 与MINCO的对比
该库在数学上与MINCO等效，但采用了更高效的算法实现。
| 特性         | SplineTrajectory                             | MINCO                      |
| --------------- | -------------------------------------------- | -------------------------- |
| **算法**   | **追赶法** (块三对角矩阵求解)     | LU分解          |
| **性能** | 更快的轨迹生成和求值           | 基准                   |
| **核心理论** |夹持样条（最小范数定理）       | 最小化控制能量     |
| **灵活性** | 完全模板化，支持**任意维度** | 固定为三维 |
| **求值**  | 优化的分段批量与系数缓存机制求值         | 标准线性查找求值        |

## 样条类型与最小能量对应
该库通过最小化一个具有明确物理意义的目标函数（即某阶导数范数的平方积分），来生成最优的样条轨迹。
| Spline Type             | MINCO Equivalent     | 
| ----------------------- | -------------------- | 
| **三次样条**  | 最小化加速度 | 
| **五次样条** | 最小化加加速度         | 
| **七次样条**| 最小化加加加速度          |

---
## 使用示例
这是一个创建和评估3D轨迹的简明示例。
```cpp
#include "SplineTrajectory.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>

int main() {
    using namespace SplineTrajectory;

    std::cout << "=== SplineTrajectory 全接口使用示例 ===" << std::endl;

    // 1. 定义3D航点和边界条件
    SplineVector<SplinePoint3d> waypoints = {
        {0.0, 0.0, 0.0}, {1.0, 2.0, 1.0}, {3.0, 1.0, 2.0}, {4.0, 3.0, 0.5}, {5.0, 0.5, 1.5}
    };
    BoundaryConditions<3> boundary; //默认速度、加速度、加加速度为0
    // 定义详细的边界条件（包含速度、加速度、Jerk）
    // 或者 BoundaryConditions<3> boundary(SplinePoint3d(0.1, 0.0, 0.0),SplinePoint3d(0.2, 0.0, 0.1)); 默认加速度和加加速度为0
    boundary.start_velocity = SplinePoint3d(0.1, 0.0, 0.0); // 三次样条只会用到速度作为边界条件 
    boundary.end_velocity = SplinePoint3d(0.2, 0.0, 0.1);
    boundary.start_acceleration = SplinePoint3d(0.0, 0.0, 0.0);// 五次样条使用速度、加速度
    boundary.end_acceleration = SplinePoint3d(0.0, 0.0, 0.0);
    boundary.start_jerk = SplinePoint3d(0.0, 0.0, 0.0); // 七次样条使用速度、加速度、加加速度
    boundary.end_jerk = SplinePoint3d(0.0, 0.0, 0.0);

    std::cout << "\n--- 构造方式对比 ---" << std::endl;
    
    // 2. 使用时间点构造样条（多种样条类型对比）
    std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0, 6.0};
    CubicSpline3D cubic_from_points(time_points, waypoints, boundary);
    QuinticSpline3D quintic_from_points(time_points, waypoints, boundary);
    SepticSpline3D septic_from_points(time_points, waypoints, boundary);

    // 使用时间段构造样条
    std::vector<double> time_segments = {1.0, 1.5, 1.5, 2.0}; // 段时长
    double start_time = 0.0;
    CubicSpline3D cubic_from_segments(time_segments, waypoints, start_time, boundary);
    QuinticSpline3D quintic_from_segments(time_segments, waypoints, start_time, boundary);
    SepticSpline3D septic_from_segments(time_segments, waypoints, start_time, boundary);

    // 3. 更新操作示例
    std::cout << "\n--- 更新操作 ---" << std::endl;
    CubicSpline3D spline_for_update;
    std::cout << "初始化状态: " << spline_for_update.isInitialized() << std::endl;
    
    // 使用时间点更新
    spline_for_update.update(time_points, waypoints, boundary);
    std::cout << "时间点更新后状态: " << spline_for_update.isInitialized() << std::endl;
    
    // 使用时间段更新
    spline_for_update.update(time_segments, waypoints, start_time, boundary);
    std::cout << "时间段更新后状态: " << spline_for_update.isInitialized() << std::endl;

    // 4. 获取基本信息
    auto& trajectory = cubic_from_points.getTrajectory();
    std::cout << "\n--- 基本信息 ---" << std::endl;
    std::cout << "样条维度: " << cubic_from_points.getDimension() << std::endl;
    std::cout << "起始时间: " << cubic_from_points.getStartTime() << std::endl;
    std::cout << "结束时间: " << cubic_from_points.getEndTime() << std::endl;
    std::cout << "轨迹时长: " << cubic_from_points.getDuration() << std::endl;
    std::cout << "航点数量: " << cubic_from_points.getNumPoints() << std::endl;
    std::cout << "样条段数: " << cubic_from_points.getNumSegments() << std::endl;
    std::cout << "样条阶数: " << trajectory.getOrder() << std::endl;

    // 5. 单点求值 - evaluate通用接口
    std::cout << "\n--- 单点evaluate通用求值 ---" << std::endl;
    double t_eval = 2.5;
    auto pos_eval = trajectory.evaluate(t_eval, 0);      // 位置（0阶导数）
    auto vel_eval = trajectory.evaluate(t_eval, 1);      // 速度（1阶导数）
    auto acc_eval = trajectory.evaluate(t_eval, 2);      // 加速度（2阶导数）
    auto jerk_eval = trajectory.evaluate(t_eval, 3);     // Jerk（3阶导数）
    auto snap_eval = trajectory.evaluate(t_eval, 4);     // Snap（4阶导数）
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "t=" << t_eval << " 位置: " << pos_eval.transpose() << std::endl;
    std::cout << "t=" << t_eval << " 速度: " << vel_eval.transpose() << std::endl;
    std::cout << "t=" << t_eval << " 加速度: " << acc_eval.transpose() << std::endl;

    // 6. 单点求值 - get系列函数重命名接口
    std::cout << "\n--- 单点get系列求值 ---" << std::endl;
    auto pos_get = trajectory.getPos(t_eval);
    auto vel_get = trajectory.getVel(t_eval);
    auto acc_get = trajectory.getAcc(t_eval);
    auto jerk_get = trajectory.getJerk(t_eval);
    auto snap_get = trajectory.getSnap(t_eval);
    
    std::cout << "get接口 t=" << t_eval << " 位置: " << pos_get.transpose() << std::endl;
    std::cout << "get接口 t=" << t_eval << " 速度: " << vel_get.transpose() << std::endl;

    // 7. 批量求值 - 传入vector<double>
    std::cout << "\n--- 批量求值vector<double> ---" << std::endl;
    std::vector<double> eval_times = {0.5, 1.5, 2.5, 3.5, 5.0};
    
    // evaluate通用接口批量求值
    auto pos_batch_eval = trajectory.evaluate(eval_times, 0);
    auto vel_batch_eval = trajectory.evaluate(eval_times, 1);
    
    // get系列批量求值
    auto pos_batch_get = trajectory.getPos(eval_times);
    auto vel_batch_get = trajectory.getVel(eval_times);
    auto acc_batch_get = trajectory.getAcc(eval_times);
    auto jerk_batch_get = trajectory.getJerk(eval_times);
    auto snap_batch_get = trajectory.getSnap(eval_times);
    
    std::cout << "批量求值点数: " << pos_batch_get.size() << std::endl;
    for (size_t i = 0; i < eval_times.size(); ++i) {
        std::cout << "t=" << eval_times[i] << " 位置: " << pos_batch_get[i].transpose() << std::endl;
    }

    // 8. 时间范围求值（包含结束时间） - evaluate带范围
    std::cout << "\n--- 时间范围evaluate求值 ---" << std::endl;
    double start_t = 0.0, end_t = 6.0, dt = 0.5;
    auto pos_range_eval = trajectory.evaluate(start_t, end_t, dt, 0);
    auto vel_range_eval = trajectory.evaluate(start_t, end_t, dt, 1);
    
    std::cout << "范围求值[" << start_t << ", " << end_t << "], dt=" << dt 
              << ", 点数: " << pos_range_eval.size() << std::endl;

    // 9. 时间范围求值（包含结束时间） - get系列带范围
    std::cout << "\n--- 时间范围get系列求值 ---" << std::endl;
    auto pos_range_get = trajectory.getPos(start_t, end_t, dt);
    auto vel_range_get = trajectory.getVel(start_t, end_t, dt);
    auto acc_range_get = trajectory.getAcc(start_t, end_t, dt);
    auto jerk_range_get = trajectory.getJerk(start_t, end_t, dt);
    auto snap_range_get = trajectory.getSnap(start_t, end_t, dt);
    
    std::cout << "get系列范围求值点数: " << pos_range_get.size() << std::endl;

    // 10. 生成时间序列（包含轨迹结束时间）
    std::cout << "\n--- 生成时间序列 ---" << std::endl;
    // 注意：generateTimeSequence会包含轨迹的结束时间点
    auto time_seq_full = trajectory.generateTimeSequence(0.8); // 从开始到结束，dt=0.8
    auto time_seq_range = trajectory.generateTimeSequence(1.0, 5.0, 0.7); // 指定范围，dt=0.7
    
    std::cout << "完整时间序列(dt=0.8): " << time_seq_full.size() << " 点，最后时间: " 
              << time_seq_full.back() << " (轨迹结束时间: " << trajectory.getEndTime() << ")" << std::endl;
    std::cout << "范围时间序列(1.0-5.0, dt=0.7): " << time_seq_range.size() << " 点，最后时间: " 
              << time_seq_range.back() << std::endl;

    // 11. 段化时间序列（高性能，包含轨迹结束时间）
    std::cout << "\n--- 段化时间序列求值 ---" << std::endl;
    // 注意：generateSegmentedTimeSequence也会包含轨迹的结束时间点
    auto segmented_seq = trajectory.generateSegmentedTimeSequence(0.0, 6.0, 0.1);
    std::cout << "段化序列总点数: " << segmented_seq.getTotalSize() 
              << " (包含轨迹结束时间)" << std::endl;
    std::cout << "段数: " << segmented_seq.segments.size() << std::endl;
    
    // 使用段化序列进行高性能批量求值
    auto pos_segmented_eval = trajectory.evaluateSegmented(segmented_seq, 0);
    auto vel_segmented_eval = trajectory.evaluateSegmented(segmented_seq, 1);
    
    // get系列段化求值
    auto pos_segmented_get = trajectory.getPos(segmented_seq);
    auto vel_segmented_get = trajectory.getVel(segmented_seq);
    auto acc_segmented_get = trajectory.getAcc(segmented_seq);
    auto jerk_segmented_get = trajectory.getJerk(segmented_seq);
    auto snap_segmented_get = trajectory.getSnap(segmented_seq);
    
    std::cout << "段化get系列求值点数: " << pos_segmented_get.size() << std::endl;

    // 12. 轨迹分析
    std::cout << "\n--- 轨迹分析 ---" << std::endl;
    double traj_length = trajectory.getTrajectoryLength();
    double traj_length_custom = trajectory.getTrajectoryLength(2.0, 4.0, 0.05);
    double cumulative_length = trajectory.getCumulativeLength(3.0);
    
    std::cout << "轨迹总长度: " << traj_length << std::endl;
    std::cout << "段[2.0, 4.0]长度: " << traj_length_custom << std::endl;
    std::cout << "累积长度到t=3.0: " << cumulative_length << std::endl;

    // 13. 导数轨迹
    std::cout << "\n--- 导数轨迹 ---" << std::endl;
    auto vel_trajectory = trajectory.derivative(1);  // 速度轨迹（1阶导数）
    auto acc_trajectory = trajectory.derivative(2);  // 加速度轨迹（2阶导数）
    
    std::cout << "速度轨迹阶数: " << vel_trajectory.getOrder() << std::endl;
    std::cout << "加速度轨迹阶数: " << acc_trajectory.getOrder() << std::endl;

    // 14. 获取内部数据
    std::cout << "\n--- 获取内部数据 ---" << std::endl;
    auto space_points = cubic_from_points.getSpacePoints();
    auto time_segments_data = cubic_from_points.getTimeSegments();
    auto cumulative_times = cubic_from_points.getCumulativeTimes();
    auto boundary_conditions = cubic_from_points.getBoundaryConditions();
    auto trajectory_copy = cubic_from_points.getTrajectoryCopy();
    auto ppoly_ref = cubic_from_points.getPPoly();
    
    std::cout << "空间点数量: " << space_points.size() << std::endl;
    std::cout << "时间段数量: " << time_segments_data.size() << std::endl;
    std::cout << "累积时间数量: " << cumulative_times.size() << std::endl;

    // 15. 能量计算
    std::cout << "\n--- 能量计算 ---" << std::endl;
    double cubic_energy = cubic_from_points.getEnergy();
    double quintic_energy = quintic_from_points.getEnergy();
    double septic_energy = septic_from_points.getEnergy();
    
    std::cout << "三次样条能量: " << cubic_energy << std::endl;
    std::cout << "五次样条能量: " << quintic_energy << std::endl;
    std::cout << "七次样条能量: " << septic_energy << std::endl;

    // 16. PPolyND静态方法
    std::cout << "\n--- PPolyND静态方法 ---" << std::endl;
    std::vector<double> test_breakpoints = {0.0, 1.0, 2.0, 3.0};
    auto zero_poly = PPoly3D::zero(test_breakpoints, 3);
    SplinePoint3d constant_val(1.0, 2.0, 3.0);
    auto constant_poly = PPoly3D::constant(test_breakpoints, constant_val);
    
    std::cout << "零多项式在t=1.5: " << zero_poly.getPos(1.5).transpose() << std::endl;
    std::cout << "常数多项式在t=1.5: " << constant_poly.getPos(1.5).transpose() << std::endl;

    std::cout << "\n=== 示例完成 ===" << std::endl;
    return 0;
    //编译：g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. SplineTrajectoryExample.cpp -o SplineTrajectoryExample
}
```
```bash
g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. SplineTrajectoryExample.cpp -o SplineTrajectoryExample
```

## 未来计划

- [ ] 增加与MINCO等效的梯度传播机制
- [x] 实现对夹持七次样条的支持 (Septic Spline, Minimum Snap)
- [ ] 实现对N维非均匀B样条的支持
- [ ] 实现从夹持样条到非均匀B样条的精确转换

## 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 致谢

- 使用 [Eigen](http://eigen.tuxfamily.org/) 进行线性代数运算
- 灵感来源于 [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) 轨迹优化
- 基于 **经典样条插值理论** 和 **最小范数定理**