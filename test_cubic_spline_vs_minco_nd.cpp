// #define EIGEN_DONT_ALIGN_STATICALLY
// #include "SplineTrajectory.hpp"
// #include "cubic_spline_nd_with_ppolynd.hpp"
#include "SplineTrajectory.hpp"
#include "gcopter/minco.hpp"
#include "gcopter/trajectory.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <functional>

// ===================================================================================
// ================================= 辅助结构体和函数 =================================
// ===================================================================================
using namespace SplineTrajectory;
using namespace minco;

// 性能测试结构体
struct PerformanceResult
{
    double total_time_us;
    double avg_time_per_query_us;
    double min_time_us;
    double max_time_us;
    int num_queries;
};

// 一致性测试结构体
struct ConsistencyResult
{
    double max_pos_diff, max_vel_diff, max_acc_diff, max_jerk_diff;
    double avg_pos_diff, avg_vel_diff, avg_acc_diff, avg_jerk_diff;
};

// 生成随机控制点
SplineVector<Eigen::VectorXd> generateRandomWaypoints(int dim, int num_points, double min_val = -10.0, double max_val = 10.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);
    SplineVector<Eigen::VectorXd> waypoints(num_points, Eigen::VectorXd::Zero(dim));
    for (int i = 0; i < num_points; ++i)
    {
        for (int d = 0; d < dim; ++d)
        {
            waypoints[i](d) = dis(gen);
        }
    }
    return waypoints;
}

// 生成随机时间段
std::vector<double> generateRandomTimeSegments(int num_segments, double min_duration = 0.5, double max_duration = 2.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_duration, max_duration);
    std::vector<double> time_segments(num_segments);
    for (int i = 0; i < num_segments; ++i)
    {
        time_segments[i] = dis(gen);
    }
    return time_segments;
}

// 生成随机边界速度
Eigen::VectorXd generateRandomVelocity(int dim, double max_vel = 5.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-max_vel, max_vel);
    Eigen::VectorXd v(dim);
    for (int d = 0; d < dim; ++d)
        v(d) = dis(gen);
    return v;
}

// 性能测试函数
PerformanceResult measurePerformance(const std::function<void(int)> &func, int num_queries)
{
    if (num_queries <= 0)
        return {0, 0, 0, 0, 0};
    std::vector<double> times;
    times.reserve(num_queries);
    auto start_total = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(i);
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3);
    }
    auto end_total = std::chrono::high_resolution_clock::now();
    PerformanceResult result;
    result.total_time_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count() / 1e3;
    result.num_queries = num_queries;
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    result.avg_time_per_query_us = sum / num_queries;
    result.min_time_us = *std::min_element(times.begin(), times.end());
    result.max_time_us = *std::max_element(times.begin(), times.end());
    return result;
}

// 打印性能结果
void printPerformance(const PerformanceResult &r, const std::string &name)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << name << " 性能: 总" << r.num_queries << "次, 总耗时: " << r.total_time_us
              << "μs, 平均: " << r.avg_time_per_query_us << "μs, 最快: " << r.min_time_us
              << "μs, 最慢: " << r.max_time_us << "μs\n";
}

// 一致性测试
template <typename VecType>
ConsistencyResult testConsistency(
    const SplineVector<VecType> &cubic_pos, const SplineVector<VecType> &cubic_vel,
    const SplineVector<VecType> &cubic_acc, const SplineVector<VecType> &cubic_jerk,
    const SplineVector<VecType> &minco_pos, const SplineVector<VecType> &minco_vel,
    const SplineVector<VecType> &minco_acc, const SplineVector<VecType> &minco_jerk)
{
    ConsistencyResult result = {0, 0, 0, 0, 0, 0, 0, 0};
    double sum_pos = 0, sum_vel = 0, sum_acc = 0, sum_jerk = 0;
    int N = cubic_pos.size();
    if (N == 0)
        return result;
    for (int i = 0; i < N; ++i)
    {
        double pos_diff = (cubic_pos[i] - minco_pos[i]).norm();
        double vel_diff = (cubic_vel[i] - minco_vel[i]).norm();
        double acc_diff = (cubic_acc[i] - minco_acc[i]).norm();
        double jerk_diff = (cubic_jerk[i] - minco_jerk[i]).norm();
        result.max_pos_diff = std::max(result.max_pos_diff, pos_diff);
        result.max_vel_diff = std::max(result.max_vel_diff, vel_diff);
        result.max_acc_diff = std::max(result.max_acc_diff, acc_diff);
        result.max_jerk_diff = std::max(result.max_jerk_diff, jerk_diff);
        sum_pos += pos_diff;
        sum_vel += vel_diff;
        sum_acc += acc_diff;
        sum_jerk += jerk_diff;
    }
    result.avg_pos_diff = sum_pos / N;
    result.avg_vel_diff = sum_vel / N;
    result.avg_acc_diff = sum_acc / N;
    result.avg_jerk_diff = sum_jerk / N;
    return result;
}

// 打印一致性结果
void printConsistency(const ConsistencyResult &r)
{
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  - 位置   max_diff: " << r.max_pos_diff << ", avg_diff: " << r.avg_pos_diff << "\n"
              << "  - 速度   max_diff: " << r.max_vel_diff << ", avg_diff: " << r.avg_vel_diff << "\n"
              << "  - 加速度 max_diff: " << r.max_acc_diff << ", avg_diff: " << r.avg_acc_diff << "\n"
              << "  - 加加.. max_diff: " << r.max_jerk_diff << ", avg_diff: " << r.avg_jerk_diff << std::endl;
    if (r.max_pos_diff < 1e-9 && r.max_vel_diff < 1e-8 && r.max_acc_diff < 1e-7 && r.max_jerk_diff < 1e-6)
    {
        std::cout << "  - 一致性检查通过!" << std::endl;
    }
    else
    {
        std::cout << "  - 一致性检查失败! 差异过大。" << std::endl;
    }
}

// ===================================================================================
// ============================= 模板化的主测试逻辑 =================================
// ===================================================================================
template <int DIM>
void runTest()
{
    std::cout << "\n--- " << DIM << " 维测试 ---" << std::endl;
    int num_segments = 10;
    int num_points = num_segments + 1;
    double start_time = 0.0;
    int fit_runs = 10;
    auto time_segments_vec = generateRandomTimeSegments(num_segments);
    auto waypoints_vec_eigen = generateRandomWaypoints(DIM, num_points);
    auto start_vel_eigen = generateRandomVelocity(DIM);
    auto end_vel_eigen = generateRandomVelocity(DIM);

    using Spline = SplineTrajectory::CubicSplineND<DIM>;
    using VectorD = typename Spline::VectorType;

    SplineVector<VectorD> waypoints;
    for (const auto &v : waypoints_vec_eigen)
        waypoints.push_back(v);

    BoundaryConditions<DIM> bv(start_vel_eigen, end_vel_eigen);

    std::cout << "2.1) 样条拟合性能 (平均 " << fit_runs << " 次):" << std::endl;
    double total_cubic_fit_time_us = 0;
    for (int i = 0; i < fit_runs; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        Spline cubic_spline(time_segments_vec, waypoints, start_time, bv);
        auto t2 = std::chrono::high_resolution_clock::now();
        total_cubic_fit_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }
    std::cout << "  - CubicSpline" << DIM << "D 拟合: " << std::fixed << std::setprecision(2)
              << total_cubic_fit_time_us / fit_runs << " μs" << std::endl;

    int num_minco_instances = (DIM + 2) / 3;
    double total_minco_fit_time_us = 0;
    for (int i = 0; i < fit_runs; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<minco::MINCO_S2NU> minco_solvers(num_minco_instances);
        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;
            int current_dim = std::min(3, DIM - start_dim);
            Eigen::MatrixXd head = Eigen::MatrixXd::Zero(3, 2), tail = Eigen::MatrixXd::Zero(3, 2);
            head.block(0, 0, current_dim, 1) = waypoints_vec_eigen.front().segment(start_dim, current_dim);
            head.block(0, 1, current_dim, 1) = start_vel_eigen.segment(start_dim, current_dim);
            tail.block(0, 0, current_dim, 1) = waypoints_vec_eigen.back().segment(start_dim, current_dim);
            tail.block(0, 1, current_dim, 1) = end_vel_eigen.segment(start_dim, current_dim);
            minco_solvers[j].setConditions(head, tail, num_segments);
            Eigen::MatrixXd minco_points = Eigen::MatrixXd::Zero(3, num_points - 2);
            for (int k = 1; k < num_points - 1; ++k)
                minco_points.block(0, k - 1, current_dim, 1) = waypoints_vec_eigen[k].segment(start_dim, current_dim);
            Eigen::VectorXd minco_times(num_segments);
            for (int k = 0; k < num_segments; ++k)
                minco_times(k) = time_segments_vec[k];
            minco_solvers[j].setParameters(minco_points, minco_times);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        total_minco_fit_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }
    std::cout << "  - MINCO 模拟 " << DIM << "D 拟合: " << std::fixed << std::setprecision(2)
              << total_minco_fit_time_us / fit_runs << " μs (" << num_minco_instances << "个实例)" << std::endl;

    Spline cubic_spline(time_segments_vec, waypoints, start_time, bv);
    std::vector<minco::MINCO_S2NU> minco_solvers_for_query(num_minco_instances);
    std::vector<Trajectory<3>> minco_trajs(num_minco_instances);
    for (int j = 0; j < num_minco_instances; ++j)
    {
        int start_dim = j * 3;
        int current_dim = std::min(3, DIM - start_dim);
        Eigen::MatrixXd head = Eigen::MatrixXd::Zero(3, 2), tail = Eigen::MatrixXd::Zero(3, 2);
        head.block(0, 0, current_dim, 1) = waypoints_vec_eigen.front().segment(start_dim, current_dim);
        head.block(0, 1, current_dim, 1) = start_vel_eigen.segment(start_dim, current_dim);
        tail.block(0, 0, current_dim, 1) = waypoints_vec_eigen.back().segment(start_dim, current_dim);
        tail.block(0, 1, current_dim, 1) = end_vel_eigen.segment(start_dim, current_dim);
        minco_solvers_for_query[j].setConditions(head, tail, num_segments);
        Eigen::MatrixXd minco_points = Eigen::MatrixXd::Zero(3, num_points - 2);
        for (int k = 1; k < num_points - 1; ++k)
            minco_points.block(0, k - 1, current_dim, 1) = waypoints_vec_eigen[k].segment(start_dim, current_dim);
        Eigen::VectorXd minco_times(num_segments);
        for (int k = 0; k < num_segments; ++k)
            minco_times(k) = time_segments_vec[k];
        minco_solvers_for_query[j].setParameters(minco_points, minco_times);
        minco_solvers_for_query[j].getTrajectory(minco_trajs[j]);
    }

    double dt = 0.01;
    auto time_seq = cubic_spline.getTrajectory().generateTimeSequence(dt);
    int num_queries = time_seq.size();

    std::cout << "\n2.2) 单点查询性能 (" << num_queries << "个点):" << std::endl;
    std::function<void(int)> cubic_func = [&](int i)
    { cubic_spline.getTrajectory().getPos(time_seq[i]); };
    auto perf_cubic = measurePerformance(cubic_func, num_queries);
    printPerformance(perf_cubic, "  - CubicSpline" + std::to_string(DIM) + "D");

    std::function<void(int)> minco_func = [&](int i)
    {
        VectorD p = VectorD::Zero(); // 修复：显式初始化为零向量
        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;
            int current_dim = std::min(3, DIM - start_dim);
            p.segment(start_dim, current_dim) = minco_trajs[j].getPos(time_seq[i]).head(current_dim);
        }
    };
    auto perf_minco = measurePerformance(minco_func, num_queries);
    printPerformance(perf_minco, "  - MINCO 模拟 " + std::to_string(DIM) + "D");

    std::cout << "\n2.3) 批量查询性能 (一次性查询 " << num_queries << " 个点):" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto cubic_results_position = cubic_spline.getTrajectory().getPos(time_seq);
    auto cubic_results_velocity = cubic_spline.getTrajectory().getVel(time_seq);
    auto cubic_results_acceleration = cubic_spline.getTrajectory().getAcc(time_seq);
    auto cubic_results_jerk = cubic_spline.getTrajectory().getJerk(time_seq);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cubic_batch_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "  - CubicSpline" << DIM << "D 批量查询 (getTrajectorySequence): " << cubic_batch_us << " μs" << std::endl;

    auto t3 = std::chrono::high_resolution_clock::now();
    SplineVector<VectorD> minco_pos(num_queries), minco_vel(num_queries), minco_acc(num_queries), minco_jerk(num_queries);

    // ############  修复：正确初始化和组装多维向量 ############
    for (size_t i = 0; i < time_seq.size(); ++i)
    {
        double t = time_seq[i];

        // 修复：显式初始化为正确尺寸的零向量
        VectorD p = VectorD::Zero();
        VectorD v = VectorD::Zero();
        VectorD a = VectorD::Zero();
        VectorD jrk = VectorD::Zero();

        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;
            int current_dim = std::min(3, DIM - start_dim);

            // 确保不会越界访问
            if (start_dim < DIM)
            {
                Eigen::Vector3d pos_3d = minco_trajs[j].getPos(t);
                Eigen::Vector3d vel_3d = minco_trajs[j].getVel(t);
                Eigen::Vector3d acc_3d = minco_trajs[j].getAcc(t);
                Eigen::Vector3d jerk_3d = minco_trajs[j].getJer(t);

                p.segment(start_dim, current_dim) = pos_3d.head(current_dim);
                v.segment(start_dim, current_dim) = vel_3d.head(current_dim);
                a.segment(start_dim, current_dim) = acc_3d.head(current_dim);
                jrk.segment(start_dim, current_dim) = jerk_3d.head(current_dim);
            }
        }
        minco_pos[i] = p;
        minco_vel[i] = v;
        minco_acc[i] = a;
        minco_jerk[i] = jrk;
    }
    // ############ End of Fix ############

    auto t4 = std::chrono::high_resolution_clock::now();
    double minco_batch_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    std::cout << "  - MINCO 逐点模拟批量查询: " << minco_batch_us << " μs" << std::endl;
    if (cubic_batch_us > 0)
        std::cout << "  - 性能比 (MINCO/CubicSpline): " << std::fixed << std::setprecision(3)
                  << minco_batch_us / cubic_batch_us << "x" << std::endl;

    std::cout << "\n2.4) 一致性检查:" << std::endl;
    auto cons = testConsistency(cubic_results_position, cubic_results_velocity,
                                cubic_results_acceleration, cubic_results_jerk,
                                minco_pos, minco_vel, minco_acc, minco_jerk);
    printConsistency(cons);

    std::cout << "\n2.5) PPolyND::derivative() 方法验证 (一阶导数):" << std::endl;
    auto ppoly = cubic_spline.getPPoly();
    auto d_ppoly = ppoly.derivative(1);
    auto vel_from_d_ppoly = d_ppoly.evaluate(time_seq, 0);
    double max_diff = 0.0, avg_diff = 0.0;
    if (!time_seq.empty())
    {
        for (size_t i = 0; i < time_seq.size(); ++i)
        {
            double diff = (cubic_results_velocity[i] - vel_from_d_ppoly[i]).norm();
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
        }
        avg_diff /= time_seq.size();
    }
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  - getVel() vs derivative().evaluate() 最大差异: " << max_diff
              << ", 平均差异: " << avg_diff << std::endl;
    if (max_diff < 1e-9)
    {
        std::cout << "  - 验证通过!" << std::endl;
    }
    else
    {
        std::cout << "  - 验证失败! 差异过大。" << std::endl;
    }

    // DIM=3 特殊测试: 轨迹能量计算函数一致性
    if constexpr (DIM == 3)
    {
        std::cout << "\n2.6) 轨迹能量计算一致性测试 (DIM=3):" << std::endl;
        
        // 获取 CubicSplineND 的能量
        double cubic_energy = cubic_spline.getEnergy();
        
        // 获取 MINCO 的能量 (只有第一个实例，因为DIM=3时只有一个MINCO实例)
        double minco_energy = 0.0;
        minco_solvers_for_query[0].getEnergy(minco_energy);
        
        // 计算差异
        double energy_diff = std::abs(cubic_energy - minco_energy);
        double relative_error = (cubic_energy != 0.0) ? (energy_diff / std::abs(cubic_energy)) : 0.0;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  - CubicSplineND 能量: " << cubic_energy << std::endl;
        std::cout << "  - MINCO 能量: " << minco_energy << std::endl;
        std::cout << "  - 绝对差异: " << energy_diff << std::endl;
        std::cout << "  - 相对误差: " << relative_error * 100 << "%" << std::endl;
        
        if (relative_error < 1e-6)
        {
            std::cout << "  - 能量一致性检查通过!" << std::endl;
        }
        else
        {
            std::cout << "  - 能量一致性检查失败! 相对误差过大。" << std::endl;
        }
    }
}

// ===================================================================================
// =================================== 主函数 ======================================
// ===================================================================================
int main()
{
    std::cout << "\n===== 多维 CubicSplineND vs MINCO 性能与一致性对比 =====\n";
    std::cout << "CubicSplineND 使用了 PPolyND 进行高效批量查询\n";
    std::cout << "MINCO 是业界标准的3D样条库，这里通过组合实例来模拟高维\n";

    runTest<1>();
    runTest<2>();
    runTest<3>();
    runTest<4>();
    runTest<5>();
    runTest<6>();
    runTest<7>();
    runTest<8>();
    runTest<9>();
    runTest<10>();

    std::cout << "\n===== 测试结束 =====\n"
              << std::endl;
    return 0;
}

// ❯ ./test_cubic_spline_vs_minco_nd

// ===== 多维 CubicSplineND vs MINCO 性能与一致性对比 =====
// CubicSplineND 使用了 PPolyND 进行高效批量查询
// MINCO 是业界标准的3D样条库，这里通过组合实例来模拟高维

// --- 1 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline1D 拟合: 1.10 μs
//   - MINCO 模拟 1D 拟合: 0.60 μs (1个实例)

// 2.2) 单点查询性能 (1206个点):
//   - CubicSpline1D 性能: 总1206次, 总耗时: 44.209μs, 平均: 0.018μs, 最快: 0.010μs, 最慢: 0.311μs
//   - MINCO 模拟 1D 性能: 总1206次, 总耗时: 42.245μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.100μs

// 2.3) 批量查询性能 (一次性查询 1206 个点):
//   - CubicSpline1D 批量查询 (getTrajectorySequence): 52.000 μs
//   - MINCO 逐点模拟批量查询: 26.000 μs
//   - 性能比 (MINCO/CubicSpline): 0.500x

// 2.4) 一致性检查:
//   - 位置   max_diff: 1.3323e-14, avg_diff: 2.5679e-15
//   - 速度   max_diff: 3.3751e-14, avg_diff: 4.6299e-15
//   - 加速度 max_diff: 8.5265e-14, avg_diff: 1.4276e-14
//   - 加加.. max_diff: 4.2633e-14, avg_diff: 1.3354e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 0.0000e+00, 平均差异: 0.0000e+00
//   - 验证通过!

// --- 2 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline2D 拟合: 0.20 μs
//   - MINCO 模拟 2D 拟合: 0.10 μs (1个实例)

// 2.2) 单点查询性能 (1099个点):
//   - CubicSpline2D 性能: 总1099次, 总耗时: 40.642μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.341μs
//   - MINCO 模拟 2D 性能: 总1099次, 总耗时: 38.417μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.030μs

// 2.3) 批量查询性能 (一次性查询 1099 个点):
//   - CubicSpline2D 批量查询 (getTrajectorySequence): 40.000 μs
//   - MINCO 逐点模拟批量查询: 47.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.175x

// 2.4) 一致性检查:
//   - 位置   max_diff: 2.2204e-14, avg_diff: 4.9493e-15
//   - 速度   max_diff: 4.0862e-14, avg_diff: 9.6648e-15
//   - 加速度 max_diff: 9.0994e-14, avg_diff: 2.8732e-14
//   - 加加.. max_diff: 1.0846e-13, avg_diff: 3.1734e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 1.4211e-14, 平均差异: 1.0528e-15
//   - 验证通过!

// --- 3 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline3D 拟合: 0.80 μs
//   - MINCO 模拟 3D 拟合: 0.10 μs (1个实例)

// 2.2) 单点查询性能 (1368个点):
//   - CubicSpline3D 性能: 总1368次, 总耗时: 52.165μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.621μs
//   - MINCO 模拟 3D 性能: 总1368次, 总耗时: 47.817μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.021μs

// 2.3) 批量查询性能 (一次性查询 1368 个点):
//   - CubicSpline3D 批量查询 (getTrajectorySequence): 51.000 μs
//   - MINCO 逐点模拟批量查询: 90.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.765x

// 2.4) 一致性检查:
//   - 位置   max_diff: 1.7585e-14, avg_diff: 7.0945e-15
//   - 速度   max_diff: 5.6579e-14, avg_diff: 1.3414e-14
//   - 加速度 max_diff: 1.1031e-13, avg_diff: 3.2179e-14
//   - 加加.. max_diff: 3.7303e-14, avg_diff: 1.5071e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 7.9441e-15, 平均差异: 1.0086e-15
//   - 验证通过!

// --- 4 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline4D 拟合: 0.40 μs
//   - MINCO 模拟 4D 拟合: 1.10 μs (2个实例)

// 2.2) 单点查询性能 (1206个点):
//   - CubicSpline4D 性能: 总1206次, 总耗时: 44.740μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.561μs
//   - MINCO 模拟 4D 性能: 总1206次, 总耗时: 55.382μs, 平均: 0.028μs, 最快: 0.010μs, 最慢: 13.216μs

// 2.3) 批量查询性能 (一次性查询 1206 个点):
//   - CubicSpline4D 批量查询 (getTrajectorySequence): 59.000 μs
//   - MINCO 逐点模拟批量查询: 99.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.678x

// 2.4) 一致性检查:
//   - 位置   max_diff: 3.6310e-14, avg_diff: 7.7041e-15
//   - 速度   max_diff: 7.4722e-14, avg_diff: 1.4270e-14
//   - 加速度 max_diff: 1.7834e-13, avg_diff: 3.8293e-14
//   - 加加.. max_diff: 1.3964e-13, avg_diff: 4.4176e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 1.4349e-14, 平均差异: 1.4343e-15
//   - 验证通过!

// --- 5 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline5D 拟合: 0.30 μs
//   - MINCO 模拟 5D 拟合: 1.10 μs (2个实例)

// 2.2) 单点查询性能 (1104个点):
//   - CubicSpline5D 性能: 总1104次, 总耗时: 40.171μs, 平均: 0.018μs, 最快: 0.010μs, 最慢: 0.190μs
//   - MINCO 模拟 5D 性能: 总1104次, 总耗时: 38.618μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.030μs

// 2.3) 批量查询性能 (一次性查询 1104 个点):
//   - CubicSpline5D 批量查询 (getTrajectorySequence): 63.000 μs
//   - MINCO 逐点模拟批量查询: 98.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.556x

// 2.4) 一致性检查:
//   - 位置   max_diff: 9.4779e-14, avg_diff: 2.0215e-14
//   - 速度   max_diff: 3.4930e-13, avg_diff: 4.4834e-14
//   - 加速度 max_diff: 7.4800e-13, avg_diff: 1.2903e-13
//   - 加加.. max_diff: 1.3024e-13, avg_diff: 4.9363e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 1.5987e-14, 平均差异: 2.7817e-15
//   - 验证通过!

// --- 6 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline6D 拟合: 0.40 μs
//   - MINCO 模拟 6D 拟合: 1.10 μs (2个实例)

// 2.2) 单点查询性能 (1105个点):
//   - CubicSpline6D 性能: 总1105次, 总耗时: 40.391μs, 平均: 0.018μs, 最快: 0.010μs, 最慢: 0.180μs
//   - MINCO 模拟 6D 性能: 总1105次, 总耗时: 42.836μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.041μs

// 2.3) 批量查询性能 (一次性查询 1105 个点):
//   - CubicSpline6D 批量查询 (getTrajectorySequence): 75.000 μs
//   - MINCO 逐点模拟批量查询: 110.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.467x

// 2.4) 一致性检查:
//   - 位置   max_diff: 6.6678e-14, avg_diff: 1.6490e-14
//   - 速度   max_diff: 1.4098e-13, avg_diff: 3.2744e-14
//   - 加速度 max_diff: 2.7663e-13, avg_diff: 7.6826e-14
//   - 加加.. max_diff: 1.8954e-13, avg_diff: 4.8653e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 2.1900e-14, 平均差异: 3.0504e-15
//   - 验证通过!

// --- 7 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline7D 拟合: 0.40 μs
//   - MINCO 模拟 7D 拟合: 2.10 μs (3个实例)

// 2.2) 单点查询性能 (966个点):
//   - CubicSpline7D 性能: 总966次, 总耗时: 35.752μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.200μs
//   - MINCO 模拟 7D 性能: 总966次, 总耗时: 34.029μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.021μs

// 2.3) 批量查询性能 (一次性查询 966 个点):
//   - CubicSpline7D 批量查询 (getTrajectorySequence): 73.000 μs
//   - MINCO 逐点模拟批量查询: 126.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.726x

// 2.4) 一致性检查:
//   - 位置   max_diff: 5.2131e-14, avg_diff: 1.4910e-14
//   - 速度   max_diff: 1.3413e-13, avg_diff: 2.9879e-14
//   - 加速度 max_diff: 3.9980e-13, avg_diff: 8.1779e-14
//   - 加加.. max_diff: 6.1274e-13, avg_diff: 2.0280e-13
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 2.6896e-14, 平均差异: 3.7410e-15
//   - 验证通过!

// --- 8 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline8D 拟合: 1.20 μs
//   - MINCO 模拟 8D 拟合: 1.30 μs (3个实例)

// 2.2) 单点查询性能 (1175个点):
//   - CubicSpline8D 性能: 总1175次, 总耗时: 44.420μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.562μs
//   - MINCO 模拟 8D 性能: 总1175次, 总耗时: 41.443μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.021μs

// 2.3) 批量查询性能 (一次性查询 1175 个点):
//   - CubicSpline8D 批量查询 (getTrajectorySequence): 108.000 μs
//   - MINCO 逐点模拟批量查询: 170.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.574x

// 2.4) 一致性检查:
//   - 位置   max_diff: 1.1818e-13, avg_diff: 1.9873e-14
//   - 速度   max_diff: 6.1681e-13, avg_diff: 5.1659e-14
//   - 加速度 max_diff: 1.6226e-12, avg_diff: 1.7504e-13
//   - 加加.. max_diff: 2.6634e-13, avg_diff: 6.8343e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 0.0000e+00, 平均差异: 0.0000e+00
//   - 验证通过!

// --- 9 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline9D 拟合: 1.20 μs
//   - MINCO 模拟 9D 拟合: 1.30 μs (3个实例)

// 2.2) 单点查询性能 (1438个点):
//   - CubicSpline9D 性能: 总1438次, 总耗时: 52.726μs, 平均: 0.018μs, 最快: 0.010μs, 最慢: 0.250μs
//   - MINCO 模拟 9D 性能: 总1438次, 总耗时: 50.752μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.021μs

// 2.3) 批量查询性能 (一次性查询 1438 个点):
//   - CubicSpline9D 批量查询 (getTrajectorySequence): 167.000 μs
//   - MINCO 逐点模拟批量查询: 268.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.605x

// 2.4) 一致性检查:
//   - 位置   max_diff: 5.9043e-14, avg_diff: 1.5751e-14
//   - 速度   max_diff: 1.6287e-13, avg_diff: 2.6465e-14
//   - 加速度 max_diff: 2.2225e-13, avg_diff: 5.3161e-14
//   - 加加.. max_diff: 7.7012e-14, avg_diff: 2.9308e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 1.0986e-14, 平均差异: 2.1554e-15
//   - 验证通过!

// --- 10 维测试 ---
// 2.1) 样条拟合性能 (平均 10 次):
//   - CubicSpline10D 拟合: 1.20 μs
//   - MINCO 模拟 10D 拟合: 2.20 μs (4个实例)

// 2.2) 单点查询性能 (1079个点):
//   - CubicSpline10D 性能: 总1079次, 总耗时: 39.840μs, 平均: 0.019μs, 最快: 0.010μs, 最慢: 0.230μs
//   - MINCO 模拟 10D 性能: 总1079次, 总耗时: 38.257μs, 平均: 0.017μs, 最快: 0.010μs, 最慢: 0.030μs

// 2.3) 批量查询性能 (一次性查询 1079 个点):
//   - CubicSpline10D 批量查询 (getTrajectorySequence): 149.000 μs
//   - MINCO 逐点模拟批量查询: 199.000 μs
//   - 性能比 (MINCO/CubicSpline): 1.336x

// 2.4) 一致性检查:
//   - 位置   max_diff: 7.0869e-14, avg_diff: 1.4482e-14
//   - 速度   max_diff: 4.0358e-13, avg_diff: 3.4961e-14
//   - 加速度 max_diff: 1.0760e-12, avg_diff: 1.1292e-13
//   - 加加.. max_diff: 3.3713e-13, avg_diff: 7.2532e-14
//   - 一致性检查通过!

// 2.5) PPolyND::derivative() 方法验证 (一阶导数):
//   - getVel() vs derivative().evaluate() 最大差异: 2.3533e-14, 平均差异: 3.0579e-15
//   - 验证通过!

// ===== 测试结束 =====
