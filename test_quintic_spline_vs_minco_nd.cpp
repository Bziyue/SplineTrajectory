// #define EIGEN_DONT_ALIGN_STATICALLY
// #include "quintic_spline_nd.hpp"
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
    double max_pos_diff, max_vel_diff, max_acc_diff, max_jerk_diff, max_snap_diff;
    double avg_pos_diff, avg_vel_diff, avg_acc_diff, avg_jerk_diff, avg_snap_diff;
};

// 批量评估性能结果
struct BatchPerformanceResult
{
    double segmented_time_generation_us;
    double quintic_batch_evaluation_us;
    double minco_batch_evaluation_us;
    double speedup_ratio;
    int total_points;
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

// 生成随机边界条件（速度和加速度）
Eigen::VectorXd generateRandomVector(int dim, double max_val = 5.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-max_val, max_val);
    Eigen::VectorXd v(dim);
    for (int d = 0; d < dim; ++d)
        v(d) = dis(gen);
    return v;
}

// 辅助函数：安全地从向量中获取数据，如果索引超出范围则重复最后一个有效值
double safeGetFromVector(const Eigen::VectorXd &vec, int index)
{
    if (vec.size() == 0)
        return 0.0;
    if (index >= vec.size())
    {
        return vec(vec.size() - 1); // 重复最后一个值
    }
    return vec(index);
}

// 辅助函数：从高维向量安全地构造3D向量
Eigen::Vector3d make3DFromND(const Eigen::VectorXd &nd_vec, int start_dim)
{
    Eigen::Vector3d result;
    for (int i = 0; i < 3; ++i)
    {
        result(i) = safeGetFromVector(nd_vec, start_dim + i);
    }
    return result;
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

// 打印批量性能结果
void printBatchPerformance(const BatchPerformanceResult &r, const std::string &name)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << name << " 批量评估性能:\n";
    std::cout << "  - 段化时间序列生成: " << r.segmented_time_generation_us << " μs\n";
    std::cout << "  - QuinticSpline 批量评估: " << r.quintic_batch_evaluation_us << " μs\n";
    std::cout << "  - MINCO 批量评估: " << r.minco_batch_evaluation_us << " μs\n";
    std::cout << "  - 总查询点数: " << r.total_points << "\n";
    std::cout << "  - 性能提升比 (MINCO/QuinticSpline): " << r.speedup_ratio << "x\n";
}

// 一致性测试（包含snap/四阶导数）
template <typename VecType>
ConsistencyResult testConsistency(
    const SplineVector<VecType> &quintic_pos, const SplineVector<VecType> &quintic_vel,
    const SplineVector<VecType> &quintic_acc, const SplineVector<VecType> &quintic_jerk,
    const SplineVector<VecType> &quintic_snap,
    const SplineVector<VecType> &minco_pos, const SplineVector<VecType> &minco_vel,
    const SplineVector<VecType> &minco_acc, const SplineVector<VecType> &minco_jerk,
    const SplineVector<VecType> &minco_snap)
{
    ConsistencyResult result = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double sum_pos = 0, sum_vel = 0, sum_acc = 0, sum_jerk = 0, sum_snap = 0;
    int N = quintic_pos.size();
    if (N == 0)
        return result;

    for (int i = 0; i < N; ++i)
    {
        double pos_diff = (quintic_pos[i] - minco_pos[i]).norm();
        double vel_diff = (quintic_vel[i] - minco_vel[i]).norm();
        double acc_diff = (quintic_acc[i] - minco_acc[i]).norm();
        double jerk_diff = (quintic_jerk[i] - minco_jerk[i]).norm();
        double snap_diff = (quintic_snap[i] - minco_snap[i]).norm();

        result.max_pos_diff = std::max(result.max_pos_diff, pos_diff);
        result.max_vel_diff = std::max(result.max_vel_diff, vel_diff);
        result.max_acc_diff = std::max(result.max_acc_diff, acc_diff);
        result.max_jerk_diff = std::max(result.max_jerk_diff, jerk_diff);
        result.max_snap_diff = std::max(result.max_snap_diff, snap_diff);

        sum_pos += pos_diff;
        sum_vel += vel_diff;
        sum_acc += acc_diff;
        sum_jerk += jerk_diff;
        sum_snap += snap_diff;
    }

    result.avg_pos_diff = sum_pos / N;
    result.avg_vel_diff = sum_vel / N;
    result.avg_acc_diff = sum_acc / N;
    result.avg_jerk_diff = sum_jerk / N;
    result.avg_snap_diff = sum_snap / N;

    return result;
}

// 打印一致性结果（包含snap）
void printConsistency(const ConsistencyResult &r)
{
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  - 位置   max_diff: " << r.max_pos_diff << ", avg_diff: " << r.avg_pos_diff << "\n"
              << "  - 速度   max_diff: " << r.max_vel_diff << ", avg_diff: " << r.avg_vel_diff << "\n"
              << "  - 加速度 max_diff: " << r.max_acc_diff << ", avg_diff: " << r.avg_acc_diff << "\n"
              << "  - 加加速 max_diff: " << r.max_jerk_diff << ", avg_diff: " << r.avg_jerk_diff << "\n"
              << "  - 四阶导 max_diff: " << r.max_snap_diff << ", avg_diff: " << r.avg_snap_diff << std::endl;

    if (r.max_pos_diff < 1e-9 && r.max_vel_diff < 1e-8 && r.max_acc_diff < 1e-7 &&
        r.max_jerk_diff < 1e-6 && r.max_snap_diff < 1e-5)
    {
        std::cout << "  - 一致性检查通过!" << std::endl;
    }
    else
    {
        std::cout << "  - 一致性检查失败! 差异过大。" << std::endl;
    }
}

// 新增：从段化时间序列提取时间点
template <int DIM>
std::vector<double> extractTimePointsFromSegmented(const typename SplineTrajectory::PPolyND<DIM>::SegmentedTimeSeq& segmented_seq)
{
    std::vector<double> time_points;
    time_points.reserve(segmented_seq.getTotalSize());
    
    for (const auto& segment : segmented_seq.segments)
    {
        for (double t : segment.times)
        {
            time_points.push_back(t);
        }
    }
    
    return time_points;
}

// 新增：测试段化批量评估性能
template <int DIM>
BatchPerformanceResult testSegmentedBatchEvaluation(
    const SplineTrajectory::QuinticSplineND<DIM>& quintic_spline,
    const std::vector<Trajectory<5>>& minco_trajs,
    double dt)
{
    using QuinticSpline = SplineTrajectory::QuinticSplineND<DIM>;
    using VectorD = typename QuinticSpline::VectorType;
    
    BatchPerformanceResult result = {0, 0, 0, 0, 0};

    double start_time = quintic_spline.getTrajectory().getStartTime();
    double end_time = quintic_spline.getTrajectory().getEndTime();

    // 1. 生成段化时间序列
    auto t1 = std::chrono::high_resolution_clock::now();
    auto segmented_seq = quintic_spline.getTrajectory().generateSegmentedTimeSequence(start_time, end_time, dt);
    auto t2 = std::chrono::high_resolution_clock::now();
    result.segmented_time_generation_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    result.total_points = segmented_seq.getTotalSize();
    
    // 2. QuinticSpline 段化批量评估
    auto t3 = std::chrono::high_resolution_clock::now();
    auto quintic_pos = quintic_spline.getTrajectory().evaluateSegmented(segmented_seq, 0);
    auto quintic_vel = quintic_spline.getTrajectory().evaluateSegmented(segmented_seq, 1);
    auto quintic_acc = quintic_spline.getTrajectory().evaluateSegmented(segmented_seq, 2);
    auto quintic_jerk = quintic_spline.getTrajectory().evaluateSegmented(segmented_seq, 3);
    auto quintic_snap = quintic_spline.getTrajectory().evaluateSegmented(segmented_seq, 4);
    auto t4 = std::chrono::high_resolution_clock::now();
    result.quintic_batch_evaluation_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    
    // 3. 提取时间点用于MINCO评估
    std::vector<double> time_points = extractTimePointsFromSegmented<DIM>(segmented_seq);
    
    // 4. MINCO 批量评估（模拟）
    auto t5 = std::chrono::high_resolution_clock::now();
    int num_minco_instances = (DIM + 2) / 3;
    SplineVector<VectorD> minco_pos(result.total_points), minco_vel(result.total_points),
        minco_acc(result.total_points), minco_jerk(result.total_points), minco_snap(result.total_points);

    for (size_t i = 0; i < time_points.size(); ++i)
    {
        double t = time_points[i];

        VectorD p = VectorD::Zero();
        VectorD v = VectorD::Zero();
        VectorD a = VectorD::Zero();
        VectorD jrk = VectorD::Zero();
        VectorD snp = VectorD::Zero();

        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;
            int current_dim = std::min(3, DIM - start_dim);

            if (start_dim < DIM)
            {
                Eigen::Vector3d pos_3d = minco_trajs[j].getPos(t);
                Eigen::Vector3d vel_3d = minco_trajs[j].getVel(t);
                Eigen::Vector3d acc_3d = minco_trajs[j].getAcc(t);
                Eigen::Vector3d jerk_3d = minco_trajs[j].getJer(t);
                Eigen::Vector3d snap_3d = minco_trajs[j].getSnap(t);

                p.segment(start_dim, current_dim) = pos_3d.head(current_dim);
                v.segment(start_dim, current_dim) = vel_3d.head(current_dim);
                a.segment(start_dim, current_dim) = acc_3d.head(current_dim);
                jrk.segment(start_dim, current_dim) = jerk_3d.head(current_dim);
                snp.segment(start_dim, current_dim) = snap_3d.head(current_dim);
            }
        }
        minco_pos[i] = p;
        minco_vel[i] = v;
        minco_acc[i] = a;
        minco_jerk[i] = jrk;
        minco_snap[i] = snp;
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    result.minco_batch_evaluation_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
    
    // 5. 计算性能提升比
    if (result.quintic_batch_evaluation_us > 0)
    {
        result.speedup_ratio = static_cast<double>(result.minco_batch_evaluation_us) / result.quintic_batch_evaluation_us;
    }
    
    // 6. 验证一致性
    std::cout << "\n    段化批量评估一致性检查:" << std::endl;
    auto cons = testConsistency(quintic_pos, quintic_vel, quintic_acc, quintic_jerk, quintic_snap,
                                minco_pos, minco_vel, minco_acc, minco_jerk, minco_snap);
    printConsistency(cons);
    
    return result;
}

// 新增：测试优化的evaluate重载函数
template <int DIM>
void testOptimizedEvaluateOverload(
    const SplineTrajectory::QuinticSplineND<DIM>& quintic_spline,
    double dt)
{
    double start_time = quintic_spline.getTrajectory().getStartTime();
    double end_time = quintic_spline.getTrajectory().getEndTime();
    
    std::cout << "\n2.6) 优化的evaluate重载函数测试 (dt=" << dt << "):" << std::endl;
    
    // 1. 使用新的evaluate重载函数
    auto t1 = std::chrono::high_resolution_clock::now();
    auto quintic_pos_opt = quintic_spline.getTrajectory().evaluate(start_time, end_time, dt, 0);
    auto quintic_vel_opt = quintic_spline.getTrajectory().evaluate(start_time, end_time, dt, 1);
    auto quintic_acc_opt = quintic_spline.getTrajectory().evaluate(start_time, end_time, dt, 2);
    auto quintic_jerk_opt = quintic_spline.getTrajectory().evaluate(start_time, end_time, dt, 3);
    auto quintic_snap_opt = quintic_spline.getTrajectory().evaluate(start_time, end_time, dt, 4);
    auto t2 = std::chrono::high_resolution_clock::now();
    double quintic_opt_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    
    // 2. 使用优化便捷函数
    auto t3 = std::chrono::high_resolution_clock::now();
    auto quintic_pos_conv = quintic_spline.getTrajectory().getPos(start_time, end_time, dt);
    auto quintic_vel_conv = quintic_spline.getTrajectory().getVel(start_time, end_time, dt);
    auto quintic_acc_conv = quintic_spline.getTrajectory().getAcc(start_time, end_time, dt);
    auto quintic_jerk_conv = quintic_spline.getTrajectory().getJerk(start_time, end_time, dt);
    auto quintic_snap_conv = quintic_spline.getTrajectory().getSnap(start_time, end_time, dt);
    auto t4 = std::chrono::high_resolution_clock::now();
    double quintic_conv_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    
    // 3. 普通方法（生成时间序列然后查询）
    auto t5 = std::chrono::high_resolution_clock::now();
    auto time_seq = quintic_spline.getTrajectory().generateTimeSequence(start_time, end_time, dt);
    auto quintic_pos_trad = quintic_spline.getTrajectory().getPos(time_seq);
    auto quintic_vel_trad = quintic_spline.getTrajectory().getVel(time_seq);
    auto quintic_acc_trad = quintic_spline.getTrajectory().getAcc(time_seq);
    auto quintic_jerk_trad = quintic_spline.getTrajectory().getJerk(time_seq);
    auto quintic_snap_trad = quintic_spline.getTrajectory().getSnap(time_seq);
    auto t6 = std::chrono::high_resolution_clock::now();
    double quintic_trad_us = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
    
    std::cout << "  - 段批量查询优化evaluate()重载函数: " << quintic_opt_us << " μs (" << quintic_pos_opt.size() << " 点)" << std::endl;
    std::cout << "  - 段批量查询优化便捷函数(getPos/Vel/Acc等): " << quintic_conv_us << " μs (" << quintic_pos_conv.size() << " 点)" << std::endl;
    std::cout << "  - 普通方法(时间序列+批量查询): " << quintic_trad_us << " μs (" << quintic_pos_trad.size() << " 点)" << std::endl;
    
    if (quintic_trad_us > 0)
    {
        std::cout << "  - 性能提升比 (普通/优化): " << std::fixed << std::setprecision(3) 
                  << quintic_trad_us / quintic_opt_us << "x" << std::endl;
        std::cout << "  - 性能提升比 (普通/便捷): " << std::fixed << std::setprecision(3) 
                  << quintic_trad_us / quintic_conv_us << "x" << std::endl;
    }
    
    // 4. 验证结果一致性
    std::cout << "\n    段批量查询优化函数结果一致性验证:" << std::endl;
    double max_pos_diff = 0, max_vel_diff = 0, max_acc_diff = 0, max_jerk_diff = 0, max_snap_diff = 0;
    
    if (quintic_pos_opt.size() == quintic_pos_trad.size())
    {
        for (size_t i = 0; i < quintic_pos_opt.size(); ++i)
        {
            max_pos_diff = std::max(max_pos_diff, (quintic_pos_opt[i] - quintic_pos_trad[i]).norm());
            max_vel_diff = std::max(max_vel_diff, (quintic_vel_opt[i] - quintic_vel_trad[i]).norm());
            max_acc_diff = std::max(max_acc_diff, (quintic_acc_opt[i] - quintic_acc_trad[i]).norm());
            max_jerk_diff = std::max(max_jerk_diff, (quintic_jerk_opt[i] - quintic_jerk_trad[i]).norm());
            max_snap_diff = std::max(max_snap_diff, (quintic_snap_opt[i] - quintic_snap_trad[i]).norm());
        }
        
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "  - 位置最大差异: " << max_pos_diff << std::endl;
        std::cout << "  - 速度最大差异: " << max_vel_diff << std::endl;
        std::cout << "  - 加速度最大差异: " << max_acc_diff << std::endl;
        std::cout << "  - 加加速最大差异: " << max_jerk_diff << std::endl;
        std::cout << "  - 四阶导最大差异: " << max_snap_diff << std::endl;
        
        if (max_pos_diff < 1e-12 && max_vel_diff < 1e-12 && max_acc_diff < 1e-12 &&
            max_jerk_diff < 1e-12 && max_snap_diff < 1e-12)
        {
            std::cout << "  - 优化函数验证通过!" << std::endl;
        }
        else
        {
            std::cout << "  - 优化函数验证失败! 存在数值差异。" << std::endl;
        }
    }
    else
    {
        std::cout << "  - 结果大小不匹配: 优化=" << quintic_pos_opt.size() 
                  << ", 普通=" << quintic_pos_trad.size() << std::endl;
    }
}

// ===================================================================================
// ============================= 模板化的主测试逻辑 =================================
// ===================================================================================
template <int DIM>
void runQuinticTest()
{
    std::cout << "\n--- " << DIM << " 维五次样条测试 ---" << std::endl;
    int num_segments = 10;
    int num_points = num_segments + 1;
    double start_time = 0.0;
    int fit_runs = 1000;

    auto time_segments_vec = generateRandomTimeSegments(num_segments);
    auto waypoints_vec_eigen = generateRandomWaypoints(DIM, num_points);
    auto start_vel_eigen = generateRandomVector(DIM, 3.0);
    auto end_vel_eigen = generateRandomVector(DIM, 3.0);
    auto start_acc_eigen = generateRandomVector(DIM, 2.0);
    auto end_acc_eigen = generateRandomVector(DIM, 2.0);

    using QuinticSpline = SplineTrajectory::QuinticSplineND<DIM>;
    using VectorD = typename QuinticSpline::VectorType;

    SplineVector<VectorD> waypoints;
    for (const auto &v : waypoints_vec_eigen)
        waypoints.push_back(v);

    BoundaryConditions<DIM> bc(start_vel_eigen, start_acc_eigen,
                               end_vel_eigen, end_acc_eigen);

    std::cout << "2.1) 样条拟合性能 (平均 " << fit_runs << " 次):" << std::endl;

    // 五次样条拟合性能测试
    double total_quintic_fit_time_us = 0;
    for (int i = 0; i < fit_runs; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        QuinticSpline quintic_spline(time_segments_vec, waypoints, start_time, bc);
        auto t2 = std::chrono::high_resolution_clock::now();
        total_quintic_fit_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }
    std::cout << "  - QuinticSpline" << DIM << "D 拟合: " << std::fixed << std::setprecision(2)
              << total_quintic_fit_time_us / fit_runs << " μs" << std::endl;

    // 五次样条拟合性能测试

    // MINCO_S3NU 拟合性能测试
    int num_minco_instances = (DIM + 2) / 3;
    double total_minco_fit_time_us = 0;
    for (int i = 0; i < fit_runs; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<minco::MINCO_S3NU> minco_solvers(num_minco_instances);
        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;

            // 使用安全的数据填充方法
            Eigen::MatrixXd head = Eigen::MatrixXd::Zero(3, 3), tail = Eigen::MatrixXd::Zero(3, 3);

            // 安全地填充头部边界条件（位置、速度、加速度）
            Eigen::Vector3d head_pos = make3DFromND(waypoints_vec_eigen.front(), start_dim);
            Eigen::Vector3d head_vel = make3DFromND(start_vel_eigen, start_dim);
            Eigen::Vector3d head_acc = make3DFromND(start_acc_eigen, start_dim);

            Eigen::Vector3d tail_pos = make3DFromND(waypoints_vec_eigen.back(), start_dim);
            Eigen::Vector3d tail_vel = make3DFromND(end_vel_eigen, start_dim);
            Eigen::Vector3d tail_acc = make3DFromND(end_acc_eigen, start_dim);

            head.col(0) = head_pos;
            head.col(1) = head_vel;
            head.col(2) = head_acc;
            tail.col(0) = tail_pos;
            tail.col(1) = tail_vel;
            tail.col(2) = tail_acc;

            minco_solvers[j].setConditions(head, tail, num_segments);

            // 安全地设置中间waypoints
            Eigen::MatrixXd minco_points = Eigen::MatrixXd::Zero(3, num_points - 2);
            for (int k = 1; k < num_points - 1; ++k)
            {
                Eigen::Vector3d waypoint_3d = make3DFromND(waypoints_vec_eigen[k], start_dim);
                minco_points.col(k - 1) = waypoint_3d;
            }

            Eigen::VectorXd minco_times(num_segments);
            for (int k = 0; k < num_segments; ++k)
                minco_times(k) = time_segments_vec[k];

            minco_solvers[j].setParameters(minco_points, minco_times);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        total_minco_fit_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }
    std::cout << "  - MINCO_S3NU 模拟 " << DIM << "D 拟合: " << std::fixed << std::setprecision(2)
              << total_minco_fit_time_us / fit_runs << " μs (" << num_minco_instances << "个实例)" << std::endl;

    // 创建用于查询的样条实例
    QuinticSpline quintic_spline(time_segments_vec, waypoints, start_time, bc);

    // 创建用于查询的MINCO实例
    std::vector<minco::MINCO_S3NU> minco_solvers_for_query(num_minco_instances);
    std::vector<Trajectory<5>> minco_trajs(num_minco_instances); // 五次样条使用Trajectory<5>

    for (int j = 0; j < num_minco_instances; ++j)
    {
        int start_dim = j * 3;

        // 使用相同的安全数据填充方法
        Eigen::MatrixXd head = Eigen::MatrixXd::Zero(3, 3), tail = Eigen::MatrixXd::Zero(3, 3);

        Eigen::Vector3d head_pos = make3DFromND(waypoints_vec_eigen.front(), start_dim);
        Eigen::Vector3d head_vel = make3DFromND(start_vel_eigen, start_dim);
        Eigen::Vector3d head_acc = make3DFromND(start_acc_eigen, start_dim);

        Eigen::Vector3d tail_pos = make3DFromND(waypoints_vec_eigen.back(), start_dim);
        Eigen::Vector3d tail_vel = make3DFromND(end_vel_eigen, start_dim);
        Eigen::Vector3d tail_acc = make3DFromND(end_acc_eigen, start_dim);

        head.col(0) = head_pos;
        head.col(1) = head_vel;
        head.col(2) = head_acc;
        tail.col(0) = tail_pos;
        tail.col(1) = tail_vel;
        tail.col(2) = tail_acc;

        minco_solvers_for_query[j].setConditions(head, tail, num_segments);

        Eigen::MatrixXd minco_points = Eigen::MatrixXd::Zero(3, num_points - 2);
        for (int k = 1; k < num_points - 1; ++k)
        {
            Eigen::Vector3d waypoint_3d = make3DFromND(waypoints_vec_eigen[k], start_dim);
            minco_points.col(k - 1) = waypoint_3d;
        }

        Eigen::VectorXd minco_times(num_segments);
        for (int k = 0; k < num_segments; ++k)
            minco_times(k) = time_segments_vec[k];

        minco_solvers_for_query[j].setParameters(minco_points, minco_times);
        minco_solvers_for_query[j].getTrajectory(minco_trajs[j]);
    }

    double dt = 0.01;
    auto time_seq = quintic_spline.getTrajectory().generateTimeSequence(dt);
    int num_queries = time_seq.size();

    std::cout << "\n2.2) 单点查询性能 (" << num_queries << "个点):" << std::endl;

    // 五次样条单点查询性能
    std::function<void(int)> quintic_func = [&](int i)
    { quintic_spline.getTrajectory().getPos(time_seq[i]); };
    auto perf_quintic = measurePerformance(quintic_func, num_queries);
    printPerformance(perf_quintic, "  - QuinticSpline" + std::to_string(DIM) + "D");

    // MINCO单点查询性能
    std::function<void(int)> minco_func = [&](int i)
    {
        VectorD p = VectorD::Zero();
        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;
            int current_dim = std::min(3, DIM - start_dim);
            if (start_dim < DIM)
            {
                Eigen::Vector3d pos_3d = minco_trajs[j].getPos(time_seq[i]);
                p.segment(start_dim, current_dim) = pos_3d.head(current_dim);
            }
        }
    };
    auto perf_minco = measurePerformance(minco_func, num_queries);
    printPerformance(perf_minco, "  - MINCO_S3NU 模拟 " + std::to_string(DIM) + "D");

    std::cout << "\n2.3) 普通批量查询性能 (一次性查询 " << num_queries << " 个点):" << std::endl;

    // 五次样条批量查询
    auto t1 = std::chrono::high_resolution_clock::now();
    auto quintic_results_position = quintic_spline.getTrajectory().getPos(time_seq);
    auto quintic_results_velocity = quintic_spline.getTrajectory().getVel(time_seq);
    auto quintic_results_acceleration = quintic_spline.getTrajectory().getAcc(time_seq);
    auto quintic_results_jerk = quintic_spline.getTrajectory().getJerk(time_seq);
    auto quintic_results_snap = quintic_spline.getTrajectory().getSnap(time_seq);
    auto t2 = std::chrono::high_resolution_clock::now();
    double quintic_batch_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "  - QuinticSpline" << DIM << "D 普通批量查询: " << quintic_batch_us << " μs" << std::endl;

    // MINCO批量查询模拟
    auto t3 = std::chrono::high_resolution_clock::now();
    SplineVector<VectorD> minco_pos(num_queries), minco_vel(num_queries),
        minco_acc(num_queries), minco_jerk(num_queries), minco_snap(num_queries);

    for (size_t i = 0; i < time_seq.size(); ++i)
    {
        double t = time_seq[i];

        VectorD p = VectorD::Zero();
        VectorD v = VectorD::Zero();
        VectorD a = VectorD::Zero();
        VectorD jrk = VectorD::Zero();
        VectorD snp = VectorD::Zero();

        for (int j = 0; j < num_minco_instances; ++j)
        {
            int start_dim = j * 3;
            int current_dim = std::min(3, DIM - start_dim);

            if (start_dim < DIM)
            {
                Eigen::Vector3d pos_3d = minco_trajs[j].getPos(t);
                Eigen::Vector3d vel_3d = minco_trajs[j].getVel(t);
                Eigen::Vector3d acc_3d = minco_trajs[j].getAcc(t);
                Eigen::Vector3d jerk_3d = minco_trajs[j].getJer(t);
                Eigen::Vector3d snap_3d = minco_trajs[j].getSnap(t); // 五次样条的四阶导数

                p.segment(start_dim, current_dim) = pos_3d.head(current_dim);
                v.segment(start_dim, current_dim) = vel_3d.head(current_dim);
                a.segment(start_dim, current_dim) = acc_3d.head(current_dim);
                jrk.segment(start_dim, current_dim) = jerk_3d.head(current_dim);
                snp.segment(start_dim, current_dim) = snap_3d.head(current_dim);
            }
        }
        minco_pos[i] = p;
        minco_vel[i] = v;
        minco_acc[i] = a;
        minco_jerk[i] = jrk;
        minco_snap[i] = snp;
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    double minco_batch_us = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    std::cout << "  - MINCO_S3NU 普通批量查询: " << minco_batch_us << " μs" << std::endl;
    if (quintic_batch_us > 0)
        std::cout << "  - 性能比 (MINCO/QuinticSpline): " << std::fixed << std::setprecision(3)
                  << minco_batch_us / quintic_batch_us << "x" << std::endl;

    std::cout << "\n2.4) 普通批量查询一致性检查:" << std::endl;
    auto cons = testConsistency(quintic_results_position, quintic_results_velocity,
                                quintic_results_acceleration, quintic_results_jerk, quintic_results_snap,
                                minco_pos, minco_vel, minco_acc, minco_jerk, minco_snap);
    printConsistency(cons);

    std::cout << "\n2.5) PPolyND::derivative() 方法验证 (一阶导数):" << std::endl;
    auto ppoly = quintic_spline.getPPoly();
    auto d_ppoly = ppoly.derivative(1);
    auto vel_from_d_ppoly = d_ppoly.evaluate(time_seq, 0);
    double max_diff = 0.0, avg_diff = 0.0;
    if (!time_seq.empty())
    {
        for (size_t i = 0; i < time_seq.size(); ++i)
        {
            double diff = (quintic_results_velocity[i] - vel_from_d_ppoly[i]).norm();
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

    // ===== 新增测试：段化批量评估 =====
    std::cout << "\n2.6) 段化批量评估性能测试:" << std::endl;
    double test_dt = 0.01;
    auto batch_result = testSegmentedBatchEvaluation<DIM>(quintic_spline,minco_trajs, test_dt);
    printBatchPerformance(batch_result, "  - " + std::to_string(DIM) + "D");

    // ===== 新增测试：优化的evaluate重载函数 =====
    testOptimizedEvaluateOverload<DIM>(quintic_spline,test_dt);

    // ===== 新增测试：不同dt下的性能对比 =====
    std::cout << "\n2.7) 不同时间步长下的段化评估性能:" << std::endl;
    std::vector<double> test_dts = {0.001,0.0001};
    
    for (double test_dt_var : test_dts)
    {
        auto batch_result_var = testSegmentedBatchEvaluation<DIM>(quintic_spline,minco_trajs, test_dt_var);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  - dt=" << test_dt_var << ": QuinticSpline=" << batch_result_var.quintic_batch_evaluation_us 
                  << "μs, MINCO=" << batch_result_var.minco_batch_evaluation_us 
                  << "μs, 提升=" << batch_result_var.speedup_ratio << "x (" << batch_result_var.total_points << "点)" << std::endl;
    }

    // DIM=3 特殊测试: 轨迹能量计算函数一致性
    if constexpr (DIM == 3)
    {
        std::cout << "\n2.8) 轨迹能量计算一致性测试 (DIM=3):" << std::endl;
        
        // 获取 QuinticSplineND 的能量
        double quintic_energy = quintic_spline.getEnergy();
        
        // 获取 MINCO_S3NU 的能量 (只有第一个实例，因为DIM=3时只有一个MINCO实例)
        double minco_energy = 0.0;
        minco_solvers_for_query[0].getEnergy(minco_energy);
        
        // 计算差异
        double energy_diff = std::abs(quintic_energy - minco_energy);
        double relative_error = (quintic_energy != 0.0) ? (energy_diff / std::abs(quintic_energy)) : 0.0;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  - QuinticSplineND 能量: " << quintic_energy << std::endl;
        std::cout << "  - MINCO_S3NU 能量: " << minco_energy << std::endl;
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
    std::cout << "\n===== 多维 QuinticSplineND vs MINCO_S3NU 性能与一致性对比 =====\n";
    std::cout << "QuinticSplineND 使用块三对角矩阵与追赶法求解，使用 PPolyND 进行高效批量查询，\n";
    std::cout << "MINCO是常用的最小控制量轨迹库，本质是夹持多项式样条 (s=3，即五次夹持样条)\n";
    std::cout << "这里通过组合多个3D实例来模拟高维五次样条\n";


    runQuinticTest<1>();
    runQuinticTest<2>();
    runQuinticTest<3>();
    runQuinticTest<4>();
    runQuinticTest<5>();
    runQuinticTest<6>();
    runQuinticTest<7>();
    runQuinticTest<8>();
    runQuinticTest<9>();
    runQuinticTest<10>();

    std::cout << "\n===== 五次样条测试结束 =====\n"
              << std::endl;
    return 0;
}