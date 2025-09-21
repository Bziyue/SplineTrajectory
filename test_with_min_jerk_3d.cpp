#include "traj_min_jerk.hpp"
#include "SplineTrajectory.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <iomanip>

using namespace std;
using namespace Eigen;
using namespace SplineTrajectory;

class RandomRouteGenerator
{
public:
    RandomRouteGenerator(Array3d l, Array3d u)
        : lBound(l), uBound(u), uniformReal(0.0, 1.0) {}

    inline MatrixXd generate(int N)
    {
        MatrixXd route(3, N + 1);
        Array3d temp;
        route.col(0).setZero();
        for (int i = 0; i < N; i++)
        {
            temp << uniformReal(gen), uniformReal(gen), uniformReal(gen);
            temp = (uBound - lBound) * temp + lBound;
            route.col(i + 1) << temp;
        }
        return route;
    }

private:
    Array3d lBound;
    Array3d uBound;
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> uniformReal;
};

VectorXd allocateTime(const MatrixXd &wayPs,
                      double vel,
                      double acc)
{
    int N = (int)(wayPs.cols()) - 1;
    VectorXd durations(N);
    if (N > 0)
    {
        Eigen::Vector3d p0, p1;
        double dtxyz, D, acct, accd, dcct, dccd, t1, t2, t3;
        for (int k = 0; k < N; k++)
        {
            p0 = wayPs.col(k);
            p1 = wayPs.col(k + 1);
            D = (p1 - p0).norm();

            acct = vel / acc;
            accd = (acc * acct * acct / 2);
            dcct = vel / acc;
            dccd = acc * dcct * dcct / 2;

            if (D < accd + dccd)
            {
                t1 = sqrt(acc * D) / acc;
                t2 = (acc * t1) / acc;
                dtxyz = t1 + t2;
            }
            else
            {
                t1 = acct;
                t2 = (D - accd - dccd) / vel;
                t3 = dcct;
                dtxyz = t1 + t2 + t3;
            }

            durations(k) = dtxyz;
        }
    }

    return durations;
}

SplineVector3D convertMatrixToSplineVector(const MatrixXd &route)
{
    SplineVector3D spline_points;
    spline_points.reserve(route.cols());
    
    for (int i = 0; i < route.cols(); ++i)
    {
        SplinePoint3d point;
        point << route(0, i), route(1, i), route(2, i);
        spline_points.push_back(point);
    }
    
    return spline_points;
}

std::vector<double> convertVectorXdToStdVector(const VectorXd &eigen_vec)
{
    std::vector<double> std_vec(eigen_vec.size());
    for (int i = 0; i < eigen_vec.size(); ++i)
    {
        std_vec[i] = eigen_vec(i);
    }
    return std_vec;
}

// Generate random time points within trajectory duration
std::vector<double> generateRandomTimePoints(double start_time, double end_time, int num_points, std::mt19937_64& gen)
{
    std::uniform_real_distribution<double> dist(start_time, end_time);
    std::vector<double> time_points;
    time_points.reserve(num_points);
    
    for (int i = 0; i < num_points; ++i)
    {
        time_points.push_back(dist(gen));
    }
    
    return time_points;
}

// Generate uniform time sequence
std::vector<double> generateUniformTimeSequence(double start_time, double end_time, double dt)
{
    std::vector<double> time_sequence;
    double current_time = start_time;
    
    while (current_time <= end_time)
    {
        time_sequence.push_back(current_time);
        current_time += dt;
    }
    
    if (time_sequence.empty() || time_sequence.back() < end_time)
    {
        time_sequence.push_back(end_time);
    }
    
    return time_sequence;
}

// Calculate maximum error between two trajectories
double calculateMaxError(const std::vector<Vector3d>& traj1, const std::vector<Vector3d>& traj2)
{
    if (traj1.size() != traj2.size()) return -1.0;
    
    double max_error = 0.0;
    for (size_t i = 0; i < traj1.size(); ++i)
    {
        double error = (traj1[i] - traj2[i]).norm();
        max_error = std::max(max_error, error);
    }
    return max_error;
}

void runEvaluationTests()
{
    std::cout << "\n=== EVALUATION PERFORMANCE AND CONSISTENCY TESTS ===" << std::endl;
    std::cout << "Using 128 segments trajectory" << std::endl;
    
    RandomRouteGenerator routeGen(Array3d(-16, -16, -16), Array3d(16, 16, 16));
    min_jerk::JerkOpt jerkOpt;
    min_jerk::Trajectory minJerkTraj;
    QuinticSpline3D quinticSpline;
    
    std::mt19937_64 gen;
    
    // Generate test trajectory with 128 segments
    const int test_segments = 128;
    MatrixXd route = routeGen.generate(test_segments);
    Matrix3d iS, fS;
    iS.setZero();
    fS.setZero();
    iS.col(0) << route.leftCols<1>();
    fS.col(0) << route.rightCols<1>();
    VectorXd ts = allocateTime(route, 3.0, 3.0);
    
    // Setup MinJerk trajectory
    jerkOpt.reset(iS, fS, route.cols() - 1);
    jerkOpt.generate(route.block(0, 1, 3, test_segments - 1), ts);
    jerkOpt.getTraj(minJerkTraj);
    
    // Setup QuinticSpline trajectory
    SplineVector3D spline_points = convertMatrixToSplineVector(route);
    std::vector<double> time_segments = convertVectorXdToStdVector(ts);
    BoundaryConditions<3> boundary;
    boundary.start_velocity.setZero();
    boundary.start_acceleration.setZero();
    boundary.end_velocity.setZero();
    boundary.end_acceleration.setZero();
    quinticSpline.update(time_segments, spline_points, 0.0, boundary);
    
    double total_duration = minJerkTraj.getTotalDuration();
    std::cout << "Trajectory duration: " << total_duration << " seconds" << std::endl;
    
    // Test parameters
    const int num_random_points = 10000;
    const double dt_batch = 0.01;
    const int num_consistency_points = 1000;
    
    std::chrono::high_resolution_clock::time_point t0, t1;
    
    // 1. Random time point evaluation performance
    std::cout << "\n1. Random Time Point Evaluation Performance (" << num_random_points << " points)" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    
    auto random_times = generateRandomTimePoints(0.0, total_duration, num_random_points, gen);
    
    // MinJerk random evaluation - Position
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<Vector3d> mj_pos_random(num_random_points);
    for (int i = 0; i < num_random_points; ++i)
    {
        mj_pos_random[i] = minJerkTraj.getPos(random_times[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mj_random_pos_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline random evaluation - Position
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_pos_random = quinticSpline.getTrajectory().getPos(random_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_random_pos_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // MinJerk random evaluation - Velocity
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<Vector3d> mj_vel_random(num_random_points);
    for (int i = 0; i < num_random_points; ++i)
    {
        mj_vel_random[i] = minJerkTraj.getVel(random_times[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mj_random_vel_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline random evaluation - Velocity
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_vel_random = quinticSpline.getTrajectory().getVel(random_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_random_vel_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // MinJerk random evaluation - Acceleration
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<Vector3d> mj_acc_random(num_random_points);
    for (int i = 0; i < num_random_points; ++i)
    {
        mj_acc_random[i] = minJerkTraj.getAcc(random_times[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mj_random_acc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline random evaluation - Acceleration
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_acc_random = quinticSpline.getTrajectory().getAcc(random_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_random_acc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    std::cout << "Evaluation Type\tMinJerk(μs)\tQuinticSpline(μs)\tRatio(MJ/QS)" << std::endl;
    std::cout << "Position\t" << std::fixed << std::setprecision(1) 
              << mj_random_pos_time << "\t\t" << qs_random_pos_time << "\t\t\t" 
              << std::setprecision(3) << (mj_random_pos_time / qs_random_pos_time) << std::endl;
    std::cout << "Velocity\t" << std::fixed << std::setprecision(1) 
              << mj_random_vel_time << "\t\t" << qs_random_vel_time << "\t\t\t" 
              << std::setprecision(3) << (mj_random_vel_time / qs_random_vel_time) << std::endl;
    std::cout << "Acceleration\t" << std::fixed << std::setprecision(1) 
              << mj_random_acc_time << "\t\t" << qs_random_acc_time << "\t\t\t" 
              << std::setprecision(3) << (mj_random_acc_time / qs_random_acc_time) << std::endl;
    
    // 2. Batch evaluation performance
    std::cout << "\n2. Batch Evaluation Performance (dt=" << dt_batch << ")" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    
    auto uniform_times = generateUniformTimeSequence(0.0, total_duration, dt_batch);
    auto segmented_times = quinticSpline.getTrajectory().generateSegmentedTimeSequence(0.0, total_duration, dt_batch);
    std::cout << "Number of evaluation points: " << uniform_times.size() << std::endl;
    
    // MinJerk batch evaluation (using for loop) - Position
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<Vector3d> mj_pos_batch(uniform_times.size());
    for (size_t i = 0; i < uniform_times.size(); ++i)
    {
        mj_pos_batch[i] = minJerkTraj.getPos(uniform_times[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mj_batch_pos_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline normal batch evaluation - Position
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_pos_batch_normal = quinticSpline.getTrajectory().getPos(uniform_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_batch_pos_normal_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline optimized batch evaluation - Position
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_pos_batch_optimized = quinticSpline.getTrajectory().getPos(segmented_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_batch_pos_optimized_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // MinJerk batch evaluation - Velocity
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<Vector3d> mj_vel_batch(uniform_times.size());
    for (size_t i = 0; i < uniform_times.size(); ++i)
    {
        mj_vel_batch[i] = minJerkTraj.getVel(uniform_times[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mj_batch_vel_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline batch evaluation - Velocity
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_vel_batch_normal = quinticSpline.getTrajectory().getVel(uniform_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_batch_vel_normal_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_vel_batch_optimized = quinticSpline.getTrajectory().getVel(segmented_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_batch_vel_optimized_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // MinJerk batch evaluation - Acceleration
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<Vector3d> mj_acc_batch(uniform_times.size());
    for (size_t i = 0; i < uniform_times.size(); ++i)
    {
        mj_acc_batch[i] = minJerkTraj.getAcc(uniform_times[i]);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mj_batch_acc_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    // QuinticSpline batch evaluation - Acceleration
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_acc_batch_normal = quinticSpline.getTrajectory().getAcc(uniform_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_batch_acc_normal_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    t0 = std::chrono::high_resolution_clock::now();
    auto qs_acc_batch_optimized = quinticSpline.getTrajectory().getAcc(segmented_times);
    t1 = std::chrono::high_resolution_clock::now();
    double qs_batch_acc_optimized_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    
    std::cout << "Eval Type\tMinJerk(μs)\tQS_Normal(μs)\tQS_Optimized(μs)\tMJ/QS_Normal\tMJ/QS_Opt" << std::endl;
    std::cout << "Position\t" << std::fixed << std::setprecision(0) 
              << mj_batch_pos_time << "\t\t" << qs_batch_pos_normal_time << "\t\t" << qs_batch_pos_optimized_time << "\t\t\t"
              << std::setprecision(2) << (mj_batch_pos_time / qs_batch_pos_normal_time) << "\t\t"
              << (mj_batch_pos_time / qs_batch_pos_optimized_time) << std::endl;
    std::cout << "Velocity\t" << std::fixed << std::setprecision(0) 
              << mj_batch_vel_time << "\t\t" << qs_batch_vel_normal_time << "\t\t" << qs_batch_vel_optimized_time << "\t\t\t"
              << std::setprecision(2) << (mj_batch_vel_time / qs_batch_vel_normal_time) << "\t\t"
              << (mj_batch_vel_time / qs_batch_vel_optimized_time) << std::endl;
    std::cout << "Acceleration\t" << std::fixed << std::setprecision(0) 
              << mj_batch_acc_time << "\t\t" << qs_batch_acc_normal_time << "\t\t" << qs_batch_acc_optimized_time << "\t\t\t"
              << std::setprecision(2) << (mj_batch_acc_time / qs_batch_acc_normal_time) << "\t\t"
              << (mj_batch_acc_time / qs_batch_acc_optimized_time) << std::endl;
    
    // 3. Consistency verification
    std::cout << "\n3. Consistency Verification (" << num_consistency_points << " points)" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    
    auto consistency_times = generateRandomTimePoints(0.0, total_duration, num_consistency_points, gen);
    
    // Get results from both methods
    std::vector<Vector3d> mj_pos_consistency(num_consistency_points);
    std::vector<Vector3d> mj_vel_consistency(num_consistency_points);
    std::vector<Vector3d> mj_acc_consistency(num_consistency_points);
    
    for (int i = 0; i < num_consistency_points; ++i)
    {
        mj_pos_consistency[i] = minJerkTraj.getPos(consistency_times[i]);
        mj_vel_consistency[i] = minJerkTraj.getVel(consistency_times[i]);
        mj_acc_consistency[i] = minJerkTraj.getAcc(consistency_times[i]);
    }
    
    auto qs_pos_consistency = quinticSpline.getTrajectory().getPos(consistency_times);
    auto qs_vel_consistency = quinticSpline.getTrajectory().getVel(consistency_times);
    auto qs_acc_consistency = quinticSpline.getTrajectory().getAcc(consistency_times);
    
    // Convert SplineVector to std::vector for error calculation
    std::vector<Vector3d> qs_pos_consistency_vec(qs_pos_consistency.begin(), qs_pos_consistency.end());
    std::vector<Vector3d> qs_vel_consistency_vec(qs_vel_consistency.begin(), qs_vel_consistency.end());
    std::vector<Vector3d> qs_acc_consistency_vec(qs_acc_consistency.begin(), qs_acc_consistency.end());
    
    double pos_max_error = calculateMaxError(mj_pos_consistency, qs_pos_consistency_vec);
    double vel_max_error = calculateMaxError(mj_vel_consistency, qs_vel_consistency_vec);
    double acc_max_error = calculateMaxError(mj_acc_consistency, qs_acc_consistency_vec);
    
    std::cout << "Derivative\tMax Error" << std::endl;
    std::cout << "Position\t" << std::scientific << std::setprecision(3) << pos_max_error << std::endl;
    std::cout << "Velocity\t" << vel_max_error << std::endl;
    std::cout << "Acceleration\t" << acc_max_error << std::endl;
    
    // energy consistency
    double mj_energy = jerkOpt.getObjective();
    double qs_energy = quinticSpline.getEnergy();
    bool energy_consistent = std::abs(mj_energy - qs_energy) < 1e-6;

    // Check if results are consistent (within reasonable tolerance)
    const double tolerance_pos = 1e-10;
    const double tolerance_vel = 1e-8;
    const double tolerance_acc = 1e-6;
    
    bool pos_consistent = pos_max_error < tolerance_pos;
    bool vel_consistent = vel_max_error < tolerance_vel;
    bool acc_consistent = acc_max_error < tolerance_acc;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nConsistency Check (Tolerance: pos=" << tolerance_pos 
              << ", vel=" << tolerance_vel << ", acc=" << tolerance_acc << "):" << std::endl;
    std::cout << "Position: " << (pos_consistent ? "PASS" : "FAIL") << std::endl;
    std::cout << "Velocity: " << (vel_consistent ? "PASS" : "FAIL") << std::endl;
    std::cout << "Acceleration: " << (acc_consistent ? "PASS" : "FAIL") << std::endl;
    std::cout << "Energy: " << (energy_consistent ? "PASS" : "FAIL") 
              << " (MinJerk: " << mj_energy << ", QuinticSpline: " << qs_energy << ")" << std::endl;

    if (pos_consistent && vel_consistent && acc_consistent && energy_consistent)
    {
        std::cout << "\n✓ All consistency tests PASSED!" << std::endl;
    }
    else
    {
        std::cout << "\n✗ Some consistency tests FAILED!" << std::endl;
    }
}

int main()
{
    RandomRouteGenerator routeGen(Array3d(-16, -16, -16), Array3d(16, 16, 16));

    min_jerk::JerkOpt jerkOpt;
    min_jerk::Trajectory minJerkTraj;
    QuinticSpline3D quinticSpline;

    MatrixXd route;
    VectorXd ts;
    Matrix3d iS, fS;
    iS.setZero();
    fS.setZero();
    
    std::chrono::high_resolution_clock::time_point tc0, tc1, tc2;
    double minJerkTime_us, quinticSplineTime_us;
    int groupSize = 1000;

    std::vector<int> test_segments;
    for (int i = 2; i <= 512; i++)
    {
        test_segments.push_back(i);
    }
    
    test_segments.push_back(1000);
    test_segments.push_back(2000);
    test_segments.push_back(3000);
    test_segments.push_back(4000);
    test_segments.push_back(5000);

    std::cout << "Performance Comparison: MinJerk vs QuinticSpline (3D)" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Segments\tMinJerk(μs)\tQuinticSpline(μs)\tRatio(Quintic/MinJerk)" << std::endl;

    for (int segments : test_segments)
    {
        minJerkTime_us = quinticSplineTime_us = 0.0;
        
        for (int j = 0; j < groupSize; j++)
        {
            // 生成随机路径点
            route = routeGen.generate(segments);
            iS.col(0) << route.leftCols<1>();
            fS.col(0) << route.rightCols<1>();
            ts = allocateTime(route, 3.0, 3.0);

            // 测试 MinJerk
            tc0 = std::chrono::high_resolution_clock::now();
            jerkOpt.reset(iS, fS, route.cols() - 1);
            jerkOpt.generate(route.block(0, 1, 3, segments - 1), ts);
            jerkOpt.getTraj(minJerkTraj);
            tc1 = std::chrono::high_resolution_clock::now();
            
            minJerkTime_us += std::chrono::duration_cast<std::chrono::microseconds>(tc1 - tc0).count();

            // 准备 QuinticSpline 数据
            SplineVector3D spline_points = convertMatrixToSplineVector(route);
            std::vector<double> time_segments = convertVectorXdToStdVector(ts);
            
            // 设置边界条件（零速度和加速度）
            BoundaryConditions<3> boundary;
            boundary.start_velocity.setZero();
            boundary.start_acceleration.setZero();
            boundary.end_velocity.setZero();
            boundary.end_acceleration.setZero();

            // 测试 QuinticSpline - 直接使用时间段构造
            tc1 = std::chrono::high_resolution_clock::now();
            quinticSpline.update(time_segments, spline_points, 0.0, boundary);
            tc2 = std::chrono::high_resolution_clock::now();
            
            quinticSplineTime_us += std::chrono::duration_cast<std::chrono::microseconds>(tc2 - tc1).count();
        }

        double avgMinJerkTime = minJerkTime_us / groupSize;
        double avgQuinticTime = quinticSplineTime_us / groupSize;
        double ratio = avgQuinticTime / avgMinJerkTime;

        std::cout << segments << "\t\t" 
                  << std::fixed << std::setprecision(2) << avgMinJerkTime << "\t\t" 
                  << avgQuinticTime << "\t\t\t" 
                  << std::setprecision(3) << ratio << std::endl;
    }

    std::cout << "\nConstruction test completed successfully!" << std::endl;
    std::cout << "Note: Lower time is better. Ratio > 1 means MinJerk is faster, ratio < 1 means QuinticSpline is faster." << std::endl;

    // Run evaluation performance and consistency tests
    runEvaluationTests();

    return 0;
}