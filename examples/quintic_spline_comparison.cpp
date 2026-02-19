/**
 * @file quintic_spline_comparison.cpp
 * @brief Compare cubic and quintic splines for the same trajectory
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    std::cout << "=== Cubic vs Quintic Spline Comparison ===" << std::endl;
    
    // Define a challenging 2D trajectory with sharp turns (row = one waypoint)
    QuinticSpline2D::MatrixType waypoints(7, 2);
    waypoints <<
        0.0, 0.0,
        2.0, 1.0,
        2.5, 3.0,
        1.0, 4.0,
        -0.5, 3.5,
        0.0, 2.0,
        1.5, 1.0;
    
    std::vector<double> time_points = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    // Set boundary conditions with specific velocities and accelerations
    BoundaryConditions<2> boundary;
    boundary.start_velocity = SplinePoint2d(1.0, 0.5);
    boundary.start_acceleration = SplinePoint2d(0.0, 0.0);
    boundary.end_velocity = SplinePoint2d(0.5, -0.2);
    boundary.end_acceleration = SplinePoint2d(0.0, 0.0);
    
    // Create both cubic and quintic splines
    CubicSpline2D cubic_spline(time_points, waypoints, boundary);
    QuinticSpline2D quintic_spline(time_points, waypoints, boundary);
    
    if (!cubic_spline.isInitialized() || !quintic_spline.isInitialized()) {
        std::cerr << "Failed to initialize splines!" << std::endl;
        return -1;
    }
    
    // Compare basic properties
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Spline Properties Comparison ===" << std::endl;
    std::cout << "Property                | Cubic      | Quintic    | Difference" << std::endl;
    std::cout << "------------------------|------------|------------|------------" << std::endl;
    
    double cubic_energy = cubic_spline.getEnergy();
    double quintic_energy = quintic_spline.getEnergy();
    std::cout << "Energy                  | " << std::setw(10) << cubic_energy 
              << " | " << std::setw(10) << quintic_energy 
              << " | " << std::setw(10) << (quintic_energy - cubic_energy) << std::endl;
    
    double cubic_length = cubic_spline.getTrajectory().getTrajectoryLength();
    double quintic_length = quintic_spline.getTrajectory().getTrajectoryLength();
    std::cout << "Length                  | " << std::setw(10) << cubic_length 
              << " | " << std::setw(10) << quintic_length 
              << " | " << std::setw(10) << (quintic_length - cubic_length) << std::endl;
    
    // Compare smoothness by looking at acceleration and jerk profiles
    std::cout << "\n=== Smoothness Analysis ===" << std::endl;
    std::vector<double> eval_times = cubic_spline.getTrajectory().generateTimeSequence(0.05);
    
    auto cubic_accelerations = cubic_spline.getTrajectory().evaluate(eval_times, Deriv::Acc);
    auto quintic_accelerations = quintic_spline.getTrajectory().evaluate(eval_times, Deriv::Acc);
    auto cubic_jerks = cubic_spline.getTrajectory().evaluate(eval_times, Deriv::Jerk);
    auto quintic_jerks = quintic_spline.getTrajectory().evaluate(eval_times, Deriv::Jerk);
    
    // Calculate RMS values for smoothness comparison
    double cubic_acc_rms = 0.0, quintic_acc_rms = 0.0;
    double cubic_jerk_rms = 0.0, quintic_jerk_rms = 0.0;
    
    for (size_t i = 0; i < cubic_accelerations.size(); ++i) {
        cubic_acc_rms += cubic_accelerations[i].squaredNorm();
        quintic_acc_rms += quintic_accelerations[i].squaredNorm();
        cubic_jerk_rms += cubic_jerks[i].squaredNorm();
        quintic_jerk_rms += quintic_jerks[i].squaredNorm();
    }
    
    cubic_acc_rms = std::sqrt(cubic_acc_rms / cubic_accelerations.size());
    quintic_acc_rms = std::sqrt(quintic_acc_rms / quintic_accelerations.size());
    cubic_jerk_rms = std::sqrt(cubic_jerk_rms / cubic_jerks.size());
    quintic_jerk_rms = std::sqrt(quintic_jerk_rms / quintic_jerks.size());
    
    std::cout << "RMS Acceleration        | " << std::setw(10) << cubic_acc_rms 
              << " | " << std::setw(10) << quintic_acc_rms 
              << " | " << std::setw(10) << (quintic_acc_rms - cubic_acc_rms) << std::endl;
    std::cout << "RMS Jerk                | " << std::setw(10) << cubic_jerk_rms 
              << " | " << std::setw(10) << quintic_jerk_rms 
              << " | " << std::setw(10) << (quintic_jerk_rms - cubic_jerk_rms) << std::endl;
    
    // Generate detailed comparison data
    std::cout << "\n=== Generating Comparison Data ===" << std::endl;
    
    auto cubic_positions = cubic_spline.getTrajectory().evaluate(eval_times, Deriv::Pos);
    auto quintic_positions = quintic_spline.getTrajectory().evaluate(eval_times, Deriv::Pos);
    auto cubic_velocities = cubic_spline.getTrajectory().evaluate(eval_times, Deriv::Vel);
    auto quintic_velocities = quintic_spline.getTrajectory().evaluate(eval_times, Deriv::Vel);
    
    // Save comparison data
    std::ofstream csv_file("spline_comparison.csv");
    csv_file << "time,cubic_x,cubic_y,quintic_x,quintic_y,cubic_vx,cubic_vy,quintic_vx,quintic_vy,"
             << "cubic_ax,cubic_ay,quintic_ax,quintic_ay,cubic_jx,cubic_jy,quintic_jx,quintic_jy\n";
    
    for (size_t i = 0; i < eval_times.size(); ++i) {
        csv_file << eval_times[i] << ","
                 << cubic_positions[i].x() << "," << cubic_positions[i].y() << ","
                 << quintic_positions[i].x() << "," << quintic_positions[i].y() << ","
                 << cubic_velocities[i].x() << "," << cubic_velocities[i].y() << ","
                 << quintic_velocities[i].x() << "," << quintic_velocities[i].y() << ","
                 << cubic_accelerations[i].x() << "," << cubic_accelerations[i].y() << ","
                 << quintic_accelerations[i].x() << "," << quintic_accelerations[i].y() << ","
                 << cubic_jerks[i].x() << "," << cubic_jerks[i].y() << ","
                 << quintic_jerks[i].x() << "," << quintic_jerks[i].y() << "\n";
    }
    csv_file.close();
    
    std::cout << "Comparison data saved to 'spline_comparison.csv'" << std::endl;
    
    // Performance comparison
    std::cout << "\n=== Performance Test ===" << std::endl;
    const int num_evaluations = 100000;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_evaluations; ++i) {
        double t = 6.0 * i / num_evaluations;
        volatile auto pos = cubic_spline.getTrajectory().evaluate(t, Deriv::Pos);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto cubic_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_evaluations; ++i) {
        double t = 6.0 * i / num_evaluations;
        volatile auto pos = quintic_spline.getTrajectory().evaluate(t, Deriv::Pos);
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto quintic_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Cubic spline:   " << cubic_duration.count() << " μs for " << num_evaluations << " evaluations" << std::endl;
    std::cout << "Quintic spline: " << quintic_duration.count() << " μs for " << num_evaluations << " evaluations" << std::endl;
    std::cout << "Performance ratio: " << (double)quintic_duration.count() / cubic_duration.count() << "x" << std::endl;
    
    return 0;
}
