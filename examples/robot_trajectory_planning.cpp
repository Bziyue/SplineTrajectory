/**
 * @file robot_trajectory_planning.cpp
 * @brief Practical example: 6-DOF robot arm trajectory planning
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "SplineTrajectory.hpp"

int main() {
    using namespace SplineTrajectory;
    
    std::cout << "=== 6-DOF Robot Arm Trajectory Planning ===" << std::endl;
    
    // Define joint space waypoints (6 joints in radians)
    SplineVector<SplinePoint6d> joint_waypoints;
    
    // Home position
    SplinePoint6d home_pos;
    home_pos << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    joint_waypoints.push_back(home_pos);
    
    // Intermediate 1
    SplinePoint6d intermediate1;
    intermediate1 << M_PI/4, M_PI/6, -M_PI/3, 0.0, M_PI/4, 0.0;
    joint_waypoints.push_back(intermediate1);
    
    // Target position
    SplinePoint6d target_pos;
    target_pos << M_PI/2, M_PI/3, -M_PI/2, M_PI/4, M_PI/2, M_PI/6;
    joint_waypoints.push_back(target_pos);
    
    // Intermediate 2
    SplinePoint6d intermediate2;
    intermediate2 << M_PI/3, M_PI/4, -M_PI/4, M_PI/6, M_PI/3, M_PI/4;
    joint_waypoints.push_back(intermediate2);
    
    // Return home
    SplinePoint6d return_home;
    return_home << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    joint_waypoints.push_back(return_home);
    
    // Define time points with varying speeds
    std::vector<double> time_points = {0.0, 2.0, 4.0, 6.0, 8.0};
    
    // Set boundary conditions for smooth start/stop
    BoundaryConditions<6> boundary_conditions;
    boundary_conditions.start_velocity = SplinePoint6d::Zero();
    boundary_conditions.start_acceleration = SplinePoint6d::Zero();
    boundary_conditions.end_velocity = SplinePoint6d::Zero();
    boundary_conditions.end_acceleration = SplinePoint6d::Zero();
    
    // Create quintic spline for smooth robot motion
    QuinticSpline6D robot_trajectory(time_points, joint_waypoints, boundary_conditions);
    
    if (!robot_trajectory.isInitialized()) {
        std::cerr << "Failed to initialize robot trajectory!" << std::endl;
        return -1;
    }
    
    std::cout << "Robot trajectory created successfully!" << std::endl;
    std::cout << "Total duration: " << robot_trajectory.getDuration() << " seconds" << std::endl;
    std::cout << "Trajectory energy: " << robot_trajectory.getEnergy() << std::endl;
    
    // Define joint limits and velocity/acceleration constraints
    SplinePoint6d joint_limits_min;
    joint_limits_min << -M_PI, -M_PI/2, -2*M_PI/3, -M_PI, -M_PI, -M_PI;
    
    SplinePoint6d joint_limits_max;
    joint_limits_max << M_PI, M_PI/2, 2*M_PI/3, M_PI, M_PI, M_PI;
    
    SplinePoint6d velocity_limits = SplinePoint6d::Constant(M_PI);  // 180 deg/s max
    SplinePoint6d acceleration_limits = SplinePoint6d::Constant(2*M_PI);  // 360 deg/s² max
    
    // Validate trajectory against constraints
    std::cout << "\n=== Trajectory Validation ===" << std::endl;
    
    std::vector<double> eval_times = robot_trajectory.getTrajectory().generateTimeSequence(0.01);
    auto positions = robot_trajectory.getTrajectory().evaluate(eval_times, Deriv::Pos);
    auto velocities = robot_trajectory.getTrajectory().evaluate(eval_times, Deriv::Vel);
    auto accelerations = robot_trajectory.getTrajectory().evaluate(eval_times, Deriv::Acc);
    
    bool position_valid = true, velocity_valid = true, acceleration_valid = true;
    double max_velocity = 0.0, max_acceleration = 0.0;
    
    for (size_t i = 0; i < positions.size(); ++i) {
        // Check position limits
        for (int j = 0; j < 6; ++j) {
            if (positions[i](j) < joint_limits_min(j) || positions[i](j) > joint_limits_max(j)) {
                position_valid = false;
            }
            if (std::abs(velocities[i](j)) > velocity_limits(j)) {
                velocity_valid = false;
            }
            if (std::abs(accelerations[i](j)) > acceleration_limits(j)) {
                acceleration_valid = false;
            }
        }
        
        max_velocity = std::max(max_velocity, velocities[i].lpNorm<Eigen::Infinity>());
        max_acceleration = std::max(max_acceleration, accelerations[i].lpNorm<Eigen::Infinity>());
    }
    
    std::cout << "Position limits:    " << (position_valid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Velocity limits:    " << (velocity_valid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Acceleration limits:" << (acceleration_valid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max velocity:       " << max_velocity << " rad/s" << std::endl;
    std::cout << "Max acceleration:   " << max_acceleration << " rad/s²" << std::endl;
    
    // Generate robot control commands
    std::cout << "\n=== Generating Robot Commands ===" << std::endl;
    
    const double control_frequency = 100.0;  // 100 Hz control loop
    const double dt = 1.0 / control_frequency;
    
    std::ofstream cmd_file("robot_commands.csv");
    cmd_file << "time,j1_pos,j2_pos,j3_pos,j4_pos,j5_pos,j6_pos,"
             << "j1_vel,j2_vel,j3_vel,j4_vel,j5_vel,j6_vel,"
             << "j1_acc,j2_acc,j3_acc,j4_acc,j5_acc,j6_acc\n";
    
    std::cout << "Generating commands at " << control_frequency << " Hz..." << std::endl;
    
    int command_count = 0;
    for (double t = 0.0; t <= robot_trajectory.getEndTime(); t += dt) {
        SplinePoint6d pos = robot_trajectory.getTrajectory().evaluate(t, Deriv::Pos);
        SplinePoint6d vel = robot_trajectory.getTrajectory().evaluate(t, Deriv::Vel);
        SplinePoint6d acc = robot_trajectory.getTrajectory().evaluate(t, Deriv::Acc);
        
        cmd_file << t;
        for (int i = 0; i < 6; ++i) cmd_file << "," << pos(i);
        for (int i = 0; i < 6; ++i) cmd_file << "," << vel(i);
        for (int i = 0; i < 6; ++i) cmd_file << "," << acc(i);
        cmd_file << "\n";
        
        command_count++;
    }
    
    cmd_file.close();
    std::cout << "Generated " << command_count << " robot commands" << std::endl;
    std::cout << "Commands saved to 'robot_commands.csv'" << std::endl;
    
    // Motion analysis
    std::cout << "\n=== Motion Analysis ===" << std::endl;
    
    // Calculate joint space distances
    std::vector<double> joint_distances(6, 0.0);
    for (size_t i = 1; i < positions.size(); ++i) {
        SplinePoint6d diff = positions[i] - positions[i-1];
        for (int j = 0; j < 6; ++j) {
            joint_distances[j] += std::abs(diff(j));
        }
    }
    
    std::cout << "Joint distances traveled:" << std::endl;
    for (int j = 0; j < 6; ++j) {
        std::cout << "  Joint " << (j+1) << ": " << joint_distances[j] << " rad (" 
                  << joint_distances[j] * 180.0 / M_PI << " deg)" << std::endl;
    }
    
    // Calculate smoothness metrics
    double total_jerk = 0.0;
    auto jerks = robot_trajectory.getTrajectory().evaluate(eval_times, Deriv::Jerk);
    for (const auto& jerk : jerks) {
        total_jerk += jerk.squaredNorm();
    }
    double rms_jerk = std::sqrt(total_jerk / jerks.size());
    
    std::cout << "RMS Jerk: " << rms_jerk << " rad/s³" << std::endl;
    std::cout << "Smoothness score: " << 1.0 / (1.0 + rms_jerk) << " (higher is better)" << std::endl;
    
    return 0;
}