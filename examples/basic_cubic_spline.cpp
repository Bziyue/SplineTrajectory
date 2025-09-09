/**
 * @file basic_cubic_spline.cpp
 * @brief Basic usage example of CubicSplineND for 3D trajectory generation
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
    
    std::cout << "=== Basic Cubic Spline Example ===" << std::endl;
    
    // Define 3D waypoints for a trajectory
    SplineVector<SplinePoint3d> waypoints = {
        SplinePoint3d(0.0, 0.0, 0.0),     // Start point
        SplinePoint3d(1.0, 2.0, 1.0),     // Waypoint 1
        SplinePoint3d(3.0, 1.0, 2.0),     // Waypoint 2
        SplinePoint3d(4.0, 3.0, 0.5),     // Waypoint 3
        SplinePoint3d(5.0, 0.0, 1.5)      // End point
    };
    
    // Define corresponding time points
    std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0, 6.0};
    
    // Set up boundary conditions
    BoundaryConditions<3> boundary_conditions;
    boundary_conditions.start_velocity = SplinePoint3d(0.5, 0.0, 0.2);
    boundary_conditions.end_velocity = SplinePoint3d(0.0, -0.5, 0.0);
    
    // Create the cubic spline
    CubicSpline3D spline(time_points, waypoints, boundary_conditions);
    
    if (!spline.isInitialized()) {
        std::cerr << "Failed to initialize spline!" << std::endl;
        return -1;
    }
    
    std::cout << "Spline created successfully!" << std::endl;
    std::cout << "Number of segments: " << spline.getNumSegments() << std::endl;
    std::cout << "Start time: " << spline.getStartTime() << std::endl;
    std::cout << "End time: " << spline.getEndTime() << std::endl;
    std::cout << "Duration: " << spline.getDuration() << std::endl;
    std::cout << "Trajectory energy: " << spline.getEnergy() << std::endl;
    
    // Evaluate at specific time points
    std::cout << "\n=== Trajectory Evaluation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    for (double t = 0.0; t <= 6.0; t += 0.5) {
        SplinePoint3d pos = spline.getTrajectory().getPos(t);
        SplinePoint3d vel = spline.getTrajectory().getVel(t);
        SplinePoint3d acc = spline.getTrajectory().getAcc(t);
        
        std::cout << "t=" << std::setw(4) << t 
                  << " pos=[" << std::setw(6) << pos.x() << "," << std::setw(6) << pos.y() << "," << std::setw(6) << pos.z() << "]"
                  << " vel=[" << std::setw(6) << vel.x() << "," << std::setw(6) << vel.y() << "," << std::setw(6) << vel.z() << "]"
                  << " |vel|=" << std::setw(6) << vel.norm() << std::endl;
    }
    
    // Calculate trajectory properties
    double trajectory_length = spline.getTrajectory().getTrajectoryLength();
    std::cout << "\nTrajectory length: " << trajectory_length << std::endl;
    
    // Batch evaluation for plotting
    std::cout << "\n=== Generating CSV Output ===" << std::endl;
    std::vector<double> eval_times = spline.getTrajectory().generateTimeSequence(0.02);
    auto positions = spline.getTrajectory().getPos(eval_times);
    auto velocities = spline.getTrajectory().getVel(eval_times);
    auto accelerations = spline.getTrajectory().getAcc(eval_times);
    
    // Save to CSV file for plotting
    std::ofstream csv_file("cubic_spline_trajectory.csv");
    csv_file << "time,x,y,z,vx,vy,vz,ax,ay,az,speed,accel_mag\n";
    
    for (size_t i = 0; i < eval_times.size(); ++i) {
        csv_file << eval_times[i] << ","
                 << positions[i].x() << "," << positions[i].y() << "," << positions[i].z() << ","
                 << velocities[i].x() << "," << velocities[i].y() << "," << velocities[i].z() << ","
                 << accelerations[i].x() << "," << accelerations[i].y() << "," << accelerations[i].z() << ","
                 << velocities[i].norm() << "," << accelerations[i].norm() << "\n";
    }
    csv_file.close();
    
    std::cout << "Trajectory saved to 'cubic_spline_trajectory.csv'" << std::endl;
    std::cout << "Generated " << eval_times.size() << " data points" << std::endl;
    
    return 0;
}