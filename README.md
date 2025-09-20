# SplineTrajectory

SplineTrajectory is a high-performance, header-only C++ library for generating smooth, N-dimensional spline trajectories. This library provides **MINCO-equivalent** cubic、quintic and septic spline interpolation with boundary conditions support, making it ideal for robotics, path planning, and trajectory generation applications.

**English** | [中文](README_zh.md)


## Key Features

- **MINCO Equivalent**: Achieves minimum acceleration, jerk, and snap trajectories, just like MINCO.
    
- **High Performance**: Outperforms traditional methods by using a specialized **block tridiagonal matrix solver** (Thomas algorithm) instead of general LU decomposition.
    
- **Template-Based**: Fully templated for **arbitrary dimensions** (1D to ND) with compile-time optimizations.
    
- **Flexible & Efficient**: Supports multiple time specifications, optimized batch evaluation, and provides derivatives (velocity, acceleration, jerk, snap).
    
- **Eigen Integration**: Seamlessly uses the Eigen library for all linear algebra operations.
    
- **Header-Only**: Easy to integrate into any project by just including the header.

## Requirements

- C++11 or later
- Eigen 3.3 or later
- CMake 3.10+ (for building examples and tests)

## Quick Start

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
SplineTrajectory also outperforms [large_scale_traj_optimizer](https://github.com/ZJU-FAST-Lab/large_scale_traj_optimizer) in both trajectory generation and evaluation. To see the test results, run ./test_with_min_jerk_3d.

For a complete motion planning toolkit that integrates this library, check out [ST-opt-tools](https://github.com/MarineRock10/ST-opt-tools). It's a motion planning toolkit featuring ESDF mapping, A* path planning, and L-BFGS trajectory optimization integrated with SplineTrajectory library.

## Comparison with MINCO
This library is mathematically equivalent to MINCO but implemented with more efficient algorithms.
| Feature         | SplineTrajectory                             | MINCO                      |
| --------------- | -------------------------------------------- | -------------------------- |
| **Algorithm**   | **Thomas Algorithm** (Block Tridiagonal)     | LU Decomposition           |
| **Performance** | **Faster** Generation & Evaluation           | Baseline                   |
| **Core Theory** | Classical Spline Theory (Minimum Norm)       | Minimum Control Effort     |
| **Flexibility** | Fully templated for **arbitrary dimensions** | Fixed to 3D |
| **Evaluation**  | Optimized segmented batch evaluation with coefficient caching        | Standard evaluation        |

## Spline Types & Energy Minimization
The library provides splines that are optimal solutions, minimizing the integral of the squared norm of a derivative, which has a direct physical meaning.

| Spline Type             | MINCO Equivalent     | 
| ----------------------- | -------------------- | 
| **Cubic** (3rd order)   | Minimum Acceleration | 
| **Quintic** (5th order) | Minimum Jerk         | 
| **Septic** (7th order)  | Minimum Snap         |

---
## Usage Example
Here's a concise example of how to create and evaluate a 3D trajectory.
```cpp
#include "SplineTrajectory.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>

int main() {
    using namespace SplineTrajectory;

    std::cout << "=== SplineTrajectory Complete Interface Usage Example ===" << std::endl;

    // 1. Define 3D waypoints and boundary conditions
    SplineVector<SplinePoint3d> waypoints = {
        {0.0, 0.0, 0.0}, {1.0, 2.0, 1.0}, {3.0, 1.0, 2.0}, {4.0, 3.0, 0.5}, {5.0, 0.5, 1.5}
    };
    
    // Define detailed boundary conditions (including velocity, acceleration, jerk)
    BoundaryConditions<3> boundary; //default velocity、acceleration and jerk are zero
    // or BoundaryConditions<3> boundary(SplinePoint3d(0.1, 0.0, 0.0),SplinePoint3d(0.2, 0.0, 0.1)); default acceleration and jerk are zero
    boundary.start_velocity = SplinePoint3d(0.1, 0.0, 0.0); // cubic splines only use velocity 
    boundary.end_velocity = SplinePoint3d(0.2, 0.0, 0.1);
    boundary.start_acceleration = SplinePoint3d(0.0, 0.0, 0.0);// quintic use velocity and acceleration
    boundary.end_acceleration = SplinePoint3d(0.0, 0.0, 0.0);
    boundary.start_jerk = SplinePoint3d(0.0, 0.0, 0.0); // septic use velocity, acceleration and jerk
    boundary.end_jerk = SplinePoint3d(0.0, 0.0, 0.0);

    std::cout << "\n--- Construction Methods Comparison ---" << std::endl;
    
    // 2. Construct splines using time points (multiple spline types comparison)
    std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0, 6.0};
    CubicSpline3D cubic_from_points(time_points, waypoints, boundary);
    QuinticSpline3D quintic_from_points(time_points, waypoints, boundary);
    SepticSpline3D septic_from_points(time_points, waypoints, boundary);

    // Construct splines using time segments
    std::vector<double> time_segments = {1.0, 1.5, 1.5, 2.0}; // Segment durations
    double start_time = 0.0;
    CubicSpline3D cubic_from_segments(time_segments, waypoints, start_time, boundary);
    QuinticSpline3D quintic_from_segments(time_segments, waypoints, start_time, boundary);
    SepticSpline3D septic_from_segments(time_segments, waypoints, start_time, boundary);

    // 3. Update operations example
    std::cout << "\n--- Update Operations ---" << std::endl;
    CubicSpline3D spline_for_update;
    std::cout << "Initial state: " << spline_for_update.isInitialized() << std::endl;
    
    // Update using time points
    spline_for_update.update(time_points, waypoints, boundary);
    std::cout << "State after time points update: " << spline_for_update.isInitialized() << std::endl;
    
    // Update using time segments
    spline_for_update.update(time_segments, waypoints, start_time, boundary);
    std::cout << "State after time segments update: " << spline_for_update.isInitialized() << std::endl;

    // 4. Get basic information
    auto& trajectory = cubic_from_points.getTrajectory();
    std::cout << "\n--- Basic Information ---" << std::endl;
    std::cout << "Spline dimension: " << cubic_from_points.getDimension() << std::endl;
    std::cout << "Start time: " << cubic_from_points.getStartTime() << std::endl;
    std::cout << "End time: " << cubic_from_points.getEndTime() << std::endl;
    std::cout << "Trajectory duration: " << cubic_from_points.getDuration() << std::endl;
    std::cout << "Number of waypoints: " << cubic_from_points.getNumPoints() << std::endl;
    std::cout << "Number of spline segments: " << cubic_from_points.getNumSegments() << std::endl;
    std::cout << "Spline order: " << trajectory.getOrder() << std::endl;

    // 5. Single point evaluation - evaluate general interface
    std::cout << "\n--- Single Point Evaluate General Interface ---" << std::endl;
    double t_eval = 2.5;
    auto pos_eval = trajectory.evaluate(t_eval, 0);      // Position (0th derivative)
    auto vel_eval = trajectory.evaluate(t_eval, 1);      // Velocity (1st derivative)
    auto acc_eval = trajectory.evaluate(t_eval, 2);      // Acceleration (2nd derivative)
    auto jerk_eval = trajectory.evaluate(t_eval, 3);     // Jerk (3rd derivative)
    auto snap_eval = trajectory.evaluate(t_eval, 4);     // Snap (4th derivative)
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "t=" << t_eval << " position: " << pos_eval.transpose() << std::endl;
    std::cout << "t=" << t_eval << " velocity: " << vel_eval.transpose() << std::endl;
    std::cout << "t=" << t_eval << " acceleration: " << acc_eval.transpose() << std::endl;

    // 6. Single point evaluation - get series function renamed interface
    std::cout << "\n--- Single Point Get Series Interface ---" << std::endl;
    auto pos_get = trajectory.getPos(t_eval);
    auto vel_get = trajectory.getVel(t_eval);
    auto acc_get = trajectory.getAcc(t_eval);
    auto jerk_get = trajectory.getJerk(t_eval);
    auto snap_get = trajectory.getSnap(t_eval);
    
    std::cout << "get interface t=" << t_eval << " position: " << pos_get.transpose() << std::endl;
    std::cout << "get interface t=" << t_eval << " velocity: " << vel_get.transpose() << std::endl;

    // 7. Batch evaluation - passing vector<double>
    std::cout << "\n--- Batch Evaluation vector<double> ---" << std::endl;
    std::vector<double> eval_times = {0.5, 1.5, 2.5, 3.5, 5.0};
    
    // evaluate general interface batch evaluation
    auto pos_batch_eval = trajectory.evaluate(eval_times, 0);
    auto vel_batch_eval = trajectory.evaluate(eval_times, 1);
    
    // get series batch evaluation
    auto pos_batch_get = trajectory.getPos(eval_times);
    auto vel_batch_get = trajectory.getVel(eval_times);
    auto acc_batch_get = trajectory.getAcc(eval_times);
    auto jerk_batch_get = trajectory.getJerk(eval_times);
    auto snap_batch_get = trajectory.getSnap(eval_times);
    
    std::cout << "Batch evaluation point count: " << pos_batch_get.size() << std::endl;
    for (size_t i = 0; i < eval_times.size(); ++i) {
        std::cout << "t=" << eval_times[i] << " position: " << pos_batch_get[i].transpose() << std::endl;
    }

    // 8. Time range evaluation (includes end time) - evaluate with range
    std::cout << "\n--- Time Range Evaluate ---" << std::endl;
    double start_t = 0.0, end_t = 6.0, dt = 0.5;
    auto pos_range_eval = trajectory.evaluate(start_t, end_t, dt, 0);
    auto vel_range_eval = trajectory.evaluate(start_t, end_t, dt, 1);
    
    std::cout << "Range evaluation [" << start_t << ", " << end_t << "], dt=" << dt 
              << ", point count: " << pos_range_eval.size() << std::endl;

    // 9. Time range evaluation (includes end time) - get series with range
    std::cout << "\n--- Time Range Get Series ---" << std::endl;
    auto pos_range_get = trajectory.getPos(start_t, end_t, dt);
    auto vel_range_get = trajectory.getVel(start_t, end_t, dt);
    auto acc_range_get = trajectory.getAcc(start_t, end_t, dt);
    auto jerk_range_get = trajectory.getJerk(start_t, end_t, dt);
    auto snap_range_get = trajectory.getSnap(start_t, end_t, dt);
    
    std::cout << "Get series range evaluation point count: " << pos_range_get.size() << std::endl;

    // 10. Generate time sequence (includes trajectory end time)
    std::cout << "\n--- Generate Time Sequence ---" << std::endl;
    // Note: generateTimeSequence will include the trajectory end time point
    auto time_seq_full = trajectory.generateTimeSequence(0.8); // From start to end, dt=0.8
    auto time_seq_range = trajectory.generateTimeSequence(1.0, 5.0, 0.7); // Specified range, dt=0.7
    
    std::cout << "Full time sequence (dt=0.8): " << time_seq_full.size() << " points, last time: " 
              << time_seq_full.back() << " (trajectory end time: " << trajectory.getEndTime() << ")" << std::endl;
    std::cout << "Range time sequence (1.0-5.0, dt=0.7): " << time_seq_range.size() << " points, last time: " 
              << time_seq_range.back() << std::endl;

    // 11. Segmented time sequence (high performance, includes trajectory end time)
    std::cout << "\n--- Segmented Time Sequence Evaluation ---" << std::endl;
    // Note: generateSegmentedTimeSequence also includes the trajectory end time point
    auto segmented_seq = trajectory.generateSegmentedTimeSequence(0.0, 6.0, 0.1);
    std::cout << "Segmented sequence total point count: " << segmented_seq.getTotalSize() 
              << " (includes trajectory end time)" << std::endl;
    std::cout << "Number of segments: " << segmented_seq.segments.size() << std::endl;
    
    // Use segmented sequence for high-performance batch evaluation
    auto pos_segmented_eval = trajectory.evaluateSegmented(segmented_seq, 0);
    auto vel_segmented_eval = trajectory.evaluateSegmented(segmented_seq, 1);
    
    // get series segmented evaluation
    auto pos_segmented_get = trajectory.getPos(segmented_seq);
    auto vel_segmented_get = trajectory.getVel(segmented_seq);
    auto acc_segmented_get = trajectory.getAcc(segmented_seq);
    auto jerk_segmented_get = trajectory.getJerk(segmented_seq);
    auto snap_segmented_get = trajectory.getSnap(segmented_seq);
    
    std::cout << "Segmented get series evaluation point count: " << pos_segmented_get.size() << std::endl;

    // 12. Trajectory analysis
    std::cout << "\n--- Trajectory Analysis ---" << std::endl;
    double traj_length = trajectory.getTrajectoryLength();
    double traj_length_custom = trajectory.getTrajectoryLength(2.0, 4.0, 0.05);
    double cumulative_length = trajectory.getCumulativeLength(3.0);
    
    std::cout << "Total trajectory length: " << traj_length << std::endl;
    std::cout << "Segment [2.0, 4.0] length: " << traj_length_custom << std::endl;
    std::cout << "Cumulative length to t=3.0: " << cumulative_length << std::endl;

    // 13. Derivative trajectories
    std::cout << "\n--- Derivative Trajectories ---" << std::endl;
    auto vel_trajectory = trajectory.derivative(1);  // Velocity trajectory (1st derivative)
    auto acc_trajectory = trajectory.derivative(2);  // Acceleration trajectory (2nd derivative)
    
    std::cout << "Velocity trajectory order: " << vel_trajectory.getOrder() << std::endl; // number of coefficients per segment
    std::cout << "Acceleration trajectory order: " << acc_trajectory.getOrder() << std::endl;

    // 14. Get internal data
    std::cout << "\n--- Get Internal Data ---" << std::endl;
    auto space_points = cubic_from_points.getSpacePoints();
    auto time_segments_data = cubic_from_points.getTimeSegments();
    auto cumulative_times = cubic_from_points.getCumulativeTimes();
    auto boundary_conditions = cubic_from_points.getBoundaryConditions();
    auto trajectory_copy = cubic_from_points.getTrajectoryCopy();
    auto ppoly_ref = cubic_from_points.getPPoly();
    
    std::cout << "Number of spatial points: " << space_points.size() << std::endl;
    std::cout << "Number of time segments: " << time_segments_data.size() << std::endl;
    std::cout << "Number of cumulative times: " << cumulative_times.size() << std::endl;

    // 15. Energy calculation
    std::cout << "\n--- Energy Calculation ---" << std::endl;
    double cubic_energy = cubic_from_points.getEnergy();
    double quintic_energy = quintic_from_points.getEnergy();
    double septic_energy = septic_from_points.getEnergy();
    
    std::cout << "Cubic spline energy: " << cubic_energy << std::endl;
    std::cout << "Quintic spline energy: " << quintic_energy << std::endl;
    std::cout << "Septic spline energy: " << septic_energy << std::endl;

    // 16. PPolyND static methods
    std::cout << "\n--- PPolyND Static Methods ---" << std::endl;
    std::vector<double> test_breakpoints = {0.0, 1.0, 2.0, 3.0};
    auto zero_poly = PPoly3D::zero(test_breakpoints, 3);
    SplinePoint3d constant_val(1.0, 2.0, 3.0);
    auto constant_poly = PPoly3D::constant(test_breakpoints, constant_val);
    
    std::cout << "Zero polynomial at t=1.5: " << zero_poly.getPos(1.5).transpose() << std::endl;
    std::cout << "Constant polynomial at t=1.5: " << constant_poly.getPos(1.5).transpose() << std::endl;

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
    //g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. SplineTrajectoryExample.cpp -o SplineTrajectoryExample
}
```
```bash
g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. SplineTrajectoryExample.cpp -o SplineTrajectoryExample
```

---
## Future Plans

- [ ] Add a gradient propagation mechanism equivalent to MINCO
    
- [x] Implement support for clamped 7th-order splines (Septic Spline, Minimum Snap)
    
- [ ] Implement support for N-dimensional Non-Uniform B-Splines
    
- [ ] Implement support for the exact conversion from clamped splines to Non-Uniform B-Splines
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Built with [Eigen](http://eigen.tuxfamily.org/) for linear algebra operations
- Inspired by [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) trajectory optimization
- Grounded in **classical spline interpolation theory** and the **minimum norm theorem**
