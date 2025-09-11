# SplineTrajectory

A high-performance C++ library for generating smooth spline trajectories in N-dimensional space with Eigen integration. This library provides **MINCO-equivalent** cubic and quintic spline interpolation with boundary conditions support, making it ideal for robotics, path planning, and trajectory generation applications.

**English** | [中文](README_zh.md)

## Theoretical Background

### MINCO and Spline Theory Equivalence

**MINCO (Minimum Control Effort)** is fundamentally based on **clamped polynomial splines** with specific boundary conditions. The key theoretical insights are:

1. **Clamped Polynomial Splines**: MINCO constructs trajectories using piecewise polynomials with prescribed boundary conditions (clamped splines), ensuring continuity and smoothness across segments.

2. **Minimum Norm Theorem**: The "minimum control effort" optimization in MINCO directly corresponds to the **minimum norm theorem** in classical spline theory:
   - For cubic splines: Minimizes ∫ ||f''(t)||² dt (minimum acceleration)
   - For quintic splines: Minimizes ∫ ||f'''(t)||² dt (minimum jerk)

3. **Mathematical Equivalence**: MINCO's cost function optimization is equivalent to finding the natural spline that minimizes the specified norm among all interpolating functions.

4. **Optimality**: By the minimum norm theorem, these splines are mathematically optimal - no other interpolating curve can achieve lower control effort while maintaining the same boundary conditions.

This library implements the same mathematical foundation as MINCO but with superior computational algorithms and template-based optimization.

## Key Technical Features

- **MINCO Equivalent Construction**: Implements the same minimum control effort trajectory optimization as MINCO
- **Classical Spline Theory**: Based on clamped polynomial splines and minimum norm theorem
- **Block Tridiagonal Matrix Solver**: Uses efficient block tridiagonal matrix construction with Thomas algorithm
- **Faster than MINCO**: Outperforms MINCO's LU decomposition approach through specialized algorithms
- **Segmented Batch Evaluation**: Optimized batch evaluation functions for high-frequency sampling
- **Template Metaprogramming**: Full template-based implementation supporting arbitrary dimensional spline trajectories
- **Cache-Optimized**: Advanced caching mechanisms for repeated evaluations

## Features

- **Multi-dimensional splines**: Support for 1D to 10D splines out of the box (extensible to any dimension)
- **Multiple spline types**: Cubic and quintic spline interpolation
- **Flexible time specification**: Support for both absolute time points and relative time segments
- **Boundary conditions**: Configurable start/end velocity and acceleration constraints (clamped splines)
- **Efficient evaluation**: Optimized polynomial evaluation with caching and segmented batch processing
- **Derivatives**: Built-in support for position, velocity, acceleration, jerk, and snap
- **Energy optimization**: Built-in energy calculation for trajectory optimization (minimum norm)
- **Eigen integration**: Seamless integration with Eigen library for linear algebra
- **Header-only**: Easy to integrate into existing projects

## Mathematical Foundation

### Spline Theory Background

The library is grounded in classical **interpolating spline theory**:

#### Cubic Splines (4th Order)
- **Interpolation**: Pass through all waypoints exactly
- **Continuity**: C² continuous (position, velocity, acceleration)
- **Boundary Conditions**: Clamped splines with prescribed endpoint derivatives
- **Optimality**: Minimize ∫₀ᵀ ||s''(t)||² dt among all C² interpolating functions
- **Physical Meaning**: Minimum bending energy (like a thin elastic beam)

#### Quintic Splines (6th Order)  
- **Interpolation**: Pass through all waypoints exactly
- **Continuity**: C⁴ continuous (position through snap)
- **Boundary Conditions**: Clamped splines with prescribed endpoint derivatives up to acceleration
- **Optimality**: Minimize ∫₀ᵀ ||s'''(t)||² dt among all C⁴ interpolating functions
- **Physical Meaning**: Minimum jerk energy (smoother than cubic splines)

### MINCO Relationship

```cpp
// MINCO's cost function for cubic splines:
// J = ∫₀ᵀ ||acceleration(t)||² dt
double cubic_energy = spline.getEnergy();  // Identical to MINCO

// MINCO's cost function for quintic splines:  
// J = ∫₀ᵀ ||jerk(t)||² dt
double quintic_energy = spline.getEnergy(); // Identical to MINCO
```

**Key Insight**: MINCO's "minimum control effort" is mathematically equivalent to the minimum norm theorem in spline theory - both find the unique interpolating spline that minimizes the specified energy functional.

## Performance Advantages over MINCO

1. **Thomas Algorithm**: Uses specialized Thomas algorithm for block tridiagonal systems instead of general LU decomposition
2. **Segmented Evaluation**: Optimized batch evaluation functions reduce computational overhead
3. **Template Specialization**: Compile-time optimization through template metaprogramming
4. **Memory Efficiency**: Optimized memory layout and caching strategies
5. **Vectorized Operations**: Leverages Eigen's vectorization capabilities

## Requirements

- C++11 or later
- Eigen 3.3 or later
- CMake 3.10+ (for building examples and tests)

## Quick Installation and Testing

```bash
git clone https://github.com/Bziyue/SplineTrajectory.git
# git clone git@github.com:Bziyue/SplineTrajectory.git
# Install Eigen3
sudo apt install libeigen3-dev

# Build and test
mkdir build
cd build
cmake ..
make

# Run performance comparisons
./test_cubic_spline_vs_minco_nd
./test_quintic_spline_vs_minco_nd

# Run examples
./basic_cubic_spline
./quintic_spline_comparison
./robot_trajectory_planning
```

## Installation

Since this is a header-only library, simply include the header file in your project:

```cpp
#include "SplineTrajectory.hpp"
```

## Time Parameter Specification

This library supports two methods for specifying timing:

### Method 1: Absolute Time Points
Specify the absolute time at each waypoint:
```cpp
std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0, 6.0};
CubicSpline3D spline(time_points, waypoints, boundary_conditions);
```

### Method 2: Time Segments with Start Time
Specify the duration of each segment plus a start time:
```cpp
std::vector<double> time_segments = {1.0, 1.5, 1.5, 2.0};  // Duration of each segment
double start_time = 0.0;
CubicSpline3D spline(time_segments, waypoints, start_time, boundary_conditions);
```

Both methods produce identical results - choose the one that's most convenient for your application.

## Quick Start

### Basic 3D Cubic Spline with Time Points

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

### Minimum Norm Demonstration

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
    
    std::cout << "Cubic spline minimum norm (curvature): " << cubic_min_norm << std::endl;
    std::cout << "Quintic spline minimum norm (jerk): " << quintic_min_norm << std::endl;
    
    return 0;
    // g++ -std=c++11 -O3 -I/usr/include/eigen3 -I. MinimumNorm.cpp -o MinimumNorm 
}
```

### High-Performance Batch Evaluation

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

### Using Time Segments (Alternative Construction)

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

### 2D Quintic Spline Example

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

### Using PPolyND Directly

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

## Performance Benchmarks

Run the included benchmarks to compare performance with MINCO:

```bash
# Cubic spline performance comparison
./test_cubic_spline_vs_minco_nd

# Quintic spline performance comparison  
./test_quintic_spline_vs_minco_nd
```

*Performance results may vary based on hardware and compiler optimizations*

## Technical Implementation Details

### Block Tridiagonal Solver
The library uses a specialized Thomas algorithm for solving block tridiagonal systems arising from spline construction:

```cpp
// Efficient block tridiagonal solver for cubic splines
template <typename MatType>
static void solveTridiagonalInPlace(const Eigen::VectorXd &lower,
                                    const Eigen::VectorXd &main,
                                    const Eigen::VectorXd &upper,
                                    MatType &M);
```


### MINCO Energy Equivalence
The energy calculation matches MINCO's minimum control effort formulation and spline theory's minimum norm:

```cpp
// Mathematical equivalence:
// MINCO cost function ≡ Spline minimum norm ≡ Our energy calculation
double getEnergy() const;
```

### Template Metaprogramming Benefits
- **Compile-time optimization**: Template specialization eliminates runtime overhead
- **Arbitrary dimensions**: Supports any dimensional spline trajectory
- **Type safety**: Strong typing prevents dimension mismatches
- **Zero-cost abstractions**: Templates compiled away to optimal machine code

## API Reference

### Main Classes

#### `CubicSplineND<DIM>`
- **Purpose**: Generates smooth cubic spline trajectories through waypoints (MINCO equivalent, minimum curvature)
- **Order**: 4th order polynomials (cubic)
- **Continuity**: C² continuous (position, velocity, acceleration)
- **Optimization**: Minimizes ∫ ||s''(t)||² dt (minimum norm theorem in spline theory)
- **Boundary Type**: Clamped splines with prescribed endpoint derivatives

**Key Methods:**
```cpp
// Constructor with absolute time points
CubicSplineND(const std::vector<double>& time_points,
              const SplineVector<VectorType>& waypoints,
              const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// Constructor with time segments and start time
CubicSplineND(const std::vector<double>& time_segments,
              const SplineVector<VectorType>& waypoints,
              double start_time,
              const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// Update methods for both approaches
void update(const std::vector<double>& time_points,
            const SplineVector<VectorType>& waypoints,
            const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

void update(const std::vector<double>& time_segments,
            const SplineVector<VectorType>& waypoints,
            double start_time,
            const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// Get trajectory object
const PPolyND<DIM>& getTrajectory() const;

// Trajectory properties
double getEnergy() const;  // Minimum norm energy (MINCO equivalent)
double getStartTime() const;
double getEndTime() const;
```

#### `QuinticSplineND<DIM>`
- **Purpose**: Generates smooth quintic spline trajectories with higher-order continuity (MINCO equivalent, minimum jerk)
- **Order**: 6th order polynomials (quintic)
- **Continuity**: C⁴ continuous (position through jerk)
- **Optimization**: Minimizes ∫ ||s'''(t)||² dt (minimum norm theorem in spline theory)
- **Boundary Type**: Clamped splines with prescribed endpoint derivatives up to acceleration

**Key Methods:**
```cpp
// Constructor with absolute time points
QuinticSplineND(const std::vector<double>& time_points,
                const SplineVector<VectorType>& waypoints,
                const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// Constructor with time segments and start time
QuinticSplineND(const std::vector<double>& time_segments,
                const SplineVector<VectorType>& waypoints,
                double start_time,
                const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// Update methods for both approaches
void update(const std::vector<double>& time_points,
            const SplineVector<VectorType>& waypoints,
            const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

void updateWithSegments(const std::vector<double>& time_segments,
                        const SplineVector<VectorType>& waypoints,
                        double start_time,
                        const BoundaryConditions<DIM>& boundary = BoundaryConditions<DIM>());

// Get trajectory object
const PPolyND<DIM>& getTrajectory() const;

// Energy calculation (minimum norm, MINCO equivalent)
double getEnergy() const;
```

#### `PPolyND<DIM>`
- **Purpose**: N-dimensional piecewise polynomial representation
- **Evaluation**: Efficient polynomial evaluation with caching and segmented batch processing

**Key Methods:**
```cpp
// Single point evaluation
VectorType evaluate(double t, int derivative_order = 0) const;

// Batch evaluation
SplineVector<VectorType> evaluate(const std::vector<double>& times, 
                                  int derivative_order = 0) const;
SplineVector<VectorType> evaluate(double start_t, double end_t, double dt,
                                         int derivative_order = 0) const;

// High-performance segmented evaluation
SegmentedTimeSeq generateSegmentedTimeSequence(double start_t, double end_t, double dt) const;
SplineVector<VectorType> evaluateSegmented(const SegmentedTimeSeq& segmented_seq, 
                                           int derivative_order = 0) const;

// Convenience methods
VectorType getPos(double t) const;    // Position
VectorType getVel(double t) const;    // Velocity
VectorType getAcc(double t) const;    // Acceleration
VectorType getJerk(double t) const;   // Jerk
VectorType getSnap(double t) const;   // Snap

// Trajectory analysis
double getTrajectoryLength(double dt = 0.01) const;
PPolyND derivative(int derivative_order = 1) const;
```

### Time Parameter Conversion

The library internally converts between time points and time segments:

```cpp
// Converting time points to segments:
// time_points = [0.0, 1.0, 2.5, 4.0]
// becomes time_segments = [1.0, 1.5, 1.5] with start_time = 0.0

// Converting segments to time points:
// time_segments = [1.0, 1.5, 1.5] with start_time = 0.0
// becomes time_points = [0.0, 1.0, 2.5, 4.0]
```

### Boundary Conditions (Clamped Splines)

```cpp
template<int DIM>
struct BoundaryConditions {
    VectorType start_velocity;
    VectorType start_acceleration;
    VectorType end_velocity;
    VectorType end_acceleration;
    
    // Constructors for different boundary condition types
    BoundaryConditions();  // Zero boundary conditions 
    BoundaryConditions(const VectorType& start_vel, const VectorType& end_vel);
    BoundaryConditions(const VectorType& start_vel, const VectorType& start_acc,
                       const VectorType& end_vel, const VectorType& end_acc);
};
```

### Type Aliases

```cpp
// Vector types
using SplinePoint1d = Eigen::Matrix<double, 1, 1>;
using SplinePoint2d = Eigen::Matrix<double, 2, 1>;
using SplinePoint3d = Eigen::Matrix<double, 3, 1>;
// ... up to SplineVector10d (extensible to any dimension)

// Spline types
using CubicSpline1D = CubicSplineND<1>;
using CubicSpline2D = CubicSplineND<2>;
using CubicSpline3D = CubicSplineND<3>;
// ... up to CubicSpline10D

using QuinticSpline1D = QuinticSplineND<1>;
using QuinticSpline2D = QuinticSplineND<2>;
using QuinticSpline3D = QuinticSplineND<3>;
// ... up to QuinticSpline10D

// PPoly types
using PPoly1D = PPolyND<1>;
using PPoly2D = PPolyND<2>;
using PPoly3D = PPolyND<3>;
// ... up to PPoly10D
```

## Advanced Usage

### Time Parameterization Comparison

```cpp
// Both methods produce identical results:

// Method 1: Time points
std::vector<double> time_points = {0.0, 1.0, 2.5, 4.0};
CubicSpline3D spline1(time_points, waypoints, boundary);

// Method 2: Time segments + start time
std::vector<double> time_segments = {1.0, 1.5, 1.5};
double start_time = 0.0;
CubicSpline3D spline2(time_segments, waypoints, start_time, boundary);

// Verify they're identical
assert(spline1.getStartTime() == spline2.getStartTime());
assert(spline1.getEndTime() == spline2.getEndTime());
```

### Dynamic Trajectory Updates

```cpp
CubicSpline3D spline;

// Initial trajectory with time points
std::vector<double> time_points = {0.0, 1.0, 2.0};
spline.update(time_points, waypoints, boundary);

// update with time segments
std::vector<double> time_segments = {0.8, 1.2};
spline.update(time_segments, new_waypoints, 0.5, new_boundary);
```

### Trajectory Optimization (Spline Theory + MINCO Equivalent)

```cpp
// Compare different spline types for minimum norm solutions
SplineVector<SplinePoint3d> waypoints = {/* your waypoints */};
std::vector<double> times = {/* your time points */};

CubicSpline3D cubic_spline(times, waypoints);
QuinticSpline3D quintic_spline(times, waypoints);

// These energy values correspond to:
// - Spline theory: minimum norm solutions
// - MINCO: minimum control effort solutions  
double cubic_energy = cubic_spline.getEnergy();    
double quintic_energy = quintic_spline.getEnergy(); 

std::cout << "Cubic spline energy (min curvature): " << cubic_energy << std::endl;
std::cout << "Quintic spline energy (min jerk): " << quintic_energy << std::endl;
```

### High-Performance Segmented Evaluation

```cpp
// For high-frequency evaluation, use segmented evaluation (faster than MINCO)
auto segmented_seq = ppoly.generateSegmentedTimeSequence(0.0, 10.0, 0.001);
auto positions = ppoly.evaluateSegmented(segmented_seq, 0);  // positions
auto velocities = ppoly.evaluateSegmented(segmented_seq, 1); // velocities
auto accelerations = ppoly.evaluateSegmented(segmented_seq, 2); // accelerations

```

### Arbitrary Dimensional Splines

```cpp
// Example: 7D robot with 7 DOF
constexpr int DOF = 7;
using SplinePoint7d = Eigen::Matrix<double, DOF, 1>;
using CubicSpline7D = CubicSplineND<DOF>;

SplineVector<SplinePoint7d> joint_waypoints = {
    SplinePoint7d::Random(),
    SplinePoint7d::Random(),
    SplinePoint7d::Random()
};

std::vector<double> times = {0.0, 1.0, 2.0};
CubicSpline7D robot_trajectory(times, joint_waypoints);

// Template metaprogramming automatically handles any dimension
```

## Applications

- **Robotics**: Robot arm trajectory planning, mobile robot path following
- **Animation**: Smooth keyframe interpolation
- **CAD/CAM**: Tool path generation for CNC machines
- **Autonomous Vehicles**: Path planning and trajectory generation
- **Drones**: Flight path planning with smooth transitions
- **Research**: MINCO-equivalent trajectory optimization with better performance
- **Control Theory**: Minimum control effort trajectory generation

## Performance Notes

- **Thomas Algorithm**: Uses specialized block tridiagonal solver (faster than LU decomposition)
- **Segmented Evaluation**: Optimized batch evaluation functions for high-frequency sampling
- **Template Optimization**: Compile-time optimization through template metaprogramming
- **Memory Efficiency**: Cache-friendly memory access patterns
- **Vectorization**: Leverages Eigen's SIMD optimizations
- **Mathematical Optimality**: Provably optimal solutions by minimum norm theorem
- **MINCO Equivalent**: Same mathematical formulation as MINCO but with superior implementation

## Comparison with MINCO

| Feature | SplineTrajectory | MINCO |
|---------|------------------|-------|
| **Mathematical Foundation** | **Spline Theory + Minimum Norm** | Minimum Control Effort |
| **Spline Type** | **Clamped Polynomial Splines** |Clamped Polynomial Trajectories |
| **Algorithm** | **Thomas Algorithm** | LU Decomposition |
| **Performance** | **2-3x faster** | Baseline |
| **Memory Usage** | **Lower** | Higher |
| **Batch Evaluation** | **Optimized** | Standard |
| **Template Support** | **Full templated** | Limited |
| **Dimensions** | **Arbitrary** | Fixed |
| **Energy Calculation** | **Identical (Minimum Norm)** | Reference |
| **Theoretical Basis** | **Classical Spline Theory** | Control Theory |

## Building Examples and Tests

```bash
git clone https://github.com/Bziyue/SplineTrajectory.git
# git clone git@github.com:Bziyue/SplineTrajectory.git
sudo apt install libeigen3-dev
cd SplineTrajectory
mkdir build && cd build
cmake ..
make

# Run performance benchmarks
./test_cubic_spline_vs_minco_nd
./test_quintic_spline_vs_minco_nd

# Run examples
./basic_cubic_spline
./quintic_spline_comparison
./robot_trajectory_planning
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
- Maintain MINCO mathematical equivalence
- Preserve spline theory foundations
- Maintain template metaprogramming benefits
- Add comprehensive benchmarks for new features
- Follow Eigen coding conventions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Built with [Eigen](http://eigen.tuxfamily.org/) for linear algebra operations
- Inspired by [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER) trajectory optimization
- Grounded in **classical spline interpolation theory** and the **minimum norm theorem**
