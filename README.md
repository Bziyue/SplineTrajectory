# SplineTrajectory

[中文](README_zh.md) · [Documentation](docs/README.md) · [Examples](examples/README.md)

A header-only C++17 / Eigen library for N-dimensional minimum-control-effort trajectories.
Cubic, quintic and septic splines minimize squared acceleration, jerk and snap, with
linear-time construction and analytic gradients. Compose optimization objectives and solve
them with your preferred numerical optimizer.

| Header | Features |
| --- | --- |
| `SplineTrajectory.hpp` | Splines, energy gradients, piecewise polynomials and sequential sampling |
| `SplineOptimizer.hpp` | Decision variables, objective composition and gradient propagation |
| `SplineConvexHull.hpp` | Bezier / MINVO control points, subdivision and derivatives |

## Build and test

Requires a C++17 compiler, CMake ≥ 3.12, Eigen ≥ 3.3 and Python 3 for the test tools.

```sh
git clone https://github.com/Bziyue/SplineTrajectory.git
cd SplineTrajectory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 3
(cd build && ctest --output-on-failure)
./build/examples/corridor_optimizer
./build/examples/esdf_optimizer
```

Examples and tests build by default. Enable performance comparisons with
`-DSPLINE_BUILD_BENCHMARKS=ON`. To use only the library, disable `BUILD_TESTING`,
`SPLINE_BUILD_EXAMPLES` and `SPLINE_BUILD_BENCHMARKS`, or include the headers directly.

## Create a spline

```cpp
#include "SplineTrajectory.hpp"

using Spline = SplineTrajectory::QuinticSplineND<3>;
Spline::CoefficientMatrix points(3, 3);
points << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
std::vector<double> durations{1.2, 1.5};
Spline spline(durations, points, 0.0);
if (!spline.isValid()) throw std::runtime_error("Invalid spline");
auto state = spline.polynomial().evaluateDerivatives<2>(0.5); // position, velocity, acceleration
```

## Create an optimizer

Using the same points and durations:

```cpp
#include "SplineOptimizer.hpp"

SplineTrajectory::SplineOptimizer<Spline> optimizer;
SplineTrajectory::SplineProblem<Spline> problem;
problem.waypoints = points;
problem.durations = durations;
if (!optimizer.prepare(problem)) throw std::runtime_error("Invalid problem");
auto x = optimizer.initialGuess();
Eigen::VectorXd gradient(x.size());
SplineTrajectory::NoObjective objective; // minimum-jerk energy only
const double cost = optimizer.costAndGradient(x, gradient, objective);
```

Add named `duration`, `integral` or `coefficient` members to an objective to combine time,
state and control-point costs. An external solver calls `costAndGradient`; after solving,
call `updateAccepted(x)` before reading the trajectory. See the [GCOPTER L-BFGS examples](examples/README.md).

## Examples and guides

| Example | Objective |
| --- | --- |
| [Basic optimizer](examples/basic_optimizer.cpp) | Minimum jerk and total duration |
| [Corridor](examples/corridor_optimizer.cpp) | Segment corridors, speed and acceleration |
| [ESDF](examples/esdf_optimizer.cpp) | Distance-field obstacle avoidance and derivative limits |
| [Dynamics](examples/dynamics_optimizer.cpp) | Speed and acceleration penalties |
| [Convex hull](examples/convex_hull_optimizer.cpp) | Bezier corridor and derivative control-point bounds |

[Sampling and gradients](docs/trajectory.md) · [Optimizer reference](docs/optimizer.md) ·
[Convex hulls](docs/convex_hull.md) · [Benchmarks](docs/benchmarks.md) · [Numerical notes](docs/numerical_notes.md)

The public headers are [MIT licensed](LICENSE). Bundled dependencies retain their
[third-party licenses](THIRD_PARTY_NOTICES.md).
