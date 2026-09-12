# SplineTrajectory

[中文](README_zh.md) · [Optimizer protocols](include/SplineOptimizerProtocols.md) · [API migration](docs/api_migration.md)

A C++17/Eigen header-only library for N-dimensional minimum-control-effort splines,
mathematically equivalent to MINCO. Cubic, quintic and septic splines minimize squared
acceleration, jerk and snap. Their specialized O(N) solvers and adjoints share owning
polynomial storage; construction does not copy a second coefficient matrix.

| Public header | Responsibility |
| --- | --- |
| `SplineTrajectory.hpp` | Interpolation, energy, adjoints, owning piecewise polynomials and sampling |
| `SplineOptimizer.hpp` | Problems, parameterization, objective evaluation and an exclusive solve workspace |
| `SplineConvexHull.hpp` | Bezier/MINVO conversion, subdivision and coefficient/time adjoints |

The optimizer and hull headers each depend only on the trajectory header. Coordinates and
application policies belong to the caller; ROS and external solvers are not library dependencies.

## Basic optimization

An optimizer owns its maps, problem snapshot and buffers. An objective is an ordinary struct
with named callable members; declare only the stages needed. No builder, context or inherited
objective interface is required.

```cpp
#include "SplineOptimizer.hpp"

struct VelocityCost {
    static constexpr int kDerivativeOrder = 1;
    double operator()(const SplineTrajectory::IntegralPointInfo &,
                      const SplineTrajectory::SampleState<3> &state,
                      SplineTrajectory::SampleGradient<3> &gradient) const {
        gradient.v += state.v;
        return 0.5 * state.v.squaredNorm();
    }
};
struct ProblemObjective { VelocityCost integral; };

int main() {
    using Spline = SplineTrajectory::QuinticSplineND<3>;
    SplineTrajectory::SplineOptimizer<Spline> optimizer;
    SplineTrajectory::SplineProblem<Spline> problem;
    problem.durations = {1.2, 1.5};
    problem.waypoints.resize(3, 3);
    problem.waypoints << 0,0,0, 1,0.2,0, 2,0,0;
    SplineTrajectory::OptimizerOptions options;
    options.integration_steps = 16;
    options.energy_weight = 0.1;
    if (!optimizer.prepare(problem, options)) return 1;
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient(x.size());
    ProblemObjective objective;
    if (!optimizer.evaluate(x, gradient, objective)) return 2;
    // An external solver repeatedly calls optimizer.costAndGradient(x, gradient, objective).
    // Use its accepted x, which may differ from the last trial passed to that callback:
    if (!optimizer.updateAccepted(x)) return 3;
    auto snapshot = optimizer.copyPolynomial();
    return snapshot.isValid() ? 0 : 4;
}
```

Supported members are `duration`, `integral`, `sample`, `coefficient`, `parameter` and `decision`.
Use `NoObjective` for an explicitly empty objective, optionally with minimum-derivative energy.
A missing integral member skips quadrature unless recording or a sample cost needs it.
The [protocol reference](include/SplineOptimizerProtocols.md) specifies signatures, additive
gradients, masks, custom maps, explicit borrowing and concurrency.

After preparation, a normal evaluation with correctly sized output uses no library heap
allocation, including its first call. User callbacks must satisfy this separately. A `sample`
member needs `prepare(problem, options, objective)` to allocate recording storage. Topology or
recording capability changes require preparation. The optimizer is movable and noncopyable;
independent concurrent solves use independent optimizers and mutable cost workspaces.

Complete examples:

- [Basic optimizer](examples/basic_optimizer.cpp): a named objective and a small external descent loop.
- [Custom parameterization](examples/parameterized_optimizer.cpp): bounded coordinates, positive
  times, shared variables, a moving time origin, integral and Bezier costs, and gradient checking.
- [Hull optimization](examples/convex_hull_optimization_demo.cpp): a C-style L-BFGS callback and
  reusable hull workspaces. Hull penalties are soft; final feasibility remains an application decision.

## Polynomials and sampling

`PPolyND<DIM, MaxCoefficients>` owns rows arranged by piece, then ascending powers of local
physical time in seconds. The compile-time capacity can exceed the actual coefficient count.
Spline traits are `kDimension`, `kDegree`, `kCoefficientCount` and `kMinDerivativeOrder`.
`MinDerivativeSplineND<DIM, S>` selects the specialized solver for S = 2, 3 or 4.

```cpp
const auto &poly = spline.polynomial();
auto sampler = poly.makeSampler<3>();
for (double t : increasing_times) {
    const auto derivatives = sampler.evaluate(t); // position, velocity, acceleration, jerk
    consume(derivatives);
}
```

Each sampler owns its cursor and current-piece derivative coefficients. Entering another piece
prepares those coefficients once; same-piece queries reuse them. Fixed-capacity samplers need
no heap allocation. Dynamic-capacity samplers allocate current-piece storage at construction.
Source const queries have no shared writes and need no warm-up, including concurrent first use.
The polynomial must outlive its samplers; any update, assignment or move invalidates all borrowed
views. `copyPolynomial()` provides an independent snapshot.

Use `evaluate(t, order)` for one derivative, `evaluateDerivatives<N>(t)` for several derivatives
with one lookup, and `evaluateInto(times, output, order)` for caller-owned batch output.
`Segment::evaluateLocal()` uses time relative to the piece start. Interior knots select the right
piece, the final knot selects the last piece, and outside queries extrapolate. Backward sampler
jumps relocate; short polynomials use linear search and larger ones use binary search.

Invalid polynomial/time queries return NaNs; derivatives above degree are zero; negative
orders throw `std::invalid_argument`. Invalid construction leaves an empty invalid polynomial.
`generateTimeSequence()` checks finite inputs, positive step, interval direction and representable
sample count; a zero-length interval returns one endpoint. `at()` checks indices.

## Hull workspaces

```cpp
SplineTrajectory::ConvexHullWorkspace<3> hull;
SplineTrajectory::HullOptions options{SplineTrajectory::ConvexHullBasis::Bezier, 0, 2};
hull.configure(poly, options); // allocations and shared-kernel cache synchronization
hull.update(poly);            // fixed-topology numeric update: no allocation or lock
// hull.backwardAdd(control_partials, coefficient_partials, duration_partials);
```

`isConfigured()` describes prepared topology; `isValid()` describes the latest numeric update;
`matches(poly, options)` checks reuse. `toBezier()` and `toMINVO()` return ready-to-use workspaces.
The immutable `ConvexHullKernel` is shared across spatial dimensions. Updates copy the source
values needed by the adjoint, so a hull does not borrow a polynomial. `backwardAdd()` mutates
scratch storage and requires exclusive workspace access; destination gradients are accumulated.
See [the mathematical derivation](docs/convex_hull_basis_zh.md) and [MINVO provenance](THIRD_PARTY_NOTICES.md).

## Build and validation

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j3
(cd build && ctest --output-on-failure)
./build/basic_optimizer
./build/parameterized_optimizer
```

`BUILD_TESTING`, `SPLINE_BUILD_EXAMPLES` and `SPLINE_BUILD_BENCHMARKS` select owned targets.
They use `-Wall -Wextra -Wpedantic -Werror` on GCC/Clang; these flags are not exported to consumers.
Only vendored dependency directories are treated as system includes. `SPLINE_NATIVE_ARCH` is
optional and off by default. Consumers relying on finite-value failure handling must preserve
IEEE NaN/Inf semantics (`-ffast-math` and `-ffinite-math-only` are incompatible with that contract).

CTest includes MINCO/finite-difference gradients, masks and boundary variables, complete objective
chains, shared time variables, first-use allocation checks, sampling, failures, copy/move behavior
and targeted compile-failure diagnostics. See [second-refactor validation](docs/refactor_v2_validation.md)
for measured scope and limitations. [First-refactor results](docs/refactor_validation.md) and
[historical hull measurements](docs/convex_hull_performance_zh.md) retain their original scope.

[MIT license](LICENSE). Three public headers are maintained here and distributed unchanged to downstream projects.
