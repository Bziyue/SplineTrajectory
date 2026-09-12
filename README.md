# SplineTrajectory

[中文](README_zh.md) · [Optimizer protocols](include/SplineOptimizerProtocols.md) · [Refactor validation](docs/refactor_validation.md)

A C++17/Eigen header-only library for N-dimensional minimum-control-effort trajectories,
mathematically equivalent to MINCO. Cubic, quintic and septic splines minimize squared
acceleration, jerk and snap respectively. Construction and adjoint propagation use an O(N)
block-tridiagonal solve; energy also has a direct analytic parameter-gradient path.

| Public header | Responsibility |
| --- | --- |
| `SplineTrajectory.hpp` | Splines, validated piecewise polynomials, derivatives, sampling, exact extraction/append, energy and adjoints |
| `SplineOptimizer.hpp` | Prepared problems, variable maps/masks, statically bound costs, quadrature and gradient assembly |
| `SplineConvexHull.hpp` | Bezier/MINVO control hulls, physical derivatives, subdivision and coefficient/time pullback |

The optimizer and hull headers independently depend on the trajectory header. ROS, geometry
policies and L-BFGS remain application dependencies. Hull penalties are soft objectives;
applications still need their own feasibility acceptance policy.

## Sampling without prewarming

```cpp
const auto &polynomial = spline.getTrajectory();
auto sampler = polynomial.makeSampler<3>();
for (double t : increasing_times) {
    const auto state = sampler.evaluate(t); // position, velocity, acceleration, jerk
    consume(state[0], state[1], state[2], state[3]);
}
```

A sampler borrows an immutable polynomial. The source must outlive it; source update,
assignment or move invalidates it, so construct another afterwards. Each traversal owns
its own sampler. Source const queries never write lazy caches, including the first query.
Fixed-order sampler construction and evaluation need no heap allocation. Dynamic-order
samplers allocate only their current-piece storage at construction. Derivative coefficients
are prepared only on entering a different piece, then reused for every query in that piece.

Forward queries advance through pieces; backward queries relocate with binary search.
Internal knots belong to the right piece; the final knot belongs to the last piece. Outside
the domain, both ordinary evaluation and samplers extrapolate. Invalid polynomials or
nonfinite query times return NaNs. Derivatives above the polynomial degree are zero.
For random access, use `evaluate(t, derivative)` or `evaluateDerivatives<N>(t)`; the latter
locates once and returns an array. Legacy `last_idx_hint` overloads remain source compatible,
but new traversal code should own a sampler instead of exposing an index.

`extractInterval()` and `appendRebased()` operate on coefficients exactly (up to floating
point roundoff). They do not decide application frame, identity or seam acceptance policy.

## Optimizing a trajectory

Define a problem and configuration, prepare a context, repeatedly evaluate candidates, then
synchronize the accepted solver vector. Configuration is copied at preparation; maps and
cost objects are borrowed. Keep the optimizer, maps and costs alive throughout their use.
Use one context per active evaluation; parallel execution also requires thread-safe costs.

```cpp
#include "SplineOptimizer.hpp"

struct VelocityCost {
    static constexpr int kDerivativeOrder = 1;
    double operator()(const SplineTrajectory::IntegralPointInfo &,
                      const SplineTrajectory::SampleState<3> &state,
                      SplineTrajectory::SampleGradient<3> &gradient) const {
        gradient.v = state.v;
        return 0.5 * state.v.squaredNorm();
    }
};

int main() {
    using Opt = SplineTrajectory::SplineOptimizer<3>;
    Opt optimizer;
    Opt::ProblemDefinition problem;
    problem.time_segments = {1.2, 1.5};
    problem.waypoints.resize(3, 3);
    problem.waypoints << 0,0,0, 1,0.2,0, 2,0,0;
    Opt::OptimizerConfig config;
    config.integral_num_steps = 16;
    config.rho_energy = 0.1;
    Opt::OptimizationContext context;
    if (!optimizer.prepareContext(problem, context, config)) return 1;
    SplineTrajectory::VoidTimeCost time_cost;
    VelocityCost integral_cost;
    const auto spec = Opt::makeEvaluateSpec(time_cost, integral_cost);
    auto x = optimizer.generateInitialGuess(context);
    Eigen::VectorXd gradient;
    const auto result = optimizer.evaluate(context, x, gradient, spec);
    if (!result) return 2;
    // Pass evaluatePrepared(context, x, gradient, spec) to an external solver.
    // After that solver returns its accepted x:
    if (!optimizer.synchronizeWorkingState(context, x)) return 3;
    const auto &accepted = optimizer.getWorkingSpline(context);
    return accepted.isInitialized() ? 0 : 4;
}
```

Local partials start at zero and exclude integration weights. `gradient.time` is the explicit
global-time partial at fixed state. `kDerivativeOrder` defaults to 4; declaring a lower order
removes unused state work, while retaining one more derivative for physical-time drift.
Quadrature bases are prepared on normalized time once and reused across pieces/evaluations.
A piece scales its coefficients once; accumulated gradients are converted back before the adjoint.
GCC/Clang flatten the internal quadrature call chain to keep statically bound costs
visible for inlining; the guarded annotation leaves other compilers on standard C++ code.

Optional waypoint, cross-sample, coefficient and trajectory costs bind with `with...Cost()`.
Coefficient costs retain their concrete type and execute before one shared spline adjoint.
Use `VoidIntegralCost` to disable quadrature (unless samples are explicitly requested).
Masks, boundary optimization, custom time/spatial maps and auxiliary state variables remain
available. See the [protocol reference](include/SplineOptimizerProtocols.md) for exact contracts.

## Failure and migration contracts

Invalid dimensions, nonfinite coefficients, collapsed cumulative knots and failed numerical
construction produce an invalid spline/status. A failed evaluation returns +infinity and a
zero gradient; failed construction clears the working spline. The prepared context can be
reused for a later valid candidate. A gradient check rejects invalid perturbations and restores
the base candidate. Always check status before consuming working state.

Migration from the old main branch:

- Replace `setConfig(config); prepareContext(problem, context)` with
  `prepareContext(problem, context, config)`. Reprepare to change configuration.
- Rename `ConvexHullBasis.hpp` to `SplineConvexHull.hpp`; include metadata via
  `SplineOptimizer.hpp` instead of downstream copies of `IntegralPointInfo.hpp`.
- Prefer the structured integral callback shown above. Legacy long callbacks remain available
  for incremental migration; all three maintained planners use the structured form.
- Use `auto`/`decltype` for composed specifications, including `.withCoefficientCost(cost)`;
  it now changes the specification type. Cost binding accepts lvalues with sufficient lifetime.
- Auxiliary maps moving the origin should accept the explicit start-time gradient in their
  extended `backward()` overload; remove application-side duplicate accumulation.

Do not compile code relying on NaN/Inf failure semantics with `-ffast-math` or
`-ffinite-math-only`. The interface target does not override consuming applications' flags.

## Build and validation

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
(cd build && ctest --output-on-failure)
./build/benchmark_optimizer 2000
./build/benchmark_sampling
```

`BUILD_TESTING`, `SPLINE_BUILD_BENCHMARKS` and `SPLINE_BUILD_EXAMPLES` can be disabled
independently. `SPLINE_NATIVE_ARCH` is off by default; comparisons must use matching flags.
CTest covers MINCO/finite-difference gradients, boundary conditions, sampler semantics and
first concurrent reads, numerical failure/recovery, masks, coefficient/hull gradients and
combined explicit start-time gradients. Assertions protecting allocation regressions remain
active in Release tests.

The [validation report](docs/refactor_validation.md) separates setup/first use from steady
sampling, fixed-candidate evaluation and complete solves. Performance depends on topology,
costs, compiler and solver behavior; a template interface alone does not imply zero overhead.
The earlier [convex-hull benchmark](docs/convex_hull_performance_zh.md) is historical data.

See [hull mathematics](docs/convex_hull_basis_zh.md),
[the optimization example](examples/convex_hull_optimization_demo.cpp),
[MIT license](LICENSE) and [MINVO attribution](THIRD_PARTY_NOTICES.md).
