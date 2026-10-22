# SplineTrajectory

Header-only C++17 / Eigen library for smooth N-dimensional minimum-control-effort trajectories.
It solves the same clamped spline problem as [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER),
with an O(N) block-tridiagonal solver, analytic energy gradients, a composable optimizer
interface, and differentiable convex-hull conversions.

**English** | [中文](README_zh.md)

[**Online API reference · English / 中文**](https://bziyue.github.io/SplineTrajectory/api/?lang=en) — signatures, data shapes, bilingual search and interactive plots. No build required.

## ✨ Overview

- Cubic, quintic and septic minimum-control-effort splines in arbitrary dimensions.
- O(N) construction and adjoint propagation to durations, waypoints and boundary states.
- Analytic energy gradients derived from Hamiltonian conservation and variational principles.
- A generic `SplineOptimizer` for reusable variable maps and objective components.
- Sequential sampling with reusable derivative coefficients and no source warm-up.
- Bezier / MINVO convex-hull conversion, subdivision and gradient propagation.

| Spline Type | Polynomial Degree | Minimizes the Integral of | MINCO Equivalent |
| --- | ---: | --- | --- |
| Cubic | 3 | Squared acceleration | MINCO S2 |
| Quintic | 5 | Squared jerk | MINCO S3 |
| Septic | 7 | Squared snap | MINCO S4 |

The library is organized into three public headers:

| Header | Features |
| --- | --- |
| `SplineTrajectory.hpp` | Splines, energy gradients, piecewise polynomials and sampling |
| `SplineOptimizer.hpp` | Decision variables, objective composition and gradient propagation |
| `SplineConvexHull.hpp` | Bezier / MINVO control points, subdivision and adjoints |

## 🔁 Gradient Propagation

`spline.backward()` maps coefficient and independent duration partials to gradients of
segment durations, all waypoints (including endpoints) and endpoint derivatives. The adjoint reuses the
construction factorization.

For sampled-state costs, `SplineOptimizer` handles quadrature, coefficient/time partials
and variable-map pullbacks. Objective components supply local gradients with respect to
position, velocity, acceleration and higher derivatives, with explicit time dependence
when needed. See [sampling and gradients](docs/trajectory.md).

## ⚡ Analytic Energy Gradients

`spline.energyGradient()` computes minimum-control-effort energy gradients directly from
the spline solution. Duration gradients use boundary information in O(1) per segment;
waypoint gradients use jumps in high-order derivatives at the joins. This avoids assembling
coefficient partials and running a separate adjoint for the energy term.

## 🧩 SplineOptimizer

`SplineOptimizer<Spline>` owns the prepared problem and reusable evaluation workspace.
An objective is a user-defined struct holding callable members such as `integral` and
`coefficient`. Each cost implements `operator()`; only supplied members participate.

- Map decision variables to positive durations, spatial points and optional auxiliary states.
- Combine energy, time, sampled-state and control-point costs.
- Supply costs and gradients to L-BFGS or another numerical solver through `evaluate()` and an explicit scalar callback.

See the quick start below and the [optimizer reference](docs/optimizer.md)
for custom maps and objective protocols.

## 📐 Spline Convex-Hull Conversion

`SplineConvexHull.hpp` converts each polynomial segment, or one of its derivatives, into
Bezier or MINVO control points without changing the represented curve.

- `toBezier()` and `toMINVO()` create control-point workspaces; included MINVO bases cover degrees 0–7.
- Binary subdivision produces tighter local bounds while preserving the trajectory.
- `prepare()` prepares conversion data; `update()` and `backwardAdd()` reuse workspace storage.
- Control-point gradients propagate to both polynomial coefficients and segment durations.

If all controls lie in a convex corridor, the entire corresponding curve piece lies in it.
Velocity and acceleration controls similarly provide conservative bounds over a whole piece.
Soft penalties require a separate feasibility check; ESDF values at controls alone do not
certify collision freedom in nonconvex free space.

![Bezier subdivision and MINVO convex hulls](docs/images/convex_hull_basis_comparison.png)

[Conversion guide](docs/convex_hull.md) · [Derivation (中文)](docs/convex_hull_basis_zh.md) ·
[Optimizer example](examples/convex_hull_optimizer.cpp)

## 📈 Benchmarks

### GCOPTER and SUPER: identical optimization problems

The benchmark loads 72 saved corridor and ESDF problems: 12 cases at each of 4, 12 and
32 segments for each problem family. Within each planner group, all backends use identical
inputs, penalties, initialization and solver settings, including that project's original
L-BFGS. GCOPTER compares quintic minimum jerk; SUPER compares septic minimum snap.
The two planner groups solve different formulations and are compared separately.

Recorded on 2026-09-12 with headers at `54abad9`: GCC 9.4, Eigen 3.3.7, C++17
`-O3 -fno-fast-math`, Ryzen 9 7945HX, one logical CPU, no LTO.
These historical results and the kernel plots below do not measure the current `main` revision.

| Group | Problem | Segments | Cost + Gradient, MINCO / Optimizer (µs) | Solve, MINCO / Optimizer (ms) |
| --- | --- | ---: | ---: | ---: |
| GCOPTER | Corridor | 4 | 11.268 / 8.697 | 0.514 / 0.407 |
| GCOPTER | Corridor | 12 | 33.939 / 25.765 | 7.954 / 6.113 |
| GCOPTER | Corridor | 32 | 90.378 / 68.347 | 54.164 / 44.063 |
| GCOPTER | ESDF | 4 | 11.113 / 8.834 | 1.199 / 0.983 |
| GCOPTER | ESDF | 12 | 33.399 / 26.085 | 5.069 / 4.052 |
| GCOPTER | ESDF | 32 | 88.526 / 69.285 | 52.435 / 43.716 |
| SUPER | Corridor | 4 | 12.716 / 8.673 | 0.463 / 0.327 |
| SUPER | Corridor | 12 | 38.071 / 25.509 | 13.631 / 8.989 |
| SUPER | Corridor | 32 | 101.295 / 67.576 | 108.449 / 81.553 |
| SUPER | ESDF | 4 | 12.490 / 8.780 | 3.502 / 2.521 |
| SUPER | ESDF | 12 | 37.359 / 25.729 | 8.296 / 5.896 |
| SUPER | ESDF | 32 | 105.469 / 72.444 | 79.921 / 59.749 |

MINCO denotes the original planner backend; Optimizer uses SplineTrajectory with
`SplineOptimizer`. Values are medians across 12 case medians, excluding file I/O and geometry
preparation. Regular fixed-candidate gradient comparisons had relative errors below `3e-13`.
Independent solves did not always reach the same final trajectory, and very short segments
expose unresolved gradient discrepancies; see [numerical notes](docs/numerical_notes.md).

[Full results, including direct Spline](docs/benchmarks.md) ·
[Corpus and reproduction commands](benchmarks/planners/README.md)

### Spline kernels

Earlier kernel measurements compare construction, general adjoints and analytic energy gradients:

![Trajectory construction speed](docs/images/construction_speed.png)

![Gradient back-propagation speed](docs/images/backprop_speed.png)

![Analytic energy gradient speed](docs/images/analytic_grad_speed.png)

Current benchmark sources and commands are listed in the [benchmark guide](docs/benchmarks.md).

## 🛠️ Refactored Projects

Related planner implementations using SplineTrajectory, each with its own integration version:

| Project | Original | Refactored |
| --- | --- | --- |
| **EGO-Planner-v2** | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/EGO-Planner-v2) | [Refactored](https://github.com/Bziyue/EGO-Planner-v2) |
| **GCOPTER** | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/GCOPTER) | [Refactored](https://github.com/Bziyue/GCOPTER) |
| **SUPER** | [HKU-MaRS](https://github.com/hku-mars/SUPER) | [Refactored](https://github.com/Bziyue/SUPER) |
| **DDR-opt** | [ZJU-FAST-Lab](https://github.com/ZJU-FAST-Lab/DDR-opt) | [Refactored](https://github.com/Bziyue/DDR-opt) |

## 🚀 Quick Start

The library requires C++17 and Eigen ≥ 3.3. The repository build uses CMake ≥ 3.12;
the test tools also require Python 3.

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
For local performance measurements, add `-DSPLINE_NATIVE_ARCH=ON` to tune repository
executables for the build machine's CPU. This option does not propagate to consumers;
see [build and integration](docs/building.md) for CMake integration and compiler settings.

### Physical input and optimization choices

`SplineInput<DIM>` contains N durations, N+1 waypoint rows, endpoint derivatives and a
`start_time` on the caller's trajectory time axis; this need not be an absolute clock.
With `spline.update(input)`, **every waypoint is an interpolation constraint**.
With `optimizer.prepare(input, options, objective)`, values selected by `options.mask` are
**optimization initial guesses**. By default, durations and interior waypoints can change;
endpoint positions and endpoint derivatives remain fixed. To keep all waypoints fixed, set
`options.mask.waypoints.assign(input.waypoints.rows(), 0)`. Auxiliary maps are a separate,
explicit mechanism that can also transform fixed physical values.

The following is a complete program. Endpoint derivatives default to zero; quintic splines
use velocity and acceleration. Both built-in weights default to **zero**: `NoObjective` means
no custom costs, and does not enable minimum-jerk energy by itself.

```cpp
#include "SplineOptimizer.hpp"
#include <cmath>
#include <iostream>

int main()
{
    using Spline = SplineTrajectory::QuinticSplineND<3>;
    SplineTrajectory::SplineInput<3> input;
    input.durations = {1.2, 1.5};
    input.waypoints.resize(3, 3);
    input.waypoints << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
    input.start_time = 2.0;
    input.boundary.start.v << 0.2, 0, 0;

    Spline spline;
    if (!spline.update(input)) return 1;
    const auto state = spline.polynomial().evaluateDerivatives<2>(2.5);
    // Equivalent construction from N+1 times on the trajectory time axis:
    Spline from_breakpoints;
    if (!from_breakpoints.updateFromBreakpoints(
            {2.0, 3.2, 4.7}, input.waypoints, input.boundary)) return 1;

    SplineTrajectory::SplineOptimizer<Spline> optimizer;
    SplineTrajectory::OptimizerOptions options;
    options.energy_weight = 1.0;   // Minimum jerk for this spline family.
    options.duration_weight = 0.5; // Linear total-duration cost.
    SplineTrajectory::NoObjective objective; // No custom cost components.
    if (!optimizer.prepare(input, options, objective)) return 1;
    auto x = optimizer.initialGuess();
    Eigen::VectorXd gradient; // Resizes once; matching sizes reuse storage.
    const auto result = optimizer.evaluate(x, gradient, objective);
    if (!result) return 1;

    // This example evaluates the initial guess. A numerical solver repeatedly
    // calls evaluate(); after solving, use its final returned x here.
    if (!optimizer.update(x)) return 1;
    // Check your planning task's constraints before accepting this candidate.
    auto trajectory = optimizer.copyPolynomial();
    auto sampler = trajectory.makeSampler<2>();
    const auto command = sampler.evaluateDerivatives(trajectory.startTime());
    std::cout << "Initial cost: " << result.cost << '\n';
    return std::isfinite(result.cost) && gradient.allFinite() &&
           state[0].allFinite() && command[0].allFinite() ? 0 : 1;
}
```

`prepare(input, objective)` uses default options; `prepare(input, options, objective)` makes
weights and variable selection explicit. Preparation copies the physical input and prepares
the library's buffers. Prepare user-owned hulls, maps and cost buffers before the solver loop
as shown in the [convex-hull example](examples/convex_hull_optimizer.cpp).

### Solve and deliver a trajectory

Use the solver's final returned decision vector in this order:

1. Call `optimizer.update(x_final)` and check success.
2. Check the planning task's constraints and accept or reject the candidate.
3. Save `auto trajectory = optimizer.copyPolynomial()`.
4. Keep that snapshot alive and unchanged; create a sampler for each control-thread traversal.

`evaluate()` replaces the current candidate. `polynomial()` borrows the last built candidate,
which may be a rejected line-search trial; it is not automatically the solver's final accepted
trajectory. Updates invalidate borrowed views and samplers. A saved polynomial owns independent
data. See the [L-BFGS examples](examples/README.md) for actual optimization and solver-status handling.

Choose costs by the planning requirement; custom functors return a `double` (cost) or an
`EvaluationResult` when they can reject a candidate, and add partials to their gradient view.

| Requirement | Entry |
| --- | --- |
| Smoothness and total duration | `energy_weight`, `duration_weight` |
| Pointwise obstacles, speed or acceleration | `integral` |
| Bezier / MINVO control constraints | `coefficient` |
| Waypoint, boundary or additional duration costs | `parameter` |
| Coupled costs across samples | `sample` |
| Regularization in decision coordinates | `decision` |

Low-level spline, polynomial and hull APIs expect valid shapes, finite inputs, positive durations
and representable arithmetic. Check spline construction status before querying. The optimizer
validates reference inputs during preparation and decisions during evaluation. Failed candidates
remain invalid while their workspace is retained for fixed-topology retries; callbacks own any
allocations they perform. See the [optimizer contracts](docs/optimizer.md).

## 🧰 Examples and Documentation

| Example | Objective |
| --- | --- |
| [Basic optimizer](examples/basic_optimizer.cpp) | Minimum jerk and total duration |
| [Corridor](examples/corridor_optimizer.cpp) | Segment corridors, speed and acceleration |
| [ESDF](examples/esdf_optimizer.cpp) | Distance-field obstacle avoidance and derivative limits |
| [Dynamics](examples/dynamics_optimizer.cpp) | Speed and acceleration penalties |
| [Convex hull](examples/convex_hull_optimizer.cpp) | Bezier corridor and derivative control-point bounds |

[Online API reference](https://bziyue.github.io/SplineTrajectory/api/?lang=en) ·
[Sampling and gradients](docs/trajectory.md) · [Optimizer reference](docs/optimizer.md) ·
[Convex hulls](docs/convex_hull.md) · [Benchmarks](docs/benchmarks.md) · [Numerical notes](docs/numerical_notes.md)

## 🔭 Future Plans

- [x] MINCO-equivalent gradient propagation
- [x] Clamped septic spline support
- [x] Spline convex-hull conversion support
- [ ] N-dimensional non-uniform B-spline support

## 📄 License

The public headers are [MIT licensed](LICENSE). Bundled dependencies retain their
[third-party licenses](THIRD_PARTY_NOTICES.md).

## 🙏 Acknowledgments

- [Eigen](https://eigen.tuxfamily.org/) for linear algebra.
- [MINCO / GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER) and [SUPER](https://github.com/hku-mars/SUPER) for the reference implementations.
- Classical spline interpolation theory and the minimum-norm formulation.
