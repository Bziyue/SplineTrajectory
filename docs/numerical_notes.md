# Numerical notes

## Conditioning

Use consistent spatial and time units. Very short segments or large duration ratios increase
roundoff sensitivity in high-order splines. Scale optimization variables and choose minimum
durations appropriate to the planning task.

## Short-segment stress case

The saved septic-spline case
[`super_esdf_small_05_short_segment.candidate`](../benchmarks/planners/data/super_esdf_small_05_short_segment.candidate)
has a shortest duration of about 0.017 s and reproduces a known gradient disagreement between
SplineTrajectory and MINCO. Use the following command to inspect costs, coefficients and
gradients at that candidate:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPLINE_BUILD_BENCHMARKS=ON
cmake --build build --target benchmark_super_planner --parallel
./build/benchmarks/planners/benchmark_super_planner \
  --data benchmarks/planners/data --output build/short-segment \
  --case esdf_small_05 \
  --candidate benchmarks/planners/data/super_esdf_small_05_short_segment.candidate --validate-only
```

The replay reports all backends and returns nonzero when comparisons fail.

## Solver results

Inspect the solver's termination status, rebuild its final vector with `update(x)`, and evaluate
the planning constraints. Save the accepted trajectory with `copyPolynomial()` for sampling.

Soft penalties trade constraint violations against cost. Convex-hull bounds cover an entire
piece when all controls satisfy the same convex constraint. For nonconvex ESDF free space,
use trajectory collision checks to assess the complete path.

Construction and evaluation report invalid or nonfinite results through `Status`.
Compiler settings are described in [build and integration](building.md).
