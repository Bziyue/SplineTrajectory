# Numerical notes

## Conditioning

Use consistent spatial and time units. Very short segments and large ratios between neighboring
durations can make high-order interpolation and gradients sensitive to roundoff. Consider
appropriate minimum durations or better-scaled variables for the application.

A saved septic-spline stress case is included at
[`super_esdf_small_05_short_segment.candidate`](../benchmarks/planners/data/super_esdf_small_05_short_segment.candidate).
Its shortest segment is about 0.017 s. On the recorded GCC 9.4 build, the direct Spline and
SplineOptimizer objectives differ from native MINCO by about `3.95e-7` relatively, while relative
gradient differences are about `2.80e-4` and `1.16e3`. This remains an unresolved numerical issue;
a high-precision reference has not established which gradients are accurate at that state.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPLINE_BUILD_BENCHMARKS=ON
cmake --build build --target benchmark_super_planner --parallel 3
./build/benchmarks/planners/benchmark_super_planner \
  --data benchmarks/planners/data --output build/short-segment \
  --case esdf_small_05 \
  --candidate benchmarks/planners/data/super_esdf_small_05_short_segment.candidate --validate-only
```

The replay reports all backends and returns nonzero when comparisons fail.

## Solver results and constraints

A solver's relative-cost stopping condition does not imply a small gradient or global optimality.
Rebuild the solver's final returned vector with `update(x)` and check success. Validate the
planning constraints before saving an independent `copyPolynomial()` for control-thread sampling.
The last evaluation may have been a rejected trial.
The planner corpus includes nonnegative SUPER stopping codes with significant same-vector
gradient differences; a close objective alone is not a gradient-equivalence check.

The examples use soft penalties. Positive sampled ESDF values do not certify continuous collision
freedom, and sampled values need not satisfy the configured margin exactly. Convex-hull bounds
apply to an entire piece only when all controls satisfy the corresponding convex set.

Invalid construction or nonfinite evaluation produces an explicit failure. Keep IEEE NaN/Inf
semantics enabled; do not compile this code with fast-math assumptions that remove finite-value checks.
