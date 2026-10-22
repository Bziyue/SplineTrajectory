# Benchmarks

The benchmarks compare spline kernels and complete trajectory optimization. All implementations
in each comparison use the same inputs and compiler settings. Results use C++17, Eigen and
Release optimization; **native** adds CPU-specific optimization with `-march=native`, while
**generic** uses the compiler's default target. The README shows native results.

## Spline kernels

Construction and gradient tests cover cubic, quintic and septic splines at **25 sizes from
4 to 10,000 segments**, with three sets of 3D waypoints and positive, nonuniform durations.
The [README plots](../README.md#spline-kernels-minco-and-large_scale_traj_optimizer) compare
SplineTrajectory with GCOPTER MINCO and `large_scale_traj_optimizer`.

| Operation | SplineTrajectory | MINCO | large_scale |
| --- | --- | --- | --- |
| Construction | `update` | `setParameters` | `generate` |
| General adjoint | `backward` | `propogateGrad` | — |
| Energy gradients | `energyDurationGradient` + `energyWaypointGradient` | Energy partials + `propogateGrad` | `getGradT` + `getGradInnerP` |

Measurements use repeated calls with prepared workspaces and report the median of repeated
batches for each input, then the median across inputs. Construction times the interpolation
step; gradient operations start from the constructed spline. Large_scale's trajectory export
is a separate operation, and its gradient-return allocations are included in gradient timings.

SplineTrajectory computes analytic energy gradients directly. MINCO propagates energy partials
through an adjoint. The energy-gradient plot compares quintic/septic Spline with MinJerk/MinSnap.
Spline gradients include all waypoint positions; common gradient comparisons cover interior
waypoints and segment durations.

At **10,000 segments**, times are **µs/call**:

| Degree | Construction: Spline / MINCO / large_scale (µs) | Adjoint: Spline / MINCO (µs) | Energy gradients: Spline / large_scale (µs) |
| ---: | ---: | ---: | ---: |
| 3 | 135.704 / 523.994 / — | 143.256 / 334.539 | — |
| 5 | 214.582 / 1123.240 / 602.896 | 212.217 / 561.306 | 20.367 / 185.747 |
| 7 | 333.401 / 2027.357 / 1267.267 | 358.002 / 905.576 | 22.884 / 275.796 |

Large_scale provides quintic and septic implementations. Cubic energy gradients are outside
the plotted comparison. Numerical checks compare energy, coefficients and common gradients;
their maximum relative differences from MINCO are `3.25e-12`, `6.98e-14` and `1.29e-11`.
Relative differences use `max(1, reference L∞ norm)` as denominator.

## Planner comparisons

Each project uses 72 saved problems: corridor and ESDF families, with 12 cases at each of
4, 12 and 32 segments. GCOPTER uses quintic minimum jerk; SUPER uses septic minimum snap.
Within each project, the backends share decision variables, geometry, costs, initial guesses,
quadrature and the project's L-BFGS settings.

- **MINCO**: the project's original backend.
- **Direct**: SplineTrajectory interpolation and gradients with the original quadrature and maps.
- **Optimizer**: SplineTrajectory with `SplineOptimizer` cost components and parameterization.

Each row reports the median across 12 case medians. Evaluation uses three rounds of nine
300-call batches; solving uses three repetitions. Timing covers objective evaluation or the
solver loop, with input preparation and result inspection performed separately.

### Native

| Project | Problem | Segments | Cost + gradient, MINCO / Direct / Optimizer (µs) | Solve, MINCO / Direct / Optimizer (ms) |
| --- | --- | ---: | ---: | ---: |
| GCOPTER | Corridor | 4 | 7.204 / 6.834 / 5.893 | 0.336 / 0.323 / 0.274 |
| GCOPTER | Corridor | 12 | 21.487 / 20.265 / 17.602 | 4.804 / 4.482 / 3.963 |
| GCOPTER | Corridor | 32 | 57.016 / 53.811 / 46.872 | 30.261 / 29.623 / 25.056 |
| GCOPTER | ESDF | 4 | 7.218 / 6.858 / 6.331 | 0.829 / 0.788 / 0.711 |
| GCOPTER | ESDF | 12 | 21.518 / 20.306 / 19.003 | 3.375 / 3.177 / 2.949 |
| GCOPTER | ESDF | 32 | 57.131 / 53.763 / 49.500 | 32.517 / 30.895 / 29.390 |
| SUPER | Corridor | 4 | 7.887 / 7.194 / 5.754 | 0.308 / 0.270 / 0.211 |
| SUPER | Corridor | 12 | 23.512 / 21.386 / 17.029 | 8.125 / 7.575 / 5.412 |
| SUPER | Corridor | 32 | 62.772 / 56.940 / 44.880 | 60.795 / 57.230 / 44.637 |
| SUPER | ESDF | 4 | 7.919 / 7.135 / 5.694 | 2.857 / 2.195 / 2.170 |
| SUPER | ESDF | 12 | 23.748 / 21.250 / 16.967 | 5.416 / 4.913 / 3.990 |
| SUPER | ESDF | 32 | 63.220 / 56.503 / 45.159 | 49.105 / 45.243 / 38.576 |

### Generic

| Project | Problem | Segments | Cost + gradient, MINCO / Direct / Optimizer (µs) | Solve, MINCO / Direct / Optimizer (ms) |
| --- | --- | ---: | ---: | ---: |
| GCOPTER | Corridor | 4 | 8.052 / 7.506 / 6.162 | 0.380 / 0.346 / 0.290 |
| GCOPTER | Corridor | 12 | 24.195 / 22.462 / 18.381 | 5.516 / 5.175 / 4.408 |
| GCOPTER | Corridor | 32 | 63.789 / 59.720 / 48.930 | 38.059 / 36.445 / 31.243 |
| GCOPTER | ESDF | 4 | 7.867 / 7.347 / 6.137 | 0.860 / 0.796 / 0.683 |
| GCOPTER | ESDF | 12 | 23.490 / 21.838 / 18.234 | 3.661 / 3.364 / 2.849 |
| GCOPTER | ESDF | 32 | 62.500 / 58.395 / 48.439 | 35.507 / 34.045 / 30.486 |
| SUPER | Corridor | 4 | 8.134 / 7.244 / 6.413 | 0.306 / 0.272 / 0.238 |
| SUPER | Corridor | 12 | 24.114 / 21.432 / 18.890 | 8.565 / 7.399 / 6.840 |
| SUPER | Corridor | 32 | 64.595 / 57.166 / 50.121 | 73.289 / 65.841 / 60.687 |
| SUPER | ESDF | 4 | 7.995 / 7.166 / 6.813 | 2.757 / 2.064 / 2.257 |
| SUPER | ESDF | 12 | 23.863 / 21.244 / 20.120 | 5.405 / 4.888 / 4.539 |
| SUPER | ESDF | 32 | 63.397 / 56.199 / 52.318 | 54.178 / 45.731 / 42.621 |

### Planner validation

Initial and perturbed candidate checks compare costs, gradients and coefficients at the same
decision vector. Maximum gradient differences are `4.13e-14` for GCOPTER and `2.21e-13` for
SUPER; central-difference errors are below `1.3e-6`.

Solve timings use each backend's returned solution. All runs reported successful or stopped
solver codes. Under the corpus's trajectory-agreement thresholds, native Optimizer and MINCO
solutions agree in 67/72 GCOPTER cases and 61/72 SUPER cases. Some short-segment SUPER solutions
exhibit gradient disagreement on reevaluation; the saved stress case and replay command are
in [numerical notes](numerical_notes.md).

## Run the benchmarks

Build from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPLINE_BUILD_BENCHMARKS=ON
cmake --build build --parallel
python3 benchmarks/planners/run_benchmarks.py --mode run \
  --build build --output build/planner-results
python3 benchmarks/planners/summarize.py build/planner-results
```

Add `-DSPLINE_NATIVE_ARCH=ON` to enable CPU-specific optimization. Kernel, sampling and
convex-hull benchmark executables are available in `build/benchmarks/`.

| Source | Coverage |
| --- | --- |
| [Kernel comparisons](../benchmarks/kernels/) | Cubic/quintic/septic MINCO and minimum-jerk/snap kernels |
| [benchmark_kernel.cpp](../benchmarks/benchmark_kernel.cpp) | Construction and adjoints |
| [benchmark_optimizer.cpp](../benchmarks/benchmark_optimizer.cpp) | Objective evaluation and solver callback overhead |
| [benchmark_sampling.cpp](../benchmarks/benchmark_sampling.cpp) | Direct and sequential sampling |
| [benchmark_convex_hull_basis.cpp](../benchmarks/benchmark_convex_hull_basis.cpp) | Basis conversion, subdivision and gradients |
| [Planner corpus](../benchmarks/planners/README.md) | Saved inputs, backend configuration and result files |

GCOPTER and SUPER source revisions are recorded in the [upstream manifest](../benchmarks/planners/upstream/manifest.json).
The large_scale comparison uses the bundled [minimum-jerk and minimum-snap implementations](../include/large_scale_traj_optimizer/).
