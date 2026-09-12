# Benchmarks

Enable benchmarks from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSPLINE_BUILD_BENCHMARKS=ON
cmake --build build --parallel 3
./build/benchmarks/benchmark_kernel 10000
./build/benchmarks/benchmark_optimizer 2000
./build/benchmarks/benchmark_sampling
./build/benchmarks/benchmark_convex_hull_basis build/hull.csv
```

| Source | Comparison |
| --- | --- |
| `benchmarks/kernels/` | Cubic/quintic/septic MINCO and minimum-jerk/snap kernels |
| `benchmarks/benchmark_kernel.cpp` | Construction and adjoints |
| `benchmarks/benchmark_optimizer.cpp` | Objective evaluation and solver callback overhead |
| `benchmarks/benchmark_sampling.cpp` | Direct and sequential sampling |
| `benchmarks/benchmark_convex_hull_basis.cpp` | Basis conversion, subdivision and backward propagation |
| [Planner corpus](../benchmarks/planners/README.md) | File-backed corridor and ESDF optimization |

## Native planner comparison

The corpus contains 72 problems: corridor and ESDF families, each with 12 cases at 4, 12 and
32 segments. Each group uses its own original L-BFGS and penalty formulas, comparing MINCO,
direct Spline and SplineOptimizer on identical inputs. GCOPTER uses quintic minimum jerk;
SUPER uses septic minimum snap. These are separate comparison groups.

```sh
python3 benchmarks/planners/run_benchmarks.py --build build --output build/planner-results
python3 benchmarks/planners/summarize.py build/planner-results
```

Representative measurements from 2026-09-12, Spline headers at `54abad9`, GCC 9.4, Eigen 3.3.7,
C++17 `-O3 -fno-fast-math`, Ryzen 9 7945HX, one logical CPU, no LTO:

| Group | Problem | Segments | Evaluation MINCO / Spline / Optimizer, µs | Solve MINCO / Spline / Optimizer, ms |
| --- | --- | ---: | ---: | ---: |
| GCOPTER | Corridor | 4 | 11.268 / 10.834 / 8.697 | 0.514 / 0.500 / 0.407 |
| GCOPTER | Corridor | 12 | 33.939 / 32.295 / 25.765 | 7.954 / 7.345 / 6.113 |
| GCOPTER | Corridor | 32 | 90.378 / 85.931 / 68.347 | 54.164 / 51.126 / 44.063 |
| GCOPTER | ESDF | 4 | 11.113 / 10.555 / 8.834 | 1.199 / 1.154 / 0.983 |
| GCOPTER | ESDF | 12 | 33.399 / 31.470 / 26.085 | 5.069 / 4.813 / 4.052 |
| GCOPTER | ESDF | 32 | 88.526 / 83.693 / 69.285 | 52.435 / 51.320 / 43.716 |
| SUPER | Corridor | 4 | 12.716 / 11.582 / 8.673 | 0.463 / 0.438 / 0.327 |
| SUPER | Corridor | 12 | 38.071 / 34.177 / 25.509 | 13.631 / 11.995 / 8.989 |
| SUPER | Corridor | 32 | 101.295 / 90.586 / 67.576 | 108.449 / 100.897 / 81.553 |
| SUPER | ESDF | 4 | 12.490 / 11.363 / 8.780 | 3.502 / 4.212 / 2.521 |
| SUPER | ESDF | 12 | 37.359 / 33.494 / 25.729 | 8.296 / 7.468 / 5.896 |
| SUPER | ESDF | 32 | 105.469 / 92.236 / 72.444 | 79.921 / 75.742 / 59.749 |

Values are medians across 12 case medians. Evaluation uses three rounds of nine 100-call batches;
solving uses five repetitions. I/O and geometry preparation are excluded. Results depend on compiler, hardware and stopping settings.

Regular fixed-candidate gradients matched to relative L∞ errors below `3e-13`, but independent
solves did not always produce the same final trajectory. Two SUPER direct-Spline ESDF cases
hit the line-search limit; their times remain in the table. Very short segments can also produce
large gradient differences despite close objective values. See [numerical notes](numerical_notes.md).

For source revisions, data format and detailed outputs, see the [planner benchmark guide](../benchmarks/planners/README.md).
The Linux utilities `compare_backends.py` and `measure_overhead.py` additionally support comparisons
with a checkout of the earlier `d504135` API; use `--help` for their build and input options.
