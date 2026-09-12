# Native planner benchmarks

Compare each project's original MINCO, L-BFGS and penalty methods with direct Spline and
SplineOptimizer on 72 file-backed corridor and ESDF problems. The programs compile as C++17
and do not require ROS or downstream checkouts.

| Group | Frozen revision | Spline |
| --- | --- | --- |
| GCOPTER `MINCO_S3NU` | `e0444f6d47b84f972ced91746b05feb36ce1fd4f` | Quintic minimum jerk |
| SUPER `traj_opt::MINCO_S4NU` | `2ad3419c127a617c6d7df6925e81a14175a9c096` | Septic minimum snap |

Within each group, all three backends use the same decision variables, boundaries, initial
vector, weights, quadrature and original L-BFGS parameters. The two groups have different
objectives and should not be compared as identical problems.

## Run

From the repository root:

```sh
python3 benchmarks/planners/run_benchmarks.py --build build --output build/planner-results
python3 benchmarks/planners/summarize.py build/planner-results
```

The Linux runner builds before measuring, pins one allowed CPU, and runs projects serially.
Defaults are three evaluation rounds, nine batches of 100 evaluations, and five solves per
backend/case. It rotates backend order and records source, input and binary hashes.

```sh
# Reuse built binaries for one case or for numerical checks only.
python3 benchmarks/planners/run_benchmarks.py --mode run --build build \
  --output build/one-case --case esdf_small_00
python3 benchmarks/planners/run_benchmarks.py --mode run --build build \
  --output build/checks --validate-only
```

The CMake targets are `benchmark_gcopter_planner` and `benchmark_super_planner`.
They are enabled by `SPLINE_BUILD_BENCHMARKS=ON`. [Measured results](../../docs/benchmarks.md) and
[numerical notes](../../docs/numerical_notes.md) describe the comparison's limits.

## Input files and objective

[data/](data/) contains six versioned `.txt` files, each holding 12 cases:

| Family | Small | Medium | Large |
| --- | --- | --- | --- |
| Corridor | 4 segments | 12 segments | 32 segments |
| ESDF | 4 segments | 12 segments | 32 segments |

Each file stores boundaries, waypoints, durations, geometry, cost weights, flatness parameters,
quadrature and solver settings. Positions are metres and time is seconds.
[ProblemData.hpp](ProblemData.hpp) implements the format; [generate_problems.py](generate_problems.py)
documents the deterministic corpus construction. Runtime always loads the saved inputs.

The objective combines exact minimum-derivative energy, total duration, trapezoidal state
penalties and the native mapped-coordinate norm penalty. Corridor cases use per-segment boxes
and the original normalized-square convex waypoint map. ESDF cases use Cartesian waypoints and
`min_j(norm(p-center_j)-radius_j)` for disjoint spheres, with a smoothed clearance penalty.
This ESDF cost is an adaptation of the original method. Geometry preprocessing is outside timing.

## Backends and outputs

- `minco`: original construction, energy partials, quadrature, adjoint and coordinate pullbacks.
- `spline_direct`: replace interpolation and its adjoint while retaining native quadrature and maps.
- `spline_optimizer`: compose the equivalent named objective and parameterization.

| Output | Contents |
| --- | --- |
| `environment.json` | Toolchain, source/data/binary hashes and run settings |
| `checks.csv` | Same-candidate objective, gradient, coefficient and finite-difference errors |
| `evaluations.csv` | Preparation, first call and steady batch P50/P95 |
| `solves.csv` | Timings, evaluation counts, termination codes and sampled geometry |
| `comparisons.csv` | Final trajectory differences and native reevaluation at returned vectors |
| `solutions/` | Full-precision first-repetition decision vectors and breakpoints |

Regular checks use the initial vector and two deterministic perturbations; both perturbations
receive full-coordinate central differences. The initial row's zero FD error denotes a skipped
FD calculation. Debug builds additionally guard structured evaluations against Eigen allocations.
Timing excludes preparation and post-solve diagnostics. P50/P95 describe batch means, not
individual-call latency tails; the first-call metric is for a new object in an already running process.

The summary uses paired case medians and excludes negative solver codes from solve ratios.
Cost agreement uses relative error `1e-4`; trajectory agreement uses relative duration `2e-3`
and sampled position difference `0.02 m`. Solver stopping is not proof of gradient equivalence.
The saved short-segment `.candidate` is a separate diagnostic; its [replay command](../../docs/numerical_notes.md)
can fail strict numerical checks.

## Source provenance

[upstream/manifest.json](upstream/manifest.json) records exact revisions and hashes.
[prepare_upstream.py](prepare_upstream.py) verifies snapshots and generates standalone bindings:

- GCOPTER include prefixes become `frozen_gcopter/` to isolate the pinned sources.
- SUPER's ROS trajectory export is omitted, retaining Eigen aliases, numerical kernels,
  separate MINCO/banded-system/L-BFGS compilation units and flatness code.
- Native quadrature receives coefficient-storage templates and one additive ESDF callback;
  the remaining penalty formulas and SUPER's joint flatness enable condition are retained.

Updating a baseline requires replacing the source snapshot, updating its manifest and repeating
the comparisons. Original MIT/LGPL notices are preserved. SUPER-derived adapters are LGPL-3.0-or-later;
see [third-party notices](../../THIRD_PARTY_NOTICES.md).
