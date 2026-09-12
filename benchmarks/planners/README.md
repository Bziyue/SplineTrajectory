# Frozen GCOPTER / SUPER planner benchmarks

These benchmarks load 72 checked-in problems and compare each original planner's numerical
optimization method against direct Spline and SplineOptimizer implementations. They are standalone
C++17 programs; neither ROS nor a downstream checkout is required. The public library headers
are not modified by this benchmark. See [measured results](../../docs/planner_benchmark_results.md).

## What is held fixed

| Comparison group | Original implementation | Spline implementation | Shared components |
| --- | --- | --- | --- |
| GCOPTER | `MINCO_S3NU` at `e0444f6d47b84f972ced91746b05feb36ce1fd4f` | `QuinticSplineND<3>` | GCOPTER L-BFGS, drag-aware flatness, position/velocity/angular-rate/tilt/thrust penalties |
| SUPER | `traj_opt::MINCO_S4NU` at `2ad3419c127a617c6d7df6925e81a14175a9c096` | `SepticSplineND<3>` | SUPER L-BFGS, its flatness/pullback and position/velocity/acceleration/jerk/angular-rate/thrust penalties |

These are **two independently matched comparison groups**. Quintic minimum jerk and septic
minimum snap are different problems; cross-project timing is not a comparison of identical objectives.
Within a group, all three backends receive the same decision vector, physical boundary derivatives,
time and spatial maps, geometry, cost weights, quadrature resolution and L-BFGS parameters.

The objective is

\[
J = w_E\int\|p^{(s)}(t)\|^2\,dt + \rho\sum_i T_i
  + \sum_{i,k} w_k\frac{T_i}{K}L(p,v,a,j)
  + \sum_{\text{mapped blocks }q}\max(\|q\|^2-1,0)^3.
\]

Here `s=3` for GCOPTER and `s=4` for SUPER. Energy is exact; penalties use each native planner's
trapezoidal quadrature (`w_k=1/2` at endpoints). Integral state drift through snap is included in
duration gradients. Native `smoothedL1` is used on constraint violations.

- **Corridor:** per-piece axis-aligned half-space boxes and the native normalized-square convex
  combination map for inner waypoints. Each waypoint has eight unconstrained coordinates.
  Fixture map boxes supply an analytic inverse for the initial point; no initialization L-BFGS
  or corridor-generation LP is hidden in the timing. All three backends use this same initial vector.
- **ESDF:** Cartesian inner waypoints and a shared analytic distance field,
  `d(p) = min_j (norm(p-center_j)-radius_j)`. All spheres in this corpus are disjoint, so this is
  their exact signed Euclidean distance field. It is nondifferentiable at sphere centers and
  nearest-sphere ties; the implementation specifies deterministic conventions there.
  The added penalty is `obstacle_weight * smoothedL1(clearance-d(p), smooth)` with its spatial
  gradient. Native corridor planes are empty in this case. All dynamic penalties, energy and
  time costs retain the project's original method.

The ESDF variant is a benchmark adaptation of the original optimization method, not a claim
that either frozen project contained this ESDF backend. It models a point trajectory; the
configured clearance supplies a margin. Soft penalties and dense samples do not certify
continuous collision freedom or exact satisfaction of that margin.

## Three backends

1. `minco`: original construction, energy partials, original coefficient quadrature, original
   adjoint, native time/spatial pullbacks, native mapped-domain penalty.
2. `spline_direct`: replace the interpolation/energy/adjoint kernel with Spline, keeping the
   original quadrature and mapping helpers. Preallocated row-major coefficient partials avoid
   a layout conversion on every evaluation.
3. `spline_optimizer`: use a named Objective with duration/integral/decision members and a
   Parameterization matching the same maps. The point penalty adapter has the same formulas;
   normalized quadrature and the specialized energy gradient belong to SplineOptimizer.

Each group uses its own original L-BFGS for **all three backends**. SUPER retains its separate
MINCO/banded-system/L-BFGS translation units; no LTO is enabled. This preserves its original
compilation structure, which differs from the header-only Spline implementation. The benchmark
is the numerical optimization core, not a ROS node, map builder, complete planning pipeline or
flight simulation. Geometry preprocessing and input I/O are excluded from evaluation/solve timing.

## Checked-in input files

[data/](data/) contains six versioned text files, each with 12 cases:

| Family | Small | Medium | Large |
| --- | ---: | ---: | ---: |
| Corridor | 4 pieces, 12 cases | 12 pieces, 12 cases | 32 pieces, 12 cases |
| ESDF | 4 pieces, 12 cases | 12 pieces, 12 cases | 32 pieces, 12 cases |

Case IDs are stable, such as `esdf_medium_07`. Cases vary path shape, times, time weight,
corridor width, obstacle centers and radii. Runtime performs no random problem generation.
[generate_problems.py](generate_problems.py) documents how the corpus was created; the committed
files, rather than regeneration with another platform's math library, are authoritative inputs.

The grammar is a sequence of whitespace-delimited named fields, with a version header and case
count. [ProblemData.hpp](ProblemData.hpp) is the strict loader. Each case stores:

- ID, scenario, scale, segment count and trapezoid resolution;
- L-BFGS memory, past window, relative-cost tolerance, gradient tolerance, iteration budget and minimum step;
- energy/time weights, smoothing width, physical flatness parameters and both projects' bounds/weights;
- ESDF margin and weight, start/end velocity/acceleration/jerk, durations and all interpolation points;
- per-waypoint map-box center/half extents, per-piece corridor lower/upper bounds and sphere center/radius.

Positions are metres; times are seconds. Boundaries are vectors written derivative by derivative.
GCOPTER uses start/end velocity and acceleration; SUPER additionally uses jerk. Time origins are zero.
The corpus sets SUPER's waypoint-attractor weight to zero; the adapter also supports that native
term with the original input waypoints as attractors and zero dead distance. Magnitude-bound and
weight ordering follows the respective native constraint function, not a shared renamed profile.

## Build and run

From the SplineTrajectory repository, using GCC 9/Eigen 3.3.7 or another supported C++17 toolchain:

```sh
python3 benchmarks/planners/run_benchmarks.py \
  --build ../.scratch/planner-benchmark/build-release \
  --output ../.scratch/planner-benchmark/results
python3 benchmarks/planners/summarize.py ../.scratch/planner-benchmark/results
```

The runner builds both binaries first, then binds itself and its children to the lowest allowed
CPU. It runs projects serially. Within each case it rotates the three backend orders for timing
rounds and solve repetitions. Defaults: 100 evaluations per batch, nine batches per round,
three rounds, five complete solves per backend. It records input/source/binary hashes and
compiler configuration in `environment.json`. `--mode run` reuses existing binaries and records
their hashes; callers must rebuild if they need changed source to take effect.

Useful alternatives:

```sh
# One exact file-backed case, with the same comparison checks:
python3 benchmarks/planners/run_benchmarks.py --mode run \
  --build ../.scratch/planner-benchmark/build-release --output ../.scratch/planner-benchmark/one \
  --case esdf_small_00 --repeats 200 --solve-repeats 7

# Check the entire corpus without timing or solving:
python3 benchmarks/planners/run_benchmarks.py --mode run \
  --build ../.scratch/planner-benchmark/build-release --output ../.scratch/planner-benchmark/checks \
  --validate-only

# CMake targets and representative CTest checks:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark_gcopter_planner benchmark_super_planner -j2
(cd build && ctest -R planner_ --output-on-failure)
```

`SPLINE_BUILD_BENCHMARKS=OFF` excludes these dependencies and targets. New targets and the native
SUPER static library explicitly compile as `-std=c++17`, with extensions disabled. Owned code
uses the repository's strict warnings. Frozen headers are system includes; original standalone
SUPER translation units are compiled separately, without applying owned-source `-Werror` to them.

## Checks and outputs

Every selected case checks the initial vector and two deterministic perturbations. One shortens
durations and enlarges mapped norms to activate dynamic/domain penalties. For **both** perturbed
vectors, all decision coordinates are checked by central differences with step
`2e-6 * max(1, abs(x_i))`. Both the original MINCO and replacement backends undergo this check.
Any nonfinite result or failed fixed-candidate tolerance causes a nonzero exit status.
In Debug builds, the structured evaluator also runs with Eigen heap allocation disabled during
these checked evaluations; preparation and caller-owned gradient storage are outside that guard.
This checks Eigen allocations, not arbitrary allocations by other libraries.

Each project writes:

| File | Contents |
| --- | --- |
| `checks.csv` | Cost, complete-gradient and coefficient relative errors; full finite-difference error |
| `evaluations.csv` | Separate constructor preparation, first evaluation, steady P50/P95 and rotating-order round |
| `solves.csv` | Solve wall time, callback/iteration/line-search counts, termination code, returned-vector objective/gradient, duration and dense geometry checks |
| `comparisons.csv` | Final objective/duration and position/velocity differences; reference cost, gradient and coefficients reevaluated at each replacement's returned vector |
| `solutions/*.txt` | Full-precision returned decision vectors and breakpoints from the first solve repetition of every backend/case |

The initial candidate's `fd_relative=0` means finite differences were not run at that vector;
the two perturbed candidates carry the finite-difference measurements. Final-vector reference
reevaluation has a separate relative tolerance of `2e-7` when both solvers return nonnegative
codes. Failed solves retain their reevaluation errors without aborting the remaining corpus.
Returned-vector gradient and coefficient errors are reported separately, without treating a
solver stopping code as proof of their equivalence. They can expose instability at very short
durations that the regular corpus candidates do not reach.

P50/P95 are quantiles of **batch mean** evaluation times; with nine batches the nearest-rank P95
is the maximum batch. They are not individual-call latency tails. Preparation includes backend
storage setup but excludes file loading/common geometry. Solve timing excludes preparation and
post-solve checking. Post-solve sampling uses 101 points per piece for geometry and 1001 normalized
whole-trajectory times for pairwise position/velocity differences. Comparison at normalized time
is explicit because optimized durations can differ.
The first-call metric uses a newly constructed backend in an already running, validated process;
it does not measure process startup or cold CPU caches.

A negative L-BFGS code is a failed or budget-limited solve and is not counted as a stopped pair.
Its returned vector is a diagnostic artifact, not an accepted solution.
Code 1 is the native past/delta stopping condition, not gradient convergence. The summary reports
per-case medians before aggregating; solve speed ratios use only pairs with nonnegative codes.
Cost agreement uses relative error `1e-4`; trajectory agreement uses duration relative error
`2e-3` and maximum position difference `0.02 m`. These are reporting thresholds, not relaxed
fixed-candidate correctness thresholds. Failed solves, unequal final trajectories and margin
violations remain in the raw outputs and the results report.

The small saved `.candidate` file in `data/` preserves a short-segment numerical stress case
encountered by the direct Spline path. It is excluded from normal `.txt` dataset loading.
Replay it using the matching executable and file-backed source case:

```sh
../.scratch/planner-benchmark/build-release/benchmarks/planners/benchmark_super_planner \
  --data benchmarks/planners/data --output ../.scratch/planner-benchmark/short-segment \
  --case esdf_small_05 --candidate benchmarks/planners/data/super_esdf_small_05_short_segment.candidate \
  --validate-only
```

Replay applies the same strict cost/gradient/coefficient comparisons at the saved vector only;
it does not perturb the vector or run finite differences at this numerically sensitive state.
It records all three backends before returning failure for finite numerical mismatches.
This known numerical stress diagnostic returns nonzero on the reported GCC 9 build; see the measured report for the
observed error and solver status. Do not use its output directory as an input dataset directory.

## Source provenance and isolation

[upstream/manifest.json](upstream/manifest.json) pins every original source's commit and SHA-256.
Those files are unedited snapshots. [prepare_upstream.py](prepare_upstream.py) verifies the hashes
and generates a standalone build directory with these explicit adaptations:

- GCOPTER include prefixes become `frozen_gcopter/`, preventing accidental selection of the
  library's older vendored `include/gcopter` copy. Only native mapping/penalty functions are
  extracted from its full planner class; corridor generation and geometry LP are not compiled.
- SUPER's MINCO declarations retain Eigen aliases but forward-declare the ROS-dependent
  trajectory type. Three unused `getTrajectory()` definitions are omitted; all construction,
  energy, banded solve and adjoint bodies remain unchanged. Banded-system includes Eigen directly.
  The original L-BFGS and flatness code is copied into the standalone build.
- Native constraint functions gain templates for coefficient/gradient storage types, including
  C++'s required dependent `template` syntax. One additive position callback supplies the ESDF
  term. The native quadrature and remaining penalty logic are retained.
- The structured point adapter ports the same formulas. SUPER's original joint condition
  enabling angular-rate and thrust penalties is deliberately retained for equivalence.

The generated bindings are build artifacts, not another maintained numerical implementation.
Updating a baseline requires explicit source replacement, manifest regeneration, adapter review
and repetition of the gradient/solution comparisons. [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
records MIT and LGPL notices. SUPER-derived benchmark adapters are LGPL-3.0-or-later; this does
not change the MIT license of the three public SplineTrajectory headers.
