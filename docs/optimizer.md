# SplineOptimizer protocols

`SplineOptimizer<Spline>` connects physical trajectory data to an external numerical solver.
Objectives and maps are ordinary user-defined C++17 types. No inheritance or type-erased callback
is required. Start with built-in smoothness and duration weights; add only the cost stages needed.

## Physical input and optimization choices

`SplineInput<DIM>` contains N positive finite `durations`, an (N+1) × DIM `waypoints` matrix,
`boundary.start/end.v/a/j`, and a finite `start_time` on the caller's trajectory time axis.
All positions, including endpoints, live in `waypoints`; boundary conditions contain derivatives.
Only derivatives supported by the selected spline participate in validation and construction.

In `spline.update(input)`, every waypoint is an interpolation constraint. In
`optimizer.prepare(input, options, objective)`, mask-selected values are optimization **initial
guesses**. The constructed candidate interpolates its current decoded waypoints, which may differ
from the original input. The default mask optimizes all durations and interior waypoints, while
endpoint positions and derivatives remain fixed as direct coordinates.

`OptimizerOptions::mask` owns the selection:

- Empty `durations` activates all durations. An explicit vector has N entries.
- Empty `waypoints` activates interior points. An explicit vector has N+1 entries, including endpoints.
- Nonzero entries are active. `start` and `end` select `v/a/j`; unsupported selections fail preparation.
- `OptimizationMask::none(N)` disables all direct coordinates. `interiorWaypoints(N)` selects only
  interior positions. An explicitly supplied auxiliary map remains active with either mask.

Auxiliary mappings may transform even fixed physical values; they are a separate advanced mechanism.
To optimize durations with every waypoint fixed, use `options.mask.waypoints.assign(N + 1, 0)`.

`OptimizerOptions` defaults are:

| Field | Default | Meaning |
| --- | --- | --- |
| `energy_weight` | `0.0` | Nonnegative weight on the spline's minimum-derivative energy |
| `duration_weight` | `0.0` | Nonnegative weight on the sum of all physical durations |
| `integration_steps` | `64` | Positive number of trapezoid intervals per segment |
| `record_samples` | `false` | Keep current sample records even without a sample cost |
| `mask` | Empty masks | Optimize durations and interior waypoints |

Both weights must be finite. `NoObjective` means **no custom costs**, not energy minimization by
itself. Built-in costs and custom costs add together. A duration weight is a soft linear cost,
not a bound or a prescribed total duration.

## Prepare, solve and deliver

```cpp
using Spline = SplineTrajectory::QuinticSplineND<3>;
SplineTrajectory::SplineOptimizer<Spline> optimizer;
SplineTrajectory::OptimizerOptions options;
options.energy_weight = 1.0;
options.duration_weight = 0.5;
SplineTrajectory::NoObjective objective;
// input is a filled SplineInput<3>.
if (const auto status = optimizer.prepare(input, options, objective); !status) return status;
auto x = optimizer.initialGuess();
Eigen::VectorXd gradient(x.size());
const auto result = optimizer.evaluate(x, gradient, objective);
if (!result) return result.status;
```

`prepare(input, objective)` uses default options. Both overloads inspect the objective type
without retaining it. Supply `NoObjective{}` explicitly when no custom stages are needed.
A sample stage requires preparation with a sample-capable objective; otherwise evaluation returns
`SamplingNotPrepared`. Options, map dimensions and sample capability remain fixed until reprepare.

Preparation copies the input, builds an initial candidate, encodes the initial decision vector
once, and sizes **library-owned** buffers. `initialGuess()` returns an owning copy and requires
successful preparation. It is unchanged by rejected trials. Prepare user-owned cost resources
separately before the solver loop; see [convex hulls](convex_hull.md). There are no automatic
cost initialization hooks. An empty output gradient may allocate on first evaluation; pre-size
it before entering a loop that must be allocation-free.

The external solver repeatedly calls `evaluate(x, gradient, objective)`. After it returns:

1. Inspect the solver's termination status and call `update(x_final)` with its final returned vector.
2. Check update success, then validate the planning task's constraints and accept/reject the candidate.
3. Save `auto trajectory = optimizer.copyPolynomial()`.
4. Keep that independent snapshot alive and unchanged while the control thread uses its sampler.

`evaluate()` builds trial candidates, including trials the solver may reject. `polynomial()`
borrows the **last built candidate**, not an automatically accepted result. `update()` rebuilds
without evaluating costs. Neither solver convergence nor a soft penalty establishes feasibility.

`problem()`, `layout()` and `activeMask()` borrow the prepared snapshot. `parameters()` borrows
current physical values; after a failed decode these may be partially updated. Preserve x when
logging rejected trials. `energy()` returns unweighted energy as `StatusOr<double>` and rejects
an invalid candidate or nonfinite energy. `recordedSamples()` is empty until a successful recording
evaluation, and after update or failure. Evaluation/state changes invalidate candidate views;
preparation, movement and destruction also invalidate prepared-data views.

The optimizer is noncopyable. Moving transfers the solve and leaves the source unprepared.
Use one optimizer and mutable cost workspace per simultaneous solve.

## Choose a cost entry

| Planning requirement | Entry |
| --- | --- |
| Smoothness and total duration | `energy_weight`, `duration_weight` |
| Pointwise obstacles, speed, acceleration, jerk or snap penalties | `integral` |
| Bezier / MINVO control constraints | `coefficient` |
| Waypoint, boundary or extra duration costs | `parameter` |
| Coupled costs across multiple sampled positions | `sample` |
| Special regularization in decision coordinates | `decision` |

An objective is a user-defined holder of **callable data members**. Each cost implements
`operator()`; the member name selects its stage. The cost type's name is arbitrary.
Direct methods such as `Objective::integral(...)` are not supported. Example:

```cpp
struct TargetDurationCost {
    double target = 5.0;
    double weight = 0.2;
    double operator()(const SplineTrajectory::ParameterView<Spline>& state,
                      SplineTrajectory::ParameterGradientView<Spline>& gradient) const {
        const double total = std::accumulate(state.durations.begin(), state.durations.end(), 0.0);
        const double error = total - target;
        gradient.durations.array() += 2.0 * weight * error;
        return weight * error * error;
    }
};
struct Objective {
    TargetDurationCost parameter;
};
```

This adds to any configured linear duration cost. Set `duration_weight = 0` if only the target
cost is wanted. Components may be values, references or `std::ref/std::cref`; borrowed objects
must outlive their use. Evaluation requires an lvalue objective; const objects work if their
components are callable through const access. `NoObjective` is the explicit empty objective.
An invalid stage signature is a compile-time error. Extra application fields are allowed.

Below, S is the spline, D its dimension, and Opt the optimizer type. Every cost returns a
`double` for unconditional success or `EvaluationResult{cost, status}` when rejection is possible.
The cost is added once. **Gradients accumulate**: use `+=`, do not resize or clear prior contributions.
Multiple terms in one stage can be composed by calling them inside one functor.

| Member | `operator()` arguments | Gradient meaning |
| --- | --- | --- |
| `integral` | `const IntegralPointInfo&, const IntegralState<D>&, IntegralGradient<D>&` | Unweighted local integrand partials |
| `sample` | `const Opt::SampleBuffer&, SampleGradientView<S>&` | Whole-batch position and explicit trajectory-time partials |
| `coefficient` | `const ParameterView<S>&, CoefficientGradientView<S>&` | Coefficient and independent duration/start-time partials |
| `parameter` | `const ParameterView<S>&, ParameterGradientView<S>&` | Physical parameter partials after the spline adjoint |
| `decision` | `const DecisionView&, Eigen::Ref<Eigen::VectorXd>` | Direct partials in decision coordinates |

### Integral costs

`IntegralState<D>` provides `p/v/a/j/s`. `IntegralGradient<D>` provides the corresponding partials
and `trajectory_time`, the **explicit time partial with state held fixed**. Each point's gradient
starts at zero. Return the integrand without quadrature weights; the optimizer handles trapezoid
weights, interval lengths, state drift and dependencies on preceding durations.

The integral functor may declare `static constexpr int kDerivativeOrder` in [0, 4], default 4.
Position is 0, velocity 1, acceleration 2, jerk 3 and snap 4. Write no gradients above the declared
order. The optimizer computes one further derivative for time drift. There is no
`beginEvaluation()` hook or separate configuration on Objective.

`IntegralPointInfo` provides `segment_index/count`, `step_index/count`, `alpha`, `segment_duration`,
`step_size`, `segment_time` (seconds from the segment start), and `trajectory_time` (on the caller's
axis). Endpoint helpers are `isSegmentStart/End()` and `isTrajectoryStart/End()`.
`waypointIndex()` returns 0…N at a waypoint, or -1 between waypoints.

### Discrete sample costs

The sample stage sees N × (integration_steps + 1) records in segment order. Each record contains
`point`, `trap_weight`, `b_p`, `p` and `v`. Interior knots appear **twice**, once per adjacent segment;
both records have the same waypoint index. This is the per-segment quadrature layout, not a unique
list of waypoint times. A discrete cost can skip `point.isSegmentStart() && !point.isTrajectoryStart()`
when it wants one record per shared knot, leaving skipped gradient rows zero.

`SampleGradientView` has `positions` (sample_count × DIM, row-major except DIM=1) and
`trajectory_times` (sample_count). It provides no velocity-gradient field. Return the complete
batch cost; the optimizer applies **no quadrature weighting** to it. It propagates position drift
through velocity and explicit sample-time partials through segment durations and start time.
Recording alone adds no cost or physical adjoint.

### Coefficient, parameter and decision costs

`ParameterView` contains read-only `polynomial`, `durations`, `waypoints`, `boundary` and
`start_time`. `CoefficientGradientView` provides `coefficients`, `durations` (with local power
coefficients held fixed) and `start_time`. Coefficient, integral and sample partials join before
one spline adjoint. The analytic energy gradient is added directly to physical gradients.

`ParameterGradientView` borrows the owning `Spline::Gradients` fields with the same access paths
as the state: `waypoints`, `durations`, `boundary.start/end.v/a/j`, plus `start_time`. There is no
second gradient store or nested `.spline`. `addWaypoint(i, partial)` adds a row/column expression
to waypoint i, including endpoint positions. Boundary gradients contain derivatives only.
`start_time` is an explicit physical partial; auxiliary maps propagate it when they move the origin.

After auxiliary/time/spatial pullbacks, `decision` receives `DecisionView{variables, layout}` and
adds to the already accumulated decision gradient. All callback views expire when the call returns.

## Time, spatial and auxiliary maps

A parameterization has `time`, `space`, and `auxiliary` members. Use `NoAuxiliaryMap` for an unused
auxiliary role. Construct `SplineOptimizer<Spline, MyParameterization> optimizer(maps)` or use
`makeOptimizer<Spline>(maps)`. Default roles are `QuadInvTimeMap`, `IdentitySpatialMap<D>` and
`NoAuxiliaryMap`; `IdentityTimeMap` is also provided. Maps are owned by value unless explicitly
borrowed with `std::ref/std::cref` (either individual roles or the whole parameterization).
Owned maps are queried through const access. Keep borrowed objects alive and dimensions/inverse
semantics unchanged during a prepared solve.

Time map:

```cpp
double forward(double tau) const;
double inverse(double duration) const;
double backward(double tau, double duration, double grad_duration) const;
```

`inverse()` encodes the initial guess during preparation. `backward()` returns the scalar chain
rule. Its duration is the result of `forward()` **before any auxiliary transformation**, cached
rather than recomputed. Fallible operations may return `StatusOr<double>` instead. The optimizer
rejects nonpositive/nonfinite forward durations and propagates map rejection.

Spatial map (Vector is S::Vector):

```cpp
int dimension(int waypoint_index) const;
Vector forward(const Eigen::Ref<const Eigen::VectorXd>& x, int waypoint_index) const;
Eigen::VectorXd inverse(const Vector& point, int waypoint_index) const;
void backward(const Eigen::Ref<const Eigen::VectorXd>& x, const Vector& grad_point,
              int waypoint_index, Eigen::Ref<Eigen::VectorXd> output) const;
```

Indices run from 0 to N; each mapped dimension is positive. `inverse()` may return a fixed-size
vector and may allocate during preparation. `backward()` overwrites its pre-sized decision slice.
Fallible value operations may return `StatusOr<VectorType>`; backward may return `Status`.
Successful maps guarantee finite, correctly shaped physical outputs.

Auxiliary map:

```cpp
int dimension() const;
StatusOr<Eigen::VectorXd> initial(const SplineInput<D>& reference) const;
Status apply(const Eigen::Ref<const Eigen::VectorXd>& z, MutableParameters<S>& state) const;
Status backward(const Eigen::Ref<const Eigen::VectorXd>& z, const ParameterView<S>& state,
                ParameterGradientView<S>& gradient, Eigen::Ref<Eigen::VectorXd> grad_z) const;
```

`apply()` runs after ordinary maps and boundary decoding, before constructing the spline. It may
transform waypoints, durations, derivatives and start time, but must preserve topology and valid
physical inputs. Each candidate first restores reference inputs when an auxiliary map is present.
`backward()` adds to initially zero `grad_z` and transforms physical gradients back to pre-auxiliary
coordinates. It returns **Status only**, not a cost. Put a switching-progress or other physical
cost in `parameter`, or a coordinate cost in `decision`. Normal map pullbacks run afterwards.
See the [parameterized example](../examples/parameterized_optimizer.cpp) for the full coupled chain.

Decision order is active durations, active waypoints, start derivatives, end derivatives, then
auxiliary coordinates. `layout().durations/waypoints` hold source indices and offsets; waypoints
also carry mapped `dof`. `boundary_offset`, `auxiliary_offset` and `dimension` locate other blocks.

## Solver boundary, failure and allocation

The library does not run an external solver. A capturing lambda or C callback invokes
`evaluate(x, gradient, objective)`, handles/records `result.status`, and returns `result.cost`.
The decision vector must match the prepared dimension. Owning gradient outputs resize as needed.
`Status` has an ErrorCode, static message and optional index; it does not allocate strings.

Failed evaluation returns infinity and zeros the gradient. Decode/construction failure invalidates
the candidate but **retains storage for fixed-topology retry**; preparation and the initial guess
remain valid. A later successful evaluation overwrites the candidate. Failed preparation leaves
an unprepared optimizer. A rejected cost can leave a numerically valid polynomial: numerical
validity is separate from objective success and planning acceptance. Do not consume failure outputs.

Prepared fixed-topology library evaluation, including rejection and retry, reuses buffers when
the gradient already matches. User maps/costs own their allocations and exceptions. Unexpected
exceptions and allocation failures are not converted into ordinary status. The library checks
reference input on prepare, decisions on evaluation, decoded durations on construction, and final
cost/gradients after accumulation; it does not repeatedly scan intermediate matrices.

`evaluate(x, g, objective, OpenMPExecutor{})` parallelizes segment integration when compiled with
OpenMP, otherwise runs serially. Integral costs and everything they access must allow concurrent
calls; exceptions must not escape worker callbacks. Other stages run outside the segment loop.
Parallel failure reporting chooses the first failed segment in segment order.

`checkGradients(optimizer, x, objective, step, tolerance)` allocates diagnostic vectors, compares
central differences and restores the base candidate even after a rejected perturbation. It returns
analytic/numeric gradients, norm/relative/max errors, offending index and status. Printing belongs
to the caller. Previous names and contracts are covered in the [migration guide](api_migration.md).
