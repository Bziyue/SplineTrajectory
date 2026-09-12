# SplineOptimizer protocols

This is the current API contract. The library requires C++17 and Eigen 3.3.7 or newer.
All examples use generic optimization terminology. Functors and parameterizations are ordinary
user types: no inheritance, builder, tuple protocol or type-erased callable is needed.

## Ownership and lifecycle

```cpp
using Spline = SplineTrajectory::SepticSplineND<3>;
using Optimizer = SplineTrajectory::SplineOptimizer<Spline>;
Optimizer optimizer;
SplineTrajectory::SplineProblem<Spline> problem;
SplineTrajectory::OptimizerOptions options;
// Fill problem.durations, problem.waypoints and optional problem.boundary/start_time/mask.
auto status = optimizer.prepare(problem, options);
auto x = optimizer.initialGuess();
Eigen::VectorXd gradient(x.size());
// objective is a named, caller-owned object.
auto result = optimizer.evaluate(x, gradient, objective);
```

Check `status` before requesting an initial guess and `result` before using a cost. `result.status`
contains `OptimizationError`, an optional offending knot index (`-1` when absent), and a static
message pointer. Status carries no owning string or allocation. `lastStatus()` also exposes the
last result for callers of `costAndGradient()`.

The optimizer privately owns its problem snapshot, prepared quadrature/layout, spline and
buffers. Default maps are owned by value. There are no internal pointers into movable members.
The optimizer is noncopyable; movement transfers the prepared solve and makes the source
unprepared. Preparation and movement invalidate previously borrowed views.

`polynomial()` returns a borrowed read-only polynomial; `copyPolynomial()` returns an allocating,
independent value. Evaluation and `updateAccepted()` replace the polynomial. `problem()`,
`layout()` and `activeMask()` borrow prepared data until preparation, movement or destruction.
Evaluation never changes that layout. A call does not retain any reference to the objective.
`energy()` returns the unweighted minimum-derivative energy of the current trajectory without
copying solver storage; an invalid trajectory returns NaN.
`parameters()` borrows the current physical values for diagnostics. After a failed decode they
may be only partially updated; use `lastStatus()` and preserve the original decision vector
when recording a rejected candidate. The view expires on the next evaluation or state change.

One optimizer supports one synchronous evaluation at a time. Independent solves need independent
optimizers and any mutable cost workspaces. An external solver's final accepted vector may differ
from its last trial evaluation: call `updateAccepted(x)` successfully before consuming its result.

## Problem, layout and options

`SplineProblem<Spline>` contains:

- `durations`: N positive finite segment durations, in seconds.
- `waypoints`: (N+1) × DIM matrix, one interpolation point per row.
- `boundary`: start/end velocity, acceleration and jerk; only derivatives supported by the
  selected spline participate in construction and validation.
- `start_time`: common finite time origin, in seconds on the caller's time axis.
- `mask`: optional `OptimizationMask`.

Empty time masks activate all durations. Empty waypoint masks activate only interior waypoints.
Explicit waypoint masks have N+1 entries, so endpoints can also be optimized. Nonzero entries
are active. `mask.start` and `mask.end` select `v/a/j`; requesting an unsupported boundary
variable fails preparation.

Decision order is active durations, active waypoints, active start derivatives, active end
derivatives, then auxiliary variables. `layout().time` and `.waypoints` carry global source
indices and offsets; each waypoint also carries its mapped `dof`. `boundary_offset`,
`auxiliary_offset` and `dimension` locate the remaining blocks.

`OptimizerOptions` contains `energy_weight` (finite and nonnegative), `integration_steps`
(positive, per segment), and `record_samples`. Options and capability are fixed at preparation.
Sample costs require `prepare(problem, options, objective)`; this overload inspects the type
without retaining the object. Basic preparation does not depend on any cost type.

## Named objective stages

Declare only the required members. They can be values, reference members, or explicit
`std::reference_wrapper` values. Borrowed objects must outlive the objective that references
them; `std::ref/std::cref` reject temporaries. The objective passed to `evaluate()` must be an
lvalue. Const objectives are supported when their actual stage members satisfy const calls.

```cpp
struct ProblemObjective {
    DurationCost duration;
    IntegralCost integral;
    HullCost coefficient;
    EndpointCost parameter;
};
```

An empty objective is explicitly `SplineTrajectory::NoObjective`; an unrelated empty struct is
rejected. A present member with an invalid signature produces a targeted compile-time error.
C++17 does not reflect other member names: extra application data are allowed, and a misspelled
member alongside another valid stage cannot be diagnosed automatically.

The following signatures use `S` for the spline, `D = S::kDimension`, `Opt = SplineOptimizer<S>`
and the names in namespace `SplineTrajectory`. Outputs are **additive**. Do not resize a gradient,
clear contributions from previous stages or include weights already handled by the optimizer.
Several costs in one stage can be implemented by direct calls inside one named functor.

| Member | Signature returning `double` | Meaning |
| --- | --- | --- |
| `duration` | `(const std::vector<double>&, Eigen::Ref<Eigen::VectorXd>)` | Cost and partials with respect to physical durations |
| `integral` | `(const IntegralPointInfo&, const SampleState<D>&, SampleGradient<D>&)` | Unweighted integrand and local partials |
| `sample` | `(const Opt::SampleBuffer&, Eigen::Ref<Opt::SampleGradMatrix>, Eigen::Ref<Eigen::VectorXd>)` | Cross-sample cost, position and explicit global-time partials |
| `coefficient` | `(const ParameterView<S>&, CoefficientGradient<S>&)` | Polynomial coefficient and independent time partials |
| `parameter` | `(const ParameterView<S>&, ParameterGradient<S>&)` | Total physical parameter partials after the spline adjoint |
| `decision` | `(const DecisionView&, Eigen::Ref<Eigen::VectorXd>)` | Direct decision-domain cost after mapped gradients have been written |

Each view is valid only during its synchronous callback. The returned cost is added once.
Sample-stage output matrices have DIM × sample_count columns, unlike the piece-major
coefficient/waypoint matrices. The corresponding time output has sample_count entries.

### Integral state and time chain

`SampleState<D>` provides `p/v/a/j/s` in physical time. `SampleGradient<D>` provides their
partials and `time`, the **explicit global-time partial at fixed state**. Each point's partials
start at zero. Return an integrand without trapezoid weights. The optimizer handles quadrature
weights, interval length, local-time drift and dependencies on earlier segment durations.

Declare `static constexpr int kDerivativeOrder` from 0 to 4. Position is zero; jerk is three;
snap is four. The default is four. The optimizer computes one additional derivative for time
drift, so jerk costs use snap internally even though they do not differentiate a snap objective.
Writing gradients above the declared order violates the protocol.

`IntegralPointInfo` provides global/local time, normalized `alpha`, segment duration, step size,
segment/step indices and counts. Segment endpoint samples are duplicated between neighboring
segments and have half weights. `isTrajectoryStart/End`, `isSegmentStart/End` and
`interiorBoundaryIndex()` support endpoint-specific policies. An optional `beginEvaluation()`
hook runs once before the segment loop. It must not allocate if the caller requires an entirely
allocation-free evaluation.

For a sample-stage partial `g_t`, the optimizer includes both explicit-time dependence and
position drift through velocity. The sum of explicit-time partials contributes once to the
common start-time gradient. Do not add that same contribution again in an auxiliary map.
Recording alone adds no objective or physical adjoint; it uses a zero integrand.

### Coefficient and physical parameter stages

`ParameterView<S>` exposes `polynomial`, physical `durations`, `waypoints`, `boundary` and
`start_time`, all read-only. Coefficients use piece-major rows in **ascending local physical-time
powers**. `CoefficientGradient<S>` exposes coefficient partials, independent duration partials
with those coefficients held fixed, and a common start-time partial. These join the integral and
sample partials before one general spline adjoint.

The specialized minimum-energy gradient is then added. `ParameterGradient<S>` exposes
`inner_points`, `durations`, `start/end` boundary gradients and `start_time`. Boundary gradients
contain `p/v/a/j`; unused higher derivatives remain zero. `addWaypoint(global_index, partial)`
handles interior points and both endpoint positions, rejecting out-of-range indices.

`decision` sees the original decision vector and immutable layout. Its output already contains
all physical gradients. This is the stage for direct mapped-domain penalties; use `+=`.

## Parameterization

A custom value object combines `time`, `space` and `auxiliary`. Use `NoAuxiliaryMap` if the
third role is unused. Missing roles are a malformed parameterization, not disabled costs.

```cpp
struct ProblemParameterization {
    PositiveDurationMap time;
    BoundedCoordinates space;
    SharedCoordinates auxiliary;
};
auto optimizer = SplineTrajectory::makeOptimizer<Spline>(ProblemParameterization{/* ... */});
// Explicit whole-object borrowing is also supported:
auto borrowed = SplineTrajectory::makeOptimizer<Spline>(std::cref(existing_parameterization));
```

Owned map roles are queried through const access. `std::ref` permits mutable borrowed roles;
`std::cref` retains const access. Borrowed objects must remain alive and address-stable. A parameterization may itself contain `std::ref/std::cref` roles. Do not mutate
map dimensions or inverse semantics during a prepared solve. Non-default-constructible maps
are supported when supplied to the constructor/factory.

### Time map

```cpp
double toTime(double tau) const;
double toTau(double duration) const;
double backward(double tau, double duration, double grad_duration) const;
```

`toTau()` is the inverse used for initial coordinates. `backward()` returns the chain rule for
one unconstrained coordinate. Its `duration` is the output of `toTime(tau)` **before** any
auxiliary transformation. This matters for maps whose derivative uses the physical duration,
such as the exponential map. `QuadInvTimeMap` is the default; `IdentityTimeMap` is also available.
Nonpositive, nonfinite or unrepresentable cumulative times are rejected before the spline kernel.

### Spatial map

```cpp
int dimension(int global_index) const;
Vector toPhysical(const Eigen::Ref<const Eigen::VectorXd>& xi, int global_index) const;
Eigen::VectorXd toUnconstrained(const Vector& point, int global_index) const;
void backwardInto(const Eigen::Ref<const Eigen::VectorXd>& xi, const Vector& grad_point,
                  int global_index, Eigen::Ref<Eigen::VectorXd> destination) const;
```

`Vector = S::Vector`. Global waypoint indices run from 0 to N, including endpoints. Each mapped
dimension must be positive. `toPhysical()` must return DIM values; `toUnconstrained()` returns
the configured mapped dimension and is allowed to allocate in preparation/initial-guess creation.
A fixed-size vector return also satisfies the inverse protocol. `backwardInto()` **overwrites**
its exclusive pre-sized decision slice; the optimizer handles accumulation between cost stages.
There is no owning-vector fallback on the hot path.

### Auxiliary map

```cpp
int dimension() const;
Eigen::VectorXd initial(const SplineProblem<S>& reference) const;
void apply(const Eigen::Ref<const Eigen::VectorXd>& z, MutableParameters<S>& state) const;
double backward(const Eigen::Ref<const Eigen::VectorXd>& z, const ParameterView<S>& state,
                ParameterGradient<S>& gradient, Eigen::Ref<Eigen::VectorXd> grad_z) const;
```

`apply()` runs after ordinary maps and boundary decoding, before spline construction. It may
transform durations, waypoints, boundary derivatives and the common time origin, but must
preserve topology. `backward()` sees their final physical values and all accumulated gradients.
It adds auxiliary gradients to initially zero `grad_z`, transforms physical gradients back to
pre-auxiliary coordinates when necessary, and can return a direct auxiliary cost. Normal time
and spatial pullbacks run afterwards. `initial()` is a cold operation and may allocate.

The [parameterized example](../examples/parameterized_optimizer.cpp) checks a coupled chain:
exponential duration coordinates → shared duration scaling → spline/integral/hull objective,
plus a movable start time. This detects using post-auxiliary durations in the ordinary time map.

## External solvers and concurrency

A capturing lambda may call `costAndGradient(x, g, objective)`. For C-style callbacks, only the
outer solver boundary needs `void*`:

```cpp
struct SolverData { Optimizer& optimizer; ProblemObjective& objective; };
double callback(void* instance, const Eigen::VectorXd& x, Eigen::VectorXd& g) {
    auto& data = *static_cast<SolverData*>(instance);
    return data.optimizer.costAndGradient(x, g, data.objective);
}
```

The solver must supply a correctly sized `g`. A wrong shape returns `DimensionMismatch` rather
than allocating in the callback. The library does not own or run the external solver.

`evaluate(x, g, objective, OpenMPExecutor{})` parallelizes segments if compiled with OpenMP;
otherwise it runs serially. The integral functor and everything it touches must allow concurrent
calls. Constness alone is not proof of thread safety. Exceptions must not escape OpenMP worker callbacks. Coefficient and parameter stages run once
outside the parallel loop. Each mutable hull workspace belongs to one active evaluation.

## Failures, diagnostics and allocation

- Failed preparation leaves an unprepared optimizer and invalid polynomial. Initial-guess access
  while unprepared throws `std::logic_error`.
- Bad input/shape, missing sample capability, numerical construction failure and nonfinite
  objective/gradient values return explicit status; failed evaluation clears its output gradient
  and returns infinity. Failed numeric construction clears the polynomial.
- A failed cost can leave a numerically valid trial polynomial. It is not an accepted solution.
  `lastStatus()` and `polynomial().isValid()` represent different facts.
- User cost exceptions propagate in serial stages; they are not converted to plausible costs. Map rejection by
  `invalid_argument` during candidate decoding becomes a numerical-failure status. Bad allocation
  and unexpected user exceptions are not swallowed.
- `checkGradients(optimizer, x, objective, step, tolerance)` allocates diagnostic vectors, performs
  central differences and restores the base candidate even after a rejected perturbation.
  It returns analytical/numerical gradients, norm/relative/max errors, offending index and status.
  Formatting and printing belong to the caller.
- Prepared normal library evaluation has no heap allocation. User maps/costs are responsible for their own behavior. Failure recovery may
  reallocate spline storage after an invalid candidate has cleared it.

For previous API names, use the [migration guide](api_migration.md).

`prepare()` encodes the initial decision vector once. `initialGuess()` returns an owning copy of
that prepared vector, so nonlinear inverse maps are not solved again. Reprepare after changing
a borrowed parameterization or the physical reference; trial evaluations leave the initial vector intact.
