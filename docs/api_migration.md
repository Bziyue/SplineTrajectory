# API migration

This guide covers the current public API, including the October 2026 consolidation. Older
names in the left column are migration references, not supported compatibility aliases.
The cubic/quintic/septic implementations, numerical formulas and partial energy computations remain.

## Physical input, gradients and outputs

| Previous API / assumption | Current API / behavior |
| --- | --- |
| Parameterized spline constructors / `Spline::create(...)` | Default construct, then `update(...)`; check Status |
| `update(knots, points, boundary)` | `updateFromBreakpoints(knots, waypoints, boundary)` |
| Separate optimizer-only `SplineProblem<Spline>` / `ProblemDefinition` | `SplineInput<DIM>` shared by spline and optimizer; `Optimizer::Problem` aliases it |
| `BoundaryConditions::start_velocity/acceleration/jerk` and end equivalents | `boundary.start.v/a/j`, `boundary.end.v/a/j`; positions only in waypoints |
| `Gradients::inner_points`, `start.p`, `end.p` | One `waypoints` matrix, (N+1) × DIM, including endpoints |
| `BoundaryGradient`, `BoundaryGradients`, earlier `BoundaryStateGrads/BoundaryDualGrads` | Removed; gradient `boundary` uses `BoundaryConditions<DIM>` with derivatives only |
| `energyInnerWaypointGradient()` or older interior-only `energyWaypointGradient()` | `energyWaypointGradient()` returns ALL N+1 waypoints |
| `energyBoundaryGradient()` including positions | Returns `BoundaryConditions<DIM>`, derivatives only |
| Waypoints declared with `CoefficientMatrix` | Prefer `WaypointMatrix`; same underlying Eigen type |
| `ParameterGradient`, nested `gradient.spline` | `ParameterGradientView`: flat waypoints/durations/boundary/start_time references |
| `CoefficientGradient` | `CoefficientGradientView`: coefficients/durations/start_time references |
| `evaluateInto`, `evaluateDerivativesInto`, `backwardInto` | Use the corresponding unsuffixed output overload |
| Owning outputs must be pre-sized | Owning overwrite resizes as needed; matching storage is reused |
| Preallocated Eigen view outputs | Where supported, unsuffixed overload overwrites without resizing |
| `*Add` outputs | Still accumulate into matching storage, never clear/resize |

`Spline::Gradients` is the owning physical gradient store; callback views borrow fields.
`addWaypoint(i, row_or_column_expression)` uses the full input row index 0…N.
`setZero(N)` prepares an accumulator; `setZero()` clears existing sizes; `size()` returns N.

`energyDurationPartials()` holds local coefficients fixed. `energyDurationGradient()` is the
total derivative through interpolation. Keep separate coefficient/time partials for a general
`spline.backward(gC, gT)`; no bundled input wrapper is required.

## Optimizer, objective and maps

| Previous API / assumption | Current API / behavior |
| --- | --- |
| `problem.mask`, `mask.time`, `layout.time` | `options.mask`, `mask.durations`, `layout().durations` |
| `OptimizationMask::allFixed(N)` | `none(N)` disables direct coordinates; auxiliary maps remain active |
| `prepare(problem)` / `prepare(problem, options)` | Pass an objective: `prepare(input, objective)` or `prepare(input, options, objective)` |
| Implicit minimum-energy objective | Both built-in weights default to zero; set `energy_weight` explicitly |
| `objective.duration` | `options.duration_weight` for linear total time; `objective.parameter` for other duration costs |
| `Objective::integral(...)` or other stage methods | Objective holds callable data members; each cost struct implements `operator()` |
| `Objective::kIntegralDerivativeOrder` | Optional `kDerivativeOrder` on the integral functor, default 4 |
| `beginIntegralEvaluation()` / `beginEvaluation()` | Removed; initialize caller-owned resources explicitly before solving |
| Callbacks always return EvaluationResult | Return `double` for unconditional success, or EvaluationResult for rejection |
| `SampleState`, `SampleGradient` for integral callbacks | `IntegralState`, `IntegralGradient` |
| `IntegralPointInfo.global_time/local_time`, integral gradient `.time` | `trajectory_time/segment_time`, gradient `.trajectory_time` |
| `interiorBoundaryIndex()` with shifted indexing | `waypointIndex()` returns 0…N at any waypoint, otherwise -1 |
| Sample callback with separate gradient matrices/vectors | `(const SampleBuffer&, SampleGradientView<Spline>&)` |
| Sample positions stored DIM × count | `gradient.positions` is count × DIM with contiguous rows; `trajectory_times` is count |
| `toTime/toTau`, `toPhysical/toUnconstrained` | `forward/inverse` on time and space maps |
| Spatial `backwardInto` | `backward(..., Eigen::Ref<VectorXd>)`; overwrite the preallocated decision slice |
| Auxiliary backward returns cost and status | Returns Status only; move costs to parameter/decision stages |
| `updateAccepted(x)` / `synchronizeWorkingState(...)` | `update(x)` with the solver's final returned vector |
| Optimizer `dimension()` | `decisionDimension()` |
| `costAndGradient()` / `lastStatus()` | `evaluate()` returns EvaluationResult directly; inspect its status |

Default masks optimize durations and interior waypoints. Input waypoints are initial guesses for
active coordinates; they are not all fixed interpolation constraints throughout optimization.
Use explicit masks to fix them. Ordinary spline construction still interpolates every input point.

Five custom stages remain: integral, sample, coefficient, parameter and decision. Gradients are
additive. Integral costs return unweighted integrands; sample costs return the whole batch cost.
Sample records still duplicate interior knots per adjacent segment. Do not silently deduplicate
them while retaining a differently indexed gradient matrix.

Maps may be owned or borrowed with explicit `std::ref/std::cref`. Total time/spatial operations
return values/void; fallible ones may return StatusOr/Status. Auxiliary apply/backward return
Status and preserve topology. Its backward only propagates gradients, including a movable start
time; the time-map backward receives the cached duration before auxiliary transforms.

For older context APIs, replace `SplineOptimizer<DIM, Spline, ...>` and `OptimizationContext`
with `SplineOptimizer<Spline, Parameterization>`. Replace `makeEvaluateSpec().with...Cost(...)`
with a user-defined objective holding functors. `generateInitialGuess(context)` becomes
`initialGuess()`, `evaluate(context,...)` becomes `evaluate(...)`, and `checkGradients` is a free
function. Mutable working-spline access is replaced by read-only `polynomial()` / `parameters()`.

## Trajectory lifetime and time axis

| Previous API / assumption | Current API / behavior |
| --- | --- |
| `PPolyND<DIM, ORDER>` interpreted as degree | Second argument is maximum coefficient count |
| `ORDER`, `COEFF_NUM` | `kDegree`, `kCoefficientCount`; also kDimension/kMinDerivativeOrder |
| `VectorType/RowVectorType/MatrixType/TrajectoryType` | `Vector/RowVector/CoefficientMatrix/Polynomial` |
| `isInitialized()`, `get...()` accessors | `isValid()` and direct names such as duration(), coefficients(), waypoints() |
| `getTrajectory/getPPoly`, corresponding Copy forms | `polynomial()` borrows, `copyPolynomial()` owns |
| `getEnergy`, `getEnergyPartialGradByCoeffs/Times`, `getEnergyGrad`, `propagateGrad` | `energy`, `energyCoefficientPartials/energyDurationPartials`, `energyGradient`, `backward` |
| `at(i)` | `operator[](i)`; caller validates index |
| Segment `evaluate()` | `evaluateLocal()`; local seconds |
| Query hint pointers | One `makeSampler<R>()` per traversal |
| Separate int / Deriv overload families | `DerivativeOrder` accepts both, defaults to position |
| `extractInterval()` resets origin | Preserves source time axis; call `rebased(0.0)` explicitly to shift |
| `appendRebased()` requires zero-origin inputs | Accepts arbitrary origins; shifts suffix to receiver's end |

Borrowing from temporary owners is deleted. Updates, assignment, moves and destruction invalidate
borrowed segments/samplers/views. A polynomial snapshot is independent. The recommended delivery
flow is final solver x → successful update(x) → task constraint acceptance → copyPolynomial() →
control-thread sampling. The final callback trial is not necessarily the accepted solver result.

## Convex hulls and allocation

| Previous API | Current API |
| --- | --- |
| `ConvexHullRepresentation` | `ConvexHullWorkspace<DIM>` |
| `configure` / `resetTopology` | `prepare(poly, HullOptions)` or `prepare(segments, coefficient_count, options)` |
| `isConfigured()` | `isPrepared()`; isValid() additionally requires numerical update |
| `kernel()` / public conversion-cache type | Internal detail; use controls, piece views, metadata and adjoints |
| Repeated manual piece offsets | `hull.piece(i).info()` / `.controls()`; older named accessors remain |
| `backwardInto(gControls, gC, gT)` | `backward(gControls, gC, gT)`; owning outputs resize/reuse |
| Repeated on-demand configuration in a cost | Prepare hull and gradients before the solver; call strict update(poly) in the loop |

`update(poly, options)` is a convenience form that prepares only on a mismatch. The one-argument
`update(poly)` requires matching prepared topology and does no allocation or cache lookup.
`toBezier`, `toMINVO` and `fromPPoly(poly, options)` return owning updated workspaces.
`backward` returns or overwrites independent coefficient/duration partials; `backwardAdd` adds
to matching outputs or a coefficient gradient view. The cache remains shared internally.

Failed decode or numerical construction invalidates the current candidate while preserving
workspace for retry. Fixed-topology library evaluation/recovery with matching output buffers
does not allocate. User cost buffers remain the caller's preparation responsibility. Status and
EvaluationResult report expected failures; unexpected exceptions and allocation failure propagate.
Low-level numerical APIs retain caller-validated input and representable-arithmetic preconditions.
