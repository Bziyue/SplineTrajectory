# API migration from d504135

Use this guide when updating code written against `d504135` or earlier.

## Polynomial and spline names

| Previous API | Current API |
| --- | --- |
| `PPolyND<DIM, ORDER>` | `PPolyND<DIM, MaxCoefficients>`; capacity, not degree |
| `Spline::ORDER`, `COEFF_NUM` | `kDegree`, `kCoefficientCount`; also `kDimension`, `kMinDerivativeOrder` |
| `VectorType`, `RowVectorType`, `MatrixType`, `TrajectoryType` | `Vector`, `RowVector`, `CoefficientMatrix`, `Polynomial` |
| `isInitialized()` | `isValid()` |
| `getDimension/getDegree/getNumCoeffs/getNumSegments()` | `dimension/degree/coefficientCount/numSegments()` |
| `getStartTime/getEndTime/getDuration()` | `startTime/endTime/duration()` |
| `getBreakpoints/getCoefficients()` | `breakpoints/coefficients()` |
| `getNumPoints/getSpacePoints/getTimeSegments/getBoundaryConditions()` | `numPoints/waypoints/durations/boundary()` |
| `getCumulativeTimes()` | `breakpoints()` |
| `getTrajectory/getPPoly()` | `polynomial()`; borrowed const lvalue only |
| `getTrajectoryCopy/getPPolyCopy()` | `copyPolynomial()`; independent owning copy |
| `getTrajectoryLength()` | `length()` |
| `Segment::evaluate()` / `getCoeffs()` | `evaluateLocal()` / `coefficients()` |
| `evaluate(t, int* hint, ...)` | Own one `makeSampler<N>()` per traversal |
| `getEnergy()` | `energy()` |
| `getEnergyPartialGradByCoeffs/Times()` | `energyCoefficientPartials/energyDurationPartials()` |
| `getEnergyGradTimes/InnerPoints/Boundary()` | `energyDurationGradient/energyWaypointGradient/energyBoundaryGradient()` |
| `getEnergyGrad()` | `energyGradient()` |
| `propagateGrad()` | `backward()`; owning return or overwritten output overload |
| `BoundaryStateGrads`, `BoundaryDualGrads` | `BoundaryGradient`, `BoundaryGradients` |
| `Gradients::times` | `Gradients::durations` |

There is one coefficient/knots owner: PPoly. Spline construction and adjoints access it through
private controlled access. Copying a spline or polynomial still deep-copies its numerical data.
Moved-from polynomials are invalid. Updates invalidate all samplers, segments and iterators.

`energyDurationPartials()` holds local coefficients fixed. `energyDurationGradient()` includes
the interpolation constraints. `backward()` receives coefficient and independent-duration
partials and overwrites physical gradients; `ConvexHullWorkspace::backwardAdd()` accumulates
partials. Distinguish those contracts when combining multiple objectives.

Shared `ErrorCode`, `Status` and `StatusOr<T>` live in `SplineTrajectory.hpp`; no planner types
are required. Replace fallible data constructors with `Type::create(...)`, check the result,
then move its value. Repeated `update()` calls return `Status`. Numerical failure invalidates
the polynomial; state rejection of output-reference operations leaves output unchanged.
A failure during numeric accumulation may leave partial output: discard it entirely.

Value-returning evaluation, energy, gradient, polynomial transform, time-sequence and hull
factory APIs now return `StatusOr<T>`. Output-reference variants return `Status`; no extra
result allocation is required. Low-level inputs are caller-validated: finite values, legal orders
and indices, matching shapes, and valid time/topology constraints are preconditions. No input
validation or checked/unchecked pair is provided. Callers guarantee representable arithmetic;
ordinary computed results are not scanned for overflow. Invalid state, unrepresentable time
knots, factorization failure and resource limits still return errors. Derivatives above degree remain zero; outside times extrapolate.
`at()` retains `StatusOr<Segment>` but, like `operator[]`, requires an already-valid index.

## Optimizer replacement

| Previous API | Current API |
| --- | --- |
| `SplineOptimizer<DIM, Spline, TimeMap, SpatialMap, AuxMap>` | `SplineOptimizer<Spline, Parameterization>` |
| `OptimizationContext` | Private storage owned by one optimizer |
| `ProblemDefinition` | `SplineProblem<Spline>` or `Optimizer::Problem` |
| `problem.time_segments`, `.bc` | `.durations`, `.boundary` |
| `OptimizerConfig` | `OptimizerOptions` |
| `integral_num_steps`, `rho_energy` | `integration_steps`, `energy_weight` |
| `prepareContext(problem, context, config)` | `prepare(problem, options)` |
| `makeEvaluateSpec(...).with...Cost(...)` | Ordinary named objective struct |
| `VoidTimeCost/VoidIntegralCost/...` | Omit that member; `NoObjective` for an empty objective |
| `generateInitialGuess(context)` | `initialGuess()` |
| `evaluate(context, x, g, spec)` | `evaluate(x, g, objective)` |
| `evaluatePrepared(...)` | `evaluate(...).cost` after checking the returned result |
| `synchronizeWorkingState(context, x)` | `updateAccepted(x)` |
| `getWorkingSpline(context)` | `polynomial()`; mutable solver access is removed |
| Member `checkGradients(...)` | Free `checkGradients(optimizer, x, objective, ...)` |
| `setRecordIntegralSamples(...)` | `options.record_samples`, then prepare |
| `getRecordedIntegralSamples(context)` | `recordedSamples()` |
| `SpatialMap::getUnconstrainedDim` | `dimension(global_index)` |
| Owning spatial backward return/fallback | `backwardInto(..., Eigen::Ref<VectorXd>)` |

Create an ordinary parameterization with `time`, `space` and `auxiliary` roles. Default maps
are owned, not borrowed from a second object's members. An advanced parameterization can be
passed by value (including non-default-constructible maps) or explicitly through `std::cref`.
The optimizer cannot be copied; moving transfers its solve. Keep externally borrowed objects
alive and do not change dimensions during a prepared solve.

Declare only the required stages: `duration`, `integral`, `sample`, `coefficient`, `parameter`,
`decision`. Member types stay concrete and can be borrowed with references or `std::ref`.
Old waypoint and trajectory costs become a `parameter` functor; add their contributions directly
inside it. Direct mapped-domain penalties belong in `decision`, which accumulates into an
already written decision gradient. No legacy long-argument integral dispatch remains: use
`SampleState<DIM>` and `SampleGradient<DIM>` with `IntegralPointInfo`.

The coefficient stage now receives a read-only `ParameterView<Spline>` and writable
`CoefficientGradient<Spline>`, rather than a mutable spline solver and several unrelated outputs.
The auxiliary protocol also receives these physical views; it must propagate gradients back
through its transforms **before** ordinary time/spatial maps. The time map's duration argument
is the pre-auxiliary `toTime(tau)` value. See [complete signatures](optimizer.md).

Pre-size `g` before entering the external solver callback. Sample objectives require
`prepare(problem, options, objective)` to determine recording capability without borrowing the
objective. Preparation is the allocation boundary. After the solver returns, check
`updateAccepted(x)` and then consume `polynomial()` or `copyPolynomial()`.

## Hull replacement

| Previous API | Current API |
| --- | --- |
| `ConvexHullRepresentation<DIM>` alias pair | `ConvexHullWorkspace<DIM>` |
| Nested `Kernel` / `ConvexHullKernel<DIM>` | Dimension-independent `ConvexHullKernel` |
| `resetTopology(poly, basis, derivative, depth, budget)` | `configure(poly, HullOptions{basis, derivative, depth}, budget)` |
| Shape-only topology configuration | `configure(segments, coefficients, HullOptions, budget)` |
| Manual field-by-field topology matching | `matches(poly, options)` |
| One initialization flag | `isConfigured()` and `isValid()` |
| Const `backward()/backwardAdd()` | Non-const operations using exclusive workspace scratch |
| `fromPPoly(poly, basis, derivative, depth)` | `fromPPoly(poly, HullOptions{basis, derivative, depth}, budget)`; budget is optional |
| Resizing `backward(partials, coefficients, durations)` | Pre-size both outputs, then call `backwardInto(partials, coefficients, durations)` |

`toBezier()` and `toMINVO()` retain their convenience arguments. The one-argument `backward()`
still returns owned gradients; `backwardInto()` overwrites pre-sized outputs and `backwardAdd()`
accumulates into them. Both require caller-validated matching shapes and never resize. `update()`
requires a valid source matching the configured topology and a representable numerical range.
Controls and accumulated gradients are not checked for overflow. The immutable kernel cache is consulted only during configuration; it is shared
across spatial dimensions. A complete tuple key avoids packed-field collisions.

## Unified failure contracts

`OptimizationError/OptimizationStatus` are replaced by the shared `ErrorCode/Status`.
Optimizer `evaluate()` keeps `EvaluationResult { cost, status }`; do not wrap it in another
`StatusOr`. Every cost callback and auxiliary backward callback returns `EvaluationResult`.
Return `{cost, Status::success()}` on success, or `{infinity, status}` on rejection.
Value-producing map operations return `StatusOr`, while map operations writing outputs return
`Status`. Propagation preserves the original code/message/index. Update custom maps and
callbacks together with their callers; no compatibility dispatch is retained.

`costAndGradient()` and `lastStatus()` are removed. External scalar solver callbacks consume
`evaluate()` directly and retain its status when needed; the solver still owns its termination
codes. `initialGuess()` remains a normal owning return: successful `prepare()` is a precondition,
checked by assertion. Failed trial evaluation does not invalidate that prepared initial vector.
Allocation failure and unexpected user exceptions are not translated into expected-failure statuses.
