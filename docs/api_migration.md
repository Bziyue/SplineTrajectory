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

Negative derivative orders throw `invalid_argument` in scalar, segment, derivative-polynomial
and batch entry points. Nonfinite numeric queries return NaNs; derivatives above degree are
zero. Invalid construction clears the polynomial. Invalid energy queries return NaN;
differentiating an invalid spline throws. `generateTimeSequence()` validates finite inputs,
positive step, forward interval and representable count. Checked indexing uses `out_of_range`.

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
| `evaluatePrepared(...)` | `costAndGradient(...)` |
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
accumulates into them. Both validate all shapes before writing and never resize. `update()` with
invalid source/topology invalidates the numerical state and throws. Numeric overflow is also
rejected. The immutable kernel cache is consulted only during configuration; it is shared
across spatial dimensions. A complete tuple key avoids packed-field collisions.
