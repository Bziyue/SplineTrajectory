# API migration from d504135

This guide covers the second refactor in SplineTrajectory. The three downstream repositories
have **not** been migrated or validated by this change. They must eventually copy the three
public headers byte-for-byte, including English Doxygen, licenses and MINVO constants.

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
is the pre-auxiliary `toTime(tau)` value. See [complete signatures](../include/SplineOptimizerProtocols.md).

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

`toBezier()`, `toMINVO()` and `fromPPoly()` retain their convenience arguments. `update()` with
invalid source/topology invalidates the numerical state and throws. Numeric overflow is also
rejected. The immutable kernel cache is consulted only during configuration; it is shared
across spatial dimensions. A complete tuple key avoids packed-field collisions.

## Downstream checklist for the follow-up migration

Paths below are relative to the parent `optimizer/` workspace. This is a migration inventory,
not a claim that these consumers already compile with the new API.

| Consumer | Relevant call sites | Required adaptation |
| --- | --- | --- |
| GCOPTER | `GCOPTER/gcopter/include/gcopter/spline_sfc_optimizer.hpp`, `include/TrajectoryOptAdapters/GcopterPenaltyIntegralCostAdapter.hpp`, project gradient-check code | Quintic parameterization; polytope map `backwardInto`; named duration/integral objective; structured gradient-check callback; accepted solution |
| SUPER | `SUPER/super_planner/include/traj_opt/spline_components.hpp`, `src/traj_opt/exp_traj_optimizer_s4.cpp`, `src/traj_opt/backup_traj_optimizer_s4.cpp`, `include/TrajectoryOptAdapters/SuperBackupAuxiliaryStateMap.hpp` | Septic objective and owned optimizer; boundary/mask layout; auxiliary views; additive decision-domain penalties |
| diff-planner-swarm | `diff-planner-swarm/src/planning/backend/optimizer/trajectory_optimizer.cpp`, `include/DiffSwarm/planning/backend/optimizer/trajectory_auxiliary_map.hpp`, `terms/bezier_corridor_cost.hpp`, `mapping/positive_duration_map.hpp` | Named objective instead of generated spec type; combined parameter costs; moving-start/shared-time adjoint; hull workspace preparation; recording capability; accepted vector |
| All three | Vendored spline headers and polynomial/sampling users | Copy the three headers unchanged, adapt renamed queries, remove obsolete adapters, update recorded upstream version/hash and third-party notes |

Downstream C-style solver user data should hold pointers/references to the named optimizer and
objective. Keep `void*` at that boundary; costs inside quadrature remain statically dispatched.
Application trajectory identity, coordinate frames, splicing and acceptance policy remain
project responsibilities. Project-specific Chinese comments in diff-planner-swarm do not apply
to the unmodified upstream public headers.
