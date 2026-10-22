# Trajectories, sampling and gradients

`CubicSplineND<DIM>`, `QuinticSplineND<DIM>` and `SepticSplineND<DIM>` interpolate waypoints and
minimize squared acceleration, jerk and snap, respectively. They share the `SplineND<DIM, DEGREE>`
interface.

## Physical input and construction

`SplineInput<DIM>` holds N positive segment durations, N+1 waypoint rows, endpoint derivatives
and a common `start_time`. Rows are points; columns are spatial coordinates. `WaypointMatrix`
and `CoefficientMatrix` share the same row-major Eigen type (column-major for DIM=1).

```cpp
using Spline = SplineTrajectory::QuinticSplineND<3>;
SplineTrajectory::SplineInput<3> input;
input.durations = {1.2, 1.5};
input.waypoints.resize(3, 3);
input.waypoints << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
input.start_time = 2.0;
input.boundary.start.v << 0.2, 0, 0;
Spline spline;
if (const auto status = spline.update(input); !status) return status;
```

`BoundaryConditions<DIM>` has `start` and `end`, each with `v/a/j`. Positions belong only to
`waypoints`. Derivatives default to zero: cubic uses velocity, quintic adds acceleration, septic
adds jerk.

`update(durations, waypoints, start_time, boundary)` accepts the same data separately.
`updateFromBreakpoints(knots, waypoints, boundary)` accepts N+1 strictly increasing times.
These times and `start_time` use the caller's trajectory time axis. All forms return `Status`
and reuse storage when the segment count stays the same. Query the spline after a successful
update. Failed construction clears the visible polynomial and retains storage for retry.

Every waypoint is an interpolation constraint here. In an optimizer, mask-selected input values
are initial guesses that may change; see [physical input and optimization choices](optimizer.md).

## Sampling and lifetime

```cpp
const auto &poly = spline.polynomial();
auto position = poly.evaluate(2.5);
auto state = poly.evaluateDerivatives<2>(2.5); // p, v, a
Spline::WaypointMatrix velocities;
poly.evaluate(std::vector<double>{2.0, 2.5, 3.0}, velocities, SplineTrajectory::Deriv::Vel);

auto trajectory = spline.copyPolynomial(); // Independent owning snapshot.
auto sampler = trajectory.makeSampler<3>();
auto times = trajectory.generateTimeSequence(0.01);
if (!times) return times.status();
for (double time : times.value()) {
    const auto pvaj = sampler.evaluateDerivatives(time);
    // Consume pvaj[0] through pvaj[3].
}
```

`polynomial()` borrows the spline's owner; `copyPolynomial()` copies it. An update, assignment,
move or destruction invalidates borrowed segments, iterators and samplers. A sampler owns its
cursor and cached derivative coefficients but borrows the source. Keep the source alive and
unchanged, and use a separate sampler for each traversal. Sequential queries reuse current-segment
coefficients; backward jumps work too.

Queries use trajectory time. `poly[i].evaluateLocal(t)` uses seconds from segment i's start.
Interior knots select the right segment; the final knot selects the last. Outside queries
extrapolate the endpoint polynomial. Derivatives above the polynomial degree are zero.
`DerivativeOrder` accepts an integer or `Deriv`, defaults to position, and supports equality.

## Outputs and allocation

| Form | Behavior |
| --- | --- |
| `auto result = function(...)` | Returns owning data; dynamic results may allocate |
| `function(..., owning_output)` | Overwrites, resizes as needed and reuses matching storage |
| `function(..., Eigen::Ref/output_block)` where supported | Overwrites a matching preallocated view; no resizing |
| `functionAdd(..., output)` | Accumulates into matching storage; no clearing or resizing |

Batch point matrices have one sample per row. `evaluateDerivatives<R>(times, output)` uses an
array of R+1 matrices, indexed by derivative order. Use output buffers separate from the inputs.

## Energy and adjoints

```cpp
const double energy = spline.energy();
auto gradient = spline.energyGradient();
// gradient.waypoints: (N+1) x DIM, including both endpoint positions.
// gradient.durations: N; gradient.boundary.start/end: v, a, j.
auto waypoint_gradient = spline.energyWaypointGradient();
auto boundary_gradient = spline.energyBoundaryGradient(); // Derivatives only.
```

`Spline::Gradients` / `SplineGradients<DIM>` owns `waypoints`, `durations` and `boundary`.
`energyWaypointGradient()` returns all N+1 position gradients; `energyBoundaryGradient()`
returns `BoundaryConditions<DIM>` containing only derivative gradients. Unsupported derivative
fields are zero. `energyDurationGradient()` computes the total duration gradient through the
interpolation constraints. These separate methods retain their on-demand computation.

`energyCoefficientPartials()` and `energyDurationPartials()` instead treat local coefficients
and durations as independent inputs. For a custom coefficient-space cost, accumulate both
partials and call `spline.backward(coefficient_partials, duration_partials, output)` to obtain
physical gradients. The adjoint reuses the construction factorization.

Overwrite forms resize a default-constructed `Spline::Gradients` automatically. For accumulation,
use `output.setZero(spline.numSegments())`, then `energyGradientAdd(output, weight)`.
`output.setZero()` clears without changing shapes; `size()` reports segment count.
`addWaypoint(i, partial)` accepts a row or column vector/expression and adds to row i, from 0 to N.

## Polynomial storage and time operations

`PPolyND<DIM, MaxCoefficients>` stores coefficients by segment, in ascending powers of local
physical time. Capacity is a coefficient count, not degree. Spline traits are `kDimension`,
`kDegree`, `kCoefficientCount` and `kMinDerivativeOrder`.

`PPolyND::create`, `zero` and `constant` return owning polynomials; `update` returns `void`.
Use finite coefficients, strictly increasing knots and matching dimensions. `operator[](i)`
returns a borrowed segment for an index in `[0, numSegments())`.

`extractInterval(begin, end)` preserves the source time axis. `rebased(new_start)` explicitly
shifts it without changing local coefficients. `appendRebased(other)` shifts an arbitrary-origin
suffix to the receiver's end, retaining its boundary values. Transforms return `StatusOr<T>`
or `Status` when they can fail.
See [numerical notes](numerical_notes.md) for conditioning.
