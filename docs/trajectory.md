# Trajectories, sampling and gradients

`CubicSplineND<DIM>`, `QuinticSplineND<DIM>` and `SepticSplineND<DIM>` interpolate waypoints and
minimize squared acceleration, jerk and snap. `MinDerivativeSplineND<DIM, S>` selects the same
implementations for `S = 2, 3, 4`.

## Construction

Rows of the waypoint matrix are points. Durations are positive seconds, one per segment;
`start_time` locates the trajectory on its time axis. Endpoint derivatives default to zero.

```cpp
using Spline = SplineTrajectory::QuinticSplineND<3>;
Spline::CoefficientMatrix points(3, 3);
points << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
SplineTrajectory::BoundaryConditions<3> boundary;
boundary.start_velocity = Eigen::Vector3d(0.2, 0, 0);
auto created = Spline::create({1.2, 1.5}, points, 0.0, boundary);
if (!created) return created.status();
auto spline = std::move(created).value();
```

`update(durations, points, start_time, boundary)` returns `Status` and reuses storage for unchanged topology.
A spline owns its `PPolyND` polynomial. `polynomial()` borrows it; `copyPolynomial()` returns an
independent snapshot. An update or move invalidates borrowed segments, iterators and samplers.

## Sampling

```cpp
const auto &poly = spline.polynomial();
auto position = poly.evaluate(0.5);
auto state = poly.evaluateDerivatives<2>(0.5);
auto sampler = poly.makeSampler<3>();
auto times = poly.generateTimeSequence(0.01);
if (!position || !state || !times) { /* Handle the corresponding status. */ }
for (double time : times.value()) {
    const auto pvaj = sampler.evaluateDerivatives(time);
    if (!pvaj) return pvaj.status();
    // Consume pvaj.value()[0] through pvaj.value()[3].
}
```

A sampler owns its cursor and derivative coefficients. Consecutive queries reuse the current
segment's coefficients; backward jumps are supported. Each traversal uses its own sampler.
No source warm-up is required. `evaluateInto(times, output, derivative)` writes batch samples
into a caller-owned matrix, one row per time.

Polynomial queries use the trajectory time axis. `poly.at(i).value().evaluateLocal(t)` uses time from
that segment's start. Interior knots select the right segment; the final knot selects the last.
Queries outside the trajectory extrapolate the endpoint polynomial.

## Energy and adjoints

```cpp
auto energy = spline.energy();
auto gradient = spline.energyGradient();
if (!energy) return energy.status();
if (!gradient) return gradient.status();
// energy.value(); gradient.value().inner_points/durations/start/end
```

For a custom coefficient-space cost, accumulate coefficient and independent duration partials,
then use `spline.backward(coefficient_partials, duration_partials, output)` to obtain physical
waypoint, duration and boundary gradients. The output is overwritten. For repeated calls,
prepare it once with `output.resetTopology(spline.numSegments())`.

`energyDurationPartials()` holds local coefficients fixed; `energyDurationGradient()` includes
the interpolation constraints. Use the latter for derivatives of the interpolating trajectory's energy.

## Polynomial storage

`PPolyND<DIM, MaxCoefficients>` stores rows by segment and ascending powers of local physical time.
The compile-time parameter is coefficient capacity, not polynomial degree. Spline traits are
`kDimension`, `kDegree`, `kCoefficientCount` and `kMinDerivativeOrder`.

Fallible value queries return `StatusOr<T>`; check the result before accessing `.value()`.
Output-reference operations return `Status`. Callers guarantee finite input values, legal orders
and indices, positive durations or strictly increasing knots, and matching input/output shapes.
The low-level APIs do not validate these preconditions. `at()` retains `StatusOr<Segment>` but
requires the same valid index as `operator[]`. Derivatives above degree remain zero.
Callers also guarantee a representable working range, including extrapolation. Evaluation,
coefficient transforms, energy and gradients do not scan their results for overflow.
Invalid object state still returns an error. Spline construction reports unrepresentable knots
or failed factorization and clears the polynomial; it does not rescan completed coefficients.
Direct polynomial `update()` copies trusted data. Creation uses `create(...)`; borrowed getters
retain their documented lifetime/preconditions. Batch state is checked once before writing,
so state failures leave output untouched. See [numerical notes](numerical_notes.md) for conditioning.
