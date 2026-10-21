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
Spline spline({1.2, 1.5}, points, 0.0, boundary);
if (!spline.isValid()) throw std::runtime_error("Invalid spline");
```

`update(durations, points, start_time, boundary)` reuses storage for unchanged topology.
A spline owns its `PPolyND` polynomial. `polynomial()` borrows it; `copyPolynomial()` returns an
independent snapshot. An update or move invalidates borrowed segments, iterators and samplers.

## Sampling

```cpp
const auto &poly = spline.polynomial();
auto position = poly.evaluate(0.5);
auto state = poly.evaluateDerivatives<2>(0.5);
auto sampler = poly.makeSampler<3>();
for (double time : poly.generateTimeSequence(0.01)) {
    const auto pvaj = sampler.evaluateDerivatives(time);
    // Consume pvaj[0], pvaj[1], pvaj[2], pvaj[3].
}
```

A sampler owns its cursor and derivative coefficients. Consecutive queries reuse the current
segment's coefficients; backward jumps are supported. Each traversal uses its own sampler.
No source warm-up is required. `evaluateInto(times, output, derivative)` writes batch samples
into a caller-owned matrix, one row per time.

Polynomial queries use the trajectory time axis. `poly.at(i).evaluateLocal(t)` uses time from
that segment's start. Interior knots select the right segment; the final knot selects the last.
Queries outside the trajectory extrapolate the endpoint polynomial.

## Energy and adjoints

```cpp
const double energy = spline.energy();
auto gradient = spline.energyGradient();
// gradient.inner_points, gradient.durations and gradient.boundary
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

Invalid numerical queries return NaNs; derivatives above the polynomial degree are zero.
Negative derivative orders throw `std::invalid_argument`, and `at()` checks indices.
Invalid construction clears the polynomial. See [numerical notes](numerical_notes.md) for conditioning.
