# Trajectories, sampling and gradients

`CubicSplineND<DIM>`, `QuinticSplineND<DIM>` and `SepticSplineND<DIM>` interpolate waypoints and
minimize squared acceleration, jerk and snap, respectively.

## Construction

Rows of the waypoint matrix are points. Durations are positive seconds, one per segment;
`start_time` locates the trajectory on its time axis. Endpoint derivatives default to zero.

```cpp
using Spline = SplineTrajectory::QuinticSplineND<3>;
Spline::CoefficientMatrix points(3, 3);
points << 0, 0, 0, 1, 0.5, 0, 2, 0, 0;
SplineTrajectory::BoundaryConditions<3> boundary;
boundary.start_velocity = Eigen::Vector3d(0.2, 0, 0);
Spline spline;
if (const auto status = spline.update({1.2, 1.5}, points, 0.0, boundary); !status)
    return status;
```

`update(durations, points, start_time, boundary)` returns `Status` and reuses storage for unchanged topology.
Alternatively, `update(knots, points, boundary)` takes N+1 absolute waypoint times.
Both overloads initialize an empty spline or update an existing one; check success before queries.
A spline owns its `PPolyND` polynomial. `polynomial()` borrows it; `copyPolynomial()` returns an
independent snapshot. An update or move invalidates borrowed segments, iterators and samplers.

## Sampling

```cpp
const auto &poly = spline.polynomial();
auto position = poly.evaluate(0.5);
auto state = poly.evaluateDerivatives<2>(0.5);
auto sampler = poly.makeSampler<3>();
auto times = poly.generateTimeSequence(0.01);
if (!times) { /* Handle time-sequence generation failure. */ }
for (double time : times.value()) {
    const auto pvaj = sampler.evaluateDerivatives(time);
    // Consume pvaj[0] through pvaj[3].
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
auto energy = spline.energy();
auto gradient = spline.energyGradient();
// energy is a double; gradient owns inner_points/durations/start/end.
```

For a custom coefficient-space cost, accumulate coefficient and independent duration partials,
then use `spline.backward(coefficient_partials, duration_partials, output)` to obtain physical
waypoint, duration and boundary gradients. Overwrite overloads resize their outputs automatically;
unchanged segment counts reuse the existing storage. A default-constructed `Spline::Gradients`
can be passed directly, and `output.size()` reports its segment count.

`energyGradientAdd(output, weight)` adds a weighted energy contribution to matching initialized
gradients. To start an accumulation from zero, use `output.setZero(spline.numSegments())`;
`output.setZero()` clears the current gradients without changing their sizes.
All spline energy and adjoint methods require successful construction and return values directly
(or `void` for output-reference overloads).

`energyDurationPartials()` holds local coefficients fixed; `energyDurationGradient()` includes
the interpolation constraints. Use the latter for derivatives of the interpolating trajectory's energy.

## Polynomial storage

`PPolyND<DIM, MaxCoefficients>` stores rows by segment and ascending powers of local physical time.
The compile-time parameter is coefficient capacity, not polynomial degree. Spline traits are
`kDimension`, `kDegree`, `kCoefficientCount` and `kMinDerivativeOrder`.

Polynomial queries require an initialized source (asserted at the public entry) and return
values directly, or `void` when writing pre-sized outputs. Polynomial transforms retain their
`StatusOr<T>`/`Status` error contracts. Borrowed `Sampler` and `Segment` views assert source
validity at construction, then return values or write pre-sized outputs directly. The source must stay alive and unchanged. `at()` and
`operator[]` both return a borrowed `Segment`; neither checks the caller-supplied index.

`PPolyND::create`, `zero`, and `constant` return owning polynomials directly; `update` returns
`void`. Callers guarantee finite input values, legal orders and indices, matching shapes,
strictly increasing knots and representable arithmetic. Derivatives above degree remain zero.
Full-capacity fixed-size polynomials expose a static Horner bound; reduced-degree and dynamic
polynomials still use the actual coefficient count.

Spline initialization and updates return `Status` for unrepresentable knots and factorization
failure, clearing the polynomial on failure. Energy and adjoints require a successful build;
they assert that precondition in debug builds without scanning results for overflow.
Single-derivative batch queries fill above-degree outputs with zero before traversing the samples.
Fixed-capacity and reduced-degree queries share the existing Horner dispatch.
See [numerical notes](numerical_notes.md) for conditioning.
