# Convex-hull representations

`SplineConvexHull.hpp` converts power-basis polynomials and their derivatives to Bezier or
MINVO control points. Both bases support subdivision and gradients with respect to coefficients
and segment durations. The included MINVO data covers polynomial degrees 0–7.

## Create and reuse a workspace

```cpp
#include "SplineConvexHull.hpp"

auto position = SplineTrajectory::toBezier(spline.polynomial(), 0, 2);
auto velocity = SplineTrajectory::toMINVO(spline.polynomial(), 1, 1);
```

The arguments after the polynomial are derivative order and binary subdivision depth.
Depth `d` creates `2^d` pieces per original segment. Controls are stored piece by piece;
`pieceInfo(i).source_segment` maps a subdivided piece back to its source corridor.
For a selectable basis or explicit memory budget, use
`ConvexHullWorkspace<3>::fromPPoly(polynomial, HullOptions{basis, derivative, depth}, budget_bytes)`.
The options match `configure()`; the positional basis/order/depth factory signature is replaced.

```cpp
SplineTrajectory::ConvexHullWorkspace<3> hull;
hull.configure(spline.polynomial(), {SplineTrajectory::ConvexHullBasis::Bezier, 0, 2});
// Repeat with the same topology:
hull.update(spline.polynomial());
hull.backwardAdd(control_partials, coefficient_partials, duration_partials);
```

`configure()` prepares storage and shared conversion data. `update()` and `backwardAdd()` reuse
that storage. The immutable conversion kernel can be shared; each concurrent evaluation needs
its own mutable workspace. The workspace retains the source values needed by its adjoint.
Fixed piece indices and normalized intervals are prepared by `configure()`; `update()` refreshes
controls and physical times. Numeric results are usable only after a successful update.

The adjoint has three output forms:

- `backward(control_partials)` allocates and returns coefficient and duration gradients.
- `backwardInto(control_partials, coefficient_partials, duration_partials)` overwrites pre-sized outputs.
- `backwardAdd(control_partials, coefficient_partials, duration_partials)` adds to pre-sized outputs.

`Into` and `Add` never resize or allocate. They check the workspace and every input/output shape
before writing; errors leave both outputs unchanged. Input and output buffers must not alias each
other or workspace storage. The former resizing three-argument `backward()` is replaced by
`backwardInto()`; size its outputs before calling, or use the allocating result overload.

## Convex constraints

A curve piece lies in the convex hull of its controls. If all its position controls lie inside
a convex corridor, the entire piece lies inside that corridor. Applying the same argument to
velocity or acceleration controls gives a conservative norm bound for the entire piece.
Subdivision tightens these bounds without changing the trajectory.

An ESDF free-space region is generally nonconvex: positive distances at all control points do
not prove that the entire curve is collision free. Soft penalties also do not guarantee exact
constraint satisfaction.

## Optimizer integration

A `coefficient` objective receives `ParameterView<Spline>` and `CoefficientGradient<Spline>`.
Prepare its hull workspace from `optimizer.polynomial()` before solving, then update controls,
accumulate control partials and call `backwardAdd()` inside the callback. Include the duration
partials of the conversion as well as coefficient partials.

The [convex-hull example](../examples/convex_hull_optimizer.cpp) combines position corridors,
velocity/acceleration control bounds, minimum jerk and total duration with GCOPTER L-BFGS.

## Basis conversion

For `p(t) = Σ c_k t^k` on a segment of duration `T`, normalize time with `u = t/T`.
The normalized power coefficients are `a_k = T^k c_k`. A fixed matrix converts `a` to Bezier or
MINVO controls. For derivative order `r`, first replace `c_k` by
`(k+r)! / k! * c_(k+r)` and use degree `degree-r`.

Subdivision composes this conversion with the affine local-time transform for each child piece.
Its transpose propagates control partials to coefficients; differentiating the powers of `T`
provides the explicit duration partials. The interpolation adjoint subsequently accounts for
the dependence of `c` on waypoint and duration variables.

[Detailed derivation (中文)](convex_hull_basis_zh.md) · [MINVO provenance](../THIRD_PARTY_NOTICES.md)
