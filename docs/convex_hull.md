# Convex-hull representations

`SplineConvexHull.hpp` converts power-basis polynomials and their derivatives to Bezier or
MINVO control points. Both support subdivision and coefficient/duration adjoints. Included
MINVO bases cover degrees 0–7. The original curve is unchanged.

## Prepare before solving

```cpp
#include "SplineConvexHull.hpp"

SplineTrajectory::ConvexHullWorkspace<3> hull;
hull.prepare(spline.polynomial(), {SplineTrajectory::ConvexHullBasis::Bezier,
                                  SplineTrajectory::Deriv::Pos, 2});
Spline::CoefficientMatrix control_partials(hull.numPieces() * hull.controlsPerPiece(), 3);
Spline::CoefficientMatrix coefficient_partials(spline.numSegments() * Spline::kCoefficientCount, 3);
Eigen::VectorXd duration_partials(spline.numSegments());

// Repeat for each candidate with the same topology:
hull.update(spline.polynomial());
control_partials.setZero();
// Accumulate the cost's control-point partials here.
hull.backward(control_partials, coefficient_partials, duration_partials);
```

`prepare(poly, options)` or `prepare(segments, coefficient_count, options)` allocates the workspace
and shares immutable conversion data. `update(poly)` computes controls for the prepared topology
and reuses storage. Prepare each cost's workspace before solving, as shown in the
[convex-hull optimizer example](../examples/convex_hull_optimizer.cpp).

For occasional calls, `update(poly, options)` prepares when `matches(poly, options)` is false.
It may allocate on first use or topology/options change. Matching repetitions reuse storage.
`toBezier(poly, derivative, depth)`, `toMINVO(...)` and
`ConvexHullWorkspace<D>::fromPPoly(poly, options)` return owning, already updated workspaces.
Reuse a workspace for repeated evaluations.

`HullOptions` contains `{basis, derivative_order, subdivision_depth}`, defaulting to Bezier,
position and depth zero. Orders accept `Deriv` or integers via `DerivativeOrder`. Order must be
between zero and the source degree; depth is in [0,16]. Each additional depth doubles the
number of pieces and their storage. Use a separate workspace for each concurrent evaluation.

## Piece views and gradients

Depth d makes 2^d pieces per source segment. Controls have one spatial point per row, ordered by
piece. A piece has `controlsPerPiece()` rows. `hull.piece(i)` is a borrowed proxy:

```cpp
const auto piece = hull.piece(i);
const auto controls = piece.controls();
const int corridor = piece.info().source_segment;
// controls.row(local) is one control point; info also provides physical start_time/duration.
```

`pieceInfo(i)`, `pieceControls(i)`, `pieceInfos()` and `controls()` provide direct access.
Piece views refer to the last update and remain valid until the workspace changes or is destroyed.

| Form | Behavior |
| --- | --- |
| `backward(control_partials)` | Returns owning `{coefficients, durations}` |
| `backward(control_partials, result)` | Overwrites an owning BackwardResult, resizing if needed |
| `backward(control_partials, coefficient_partials, duration_partials)` | Overwrites owning outputs; matching sizes reuse storage |
| `backwardAdd(control_partials, coefficient_partials, duration_partials)` | Adds to matching preallocated outputs; no resizing or clearing |
| `backwardAdd(control_partials, gradient_view)` | Adds directly to a coefficient-stage gradient view |

`spline.backward(gC, gT)` maps both coefficient and independent duration partials through
interpolation to waypoint, boundary and duration gradients.

For costs that depend directly on piece start times/durations, `backwardPieceTimesAdd()` adds
those separate partials to source duration/start-time gradients. It also accepts a coefficient
gradient view. This is additional to the control conversion's duration partials.

Use finite control partials matching `controls()`, with separate output buffers.

## Optimizer integration

A user-defined objective holds a callable `coefficient` member. Its `operator()` receives
`ParameterView<Spline>` and `CoefficientGradientView<Spline>`. Prepare its hulls and gradients
after successful optimizer preparation, then use fixed-topology `hull.update(state.polynomial)`
and `hull.backwardAdd(control_partials, gradient)` in the callback.

A curve piece lies in the convex hull of its controls. If every position control lies inside one
convex corridor, the entire piece lies there. Velocity/acceleration controls give conservative
bounds on those derivatives. Subdivision tightens the bounds. Apply these bounds to convex
constraints; use trajectory collision checks for nonconvex ESDF free space.

## Basis conversion

For p(t) = Σ c_k t^k on a segment of duration T, normalize with u=t/T, so a_k=T^k c_k. A fixed
matrix converts normalized power coefficients to controls. Derivative order r first replaces
c_k by (k+r)!/k! c_(k+r) and reduces the degree by r. Subdivision composes this conversion with
an affine local-time transform. Its transpose propagates control partials; differentiating T's
powers gives independent duration partials. The spline adjoint then accounts for interpolation.

[Detailed derivation (中文)](convex_hull_basis_zh.md) · [MINVO provenance](../THIRD_PARTY_NOTICES.md)
