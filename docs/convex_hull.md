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

`prepare(poly, options)` or `prepare(segments, coefficient_count, options)` allocates workspace
and acquires shared immutable conversion data. It sets `isPrepared()`, but controls are invalid
until update. The one-argument `update(poly)` requires matching prepared topology, returns void,
and reuses storage without allocation or cache locking. Do this preparation for **each user-owned
cost workspace** before solving; `optimizer.prepare()` only prepares the library's own buffers.
The [convex-hull optimizer example](../examples/convex_hull_optimizer.cpp) shows that boundary.

For occasional calls, `update(poly, options)` prepares when `matches(poly, options)` is false.
It may allocate on first use or topology/options change. Matching repetitions reuse storage.
`toBezier(poly, derivative, depth)`, `toMINVO(...)` and
`ConvexHullWorkspace<D>::fromPPoly(poly, options)` return owning, already updated workspaces.
They do not retain a source reference. Repeated factories allocate; reuse a workspace in a solver.

`HullOptions` contains `{basis, derivative_order, subdivision_depth}`, defaulting to Bezier,
position and depth zero. Orders accept `Deriv` or integers via `DerivativeOrder`. Order must be
between zero and the source degree; depth is in [0,16]. Callers own feasible memory/index limits.
There is no memory-budget API or public kernel pointer. Cached conversion data remain shared;
each concurrent evaluation needs its own mutable workspace.

## Piece views and gradients

Depth d makes 2^d pieces per source segment. Controls have one spatial point per row, ordered by
piece. A piece has `controlsPerPiece()` rows. `hull.piece(i)` is a borrowed proxy:

```cpp
const auto piece = hull.piece(i);
const auto controls = piece.controls();
const int corridor = piece.info().source_segment;
// controls.row(local) is one control point; info also provides physical start_time/duration.
```

`pieceInfo(i)`, `pieceControls(i)`, `pieces()` and `controls()` remain available. Update/prepare,
movement and destruction invalidate borrowed numeric views. Inspect controls only after update.

| Form | Behavior |
| --- | --- |
| `backward(control_partials)` | Returns owning `{coefficients, durations}` |
| `backward(control_partials, result)` | Overwrites an owning BackwardResult, resizing if needed |
| `backward(control_partials, coefficient_partials, duration_partials)` | Overwrites owning outputs; matching sizes reuse storage |
| `backwardAdd(control_partials, coefficient_partials, duration_partials)` | Adds to matching preallocated outputs; no resizing or clearing |
| `backwardAdd(control_partials, gradient_view)` | Adds directly to a coefficient-stage gradient view |

Keep **both** coefficient and independent duration partials. `spline.backward(gC, gT)` then maps
that pair through interpolation to waypoint, boundary and duration gradients. The explicit
composition avoids a second representation of the same gradients.

For costs that depend directly on piece start times/durations, `backwardPieceTimesAdd()` adds
those separate partials to source duration/start-time gradients. It also accepts a coefficient
gradient view. This is additional to the control conversion's duration partials.

All adjoints require updated controls, finite matching inputs and nonaliasing outputs. Add never
allocates or resizes; owning overwrite may resize. Shapes and output arithmetic are not rescanned.
Allocation failures propagate. Prepare and numerical updates require exclusive workspace access.

## Optimizer integration and feasibility

A user-defined objective holds a callable `coefficient` member. Its `operator()` receives
`ParameterView<Spline>` and `CoefficientGradientView<Spline>`. Prepare its hulls and gradients
after successful optimizer preparation, then use fixed-topology `hull.update(state.polynomial)`
and `hull.backwardAdd(control_partials, gradient)` in the callback.

A curve piece lies in the convex hull of its controls. If every position control lies inside one
convex corridor, the entire piece lies there. Velocity/acceleration controls give conservative
bounds on those derivatives. Subdivision tightens the bounds. Soft penalties still require a
separate feasibility check after rebuilding the solver's final candidate. ESDF free space is
usually nonconvex: positive control-point distances alone do not prove collision freedom.

## Basis conversion

For p(t) = Σ c_k t^k on a segment of duration T, normalize with u=t/T, so a_k=T^k c_k. A fixed
matrix converts normalized power coefficients to controls. Derivative order r first replaces
c_k by (k+r)!/k! c_(k+r) and reduces the degree by r. Subdivision composes this conversion with
an affine local-time transform. Its transpose propagates control partials; differentiating T's
powers gives independent duration partials. The spline adjoint then accounts for interpolation.

[Detailed derivation (中文)](convex_hull_basis_zh.md) · [MINVO provenance](../THIRD_PARTY_NOTICES.md)
