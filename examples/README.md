# Examples

Build from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target examples --parallel 3
./build/examples/basic_optimizer
./build/examples/corridor_optimizer
./build/examples/esdf_optimizer
./build/examples/dynamics_optimizer
./build/examples/convex_hull_optimizer
```

| Source | Demonstrates |
| --- | --- |
| [trajectory_sampling.cpp](trajectory_sampling.cpp) | Create a spline, sample position/velocity/acceleration and evaluate energy gradients |
| [basic_optimizer.cpp](basic_optimizer.cpp) | Minimum jerk plus total duration |
| [corridor_optimizer.cpp](corridor_optimizer.cpp) | Segment-specific boxes and sampled speed/acceleration penalties |
| [esdf_optimizer.cpp](esdf_optimizer.cpp) | ESDF queries, obstacle gradients and derivative limits |
| [dynamics_optimizer.cpp](dynamics_optimizer.cpp) | Time allocation with fixed waypoints and velocity/acceleration limits |
| [convex_hull_optimizer.cpp](convex_hull_optimizer.cpp) | Bezier control-point bounds with coefficient and time adjoints |
| [parameterized_optimizer.cpp](parameterized_optimizer.cpp) | Custom coordinate maps, auxiliary variables and an objective gradient check |

The optimization examples use two-dimensional quintic splines for compact geometry; the library
supports other dimensions and spline families. Penalties are soft. Sampled ESDF/corridor checks
are observations, while control points in a convex set can bound an entire polynomial piece.

## Add your own objective

An objective is an ordinary struct. Use `duration` for time costs, `integral` for sampled state
costs, and `coefficient` for polynomial or convex-hull costs. Each callback adds its partials to
the supplied gradient; SplineOptimizer handles the interpolation and decision-variable chain.

The shared [L-BFGS driver](support/OptimizationExample.hpp) shows the complete solver boundary:
prepare the optimizer, obtain `initialGuess()`, call `costAndGradient()` from the GCOPTER callback,
check the solver status and synchronize the returned vector with `updateAccepted()`.
Cost functions remain in the individual examples; [Penalties.hpp](support/Penalties.hpp) contains
the shared box and derivative penalties.

For an ESDF map, replace `CircleDistance::query()` with the map's distance and spatial gradient
in the same coordinate frame. For MINVO controls, change the basis in `HullCost` configuration
from `ConvexHullBasis::Bezier` to `ConvexHullBasis::MINVO`.

The five optimization examples accept `--check-gradients` to check their initial objective
before solving. Full callback signatures are in the [optimizer reference](../docs/optimizer.md).
