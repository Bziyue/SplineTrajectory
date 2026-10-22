# Examples

Build from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target examples --parallel
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

The optimization examples use two-dimensional quintic splines and soft penalties.
The library also supports other dimensions and spline families.

## Add your own objective

An objective is a user-defined holder of callable members. Each cost implements `operator()`.
Use `options.duration_weight` for linear total-duration cost, `parameter` for other duration
costs, `integral` for pointwise state costs, and `coefficient` for polynomial or convex-hull costs.
Both built-in weights default to zero; set `energy_weight` explicitly for smoothness. Each callback adds its partials to
the supplied gradient; SplineOptimizer handles the interpolation and decision-variable chain.

The shared [L-BFGS driver](support/OptimizationExample.hpp) prepares the optimizer, obtains
`initialGuess()`, evaluates costs in the solver callback and rebuilds the returned vector with
`update()`. Validate the trajectory, then save it with `copyPolynomial()` for sampling.
Cost functions remain in the individual examples; [Penalties.hpp](support/Penalties.hpp) contains
the shared box and derivative penalties.

For an ESDF map, replace `CircleDistance::query()` with the map's distance and spatial gradient
in the same coordinate frame. For MINVO controls, change the basis in `HullCost` configuration
from `ConvexHullBasis::Bezier` to `ConvexHullBasis::MINVO`.

The optimization examples accept `--check-gradients` to check their initial objective
before solving. Full callback signatures are in the [optimizer reference](../docs/optimizer.md).

`SplineInput` waypoints are interpolation constraints for direct spline construction. In optimization,
default masks allow durations and interior waypoints to change. Prepare user-owned cost buffers
before solving: the convex-hull example explicitly prepares hulls and control gradients, then
uses fixed-topology `update(poly)` in each cost call.
