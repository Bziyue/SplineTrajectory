# Tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 3
(cd build && ctest --output-on-failure)
```

| Directory / CTest label | Coverage |
| --- | --- |
| `trajectory/` / `trajectory` | Polynomials, sampling and lifetime contracts |
| `gradients/` / `gradients` | Energy, waypoint, duration and boundary derivatives |
| `optimizer/` / `optimizer` | Objective composition, maps, workspace and numerical failures |
| `convex_hull/` / `convex_hull` | Bezier/MINVO values, subdivision and adjoints |
| `compile/` / `compile` | Public headers, callback contracts and both English/Chinese README programs |
| `support/` | Shared checks and allocation counters |
| `../examples/` / `examples` | Runnable examples and objective gradients |

Run a category with `ctest -L`, or a test with `ctest -R`:

```sh
(cd build && ctest -L optimizer --output-on-failure)
(cd build && ctest -L examples --output-on-failure)
(cd build && ctest -R test_sampler --output-on-failure)
```

Assertion-dependent checks remain enabled in Release; numerical-failure tests also exercise normal Release semantics. Performance measurements are in
[benchmarks](../docs/benchmarks.md) and are enabled separately.

Optimizer allocation checks cover prepared success and rejected-candidate recovery for all three
spline families, using both standard heap counters and Eigen's runtime no-malloc guard.
