# Build and integration

Requirements: C++17, CMake ≥ 3.12 and Eigen ≥ 3.3. Tests and benchmark tooling also use Python 3.
The library has no ROS dependency.

```sh
git clone https://github.com/Bziyue/SplineTrajectory.git
cd SplineTrajectory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
(cd build && ctest --output-on-failure)
```

| Option | Default | Purpose |
| --- | --- | --- |
| `BUILD_TESTING` | `ON` | Numerical and API tests |
| `SPLINE_BUILD_EXAMPLES` | `ON` | Runnable examples |
| `SPLINE_BUILD_BENCHMARKS` | `OFF` | Kernel and planner comparisons |
| `SPLINE_NATIVE_ARCH` | `OFF` | Enable CPU-specific tuning for repository executables |

For Eigen installations outside the default search paths, set `-DEigen3_DIR=/path/to/Eigen3/cmake`.
Test executables are written to `build/tests/`, examples to `build/examples/`, and benchmarks to
`build/benchmarks/`.

## Use the library

Copy the three `include/Spline*.hpp` files into your project, or add this repository as a CMake
subdirectory and link its interface target:

```cmake
add_subdirectory(external/SplineTrajectory EXCLUDE_FROM_ALL)
target_link_libraries(your_target PRIVATE SplineTrajectory::SplineTrajectory)
```

The interface target supplies the headers, Eigen dependency and C++17 requirement.
Test, example and benchmark options control the repository's own executables.

The examples use the bundled GCOPTER L-BFGS solver; library users can choose their own solver.
Compile with IEEE NaN/Inf semantics (`-fno-fast-math`) to support finite-value checks.
