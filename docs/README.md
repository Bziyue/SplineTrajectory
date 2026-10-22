# Documentation

| Guide | Contents |
| --- | --- |
| [Online API reference · English / 中文](https://bziyue.github.io/SplineTrajectory/api/?lang=en) | Bilingual signatures, shapes, contracts, examples and interactive plots; also available offline by opening `docs/api/index.html` after cloning. |
| [Build and integration](building.md) | Requirements, CMake options and library integration |
| [Trajectory](trajectory.md) | Construction, sampling, energy and adjoints |
| [Optimizer](optimizer.md) | Objective stages, parameterization and solver callbacks |
| [Convex hulls](convex_hull.md) | Bezier / MINVO bounds, subdivision and gradients |
| [Examples](../examples/README.md) | Runnable trajectory optimization problems |
| [Tests](../tests/README.md) | Test suites and commands |
| [Benchmarks](benchmarks.md) | Reproducible performance and numerical comparisons |
| [Numerical notes](numerical_notes.md) | Conditioning, solver status and constraint interpretation |
| [API migration](api_migration.md) | Updating previous API names |

GitHub Pages publishes `main:/docs` directly, with `.nojekyll` preserving the static files.
The site root redirects to `api/`, retaining the language query and section anchor.
Edit the static files in `docs/api/` and push to `main` to publish documentation changes;
no generator or separate deployment branch is required.
