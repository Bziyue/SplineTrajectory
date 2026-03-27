# SplineOptimizer Protocol Reference

This file documents the callable and mapper protocols used by `SplineOptimizer`.

These are reference interfaces only:

- user-defined functors and lambdas do not need to inherit from them
- they only need to satisfy the signatures checked by `SplineOptimizer` type traits

The current optimizer API also uses a few small integration types:

- `OptimizationContext`: caller-owned prepared problem plus mutable runtime state
- `EvaluateSpec`: binds borrowed cost functors and an executor
- `ErrorCode`: structured failure categories for setup and evaluation
- `Status`: returned by setup/validation style APIs
- `EvaluationResult`: returned by `evaluate(...)`
- `GradientCheckResult`: returned by `checkGradients(...)`

Typical flow:

1. create and initialize a `SplineOptimizer`
2. create one `OptimizationContext` per evaluation context
3. build an `EvaluateSpec` with `makeEvaluateSpec(...)`
4. call `evaluate(...)` and inspect `EvaluationResult::ok`

`EvaluateSpec` borrows cost objects through reference-like storage, so cost
functors passed to `makeEvaluateSpec(...)` and `with...Cost(...)` must be
lvalues that outlive the spec.

## TimeMap Protocol

```cpp
struct TimeMapProtocol
{
    // Convert unconstrained variable 'tau' to physical time 'T'
    double toTime(double tau) const;

    // Convert physical time 'T' to unconstrained variable 'tau'
    double toTau(double T) const;

    // Chain rule: dCost/dtau = (dCost/dT) * (dT/dtau)
    double backward(double tau, double T, double gradT) const;
};
```

## SpatialMap Protocol

Global absolute indexing:

- `index = 0`: start point
- `index = 1 ... N - 1`: inner waypoints
- `index = N`: end point

```cpp
struct SpatialMapProtocol
{
    // Dimension of unconstrained variable xi for the point at global index
    int getUnconstrainedDim(int index) const;

    // Forward: xi (unconstrained) -> p (physical)
    Eigen::VectorXd toPhysical(const Eigen::VectorXd& xi, int index) const;

    // Backward: p (physical) -> xi (unconstrained), used for initial guess
    Eigen::VectorXd toUnconstrained(const Eigen::VectorXd& p, int index) const;

    // Chain rule: dCost/dxi = (dCost/dp) * (dp/dxi)
    Eigen::VectorXd backwardGrad(const Eigen::VectorXd& xi,
                                 const Eigen::VectorXd& grad_p,
                                 int index) const;
};
```

## IntegralCost Protocol

```cpp
struct IntegralCostProtocol
{
    double operator()(double t,
                      double t_global,
                      int segment_index,
                      int step_in_seg,
                      const Eigen::VectorXd& p,
                      const Eigen::VectorXd& v,
                      const Eigen::VectorXd& a,
                      const Eigen::VectorXd& j,
                      const Eigen::VectorXd& s,
                      Eigen::VectorXd& gp,
                      Eigen::VectorXd& gv,
                      Eigen::VectorXd& ga,
                      Eigen::VectorXd& gj,
                      Eigen::VectorXd& gs,
                      double& gt) const;
};
```

## TimeCost Protocol

```cpp
struct TimeCostProtocol
{
    double operator()(const std::vector<double>& Ts,
                      Eigen::VectorXd& grad) const;
};
```

## WaypointsCost Protocol

```cpp
struct WaypointsCostProtocol
{
    double operator()(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& waypoints,
                      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& grad_q) const;
};
```

## SampleCost Protocol

```cpp
struct SampleCostProtocol
{
    template <typename SamplesType>
    double operator()(const SamplesType& samples,
                      Eigen::Matrix<double, 3, Eigen::Dynamic>& grad_p,
                      Eigen::VectorXd& grad_t_global) const;
};
```

`samples` is the recorded integration buffer. Each sample provides:

- `seg_idx`: segment index
- `step_in_seg`: sample step inside the segment
- `sample_buffer_index`: index inside the recorded sample buffer
- `alpha`: normalized time in the segment, in `[0, 1]`
- `t_local`: local time within the segment
- `t_global`: absolute time
- `trap_weight`: trapezoidal integration weight factor
- `dt`: segment integration step size
- `b_p`: position basis row for the spline coefficients
- `p`, `v`: sampled position and velocity

Under the current optimizer data model, `SampleCost` is a discrete cost on:

- sampled position `p`
- sampled global time `t_global`

It does not currently expose independent sample-backward channels for
`v/a/j/s`.

## TrajectoryCost Protocol

```cpp
struct TrajectoryCostProtocol
{
    // The concrete spline type is determined by SplineOptimizer<DIM, SplineType, ...>
    double operator()(const QuinticSplineND<3>& spline,
                      const std::vector<double>& Ts,
                      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& waypoints,
                      double start_time,
                      const BoundaryConditions<3>& bc,
                      QuinticSplineND<3>::Gradients& grads) const;
};
```

## AuxiliaryStateMap Protocol

```cpp
struct AuxiliaryStateMapProtocol
{
    // Dimension of the auxiliary optimization variable block
    int getDimension() const;

    // Create initial unconstrained auxiliary variables from the reference state
    Eigen::VectorXd getInitialValue(const std::vector<double>& ref_times,
                                    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& ref_waypoints,
                                    double ref_start_time,
                                    const BoundaryConditions<3>& ref_bc) const;

    // Apply the auxiliary variables onto the working state before spline update
    void apply(const Eigen::VectorXd& z,
               std::vector<double>& times,
               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& waypoints,
               double& start_time,
               BoundaryConditions<3>& bc) const;

    // Back-propagate gradients from the working state to z
    // The concrete spline type is determined by SplineOptimizer<DIM, SplineType, ...>
    double backward(const Eigen::VectorXd& z,
                    const QuinticSplineND<3>& spline,
                    const std::vector<double>& times,
                    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& waypoints,
                    double start_time,
                    const BoundaryConditions<3>& bc,
                    QuinticSplineND<3>::Gradients& grads,
                    Eigen::VectorXd& grad_z) const;
};
```

## Runtime Integration Notes

`SplineOptimizer` no longer owns an internal prepared problem or runtime state.
Callers should provide one `OptimizationContext` per active evaluation flow:

```cpp
using Optimizer = SplineTrajectory::SplineOptimizer<3>;

Optimizer optimizer;
Optimizer::OptimizationContext context;

auto status = optimizer.prepareContext(problem, context);
if (!status)
{
    std::cerr << status.message << std::endl;
}

auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost);
auto x = optimizer.generateInitialGuess(context);
auto result = optimizer.evaluate(context, x, grad, spec);
if (!result)
{
    std::cerr << result.message << std::endl;
}
```

`prepareContext(...)` snapshots the config-dependent mapper and scalar state
needed by later evaluation, including:

- `TimeMap`
- `SpatialMap`
- `AuxiliaryStateMap`
- `rho_energy`
- `integral_num_steps`

Later `setConfig(...)` calls affect newly prepared contexts, but do not mutate
existing prepared contexts.

Setup and validation style APIs use `Status`:

```cpp
Optimizer::OptimizerConfig config;
config.rho_energy = 0.05;
config.integral_num_steps = 32;
optimizer.setConfig(config);

Optimizer::ProblemDefinition problem;
problem.time_segments = durations;
problem.waypoints = waypoints;
problem.start_time = 0.0;
problem.bc = bc;

Optimizer::OptimizationContext context;
auto status = optimizer.prepareContext(problem, context);
if (!status)
{
    std::cerr << static_cast<int>(status.code) << std::endl;
    std::cerr << status.message << std::endl;
}
```

If you want an explicit mask for just this problem, attach it directly to the
problem definition before preparing the context:

```cpp
Optimizer::OptimizationMask mask;
mask.time.assign(durations.size(), static_cast<uint8_t>(1));
mask.waypoints.assign(durations.size() + 1, static_cast<uint8_t>(1));
mask.waypoints.front() = static_cast<uint8_t>(0);
mask.waypoints.back() = static_cast<uint8_t>(0);

problem.mask = mask;

auto status = optimizer.prepareContext(problem, context);
```

If your upstream pipeline provides absolute time points instead of durations,
build the problem with the helper and still use the same `prepareContext(...)`
entrypoint:

```cpp
std::vector<double> time_points = {0.0, 1.2, 2.7, 4.0};
auto problem = Optimizer::makeProblemFromTimePoints(time_points, waypoints, bc);
auto status = optimizer.prepareContext(problem, context);
```

Evaluation uses `EvaluationResult` with the same error code pattern:

```cpp
auto result = optimizer.evaluate(context, x, grad, spec);
if (!result)
{
    std::cerr << static_cast<int>(result.code) << std::endl;
    std::cerr << result.message << std::endl;
}
```

Optional costs and a custom executor can be attached fluently:

```cpp
auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost)
                .withWaypointsCost(waypoints_cost)
                .withSampleCost(sample_cost)
                .withTrajectoryCost(trajectory_cost)
                .withExecutor(OpenMPExecutor{});
```

When using `OpenMPExecutor`, the borrowed `integral_cost` object must be
thread-safe because the same functor instance is invoked concurrently across
segments.
