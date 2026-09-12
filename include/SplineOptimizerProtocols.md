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

For allocation-free pullback, a spatial map may additionally provide:

```cpp
void backwardGradInto(const Eigen::Ref<const Eigen::VectorXd>& xi,
                      const Eigen::Vector3d& grad_p, int index,
                      Eigen::Ref<Eigen::VectorXd> grad_xi) const;
```

The output is a borrowed, correctly sized decision-vector slice. Write the full
pullback there; do not resize it. The optimizer selects this overload at compile
time, falling back to owning `backwardGrad()` for existing maps. `toPhysical()`
may also accept a const Eigen Ref and return a fixed-size vector. Optional domain
penalties must add to the existing decision gradient instead of clearing it.

## IntegralCost Protocol

An integral cost may optionally implement `beginEvaluation() const`. The optimizer
calls it once after decoding the current decision vector and before visiting any
integration sample. It is only for resetting adapter-owned diagnostics; adapters
must not borrow or retain optimizer working state:

```cpp
void beginEvaluation() const;
```

```cpp
struct IntegralCostProtocol
{
    static constexpr int kDerivativeOrder = 3;
    double operator()(const SplineTrajectory::IntegralPointInfo& point,
                      const SplineTrajectory::SampleState<3>& state,
                      SplineTrajectory::SampleGradient<3>& gradient) const;
};
```

`SampleGradient` starts at zero; add unweighted integrand partials to `p/v/a/j/s`
and the explicit global-time partial to `time`. The optimizer supplies trapezoidal
weights and the state time drift. `kDerivativeOrder` defaults to 4 and may be 0--4;
the framework also computes the next derivative needed by that drift. State above
the declared order must not be consumed. Legacy callbacks with separate state and
gradient arguments remain supported for downstream migration.

Normalized basis rows are prepared once for the fixed quadrature rule. Each piece
scales coefficients once, reuses them across samples, and converts coefficient
partials back to the physical polynomial before the adjoint. `VoidIntegralCost`
skips quadrature unless sample recording/cross-sample costs request it.
`VoidTimeCost` explicitly disables the duration objective. Disabled coefficient,
trajectory and auxiliary paths are discarded at compile time.

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

- `point`: immutable integration metadata (`segment_index`, `segment_count`,
  `step_index`, `step_count`, `alpha`, `segment_duration`, `step_size`,
  `local_time`, `global_time`, and exact endpoint helpers)
- `trap_weight`: trapezoidal integration weight factor
- `b_p`: position basis row for the spline coefficients
- `p`, `v`: sampled position and velocity

Under the current optimizer data model, `SampleCost` is a discrete cost on:

- sampled position `p`
- sampled global time `t_global`

It does not currently expose independent sample-backward channels for
`v/a/j/s`.

## CoefficientCost Protocol

Use this protocol for Bezier/MINVO hull costs and other objectives whose natural
gradient is expressed on the spline polynomial coefficients:

```cpp
struct CoefficientCostProtocol
{
    double operator()(QuinticSplineND<3>& spline,
                      const std::vector<double>& Ts,
                      double start_time,
                      QuinticSplineND<3>::MatrixType& grad_coeffs,
                      Eigen::VectorXd& grad_times,
                      double& grad_start_time) const;
};
```

Attach an lvalue callable with:

```cpp
auto spec = Optimizer::makeEvaluateSpec(time_cost, integral_cost)
    .withCoefficientCost(coefficient_cost);
```

The callback is borrowed as its concrete type and invoked directly; there is no
`void*` or function-pointer erasure. Binding does not allocate. It runs before the spline adjoint and must add to the
provided gradients rather than clear or resize them. `SplineOptimizer` then calls
`propagateGrad()` exactly once for all integral, sample, and coefficient costs.

## TrajectoryCost Protocol

```cpp
struct TrajectoryCostProtocol
{
    // The concrete spline type is determined by SplineOptimizer<DIM, SplineType, ...>
    double operator()(QuinticSplineND<3>& spline,
                      const std::vector<double>& Ts,
                      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& waypoints,
                      double start_time,
                      const BoundaryConditions<3>& bc,
                      QuinticSplineND<3>::Gradients& grads) const;
};
```

`TrajectoryCost` runs after the spline coefficient adjoint and works directly on
the final spline parameter gradients. New polynomial-coefficient or convex-hull
costs should use `CoefficientCost`; this avoids a second `propagateGrad()` call.
Existing trajectory callables remain compatible.

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

If the auxiliary map changes the common trajectory start time, it can use the
extended backward overload:

```cpp
double backward(const Eigen::VectorXd& z,
                const QuinticSplineND<3>& spline,
                const std::vector<double>& times,
                const MatrixType& waypoints,
                double start_time,
                const BoundaryConditions<3>& bc,
                QuinticSplineND<3>::Gradients& grads,
                double grad_start_time,
                Eigen::VectorXd& grad_z) const;
```

The optimizer detects this overload at compile time and forwards the sum of explicit
start-time partials from integral, cross-sample and coefficient costs. State-induced
time changes are already in the spline parameter gradients; the map applies its
boundary-motion Jacobian as well. Do not repeat explicit-time accumulation outside the library.

## Runtime Integration Notes

`SplineOptimizer` no longer owns an internal prepared problem or runtime state.
Callers should provide one `OptimizationContext` per active evaluation flow:

```cpp
using Optimizer = SplineTrajectory::SplineOptimizer<3>;

Optimizer optimizer;
Optimizer::OptimizationContext context;

auto status = optimizer.prepareContext(problem, context, config);
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

`setConfig()` and active mirrors are removed. Configuration is passed only to
`prepareContext`; changing the caller's configuration does not change prepared values.
Explicit map pointers are borrowed, not cloned; the optimizer and maps must outlive
contexts using their defaults or references.

For a valid problem, `prepareContext(...)` also sizes the reusable evaluation
buffers and performs one fixed-topology spline update. Consequently the first
optimizer callback uses the same preallocated coefficient, gradient, sample-time,
and auxiliary-gradient storage as later callbacks. The caller-owned output gradient
and optional sample/coefficient adapters may still allocate on their first use;
measure setup and the first callback separately from steady evaluation.

Setup and validation style APIs use `Status`:

```cpp
Optimizer::OptimizerConfig config;
config.rho_energy = 0.05;
config.integral_num_steps = 32;

Optimizer::ProblemDefinition problem;
problem.time_segments = durations;
problem.waypoints = waypoints;
problem.start_time = 0.0;
problem.bc = bc;

Optimizer::OptimizationContext context;
auto status = optimizer.prepareContext(problem, context, config);
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

auto status = optimizer.prepareContext(problem, context, config);
```

If your upstream pipeline provides absolute time points instead of durations,
build the problem with the helper and still use the same `prepareContext(...)`
entrypoint:

```cpp
std::vector<double> time_points = {0.0, 1.2, 2.7, 4.0};
auto problem = Optimizer::makeProblemFromTimePoints(time_points, waypoints, bc);
auto status = optimizer.prepareContext(problem, context, config);
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

LBFGS-style inner loops may use `evaluatePrepared(...)` after a successful
`prepareContext(...)`. It skips the duplicate defensive decode performed by
`evaluate(...)`; the context and finite decision-vector preconditions therefore
belong to the caller. After an external optimizer returns, call
`synchronizeWorkingState(context, x)` to rebuild only the output spline without
paying for another complete cost/gradient integration.

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

## 中文契约速查

配置仅在 `prepareContext(problem, context, config)` 冻结；映射和代价借用，
每个并发评价独占上下文。结构化积分回调使用 `SampleState` 与每次清零的
`SampleGradient`，后者是未乘积分权重的偏导。`kDerivativeOrder` 只裁掉不用的状态，
时间梯度仍计算下一阶导数。系数代价在公共伴随之前直接静态调用，轨迹参数代价在其后累加。
跨样本代价的独立输出仅为位置和显式全局时间偏导。移动起点的全部显式时间偏导统一传入
扩展辅助映射 `backward`；业务层不可重复补算。自定义空间映射可用 `backwardGradInto`
直接写已分配决策梯度切片，域惩罚必须保留并累加已有梯度。失败状态、借用期限与接受解同步
见 [中文 README](../README_zh.md)。
