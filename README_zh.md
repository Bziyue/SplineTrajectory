# SplineTrajectory

[English](README.md) · [优化器协议](include/SplineOptimizerProtocols.md) · [本次验证](docs/refactor_validation.md)

C++17 / Eigen 纯头文件、N 维最小控制量样条库，与 MINCO 数学等价。
三次、五次、七次样条分别最小化加速度、jerk、snap 平方积分；构造和伴随传播使用 O(N)
块三对角求解，能量另有直接作用于物理参数的解析梯度路径。

| 公共头文件 | 职责 |
| --- | --- |
| `SplineTrajectory.hpp` | 样条、合法性校验、分段多项式、导数和采样、精确截取追加、能量和伴随传播 |
| `SplineOptimizer.hpp` | 问题与上下文、变量映射和掩码、静态代价绑定、积分与梯度组装 |
| `SplineConvexHull.hpp` | Bezier/MINVO 控制凸包、物理导数、细分及系数/时间梯度传播 |

优化器和凸包头文件各自依赖轨迹头文件；ROS、业务走廊、L-BFGS 和最终可行性验收留在项目层。
凸包包含性质不等于软惩罚优化结果一定可行。

## 无需预热的顺序采样

```cpp
const auto &poly = spline.getTrajectory();
auto sampler = poly.makeSampler<3>();
for (double t : increasing_times) {
    const auto state = sampler.evaluate(t);
    consume(state[0], state[1], state[2], state[3]);
}
```

采样器只借用轨迹，源轨迹必须持续存活且不可修改。源更新、赋值或移动后应重新创建采样器。
每个遍历者独立持有采样器；多个读取者从第一次 const 查询起就无需预热，不写共享惰性缓存。
固定阶数在构造和采样循环中均不分配堆内存；动态阶数仅在采样器构造时分配当前段存储。
进入新段时一次准备所需导数系数，同段查询复用。向前查询逐段推进，时间回退则二分重新定位。

内部断点归右段，末端点归最后一段；域外沿相邻段外推。非法轨迹或非有限查询时间返回 NaN，
高于多项式次数的导数为零。随机访问使用 `evaluate(t, derivative)` 或一次选段的
`evaluateDerivatives<N>(t)`。为兼容现有调用保留 `last_idx_hint`，新代码推荐使用独立采样器。
精确截取和追加是系数运算，业务坐标系、轨迹身份和接缝可接受性由调用方处理。

## 优化生命周期与梯度

按「定义问题与配置 → `prepareContext(problem, context, config)` → 多次 `evaluate()` →
对求解器接受的 x 调用 `synchronizeWorkingState()` → `getWorkingSpline()`」使用。
上下文拥有准备结果和缓冲，显式传入的映射与代价使用借用语义；保持优化器及被借用对象存活。
每个活跃评价流独占上下文，并行积分还要求代价可并发调用。

推荐积分接口为 `(IntegralPointInfo, SampleState<DIM>, SampleGradient<DIM>&)`：
`p/v/a/j/s` 为物理状态及对应未加权偏导，`time` 为固定状态下显式全局时间偏导。
每次调用前梯度清零，积分权重和时间漂移由优化器统一处理。代价声明 `kDerivativeOrder`
后只计算需要的状态，但时间梯度始终保留额外一阶导数；未声明时默认为 4。
归一化积分基在准备阶段只计算一次，每段缩放一次系数，并在伴随传播前把梯度换回物理系数。
GCC/Clang 对内部积分调用链使用局部 flatten 属性，避免结构化回调使原有内联优化丢失；
其他编译器使用标准 C++ 路径，实际性能须重新测量。

可组合路点、跨样本、系数和轨迹参数代价。系数代价保留具体类型，在一次公共伴随传播前累计；
`VoidIntegralCost` 显式禁用积分，无样本记录要求时完全跳过积分循环。边界优化、掩码、共享时长、
空间映射与附加变量均保留。移动起点的积分、跨样本和系数显式时间偏导统一交给辅助映射反传。
完整可编译示例见 [英文 README](README.md)，精确签名及中英说明见 [协议文档](include/SplineOptimizerProtocols.md)。

## 迁移与失败语义

- 配置直接传入 `prepareContext`，移除 `setConfig` 和 active 配置镜像；修改配置须重新准备。
- `ConvexHullBasis.hpp` 改为 `SplineConvexHull.hpp`，独立的 `IntegralPointInfo.hpp` 由优化器头提供。
- 旧长参数积分回调仍可用于增量迁移；三个维护中的项目均使用结构化回调。
- 组合规格用 `auto` 或完整 `decltype` 表示，`withCoefficientCost` 现在改变规格类型；绑定的代价须为有效左值。
- 移动起点使用扩展 `backward(..., gradients, grad_start_time, grad_aux)`，删除业务层重复的显式时间补算。

非有限系数、非法维数、累计断点重合或求解失败会返回无效样条/状态。
无效评价返回正无穷和零梯度，构造失败清除工作轨迹，已准备上下文仍可用于后续合法候选。
梯度检查遇到非法扰动返回失败并恢复基准候选。读取工作轨迹前必须检查操作成功。
依赖这些 NaN/Inf 语义的消费方不能开启 `-ffast-math` / `-ffinite-math-only`；库接口目标不覆盖应用编译选项。

## 构建与验证

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
(cd build && ctest --output-on-failure)
./build/benchmark_optimizer 2000
./build/benchmark_sampling
```

测试、benchmark 和示例分别由 `BUILD_TESTING`、`SPLINE_BUILD_BENCHMARKS`、
`SPLINE_BUILD_EXAMPLES` 控制。默认关闭 `SPLINE_NATIVE_ARCH`；比较时统一编译选项。
CTest 覆盖 MINCO/差分梯度、边界、采样器首次并发读取、数值失败恢复、掩码和凸包完整梯度链。

[本次验证报告](docs/refactor_validation.md) 区分准备/首次使用、稳定采样、固定候选评价与完整求解；
模板绑定本身不能保证整条评价路径零成本。[凸包历史 benchmark](docs/convex_hull_performance_zh.md)
保留原实验适用范围。另见 [凸包数学说明](docs/convex_hull_basis_zh.md)、
[优化示例](examples/convex_hull_optimization_demo.cpp)、[MIT 许可证](LICENSE) 和 [MINVO 来源](THIRD_PARTY_NOTICES.md)。
