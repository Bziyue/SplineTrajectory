# SplineTrajectory

[English](README.md) · [优化器协议](include/SplineOptimizerProtocols.md) · [API 迁移](docs/api_migration.md)

C++17 / Eigen 纯头文件、N 维最小控制量样条库，与 MINCO 数学等价。
三次、五次、七次样条分别最小化加速度、jerk、snap 的平方积分。
三种 O(N) 专用求解与伴随内核保留；PPoly 独占系数和断点，构造后不再复制第二份系数。

| 公共头文件 | 职责 |
| --- | --- |
| `SplineTrajectory.hpp` | 插值、能量、伴随传播、分段多项式与采样 |
| `SplineOptimizer.hpp` | 问题、变量映射、代价评价与独占求解工作区 |
| `SplineConvexHull.hpp` | Bezier/MINVO 转换、细分与系数/时间反传 |

优化器和凸包头分别只依赖轨迹头；坐标系、ROS、外部求解器、业务约束和最终可行性由项目层负责。
公共接口采用通用数学语义，适用于不同轨迹优化问题。

## 简洁的优化接口

`SplineOptimizer<Spline>` 推导维度和次数，按值持有默认映射、问题快照与缓冲。
高级用法只增加一个具名 `Parameterization` 类型；外部对象可以通过 `std::ref/std::cref` 显式借用。
目标是普通结构体，按需声明 `duration/integral/sample/coefficient/parameter/decision` 成员。
缺少的阶段在编译期移除；签名错误产生指定阶段的诊断；空目标明确使用 `NoObjective`。

生命周期为 `prepare()` → `initialGuess()` → 重复 `evaluate()` → `updateAccepted(x)` → `polynomial()`。
不再要求调用方维护 Context、EvaluateSpec 或 builder 链。传入评价的 Objective 只在同步调用期间借用。
`costAndGradient()` 是外部求解器的标量适配，调用同一评价流程；最后一个试探点可能被求解器拒绝，
因此交付轨迹前必须以求解器接受的 x 调用 `updateAccepted()`。

准备完成且输出梯度已正确分配后，正常库内评价从首次调用起不分配堆内存；用户 map/cost 自己负责其分配行为。
带 sample 阶段时使用 `prepare(problem, options, objective)` 准备记录缓冲。
拓扑或记录能力改变时重新准备，评价不会偷偷扩大缓冲。优化器可移动、禁止隐式深复制。
每个并发求解独占优化器和可变代价工作区；并行积分还要求积分代价可并发调用。

[基础示例](examples/basic_optimizer.cpp) 给出完整求解循环；
[自定义映射示例](examples/parameterized_optimizer.cpp) 组合有界坐标、共享时间变量、移动起点、积分与 Bezier 代价；
[凸包优化示例](examples/convex_hull_optimization_demo.cpp) 展示 C 风格 L-BFGS 接入。
[英文 README](README.md) 含最小可编译代码；全部签名及时间梯度契约见[协议文档](include/SplineOptimizerProtocols.md)。

## 无需预热的采样

```cpp
const auto &poly = spline.polynomial();
auto sampler = poly.makeSampler<3>();
for (double t : increasing_times) {
    const auto state = sampler.evaluate(t);
    consume(state[0], state[1], state[2], state[3]);
}
```

采样器独占游标和当前段导数系数，进入另一段时准备，同段重复利用。
固定容量采样器构造与查询都不分配，动态容量仅在构造时分配当前段存储。
源轨迹 const 查询从首次读取起就没有共享写入。裸 `int* hint` 接口已删除。
源必须持续存活，源更新、赋值或移动后重新创建采样器。需要独立快照时使用 `copyPolynomial()`。

单阶查询使用 `evaluate(t, order)`，多阶查询使用 `evaluateDerivatives<N>(t)`；
`evaluateInto(times, output, order)` 写入调用方预分配矩阵，每个时间占一行。
`Segment::evaluateLocal()` 明确使用段内时间。内部断点归右段、末端归最后一段、域外外推；
时间回退时重新定位，短轨迹线性查找，较长轨迹二分查找。

`PPolyND<DIM, MaxCoefficients>` 的第二个参数是系数容量；实际系数数可更少。
样条暴露 `kDimension/kDegree/kCoefficientCount/kMinDerivativeOrder`。
系数按段、局部物理时间升幂排列。非法查询返回 NaN，超过次数的导数为零，负阶数抛 `invalid_argument`。
时间序列检查步长、区间与样本数溢出；零长度区间返回一个端点。

## 凸包与失败契约

`ConvexHullWorkspace<DIM>` 区分未配置、已配置和当前数值有效。
`configure(poly, HullOptions)` 允许分配和缓存同步；固定拓扑 `update/backwardAdd` 无库内分配和锁。
`matches()` 检查复用条件；不依赖 DIM 的不可变核可跨维共享。
反传会修改工作区，不能并发使用同一对象；输出梯度为累加。
源数值快照服务时间反传，工作区不借用源轨迹。凸包软惩罚不能替代最终可行性验收。

构造失败清除当前多项式。优化器可恢复失败返回轻量状态，评价失败返回正无穷及零梯度。
目标代价失败与轨迹数值有效性分别表达，读取接受轨迹以 `updateAccepted` 成功为准。
合法候选可以在失败后继续评价。梯度检查遇到非法扰动返回失败并恢复基准候选。
依赖这些失败语义时不能启用破坏 NaN/Inf 检查的 fast-math 选项。

## 构建与验证

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j3
(cd build && ctest --output-on-failure)
./build/basic_optimizer
./build/parameterized_optimizer
```

测试、示例、benchmark 分别由 `BUILD_TESTING/SPLINE_BUILD_EXAMPLES/SPLINE_BUILD_BENCHMARKS` 控制。
项目自有目标使用严格警告并按错误处理，告警选项不经 INTERFACE 传播给下游。
只有准确的第三方目录标为 SYSTEM，三个主库头文件保留检查。

[第二轮验证报告](docs/refactor_v2_validation.md) 说明本次环境、性能和限制；
[第一轮报告](docs/refactor_validation.md) 与[凸包历史性能](docs/convex_hull_performance_zh.md) 保留历史范围。
另见[凸包数学说明](docs/convex_hull_basis_zh.md)、[许可证](LICENSE) 与 [MINVO 来源](THIRD_PARTY_NOTICES.md)。
三个下游后续直接同步这三个英文注释的原始头文件，本轮只完成主库工作。
