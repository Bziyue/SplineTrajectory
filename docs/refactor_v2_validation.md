# 第二轮重构验证（2026-09-12）

本轮完成 SplineTrajectory 主库的接口和所有权重构。Optimizer 保持与冻结旧版接近的稳定评价速度，
使用端不再需要 EvaluateSpec、Context 或多项 cost 模板参数。公共分发仍为三个头文件。
SUPER、GCOPTER、diff-planner-swarm 本轮没有修改、同步或构建；后续工作见 [API 迁移表](api_migration.md)。

## 版本、环境和统计口径

- 冻结基线：`d5041355f08e28a9a62d088388f27af147d1a16c`，独立 detached worktree。
  第一轮已经完成的凸包分支合并和下游修复保留，不重新实现。
- 当前实现：包含本报告的第二轮重构提交；源头文件的最终 SHA-256 另见本页末尾。
- 主验证环境：Docker `ubuntu2004`，GCC 9.4.0、Eigen 3.3.7、CMake 3.16，C++17。
  宿主机 AMD Ryzen 9 7945HX；性能程序绑定 CPU 0，单线程，构建完成后才测量。
- 次工具链：宿主机 GCC 15 / Eigen 3.4，Release 测试和示例；没有把 Clang 列为已验证环境。
- 性能编译：`-std=c++17 -O3 -DNDEBUG -fno-fast-math`，没有启用 `-march=native`。
  正式受控构建额外启用 `-Wall -Wextra -Wpedantic -Werror`，不经 INTERFACE 传播给下游。
- 固定候选采用相同输入、16 步梯形积分、能量权重 0.01，比较 MINCO、手写新 Spline、手写旧 Spline、
  旧 Optimizer、新 Optimizer。旧 harness 只适配次数、导数阶数和记录开关，冻结头文件未修改。
- 每次程序运行测 15 组重复调用；CSV 的 P50 为组均值的中位数，P95 为最近秩法对应的第 15 组。
  **这些是批次平均耗时的分位数，不是单次调用尾延迟**。准备和首次调用为独立冷路径测量。
- 全矩阵先以每组 1500 次执行；随后对稳定评价做 5 轮交错复测，每组 500 次，轮换 old/new/direct 顺序。
  下列稳定评价主表使用 5 轮 P50 的中位数，降低频率变化和测量顺序的影响。

本机原始工件位于工作区 `.scratch/spline-v2/`，不随 Git 分发。仓库保留方法、结论和重现工具，
不把本机 CSV 路径当作跨电脑附件。旧历史报告见 [第一轮验证](refactor_validation.md)。

## 实现和正确性验收

| 内容 | 最终行为 / 验证 |
| --- | --- |
| Optimizer 模板 | `SplineOptimizer<Spline, Parameterization>`；普通 Objective 具名成员，按实际签名静态绑定 |
| 工作区 | 单对象按值拥有 maps、问题快照和缓冲；显式借用、禁止复制、移动后原对象未准备 |
| 评价流程 | 同一候选一次解码、一次构造，系数相关梯度一次通用伴随；计数回归直接检查调用次数 |
| 代价阶段 | duration/integral/sample/coefficient/parameter/decision；梯度累加和辅助变量完整链有限差分 |
| 样条存储 | PPoly 唯一拥有系数和断点，专用求解器内部填充；失败清空，复制保持值语义 |
| 顺序采样 | 独占 Sampler，首次无需预热，同段复用导数；删除外置 hint，支持预分配批量输出 |
| 凸包 | DIM 无关共享 Kernel；configure/update/backward 职责分开，反传显式修改 workspace |
| 输入契约 | 负导数、非法步长、空时间点、累计断点不可表示、系数溢出、失败恢复和 checked access |
| 借用契约 | 临时对象的 polynomial/Sampler 借用编译失败；源更新或移动后要求重新取得视图 |
| 接受解 | 求解器回调与最终接受解分开，`updateAccepted(x)` 重新构造接受向量 |

GCC 9 Release 全部受控目标构建通过；Debug 测试和示例、GCC 15 Release 测试和示例构建通过，
三种配置各 **11/11 CTest 通过**。基线为 **9/9 CTest 通过**。
新增测试覆盖动态路点参数维数、多面体映射写入、默认/自定义/借用映射、组合梯度、对象生命周期和失败行为。
MINCO 对照保留三/五/七次轨迹、能量、时长、路点和边界梯度；凸包还检查 DIM=1/2/6/10 的前向/伴随内积等式。

分配检查包括：Eigen 运行时禁止分配，以及对标准容器所用 C++ new/delete 的计数。
正常固定拓扑、预分配输出下，首次评价与后续评价通过；sample 开启/关闭切换、辅助变量、
凸包 `update/backwardAdd`、第一次样条反传均有覆盖。
这不是任意用户回调的全进程 malloc 拦截：用户 cost/map 自行分配、异常构造、重新准备、快照和诊断不在零分配保证内。

OpenMP 以 4 线程运行目标契约测试通过，积分串/并行结果一致。每个并发求解仍须独占 Optimizer 和可变 cost 工作区，
并行积分回调不能让异常逃出工作线程。

独立头文件和错误用法检查也启用严格警告。GCC 9 的实际诊断行数：

| 输入 | 诊断行数 |
| --- | ---: |
| 三个头分别独立 include | 各 0 |
| 不含支持阶段的 Objective | 12 |
| 错误 duration 签名 | 6 |
| 错误 integral 签名 | 7 |
| 不支持的积分导数阶数 | 7 |
| 临时 polynomial/Sampler 借用 | 各 9 |

负例检查命中预期阶段说明或 deleted 接口，没有靠隐藏回溯选项压缩输出。
另以高级参数化示例审计 `-Wconversion -Wshadow -Werror`，通过；不宣称这两个额外选项已覆盖所有历史 benchmark。

## 稳定评价、准备与记录

3 维、32 段，单位 µs/次。最后一列是 5 轮新实现批次 P95 的中位数。

| 次数 | 代价最高导数 | 旧 Optimizer P50 | 新 Optimizer P50 | 手写新 Spline P50 | 新 P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0 | 6.191 | 6.017 | 15.150 | 6.051 |
| 3 | 3 | 13.621 | 13.528 | 18.577 | 13.713 |
| 3 | 4 | 15.205 | 15.076 | 19.101 | 15.283 |
| 5 | 0 | 8.698 | 8.679 | 17.560 | 8.850 |
| 5 | 3 | 17.271 | 17.207 | 19.690 | 17.273 |
| 5 | 4 | 19.754 | 19.697 | 20.932 | 19.990 |
| 7 | 0 | 12.775 | 12.832 | 23.181 | 12.911 |
| 7 | 3 | 23.048 | 23.041 | 26.042 | 24.093 |
| 7 | 4 | 26.406 | 26.368 | 27.871 | 26.577 |

覆盖 2/8/32/128 段、3/5/7 次、0/3/4 阶代价的 36 个组合中，新/旧 P50 比为 **0.944–1.016**，
新/手写新 Spline 为 **0.397–0.995**。本轮没有发现稳定、显著的 Optimizer 抽象退化。
手写路径使用物理时间基，Optimizer 复用准备好的归一化积分基，因此后一项比较是完整评价路径收益，
不能解释成“单纯模板分发节省这些时间”，也不证明所有机器、所有 functor 都严格零纳秒开销。
上一轮局部 `flatten/noinline` 保留，不采用全局强制内联。

冷路径示例（七次、32 段、jerk，5 轮中位数）：准备旧 29.683 → 新 30.870 µs，
首次评价旧 26.121 → 新 24.934 µs。这里准备计时还包含问题 fixture 和初值/输出向量分配，
不等同于单独的 `prepare()` 函数耗时。首次反传所需的库缓冲已提前分配，不能把它当作隐藏预热。

记录能力另测旧/新、三种次数、四种段数。32 段 jerk、开启记录时的 P50 分别为：
三次 14.605/14.671 µs、五次 18.968/18.889 µs、七次 25.264/25.243 µs。
这些来自未交错的完整矩阵；不能拿不同轮次的关闭值相减来精确估计记录开销。
记录准备能力与当前是否记录已区分：后续无 sample 的目标且选项未要求记录时不执行记录循环，
`recordedSamples()` 在首次记录前、失败、非记录评价和 `updateAccepted()` 后返回空视图。

## 构造、反传和采样

内核基准覆盖 DIM=1/3、3/5/7 次、2/8/32/128 段，分别记录首次构造、稳定构造、反传和对象尺寸。
DIM=3、七次、128 段：构造 9.456 → 8.818 µs，反传 12.452 → 12.428 µs。
三次 32/128 段构造约慢 4%–5%，五次接近持平，七次受去除复制影响改善较明显。
输入与有限值校验保留；本轮不声称所有独立内核操作更快。反传计时含两版共有的输出 checksum 归约。

DIM=3 七次对象 `sizeof` 从 512 减到 472 字节；128 段去除了 24,576 字节的第二份系数和
1,032 字节的第二份断点。这里是确定移除的存储，不是整个进程峰值内存差：伴随工作区提前分配改变了冷路径内存时机。

采样以 7 轮 old/new 交错程序运行复测。32 段的中位数，单位 ns/点：

| 查询 | 顺序旧/新 | 随机旧/新 |
| --- | ---: | ---: |
| 分开求各导数 | 20.78 / 20.90 | 39.00 / 39.77 |
| 一次组合求值 | 20.66 / 20.60 | 36.51 / 36.15 |
| Sampler | 14.13 / 14.07 | 44.52 / 44.52 |
| 基准中的 fused Horner 实验 | 15.67 / 15.73 | 33.96 / 34.31 |

单轮曾出现 Sampler 异常变慢；热段生成代码对比和 7 轮复测未复现，最终采用交错结果。
Sampler 的优势在顺序同段复用，随机跳转重新准备导数可能比直接组合查询慢。
没有据此把普通查询统一改成 Sampler，也没有把 fused 实验引入正式实现。
动态容量 Sampler 只在构造时为实际需要的阶数/行数分配；固定容量的上述计时不包含这条动态构造路径。

## 凸包更新

原来的动态矩阵转换后整块扫描有限值，在新增契约检查后出现可重复开销。
最终改为逐控制点用固定 DIM 向量累加转换结果，并在写回前检查该向量，合并计算与有限值检查。
反向矩阵算法不变，源系数/时长快照保留。此表达式改写与公共存储重构分开记录。

完整基准覆盖分段数、导数、基、细分和次数；前向/反传有限差分、无分配及多维内积回归通过。
64 段、DIM=3、位置凸包、无细分的正式基准如下，单位 µs。每项为 9 批自适应重复的中位数。

| 次数/基 | update 旧/新 | backwardAdd 旧/新 |
| --- | ---: | ---: |
| 3 / Bezier | 1.733 / 1.663 | 1.990 / 2.005 |
| 3 / MINVO | 1.789 / 1.676 | 1.986 / 1.943 |
| 5 / Bezier | 3.065 / 2.743 | 3.244 / 3.368 |
| 5 / MINVO | 3.127 / 2.879 | 3.293 / 3.321 |
| 7 / Bezier | 4.933 / 4.271 | 4.930 / 5.103 |
| 7 / MINVO | 4.983 / 4.307 | 5.348 / 5.103 |

前向 update 在这些用例中比冻结旧版快约 4%–14%；反传有小幅正负波动，不能声称反传全面提速。
DIM=10 的局部对照也支持逐行实现，且多维正确性测试通过；完整性能表仍仅代表 DIM=3。

MINVO 0–7 次转换表及来源函数与冻结基线逐字节一致，已有许可证保留，Optimizer 补齐相同 MIT 文件头；核缓存只在 configure 阶段加锁。
配置缓存键改为完整四元组，避免字段打包碰撞；容量、索引、预算和数值溢出有明确拒绝行为。

## 完整求解与示例

L-BFGS 使用工作区 GCOPTER `9f999be4d4c277a1ef9ac2259b9890c66813d32b` 中的头文件。
各用例 7 次，mem=16、past=3、delta=1e-6、梯度阈值 1e-8、最多 500 次迭代。
旧/新在每个用例的迭代数、评价数、终止码、目标值相同；耗时单位 µs，取中位数。

| 次数/段数 | 旧/新耗时 | 迭代 | 评价/线搜索评价 | 终止码 | 目标值 | 梯度无穷范数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3/2 | 26.12 / 26.12 | 19 | 23/22 | 1 | 0.956428 | 7.65e-6 |
| 3/8 | 530.73 / 529.54 | 122 | 134/133 | 1 | 3.189248 | 0.0501 |
| 3/32 | 7969.23 / 8010.79 | 500 | 529/528 | -1008 | 46.171412 | 9.7153 |
| 5/2 | 42.74 / 37.99 | 23 | 27/26 | 1 | 1.375332 | 1.89e-4 |
| 5/8 | 763.44 / 758.69 | 149 | 155/154 | 1 | 4.457366 | 0.0185 |
| 5/32 | 10042.28 / 10029.22 | 500 | 536/535 | -1008 | 70.400470 | 62.5736 |
| 7/2 | 61.74 / 61.74 | 29 | 32/31 | 1 | 1.981028 | 1.78e-4 |
| 7/8 | 2678.57 / 2654.83 | 382 | 414/413 | 1 | 6.318988 | 0.6084 |
| 7/32 | 13364.38 / 13398.81 | 500 | 545/544 | -1008 | 143.265369 | 94.5095 |

`1` 表示 past/delta 目标变化条件触发，不能当作梯度阈值收敛；`-1008` 是达到最大迭代数，
**32 段三个用例没有收敛，不计为正确性成功案例**。这里保留它们用于比较相同求解路径的成本。

基础 optimizer 示例完成外部下降循环和接受解重建。高级参数化示例包含指数时间、受限空间坐标、
共享时长缩放、移动起点、积分及 Bezier 代价，完整梯度最大绝对误差 `2.56185e-9`。
凸包示例使用具名 C 回调用户数据；ESDF/corridor 梯度检查最大绝对误差分别 `4.00802e-6`、`1.97194e-9`。
两个示例求解都在 300 次预算处终止，终端梯度无穷范数约 `1.95e-4`、`3.08e-2`，
程序明确输出 `iteration_limit`，没有将软惩罚或有限迭代结果表述成可行性/收敛保证。

## 编译成本

干净编译七次 jerk Optimizer 基准翻译单元，关闭 L-BFGS 分支，旧/新各 3 次顺序执行。
该 TU 同时包含 MINCO 参考头，结果只代表这个实际 TU。使用编译器子进程 `wait4` 记录 RSS，
不是累计读取前一个进程的最大值。

| 指标 | 旧中位数 | 新中位数 |
| --- | ---: | ---: |
| 编译 wall time | 6.083 s | 5.893 s |
| 峰值 RSS | 734,928 KiB | 724,412 KiB |
| `.o` 文件大小 | 205,320 B | 158,736 B |

编译时间约减少 3.1%，目标文件约减少 22.7%。主要结构收益是公开类型和生命周期简化，
不能把这组 TU 数据外推成所有下游编译时间都会大幅减少。

## 重现步骤

宿主机 `optimizer/` 映射到容器 `/home/beiyue/ubuntu2004-shared/workspace/optimizer`。
先在宿主机工作区创建基线（worktree 内 `.git` 的宿主路径不能直接在容器解析，故显式传基线 revision）：

```sh
git -C SplineTrajectory worktree add --detach ../.scratch/spline-v2/baseline d5041355f08e28a9a62d088388f27af147d1a16c
docker exec -it -u beiyue ubuntu2004 bash
```

以下在容器内执行；已有构建路径时保持同一容器路径，不混用挂载别名。

```sh
cd /home/beiyue/ubuntu2004-shared/workspace/optimizer
cmake -S SplineTrajectory -B .scratch/spline-v2/build-release -DCMAKE_BUILD_TYPE=Release
cmake --build .scratch/spline-v2/build-release -j3
(cd .scratch/spline-v2/build-release && ctest --output-on-failure)

cmake -S SplineTrajectory -B .scratch/spline-v2/build-debug -DCMAKE_BUILD_TYPE=Debug -DSPLINE_BUILD_BENCHMARKS=OFF
cmake --build .scratch/spline-v2/build-debug --target tests examples -j3
(cd .scratch/spline-v2/build-debug && ctest --output-on-failure)

python3 SplineTrajectory/benchmarks/compare_backends.py \
  --baseline .scratch/spline-v2/baseline --baseline-revision d5041355f08e28a9a62d088388f27af147d1a16c \
  --output .scratch/spline-v2/comparison --mode all --jobs 3 --repeats 1500 --solve-repeats 7 \
  --lbfgs-header GCOPTER/gcopter/include/gcopter/lbfgs.hpp
python3 SplineTrajectory/benchmarks/measure_overhead.py evaluate --output .scratch/spline-v2/comparison
python3 SplineTrajectory/benchmarks/measure_overhead.py sampling --output .scratch/spline-v2/comparison
python3 SplineTrajectory/benchmarks/measure_overhead.py compile --output .scratch/spline-v2/comparison --baseline .scratch/spline-v2/baseline

taskset -c 0 .scratch/spline-v2/build-release/benchmark_convex_hull_basis .scratch/spline-v2/hull-final.csv
.scratch/spline-v2/build-release/basic_optimizer
.scratch/spline-v2/build-release/parameterized_optimizer
.scratch/spline-v2/build-release/convex_hull_optimization_demo .scratch/spline-v2/hull-demo
```

缺少 GCOPTER 求解器头时，传 `--solve-repeats 0` 可独立测库内评价；这不执行完整求解。
旧凸包基准用同样 CMake/Release 配置构建冻结 worktree 后执行其 `benchmark_convex_hull_basis`。
次工具链在宿主机配置 `-DCMAKE_CXX_COMPILER=/usr/bin/x86_64-linux-gnu-g++-15`、
`-DCMAKE_BUILD_TYPE=Release -DSPLINE_BUILD_BENCHMARKS=OFF`，随后构建 tests/examples 并运行 CTest。
OpenMP 独立构建 `test_optimizer_contracts.cpp`，增加 `-fopenmp`，以 `OMP_NUM_THREADS=4` 执行。
额外警告审计使用上述主工具链对 `examples/parameterized_optimizer.cpp` 执行
`-fsyntax-only -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`。

最终源头文件：

| 头文件 | SHA-256 |
| --- | --- |
| SplineTrajectory.hpp | `9d5c3eca996a13104e13bcc668cf73e50367db5cb53faf4713f0cd1c041106dc` |
| SplineOptimizer.hpp | `e0cbe1bf298adbd8022d02058be6fa0864409fd74cb73782c7ef72225348d5e9` |
| SplineConvexHull.hpp | `221a30e2247f6c71218d7e02e7228e802a60c820574f11f24b442da9bbb09c33` |
