# 样条重构验证（2026-09-12）

本次整合凸包分支和 diff-planner-swarm 的通用数值修复，保留三个公共头文件，新增独立顺序采样器，
并重构静态代价绑定、按需导数、归一化积分基和显式起始时间反传。结果以本页记录的版本、输入与环境为准。

## 版本与环境

| 用途 | 固定基线 |
| --- | --- |
| 原主库 | SplineTrajectory `5dde128` |
| 凸包功能 | `feature/convex-hull-bases` / `fef5574`，保留原 5 个提交的历史 |
| 下游数值修复 | diff-planner-swarm `cdbb987`，包含 `1616ced` / `31f109b` |
| SUPER 当前集成基线 | `a608fb1` |
| GCOPTER 当前集成基线 | `766d390` |
| 原 MINCO | GCOPTER `e0444f6`；另保留 SUPER `2ad3419` worktree 供历史核对 |

Ubuntu 20.04 Docker `ubuntu2004`，GCC 9.4.0、Eigen 3.3.7、C++17、ROS Noetic；
CPU 为 AMD Ryzen 9 7945HX。独立比较统一使用 `-O3 -DNDEBUG -fno-fast-math`，未开启 `-march=native`。
测量串行绑定 CPU 0，在构建结束后执行。ROS 验证使用隔离 master `127.0.0.1:11417`，无飞行或碰撞仿真。

宿主机工作区 `/home/beiyue/Docker-WorkSpace/ubuntu2004-shared/workspace/optimizer`
在容器中为 `/home/beiyue/ubuntu2004-shared/workspace/optimizer`。
原始日志、逐轮 CSV、冻结 worktree、诊断探针与优化归档保存在工作区 `.scratch/spline-refactor/`，
这些是本机工件；跨电脑复现使用本页命令及已纳入版本控制的 benchmark 源码。
下游 VERSION.md 记录本次公共头文件来源提交和 SHA-256。

## 退化原因和处理

模板绑定本身不会消除运行路径中的工作。本次找到并处理了：

- 凸包系数代价原来的 `void*` / 函数指针擦除：改成具体类型借用和直接静态调用。
- 配置镜像、重复解码、无效代价循环：准备上下文冻结配置，每次候选解码构造一次，禁用代价用显式 Void 类型。
- 每个积分点反复构造时间幂和完整高阶状态：准备归一化积分基，每段只缩放一次系数，按代价所需阶数计算状态。
  时间漂移保留下一阶导数，不能因代价仅依赖 jerk 就删除 snap。
- 空间映射返回动态临时梯度：增加 `backwardGradInto`，下游直接写决策梯度切片。
- 编译器丢失内联：在实际 GCOPTER 固定候选中，结构化接口使 GCC 9 将平坦性反传拆出积分循环。
  对外层代价函数单独强制内联收益很小；对内部积分函数局部使用 GCC/Clang `flatten` 后，
  诊断中位数从约 7.8 μs 降至 7.0 μs，旧版约 7.4 μs。其他编译器保留标准 C++ 路径，性能须实测。

同时修复两处梯度契约：SUPER 多面体域惩罚曾清零并覆盖已有物理梯度，现改为累加，旧版运行新增回归会失败；
移动起点的积分、跨样本与系数显式时间偏导统一进入辅助映射，删除 diff-planner-swarm 外部重复补算。
前一修复会改变求解路径，因此不能将完整求解耗时变化全部解释成单次评价提速。

## 采样

`Sampler<N>` 借用只读源轨迹，只在切换段时准备所需导数系数，同段采样复用。
源更新、赋值或移动后重建 sampler；每个遍历者独占其状态。源的首次 const 读取不需要预热。
固定阶数构造/循环无堆分配，动态阶数在 sampler 构造时分配当前段存储。

七次、三维、32 段，每组 20,000 点，21 轮；下表为每点耗时中位数。所有路径使用相同多项式和查询时间。

| 版本 / 方法 | 顺序 ns/点 | 随机 ns/点 |
| --- | ---: | ---: |
| 旧 main 分别求 P/V/A/J | 30.34 | 46.36 |
| 旧 diff 组合求值 | 20.48 | 36.15 |
| 新组合求值 | 20.60 | 36.15 |
| 新 sampler | 14.25 | 44.64 |
| 比较用联合 Horner | 15.79 | 34.31 |

联合 Horner 为 benchmark 内的独立比较实现。按段准备导数的方案在顺序遍历中更快，因此选作 sampler；
随机跳转会反复准备段系数，推荐普通组合求值。旧 main 的普通查询包含共享导数缓存行为，不能据其稳定性能
推导首次并发读取安全性。CSV 单独保留首次调用耗时；表中为稳定循环，未把首次成本计入其中。

## 构造与反传内核

`test_Grad 5 10000`，固定种子和 5 段，7 轮、每轮 10,000 次，下表为均值，单位 μs；
构造与传播分别测量，梯度输入在传播循环前固定。它衡量内核，不能替代完整优化器比较。

| 次数 | MINCO 构造 | Spline 构造 | MINCO 反传 | Spline 反传 |
| --- | ---: | ---: | ---: | ---: |
| 3 | 0.302 | 0.128 | 0.187 | 0.113 |
| 5 | 0.641 | 0.183 | 0.399 | 0.238 |
| 7 | 1.114 | 0.344 | 0.657 | 0.441 |

## 固定候选评价

`benchmark_optimizer.cpp` 使用同一组时间、路点、边界、变量排列、二次 P/V/A/J 代价、能量权重 0.01，
每段 16 个梯形区间。手写 MINCO / 手写 Spline 与两个旧框架、新框架共用输入和代价公式。
每个规模 15 轮，每轮 2,000 次；下表为单次代价和完整梯度评价中位数，单位 μs。

| 次数 / 段数 | 手写 MINCO | 手写 Spline | 旧 main | 旧下游 | 新优化器 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 / 2 | 1.366 | 1.194 | 1.446 | 1.253 | 1.152 |
| 5 / 8 | 5.577 | 4.853 | 5.292 | 4.885 | 4.353 |
| 5 / 32 | 22.640 | 19.218 | 20.445 | 19.298 | 17.104 |
| 5 / 128 | 90.678 | 76.799 | 81.496 | 76.868 | 68.297 |
| 7 / 2 | 1.885 | 1.560 | 1.846 | 1.660 | 1.555 |
| 7 / 8 | 7.428 | 6.302 | 6.739 | 6.558 | 5.909 |
| 7 / 32 | 30.324 | 24.990 | 26.133 | 25.702 | 23.249 |
| 7 / 128 | 159.824 | 99.993 | 103.803 | 102.552 | 92.023 |

各路径固定候选的代价和梯度校验和通过相对 1e-8 检查；完整梯度由 MINCO/有限差分回归独立验证。
CSV 同时报告准备、首次调用以及批均值的 P50/P95，首次两项为单次观测，不能用于细小差异判断。
新版本的准备阶段承担合法性验证和积分基预计算；短生命周期问题应同时考虑准备与首次评价成本。

## 完整求解

共享求解器实验使用同一 GCOPTER L-BFGS，memory=16、past=3、delta=1e-6、g_epsilon=1e-8、
最多 500 次迭代；每组 11 次。CSV 记录总耗时、评价数、迭代数、线搜索评价数、结束码、代价和梯度无穷范数。

| 次数 / 段数 | 手写 MINCO μs / 评价数 | 手写 Spline μs / 评价数 | 旧下游 μs / 评价数 | 新优化器 μs / 评价数 |
| --- | ---: | ---: | ---: | ---: |
| 5 / 2 | 42.74 / 27 | 37.99 / 27 | 43.93 / 27 | 37.99 / 27 |
| 5 / 8 | 946.29 / 156 | 823.99 / 155 | 846.55 / 156 | 764.63 / 155 |
| 7 / 2 | 70.05 / 32 | 62.93 / 32 | 65.30 / 32 | 60.55 / 32 |
| 7 / 8 | 2988.46 / 377 | 2752.18 / 402 | 2946.90 / 420 | 2666.70 / 414 |

32 段压力场景所有版本均达到 500 次迭代限制（-1008），不能列作收敛成功或按最终代价等价比较。
这些数据保留为相同预算下的压力结果。2/8 段场景满足同一相对代价终止条件；浮点运算顺序可能改变非线性
求解的评价数与终点，固定候选评价表更适合归因框架成本。

三个实际消费项目使用以下代表性问题：

- GCOPTER：三组重叠箱形走廊端点扰动，物理参数和权重取当前 global_planning 配置，21 次/组。
  旧 MINCO 的几何头只有比较器缺少 const、不能直接用 GCC 9 的 C++17 编译；测量统一预包含现行几何头，
  该文件与旧版的差异仅为比较器的 const。内部几何 LP 的固定种子 RNG 在重复 setup 间推进，各版本使用相同调用序列。
- SUPER：现有 `backend_equivalence`，扰动 -1/0/+1、accuracy=5e-6、11 次/组，使用原稳定性门限；
  耗时取库内部 optimize 区间，排除 ROS 进程启动。历史 MINCO 完整 SUPER 业务接口没有混入这个 A/B。
- diff-planner-swarm：现有 optimization_dataset_benchmark，2/8/16 段、31 次，分别验证关闭记录、同步记录、
  异步记录；比较关闭记录的 solve_ms。归档时的顺序采样也经过同步/异步完整路径。

| 项目 / 用例 | 旧版中位耗时 μs | 新版中位耗时 μs | 耗时变化 | 评价数（旧 → 新） |
| --- | ---: | ---: | ---: | --- |
| GCOPTER / -1 | 371.63 | 344.32 | -7.3% | 未对业务求解器插桩 |
| GCOPTER / 0 | 423.87 | 400.12 | -5.6% | 未对业务求解器插桩 |
| GCOPTER / 1 | 370.44 | 346.69 | -6.4% | 未对业务求解器插桩 |
| SUPER / -1 | 1429.52 | 1477.01 | +3.3% | 154 → 164 |
| SUPER / 0 | 1688.35 | 1561.31 | -7.5% | 187 → 173 |
| SUPER / 1 | 2226.20 | 2248.76 | +1.0% | 250 → 260 |
| DIFF / 2 | 106.86 | 94.98 | -11.1% | 16 → 16 |
| DIFF / 8 | 1370.15 | 1237.18 | -9.7% | 60 → 60 |
| DIFF / 16 | 8672.11 | 7747.19 | -10.7% | 188 → 188 |

GCOPTER 三组较旧集成快约 6%–7%；原 MINCO 同组中位数为 369.25 / 427.43 / 370.44 μs，均返回有限代价且端点检查通过。业务接口的公开代价是否包含范数惩罚有历史差异；严格相同代价的比较以共享求解器实验为准。
SUPER 单次评价改善与梯度修复带来的评价次数变化叠加，总耗时有快有慢，不能宣称所有场景加速。
diff-planner-swarm 的三组评价数及最终代价与基线一致，关闭归档时中位耗时下降约 10%–11%。
三个规划器的原有成功、稳定性或归档验收均通过，本次代表性场景未发现整体系统性退化。

## 正确性与构建

- 主库 Debug 和 Release：各 9/9 CTest，通过系数/能量/边界/伴随梯度、掩码、凸包反传、组合显式起始时间梯度。
- 采样器：固定与动态阶数、1/2/3 维、内部断点、端点、外推、重复时间、回退、独立首次并发读取；
  独立幂级数参考校验位置至五阶导数，固定/动态阶数循环禁用 Eigen 分配仍通过。
- 数值失败：无效系数、时间溢出、累计断点重合、失败后恢复；Release 不依赖断言，评价返回失败/+infinity/零梯度。
- diff-planner-swarm：25/25 CTest，含数值失败的 NDEBUG 回归、辅助映射、Bezier 代价、接续与归档重放。
- Noetic：SUPER backend_equivalence、GCOPTER global_planning、diffswarm_planner_node 和 codec 目标构建通过；
  两个空间映射回归、SUPER 稳定性检查和 ROS codec 运行通过。SUPER 修正了干净构建中缺失消息生成依赖的问题。
- `test_Grad` 默认只执行短回归，额外参数显式启用性能循环；随机输入固定种子，失败返回非零状态。
  不再让 Debug CTest 隐式执行数万次性能循环。

未运行飞行、GPU 渲染或 PCD 碰撞仿真；上述结论覆盖数值库、代表性优化问题和 ROS 构建/编解码链路。

## 复现

在工作区创建固定版本的 detached worktree，保持现有工作目录不变：

```sh
git -C SplineTrajectory worktree add --detach ../.scratch/spline-refactor/spline-main 5dde128
git -C GCOPTER worktree add --detach ../.scratch/spline-refactor/gcopter-minco e0444f6
git -C GCOPTER worktree add --detach ../.scratch/spline-refactor/gcopter-current 766d390
git -C SUPER worktree add --detach ../.scratch/spline-refactor/super-current a608fb1
git -C diff-planner-swarm worktree add --detach ../.scratch/spline-refactor/diff-current cdbb987
```

容器中统一使用映射后的工作区路径，并显式加载 `/opt/ros/noetic/setup.bash`。
普通主库构建见 README；Debug 将 CMAKE_BUILD_TYPE 改为 Debug。

```sh
python3 SplineTrajectory/benchmarks/compare_backends.py --mode all
# Kernels: construction and adjoints are timed separately on identical data.
taskset -c 0 .scratch/spline-refactor/build-spline-release/test_Grad 5 10000
```

`compare_backends.py` 构建四条实现路径，并额外区分旧主库与旧下游头文件。编译完成后才开始单核运行。
可用 `--mode build` / `--mode run` 分开构建与测量；已有基线不可用时明确报错，不修改仓库历史。

实际 GCOPTER 基准编译示例（old 替换为对应 worktree/include；MINCO 加 `-DBENCH_MINCO`）：

```sh
g++ -std=c++17 -O3 -DNDEBUG -fno-fast-math -I/usr/include/eigen3 \
  -include "$PWD/GCOPTER/gcopter/include/gcopter/geo_utils.hpp" \
  -IGCOPTER/gcopter/include GCOPTER/gcopter/tests/spline_solve_benchmark.cpp \
  -o .scratch/spline-refactor/gcopter-solve-new
```

catkin-baseline/src 链接冻结 SUPER 的 super_planner、rog_map、mars_quadrotor_msgs 与冻结 GCOPTER 的 gcopter；
catkin-current/src 链接现行相同包，以及 diff-planner-swarm 的 diffswarm_ros1、diffswarm_msgs。
先生成 baseline 的 `quadrotor_msgs_generate_messages_cpp`，再构建其 `backend_equivalence global_planning`；
当前树直接构建同名目标和 `diffswarm_planner_node diffswarm_swarm_trajectory_codec_test`
以及 `super_spline_map_regression gcopter_spline_map_regression`。均为 Release、`-j2`。

分别在 build-diff-baseline / build-diff-current 构建冻结/现行 diff-planner-swarm 的
`optimization_dataset_benchmark` 目标。启动隔离 ROS master 并运行：

```sh
ROS_IP=127.0.0.1 ROS_HOSTNAME=127.0.0.1 roscore -p 11417
# In another shell after sourcing Noetic:
python3 SplineTrajectory/benchmarks/compare_planners.py
```

此脚本要求三组 GCOPTER 基准二进制及上述构建目录已存在；所有原始输出留在本机 scratch。
重复执行前应选择新的输出工作目录或归档上轮 DiffSwarm 数据集目录，防止把旧归档当作本轮结果。
