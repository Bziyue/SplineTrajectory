# Bezier / MINVO 凸包表示性能报告

## 测试环境

- CPU：AMD Ryzen 9 7945HX，固定在 CPU 4
- 编译器：GCC 15.2.0
- 编译参数：`-O3 -march=native -ffast-math -DNDEBUG`
- Eigen，单线程
- 输入：`PPolyND<3>`，每段时长约 0.75–1.25 s
- 每项为 9 个批次的中位数；每批自动增加迭代次数至约 18 ms

原始数据位于
[`convex_hull_benchmark.csv`](convex_hull_benchmark.csv)，基准程序位于
[`benchmark_convex_hull_basis.cpp`](../benchmarks/benchmark_convex_hull_basis.cpp)。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark_convex_hull_basis -j2
taskset -c 4 ./build/benchmark_convex_hull_basis \
  docs/convex_hull_benchmark.csv
```

报告区分三种路径：

- one-shot：`fromPPoly()`，包含表示对象和输出内存的构造；
- update：固定拓扑后原地更新控制点，不重新分配；
- backwardAdd：向预分配的系数/时长梯度加和，不重新分配、不清零。

共享的不可变转换核已在预热阶段建立，因此不计入优化循环热路径。

## 不同段数

五次三维位置轨迹，深度为 0，单位为微秒：

| 段数 | Bezier one-shot | MINVO one-shot | Bezier update | MINVO update | Bezier backwardAdd | MINVO backwardAdd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.334 | 0.327 | 0.050 | 0.050 | 0.049 | 0.050 |
| 4 | 0.461 | 0.450 | 0.183 | 0.183 | 0.188 | 0.188 |
| 16 | 1.027 | 1.013 | 0.721 | 0.721 | 0.743 | 0.745 |
| 64 | 3.268 | 3.249 | 2.859 | 2.889 | 2.996 | 3.020 |
| 256 | 12.256 | 12.232 | 11.791 | 11.793 | 12.258 | 12.254 |
| 512 | 24.355 | 24.510 | 23.829 | 23.798 | 24.166 | 24.360 |
| 1024 | 49.178 | 49.037 | 47.854 | 48.172 | 48.726 | 49.667 |

16–1024 段区间内，原地位置更新约为 44–47 ns/段，反传约为
46–49 ns/段。Bezier 与 MINVO 使用同一个堆叠算子执行框架，所以性能几乎
一致。

## 获取指定阶导数表示

下面是 64 段五次三维轨迹的 one-shot 时间：

| 操作 | Bezier [µs] | MINVO [µs] |
| :--- | ---: | ---: |
| 位置，$r=0$ | 3.268 | 3.249 |
| 速度，$r=1$ | 2.519 | 2.508 |
| 加速度，$r=2$ | 1.926 | 1.917 |

单独生成幂基导数 `PPolyND::derivative(1/2)` 分别为 0.502 µs 和
0.369 µs。凸包转换并不先生成这个中间对象，而是把下降阶乘、递推得到的
$T^j$ 和共享转换核融合在一次段循环中。导数阶越高，控制点数越少。

`PPolyND` 的导数系数缓存也保留既有容量；同一拓扑仅更新系数值时，不再
清空并重新申请各阶导数矩阵，且只依赖阶数的下降阶乘表不会因系数变化
而失效。

## Bezier 与 MINVO 细分

64 段五次三维位置轨迹：

| 基 | 深度 | piece 数 | 控制点数 | one-shot [µs] | update [µs] | backwardAdd [µs] |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bezier | 0 | 64 | 384 | 3.269 | 2.860 | 2.999 |
| Bezier | 1 | 128 | 768 | 7.130 | 6.100 | 6.313 |
| Bezier | 2 | 256 | 1536 | 9.156 | 8.242 | 8.872 |
| Bezier | 3 | 512 | 3072 | 14.563 | 12.768 | 14.983 |
| Bezier | 4 | 1024 | 6144 | 24.593 | 21.681 | 28.464 |
| MINVO | 0 | 64 | 384 | 3.265 | 2.847 | 2.991 |
| MINVO | 1 | 128 | 768 | 6.665 | 6.154 | 6.362 |
| MINVO | 2 | 256 | 1536 | 9.062 | 8.721 | 9.039 |
| MINVO | 3 | 512 | 3072 | 14.276 | 12.645 | 14.835 |
| MINVO | 4 | 1024 | 6144 | 24.373 | 21.760 | 28.575 |

细分不再在每次调用时构造 de Casteljau 树。拓扑建立时把每个叶片的仿射
限制矩阵和 Bezier/MINVO 基矩阵合成一个共享稠密算子 $K$；热路径只有
$KZ$，反向只有 $K^TG$。这也使 MINVO 获得与 Bezier 相同的任意二分能力。

## 3/5/7 次特化

64 段三维位置轨迹，不细分：

| 次数 | 基 | one-shot [µs] | update [µs] | backwardAdd [µs] |
| ---: | :--- | ---: | ---: | ---: |
| 3 | Bezier | 1.929 | 1.623 | 1.737 |
| 3 | MINVO | 1.933 | 1.639 | 1.736 |
| 5 | Bezier | 3.374 | 2.943 | 3.125 |
| 5 | MINVO | 3.357 | 2.855 | 2.971 |
| 7 | Bezier | 5.282 | 4.676 | 4.535 |
| 7 | MINVO | 5.188 | 4.676 | 4.543 |

统一接口
`MinDerivativeSplineND<DIM, S>` 在编译期分别选择现有的 cubic、quintic、
septic 专用求解器；因此没有运行时分支或虚函数开销。三类求解器现在都
直接原位填充持久化系数矩阵，优化器在 `prepareContext()` 中预热固定段数
的求解器，并复用能量、采样全局时间和 auxiliary 梯度缓冲。

测试还使用 `EIGEN_RUNTIME_NO_MALLOC` 在 warm-up 后关闭 Eigen 动态分配，
连续执行 quintic `update()`、深度 2 Bezier `update()`、
`backwardAdd()` 和 `propagateGrad()`；该组合热路径通过无分配断言。

## 复杂度

令源段数为 $N$，导数后每段控制点数为 $q=m+1$，维数为 $D$，细分深度为
$s$，叶片数为 $L=2^s$。共享核的堆叠矩阵为

$$
K\in\mathbb R^{Lq\times q}.
$$

于是：

- `update()`：$O(NLq^2D)$；
- `backwardAdd()`：$O(NLq^2D)$；
- 输出控制点工作区：$O(NLqD)$；
- 每种拓扑共享核：$O(Lq^2)$；
- 时长幂使用乘法递推，不调用 `pow()`。

`resetTopology()` 按实际矩阵和工作区字节数检查内存预算，而不是使用固定
的细分深度上限。正常优化应在准备阶段调用一次，迭代中只调用
`update()`/`backwardAdd()`。
