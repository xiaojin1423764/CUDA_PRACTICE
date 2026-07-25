# Softmax 优化实验记录

本文记录 [code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu) 的当前性能分析、CUB baseline 对比，以及后续 blocks 参数扫描和 row-wise softmax 实验。

## 当前实现

当前一维 softmax 每轮启动 5 个 kernel：

```text
max_first_pass_kernel
max_final_pass_kernel
exp_sum_kernel
sum_final_pass_kernel
normalize_kernel
```

测试规模：

```bash
./bin/Softmax 500000 200 20
```

一次代表性结果：

```text
n=500000 output_sum=0.999999919 max_abs_error=3.637979e-12 max_rel_error=4.254749e-07
softmax avg_us=52.877 bandwidth_gbps=189.12 partial_blocks=420
```

## 当前 NCU 结论

来自 [softmax_500k_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/softmax_500k_ncu_detailed.ncu-rep)。

| Kernel | Grid x Block | Duration | Memory Throughput | DRAM Throughput | SM Throughput | Achieved Occupancy |
|---|---:|---:|---:|---:|---:|---:|
| `max_first_pass_kernel` | `420 x 256` | `6.56 us` | `320.23 GB/s` | `35.82%` | `8.96%` | `70.40%` |
| `max_final_pass_kernel` | `1 x 256` | `2.30 us` | `9.11 GB/s` | `1.05%` | `0.05%` | `15.11%` |
| `exp_sum_kernel` | `420 x 256` | `7.30 us` | `288.56 GB/s` | `32.45%` | `9.49%` | `72.66%` |
| `sum_final_pass_kernel` | `1 x 256` | `2.40 us` | `12.05 GB/s` | `1.35%` | `0.04%` | `16.76%` |
| `normalize_kernel` | `420 x 256` | `5.06 us` | `428.10 GB/s` | `47.87%` | `9.41%` | `92.01%` |

当前判断：

- `exp_sum_kernel` 是单个耗时最高的 kernel。
- `max_final_pass_kernel` 和 `sum_final_pass_kernel` 是结构性低利用率 kernel，因为只启动 `1 x 256`。
- `normalize_kernel` 相对健康，occupancy 最高。
- 所有 kernel 的 local memory spilling 为 0。
- 主路径 kernel branch efficiency 接近 100%，分支发散不是主要瓶颈。

## CUB Baseline

CUB baseline 文件：[code/lib_Softmax.cu](/home/xj/advanced_cuda/code/lib_Softmax.cu)

它使用：

```text
CUB DeviceReduce::Max
exp_shift_kernel
CUB DeviceReduce::Sum
normalize_kernel
```

计时区不包含 `cudaMalloc/cudaFree`、CUB temp storage 查询/分配和 H2D/D2H 拷贝。

同规模运行：

```bash
./bin/lib_Softmax 500000 200 20
```

一次代表性结果：

```text
n=500000 output_sum=0.999999919 max_abs_error=3.637979e-12 max_rel_error=4.254749e-07
cub softmax avg_us=56.673 bandwidth_gbps=176.45 blocks=420 temp_bytes=8703
```

当前判断：

- 手写版和 CUB baseline 在 `n=500000` 下性能接近。
- CUB baseline 一轮实际是 6 个 kernel，因为 Max 和 Sum 各拆成大 reduction + single-tile final reduction。
- 手写版一轮是 5 个 kernel，但 reduction 内核本身不如 CUB 成熟。
- 当前瓶颈不只是单个 reduction 写法，而是全局 softmax 的多阶段同步结构。

## 下一步实验

### 1. Blocks 参数扫描

目标：找出 `blocks` 对一维全局 softmax 的影响。

候选值：

```text
70, 140, 280, 420, 560
```

这些值对应当前 70 SM GPU 的：

```text
SM * 1, SM * 2, SM * 4, SM * 6, SM * 8
```

关注：

- `avg_us`
- `bandwidth_gbps`
- correctness error
- final reduction 的 partial 数量变化

实现方式：

[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu) 已新增第 4 个可选参数：

```bash
./bin/Softmax [n] [benchmark_iters] [warmup_iters] [forced_blocks]
```

例如：

```bash
./bin/Softmax 500000 200 20 140
```

表示强制使用 `blocks=140`，而不是自动选择的 `420`。

### Blocks 扫描结果

顺序运行命令：

```bash
for b in 70 140 280 420 560; do
  for r in 1 2 3; do
    ./bin/Softmax 500000 200 20 $b
  done
done
```

结果摘要：

| blocks | Run 1 | Run 2 | Run 3 | Best | Median |
|---:|---:|---:|---:|---:|---:|
| 70 | `43.027 us` | `51.585 us` | `97.360 us` | `43.027 us` | `51.585 us` |
| 140 | `45.630 us` | `119.640 us` | `95.211 us` | `45.630 us` | `95.211 us` |
| 280 | `48.101 us` | `51.372 us` | `113.287 us` | `48.101 us` | `51.372 us` |
| 420 | `126.653 us` | `55.710 us` | `51.440 us` | `51.440 us` | `55.710 us` |
| 560 | `47.358 us` | `84.367 us` | `51.616 us` | `47.358 us` | `51.616 us` |

观察：

- 运行波动很大，说明 `n=500000` 的 kernel 很短，容易受 GPU 时钟、系统状态和其他图形/显示负载影响。
- 从 best case 看，`70/140/280/560` 都出现过 `43-48 us` 的较快结果。
- 从 median 看，除 `140` 的异常波动外，`70/280/420/560` 都大致在 `51-56 us`。
- 因此当前不能说某个 blocks 值稳定压倒其他值。

为了看结构差异，额外采集了 `blocks=70` 的 NCU：

- [softmax_500k_blocks70_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/softmax_500k_blocks70_ncu_detailed.ncu-rep)

关键指标对比：

| Kernel | blocks=70 Duration | blocks=420 Duration | 说明 |
|---|---:|---:|---|
| `max_first_pass_kernel` | `5.12 us` | `6.56 us` | blocks 少时 partial 数少，第一阶段略快 |
| `max_final_pass_kernel` | `1.98 us` | `2.30 us` | partial 从 420 降到 70，final max 更轻 |
| `exp_sum_kernel` | `8.38 us` | `7.30 us` | blocks 少时每个 block 处理更多元素，变慢 |
| `sum_final_pass_kernel` | `1.95 us` | `2.40 us` | partial 更少，final sum 更轻 |
| `normalize_kernel` | `13.82 us` | `5.06 us` | blocks=70 并行度不足，normalize 明显变慢 |

结论：

- 减少 blocks 确实能降低 final reduction 的固定开销。
- 但 blocks 太少会让主路径 kernel 并行度不足，尤其是 `normalize_kernel` 明显变慢。
- 对当前 `n=500000`，`blocks=420` 虽然 final reduction 多归约一些 partial，但整体并行度更均衡。
- blocks 扫描的主要价值是暴露 tradeoff：`partial 数量` 和 `主 kernel 并行度` 不能只优化一边。

### 2. Row-Wise Softmax

目标：验证真实 attention 场景中更常见的 row-wise softmax 是否更适合 GPU。

一维全局 softmax 需要跨所有元素做全局 max 和 sum，因此天然需要多阶段全局同步。Row-wise softmax 可以让一个 block 负责一行，把 max、sum、normalize 限制在 block 内完成，减少全局同步。

计划测试：

```text
rows = 512
cols = 1024
total elements = 524288
```

该规模和 `n=500000` 接近，便于粗略比较。

实现文件：

- [code/RowWiseSoftmax.cu](/home/xj/advanced_cuda/code/RowWiseSoftmax.cu)

编译：

```bash
nvcc -O3 -arch=sm_80 -lineinfo \
  code/RowWiseSoftmax.cu \
  -o bin/RowWiseSoftmax
```

运行：

```bash
./bin/RowWiseSoftmax 512 1024 200 20
```

结果：

```text
rows=512 cols=1024 elements=524288
max_row_sum_error=1.226158e-08
max_abs_error=9.313226e-10
max_rel_error=5.235603e-07
rowwise softmax avg_us=11.776 bandwidth_gbps=890.47
```

NCU 报告：

- [rowwise_softmax_512x1024_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/rowwise_softmax_512x1024_ncu_detailed.ncu-rep)

NCU 关键指标：

| Kernel | Grid x Block | Duration | Memory Throughput | DRAM Throughput | SM Throughput | Achieved Occupancy |
|---|---:|---:|---:|---:|---:|---:|
| `rowwise_softmax_kernel` | `512 x 256` | `8.54 us` | `311.55 GB/s` | `34.90%` | `23.60%` | `70.70%` |

和一维全局 softmax 对比：

| Version | Elements | Kernel Count | Representative Avg |
|---|---:|---:|---:|
| Global softmax | `500000` | `5` | `~50-60 us` 常见区间 |
| Row-wise softmax | `524288` | `1` | `11.776 us` |

Row-wise 版本快很多的原因：

- 一个 block 处理一行，max、sum、normalize 都在 block 内完成。
- 不需要 `max_final_pass_kernel` 和 `sum_final_pass_kernel`。
- 不需要在多个 kernel 之间做全局同步。
- 一轮只启动一个 kernel，而不是 5 个 kernel。
- 当前 `512 x 1024` 的 grid 有 512 个 block，比 `blocks=70` 更容易给 70 个 SM 持续供给工作。

重要结论：

一维全局 softmax 是不利于 GPU 的形态，因为全数组 max 和 sum 天然需要全局同步。Transformer attention 中常见的 row-wise softmax 更适合 GPU，因为每行 softmax 的归约范围有限，可以把同步限制在 block 内。

## 阶段结论

当前最重要的结论不是“哪个 blocks 数字绝对最好”，而是：

```text
一维全局 softmax 的主要瓶颈来自多阶段全局同步和短 kernel 固定开销。
```

blocks 参数扫描说明：

```text
减少 blocks 可以降低 final reduction 开销，但会降低主 kernel 并行度；
增加 blocks 可以提高主 kernel 并行度，但会增加 partial 数和 final reduction 开销。
```

row-wise softmax 说明：

```text
改变问题组织方式比微调一维全局 softmax 的 blocks 更有效。
```

后续如果目标是 attention/LLM 算子，应该优先继续优化 row-wise softmax，而不是在一维全局 softmax 上过度调参。

## Row-Wise Shared-Memory 变体

进一步尝试了一个 shared-memory row-wise 版本：

- 文件：[code/RowWiseSoftmax.cu](/home/xj/advanced_cuda/code/RowWiseSoftmax.cu)
- `use_shared=0`：原始 row-wise 版本，`exp(x - max)` 先写入 global output，再原地 normalize。
- `use_shared=1`：shared-memory 版本，`exp(x - max)` 先写入 shared memory，最后只把 normalize 后的结果写一次 global output。

运行方式：

```bash
./bin/RowWiseSoftmax 512 1024 200 20 0
./bin/RowWiseSoftmax 512 1024 200 20 1
```

复测结果：

| Mode | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| `global` | `11.481 us` | `10.442 us` | `19.665 us` | `11.481 us` |
| `shared` | `28.052 us` | `22.389 us` | `25.911 us` | `25.911 us` |

NCU 报告：

- global 版本：[rowwise_softmax_512x1024_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/rowwise_softmax_512x1024_ncu_detailed.ncu-rep)
- shared 版本：[rowwise_softmax_shared_512x1024_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/rowwise_softmax_shared_512x1024_ncu_detailed.ncu-rep)

NCU 关键指标：

| Mode | Kernel | Duration | Memory Throughput | DRAM Throughput | SM Throughput | Achieved Occupancy |
|---|---|---:|---:|---:|---:|---:|
| `global` | `rowwise_softmax_kernel` | `8.54 us` | `311.55 GB/s` | `34.90%` | `23.60%` | `70.70%` |
| `shared` | `rowwise_softmax_shared_kernel` | `11.26 us` | `223.34 GB/s` | `24.95%` | `17.94%` | `72.30%` |

结论：

- shared-memory 版本没有变快，反而更慢。
- 虽然 shared 版本少了一次 global output 的读写，但它增加了 shared memory 写入/读取和额外同步。
- 对 `cols=1024` 这种规模，global 版本的中间结果很可能还能受 cache/L2 帮助，额外 shared memory 路径不划算。
- NCU 也显示 shared 版本 duration 更长，memory throughput 和 SM throughput 都更低。

当前建议：

```text
保留 global row-wise 版本作为当前 row-wise baseline。
不要把 shared-memory 暂存 exp 作为下一步主优化方向。
```

更值得继续的方向：

- 做 warp-level row-wise softmax，适合 `cols <= 1024` 的 attention 行。
- 对 `cols` 扫描，例如 `128/256/512/1024/2048/4096`。
- 针对 attention 场景做 masked row-wise softmax。
- 将 scale、mask、softmax 融合到一个 kernel。
