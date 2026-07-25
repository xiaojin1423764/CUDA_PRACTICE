# Softmax.cu Nsight Systems / Nsight Compute 分析

分析对象：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu)

测试环境：

- GPU：NVIDIA GeForce RTX 5070 Ti
- SM 数量：70
- CUDA 编译工具：CUDA 13.0
- 编译命令：

```bash
nvcc -O3 -arch=sm_80 -lineinfo code/Softmax.cu -o bin/Softmax
```

测试规模：

```bash
./bin/Softmax 500000 100 10
```

正确性结果：

```text
n=500000
output_sum=0.999999919
max_abs_error=3.637979e-12
max_rel_error=4.254749e-07
partial_blocks=420
```

非 profiler 下程序自带 CUDA event 计时存在波动，本次观察到约 `48.9 us` 到 `79.3 us`。后续 kernel 级分析主要以 Nsight Compute 单 kernel 指标为准。

## 代码结构

当前 softmax 每轮 benchmark 启动 5 个 kernel：

1. `max_first_pass_kernel`：读取输入，做 block 级 max reduction，输出 partial max。
2. `max_final_pass_kernel`：单 block 归约所有 partial max，得到全局 max。
3. `exp_sum_kernel`：计算 `exp(x - max)` 写入 output，同时做 block 级 sum reduction。
4. `sum_final_pass_kernel`：单 block 归约所有 partial sum，得到全局分母。
5. `normalize_kernel`：将 output 中的 exp 分子除以全局分母。

对应代码位置：

- `max_first_pass_kernel`：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu:97)
- `max_final_pass_kernel`：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu:132)
- `exp_sum_kernel`：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu:154)
- `sum_final_pass_kernel`：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu:197)
- `normalize_kernel`：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu:214)
- benchmark 循环：[code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu:257)

## Nsight Systems

采集命令：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o reports/softmax_500k_nsys \
  ./bin/Softmax 500000 20 5
```

生成文件：

- [softmax_500k_nsys.nsys-rep](/home/xj/advanced_cuda/reports/softmax_500k_nsys.nsys-rep)
- [softmax_500k_nsys.sqlite](/home/xj/advanced_cuda/reports/softmax_500k_nsys.sqlite)

CUDA API 摘要：

```text
cudaMalloc:        4 calls, 190.744 ms total, 95.3% CUDA API time
cudaMemcpy:        2 calls,   3.303 ms total,  1.6% CUDA API time
cudaLaunchKernel: 125 calls,  2.932 ms total,  1.5% CUDA API time
cudaFree:          4 calls,   0.793 ms total,  0.4% CUDA API time
```

结论：

- `nsys` 统计的是完整程序 host 侧 CUDA API 时间，`cudaMalloc` 占比最高。这是因为 `main` 中输入、输出和 workspace 都在 benchmark 前后分配释放。
- benchmark 内部已经复用 workspace，所以程序打印的 `avg_us` 不包含 `cudaMalloc/cudaFree`。
- 采集命令使用 `20` 次 benchmark 和 `5` 次 warmup。每轮 5 个 kernel，所以 `cudaLaunchKernel` 共 `25 * 5 = 125` 次，和报告一致。
- 当前程序没有 NVTX range，因此 `nsys` 无法按 warmup/benchmark 阶段分段聚合。后续建议加 NVTX 标记。

注意：当前 `nsys` 摘要没有成功导出 GPU kernel/memcpy 汇总表，因此 kernel 级结论以 `ncu` 为准。

## Nsight Compute

采集命令：

```bash
ncu \
  --set detailed \
  --target-processes all \
  --force-overwrite \
  -o reports/softmax_500k_ncu_detailed \
  ./bin/Softmax 500000 1 0
```

生成文件：

- [softmax_500k_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/softmax_500k_ncu_detailed.ncu-rep)
- [softmax_500k_ncu_details.txt](/home/xj/advanced_cuda/reports/softmax_500k_ncu_details.txt)

`ncu` 会通过多 pass 重放 kernel，程序打印的 `avg_us` 会严重失真，不能用于性能判断。

### Kernel 指标摘要

| Kernel | Grid x Block | Duration | Memory Throughput | DRAM Throughput | SM Throughput | Achieved Occupancy | Registers / Thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| `max_first_pass_kernel` | `420 x 256` | `6.56 us` | `320.23 GB/s` | `35.82%` | `8.96%` | `70.40%` | `19` |
| `max_final_pass_kernel` | `1 x 256` | `2.30 us` | `9.11 GB/s` | `1.05%` | `0.05%` | `15.11%` | `16` |
| `exp_sum_kernel` | `420 x 256` | `7.30 us` | `288.56 GB/s` | `32.45%` | `9.49%` | `72.66%` | `24` |
| `sum_final_pass_kernel` | `1 x 256` | `2.40 us` | `12.05 GB/s` | `1.35%` | `0.04%` | `16.76%` | `16` |
| `normalize_kernel` | `420 x 256` | `5.06 us` | `428.10 GB/s` | `47.87%` | `9.41%` | `92.01%` | `24` |

补充观察：

- 所有 kernel 的 local memory spilling 都是 `0`。
- 主路径 kernel 的 branch efficiency 接近或等于 `100%`。
- `max_final_pass_kernel` 和 `sum_final_pass_kernel` 被 `ncu` 明确提示 grid 太小：只有 1 个 block，无法填满 70 个 SM。

## 性能解读

### 1. 主要开销来自多阶段全局同步

Softmax 的数据依赖是：

```text
global max -> exp and sum -> global sum -> normalize
```

当前实现用 5 个 kernel 表达这个依赖。每个 kernel launch 都是一次全局同步边界。对 `n=500000` 这种中等规模输入，launch overhead 和两个单 block final reduction 的固定开销都比较明显。

从 `ncu` 单 kernel 时间看，5 个 kernel 的 GPU 执行时间合计约：

```text
6.56 + 2.30 + 7.30 + 2.40 + 5.06 = 23.62 us
```

程序自带 event 计时高于这个数，原因包括 kernel launch、运行波动、时钟状态、采样差异以及 profiler/非 profiler 条件不同。

### 2. 主路径不是 compute-bound

`max_first_pass_kernel`、`exp_sum_kernel`、`normalize_kernel` 的 SM Throughput 只有约 `9%`，DRAM Throughput 约 `32% - 48%`。这说明当前规模下 GPU 没有被算力打满，也没有完全打满显存峰值带宽。

主要原因：

- 输入规模只有 `500000` 个 float，约 `2 MB`，单个 kernel 工作量偏小。
- 每个主路径 kernel 只有 `420` 个 block，在 70 个 SM 上约 6 个 block/SM，能跑满一波，但持续时间很短。
- reduction kernel 存在跨 warp/block 汇总和尾部判断，调度和同步开销占比不低。

### 3. 两个 final reduction 是明显固定开销

`max_final_pass_kernel` 和 `sum_final_pass_kernel` 都只启动 `1 x 256`，每次只归约 `420` 个 partial 值。

它们的问题不是绝对耗时大，而是相对整个 softmax 耗时明显：

```text
max_final_pass_kernel: 2.30 us
sum_final_pass_kernel: 2.40 us
合计: 4.70 us
```

这两个 kernel 只用到 1 个 block，occupancy 分别只有 `15.11%` 和 `16.76%`，对 70 SM 的 GPU 来说硬件利用率很低。

### 4. normalize kernel 利用率最好

`normalize_kernel` 是最简单的线性读写：

```cpp
output[i] *= inv_sum;
```

它的 achieved occupancy 达到 `92.01%`，memory throughput `428.10 GB/s`，是当前 5 个 kernel 里带宽表现最好的。它没有 reduction，也没有 shared memory，同步和控制流更少。

### 5. exp_sum kernel 比 normalize 慢但合理

`exp_sum_kernel` 做了三件事：

- 读 input。
- 计算 `__expf(x - max)`。
- 写 output 并做 block 级 sum reduction。

因此它比 `normalize_kernel` 多了 SFU/exp 指令、block reduction 和 partial sum 写回。`7.30 us` 高于 normalize 的 `5.06 us` 是合理的。

## 优化建议

优先级从高到低：

### 1. 加 NVTX range

在 benchmark 中加：

```cpp
nvtxRangePushA("warmup");
...
nvtxRangePop();

nvtxRangePushA("benchmark");
...
nvtxRangePop();
```

这样 `nsys` 可以按阶段聚合，避免初始化、warmup、benchmark 混在一起。

### 2. 用 CUB 做 baseline

建议和 `code/lib_Softmax.cu` 的 Thrust/CUB 版本对比，至少记录：

- correctness error。
- average latency。
- kernel 数量。
- `ncu` 单 kernel 指标。

这样能判断当前手写版本主要差在算法结构、kernel fusion、还是单 kernel 实现。

### 3. 减少 kernel 数量

当前 5 kernel 版本清晰但 launch 和中间归约开销较多。可尝试：

- 对较小 `n` 使用单 block 或少 block fused softmax。
- 对中等 `n` 用 cooperative groups 做 grid-level sync，前提是使用 cooperative launch 并满足硬件限制。
- 对二维 softmax 场景按 row/block 处理，每个 block 完成一行或一段，减少全局同步。

对一维全数组 softmax，由于必须全局 max 和全局 sum，完全融合比较难；但对于 Transformer 里的 row-wise softmax，融合空间会大很多。

### 4. 调整 grid 策略

当前 `partial_blocks=420`，来源是 occupancy 上限。对 `n=500000` 来说，每个线程实际处理约 4 到 5 个 float，kernel 很短。

可以实验：

- 固定更少 block，让每个线程处理更多元素，减少 partial 数量和 final reduction 开销。
- 固定更多 block，提高短 kernel 的并行度，但会增加 partial 数量。
- 分别测试 `blocks = SM * {1,2,4,6,8}`。

目标不是盲目提高 occupancy，而是找到总耗时最低点。

### 5. 对 normalize 使用 vectorized load/store

`max_first_pass_kernel` 和 `exp_sum_kernel` 已经使用 `float4` 主路径；`normalize_kernel` 目前是逐元素处理。可以改成每个线程处理 `float4`，减少指令数并改善访存事务表达。

注意需要处理尾部和对齐。`cudaMalloc` 返回指针对齐通常足够，但 `output + i` 的 `i` 也要保证 4 元素对齐。

### 6. 扩大测试规模做 roofline 判断

建议补充：

```bash
./bin/Softmax 500000 100 10
./bin/Softmax 5000000 100 10
./bin/Softmax 50000000 50 5
```

如果规模增大后 DRAM Throughput 明显上升，说明当前 `500000` 主要受短 kernel 和 launch/reduction 固定开销影响。如果规模增大后仍然低，则需要进一步查访存合并、指令 mix 和调度 stall。

## 当前结论

当前实现正确，结构清晰，workspace 也已经在 benchmark 前分配，避免了 `cudaMalloc/cudaFree` 污染 kernel 计时。

主要性能问题不是寄存器溢出、分支发散或严重 shared memory 问题，而是：

1. 5 个 kernel 带来的多次全局同步和 launch 边界。
2. 两个单 block final reduction 对 70 SM GPU 利用率很低。
3. `n=500000` 规模偏小，主路径 kernel 运行时间很短，难以打满 GPU。

下一步最有价值的工作是：加 NVTX、做 CUB baseline、扫描不同 block 数，并把 `normalize_kernel` 改成 `float4` 版本做 A/B benchmark。
