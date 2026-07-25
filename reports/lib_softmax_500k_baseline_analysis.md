# lib_Softmax.cu CUB Baseline 分析

分析对象：[code/lib_Softmax.cu](/home/xj/advanced_cuda/code/lib_Softmax.cu)

这个文件已从原来的 `n=5` API 示例改成可和 [code/Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu) 公平对比的 CUB baseline。

## 公平对比口径

当前 `lib_Softmax.cu` 的 benchmark 口径和 `Softmax.cu` 对齐：

```bash
./bin/lib_Softmax [n] [benchmark_iters] [warmup_iters]
./bin/Softmax     [n] [benchmark_iters] [warmup_iters]
```

计时区不包含：

- `cudaMalloc` / `cudaFree`
- CUB temp storage size 查询
- CUB temp storage 分配
- H2D / D2H 拷贝
- CPU reference 计算

计时区只包含：

- `cub::DeviceReduce::Max`
- `exp_shift_kernel`
- `cub::DeviceReduce::Sum`
- `normalize_kernel`

对应实现位置：

- CUB temp storage 查询：[code/lib_Softmax.cu](/home/xj/advanced_cuda/code/lib_Softmax.cu:107)
- CUB softmax 主路径：[code/lib_Softmax.cu](/home/xj/advanced_cuda/code/lib_Softmax.cu:121)
- benchmark 计时循环：[code/lib_Softmax.cu](/home/xj/advanced_cuda/code/lib_Softmax.cu:144)

## 编译与运行

编译：

```bash
nvcc -O3 -arch=sm_80 -lineinfo code/lib_Softmax.cu -o bin/lib_Softmax
```

同规模对比：

```bash
./bin/lib_Softmax 500000 200 20
./bin/Softmax 500000 200 20
```

本次结果：

```text
CUB baseline:
n=500000 output_sum=0.999999919 max_abs_error=3.637979e-12 max_rel_error=4.254749e-07
cub softmax avg_us=56.673 bandwidth_gbps=176.45 blocks=420 temp_bytes=8703

Hand-written Softmax.cu:
n=500000 output_sum=0.999999919 max_abs_error=3.637979e-12 max_rel_error=4.254749e-07
softmax avg_us=52.877 bandwidth_gbps=189.12 partial_blocks=420
```

结论：在这次运行中，手写版本略快于 CUB baseline，但两者非常接近。由于短 kernel 对 GPU 时钟和系统状态敏感，建议看多次运行的中位数，不要只看单次结果。

## Nsight 报告

生成文件：

- [lib_softmax_500k_nsys.nsys-rep](/home/xj/advanced_cuda/reports/lib_softmax_500k_nsys.nsys-rep)
- [lib_softmax_500k_nsys.sqlite](/home/xj/advanced_cuda/reports/lib_softmax_500k_nsys.sqlite)
- [lib_softmax_500k_ncu_detailed.ncu-rep](/home/xj/advanced_cuda/reports/lib_softmax_500k_ncu_detailed.ncu-rep)
- [lib_softmax_500k_ncu_details.txt](/home/xj/advanced_cuda/reports/lib_softmax_500k_ncu_details.txt)

采集命令：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o reports/lib_softmax_500k_nsys \
  ./bin/lib_Softmax 500000 20 5
```

```bash
ncu \
  --set detailed \
  --target-processes all \
  --force-overwrite \
  -o reports/lib_softmax_500k_ncu_detailed \
  ./bin/lib_Softmax 500000 1 0
```

## Kernel 结构

CUB baseline 一轮 softmax 实际启动 6 个 kernel：

```text
CUB Max: DeviceReduceKernel
CUB Max: DeviceReduceSingleTileKernel
exp_shift_kernel
CUB Sum: DeviceReduceKernel
CUB Sum: DeviceReduceSingleTileKernel
normalize_kernel
```

手写 `Softmax.cu` 一轮是 5 个 kernel：

```text
max_first_pass_kernel
max_final_pass_kernel
exp_sum_kernel
sum_final_pass_kernel
normalize_kernel
```

差异点：

- CUB 的 max/sum reduction 都是两段式，所以 Max 和 Sum 各 2 个 kernel。
- 手写版本把 `exp` 和 partial sum 放在同一个 kernel 里，因此比 CUB baseline 少一次单独的逐元素 sum 输入准备逻辑。
- CUB reduction kernel 本身更成熟，但多一次 kernel launch，且仍有 single-tile final reduction 的固定开销。

## NCU 关键观察

CUB Max 第一段 reduction：

```text
Grid x Block: 123 x 256
Duration: 4.51 us
Memory Throughput: 456.17 GB/s
DRAM Throughput: 51.49%
Achieved Occupancy: 24.48%
Registers / Thread: 40
```

CUB Max final single-tile：

```text
Grid x Block: 1 x 256
Duration: 2.69 us
Achieved Occupancy: 17.26%
```

`exp_shift_kernel`：

```text
Grid x Block: 420 x 256
Duration: 5.44 us
Memory Throughput: 587.81 GB/s
DRAM Throughput: 65.94%
Achieved Occupancy: 68.10%
Registers / Thread: 22
```

CUB Sum final single-tile：

```text
Grid x Block: 1 x 256
Duration: 2.30 us
Achieved Occupancy: 16.85%
```

`normalize_kernel`：

```text
Grid x Block: 420 x 256
Duration: 5.25 us
Memory Throughput: 398.59 GB/s
DRAM Throughput: 44.95%
Achieved Occupancy: 69.53%
Registers / Thread: 16
```

注意：`ncu` 输出中 CUB Sum 第一段 reduction 的完整段落较长，报告文件中可查看完整指标。它和 CUB Max 第一段类似，都是多 block reduction，再接 single-tile final reduction。

## NSYS 关键观察

`nsys` 的 CUDA API 统计：

```text
cudaMalloc:        5 calls, 220.047 ms total, 97.2% CUDA API time
cudaLaunchKernel: 150 calls, 2.686 ms total
cudaMemcpy:        2 calls, 2.353 ms total
cudaFree:          5 calls, 0.856 ms total
```

这说明完整程序视角下，初始化分配仍然很重。但这不影响 benchmark 的 kernel-only 对比，因为计时区在分配和拷贝之后。

`nsys` 还识别到了 CUB 自带 NVTX range：

```text
CCCL:cub::DeviceReduce::Max
CCCL:cub::DeviceReduce::Sum
```

这对后续看 timeline 很有帮助。手写 `Softmax.cu` 目前没有 NVTX，建议后续补上。

## 和手写版本的关系

这个 CUB baseline 的价值是回答两个问题：

1. 手写 reduction 是否明显输给成熟库实现。
2. 当前 softmax 的主要瓶颈是 reduction kernel 本身，还是多阶段 softmax 结构。

从当前结果看：

- CUB 的大 reduction kernel 单独看比手写第一阶段 reduction 更快。
- 但 CUB Max 和 Sum 各有 single-tile final kernel，固定开销仍然存在。
- CUB baseline 一轮 6 个 kernel，手写版本一轮 5 个 kernel。
- 最终端到端两者接近，说明当前主要瓶颈不只是 block 内 reduction 写法，而是全局 softmax 必须跨 kernel 完成 max、sum、normalize 这些同步阶段。

## 后续建议

建议后续做三组实验：

1. 多次运行取中位数：

```bash
./bin/lib_Softmax 500000 200 20
./bin/Softmax 500000 200 20
```

2. 扩大输入规模：

```bash
./bin/lib_Softmax 5000000 100 10
./bin/Softmax 5000000 100 10
```

3. 给手写版本加 NVTX：

```text
warmup
benchmark
max
exp_sum
normalize
```

如果规模变大后 CUB 明显领先，说明手写 reduction 的吞吐还有差距；如果两者仍接近，则应该优先考虑减少 kernel 数量、改 row-wise softmax、或做更贴近 Transformer attention 的 fused softmax。
