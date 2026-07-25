# Nsight Systems / Nsight Compute 常用报告数据规范

> 来源说明：用户给的视频是 B 站《彻底搞懂！CUDA编程之：如何安装Nsight system及compute》（BV1UP411s7nE）。当前环境无法直接取得完整视频字幕，因此本文把视频主题中常见的 `nsys` / `ncu` 使用方式，结合本仓库已有报告和 NVIDIA 官方 CLI 口径整理成后续 profiling 的固定规范。

## 快速命令

后续分析 CUDA profiling 数据时，默认优先使用本节的编译、NSYS、NCU 指令生成报告；只有在需要缩短采集时间、限定特定 kernel、或补充专项指标时，才在这些命令基础上调整参数。

编译时保留源码行号和调试符号，方便 Nsight 报告关联源码：

```bash
nvcc -O3 -lineinfo -Xcompiler -g \
  -o GEMM_navie_benchmark \
  GEMM_navie_benchmark.cu
```

用 Nsight Systems 尽量收集系统级 timeline、CUDA API、kernel、memcpy 和 GPU metrics：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-device=all \
  --stats=true \
  --force-overwrite=true \
  -o GEMM_navie_nsys_full \
  ./GEMM_navie_benchmark
```

用 Nsight Compute 尽量收集 kernel 级详细指标：

```bash
ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  -o GEMM_navie_ncu_full \
  ./GEMM_navie_benchmark
```

`ncu --set full` 会比较慢，日常迭代可以先把 `full` 换成 `detailed`。

## 总体原则

- `nsys` 看系统级时间线：程序是否真的跑在 GPU 上、CUDA API 和 GPU kernel 是否同步/阻塞、kernel 与 memcpy 的时间占比、CPU/GPU 是否有空洞。
- `ncu` 看单个 CUDA kernel：kernel 的耗时、SM 利用率、访存吞吐、cache 命中率、occupancy、寄存器/共享内存/线程块配置。
- 先用 `nsys` 定位瓶颈 kernel，再用 `ncu` 对目标 kernel 做细粒度分析。
- 每次报告都保留原始 `.nsys-rep` / `.ncu-rep`，同时导出文本或 CSV 摘要，方便贴到笔记、PR 或 benchmark 记录里。

## 推荐文件命名

以可执行文件名、优化版本、数据规模、工具名组成文件名：

```bash
<program>_<case>_nsys.nsys-rep
<program>_<case>_nsys.sqlite
<program>_<case>_nsys_stats.txt
<program>_<case>_ncu.ncu-rep
<program>_<case>_ncu_details.txt
```

示例：

```bash
GEMM_navie_nsys_gpu_metrics.nsys-rep
GEMM_navie_nsys_gpu_metrics.sqlite
GEMM_navie_ncu_memory.ncu-rep
```

## NSYS 采集命令

基础采集：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o <program>_<case>_nsys \
  ./<program> [args...]
```

需要 GPU metrics 时使用：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --gpu-metrics-device=all \
  --stats=true \
  --force-overwrite=true \
  -o <program>_<case>_nsys_gpu_metrics \
  ./<program> [args...]
```

常用摘要导出：

```bash
nsys stats \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_kern_gb_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_gpu_mem_size_sum \
  --report nvtx_sum \
  --format table \
  <program>_<case>_nsys.nsys-rep
```

如果要生成 CSV：

```bash
nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum \
  --format csv \
  --output . \
  <program>_<case>_nsys.nsys-rep
```

## NSYS 必看数据

### CUDA API Summary

对应 `cuda_api_sum`。

记录这些字段：

- `Time`：该 API 在 CUDA API 总时间里的占比。
- `Total Time`：该 API 总耗时。
- `Instances` / `Num Calls`：调用次数。
- `Avg` / `Med` / `Min` / `Max` / `StdDev`：单次调用耗时分布。
- `Name`：API 名称，例如 `cudaMalloc`、`cudaMemcpy`、`cudaDeviceSynchronize`、kernel launch API。

重点判断：

- `cudaMalloc` / `cudaFree` 是否过多，必要时改成复用 buffer。
- `cudaMemcpy` / `cudaMemcpyAsync` 是否占主导，必要时减少拷贝或改异步流水。
- `cudaDeviceSynchronize` 是否频繁出现，必要时缩小同步范围。

### CUDA GPU Kernel Summary

对应 `cuda_gpu_kern_sum`。

记录这些字段：

- `Time`：该 kernel 在所有 GPU kernel 时间里的占比。
- `Total Time`：该 kernel 所有实例总耗时。
- `Instances`：kernel 调用次数。
- `Avg` / `Med` / `Min` / `Max` / `StdDev`：kernel 单次耗时分布。
- `Name`：kernel 名称。

如果需要同时看 launch 配置，导出 `cuda_gpu_kern_gb_sum`，额外关注：

- `Grid X/Y/Z`
- `Block X/Y/Z`
- kernel name

重点判断：

- 哪个 kernel 是主耗时。
- kernel 是否被调用过多。
- 同一个 kernel 耗时波动是否异常。
- grid/block 配置是否符合预期。

### CUDA GPU MemOps

按时间看 `cuda_gpu_mem_time_sum`，按大小看 `cuda_gpu_mem_size_sum`。

时间字段：

- `Time`
- `Total Time`
- `Count`
- `Avg` / `Med` / `Min` / `Max` / `StdDev`
- `Operation`

大小字段：

- `Total`
- `Count`
- `Avg` / `Med` / `Min` / `Max` / `StdDev`
- `Operation`

重点判断：

- HtoD / DtoH / DtoD 拷贝量和耗时是否合理。
- memset 是否异常频繁。
- 拷贝是否可以和 kernel 重叠。

### NVTX Summary

对应 `nvtx_sum`。

建议所有 benchmark 后续都加 NVTX range，例如：

```cpp
nvtxRangePushA("warmup");
// warmup kernels
nvtxRangePop();

nvtxRangePushA("benchmark");
// measured kernels
nvtxRangePop();
```

记录：

- range 名称。
- range 总耗时。
- range 调用次数。
- range 内 CUDA API / kernel 的对应关系。

## NCU 采集命令

基础 kernel 分析：

```bash
ncu \
  --set basic \
  --target-processes all \
  --force-overwrite \
  -o <program>_<case>_ncu_basic \
  ./<program> [args...]
```

更适合做优化记录的详细分析：

```bash
ncu \
  --set detailed \
  --target-processes all \
  --force-overwrite \
  -o <program>_<case>_ncu_detailed \
  ./<program> [args...]
```

只分析目标 kernel：

```bash
ncu \
  --set detailed \
  --kernel-name '<kernel_regex>' \
  --target-processes all \
  --force-overwrite \
  -o <program>_<case>_ncu_<kernel_name> \
  ./<program> [args...]
```

导出文本：

```bash
ncu --import <program>_<case>_ncu_detailed.ncu-rep --page details
```

## NCU 必看数据

### GPU Speed Of Light Throughput

记录：

- `Duration`
- `Elapsed Cycles`
- `SM Frequency`
- `DRAM Frequency`
- `Compute (SM) Throughput`
- `Memory Throughput`
- `DRAM Throughput`
- `L1/TEX Cache Throughput`
- `L2 Cache Throughput`
- `SM Active Cycles`

读法：

- `Compute (SM) Throughput` 高、`Memory Throughput` 低：更像计算瓶颈。
- `Memory Throughput` / `DRAM Throughput` 高：更像访存瓶颈。
- 二者都低：通常要看 occupancy、warp stall、launch 配置或同步问题。

### Memory Workload Analysis

记录：

- `Memory Throughput`
- `Mem Busy`
- `Max Bandwidth`
- `L1/TEX Hit Rate`
- `L2 Hit Rate`
- `Mem Pipes Busy`
- `Local Memory Spilling Requests`
- `Local Memory Spilling Request Overhead`

读法：

- `Local Memory Spilling` 非 0：优先检查寄存器压力、局部数组、编译选项。
- `L1/TEX Hit Rate` / `L2 Hit Rate` 低：检查访存复用和合并访问。
- `Mem Pipes Busy` 高：访存管线压力大。

### Launch Statistics

记录：

- `Block Size`
- `Grid Size`
- `Threads`
- `Registers Per Thread`
- `Static Shared Memory Per Block`
- `Dynamic Shared Memory Per Block`
- `Shared Memory Configuration Size`
- `Waves Per SM`
- `# SMs`

读法：

- `Registers Per Thread` 过高可能限制 occupancy。
- shared memory per block 过高也可能限制每个 SM 的并发 block。
- `Waves Per SM` 太低时，kernel 可能不足以填满 GPU。

### Occupancy

记录：

- `Theoretical Occupancy`
- `Achieved Occupancy`
- `Theoretical Active Warps per SM`
- `Achieved Active Warps Per SM`
- `Block Limit Registers`
- `Block Limit Shared Mem`
- `Block Limit Warps`
- `Block Limit SM`

读法：

- `Achieved Occupancy` 明显低于 `Theoretical Occupancy`：检查访存等待、分支发散、同步或 launch 规模。
- `Block Limit Registers` 低：寄存器限制并发。
- `Block Limit Shared Mem` 低：共享内存限制并发。
- occupancy 高不等于性能一定好，还要结合吞吐和 stall 原因。

### Roofline / Workload Analysis

使用 `--set detailed` 或 `--set roofline` 时记录：

- achieved FLOP/s 或 peak 百分比。
- arithmetic intensity。
- roofline 图中是靠近 compute roof 还是 memory roof。
- Nsight Compute 给出的 `OPT` / `INF` 提示和估计 speedup。

读法：

- 靠近 memory roof：优先优化访存合并、数据复用、shared memory tiling。
- 靠近 compute roof：优先优化指令、tensor core、展开、数据类型。
- 离两个 roof 都远：优先找 occupancy、stall、launch、小 kernel 开销。

## 本仓库当前报告示例

当前 `GEMM_navie_ncu_memory.ncu-rep` 中，`gemm_naive_kernel` 的常见字段示例：

- `Duration` 约 920 us。
- `Compute (SM) Throughput` 约 91%。
- `Memory Throughput` 约 91%。
- `DRAM Throughput` 约 1% 到 2%。
- `L1/TEX Cache Throughput` 约 92%。
- `L2 Cache Throughput` 约 18% 到 23%。
- `Memory Workload Analysis` 中 `Local Memory Spilling Requests = 0`。
- `Launch Statistics` 中 `Block Size = 256`、`Grid Size = 4096`、`Registers Per Thread = 24`、`Threads = 1048576`。
- `Occupancy` 中 `Theoretical Occupancy = 100%`、`Achieved Occupancy` 约 96%。

这个例子说明：该 kernel 的占用率很高，Compute/Memory Throughput 也高；继续优化时不能只看 occupancy，应结合 NCU 的 uncoalesced load 提示、roofline 和访存访问模式继续判断。

## 后续生成报告时的固定输出

每次 profiling 后，在记录里至少包含：

```text
环境:
- GPU:
- CUDA:
- nsys:
- ncu:
- 编译命令:
- 运行命令:
- 输入规模:

NSYS:
- Top CUDA APIs:
- Top GPU kernels:
- MemOps by time:
- MemOps by size:
- 是否有明显同步/空洞:

NCU:
- 目标 kernel:
- Duration:
- Compute (SM) Throughput:
- Memory Throughput:
- DRAM Throughput:
- L1/TEX Hit Rate:
- L2 Hit Rate:
- Registers Per Thread:
- Shared Memory Per Block:
- Theoretical Occupancy:
- Achieved Occupancy:
- 主要 OPT/INF 提示:

结论:
- 当前主要瓶颈:
- 下一步优化方向:
```

## 参考资料

- NVIDIA Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/
- NVIDIA Nsight Systems Documentation: https://docs.nvidia.com/nsight-systems/
- NVIDIA Nsight Compute User Guide: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- NVIDIA Nsight Compute CLI: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
