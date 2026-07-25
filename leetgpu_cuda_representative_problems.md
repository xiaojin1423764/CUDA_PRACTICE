
# LeetGPU CUDA 代表性中档/难题整理

来源：LeetGPU 官方 `AlphaGPU/leetgpu-challenges` 仓库题面。本文不是全题库复刻，而是按 CUDA 程序设计和优化训练价值筛出的代表题摘要。

## 推荐训练顺序

1. Reduction
2. Prefix Sum
3. Softmax
4. GEMM
5. 2D/3D Convolution
6. Top-K Selection
7. INT8/INT4 MatMul
8. Sorting / Radix Sort
9. Attention 系列
10. BFS / All-Pairs Shortest Paths
11. FFT
12. GPT-2 / Llama Transformer Block

---

## 题目掌握深度要求

这份题单的目标不是只把题目跑通，而是逐步建立 CUDA 算子实现和性能分析能力。不同题目需要投入的深度不同，建议按下面三个层次验收。

### 第一层：能写对

要求：

- 能独立写出 CUDA 实现。
- 能处理题目约束内的边界情况，例如非 2 的幂长度、不整除 tile、空输入、小输入、大输入。
- 有 CPU baseline 或库函数结果做 correctness check。
- 有基本 benchmark，能区分 warmup 和正式计时。

适用题目：

- 2D/3D Convolution
- Top-K Selection
- Sorting / Radix Sort
- BFS / All-Pairs Shortest Paths
- FFT

这些题目第一轮先掌握算法结构和并行化方式，不要求一开始就接近工业库性能。后续如果研究方向需要图计算、信号处理、排序库或数据库/GPU analytics，再深入优化。

### 第二层：能解释实现细节

要求：

- 能解释数据如何映射到 thread / warp / block。
- 能说明 global memory 访问是否 coalesced。
- 能说明 shared memory 用在哪里，是否可能有 bank conflict。
- 能解释 warp shuffle、block reduction、分层 scan、跨 block 合并等设计。
- 能说明 kernel launch 配置为什么这样选。
- 能判断主要瓶颈大概来自算力、显存带宽、同步、分支还是 launch overhead。

必须达到这一层的题目：

- Reduction
- Prefix Sum
- Softmax

这三类是后续大多数 CUDA 算子的基础模式。它们不一定要做到极限性能，但必须能把实现细节讲清楚。

### 第三层：能用 profiling 证明并优化

要求：

- 能用 Nsight Systems 判断端到端瓶颈、CUDA API 开销、kernel 时间占比和 memcpy 时间占比。
- 能用 Nsight Compute 分析 SM 利用率、memory throughput、L2 hit rate、occupancy、warp stall reason、register/shared memory 使用。
- 能和 `cuBLAS`、`cuDNN`、`CUB`、`CUTLASS`、Triton 或其他合理 baseline 对比。
- 能根据 profiling 数据提出优化，而不是只凭经验改代码。
- 能记录每轮优化前后的性能变化和原因。

必须重点深挖的题目：

- GEMM
- Attention 系列
- INT8 Quantized MatMul
- INT4 Weight-Only Quantized MatMul
- GPT-2 / Llama Transformer Block

这些题目直接决定后续能否进入高性能 AI 算子、LLM 推理和系统级优化。尤其是 GEMM 和 Attention，建议做到能解释 Tensor Core、tiling、数据布局、访存复用、数值稳定性和主要性能瓶颈。

### 推荐验收标准

每完成一个题目，至少记录：

- 正确性测试覆盖了哪些输入规模和边界情况。
- kernel 的 grid/block 配置。
- 每个 kernel 的主要职责。
- 是否使用 shared memory、warp shuffle、vectorized load/store、Tensor Core 或库 primitive。
- benchmark 输入规模、warmup 次数、正式计时次数、平均耗时。
- 与 CPU baseline 或库实现的结果误差。
- 如果进入第三层，还要保存 Nsight 报告和性能分析结论。

简化判断：

```text
Reduction / Prefix Sum / Softmax：至少第二层
GEMM / Attention / INT8/INT4 MatMul / Llama Block：尽量第三层
Conv / Top-K / Sort / BFS / FFT：第一轮第一到第二层即可
```

---

## 文件命名规则

- `code/`：保存 CUDA 源码。之后新增代码也放在这里。
- `bin/`：保存编译产物和可执行文件。之后编译输出也放在这里。
- `reports/`：保存 Nsight Compute / Nsight Systems 等性能报告。之后性能报告也放在这里。
- 不直接调用 Thrust/CUB 等库完成整题的实现，使用题目名命名，例如 `code/Reduction.cu`、`code/PrefixSum.cu`。
- 主要用于学习库 API、直接调用库函数完成题目的实现，加 `lib_` 前缀，例如 `code/lib_Reduction.cu`、`code/lib_PrefixSum.cu`。

## Medium

### 4. Reduction

CUDA 训练重点：并行归约、shared memory、warp divergence、非 2 的幂长度处理、多级归约。

题目描述：给定一个 `float32` 数组，在 GPU 上并行求和，输出单个总和。

示例：

```text
Input:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0

Input:  [-2.5, 1.5, -1.0, 2.0]
Output: 0.0
```

约束/性能点：`1 <= N <= 100,000,000`，性能测试 `N = 4,194,304`。

参考实现：[Reduction.cu](/home/xj/advanced_cuda/code/Reduction.cu)。该版本不调用 Thrust/CUB 的现成 reduce，而是手写两阶段归约：

- 第一阶段：每个 thread 读取 4 个 `float`，使用 `float4` 合并读取，block 内用 warp shuffle + 少量 shared memory 归约，输出每个 block 的 partial sum。
- 第二阶段：单个 block 归约所有 partial sum，写出最终结果。
- block 数通过 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 按当前 GPU 估算，避免为了小输入启动过多 block，也避免大输入只有少量 block 导致 SM 吃不满。
- 测试程序把中间 workspace 预分配后复用，避免 `cudaMalloc` / `cudaFree` 污染 kernel 性能计时。
- 调包示例：[lib_Reduction.cu](/home/xj/advanced_cuda/code/lib_Reduction.cu)，用于学习 CUB `DeviceReduce::Sum` API。
- 性能报告：[Nsight Compute](/home/xj/advanced_cuda/reports/leetgpu_reduction_compare_ncu_detailed.ncu-rep)、[Nsight Systems](/home/xj/advanced_cuda/reports/leetgpu_reduction_compare_nsys.nsys-rep)。

#### 完成标志（2026-07-25）

- **实现状态：已实现；第一层验收部分完成，第二层实现说明完成。** 本轮在 RTX 5070 Ti 上以 `N=4,194,304`、`5` 次 warmup 和 `20` 次正式计时运行。CPU 参考值、手写 GPU 和 CUB GPU 的和均为 `4,194,304`，两种 GPU 结果误差均为 `0`；手写版 `21.211 us`（`790.96 GB/s`），CUB `20.109 us`（`834.32 GB/s`），手写版为 CUB 时间的 `1.055x`。
- **第一层问题：正确性与边界是否覆盖？** 当前入口验证了题目性能规模，并与 CPU 累加和及 CUB 对照；实现的首阶段会以掩码处理最后一个不满 `float4` 的读取。尚未把 `N=1`、非 2 的幂、随机/负数输入写入自动化测试，因此不将第一层标为完全验收。
- **第二层问题：线程、访存和同步如何设计？** 每个 thread 读取 4 个连续 `float`，`float4` 主路径使 warp 的全局读取连续合并；一个 block 先在寄存器中累计，再用 warp shuffle 完成 warp 内归约，只将各 warp 的部分和写入 shared memory，最后由第一个 warp 汇总。第一阶段输出 block partial sum，第二阶段由单 block 写出总和；block 数按 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 和 SM 数量估算，以兼顾并行度与 partial 数量。shared memory 仅按 warp 编号线性访问，使用量很小，也不存在同一 warp 的 bank conflict 模式。
- **第三层问题：是否有 profiling 依据？** 已保存 Nsight Systems/Compute 报告，并有 CUB 对照；该题的第三层不是题单强制项。后续要完成第一层，应扩展测试入口并固定记录不同输入形状下的正确性。

单文件测试编译：

```bash
nvcc -O3 -arch=sm_80 -lineinfo \
  code/Reduction.cu \
  -o bin/Reduction
```

运行：

```bash
./bin/Reduction 100 10
```

其中第一个参数是 benchmark 次数，第二个参数是 warmup 次数。

### 16. Prefix Sum

CUDA 训练重点：scan、块内同步、跨 block 累加、分层算法、bank conflict。

题目描述：计算 `float32` 数组的 inclusive prefix sum。输入 `[a,b,c,d]` 输出 `[a,a+b,a+b+c,a+b+c+d]`。

示例：

```text
Input:  [1.0, 2.0, 3.0, 4.0]
Output: [1.0, 3.0, 6.0, 10.0]

Input:  [5.0, -2.0, 3.0, 1.0, -4.0]
Output: [5.0, 3.0, 6.0, 7.0, 3.0]
```

约束/性能点：`1 <= N <= 100,000,000`，性能测试 `N = 250,000`。

参考实现：[PrefixSum.cu](/home/xj/advanced_cuda/code/PrefixSum.cu)。该版本不直接调用 Thrust/CUB 的 device-level scan，而是用 CUB block-level primitive 做块内扫描，再递归扫描 block sums 并加回跨 block offset。

调包示例：[lib_PrefixSum.cu](/home/xj/advanced_cuda/code/lib_PrefixSum.cu)，用于学习 Thrust `thrust::inclusive_scan` 和 CUB `DeviceScan::InclusiveSum` API。

#### 完成标志（2026-07-25）

- **实现状态：已实现；第一层验收部分完成，第二层实现说明完成。** 在 RTX 5070 Ti 上，`N=250,000`、`5` 次 warmup 和 `20` 次正式计时得到末元素 `250000`、最大误差 `0`、平均 `503.042 us`（`3.98 GB/s`）。块边界验证 `N=1`、`2047`、`2048`、`2049` 的末元素均等于期望值，最大误差均为 `0`。
- **第一层问题：正确性与边界是否覆盖？** 测试覆盖了单元素、一个 block 内的尾部、恰好一个 block、跨 block 的首个元素；测试输入为全 `1`，CPU 参考值为 `i + 1`。仍缺少随机、负数和大规模自动化对照，因此第一层保留为部分验收。
- **第二层问题：线程、访存和同步如何设计？** 一个 block 为 `256` threads，每 thread 处理 `8` 个元素，即每 block `2048` 个元素。`cub::BlockLoad/BlockStore` 的 warp-transpose 布局用于块内合并读写；thread 先在寄存器内扫描其 8 项，再以 warp shuffle 扫描 thread sum。每个 warp 只写一个 shared-memory partial sum，第一个 warp 扫描这些 partial sum 后广播 offset。递归扫描 block sums，再由 `add_block_offsets_kernel` 加回前一 block 的总和，解决跨 block 依赖。主要瓶颈应为多轮 kernel launch 与递归临时分配，而非单个块内扫描的计算量。
- **第三层问题：是否有 profiling 依据？** 此题第三层不是强制项，当前未保存 Nsight 报告。另一个待优化点是递归调用在 benchmark 内部执行 `cudaMalloc/cudaFree`；应改为预分配分层 workspace 后再比较 kernel 性能。

单文件测试编译：

```bash
nvcc -O3 -arch=sm_80 -lineinfo \
  code/PrefixSum.cu \
  -o bin/PrefixSum
```

运行：

```bash
./bin/PrefixSum 250000 100 10
```

### 5. Softmax

CUDA 训练重点：max reduction、sum reduction、数值稳定、exp 计算、内存带宽优化。

题目描述：在 GPU 上计算一维数组 softmax，要求使用 max trick 避免指数溢出。

示例：

```text
Input:  [1.0, 2.0, 3.0], N = 3
Output: [0.090, 0.244, 0.665] 近似

Input:  [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
Output: [2.047e-09, 3.038e-07, 4.509e-05, 6.693e-03, 9.933e-01] 近似
```

约束/性能点：`1 <= N <= 500,000`，性能测试 `N = 500,000`。

参考实现：[Softmax.cu](/home/xj/advanced_cuda/code/Softmax.cu)。该版本不直接调用 Thrust/CUB 的整题实现，而是手写三阶段 softmax：

- 第一阶段：全局 max reduction，使用 warp shuffle + shared memory 做 block 内归约。
- 第二阶段：用 max trick 写出 `exp(x - max)`，同时归约指数和。
- 第三阶段：用指数和原地归一化输出。
- 对主路径使用 `float4` 合并读写，workspace 预分配后复用，避免 benchmark 中反复 `cudaMalloc` / `cudaFree`。

调包示例：[lib_Softmax.cu](/home/xj/advanced_cuda/code/lib_Softmax.cu)，用于学习 Thrust `max_element` / `transform` / `reduce` 和 CUB `DeviceReduce::Max` / `DeviceReduce::Sum` API。

#### 完成标志（2026-07-25）

- **实现状态：第一层和第二层已验收，已补充第三层 profiling 记录。** 在 RTX 5070 Ti 上，`N=500,000`、`5` 次 warmup 和 `20` 次正式计时得到 `output_sum=0.999999919`、最大绝对误差 `3.637979e-12`、最大相对误差 `4.254749e-07`，平均 `66.362 us`（`150.69 GB/s`）；`N=1` 时输出和为 `1`、绝对/相对误差均为 `0`。
- **第一层问题：正确性与边界是否覆盖？** CPU 参考实现和 GPU 均采用 max trick，CPU 使用 double 累加；实测覆盖单元素和题目性能规模 `500,000`，后者也是非 2 的幂。输出和接近 1 且逐元素误差很小，满足数值稳定性与正确性检查；benchmark 将 warmup 与正式计时分开，workspace 在计时前分配。
- **第二层问题：线程、访存和同步如何设计？** 主路径使用 `256` threads/block 和 `float4` 连续合并读取。`max_first_pass_kernel` 与 `exp_sum_kernel` 先在寄存器累积，每个 warp 用 shuffle 归约、各 warp 的结果写 shared memory，再由一个 warp 汇总；它们分别产生全局 max 和指数和的 partial 值。`max_final_pass_kernel`、`sum_final_pass_kernel` 各以单 block 合并 partial 值，`normalize_kernel` 写回归一化结果。主路径访存连续，shared memory 只保存 warp partial；主要瓶颈是全局 max、全局 sum、归一化之间的 5 次 kernel launch 和两个单 block final reduction，而不是分支发散。
- **第三层问题：profiling 和 baseline 说明了什么？** [Nsight Systems/Compute 分析](/home/xj/advanced_cuda/reports/softmax_500k_profile_analysis.md) 已记录。`N=500,000` 时，NCU 显示 5 个 kernel 合计约 `23.62 us`；两个单 block final reduction 各约 `2.30-2.40 us`，occupancy 仅约 `15%-17%`，是明显的固定开销。与 [CUB baseline](/home/xj/advanced_cuda/reports/lib_softmax_500k_baseline_analysis.md) 对比的代表结果为手写 `52.877 us`、CUB `56.673 us`；两者接近，说明当前应优先减少多阶段同步/launch 开销，而非仅微调 block 内 reduction。

**首选方案：改为 row-wise fused softmax。** 面向实际模型的二维输入，让一个 block 负责一行，在 block 内以 warp shuffle 和 shared memory 完成 `max -> exp/sum -> normalize`。这样每行只需 1 个 kernel，而不是当前一维全局实现的 5 个 kernel；可消除写出 global partial 值、两次单 block final reduction，以及它们之间的全局同步。
**超长单行：保留分段归约，但减少中间阶段。** 当一行无法由一个 block 处理时，合并 `exp` 计算与 partial sum，复用指数结果，避免额外的全局读写。只有在 profiling 证明 launch 固定开销仍占主导时，再评估


单文件测试编译：

```bash
nvcc -O3 -arch=sm_80 -lineinfo \
  code/Softmax.cu \
  -o bin/Softmax
```

运行：

```bash
./bin/Softmax 500000 100 10
```

### 22. GEMM

CUDA 训练重点：shared memory tiling、coalesced global load、寄存器复用、FP16 输入 FP32 累加、WMMA/Tensor Core。

题目描述：实现 FP16 GEMM：

```text
C = alpha * (A x B) + beta * C_initial
```

其中 `A` 为 `M x K`，`B` 为 `K x N`，`C` 为 `M x N`，矩阵 row-major，最终 `C` 写回 FP16。

示例：

```text
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[1, 2],
     [3, 4],
     [5, 6]]

C_initial = [[1, 1],
             [1, 1]]

alpha = 1.0, beta = 0.0

Output C = [[22, 28],
            [49, 64]]
```

约束/性能点：`16 <= M,N,K <= 4096`，性能测试 `M=N=K=1024`。

参考实现：[GEMM.cu](/home/xj/advanced_cuda/code/GEMM.cu) 和 [lib_GEMM.cu](/home/xj/advanced_cuda/code/lib_GEMM.cu)。前者使用 WMMA 实现 FP16 输入、FP32 累加的 Tensor Core GEMM；后者使用 cuBLAS `cublasGemmEx`，通过 `C^T = B^T x A^T` 直接处理 row-major 数据，作为库 baseline。

单文件测试编译：

```bash
nvcc -O3 -arch=sm_120 -lineinfo code/GEMM.cu -o bin/GEMM
nvcc -O3 -arch=sm_120 -lineinfo code/lib_GEMM.cu -o bin/lib_GEMM -lcublas
```

性能测试命令：

```bash
./bin/GEMM 1024 1024 1024 100 20
./bin/lib_GEMM 1024 1024 1024 100 20
```

- **测试结果（2026-07-25）：第一层和第二层已验收。** 在 RTX 5070 Ti（SM 12.0，CUDA Toolkit 13.0）上，以 `M=N=K=1024`、`20` 次 warmup、`100` 次正式迭代测得：手写 WMMA GEMM 平均 `112.603 us`、`19.07 TFLOPS`；cuBLAS GEMM 平均 `95.950 us`、`22.38 TFLOPS`。手写核达到 cuBLAS 吞吐的约 `85.2%`，耗时高约 `17.4%`。两者均采样校验 `512` 个输出元素，最大绝对误差 `1.531219e-02`、最大相对误差 `4.522427e-04`。
- **实现检查：** 手写核以 `128 x 64 x 32` shared-memory tile 组织，8 个 warp 各计算四个 `16 x 16` WMMA accumulator fragment；A 以 row-major 读取，B 在 cooperative load 时转存为 column-major shared tile，减少 Tensor Core 读取时的布局转换。epilogue 融合 `alpha=1.0`、`beta=0.25` 和 FP16 写回。两个程序都在计时后从原始 `C_initial` 重新运行一次并与 CPU FP32 参考做采样校验。
- **Profiling 状态：** 按当前要求，未运行 `nsys` 或 `ncu`。

### 10. 2D Convolution

CUDA 训练重点：stencil 访问、halo 区域、shared memory tile、边界处理、访存复用。

题目描述：对二维 `float32` 矩阵做 valid convolution。输出尺寸：

```text
output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1
```

示例 1：

```text
input =
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]

kernel =
[[0, 1],
 [1, 0]]

Output =
[[ 6,  8],
 [12, 14]]
```

示例 2：

```text
input =
[[1, 1, 1, 1],
 [1, 2, 3, 1],
 [1, 4, 5, 1],
 [1, 1, 1, 1]]

kernel = [[1, 0, 1]]

Output =
[[2, 2],
 [4, 3],
 [6, 5],
 [2, 2]]
```

约束/性能点：输入行列最大 `3072`，kernel 行列最大 `31`，性能测试 `3072 x 3072` 输入和 `15 x 15` kernel。

### 11. 3D Convolution

CUDA 训练重点：三维索引展开、3D tile、更多 halo、缓存复用、寄存器压力。

题目描述：对三维体数据做 valid convolution：

```text
output(i,j,k) = sum input(i+d,j+r,k+c) * kernel(d,r,c)
```

示例：

```text
Input volume: 2 x 2 x 2
d=0: [[1, 2],
      [3, 4]]
d=1: [[5, 6],
      [7, 8]]

Kernel: 全 1 的 2 x 2 x 2

Output: [36]
```

约束/性能点：输入每维最大 `256`，kernel 每维最大 `5`。

### 29. Top K Selection

CUDA 训练重点：局部 top-k、排序网络、小数组寄存器操作、block 间合并、减少全局排序开销。

题目描述：给定长度为 `N` 的 `float32` 数组，选出最大的 `k` 个元素，按降序写入输出。

示例：

```text
input = [1.0, 5.0, 3.0, 2.0, 4.0], N = 5, k = 3
output = [5.0, 4.0, 3.0]

input = [7.2, -1.0, 3.3, 8.8, 2.2], N = 5, k = 2
output = [8.8, 7.2]
```

约束/性能点：`1 <= N <= 100,000,000`，性能测试 `N = 50,000,000, k = 100`。

### 32. INT8 Quantized MatMul

CUDA 训练重点：int8 量化矩阵乘、int32 累加、scale/zero-point、clamp/round、低精度推理。

题目描述：实现 INT8 量化矩阵乘。核心公式：

```text
Cq(i,j) = clamp(round(sum_k((Aik - zA) * (Bkj - zB)) * sA * sB / sC) + zC, -128, 127)
```

示例 1：

```text
A = [[1, 2],
     [3, 4]]
B = [[5, 6],
     [7, 8]]

scale_A = 0.1, scale_B = 0.2, scale_C = 0.05
zero_point_A = zero_point_B = zero_point_C = 0

Output C = [[19, 22],
            [43, 50]]
```

示例 2：

```text
A = [[1, 2]]
B = [[3],
     [4]]

scale_A = scale_B = scale_C = 1.0
zero_point_A = 1, zero_point_B = 3, zero_point_C = 5

Output C = [[6]]
```

约束/性能点：`1 <= M,N,K <= 4096`，性能测试 `M=8192, N=4096, K=2048`。

### 81. INT4 Weight-Only Quantized MatMul

CUDA 训练重点：INT4 unpack、weight-only dequant、group-wise scale、W4A16 LLM 推理、吞吐和带宽平衡。

题目描述：实现 W4A16 矩阵乘。输入激活 `x` 是 FP16，权重 `w_q` 是 packed INT4。每个 byte 存两个 INT4 权重，高 4 位对应偶数 k，低 4 位对应奇数 k。真实 signed int4 值为 `nibble - 8`。按 group dequant：

```text
W[n,k] = (w_q_nibble[n,k] - 8) * scales[n, k / group_size]
y = x x W^T
```

示例：

```text
M=2, N=4, K=4, group_size=2

x =
[[1, 0, 1, 0],
 [0, 1, 0, 1]]

w_q =
[[0x99, 0x99],
 [0xAA, 0xAA],
 [0x77, 0x77],
 [0x88, 0x88]]

对应 W_int4 =
[[ 1,  1,  1,  1],
 [ 2,  2,  2,  2],
 [-1, -1, -1, -1],
 [ 0,  0,  0,  0]]

scales 全为 0.5

Output y =
[[1.0, 2.0, -1.0, 0.0],
 [1.0, 2.0, -1.0, 0.0]]
```

约束/性能点：`M,N,K <= 8192`，`group_size` 属于 `{2,4,8,16,32,64,128}`，性能测试 `M=N=K=4096, group_size=128`。

---

## Hard

### 15. Sorting

CUDA 训练重点：并行排序、分桶/merge/network 思路、global scatter、shared memory 分块、稳定性和吞吐。

题目描述：将 `float32` 数组按升序排序，结果写回原数组。算法不限。

示例：

```text
Input:  data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0], N = 6
Output: data = [1.0, 2.0, 4.0, 5.0, 8.0, 9.0]
```

约束/性能点：`1 <= N <= 1,000,000`，性能测试 `N = 1,000,000`。

### 36. Radix Sort

CUDA 训练重点：histogram、prefix sum、scatter、double buffering、bit-pass 调度。

题目描述：在 GPU 上用 radix sort 对 `uint32` 数组升序排序。

示例：

```text
Input:  [170, 45, 75, 90, 2, 802, 24, 66]
Output: [2, 24, 45, 66, 75, 90, 170, 802]

Input:  [1, 4, 1, 3, 555, 1000, 2]
Output: [1, 1, 2, 3, 4, 555, 1000]
```

约束/性能点：`1 <= N <= 100,000,000`，性能测试 `N = 50,000,000`。

### 12. Multi-Head Attention

CUDA 训练重点：batched GEMM、row-wise softmax、Q/K/V head 分块、减少中间矩阵写回、attention 访存模式。

题目描述：实现多头自注意力：

```text
head_i = softmax(Q_i K_i^T / sqrt(d_k)) V_i
MultiHead = concat(head_1, ..., head_h)
```

示例 1：

```text
N = 2, d_model = 4, h = 2

Q = [[1, 0, 2, 3],
     [4, 5, 6, 7]]

K = [[1, 2, 3, 4],
     [5, 6, 7, 8]]

V = [[0.5, 1.0, 1.5, 2.0],
     [2.5, 3.0, 3.5, 4.0]]

Output =
[[2.39, 2.89, 3.50, 4.00],
 [2.50, 3.00, 3.50, 4.00]]
```

示例 2：

```text
N = 1, d_model = 2, h = 1
Q = [[1, 1]], K = [[1, 1]], V = [[2, 3]]
Output = [[2, 3]]
```

约束/性能点：`N <= 10000`，`d_model <= 1024`，`d_model % h == 0`，性能测试 `N=1024, d_model=1024`。

### 53. Causal Self-Attention

CUDA 训练重点：causal mask、三角区域计算、row-wise stable softmax、attention fusion。

题目描述：实现 causal self-attention：

```text
Attention_causal(Q,K,V) = softmax(masked(QK^T / sqrt(d))) V
```

其中位置 `i` 只能 attend 到 `j <= i` 的 key。

示例 1：

```text
Q = [[1,0,0,0],
     [0,1,0,0]]

K = [[1,0,0,0],
     [0,1,0,0]]

V = [[1,2,3,4],
     [5,6,7,8]]

Output =
[[1.0,       2.0,       3.0,       4.0],
 [3.4898374, 4.4898374, 5.4898374, 6.4898374]]
```

示例 2：

```text
Q = [[0,0],
     [1,1]]
K = [[1,0],
     [0,1]]
V = [[3,4],
     [5,6]]

Output =
[[3,4],
 [5,6]]
```

约束/性能点：`M <= 10000`，`d <= 128`，性能测试 `M = 5000`。

### 59. Sliding Window Self-Attention

CUDA 训练重点：稀疏/局部 attention、窗口边界、减少 O(N^2) 计算、locality。

题目描述：每个 query 只 attend 到 `[i-window_size, i+window_size]` 范围内的 key/value。

示例 1：

```text
Q = [[1,0,0,0],
     [0,1,0,0]]

K = [[1,0,0,0],
     [0,1,0,0]]

V = [[1,2,3,4],
     [5,6,7,8]]

window_size = 1

Output =
[[2.5101628, 3.5101628, 4.510163,  5.510163],
 [3.4898374, 4.4898376, 5.4898376, 6.489837]]
```

示例 2：

```text
Q = [[0,0,0],
     [0,1,0]]

K = [[1,0,0],
     [0,1,0]]

V = [[1,2,3],
     [5,6,7]]

window_size = 1

Output =
[[3.0,       4.0,      5.0],
 [3.5618298, 4.56183,  5.5618296]]
```

约束/性能点：矩阵 `Q,K,V` 为 `M x d`，适合练局部窗口和边界处不等长 softmax。

### 46. BFS Shortest Path

CUDA 训练重点：frontier、atomic、负载不均衡、不规则访存、图算法并行化。

题目描述：在有障碍的二维网格上做 BFS，返回从起点到终点的最短步数；无路返回 `-1`。网格 `0` 表示空地，`1` 表示障碍，只能上下左右移动。

示例 1：

```text
grid =
[[0, 0, 0, 0],
 [1, 1, 0, 1],
 [0, 0, 0, 0],
 [0, 1, 1, 0]]

start = (0,0), end = (3,3)
Output: 6
```

示例 2：

```text
grid =
[[0, 1, 0],
 [1, 1, 1],
 [0, 0, 0]]

start = (0,0), end = (0,2)
Output: -1
```

约束/性能点：`rows, cols <= 1000`，性能测试 `500 x 500`。

### 73. All-Pairs Shortest Paths

CUDA 训练重点：blocked Floyd-Warshall、tile 依赖、shared memory、二维矩阵更新。

题目描述：给定加权有向图的 `N x N` 距离矩阵，计算任意两点最短路。核心更新：

```text
output[i][j] = min(output[i][j], output[i][k] + output[k][j])
```

示例：

```text
N = 4
dist =
[[0,   5,   inf, 10 ],
 [inf, 0,   3,   inf],
 [inf, inf, 0,   1  ],
 [inf, inf, inf, 0  ]]

Output =
[[0,   5,   8,   9],
 [inf, 0,   3,   4],
 [inf, inf, 0,   1],
 [inf, inf, inf, 0]]
```

约束/性能点：`N <= 4096`，无负环，性能测试 `N = 2048`。

### 39. Fast Fourier Transform

CUDA 训练重点：蝶形计算、stride access、bit reversal、分阶段同步、复数数据布局。

题目描述：实现 1D complex FFT。输入和输出均为 interleaved real/imag：

```text
[real0, imag0, real1, imag1, ...]
```

示例 1：

```text
N = 4
signal = [1,0, 0,0, 0,0, 0,0]
Output = [1,0, 1,0, 1,0, 1,0]
```

示例 2：

```text
N = 2
signal = [1,0, 1,0]
Output = [2,0, 0,0]
```

约束/性能点：`N <= 262,144`，误差 `1e-3`，性能测试 `N = 262,144`。

### 74. GPT-2 Transformer Block

CUDA 训练重点：端到端 transformer block、LayerNorm、QKV projection、MHA、GELU FFN、residual、kernel fusion。

题目描述：实现一个 GPT-2 decoder block。输入 `x` 形状为 `(seq_len, 768)`，权重打包在一个连续 buffer 中。结构：

```text
x'     = x + MultiHeadAttn(LN1(x))
output = x' + FeedForward(LN2(x'))
```

关键子操作：

```text
LayerNorm epsilon = 1e-5
QKV projection: 768 -> 2304
12 heads, head_dim = 64
FFN: 768 -> 3072 -> 768
GELU 使用 tanh approximation
```

示例：

```text
Input:
  x.shape       = (4, 768)
  weights.shape = (7,087,872,)
  seq_len       = 4

Output:
  output.shape  = (4, 768)
```

约束/性能点：`1 <= seq_len <= 4096`，全部 `float32`，性能测试 `seq_len = 1024`。

### 93. Llama Transformer Block

CUDA 训练重点：RMSNorm、GQA、RoPE、causal attention、SwiGLU、LLM block fusion。

题目描述：实现一个 Llama-style decoder block。输入 `x` 形状为 `(seq_len, 512)`，权重打包，且给定预计算 RoPE `cos/sin`。结构：

```text
x'     = x + Attn(RMSNorm1(x), cos, sin)
output = x' + FFN(RMSNorm2(x'))
```

关键子操作：

```text
RMSNorm epsilon = 1e-5，无 bias
8 个 query heads，2 个 key/value heads
head_dim = 64
K/V heads repeat 4x 以匹配 Q heads
Q/K 应用 RoPE
causal mask
FFN 使用 SwiGLU: SiLU(gate) * up -> down
```

示例：

```text
Input:
  x.shape       = (4, 512)
  weights.shape = (2,819,072,)
  cos.shape     = (4, 32)
  sin.shape     = (4, 32)
  seq_len       = 4

Output:
  output.shape  = (4, 512)
```

约束/性能点：`1 <= seq_len <= 4096`，全部 `float32`，性能测试 `seq_len = 2048`。
