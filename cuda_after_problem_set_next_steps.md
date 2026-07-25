# CUDA 题单完成后的六步进阶路线

本文整理的是完成当前题单之后的进阶学习方向。假设已经完成：

```text
Reduction -> Prefix Sum -> Softmax -> GEMM -> 2D/3D Convolution
-> Top-K -> INT8/INT4 MatMul -> Sorting/Radix Sort
-> Attention -> BFS/APSP -> FFT -> GPT-2/Llama Transformer Block
```

完成这些题目后，重点不再是继续刷零散 CUDA 题，而是进入算子性能工程、框架理解和真实系统复现。

## 1. 算子性能工程

目标：从“能写出来”提升到“知道为什么快、为什么慢、离工业库差在哪里”。

重点内容：

- 对标 `cuBLAS`、`cuDNN`、`CUB`、`CUTLASS` 的性能。
- 系统使用 Nsight Systems 和 Nsight Compute。
- 分析 SM 利用率、memory throughput、L2 hit rate、occupancy、warp stall reason。
- 做 roofline analysis，判断 kernel 是 compute-bound 还是 memory-bound。
- 学习 `cp.async`、double buffering、persistent kernel、warp specialization。
- 针对不同 GPU 架构分别调参，例如 A100、H100、RTX 30/40 系列。

建议练习：

- 把手写 GEMM 和 `cuBLAS` 做性能对比。
- 把手写 softmax、scan、sort 和 CUB 做对比。
- 给每个 kernel 写 profiling 记录，解释主要瓶颈。

阶段产出：

- 一套 benchmark 脚本。
- 每个核心算子的 Nsight 报告。
- 一份性能分析笔记，说明瓶颈和优化依据。

## 2. CUTLASS / Triton / TVM

目标：理解现代高性能算子库和算子生成框架如何组织代码。

建议顺序：

```text
CUTLASS -> Triton -> TVM / MLIR / XLA
```

重点内容：

- tile hierarchy。
- threadblock / warp / MMA layout。
- Tensor Core MMA 数据布局。
- epilogue fusion。
- autotuning。
- kernel codegen。
- layout transformation。

学习方式：

- 不要只停留在调用 API。
- 先用手写 GEMM 的经验理解 CUTLASS 模板结构。
- 再用 Triton 重写 GEMM、softmax、attention。
- 最后再看 TVM、MLIR、XLA 这类编译器系统。

阶段产出：

- 一个 CUTLASS GEMM 示例和性能记录。
- 一个 Triton softmax / attention 实现。
- 一篇对比笔记：手写 CUDA、CUTLASS、Triton 的开发成本和性能差异。

## 3. LLM 推理系统

目标：从单个 CUDA kernel 扩展到完整 LLM 推理链路。

重点内容：

- KV cache 管理。
- paged attention。
- continuous batching。
- speculative decoding。
- tensor parallel。
- pipeline parallel。
- quantized inference。
- MoE routing。
- prefix cache。
- CUDA Graph 降低 launch overhead。

推荐参考方向：

```text
vLLM
TensorRT-LLM
SGLang
llama.cpp CUDA backend
FlashInfer
```

建议练习：

- 实现一个简化版 paged attention。
- 实现一个小型 continuous batching scheduler。
- 对比普通 attention、FlashAttention、paged attention 的内存访问差异。
- 做一个小型 Llama block 推理 benchmark。

阶段产出：

- 一个 mini LLM inference demo。
- 一个 KV cache 管理模块。
- 一个 attention kernel benchmark。
- 一份端到端延迟和吞吐分析。

## 4. 分布式 GPU 通信

目标：从单卡算子优化进入多卡训练和多卡推理。

重点内容：

- NCCL。
- AllReduce、AllGather、ReduceScatter。
- ring / tree 通信算法。
- compute-communication overlap。
- GPUDirect RDMA。
- tensor parallel。
- data parallel。
- sequence parallel。
- 多机多卡 profiling。

建议练习：

- 写 NCCL AllReduce benchmark。
- 比较不同 message size 下的带宽和延迟。
- 实现计算和通信重叠的小例子。
- 分析 tensor parallel 中 GEMM 和通信的时间占比。

阶段产出：

- NCCL benchmark。
- 单机多卡通信分析报告。
- 一个 compute-communication overlap demo。

## 5. GPU 编译器与底层 ISA

目标：理解 CUDA 代码最终如何映射到 PTX、SASS 和硬件执行。

重点内容：

- PTX。
- SASS。
- `cuobjdump`。
- `nvdisasm`。
- register allocation。
- instruction scheduling。
- memory barrier。
- warp-level execution。
- Tensor Core MMA 指令。

建议练习：

- 对一个简单 reduction kernel 反汇编，观察 PTX 和 SASS。
- 对比不同编译选项下的寄存器数量和指令变化。
- 分析一个 WMMA kernel 对应的 MMA 指令。
- 观察 shared memory 访问、global memory load/store、barrier 指令。

阶段产出：

- 一份 PTX/SASS 阅读笔记。
- 一个 kernel 编译参数对性能影响的实验。
- 一个 Tensor Core 指令级分析记录。

## 6. 真实项目复现

目标：把题目里的单点能力整合成可维护、可测试、可 profiling 的工程项目。

推荐项目：

```text
实现 mini-CUTLASS GEMM
实现 mini-FlashAttention
实现 PyTorch CUDA extension 算子库
实现简化版 vLLM scheduler + paged attention
实现 INT4 Llama inference kernel
实现 NCCL AllReduce benchmark
```

项目要求：

- 有清晰的代码结构。
- 有 correctness test。
- 有 benchmark。
- 有 Nsight profiling 报告。
- 有和主流库或 baseline 的性能对比。
- 有 README 说明设计、限制和结果。

建议优先级：

```text
mini-FlashAttention
-> PyTorch CUDA extension 算子库
-> INT4 Llama inference kernel
-> 简化版 vLLM scheduler + paged attention
-> NCCL AllReduce benchmark
```

阶段产出：

- 一个可运行的项目仓库。
- 一组端到端 benchmark。
- 一份工程复盘文档。

## 推荐总路线

如果目标是 AI 算子和 LLM 推理加速，推荐按下面的顺序推进：

```text
单算子极限优化
-> CUTLASS / Triton
-> FlashAttention / LLM 推理
-> PyTorch Extension
-> 多 GPU / NCCL
-> 编译器 / PTX / SASS
```

如果只选择一个最值得投入的方向，建议从 GEMM 和 Attention 出发，做一个小型 LLM 推理加速项目。它能把 CUDA kernel、Tensor Core、量化、KV cache、profiling、多 stream 和系统设计串起来。
