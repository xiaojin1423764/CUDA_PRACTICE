# MPI 从入门到精通学习路线

本文整理 MPI 的系统学习路线，目标是从能写基础并行程序，逐步过渡到能分析通信性能、设计分布式算法，并理解 MPI 与 CUDA/NCCL 在多机多卡系统中的关系。

MPI 不是一个具体库，而是一套分布式内存并行编程标准。常见实现包括 Open MPI、MPICH、Intel MPI、MVAPICH 等。它主要用于多进程、多节点通信，是 HPC、科学计算、大模型训练和多机推理系统里的基础组件。

## 总体路线

推荐学习顺序：

```text
MPI 基础模型
-> 点对点通信
-> 集合通信
-> 非阻塞通信
-> 通信与计算重叠
-> 派生数据类型
-> communicator / group / topology
-> 并行算法与 domain decomposition
-> MPI-IO
-> MPI + OpenMP / CUDA / NCCL
-> 性能分析与工程化
```

## 1. MPI 基础模型

目标：理解 MPI 程序的基本运行方式。

重点内容：

- 进程模型：MPI 程序通常是多进程并行，而不是多线程并行。
- `rank`：当前进程在 communicator 中的编号。
- `size`：communicator 中的进程总数。
- `MPI_COMM_WORLD`：默认包含所有进程的 communicator。
- `mpicc` / `mpicxx`：MPI 编译器包装器。
- `mpirun` / `mpiexec`：启动 MPI 程序。

最小程序结构：

```cpp
#include <mpi.h>
#include <cstdio>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::printf("Hello from rank %d / %d\n", rank, size);

  MPI_Finalize();
  return 0;
}
```

编译运行：

```bash
mpicxx hello.cpp -o hello
mpirun -np 4 ./hello
```

阶段要求：

- 能解释每个进程执行的是同一份程序。
- 能理解 SPMD 模型，即 Single Program Multiple Data。
- 能用 `rank` 区分不同进程的工作。

## 2. 点对点通信

目标：掌握两个进程之间的数据发送和接收。

核心 API：

- `MPI_Send`
- `MPI_Recv`
- `MPI_Isend`
- `MPI_Irecv`
- `MPI_Wait`
- `MPI_Waitall`
- `MPI_Probe`

重点概念：

- source / destination。
- tag。
- communicator。
- blocking send/recv。
- non-blocking send/recv。
- deadlock。
- message matching。

必须掌握的问题：

- 为什么两个进程互相 `MPI_Send` 可能死锁。
- `MPI_Send` 和 `MPI_Isend` 的区别。
- `MPI_Recv` 如何通过 source、tag、communicator 匹配消息。
- buffer 在非阻塞通信完成前为什么不能随便修改。

建议练习：

- rank 0 向 rank 1 发送一个数组。
- 相邻 rank 做 ring communication。
- 实现 ping-pong latency benchmark。
- 实现 bandwidth benchmark。

阶段产出：

- 一个点对点通信 demo。
- 一个 latency/bandwidth benchmark。
- 一份死锁案例和修复方式记录。

## 3. 集合通信

目标：掌握多个进程之间的常见通信模式。

核心 API：

- `MPI_Bcast`
- `MPI_Reduce`
- `MPI_Allreduce`
- `MPI_Gather`
- `MPI_Allgather`
- `MPI_Scatter`
- `MPI_Alltoall`
- `MPI_Reduce_scatter`
- `MPI_Barrier`

重点概念：

- root process。
- collective operation。
- 所有进程必须以一致顺序调用集合通信。
- 集合通信通常比手写点对点通信更容易优化。

必须掌握的问题：

- `Reduce` 和 `Allreduce` 的区别。
- `Gather` 和 `Allgather` 的区别。
- `Scatter` 和 `Bcast` 的区别。
- 为什么集合通信调用顺序不一致会导致 hang。
- `MPI_Barrier` 什么时候有必要，什么时候会伤害性能。

建议练习：

- 用 `MPI_Reduce` 计算全局 sum/max/min。
- 用 `MPI_Allreduce` 实现分布式 dot product。
- 用 `MPI_Scatter` + local compute + `MPI_Gather` 实现简单数据并行。
- 对比手写 ring allreduce 和 `MPI_Allreduce`。

阶段产出：

- 常见 collective API 示例。
- 一个分布式向量求和程序。
- 一个 AllReduce benchmark。

## 4. 非阻塞通信与通信计算重叠

目标：减少等待时间，把通信隐藏在计算后面。

核心 API：

- `MPI_Isend`
- `MPI_Irecv`
- `MPI_Test`
- `MPI_Wait`
- `MPI_Waitall`
- `MPI_Waitsome`

重点内容：

- 提前 post receive。
- 计算 interior region，同时交换 boundary / halo。
- 通信完成后再计算 boundary region。
- 避免过早 `MPI_Wait` 破坏 overlap。

典型模式：

```text
post Irecv
post Isend
compute independent work
wait communication
compute dependent work
```

建议练习：

- 实现 1D stencil 的 halo exchange。
- 实现 2D stencil 的上下左右边界交换。
- 对比阻塞通信和非阻塞通信版本耗时。
- 用 profiling 判断是否真的产生 overlap。

阶段产出：

- 一个 halo exchange demo。
- 一份 blocking vs non-blocking 的性能对比。

## 5. 派生数据类型

目标：优雅表达非连续内存布局，减少手动 pack/unpack。

核心 API：

- `MPI_Type_contiguous`
- `MPI_Type_vector`
- `MPI_Type_create_subarray`
- `MPI_Type_commit`
- `MPI_Type_free`

适用场景：

- 发送矩阵的一列。
- 发送 2D/3D 子块。
- 发送结构体中的部分字段。
- stencil / domain decomposition 中的边界区域。

必须掌握的问题：

- extent 和 true extent 的区别。
- 为什么 datatype 创建后要 `MPI_Type_commit`。
- derived datatype 是否一定比手动 pack 更快。

建议练习：

- 用 `MPI_Type_vector` 发送矩阵列。
- 用 `MPI_Type_create_subarray` 发送 2D tile。
- 对比 derived datatype 和手动 pack/unpack。

阶段产出：

- 一个矩阵边界交换 demo。
- 一份 datatype 性能对比记录。

## 6. Communicator、Group 与拓扑

目标：管理复杂并行程序里的进程组织。

核心 API：

- `MPI_Comm_split`
- `MPI_Comm_dup`
- `MPI_Comm_create`
- `MPI_Cart_create`
- `MPI_Cart_shift`
- `MPI_Comm_free`

重点内容：

- 把全局进程划分成多个子通信域。
- 为不同并行维度创建不同 communicator。
- 使用笛卡尔拓扑管理网格邻居。
- 在多维 domain decomposition 中简化邻居通信。

建议练习：

- 把 16 个进程分成 4 组，每组内部做 `Allreduce`。
- 用 `MPI_Cart_create` 创建 2D 进程网格。
- 用 `MPI_Cart_shift` 找上下左右邻居。

阶段产出：

- 一个 2D Cartesian communicator demo。
- 一个分组 collective demo。

## 7. 并行算法与 Domain Decomposition

目标：从 API 学习进入真实并行算法设计。

重点算法：

- parallel reduction。
- distributed prefix sum。
- stencil computation。
- distributed matrix-vector multiplication。
- distributed matrix multiplication。
- parallel sorting。
- graph traversal。
- iterative solver，例如 Jacobi、CG。

重点思想：

- 数据如何切分。
- 每个 rank 存哪些数据。
- 哪些数据需要通信。
- 通信频率和通信量是多少。
- load balance 是否合理。
- strong scaling 和 weak scaling。

建议练习：

- 1D/2D heat equation。
- 分布式矩阵向量乘。
- Cannon / SUMMA 矩阵乘。
- 分布式 Jacobi solver。

阶段产出：

- 一个完整 domain decomposition 项目。
- strong scaling / weak scaling 曲线。
- 一份通信量分析。

## 8. MPI-IO

目标：理解多进程并行读写文件，避免所有进程都挤到 rank 0。

核心 API：

- `MPI_File_open`
- `MPI_File_read_at`
- `MPI_File_write_at`
- `MPI_File_set_view`
- `MPI_File_read_all`
- `MPI_File_write_all`
- `MPI_File_close`

重点内容：

- independent I/O。
- collective I/O。
- file view。
- offset。
- 并行文件系统。

建议练习：

- 每个 rank 写自己的数据到同一个文件不同 offset。
- 用 file view 写 2D 子块。
- 对比 rank 0 汇总写文件和 MPI-IO collective write。

阶段产出：

- 一个并行写文件 demo。
- 一个 I/O bandwidth benchmark。

## 9. MPI + OpenMP / CUDA / NCCL

目标：理解现代 HPC 和 AI 系统里的混合并行。

常见组合：

```text
MPI + OpenMP：多节点 + 单节点多核 CPU
MPI + CUDA：多节点 + 单节点 GPU
MPI + NCCL：多节点多 GPU 通信
MPI + CUDA-aware MPI：直接传 GPU buffer
```

重点内容：

- 每个 MPI rank 绑定一个 GPU。
- rank 到 GPU 的映射关系。
- CPU buffer 通信和 GPU buffer 通信的区别。
- CUDA-aware MPI。
- GPUDirect RDMA。
- MPI 负责跨节点控制和部分通信，NCCL 负责 GPU collective。
- 多进程和多线程混用时的亲和性绑定。

典型启动方式：

```bash
mpirun -np 8 ./program
```

程序中常见 GPU 绑定方式：

```cpp
int local_rank = 0;  // 通常从环境变量读取
cudaSetDevice(local_rank);
```

建议练习：

- 每个 MPI rank 绑定一张 GPU。
- 每个 rank 在 GPU 上做 local reduction，再用 `MPI_Allreduce` 合并。
- 测试 CUDA-aware MPI 是否支持直接传 GPU pointer。
- 对比 `MPI_Allreduce` 和 `NCCL AllReduce`。

阶段产出：

- 一个 MPI + CUDA demo。
- 一个 GPU buffer 通信测试。
- 一份 MPI 和 NCCL 的职责对比笔记。

## 10. 性能分析与工程化

目标：能定位 MPI 程序慢在哪里，并把程序组织成可维护工程。

重点指标：

- latency。
- bandwidth。
- message size。
- communication/computation ratio。
- load imbalance。
- synchronization overhead。
- strong scaling efficiency。
- weak scaling efficiency。

常用工具方向：

- MPI profiling interface，PMPI。
- mpiP。
- TAU。
- Score-P。
- Vampir。
- Nsight Systems，适合 MPI + CUDA 程序。
- 实现自己的轻量计时器和日志。

工程要求：

- 所有 MPI 调用检查返回值。
- benchmark 区分 warmup 和正式计时。
- 记录进程数、节点数、线程数、GPU 数、输入规模。
- 不要在计时区频繁打印。
- 避免让 rank 0 成为不必要瓶颈。
- 对 collective 调用顺序保持严格一致。
- 对非阻塞通信的 request 生命周期进行清晰管理。

阶段产出：

- 一个 MPI benchmark 框架。
- strong/weak scaling 报告。
- 一份性能瓶颈分析。

## 推荐项目路线

建议按下面的项目逐步推进：

```text
Hello MPI
-> ping-pong latency/bandwidth benchmark
-> distributed vector reduction
-> 1D/2D halo exchange
-> 2D heat equation
-> distributed matrix-vector multiplication
-> distributed matrix multiplication
-> MPI-IO parallel writer
-> MPI + CUDA local compute
-> MPI + CUDA/NCCL multi-GPU benchmark
```

## 掌握深度要求

第一层：能写基础程序

- 会初始化和结束 MPI。
- 会获取 rank/size。
- 会使用 send/recv 和常见 collective。
- 能运行多进程程序。

第二层：能设计通信模式

- 会选择点对点通信还是集合通信。
- 能避免常见死锁。
- 能写非阻塞通信。
- 能做 halo exchange。
- 能解释通信量和同步点。

第三层：能做性能分析

- 能测 latency 和 bandwidth。
- 能做 strong/weak scaling。
- 能识别 load imbalance。
- 能判断通信是否成为瓶颈。
- 能通过 overlap、数据划分、collective 优化改进性能。

第四层：能做真实系统集成

- 能写 MPI + CUDA 程序。
- 能理解 MPI 和 NCCL 的分工。
- 能管理多节点多 GPU rank 映射。
- 能用 profiling 工具分析端到端性能。
- 能把并行算法组织成可维护项目。

## 与 CUDA 学习路线的关系

CUDA 主要解决单 GPU 或单节点多 GPU 内部的计算效率问题。MPI 主要解决多进程、多节点之间的数据分发和通信问题。

在 AI/HPC 系统中，两者通常这样分工：

```text
CUDA：单个 rank 内部的 GPU kernel 计算
NCCL：GPU collective，尤其是多 GPU AllReduce / AllGather / ReduceScatter
MPI：多进程启动、跨节点通信、任务划分、控制流、部分 CPU/GPU buffer 通信
```

如果目标是多机多卡训练或推理，推荐学习顺序：

```text
CUDA 单卡算子
-> NCCL collective
-> MPI 基础通信
-> MPI + CUDA
-> 多节点多 GPU benchmark
-> 分布式训练 / 推理系统
```

## 最小验收清单

完成 MPI 入门到进阶后，至少应该能独立完成：

- 写一个无死锁的点对点通信程序。
- 写一个使用 `MPI_Allreduce` 的分布式 reduction。
- 写一个非阻塞 halo exchange。
- 写一个 2D domain decomposition 程序。
- 写一个 MPI-IO 并行写文件程序。
- 写一个 MPI + CUDA 多 GPU demo。
- 测量 latency、bandwidth、strong scaling、weak scaling。
- 解释程序瓶颈来自计算、通信、同步还是负载不均衡。

真正进入熟练阶段的标志是：看到一个分布式问题时，能先判断数据怎么切、通信怎么走、同步在哪里、性能瓶颈大概会出现在哪，而不是只会调用几个 MPI API。
