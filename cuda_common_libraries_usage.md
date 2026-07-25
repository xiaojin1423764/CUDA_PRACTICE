# CUDA 常用库用法速查

本文按日常 CUDA 开发中最常见的库整理：每节包含用途、头文件、链接方式、典型调用流程和容易踩坑的点。示例以 C++/CUDA 为主，默认使用 `nvcc` 编译。

## 编译和链接基础

最小 CUDA 程序：

```bash
nvcc -O3 -arch=sm_80 main.cu -o main
```

常用参数：

- `-O3`：开启优化。
- `-arch=sm_80`：指定目标 GPU 架构，A100 常用 `sm_80`，RTX 30 系列常用 `sm_86`，H100 常用 `sm_90`。
- `-lineinfo`：保留源码行号，便于 Nsight Compute / Nsight Systems 分析。
- `-Xcompiler -fopenmp`：把参数传给 host 编译器。
- `-I<path>`：添加头文件路径。
- `-L<path>`：添加库路径。
- `-l<name>`：链接库，例如 `-lcublas`。

推荐调试宏：

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)
```

kernel launch 后建议检查：

```cpp
my_kernel<<<grid, block>>>(args...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## CUDA Runtime API

用途：设备管理、显存分配、数据拷贝、stream/event、kernel launch。大多数 CUDA C++ 程序都直接使用 Runtime API。

头文件：

```cpp
#include <cuda_runtime.h>
```

链接：通常由 `nvcc` 自动链接 `cudart`。手动链接时使用：

```bash
nvcc main.cu -lcudart -o main
```

常用流程：

```cpp
float *d_x = nullptr;
size_t bytes = n * sizeof(float);

CUDA_CHECK(cudaSetDevice(0));
CUDA_CHECK(cudaMalloc(&d_x, bytes));
CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

kernel<<<grid, block>>>(d_x, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

CUDA_CHECK(cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaFree(d_x));
```

常用 API：

- `cudaGetDeviceCount` / `cudaGetDeviceProperties`：查询 GPU。
- `cudaMalloc` / `cudaFree`：设备显存。
- `cudaMallocHost` / `cudaFreeHost`：page-locked host memory，用于更高效异步拷贝。
- `cudaMemcpy` / `cudaMemcpyAsync`：同步/异步拷贝。
- `cudaStreamCreate` / `cudaStreamDestroy`：创建 stream。
- `cudaEventCreate` / `cudaEventRecord` / `cudaEventElapsedTime`：计时。
- `cudaMemset` / `cudaMemsetAsync`：初始化设备内存。

注意：

- `cudaMemcpyAsync` 要真正和计算重叠，host 端内存通常需要是 pinned memory。
- 默认 stream 的同步语义容易影响并发，复杂程序中建议显式创建 stream。
- `cudaMalloc` / `cudaFree` 开销较高，性能敏感路径应复用 buffer 或使用内存池。

## CUDA Driver API

用途：更底层地管理 CUDA context、module、function 和 device memory。多数业务代码不需要直接使用，但插件系统、JIT、框架 runtime 会用到。

头文件：

```cpp
#include <cuda.h>
```

链接：

```bash
nvcc main.cu -lcuda -o main
```

基本流程：

```cpp
CUdevice dev;
CUcontext ctx;

cuInit(0);
cuDeviceGet(&dev, 0);
cuCtxCreate(&ctx, 0, dev);

// cuModuleLoad, cuModuleGetFunction, cuLaunchKernel ...

cuCtxDestroy(ctx);
```

注意：

- Runtime API 和 Driver API 可以混用，但 context 管理要清楚。
- Driver API 返回 `CUresult`，错误处理和 Runtime API 不同。

## cuBLAS

用途：BLAS 线性代数库，常用于向量运算、矩阵乘法、batched GEMM、Tensor Core GEMM。

头文件：

```cpp
#include <cublas_v2.h>
```

链接：

```bash
nvcc main.cu -lcublas -o main
```

常用流程：

```cpp
cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f;
float beta = 0.0f;

// C = alpha * A * B + beta * C
// cuBLAS 默认按 column-major 理解矩阵。
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            A, m,
            B, k,
            &beta,
            C, m);

cublasDestroy(handle);
```

常用 API：

- `cublasSgemm` / `cublasDgemm`：FP32 / FP64 GEMM。
- `cublasGemmEx`：混合精度 GEMM，可用 Tensor Core。
- `cublasLtMatmul`：更灵活的 GEMM 接口，适合调优 layout、epilogue、算法选择。
- `cublasSaxpy` / `cublasSdot` / `cublasSnrm2`：常见向量操作。

注意：

- cuBLAS 默认 column-major；C/C++ row-major 矩阵常需要交换 A/B、转置标记或调整 leading dimension。
- 要使用 Tensor Core，数据类型、对齐、维度和 math mode 都要满足条件。
- 多 stream 程序中使用 `cublasSetStream(handle, stream)`。

## cuDNN

用途：深度学习算子库，包含 convolution、activation、normalization、RNN、attention 等。通常被 PyTorch/TensorFlow 调用，手写推理或算子验证时也会直接用。

头文件：

```cpp
#include <cudnn.h>
```

链接：

```bash
nvcc main.cu -lcudnn -o main
```

典型流程：

```cpp
cudnnHandle_t handle;
cudnnCreate(&handle);

cudnnTensorDescriptor_t x_desc;
cudnnCreateTensorDescriptor(&x_desc);
cudnnSetTensor4dDescriptor(x_desc,
                           CUDNN_TENSOR_NCHW,
                           CUDNN_DATA_FLOAT,
                           n, c, h, w);

// 创建 filter / convolution / output descriptor
// 查询算法和 workspace
// 调用 cudnnConvolutionForward

cudnnDestroyTensorDescriptor(x_desc);
cudnnDestroy(handle);
```

常用对象：

- `cudnnHandle_t`：库句柄。
- `cudnnTensorDescriptor_t`：tensor 形状、layout、数据类型。
- `cudnnFilterDescriptor_t`：卷积核描述。
- `cudnnConvolutionDescriptor_t`：stride、padding、dilation、卷积模式。

注意：

- descriptor 必须和实际内存 layout 一致。
- 大多数高性能算法需要额外 workspace。
- 多 stream 程序中使用 `cudnnSetStream(handle, stream)`。

## cuFFT

用途：FFT / IFFT，支持 1D、2D、3D、batched FFT，常用于信号处理、频域卷积、谱方法。

头文件：

```cpp
#include <cufft.h>
```

链接：

```bash
nvcc main.cu -lcufft -o main
```

1D complex-to-complex 示例：

```cpp
cufftHandle plan;
cufftPlan1d(&plan, n, CUFFT_C2C, batch);

// d_data 类型通常是 cufftComplex*
cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

cufftDestroy(plan);
```

常用 API：

- `cufftPlan1d` / `cufftPlan2d` / `cufftPlan3d`：简单 plan。
- `cufftPlanMany`：batched、多维、复杂 stride 场景。
- `cufftExecC2C`：complex-to-complex。
- `cufftExecR2C` / `cufftExecC2R`：real/complex 互转。

注意：

- cuFFT 不自动做归一化，反变换后通常需要自己除以元素数。
- plan 创建有开销，循环中应复用 plan。
- 多 stream 程序中使用 `cufftSetStream(plan, stream)`。

## Thrust

用途：类似 C++ STL 的并行算法库，适合快速写 sort、scan、reduce、transform、copy 等操作。

头文件：

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
```

链接：通常不需要额外库，使用 `nvcc` 编译即可。

示例：

```cpp
thrust::device_vector<float> x(n);

thrust::fill(x.begin(), x.end(), 1.0f);
float sum = thrust::reduce(x.begin(), x.end(), 0.0f, thrust::plus<float>());
thrust::exclusive_scan(x.begin(), x.end(), x.begin());
thrust::sort(x.begin(), x.end());
```

和原始指针配合：

```cpp
thrust::device_ptr<float> p = thrust::device_pointer_cast(d_x);
float sum = thrust::reduce(p, p + n, 0.0f);
```

注意：

- Thrust 很适合原型和中等复杂度逻辑，但对极致性能和内存分配控制不如手写 kernel / CUB。
- 在已有 stream 上运行时，可使用 execution policy，例如 `thrust::cuda::par.on(stream)`。

## CUB

用途：高性能 CUDA primitives，包括 block/warp/device 级 reduce、scan、sort、histogram、select 等。很多基础并行算法优先考虑 CUB。

头文件：

```cpp
#include <cub/cub.cuh>
```

链接：通常不需要额外库，使用 `nvcc` 编译即可。

DeviceReduce 示例：

```cpp
void *temp_storage = nullptr;
size_t temp_bytes = 0;

cub::DeviceReduce::Sum(temp_storage, temp_bytes, d_in, d_out, n);
CUDA_CHECK(cudaMalloc(&temp_storage, temp_bytes));
cub::DeviceReduce::Sum(temp_storage, temp_bytes, d_in, d_out, n);

CUDA_CHECK(cudaFree(temp_storage));
```

BlockReduce 示例：

```cpp
template <int BLOCK_SIZE>
__global__ void reduce_kernel(const float *x, float *partial, int n) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? x[i] : 0.0f;
  float sum = BlockReduce(temp).Sum(v);

  if (threadIdx.x == 0) {
    partial[blockIdx.x] = sum;
  }
}
```

注意：

- Device 级 API 通常先调用一次获取临时空间大小，再分配 workspace，再执行。
- Block/Warp 级 API 适合嵌入自定义 kernel。
- CUB 比 Thrust 更贴近底层，也更适合性能敏感代码。

## NCCL

用途：多 GPU / 多机 GPU 通信，常用于深度学习分布式训练中的 all-reduce、broadcast、all-gather、reduce-scatter。

头文件：

```cpp
#include <nccl.h>
```

链接：

```bash
nvcc main.cu -lnccl -o main
```

单进程多 GPU all-reduce 简化流程：

```cpp
ncclComm_t comms[num_gpus];
int devices[num_gpus] = {0, 1};

ncclCommInitAll(comms, num_gpus, devices);

for (int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(devices[i]);
  ncclAllReduce(sendbuff[i], recvbuff[i], count,
                ncclFloat, ncclSum, comms[i], streams[i]);
}

for (int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(devices[i]);
  cudaStreamSynchronize(streams[i]);
  ncclCommDestroy(comms[i]);
}
```

常用 API：

- `ncclAllReduce`：所有 GPU 求和/最大值等，再广播结果。
- `ncclBroadcast`：一个 rank 广播给所有 rank。
- `ncclAllGather`：每个 rank 收集所有 rank 的数据。
- `ncclReduceScatter`：reduce 后按 rank 切分。
- `ncclGroupStart` / `ncclGroupEnd`：把多次 NCCL 调用打包提交。

注意：

- NCCL 调用是异步入 stream 的，错误和阻塞常在后续同步时暴露。
- 多进程多机需要 rank、unique id、通信初始化，通常由 MPI、torchrun 或框架管理。
- 每个 rank 的 device、stream、buffer 要对应清楚。

## cuRAND

用途：GPU 上生成随机数，支持 uniform、normal、log-normal、Poisson，也支持在 kernel 内使用 RNG state。

头文件：

```cpp
#include <curand.h>
#include <curand_kernel.h>
```

链接：

```bash
nvcc main.cu -lcurand -o main
```

host API 示例：

```cpp
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

curandGenerateUniform(gen, d_x, n);

curandDestroyGenerator(gen);
```

device API 示例：

```cpp
__global__ void rng_kernel(float *out, int n, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  out[i] = curand_uniform(&state);
}
```

注意：

- 每个线程初始化 RNG state 有成本，频繁调用时应持久化 state。
- `curand_uniform` 返回 `(0, 1]`，不是 `[0, 1)`。
- 需要可复现实验时固定 seed，并明确 sequence / offset 策略。

## cuSPARSE

用途：稀疏矩阵运算，包含 SpMV、SpMM、稀疏格式转换等。

头文件：

```cpp
#include <cusparse.h>
```

链接：

```bash
nvcc main.cu -lcusparse -o main
```

SpMV 典型流程：

```cpp
cusparseHandle_t handle;
cusparseCreate(&handle);

cusparseSpMatDescr_t matA;
cusparseDnVecDescr_t vecX, vecY;

cusparseCreateCsr(&matA,
                  rows, cols, nnz,
                  d_csr_offsets,
                  d_csr_columns,
                  d_csr_values,
                  CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO,
                  CUDA_R_32F);

cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);

float alpha = 1.0f;
float beta = 0.0f;
size_t buffer_size = 0;
void *buffer = nullptr;

cusparseSpMV_bufferSize(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        &buffer_size);
cudaMalloc(&buffer, buffer_size);

cusparseSpMV(handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matA, vecX, &beta, vecY,
             CUDA_R_32F,
             CUSPARSE_SPMV_ALG_DEFAULT,
             buffer);

cudaFree(buffer);
cusparseDestroyDnVec(vecY);
cusparseDestroyDnVec(vecX);
cusparseDestroySpMat(matA);
cusparseDestroy(handle);
```

注意：

- 新版 cuSPARSE 推荐使用 generic API，如 `cusparseSpMV` / `cusparseSpMM`。
- 稀疏格式、索引类型和 index base 必须和输入一致。
- 很多 API 需要 workspace，流程和 CUB 类似。

## NVTX

用途：给代码区间打标记，在 Nsight Systems / Nsight Compute 中更容易看 timeline 和关联业务逻辑。

头文件：

```cpp
#include <nvtx3/nvToolsExt.h>
```

链接：

```bash
nvcc main.cu -lnvToolsExt -o main
```

示例：

```cpp
nvtxRangePushA("H2D copy");
cudaMemcpyAsync(d_x, h_x, bytes, cudaMemcpyHostToDevice, stream);
nvtxRangePop();

nvtxRangePushA("kernel");
kernel<<<grid, block, 0, stream>>>(d_x, n);
nvtxRangePop();
```

注意：

- NVTX 标记本身不替代同步；它只是 profiling 标记。
- 粗粒度阶段标记通常比每个小函数都标记更有用。

## CUTLASS

用途：NVIDIA 开源 CUDA 模板库，主要用于高性能 GEMM、convolution、Tensor Core kernel 组合和定制。适合学习工业级矩阵乘法实现，也适合定制 fused GEMM。

头文件路径依赖 CUTLASS 安装位置，常见形式：

```cpp
#include <cutlass/gemm/device/gemm.h>
```

编译示例：

```bash
nvcc -O3 -arch=sm_80 \
  -I/path/to/cutlass/include \
  -I/path/to/cutlass/tools/util/include \
  main.cu -o main
```

简化示例：

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float>;

Gemm gemm_op;
Gemm::Arguments args({M, N, K},
                     {A, K},
                     {B, N},
                     {C, N},
                     {C, N},
                     {alpha, beta});

cutlass::Status status = gemm_op(args);
```

注意：

- CUTLASS 类型参数很多，优先从官方 examples 或已有工程模板改。
- layout、leading dimension、元素类型、accumulator 类型要严格匹配。
- 如果只是调用标准 GEMM，cuBLAS / cuBLASLt 通常更省事；需要定制 epilogue 或研究 kernel 时再考虑 CUTLASS。

## WMMA

用途：直接在 CUDA kernel 中使用 Tensor Core 的 warp-level matrix multiply-accumulate。适合学习 Tensor Core 或写小型定制 matmul。

头文件：

```cpp
#include <mma.h>
using namespace nvcuda;
```

编译：

```bash
nvcc -O3 -arch=sm_80 main.cu -o main
```

基本结构：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
```

注意：

- WMMA 是 warp 级操作，通常一个 warp 负责一个或多个 tile。
- 输入矩阵维度、对齐和 layout 要满足 Tensor Core 要求。
- 工程上写高性能 GEMM 时，tiling、shared memory、寄存器复用和流水都要配合设计；WMMA 只是核心计算指令接口。

## 常用库选择建议

| 任务 | 首选库 | 说明 |
| --- | --- | --- |
| 显存管理、stream、event | CUDA Runtime API | 大多数程序的基础接口 |
| 标准 GEMM / BLAS | cuBLAS / cuBLASLt | 优先用成熟库，通常比手写快 |
| 深度学习卷积/归一化/激活 | cuDNN | 框架底层常用库 |
| FFT | cuFFT | plan 要复用 |
| sort / scan / reduce 原型 | Thrust | 写法简单，接近 STL |
| 高性能基础并行 primitive | CUB | 更适合性能敏感代码 |
| 多 GPU 通信 | NCCL | 分布式训练和多卡同步常用 |
| 随机数 | cuRAND | host API 简单，device API 灵活 |
| 稀疏矩阵 | cuSPARSE | SpMV / SpMM / 格式转换 |
| profiling 标记 | NVTX | 配合 Nsight 看 timeline |
| 定制 Tensor Core GEMM | CUTLASS / WMMA | CUTLASS 工程化，WMMA 更底层 |

## 常见开发流程

1. 先用 CUDA Runtime API 写清楚数据分配、拷贝、kernel launch 和同步。
2. 遇到标准矩阵乘法、FFT、稀疏矩阵、随机数、通信等需求时，优先调用成熟库。
3. 对业务阶段加 NVTX 标记。
4. 用 Nsight Systems 找整体瓶颈，用 Nsight Compute 看目标 kernel。
5. 只有成熟库不满足融合、layout、访存或延迟需求时，再手写 kernel、CUB primitive、CUTLASS 或 WMMA。

## 常见链接参数汇总

```bash
# Runtime API
nvcc main.cu -lcudart -o main

# cuBLAS
nvcc main.cu -lcublas -o main

# cuDNN
nvcc main.cu -lcudnn -o main

# cuFFT
nvcc main.cu -lcufft -o main

# NCCL
nvcc main.cu -lnccl -o main

# cuRAND
nvcc main.cu -lcurand -o main

# cuSPARSE
nvcc main.cu -lcusparse -o main

# NVTX
nvcc main.cu -lnvToolsExt -o main
```

实际项目中常组合链接：

```bash
nvcc -O3 -arch=sm_80 -lineinfo main.cu \
  -lcublas -lcudnn -lcufft -lcurand -lcusparse -lnccl -lnvToolsExt \
  -o main
```

如果库不在默认路径，增加：

```bash
nvcc main.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -o main
```
