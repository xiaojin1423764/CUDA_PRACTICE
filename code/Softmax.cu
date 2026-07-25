#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>

// 统一检查 CUDA Runtime API 的返回值，出错时打印位置并退出。
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 4;

// warp 内求和归约。
// 每个线程先持有一个局部值，通过 shuffle 指令读取同 warp 中其他线程的值。
// 归约完成后，lane 0 持有整个 warp 的 sum。
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// warp 内最大值归约。
// softmax 需要先求全局 max，用 max trick 计算 exp(x - max)，避免指数溢出。
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

// block 内求和归约。
// 1. 每个 warp 用 warp_reduce_sum 得到一个 warp sum。
// 2. 每个 warp 的 lane 0 把结果写入 shared memory。
// 3. 第一个 warp 再把所有 warp sum 归约成 block sum。
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float warp_sums[BLOCK_SIZE / 32];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) {
    warp_sums[warp] = v;
  }
  __syncthreads();

  v = (threadIdx.x < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    v = warp_reduce_sum(v);
  }
  return v;
}

// block 内最大值归约。
// 结构和 block_reduce_sum 相同，区别是归约操作为 max，
// 非有效线程使用 -FLT_MAX 作为中性值。
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float warp_maxes[BLOCK_SIZE / 32];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_max(v);
  if (lane == 0) {
    warp_maxes[warp] = v;
  }
  __syncthreads();

  v = (threadIdx.x < BLOCK_SIZE / 32) ? warp_maxes[lane] : -FLT_MAX;
  if (warp == 0) {
    v = warp_reduce_max(v);
  }
  return v;
}

// 第一阶段：全局 max 的 block 级 partial reduction。
//
// 每个 block 通过 grid-stride loop 处理输入数组的一部分。
// 每个线程一次负责 kItemsPerThread 个 float；主路径使用 float4 合并读取。
// 每个 block 最终输出一个 partial max 到 partial_max[blockIdx.x]。
__global__ void max_first_pass_kernel(const float *__restrict__ input,
                                      float *__restrict__ partial_max,
                                      int n) {
  int tid = threadIdx.x;
  int base = (blockIdx.x * blockDim.x + tid) * kItemsPerThread;
  int stride = gridDim.x * blockDim.x * kItemsPerThread;

  float local_max = -FLT_MAX;
  for (int i = base; i < n; i += stride) {
    if (i + 3 < n) {
      float4 v = *reinterpret_cast<const float4 *>(input + i);
      local_max = fmaxf(local_max, v.x);
      local_max = fmaxf(local_max, v.y);
      local_max = fmaxf(local_max, v.z);
      local_max = fmaxf(local_max, v.w);
    } else {
// 尾部不足 4 个元素时逐个处理，避免越界读取。
#pragma unroll
      for (int j = 0; j < kItemsPerThread; ++j) {
        int idx = i + j;
        if (idx < n) {
          local_max = fmaxf(local_max, input[idx]);
        }
      }
    }
  }

  local_max = block_reduce_max<kBlockSize>(local_max);
  if (tid == 0) {
    partial_max[blockIdx.x] = local_max;
  }
}

// 第二阶段：用单个 block 归约所有 partial max，得到真正的全局 max。
// 结果写入 max_value[0]，供 exp_sum_kernel 使用。
__global__ void max_final_pass_kernel(const float *__restrict__ partial_max,
                                      float *__restrict__ max_value,
                                      int n) {
  int tid = threadIdx.x;
  float local_max = -FLT_MAX;

  for (int i = tid; i < n; i += blockDim.x) {
    local_max = fmaxf(local_max, partial_max[i]);
  }

  local_max = block_reduce_max<kBlockSize>(local_max);
  if (tid == 0) {
    *max_value = local_max;
  }
}

// 第三阶段的前半部分：
// 计算未归一化的 softmax 分子 output[i] = exp(input[i] - max_value[0])，
// 同时对这些分子做 block 级求和，输出 partial_sum[blockIdx.x]。
//
// 这里使用 __expf，它是 CUDA 的 fast math 单精度 exp，速度较快；
// 对本题 float32 softmax benchmark 更适合。
__global__ void exp_sum_kernel(const float *__restrict__ input,
                               float *__restrict__ output,
                               float *__restrict__ partial_sum,
                               const float *__restrict__ max_value, int n) {
  int tid = threadIdx.x;
  int base = (blockIdx.x * blockDim.x + tid) * kItemsPerThread;
  int stride = gridDim.x * blockDim.x * kItemsPerThread;
  float max_v = *max_value;
  float local_sum = 0.0f;

  for (int i = base; i < n; i += stride) {
    if (i + 3 < n) {
      float4 v = *reinterpret_cast<const float4 *>(input + i);
      float4 e;
      e.x = __expf(v.x - max_v);
      e.y = __expf(v.y - max_v);
      e.z = __expf(v.z - max_v);
      e.w = __expf(v.w - max_v);
      *reinterpret_cast<float4 *>(output + i) = e;
      local_sum += e.x + e.y + e.z + e.w;
    } else {
// 尾部不足 4 个元素时逐个计算 exp 和局部 sum。
#pragma unroll
      for (int j = 0; j < kItemsPerThread; ++j) {
        int idx = i + j;
        if (idx < n) {
          float e = __expf(input[idx] - max_v);
          output[idx] = e;
          local_sum += e;
        }
      }
    }
  }

  local_sum = block_reduce_sum<kBlockSize>(local_sum);
  if (tid == 0) {
    partial_sum[blockIdx.x] = local_sum;
  }
}

// 第三阶段的中间归约：
// 用单个 block 把所有 partial sum 合并成全局分母 sum(exp(x - max))。
// 结果写入 sum_value[0]。
__global__ void sum_final_pass_kernel(const float *__restrict__ partial_sum,
                                      float *__restrict__ sum_value, int n) {
  int tid = threadIdx.x;
  float local_sum = 0.0f;

  for (int i = tid; i < n; i += blockDim.x) {
    local_sum += partial_sum[i];
  }

  local_sum = block_reduce_sum<kBlockSize>(local_sum);
  if (tid == 0) {
    *sum_value = local_sum;
  }
}

// 第三阶段的最后一步：
// output 当前保存 exp(x - max)，这里原地除以全局分母，得到最终 softmax。
__global__ void normalize_kernel(float *__restrict__ output,
                                 const float *__restrict__ sum_value, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  float inv_sum = 1.0f / (*sum_value);

  for (int i = tid; i < n; i += stride) {
    output[i] *= inv_sum;
  }
}

int div_up(int x, int y) { return (x + y - 1) / y; }

// 根据输入规模和当前 GPU 的 occupancy 选择启动 block 数。
// 数据量小时避免启动过多 block；数据量大时让 SM 有足够并行工作。
int choose_grid_size(int n) {
  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, max_first_pass_kernel, kBlockSize, 0));

  int occupancy_blocks = prop.multiProcessorCount * blocks_per_sm;
  int data_blocks = div_up(n, kBlockSize * kItemsPerThread);
  int blocks = data_blocks < occupancy_blocks ? data_blocks : occupancy_blocks;
  return blocks > 0 ? blocks : 1;
}

double bandwidth_gbps(int n, float avg_us) {
  // 粗略估算主路径访存量：
  // max 阶段读 input 1 次；
  // exp_sum 阶段读 input 1 次、写 output 1 次；
  // normalize 阶段读 output 1 次、写 output 1 次。
  // partial 和标量 workspace 很小，相对 n 可忽略。
  double bytes = static_cast<double>(n) * sizeof(float) * 5.0;
  return bytes / (avg_us * 1.0e-6) / 1.0e9;
}

// benchmark 版本直接展开 softmax 的所有 kernel。
// workspace 由调用方预分配，避免把 cudaMalloc/cudaFree 计入 kernel 性能。
float benchmark_softmax(const float *d_input, float *d_output, float *d_partial,
                        float *d_scalars, int n, int partial_count,
                        int warmup_iters, int benchmark_iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    max_first_pass_kernel<<<partial_count, kBlockSize>>>(d_input, d_partial, n);
    max_final_pass_kernel<<<1, kBlockSize>>>(d_partial, d_scalars,
                                             partial_count);
    exp_sum_kernel<<<partial_count, kBlockSize>>>(d_input, d_output, d_partial,
                                                  d_scalars, n);
    sum_final_pass_kernel<<<1, kBlockSize>>>(d_partial, d_scalars + 1,
                                             partial_count);
    normalize_kernel<<<partial_count, kBlockSize>>>(d_output, d_scalars + 1, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < benchmark_iters; ++i) {
    max_first_pass_kernel<<<partial_count, kBlockSize>>>(d_input, d_partial, n);
    max_final_pass_kernel<<<1, kBlockSize>>>(d_partial, d_scalars,
                                             partial_count);
    exp_sum_kernel<<<partial_count, kBlockSize>>>(d_input, d_output, d_partial,
                                                  d_scalars, n);
    sum_final_pass_kernel<<<1, kBlockSize>>>(d_partial, d_scalars + 1,
                                             partial_count);
    normalize_kernel<<<partial_count, kBlockSize>>>(d_output, d_scalars + 1, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms * 1000.0f / benchmark_iters;
}

}  // namespace

// d_partial 至少需要 softmax_workspace_size(n) 个 float。
int softmax_workspace_size(int n) {
  if (n <= 0) {
    return 1;
  }
  return choose_grid_size(n);
}

// 带 workspace 的 softmax 主实现。
//
// d_partial:
//   长度至少为 blocks，复用来保存 partial max 和 partial sum。
//
// d_scalars:
//   长度至少为 2。
//   d_scalars[0] 保存全局 max。
//   d_scalars[1] 保存全局 sum(exp(x - max))。
//
// 执行顺序：
//   1. max_first_pass_kernel + max_final_pass_kernel 求全局 max。
//   2. exp_sum_kernel + sum_final_pass_kernel 求 exp 分子和全局分母。
//   3. normalize_kernel 原地归一化 output。
void softmax_with_workspace_and_blocks(const float *d_input, float *d_output,
                                       float *d_partial, float *d_scalars,
                                       int n, int blocks,
                                       cudaStream_t stream = 0) {
  if (n <= 0) {
    return;
  }

  // 1. 求全局最大值 max(x)。
  max_first_pass_kernel<<<blocks, kBlockSize, 0, stream>>>(d_input, d_partial,
                                                           n);
  CUDA_CHECK(cudaGetLastError());

  max_final_pass_kernel<<<1, kBlockSize, 0, stream>>>(d_partial, d_scalars,
                                                      blocks);
  CUDA_CHECK(cudaGetLastError());

  // 2. 计算 exp(x - max)，并归约得到分母。
  exp_sum_kernel<<<blocks, kBlockSize, 0, stream>>>(d_input, d_output,
                                                    d_partial, d_scalars, n);
  CUDA_CHECK(cudaGetLastError());

  sum_final_pass_kernel<<<1, kBlockSize, 0, stream>>>(d_partial, d_scalars + 1,
                                                      blocks);
  CUDA_CHECK(cudaGetLastError());

  // 3. output[i] = exp(x_i - max) / sum(exp(x - max))。
  normalize_kernel<<<blocks, kBlockSize, 0, stream>>>(d_output, d_scalars + 1,
                                                      n);
  CUDA_CHECK(cudaGetLastError());
}

// 自动选择 blocks 的 workspace 版本，适合外部代码预分配并复用 workspace。
void softmax_with_workspace(const float *d_input, float *d_output,
                            float *d_partial, float *d_scalars, int n,
                            cudaStream_t stream = 0) {
  int blocks = softmax_workspace_size(n);
  softmax_with_workspace_and_blocks(d_input, d_output, d_partial, d_scalars, n,
                                    blocks, stream);
}

// 简单易用版本：内部申请并释放 workspace。
// 如果在 benchmark 或循环中频繁调用，应优先使用 softmax_with_workspace。
void softmax(const float *d_input, float *d_output, int n,
             cudaStream_t stream = 0) {
  if (n <= 0) {
    return;
  }

  int partial_count = softmax_workspace_size(n);
  float *d_partial = nullptr;
  float *d_scalars = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partial, partial_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scalars, 2 * sizeof(float)));

  softmax_with_workspace_and_blocks(d_input, d_output, d_partial, d_scalars, n,
                                    partial_count, stream);

  CUDA_CHECK(cudaFree(d_scalars));
  CUDA_CHECK(cudaFree(d_partial));
}

int main(int argc, char **argv) {
  int n = 500000;
  int benchmark_iters = 100;
  int warmup_iters = 10;

  // 命令行参数：
  //   ./Softmax [n] [benchmark_iters] [warmup_iters]
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    benchmark_iters = std::atoi(argv[2]);
  }
  if (argc > 3) {
    warmup_iters = std::atoi(argv[3]);
  }
  if (n < 1) {
    n = 1;
  }
  if (benchmark_iters < 1) {
    benchmark_iters = 1;
  }
  if (warmup_iters < 0) {
    warmup_iters = 0;
  }

  float *h_input = static_cast<float *>(std::malloc(n * sizeof(float)));
  float *h_output = static_cast<float *>(std::malloc(n * sizeof(float)));
  float *h_expected = static_cast<float *>(std::malloc(n * sizeof(float)));
  if (!h_input || !h_output || !h_expected) {
    std::fprintf(stderr, "host allocation failed\n");
    return EXIT_FAILURE;
  }

  // 构造测试输入。数值范围大约为 [-5.12, 5.11]，
  // 足以覆盖 softmax 的指数差异，同时不会让参考结果大量下溢为 0。
  for (int i = 0; i < n; ++i) {
    h_input[i] = static_cast<float>((i % 1024) - 512) * 0.01f;
  }

  // CPU 参考实现同样使用 max trick，并用 double 累加降低验证端误差。
  float host_max = h_input[0];
  for (int i = 1; i < n; ++i) {
    host_max = std::max(host_max, h_input[i]);
  }

  double host_sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double e = std::exp(static_cast<double>(h_input[i] - host_max));
    h_expected[i] = static_cast<float>(e);
    host_sum += e;
  }
  for (int i = 0; i < n; ++i) {
    h_expected[i] = static_cast<float>(h_expected[i] / host_sum);
  }

  float *d_input = nullptr;
  float *d_output = nullptr;
  float *d_partial = nullptr;
  float *d_scalars = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

  // workspace 在计时前分配，benchmark 只统计 kernel 执行时间。
  int partial_count = softmax_workspace_size(n);
  CUDA_CHECK(cudaMalloc(&d_partial, partial_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scalars, 2 * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float avg_us =
      benchmark_softmax(d_input, d_output, d_partial, d_scalars, n,
                        partial_count, warmup_iters, benchmark_iters);

  CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 检查两类正确性：
  // 1. softmax 输出总和应接近 1。
  // 2. GPU 输出应接近 CPU reference。
  double output_sum = 0.0;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    output_sum += h_output[i];
    float abs_error = std::fabs(h_output[i] - h_expected[i]);
    float rel_error = abs_error / std::max(h_expected[i], 1.0e-20f);
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }

  std::printf("n=%d output_sum=%.9f max_abs_error=%e max_rel_error=%e\n", n,
              output_sum, max_abs_error, max_rel_error);
  std::printf("softmax avg_us=%.3f bandwidth_gbps=%.2f partial_blocks=%d\n",
              avg_us, bandwidth_gbps(n, avg_us), partial_count);

  int print_n = std::min(n, 8);
  std::printf("first_values:");
  for (int i = 0; i < print_n; ++i) {
    std::printf(" %.9e", h_output[i]);
  }
  std::printf("\n");

  CUDA_CHECK(cudaFree(d_scalars));
  CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));
  std::free(h_expected);
  std::free(h_output);
  std::free(h_input);
  return 0;
}
