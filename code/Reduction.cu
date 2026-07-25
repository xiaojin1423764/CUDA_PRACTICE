#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
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

namespace {

constexpr int kBlockSize = 256;
constexpr int kItemsPerThread = 4;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

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

__global__ void reduce_first_pass(const float *__restrict__ x,
                                  float *__restrict__ partial,
                                  int n) {
  int tid = threadIdx.x;
  int base = (blockIdx.x * blockDim.x + tid) * kItemsPerThread;
  int stride = gridDim.x * blockDim.x * kItemsPerThread;

  float sum = 0.0f;

  for (int i = base; i < n; i += stride) {
    if (i + 3 < n) {
      float4 v = *reinterpret_cast<const float4 *>(x + i);
      sum += v.x + v.y + v.z + v.w;
    } else {
#pragma unroll
      for (int j = 0; j < kItemsPerThread; ++j) {
        int idx = i + j;
        if (idx < n) {
          sum += x[idx];
        }
      }
    }
  }

  sum = block_reduce_sum<kBlockSize>(sum);
  if (tid == 0) {
    partial[blockIdx.x] = sum;
  }
}

__global__ void reduce_final_pass(const float *__restrict__ partial,
                                  float *__restrict__ out,
                                  int n) {
  int tid = threadIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < n; i += blockDim.x) {
    sum += partial[i];
  }

  sum = block_reduce_sum<kBlockSize>(sum);
  if (tid == 0) {
    *out = sum;
  }
}

int choose_grid_size(int n) {
  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int blocks_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, reduce_first_pass, kBlockSize, 0));

  int occupancy_blocks = prop.multiProcessorCount * blocks_per_sm;
  int data_blocks =
      (n + kBlockSize * kItemsPerThread - 1) / (kBlockSize * kItemsPerThread);

  int blocks = data_blocks < occupancy_blocks ? data_blocks : occupancy_blocks;
  return blocks > 0 ? blocks : 1;
}

}  // namespace

int reduction_workspace_size(int n) {
  if (n <= 0) {
    return 1;
  }
  return choose_grid_size(n);
}

void reduction_with_workspace(const float *d_input, float *d_output,
                              float *d_partial, int n,
                              cudaStream_t stream = 0) {
  if (n <= 0) {
    CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(float), stream));
    return;
  }

  int blocks = choose_grid_size(n);
  reduce_first_pass<<<blocks, kBlockSize, 0, stream>>>(d_input, d_partial, n);
  CUDA_CHECK(cudaGetLastError());

  reduce_final_pass<<<1, kBlockSize, 0, stream>>>(d_partial, d_output, blocks);
  CUDA_CHECK(cudaGetLastError());
}

void reduction_with_workspace_and_blocks(const float *d_input, float *d_output,
                                         float *d_partial, int n, int blocks,
                                         cudaStream_t stream = 0) {
  if (n <= 0) {
    CUDA_CHECK(cudaMemsetAsync(d_output, 0, sizeof(float), stream));
    return;
  }

  reduce_first_pass<<<blocks, kBlockSize, 0, stream>>>(d_input, d_partial, n);
  CUDA_CHECK(cudaGetLastError());

  reduce_final_pass<<<1, kBlockSize, 0, stream>>>(d_partial, d_output, blocks);
  CUDA_CHECK(cudaGetLastError());
}

void reduction(const float *d_input, float *d_output, int n,
               cudaStream_t stream = 0) {
  int partial_count = reduction_workspace_size(n);
  float *d_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&d_partial, partial_count * sizeof(float)));

  reduction_with_workspace(d_input, d_output, d_partial, n, stream);

  CUDA_CHECK(cudaFree(d_partial));
}

double bandwidth_gbps(int n, float avg_us) {
  double bytes = static_cast<double>(n) * sizeof(float);
  return bytes / (avg_us * 1.0e-6) / 1.0e9;
}

float benchmark_custom(const float *d_input, float *d_output, float *d_partial,
                       int n, int partial_count, int warmup_iters,
                       int benchmark_iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    reduction_with_workspace_and_blocks(d_input, d_output, d_partial, n,
                                        partial_count);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < benchmark_iters; ++i) {
    reduction_with_workspace_and_blocks(d_input, d_output, d_partial, n,
                                        partial_count);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms * 1000.0f / benchmark_iters;
}

float benchmark_cub(const float *d_input, float *d_output, void *d_temp_storage,
                    size_t temp_storage_bytes, int n, int warmup_iters,
                    int benchmark_iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      d_input, d_output, n));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < benchmark_iters; ++i) {
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      d_input, d_output, n));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms * 1000.0f / benchmark_iters;
}

int main(int argc, char **argv) {
  constexpr int n = 4 * 1024 * 1024;
  int warmup_iters = 10;
  int benchmark_iters = 100;
  if (argc > 1) {
    benchmark_iters = std::atoi(argv[1]);
  }
  if (argc > 2) {
    warmup_iters = std::atoi(argv[2]);
  }
  if (benchmark_iters < 1) {
    benchmark_iters = 1;
  }
  if (warmup_iters < 0) {
    warmup_iters = 0;
  }

  float *h_input = static_cast<float *>(std::malloc(n * sizeof(float)));
  if (!h_input) {
    return 1;
  }

  double expected = 0.0;
  for (int i = 0; i < n; ++i) {
    h_input[i] = 1.0f;
    expected += h_input[i];
  }

  float *d_input = nullptr;
  float *d_custom_output = nullptr;
  float *d_cub_output = nullptr;
  float *d_partial = nullptr;
  void *d_cub_temp = nullptr;
  float h_custom_output = 0.0f;
  float h_cub_output = 0.0f;

  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_custom_output, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cub_output, sizeof(float)));
  int partial_count = reduction_workspace_size(n);
  CUDA_CHECK(cudaMalloc(&d_partial, partial_count * sizeof(float)));

  size_t cub_temp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, cub_temp_bytes, d_input,
                                    d_cub_output, n));
  CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float custom_avg_us =
      benchmark_custom(d_input, d_custom_output, d_partial, n, partial_count,
                       warmup_iters, benchmark_iters);
  float cub_avg_us =
      benchmark_cub(d_input, d_cub_output, d_cub_temp, cub_temp_bytes, n,
                    warmup_iters, benchmark_iters);

  CUDA_CHECK(cudaMemcpy(&h_custom_output, d_custom_output, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_cub_output, d_cub_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::printf("expected=%f\n", expected);
  std::printf("custom gpu=%f avg_us=%.3f bandwidth_gbps=%.2f "
              "partial_blocks=%d\n",
              h_custom_output, custom_avg_us,
              bandwidth_gbps(n, custom_avg_us), partial_count);
  std::printf("cub    gpu=%f avg_us=%.3f bandwidth_gbps=%.2f "
              "temp_bytes=%zu\n",
              h_cub_output, cub_avg_us, bandwidth_gbps(n, cub_avg_us),
              cub_temp_bytes);
  std::printf("custom_vs_cub_time=%.3fx custom_error=%f cub_error=%f\n",
              custom_avg_us / cub_avg_us,
              static_cast<float>(h_custom_output - expected),
              static_cast<float>(h_cub_output - expected));

  CUDA_CHECK(cudaFree(d_cub_temp));
  CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFree(d_cub_output));
  CUDA_CHECK(cudaFree(d_custom_output));
  CUDA_CHECK(cudaFree(d_input));
  std::free(h_input);
  return 0;
}
