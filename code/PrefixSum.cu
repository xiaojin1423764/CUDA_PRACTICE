#include <algorithm>
#include <cmath>
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
constexpr int kItemsPerThread = 8;
constexpr int kItemsPerBlock = kBlockSize * kItemsPerThread;

__device__ __forceinline__ float warp_inclusive_scan(float v) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    float other = __shfl_up_sync(0xffffffffu, v, offset);
    if ((threadIdx.x & 31) >= offset) {
      v += other;
    }
  }
  return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_inclusive_scan(float v,
                                                      float &block_sum) {
  __shared__ float warp_sums[BLOCK_SIZE / 32];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_inclusive_scan(v);
  if (lane == 31) {
    warp_sums[warp] = v;
  }
  __syncthreads();

  float warp_prefix = 0.0f;
  if (warp == 0) {
    float warp_sum = (threadIdx.x < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
    float scanned = warp_inclusive_scan(warp_sum);
    if (threadIdx.x < BLOCK_SIZE / 32) {
      warp_sums[lane] = scanned;
    }
  }
  __syncthreads();

  if (warp > 0) {
    warp_prefix = warp_sums[warp - 1];
  }
  block_sum = warp_sums[BLOCK_SIZE / 32 - 1];
  return v + warp_prefix;
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void scan_blocks_kernel(const float *__restrict__ input,
                                   float *__restrict__ output,
                                   float *__restrict__ block_sums, int n) {
  using BlockLoad =
      cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                     cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStore =
      cub::BlockStore<float, BLOCK_SIZE, ITEMS_PER_THREAD,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  } temp;

  float items[ITEMS_PER_THREAD];
  int block_offset = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
  int valid_items = n - block_offset;
  valid_items = valid_items > 0 ? min(valid_items, BLOCK_SIZE * ITEMS_PER_THREAD)
                                : 0;

  BlockLoad(temp.load).Load(input + block_offset, items, valid_items, 0.0f);
  __syncthreads();

  float thread_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    thread_sum += items[i];
    items[i] = thread_sum;
  }

  float block_sum = 0.0f;
  float thread_prefix_sum = block_inclusive_scan<BLOCK_SIZE>(thread_sum,
                                                             block_sum);
  float thread_offset = thread_prefix_sum - thread_sum;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    items[i] += thread_offset;
  }
  __syncthreads();

  BlockStore(temp.store).Store(output + block_offset, items, valid_items);

  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = block_sum;
  }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void add_block_offsets_kernel(float *__restrict__ output,
                                         const float *__restrict__ offsets,
                                         int n) {
  int block_offset = blockIdx.x * BLOCK_SIZE * ITEMS_PER_THREAD;
  int valid_items = n - block_offset;
  valid_items = valid_items > 0 ? min(valid_items, BLOCK_SIZE * ITEMS_PER_THREAD)
                                : 0;

  if (blockIdx.x == 0 || valid_items <= 0) {
    return;
  }

  float offset = offsets[blockIdx.x - 1];
  int linear_tid = threadIdx.x;
  for (int i = linear_tid; i < valid_items; i += BLOCK_SIZE) {
    output[block_offset + i] += offset;
  }
}

int div_up(int x, int y) { return (x + y - 1) / y; }

void prefix_sum_recursive(const float *d_input, float *d_output, int n,
                          cudaStream_t stream) {
  if (n <= 0) {
    return;
  }

  int num_blocks = div_up(n, kItemsPerBlock);
  float *d_block_sums = nullptr;
  CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));

  scan_blocks_kernel<kBlockSize, kItemsPerThread>
      <<<num_blocks, kBlockSize, 0, stream>>>(d_input, d_output, d_block_sums,
                                              n);
  CUDA_CHECK(cudaGetLastError());

  if (num_blocks > 1) {
    float *d_block_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * sizeof(float)));

    prefix_sum_recursive(d_block_sums, d_block_offsets, num_blocks, stream);

    add_block_offsets_kernel<kBlockSize, kItemsPerThread>
        <<<num_blocks, kBlockSize, 0, stream>>>(d_output, d_block_offsets, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_block_offsets));
  }

  CUDA_CHECK(cudaFree(d_block_sums));
}

double bandwidth_gbps(int n, float avg_us) {
  double bytes = static_cast<double>(n) * sizeof(float) * 2.0;
  return bytes / (avg_us * 1.0e-6) / 1.0e9;
}

float benchmark_prefix_sum(const float *d_input, float *d_output, int n,
                           int warmup_iters, int benchmark_iters) {
  for (int i = 0; i < warmup_iters; ++i) {
    prefix_sum_recursive(d_input, d_output, n, 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (int i = 0; i < benchmark_iters; ++i) {
    prefix_sum_recursive(d_input, d_output, n, 0);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms * 1000.0f / benchmark_iters;
}

}  // namespace

void prefix_sum(const float *d_input, float *d_output, int n,
                cudaStream_t stream = 0) {
  prefix_sum_recursive(d_input, d_output, n, stream);
}

int main(int argc, char **argv) {
  int n = 250000;
  int benchmark_iters = 100;
  int warmup_iters = 10;

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
  if (!h_input || !h_output) {
    std::fprintf(stderr, "host allocation failed\n");
    return EXIT_FAILURE;
  }

  for (int i = 0; i < n; ++i) {
    h_input[i] = 1.0f;
  }

  float *d_input = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  float avg_us =
      benchmark_prefix_sum(d_input, d_output, n, warmup_iters, benchmark_iters);
  CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    float expected = static_cast<float>(i + 1);
    max_error = std::max(max_error, std::fabs(h_output[i] - expected));
  }

  std::printf("n=%d last=%f expected_last=%f max_error=%f\n", n,
              h_output[n - 1], static_cast<float>(n), max_error);
  std::printf("prefix_sum avg_us=%.3f bandwidth_gbps=%.2f\n", avg_us,
              bandwidth_gbps(n, avg_us));

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));
  std::free(h_output);
  std::free(h_input);
  return 0;
}
