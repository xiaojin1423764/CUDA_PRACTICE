#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// This file implements the row-major problem definition directly:
//   C[M, N] = alpha * A[M, K] * B[K, N] + beta * C_initial[M, N].
// A, B, and C use FP16 storage; Tensor Core MMA accumulates into FP32.

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

namespace wmma = nvcuda::wmma;

// One block computes a 128 x 64 output tile.  Eight warps each own four
// 16 x 16 WMMA accumulator tiles (a 32 x 32 warp tile), improving shared-tile
// reuse substantially over one-accumulator-per-warp designs.
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsM = kBlockM / 32;
constexpr int kWarpsN = kBlockN / 32;

int div_up(int x, int y) { return (x + y - 1) / y; }

// The shared B tile is deliberately stored as [column][k].  WMMA's matrix_b
// fragment below is column-major, while the input B is row-major.  Transposing
// during the cooperative global load lets tensor cores consume B without a
// per-MMA transpose or strided global loads.
__global__ void gemm_wmma_kernel(const __half *__restrict__ a,
                                 const __half *__restrict__ b,
                                 const __half *__restrict__ c_initial,
                                 __half *__restrict__ c, int m, int n, int k,
                                 float alpha, float beta) {
  // Input tiles and FP32 output accumulators have non-overlapping lifetimes.
  // Overlaying them keeps shared-memory use at 32 KiB instead of 44 KiB:
  // input tiles need 12 KiB during MMA, then the FP32 accumulator needs 32 KiB
  // during the epilogue.  The barriers below make the reuse safe.
  __shared__ union {
    struct {
      __half a[kBlockM][kBlockK];
      __half b_transposed[kBlockN][kBlockK];
    } tiles;
    float accumulator[kBlockM][kBlockN];
  } shared;

  const int block_row = blockIdx.y * kBlockM;
  const int block_col = blockIdx.x * kBlockN;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int warp_row = warp / kWarpsN;  // 0..3, each warp covers 32 rows.
  const int warp_col = warp % kWarpsN;  // 0..1, each warp covers 32 columns.

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator[2][2];
#pragma unroll
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < 2; ++tile_n) {
      wmma::fill_fragment(accumulator[tile_m][tile_n], 0.0f);
    }
  }

  // Each K tile is zero-padded in shared memory.  This preserves the WMMA
  // 16-wide contract for arbitrary K and also handles M/N edge tiles safely.
  for (int k_base = 0; k_base < k; k_base += kBlockK) {
    for (int index = tid; index < kBlockM * kBlockK; index += blockDim.x) {
      const int row = index / kBlockK;
      const int kk = index % kBlockK;
      const int global_row = block_row + row;
      const int global_k = k_base + kk;
      shared.tiles.a[row][kk] = (global_row < m && global_k < k)
                                    ? a[global_row * k + global_k]
                                    : __float2half(0.0f);
    }
    for (int index = tid; index < kBlockN * kBlockK; index += blockDim.x) {
      const int col = index / kBlockK;
      const int kk = index % kBlockK;
      const int global_col = block_col + col;
      const int global_k = k_base + kk;
      shared.tiles.b_transposed[col][kk] =
          (global_col < n && global_k < k) ? b[global_k * n + global_col]
                                            : __float2half(0.0f);
    }
    __syncthreads();

    // All eight warps execute this same loop.  Each warp computes a 32 x 32
    // region with four MMA fragments.  An A tile is reused by two warp columns,
    // while a B tile is reused by four warp rows.
#pragma unroll
    for (int kk = 0; kk < kBlockK; kk += 16) {
#pragma unroll
      for (int tile_m = 0; tile_m < 2; ++tile_m) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
            a_fragment;
        wmma::load_matrix_sync(
            a_fragment, &shared.tiles.a[warp_row * 32 + tile_m * 16][kk],
            kBlockK);
#pragma unroll
        for (int tile_n = 0; tile_n < 2; ++tile_n) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
              b_fragment;
          wmma::load_matrix_sync(
              b_fragment,
              &shared.tiles.b_transposed[warp_col * 32 + tile_n * 16][kk],
              kBlockK);
          wmma::mma_sync(accumulator[tile_m][tile_n], a_fragment, b_fragment,
                         accumulator[tile_m][tile_n]);
        }
      }
    }
    __syncthreads();  // Shared storage is overwritten by the next K tile.
  }

  // Each warp writes four disjoint 16 x 16 regions.  This begins only after
  // all K tiles are consumed, so the union can safely switch to accumulator.
#pragma unroll
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < 2; ++tile_n) {
      wmma::store_matrix_sync(
          &shared.accumulator[warp_row * 32 + tile_m * 16]
                             [warp_col * 32 + tile_n * 16],
          accumulator[tile_m][tile_n], kBlockN, wmma::mem_row_major);
    }
  }
  __syncthreads();

  // Fusing alpha/beta avoids an additional read/write kernel.  Values are
  // intentionally converted only at the final store, so the dot product and
  // alpha/beta arithmetic retain FP32 precision.
  for (int index = tid; index < kBlockM * kBlockN; index += blockDim.x) {
    const int row = index / kBlockN;
    const int col = index % kBlockN;
    const int global_row = block_row + row;
    const int global_col = block_col + col;
    if (global_row < m && global_col < n) {
      const int offset = global_row * n + global_col;
      const float old_c = __half2float(c_initial[offset]);
      c[offset] = __float2half_rn(alpha * shared.accumulator[row][col] +
                                   beta * old_c);
    }
  }
}

void launch_gemm(const __half *a, const __half *b, const __half *c_initial,
                 __half *c, int m, int n, int k, float alpha, float beta,
                 cudaStream_t stream = 0) {
  const dim3 block(kThreads);
  const dim3 grid(div_up(n, kBlockN), div_up(m, kBlockM));
  gemm_wmma_kernel<<<grid, block, 0, stream>>>(a, b, c_initial, c, m, n, k,
                                                alpha, beta);
  CUDA_CHECK(cudaGetLastError());
}

// Deterministic values make failures reproducible without a random-number
// dependency.  The range is small enough that FP16 input quantization is
// representative without overflowing the output in normal benchmark sizes.
float input_value(int index, int salt) {
  return static_cast<float>((index * 17 + salt * 31) % 101 - 50) * 0.02f;
}

float reference_element(const __half *a, const __half *b,
                        const __half *c_initial, int row, int col, int n,
                        int k, float alpha, float beta) {
  float sum = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    sum += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  return alpha * sum + beta * __half2float(c_initial[row * n + col]);
}

// For small matrices check every element.  For performance-size matrices,
// checking evenly distributed samples keeps validation inexpensive while still
// exercising interior and boundary tiles.
void validate_result(const __half *a, const __half *b, const __half *c_initial,
                     const __half *c, int m, int n, int k, float alpha,
                     float beta) {
  const long long elements = static_cast<long long>(m) * n;
  const int checks = elements <= 65536 ? static_cast<int>(elements) : 512;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;

  for (int sample = 0; sample < checks; ++sample) {
    const long long flat = checks == 1
                               ? 0
                               : sample * (elements - 1) / (checks - 1);
    const int row = static_cast<int>(flat / n);
    const int col = static_cast<int>(flat % n);
    const float expected =
        reference_element(a, b, c_initial, row, col, n, k, alpha, beta);
    const float actual = __half2float(c[flat]);
    const float abs_error = std::fabs(actual - expected);
    const float rel_error = abs_error / std::max(std::fabs(expected), 1.0e-3f);
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
  }
  std::printf("validation samples=%d max_abs_error=%e max_rel_error=%e\n",
              checks, max_abs_error, max_rel_error);
}

float benchmark_gemm(const __half *a, const __half *b, const __half *c_initial,
                     __half *c, int m, int n, int k, float alpha, float beta,
                     int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch_gemm(a, b, c_initial, c, m, n, k, alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    launch_gemm(a, b, c_initial, c, m, n, k, alpha, beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms * 1000.0f / iterations;
}

}  // namespace

int main(int argc, char **argv) {
  int m = 256;
  int n = 256;
  int k = 256;
  int iterations = 100;
  int warmup = 10;
  if (argc > 1) m = std::atoi(argv[1]);
  if (argc > 2) n = std::atoi(argv[2]);
  if (argc > 3) k = std::atoi(argv[3]);
  if (argc > 4) iterations = std::atoi(argv[4]);
  if (argc > 5) warmup = std::atoi(argv[5]);
  if (m < 1 || n < 1 || k < 1 || iterations < 1 || warmup < 0) {
    std::fprintf(stderr, "usage: %s [M N K iterations warmup], all sizes > 0\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 7) {
    std::fprintf(stderr, "WMMA Tensor Cores require compute capability 7.0+\n");
    return EXIT_FAILURE;
  }

  const size_t a_elements = static_cast<size_t>(m) * k;
  const size_t b_elements = static_cast<size_t>(k) * n;
  const size_t c_elements = static_cast<size_t>(m) * n;
  const float alpha = 1.0f;
  const float beta = 0.25f;

  __half *h_a = static_cast<__half *>(std::malloc(a_elements * sizeof(__half)));
  __half *h_b = static_cast<__half *>(std::malloc(b_elements * sizeof(__half)));
  __half *h_c_initial =
      static_cast<__half *>(std::malloc(c_elements * sizeof(__half)));
  __half *h_c = static_cast<__half *>(std::malloc(c_elements * sizeof(__half)));
  if (!h_a || !h_b || !h_c_initial || !h_c) {
    std::fprintf(stderr, "host allocation failed\n");
    return EXIT_FAILURE;
  }
  for (size_t i = 0; i < a_elements; ++i) h_a[i] = __float2half(input_value(i, 1));
  for (size_t i = 0; i < b_elements; ++i) h_b[i] = __float2half(input_value(i, 2));
  for (size_t i = 0; i < c_elements; ++i)
    h_c_initial[i] = __float2half(input_value(i, 3));

  __half *d_a = nullptr;
  __half *d_b = nullptr;
  __half *d_c_initial = nullptr;
  __half *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_b, b_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_c_initial, c_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_c, c_elements * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, a_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, b_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c_initial, h_c_initial, c_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));

  const float avg_us = benchmark_gemm(d_a, d_b, d_c_initial, d_c, m, n, k,
                                      alpha, beta, warmup, iterations);
  // Re-run once from the original C so validation checks the documented beta
  // semantics rather than the result of repeated benchmark updates.
  launch_gemm(d_a, d_b, d_c_initial, d_c, m, n, k, alpha, beta);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, c_elements * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  validate_result(h_a, h_b, h_c_initial, h_c, m, n, k, alpha, beta);

  const double tflops = 2.0 * static_cast<double>(m) * n * k / avg_us / 1.0e6;
  std::printf("hand_wmma_gemm M=%d N=%d K=%d alpha=%.2f beta=%.2f "
              "avg_us=%.3f tflops=%.2f\n",
              m, n, k, alpha, beta, avg_us, tflops);

  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_c_initial));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  std::free(h_c);
  std::free(h_c_initial);
  std::free(h_b);
  std::free(h_a);
  return 0;
}
