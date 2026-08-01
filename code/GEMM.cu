#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// 本文件直接实现题目规定的 row-major GEMM：
//   C[M, N] = alpha * A[M, K] * B[K, N] + beta * C_initial[M, N]。
// A、B、C 的存储类型均为 FP16；WMMA Tensor Core 的乘加累加器为 FP32，
// 仅在最终写回 C 时转回 FP16，从而避免 FP16 累加带来的明显精度损失。

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

// 一个 thread block 计算 C 的 128 x 64 输出块，block 内共有 256 个线程、8 个 warp。
// 每个 warp 负责 32 x 32 的子块，并持有 2 x 2 个 16 x 16 WMMA accumulator fragment。
// 因而一个 warp 会发射 4 次矩阵乘加；相较于每个 warp 仅计算一个 fragment，
// 这种安排能让同一份 shared-memory A/B 子块被更多 MMA 操作复用。
constexpr int kBlockM = 128;
constexpr int kBlockN = 64;
constexpr int kBlockK = 32;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsN = kBlockN / 32;

// 正整数向上取整，用于计算不能被 tile 大小整除时所需的 grid 尺寸。
int div_up(int x, int y) { return (x + y - 1) / y; }

// shared B tile 故意按 [列][K] 保存。输入 B 是 row-major，但下面的 WMMA matrix_b
// fragment 要求 column-major。协作加载 global memory 时完成这次布局转换，
// 后续 Tensor Core 可按连续的 column-major tile 读取 B，避免每次 MMA 再转置，
// 也避免从 global memory 进行跨步读取。
__global__ void gemm_wmma_kernel(const __half *__restrict__ a,
                                 const __half *__restrict__ b,
                                 const __half *__restrict__ c_initial,
                                 __half *__restrict__ c, int m, int n, int k,
                                 float alpha, float beta) {
  // 输入 tile 与 FP32 输出暂存的生命周期不重叠，因此使用 union 复用 shared memory。
  // MMA 阶段：A 为 128x32、B 为 64x32，合计 12 KiB（每个元素 2 字节）。
  // epilogue 阶段：FP32 accumulator 为 128x64，使用 32 KiB（每个元素 4 字节）。
  // 复用后 shared memory 峰值为 32 KiB，而不是二者相加的 44 KiB；下方的两次
  // __syncthreads() 保证前一阶段的所有读写完成后才切换这块存储的解释方式。
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
  // warp 编号按输出 tile 的二维坐标映射：warp_row 为 0..3，warp_col 为 0..1。
  // 因此每个 warp 覆盖本 block 内固定的 32 行 x 32 列输出区域。
  const int warp_row = warp / kWarpsN;
  const int warp_col = warp % kWarpsN;

  // 四个 accumulator fragment 位于寄存器中，逻辑布局为 [2][2] 个 16x16 子块。
  // 初值必须显式清零，随后每个 K tile 的 mma_sync 都会累加到这些寄存器上。
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator[2][2];
#pragma unroll
  for (int tile_m = 0; tile_m < 2; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < 2; ++tile_n) {
      wmma::fill_fragment(accumulator[tile_m][tile_n], 0.0f);
    }
  }

  // 沿 K 维以 32 为步长迭代。对边界 tile 中越界的 A/B 元素填 0：
  // WMMA 的矩阵维度必须是 16 的倍数，填零同时支持任意 K、M、N，而不会影响
  // 合法元素的点积结果。每个线程按 stride=blockDim.x 搬运多个连续元素，
  // 所有线程合起来完成 A、B tile 的协作加载。
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

    // A/B tile 载入完成后，8 个 warp 都执行相同的 K 内循环。每个 warp 计算 32x32
    // 区域中的四个 fragment。相同的 A 子块会被两个不同的 warp_col 复用，
    // 相同的 B 子块会被四个不同的 warp_row 复用，因此 shared memory tile 的
    // 读取复用率显著高于直接从 global memory 为每个 warp 单独加载。
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
    // 所有 warp 用完当前 K tile 后才能覆盖 shared memory，进入下一轮 K tile。
    __syncthreads();
  }

  // K 维全部累加完后，每个 warp 将四个互不重叠的 16x16 fragment 写入 shared。
  // 到这里已不会再读取输入 tile，所以 union 可以安全地切换为 FP32 accumulator。
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

  // 将 alpha * (A*B) + beta * C_initial 融合到同一 kernel 的 epilogue，
  // 避免额外 kernel 对输出矩阵的一次读写。点积、alpha/beta 运算均保持 FP32，
  // 只在最终写回全局 C 时用 __float2half_rn 执行 round-to-nearest 转换。
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
  // grid.x 对应 N 方向的 64 列 tile，grid.y 对应 M 方向的 128 行 tile。
  // 保留 stream 参数以便该函数可用于非默认流；benchmark 中默认使用 stream 0。
  const dim3 block(kThreads);
  const dim3 grid(div_up(n, kBlockN), div_up(m, kBlockM));
  gemm_wmma_kernel<<<grid, block, 0, stream>>>(a, b, c_initial, c, m, n, k,
                                                alpha, beta);
  CUDA_CHECK(cudaGetLastError());
}

// 用确定性伪数据初始化输入，避免引入随机数依赖，也使错误可稳定复现。
// 元素范围约为 [-1.0, 1.0]，既能体现 FP16 量化，又不会在常规测试规模下使输出溢出。
float input_value(int index, int salt) {
  return static_cast<float>((index * 17 + salt * 31) % 101 - 50) * 0.02f;
}

float reference_element(const __half *a, const __half *b,
                        const __half *c_initial, int row, int col, int n,
                        int k, float alpha, float beta) {
  // CPU 参考路径将每个 FP16 输入显式提升到 float 再累加，用于验证 GPU 输出。
  // 这是逐元素 O(K) 实现，只在采样点上调用，不参与性能计时。
  float sum = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    sum += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  return alpha * sum + beta * __half2float(c_initial[row * n + col]);
}

// 小矩阵逐元素检查；性能规模只检查均匀分布的 512 个样本。
// 采样同时覆盖首尾、内部和可能的边界 tile，避免完整 CPU GEMM 干扰 benchmark 时间。
void validate_result(const __half *a, const __half *b, const __half *c_initial,
                     const __half *c, int m, int n, int k, float alpha,
                     float beta) {
  const long long elements = static_cast<long long>(m) * n;
  const int checks = elements <= 65536 ? static_cast<int>(elements) : 512;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;

  for (int sample = 0; sample < checks; ++sample) {
    // 将 [0, M*N-1] 均匀映射到 checks 个样本，随后恢复其 row-major 坐标。
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
  // warmup 使 CUDA context、JIT/缓存和 GPU 时钟进入稳定状态；其耗时不计入结果。
  for (int i = 0; i < warmup; ++i) {
    launch_gemm(a, b, c_initial, c, m, n, k, alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // CUDA event 在同一 stream 上记录，测得的是 GPU 执行时间，不含 CPU 提交开销。
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

}  // namespace：内部辅助函数不暴露给其他编译单元。

int main(int argc, char **argv) {
  // 参数依次为 M、N、K、正式迭代次数、warmup 次数；未传入时使用较小默认规模。
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

  // WMMA API 从 Volta（compute capability 7.0）起支持，提前给出明确错误信息。
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 7) {
    std::fprintf(stderr, "WMMA Tensor Cores require compute capability 7.0+\n");
    return EXIT_FAILURE;
  }

  // A、B、C 均连续按 row-major 分配；C_initial 单独保存，确保 beta 语义可验证。
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

  // 所有 host-device 拷贝发生在计时前，benchmark 仅衡量 GEMM kernel 本身。
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
  // benchmark 会连续更新 C（beta 非零），因此校验前从原始 C_initial 重跑一次。
  // 这样检查的是题目定义的 beta 语义，而不是前一次迭代输出再次作为 C 的结果。
  launch_gemm(d_a, d_b, d_c_initial, d_c, m, n, k, alpha, beta);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, c_elements * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  validate_result(h_a, h_b, h_c_initial, h_c, m, n, k, alpha, beta);

  // 每个 FMA 按 2 次浮点操作计算；avg_us 的单位为微秒，故除以 1e6 得 TFLOPS。
  const double tflops = 2.0 * static_cast<double>(m) * n * k / avg_us / 1.0e6;
  std::printf("hand_wmma_gemm M=%d N=%d K=%d alpha=%.2f beta=%.2f "
              "avg_us=%.3f tflops=%.2f\n",
              m, n, k, alpha, beta, avg_us, tflops);

  // 按与分配相反的逻辑顺序释放 device 和 host 资源。
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
