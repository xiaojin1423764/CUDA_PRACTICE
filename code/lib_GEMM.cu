#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 这是与 GEMM.cu 相同题目的 cuBLAS baseline，数学定义为：
//   C[M, N] = alpha * A[M, K] * B[K, N] + beta * C_initial[M, N]。
// 题目中的矩阵存储为 row-major，而传统 cuBLAS GEMM 接口采用 column-major。
// gemm_row_major 利用 C^T = B^T * A^T 的恒等式，只改变内存解释方式，
// 不额外启动转置 kernel，也不分配临时转置矩阵。

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t status = (call);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__,      \
                   __LINE__, static_cast<int>(status));                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace {

float input_value(int index, int salt) {
  // 确定性伪数据保证手写实现和 cuBLAS baseline 使用完全相同的输入，便于复现。
  // 数值约位于 [-1.0, 1.0]，能体现 FP16 量化，又不会使一般测试中的结果溢出。
  return static_cast<float>((index * 17 + salt * 31) % 101 - 50) * 0.02f;
}

float reference_element(const __half *a, const __half *b,
                        const __half *c_initial, int row, int col, int n,
                        int k, float alpha, float beta) {
  // CPU 参考路径将 FP16 元素提升为 float 再累加。它只用于校验采样点，
  // 不在 CUDA event 的计时区间内，因此不会影响 cuBLAS 性能结果。
  float sum = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    sum += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
  }
  return alpha * sum + beta * __half2float(c_initial[row * n + col]);
}

void validate_result(const __half *a, const __half *b, const __half *c_initial,
                     const __half *c, int m, int n, int k, float alpha,
                     float beta) {
  // 小矩阵完整比对；性能规模均匀抽取 512 个位置，兼顾校验成本与覆盖范围。
  const long long elements = static_cast<long long>(m) * n;
  const int checks = elements <= 65536 ? static_cast<int>(elements) : 512;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  for (int sample = 0; sample < checks; ++sample) {
    // 从一维 row-major 下标恢复 (row, col)，样本覆盖输出矩阵首尾和内部区域。
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

// 连续 row-major 缓冲区可被 cuBLAS 看作其转置的 column-major 视图。例如：
// row-major A[M,K] 的线性地址为 A[row * K + col]，正好等于 column-major
// A^T[K,M] 的地址 A[col + row * K]。因此 row-major A[M,K] * B[K,N] 可写为：
//
//   C^T[N,M] = B^T[N,K] * A^T[K,M]
//
// 这里的转置仅是逻辑解释：原始 row-major B 的行跨度 N 变成 column-major B^T
// 的 leading dimension（ldb）N；A 的行跨度 K 变成 A^T 的 leading dimension（lda）K。
// 输出 C 同理使用 leading dimension（ldc）N。整个过程没有额外 global memory 拷贝。
void gemm_row_major(cublasHandle_t handle, const __half *a, const __half *b,
                    __half *c, int m, int n, int k, float alpha, float beta) {
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      // column-major C^T 的形状为 N x M；两侧均已通过内存解释变成转置视图，
      // 所以这里使用 CUBLAS_OP_N，而不是再要求 cuBLAS 做转置。
      n, m, k,
      &alpha,
      // 第一个操作数：row-major B 缓冲区解释为 column-major B^T[N,K]，ldb=N。
      b, CUDA_R_16F, n,
      // 第二个操作数：row-major A 缓冲区解释为 column-major A^T[K,M]，lda=K。
      a, CUDA_R_16F, k,
      &beta, c, CUDA_R_16F, n,
      // 外部矩阵均是 FP16，但 CUBLAS_COMPUTE_32F 指定 FP32 累加；默认 Tensor Op
      // 算法会在硬件和形状允许时使用 Tensor Core。
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

float benchmark_gemm(cublasHandle_t handle, const __half *a, const __half *b,
                     __half *c, int m, int n, int k, float alpha, float beta,
                     int warmup, int iterations) {
  // warmup 使 CUDA context、cuBLAS 内部选择和 GPU 时钟进入稳定状态，其耗时不计入。
  for (int i = 0; i < warmup; ++i) {
    gemm_row_major(handle, a, b, c, m, n, k, alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // event 与 cuBLAS handle 均使用默认 stream，测量的是 GPU GEMM 执行时间，
  // 不包含 host 侧函数调用和参数提交开销。
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    gemm_row_major(handle, a, b, c, m, n, k, alpha, beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return elapsed_ms * 1000.0f / iterations;
}

}  // namespace：辅助函数仅供本编译单元的 benchmark 使用。

int main(int argc, char **argv) {
  // 参数依次为 M、N、K、正式迭代次数、warmup 次数；默认值适合快速冒烟测试。
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

  // 三个矩阵按题目定义的 row-major 布局分配；C_initial 单独保留给 beta 与校验使用。
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

  // 内存分配和 H2D 拷贝全部发生在计时前，避免把数据传输计入 GEMM 吞吐。
  __half *d_a = nullptr;
  __half *d_b = nullptr;
  __half *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_b, b_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_c, c_elements * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, a_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, b_elements * sizeof(__half), cudaMemcpyHostToDevice));

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  // handle 和 CUDA event 均绑定默认 stream，保证 event 与 GEMM 的先后顺序正确。
  CUBLAS_CHECK(cublasSetStream(handle, 0));
  // 请求 Tensor Core 数学模式；是否实际使用仍由数据类型、矩阵尺寸和内部算法决定。
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // c 只在计时前初始化一次。beta 非零时每轮 GEMM 都会更新 C，但这不影响单次
  // kernel 耗时；最终校验前会从 C_initial 恢复，避免累计结果影响 beta 语义检查。
  CUDA_CHECK(cudaMemcpy(d_c, h_c_initial, c_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  const float avg_us =
      benchmark_gemm(handle, d_a, d_b, d_c, m, n, k, alpha, beta, warmup,
                     iterations);

  CUDA_CHECK(cudaMemcpy(d_c, h_c_initial, c_elements * sizeof(__half),
                        cudaMemcpyHostToDevice));
  gemm_row_major(handle, d_a, d_b, d_c, m, n, k, alpha, beta);
  CUDA_CHECK(cudaMemcpy(h_c, d_c, c_elements * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  validate_result(h_a, h_b, h_c_initial, h_c, m, n, k, alpha, beta);

  // 每个 FMA 计算为 2 次浮点操作；avg_us 单位为微秒，故除以 1e6 转为 TFLOPS。
  const double tflops = 2.0 * static_cast<double>(m) * n * k / avg_us / 1.0e6;
  std::printf("cublas_gemm M=%d N=%d K=%d alpha=%.2f beta=%.2f "
              "avg_us=%.3f tflops=%.2f\n",
              m, n, k, alpha, beta, avg_us, tflops);

  // 在释放矩阵缓冲区前销毁 cuBLAS handle，并释放全部 host/device 资源。
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  std::free(h_c);
  std::free(h_c_initial);
  std::free(h_b);
  std::free(h_a);
  return 0;
}
