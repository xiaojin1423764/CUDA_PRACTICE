#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Library baseline for the same row-major FP16 GEMM problem as GEMM.cu:
//   C[M, N] = alpha * A[M, K] * B[K, N] + beta * C_initial[M, N].
// cuBLAS exposes column-major BLAS interfaces, so gemm_row_major below uses
// the transpose identity C^T = B^T * A^T without physically transposing data.

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

// cuBLAS sees the same contiguous row-major buffer as a column-major view of
// its transpose.  Therefore row-major A[M,K] * B[K,N] is expressed as:
//
//   C^T[N,M] = B^T[N,K] * A^T[K,M]
//
// The logical transpose is only a change of interpretation: B has leading
// dimension N and A has leading dimension K in their original row-major
// buffers.  No transpose kernel or temporary matrix is needed.
void gemm_row_major(cublasHandle_t handle, const __half *a, const __half *b,
                    __half *c, int m, int n, int k, float alpha, float beta) {
  CUBLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      // Column-major dimensions of C^T.
      n, m, k,
      &alpha,
      // First operand is the column-major view B^T (the row-major B buffer).
      b, CUDA_R_16F, n,
      // Second operand is the column-major view A^T (the row-major A buffer).
      a, CUDA_R_16F, k,
      &beta, c, CUDA_R_16F, n,
      // FP32 accumulation is required even though all external matrices use
      // FP16 storage.  Tensor-op math permits Tensor Cores where applicable.
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

float benchmark_gemm(cublasHandle_t handle, const __half *a, const __half *b,
                     __half *c, int m, int n, int k, float alpha, float beta,
                     int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    gemm_row_major(handle, a, b, c, m, n, k, alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
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
  __half *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_b, b_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_c, c_elements * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, a_elements * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, b_elements * sizeof(__half), cudaMemcpyHostToDevice));

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  // The default stream keeps CUDA event timing and GEMM submission ordered.
  CUBLAS_CHECK(cublasSetStream(handle, 0));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // c is deliberately initialized once before timing.  Repeated GEMMs update
  // C when beta != 0, but that does not affect kernel timing.  It is restored
  // from C_initial before the final correctness check below.
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

  const double tflops = 2.0 * static_cast<double>(m) * n * k / avg_us / 1.0e6;
  std::printf("cublas_gemm M=%d N=%d K=%d alpha=%.2f beta=%.2f "
              "avg_us=%.3f tflops=%.2f\n",
              m, n, k, alpha, beta, avg_us, tflops);

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
