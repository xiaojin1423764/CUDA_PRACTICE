#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

// 统一检查 CUDA Runtime API 的返回值，方便定位 CUDA 调用错误。
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Thrust transform 的一元函数：
// 输入 x，输出 exp(x - max_value)。减去 max_value 是 softmax 的数值稳定写法。
struct ExpShift {
  float max_value;

  __host__ __device__ float operator()(float x) const {
    return expf(x - max_value);
  }
};

// Thrust transform 的一元函数：
// 输入未归一化的 exp 值，输出 exp / sum。
struct DivideBy {
  float denominator;

  __host__ __device__ float operator()(float x) const {
    return x / denominator;
  }
};

// CUB 只负责 reduction，不负责逐元素 exp。
// 因此 CUB 版本仍需要一个简单 kernel 计算 output[i] = exp(input[i] - max_value)。
__global__ void exp_shift_kernel(const float *__restrict__ input,
                                 float *__restrict__ output, float max_value,
                                 int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = tid; i < n; i += stride) {
    output[i] = expf(input[i] - max_value);
  }
}

// CUB 求出 sum 后，使用这个 kernel 原地归一化 output。
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

// Thrust 调包版本。
//
// 计算流程：
// 1. thrust::max_element 求全局 max。
// 2. thrust::transform 计算 exp(x - max)。
// 3. thrust::reduce 求 sum(exp(x - max))。
// 4. thrust::transform 原地除以 sum。
//
// 这个版本代码最短，适合学习 Thrust 的高层算法接口。
void softmax_thrust(const float *h_input, float *h_output, int n) {
  thrust::device_vector<float> d_input(h_input, h_input + n);
  thrust::device_vector<float> d_output(n);

  // max_element 返回 device_vector 迭代器；解引用会把单个结果拷回 host。
  float max_value = *thrust::max_element(d_input.begin(), d_input.end());

  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(),
                    ExpShift{max_value});

  float sum_value = thrust::reduce(d_output.begin(), d_output.end(), 0.0f,
                                   thrust::plus<float>());

  thrust::transform(d_output.begin(), d_output.end(), d_output.begin(),
                    DivideBy{sum_value});

  thrust::copy(d_output.begin(), d_output.end(), h_output);
}

// CUB 调包版本。
//
// CUB 的 DeviceReduce 可以高效完成 Max 和 Sum，但它不是完整 softmax API；
// 逐元素 exp 和 normalize 仍用两个简单 CUDA kernel 完成。
//
// CUB reduction 的标准调用方式是两步：
// 1. temp storage 传 nullptr，查询临时空间大小。
// 2. cudaMalloc 临时空间，再调用一次真正执行 reduction。
void softmax_cub(const float *h_input, float *h_output, int n) {
  float *d_input = nullptr;
  float *d_output = nullptr;
  float *d_max = nullptr;
  float *d_sum = nullptr;
  void *d_max_temp = nullptr;
  void *d_sum_temp = nullptr;
  size_t max_temp_bytes = 0;
  size_t sum_temp_bytes = 0;

  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  // 求全局最大值 max(input)。
  CUDA_CHECK(cub::DeviceReduce::Max(nullptr, max_temp_bytes, d_input, d_max, n));
  CUDA_CHECK(cudaMalloc(&d_max_temp, max_temp_bytes));
  CUDA_CHECK(cub::DeviceReduce::Max(d_max_temp, max_temp_bytes, d_input, d_max,
                                    n));

  // 这里把 max 拷回 host 作为 kernel 参数，示例更直接。
  // 性能敏感实现可以让 exp_shift_kernel 从 device scalar 读取，避免这次 D2H 同步。
  float h_max = 0.0f;
  CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

  int block_size = 256;
  int blocks = div_up(n, block_size);
  exp_shift_kernel<<<blocks, block_size>>>(d_input, d_output, h_max, n);
  CUDA_CHECK(cudaGetLastError());

  // 对未归一化的 exp 输出求和，得到 softmax 分母。
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, sum_temp_bytes, d_output, d_sum,
                                    n));
  CUDA_CHECK(cudaMalloc(&d_sum_temp, sum_temp_bytes));
  CUDA_CHECK(cub::DeviceReduce::Sum(d_sum_temp, sum_temp_bytes, d_output, d_sum,
                                    n));

  // output[i] = exp(input[i] - max) / sum(exp(input - max))。
  normalize_kernel<<<blocks, block_size>>>(d_output, d_sum, n);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_sum_temp));
  CUDA_CHECK(cudaFree(d_max_temp));
  CUDA_CHECK(cudaFree(d_sum));
  CUDA_CHECK(cudaFree(d_max));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));
}

int main() {
  // 使用题面中的第二个样例，方便直接对照 softmax 结果。
  constexpr int n = 5;
  float h_input[n] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
  float h_thrust[n] = {};
  float h_cub[n] = {};

  softmax_thrust(h_input, h_thrust, n);
  softmax_cub(h_input, h_cub, n);

  std::printf("input : ");
  for (int i = 0; i < n; ++i) {
    std::printf("%.1f ", h_input[i]);
  }

  std::printf("\nthrust: ");
  for (int i = 0; i < n; ++i) {
    std::printf("%.9e ", h_thrust[i]);
  }

  std::printf("\ncub   : ");
  for (int i = 0; i < n; ++i) {
    std::printf("%.9e ", h_cub[i]);
  }
  std::printf("\n");

  // 两个库版本应只存在很小的浮点误差。
  float max_abs_error = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_abs_error = std::max(max_abs_error, std::fabs(h_thrust[i] - h_cub[i]));
  }
  std::printf("thrust_vs_cub_max_abs_error=%e\n", max_abs_error);

  return 0;
}
