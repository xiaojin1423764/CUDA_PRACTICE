#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

void prefix_sum_thrust(const float *h_input, float *h_output, int n) {
  thrust::device_vector<float> d_input(h_input, h_input + n);
  thrust::device_vector<float> d_output(n);

  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());

  thrust::copy(d_output.begin(), d_output.end(), h_output);
}

void prefix_sum_cub(const float *h_input, float *h_output, int n) {
  float *d_input = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // First call: query how much temporary device memory CUB needs.
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_input,
                                           d_output, n));

  // Second call: provide the temporary memory and run inclusive prefix sum.
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                           d_input, d_output, n));

  CUDA_CHECK(cudaMemcpy(h_output, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));
}

int main() {
  constexpr int n = 8;
  float h_input[n] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
  float h_thrust[n] = {};
  float h_cub[n] = {};

  prefix_sum_thrust(h_input, h_thrust, n);
  prefix_sum_cub(h_input, h_cub, n);

  std::printf("input : ");
  for (int i = 0; i < n; ++i) {
    std::printf("%.0f ", h_input[i]);
  }

  std::printf("\nthrust: ");
  for (int i = 0; i < n; ++i) {
    std::printf("%.0f ", h_thrust[i]);
  }

  std::printf("\ncub   : ");
  for (int i = 0; i < n; ++i) {
    std::printf("%.0f ", h_cub[i]);
  }
  std::printf("\n");

  return 0;
}
