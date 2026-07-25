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

int main(int argc, char **argv) {
  int n = 1024;
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }

  // Host input array. malloc returns void*, so cast it to float* in C++.
  float *h_input = static_cast<float *>(std::malloc(n * sizeof(float)));
  for (int i = 0; i < n; ++i) {
    h_input[i] = 1.0f;
  }

  // Device input array and one-float device output.
  float *d_input = nullptr;
  float *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

  // CUB uses temporary device memory while reducing.
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // First call only asks CUB how many bytes of temporary storage it needs.
  // d_temp_storage is nullptr, and temp_storage_bytes is filled by CUB.
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_input,
                                    d_output, n));

  // Allocate that temporary storage, then call Sum again to really compute:
  // d_output[0] = d_input[0] + d_input[1] + ... + d_input[n - 1].
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input,
                                    d_output, n));

  // Copy the one-float result back to host memory and print it.
  float h_output = 0.0f;
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::printf("sum = %f\n", h_output);

  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));
  std::free(h_input);
  return 0;
}
