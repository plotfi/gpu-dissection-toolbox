#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

#include "error-macro.h"

// compile nvcc *.cu -o test

using READ_TYPE = unsigned;
#define READ_COUNT 512

__global__ void global_latency(READ_TYPE *my_array, int array_length,
                               int iterations, unsigned *duration,
                               READ_TYPE *index) {

  __shared__ unsigned s_tvalue[READ_COUNT];
  __shared__ READ_TYPE s_index[READ_COUNT];

  for (unsigned k = 0; k < READ_COUNT; k++) {
    s_tvalue[k] = 0;
    s_index[k] = 0;
  }

  // WARM UP
  READ_TYPE j = 0;
  for (int k = 0; k < iterations * READ_COUNT; k++) {
    j = my_array[j];
  }

  // P-Chase
  j = 0;
  int start_k = -iterations * 512;
  int end_k = iterations * READ_COUNT;
  for (int k = start_k; k < end_k; k++) {
    if (k >= 0) {
      int start_time = clock();
      j = my_array[j];
      s_index[k] = j;
      int end_time = clock();
      s_tvalue[k] = static_cast<unsigned>(end_time - start_time);
    } else {
      j = my_array[j];
    }
  }

  my_array[array_length] = j;
  my_array[array_length + 1] = my_array[j];

  for (int k = 0; k < READ_COUNT; k++) {
    duration[k] = s_tvalue[k];
    index[k] = s_index[k];
  }
}

template <typename T> auto DeviceMalloc(size_t count) -> T * {
  T *t = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&t, sizeof(T) * count),
             "on-device allocation");
  return t;
}

template <typename T>
auto MemCopyDeviceToHostAndFree(T *src, size_t count) -> std::vector<T> {
  std::vector<T> dst(count);
  CUDA_CHECK(cudaMemcpy((void *)dst.data(), (void *)src, sizeof(T) * count,
                        cudaMemcpyDeviceToHost),
             "device to host copy");
  CUDA_CHECK(cudaFree(src), "free device memory");
  return dst;
}

void parametric_measure_global(unsigned N, int iterations, unsigned stride) {

  // initialize array elements on CPU with pointers into d_a
  std::vector<READ_TYPE> h_a(N + 2, 0);
  for (unsigned i = 0; i < N; i++) {
    h_a[i] = (i + stride) % N;
  }

  // The read type granularity has to be the same as the offset s_index (ie
  // stride) type, but the duration is the type of the clock() (ie int)
  auto d_a = DeviceMalloc<READ_TYPE>(N + 2);
  auto duration = DeviceMalloc<unsigned>(READ_COUNT);
  auto d_index = DeviceMalloc<READ_TYPE>(READ_COUNT);

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sizeof(READ_TYPE) * N,
                        cudaMemcpyHostToDevice),
             "host to device copy");

  // launch kernel
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1, 1, 1);
  global_latency<<<Dg, Db>>>(d_a, N, iterations, duration, d_index);

  CUDA_CHECK(cudaGetLastError(), "kernel launch");
  CUDA_CHECK(cudaFree(d_a), "free device memory");

  auto h_timeinfo = MemCopyDeviceToHostAndFree<READ_TYPE>(duration, READ_COUNT);
  auto h_index = MemCopyDeviceToHostAndFree<READ_TYPE>(d_index, READ_COUNT);

  CUDA_CHECK(cudaDeviceReset(), "deinitialize the device");

  printf("\n=====%10.4f KB array, warm TLB, read %u element====\n",
         sizeof(READ_TYPE) * (float)N / 1024, READ_COUNT);
  printf("Stride = %u element, %lu byte\n", stride, stride * sizeof(READ_TYPE));

  for (unsigned i = 0; i < READ_COUNT; i++) {
    printf("%u\t %u\n", h_index[i], h_timeinfo[i]);
  }

  printf("\n===============================================\n\n");
}

auto pchase_mei_host(int argc, char **argv) -> int {
  // stride in element
  int iterations = 1;

  // 1. overflow cache with 1 element. stride=1, N=4097
  // 2. overflow cache with cache lines. stride=32, N_min=16*256, N_max=24*256
  unsigned stride = 128 / sizeof(READ_TYPE);

  unsigned START_N = 16 * 256;
  unsigned END_N = 24 * 256;

  for (unsigned N = START_N; N <= END_N; N += stride) {
    parametric_measure_global(N, iterations, stride);
  }

  return 0;
}

auto main(int argc, char **argv) -> int { return pchase_mei_host(argc, argv); }
