#include <assert.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <vector>

#include <stdint.h>

#define WARP_SIZE 8
#define L1_SIZE 32
#define THREADS_NUM 1

/// The following are a number renditions of the P-Chase algorithm from a number
/// of publications: Dissecting the NVIDIA Volta GPU Architecture via
/// Microbenchmarking: https://arxiv.org/abs/1804.06826 Dissecting GPU Memory
/// Hierarchy through Microbenchmarking: https://arxiv.org/abs/1509.02308
/// Capturing the Memory Topology of GPUs: https://hgpu.org/?p=27501
__global__ void l1_bw(uint32_t *startClk, uint32_t *stopClk, double *dsink,
                      uint64_t *posArray) { // thread index


  uint32_t tid = threadIdx.x;
  // a register to avoid compiler optimization
  double sink = 0;

  // populate l1 cache to warm up
  #pragma unroll 1
  for (uint32_t i = tid; i < L1_SIZE; i += THREADS_NUM) {
    double *ptr = (double*)(posArray + i);
    asm volatile("{\n\t\t"
                 ".reg .f64 data;\n\t\t"
                 "ld.global.ca.f64 data, [%1];\n\t\t"
                 "add.f64 %0, data, %0;\n\t"
                 "}\n"
                 : "+d"(sink)
                 : "l"(ptr)
                 : "memory");
  }

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  // load data from l1 cache and accumulate
  #pragma unroll 1
  for (uint32_t i = 0; i < L1_SIZE; i += THREADS_NUM) {
    double *ptr = (double*)(posArray + i); // every warp loads all data in l1 cache
    for (uint32_t j = 0; j < THREADS_NUM; j += WARP_SIZE) {
      uint32_t offset = (tid + j) % THREADS_NUM;
      asm volatile("{\n\t\t"
                   ".reg .f64 data;\n\t\t"
                   "ld.global.ca.f64 data, [%1];\n\t\t"
                   "add.f64 %0, data, %0;\n\t"
                   "}\n"
                   : "+d"(sink)
                   : "l"(ptr + offset)
                   : "memory");
    }
  }


  // synchronize all threads
  asm volatile("bar.sync 0;");
  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");
  // write time and data back to memory
  startClk[tid] = start;
  stopClk[tid] = stop;
  dsink[tid] = sink;
}

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= numElements)
    return;
  float a = A[i];
  float b = B[i];
  printf("(gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d): %f "
         "+ %f\n",
         gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, a, b);
  C[i] = a + b;
}

#define CUDA_CHECK(_err, msg)                                                  \
  do {                                                                         \
    cudaError_t __err = (_err);                                                \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Failed ");                                              \
      fprintf(stderr, msg);                                                    \
      fprintf(stderr, " (error code %s)!\n", cudaGetErrorString(__err));       \
      assert(false && "CUDA_CHECK FAILED");                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (false)

auto main(int argc, char **argv) -> int {

  if (argc != 3) {
    std::cerr << "Wrong arg count. Example:\n";
    std::cerr << argv[0] << " <number of elements> <threads per block>\n";
    return 0;
  }

  const int numElements = atoi(argv[1]);
  const int threadsPerBlock = atoi(argv[2]);
  const int blocksPerGrid =
      (numElements + threadsPerBlock - 1) / threadsPerBlock;
  const size_t size = numElements * sizeof(float);

  std::vector<float> h_A(size);
  std::vector<float> h_B(size);
  std::vector<float> h_C(size);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  for (int i = 0; i < numElements; ++i)
    for (auto a : {&h_A, &h_B})
      (*a)[i] = rand() / (float)RAND_MAX;

  for (auto d : {&d_A, &d_B, &d_C})
    CUDA_CHECK(cudaMalloc((void **)d, size), "on-device allocation");

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice),
             "host to device copy");
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice),
             "host to device copy");

  std::cout << "CUDA kernel launch with " << blocksPerGrid << " blocks of "
            << threadsPerBlock << " threads for vector add of " << numElements
            << " elements\n";
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  CUDA_CHECK(cudaGetLastError(), "kernel launch");

  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost),
             "device to host copy");

#if 1
  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) <= 1e-5)
      continue;
    std::cerr << "Verification failure at index " << i << "\n";
    assert(false);
    exit(EXIT_FAILURE);
  }
  printf("Test PASSED\n");
#endif

  for (auto d : {d_A, d_B, d_C})
    CUDA_CHECK(cudaFree(d), "free device memory");
  CUDA_CHECK(cudaDeviceReset(), "deinitialize the device");
  printf("Done\n");
}
