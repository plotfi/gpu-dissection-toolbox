#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

auto vector_add_host(int argc, char **argv) -> int {

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

  return 0;
}

