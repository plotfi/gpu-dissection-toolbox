#include <cassert>
#include <exception>
#include <iostream>
#include <cstdio>

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

