/// This pchase implementation is from:
/// Dissecting GPU Memory Hierarchy through Microbenchmarking
/// https://arxiv.org/abs/1509.02308
/// This is the Xinxin Mei Fine Pointer Chasing paper

/// Based on the classic P-Chase:
///
/// start_time = clock();
/// for (unsigned k = 0; k < m; k++) {
///   j = A[j];
/// }
/// end_time = clock();
/// tvalue = (end_time - start_time) / iterations;
///

#include <cuda_runtime.h>
#include <cstdint>

#define measureSize 32

__global__ void pchase(unsigned *array,
                       unsigned array_length,
                       uint64_t *duration) {
  __shared__ long long s_tvalue[measureSize];
  __shared__ unsigned int s_index[measureSize];

  uint32_t tid = threadIdx.x;
  unsigned *ptr = nullptr;
  unsigned j = 0;

  // preheat the data
  for (int k = 0; k < array_length; k++) {
    ptr = array + j;
    j = ptr[0];
  }

  #pragma unroll 1
  for (unsigned k = 0; k < array_length; k++) {
    unsigned start_time = 0;
    unsigned end_time = 0;

    start_time = clock();
    j = array[j];

    // TODO: WHY DO WE DO THIS???
    // store the array index
    s_index[k] = j;
    end_time = clock();

    // store the access latency
    s_tvalue[k] = end_time - start_time;
  }

  unsigned time_sum = 0;
  for (int k = 0; k < array_length; k++) {
    time_sum += s_tvalue[k];
  }
  duration[tid] = time_sum;
}

#if 0
// INIT
for (i=0; i < array_size; i++) {
  A[i] = (i + stride) % array_size;
}
#endif

