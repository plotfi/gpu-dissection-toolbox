/// This pchase implementation is from:
/// Capturing the Memory Topology of GPUs
/// https://mediatum.ub.tum.de/doc/1689994/1689994.pdf
/// I call this the German Inline PTX Fine P-Chase

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

  asm volatile(".reg .u64 smem_ptr64;\n\t"
               "cvta.to.shared.u64 smem_ptr64, %0;\n\t"
               :: "l"(s_index));

  uint32_t tid = threadIdx.x;
  unsigned *ptr = nullptr;
  unsigned j = 0;

  // preheat the data
  for (int k = 0; k < array_length; k++) {
    ptr = array + j;
    asm volatile("ld.global.ca.u32 %0, [%1];"
                 : "=r"(j) : "l"(ptr) : "memory");
  }

  #pragma unroll 1
  for (unsigned k = 0; k < array_length; k++) {
    unsigned start_time = 0;
    unsigned end_time = 0;

    ptr = array + j;
    asm volatile ("mov.u32 %0, %%clock;\n\t"
        "ld.global.ca.u32 %1, [%3];\n\t"
        "st.shared.u32 [smem_ptr64], %1;"
        "mov.u32 %2, %%clock;\n\t"
        "add.u64 smem_ptr64, smem_ptr64, 4;"
        : "=r"(start_time), "=r"(j), "=r"(end_time)
        : "l"(ptr) : "memory");

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

