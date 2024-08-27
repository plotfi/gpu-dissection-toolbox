///
/// Puyan Lotfi - puyan@puyan.org
///

/// This pchase implementation is from:
/// Dissecting GPU Memory Hierarchy through Microbenchmarking
/// https://arxiv.org/abs/1509.02308
/// This is the Xinxin Mei Fine Pointer Chasing paper
///
/// The inline PTX portions come from:
/// Capturing the Memory Topology of GPUs
/// https://mediatum.ub.tum.de/doc/1689994/1689994.pdf
///
/// Also looked at the inline asm non-fined grained kernel from Citadel:
/// Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking
/// https://arxiv.org/abs/1804.06826
///


/// Based on the classic P-Chase:
///
/// start_time = clock();
/// for (unsigned k = 0; k < m; k++) {
///   k = A[k];
/// }
/// end_time = clock();
/// tvalue = (end_time - start_time) / iterations;
///

#include <cuda_runtime.h>
#include <cstdint>

#define measureSize 32

#define INLINE_PTX 1
#define SYNC_PTX 1

#if SYNC_PTX
#define SYNC_THREADS() asm volatile("bar.sync 0;")
#else
#define SYNC_THREADS()
#endif

__global__ void pchase(unsigned *array,
                       unsigned array_length,
                       unsigned *sink_ptr,
                       uint64_t *duration) {
  __shared__ long long s_tvalue[measureSize];
  __shared__ unsigned int s_index[measureSize];

#if INLINE_PTX
  asm volatile("{\n\t"
               "  // WIRE UP SMEM\n\t"
               "  .reg .u64 smem_ptr64;\n\t"
               "  cvta.to.shared.u64 smem_ptr64, %0;\n\t"
               "}"
               :: "l"(s_index));
#endif

  uint32_t tid = threadIdx.x;
  unsigned *ptr = nullptr;
  unsigned j = 0;
  unsigned sink = 0;

  // preheat the data
  #pragma unroll 1
  for (unsigned k = 0; k < array_length; k++) {
    ptr = array + j;

#if INLINE_PTX
    asm volatile("{\n\t"
                 "  // WARM UP\n\t"
                 "  ld.global.ca.u32 %0, [%1];\n\t"
                 "}"
                 : "=r"(j) : "l"(ptr) : "memory");
#else
    j = ptr[0];
#endif

    sink += j;
  }
  SYNC_THREADS();

  #pragma unroll 1
  for (unsigned k = 0; k < array_length; k++) {
    unsigned start_time = 0;
    unsigned end_time = 0;

#if INLINE_PTX
    ptr = array + j;
    asm volatile("{\n\t"
        "  // P-CHASE BODY\n\t"
        "  mov.u32 %0, %%clock;\n\t"
        "  ld.global.ca.u32 %1, [%3];\n\t"
        "  st.shared.u32 [smem_ptr64], %1;\n\t"
        "  mov.u32 %2, %%clock;\n\t"
        "  add.u64 smem_ptr64, smem_ptr64, 4;\n\t"
        "}"
        : "=r"(start_time), "=r"(j), "=r"(end_time)
        : "l"(ptr) : "memory");
#else
    start_time = clock();
    j = array[j];

    // TODO: WHY DO WE DO THIS???
    // store the array index
    s_index[k] = j;
    end_time = clock();
#endif

    // store the access latency
    s_tvalue[k] = end_time - start_time;
    sink += j;
  }
  SYNC_THREADS();

  unsigned time_sum = 0;
  #pragma unroll 1
  for (unsigned k = 0; k < array_length; k++) {
    time_sum += s_tvalue[k];
  }
  SYNC_THREADS();

  duration[tid] = time_sum;
  sink_ptr[tid] = sink;
}

#if 0
// INIT
for (i=0; i < array_size; i++) {
  A[i] = (i + stride) % array_size;
}
#endif

