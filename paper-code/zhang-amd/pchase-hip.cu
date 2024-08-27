/// This pchase implementation is from:
/// DELTA: Validate GPU Memory Profiling with Microbenchmarks
/// https://xianweiz.github.io/doc/papers/delta_memsys20.pdf
/// This is the AMD HIP Pointer Chasing paper

#include <cuda_runtime.h>
#include <cstdint>

__global__ void hip_pchase(unsigned *array,
                           unsigned array_length,
                           uint64_t *duration) {
  uint32_t tid = threadIdx.x;
  unsigned j = 0;

  uint64_t time = clock();

  #pragma unroll 1
  for (unsigned k = 0; k < array_length; k++) {
    j = array[j];
  }

  // GET THE RIGHT TID
  duration[tid] = clock() - time;
}

#if 0
// Setup Host Code
auto main() -> int {

  unsigned N = 32;
  unsigned stride = 8;
  unsigned array[1024];

  /* setup: initialize array on CPU with the stride */
  for (unsigned i = 0; i < N; i++) {
    array[i] = static_cast<unsigned>((i + stride) % N);
  }

  /* copy array to GPU, launch kernel (not shown) */
}
#endif

