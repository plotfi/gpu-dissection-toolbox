/*

   This code comes from the Zhang AMD paper:

   https://xianweiz.github.io/doc/papers/delta_memsys20.pdf

*/

#include <stdint.h>

using _TYPE = unsigned;

uint64_t clock() {
  return 0;
}

/* kernel: serially reading array elements */

__global__ void pchase_RO(_TYPE *array, unsigned array_length, unsigned m,
                          uint64_t *duration) {

  _TYPE j = 0;

  // GET THE RIGHT TID:
  unsigned tid = 0;

  unsigned k;
  uint64_t time = clock();

  for (k = 0; k < m; k++) {
    j = array[j];
  }

  // GET THE RIGHT TID
  duration[tid] = clock() - time;
}

// Setup Host Code
auto main() -> int {

  unsigned N = 32;
  unsigned stride = 8;
  _TYPE array[1024];

  /* setup: initialize array on CPU with the stride */
  for (unsigned i = 0; i < N; i++) {
    array[i] = static_cast<_TYPE>((i + stride) % N);
  }

  /* copy array to GPU, launch kernel (not shown) */
}


