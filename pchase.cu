#include <stdint.h>

#define WARP_SIZE 8
#define L1_SIZE 32
#define THREADS_NUM 1

/// The following are a number renditions of the P-Chase algorithm from a number
/// of publications: Dissecting the NVIDIA Volta GPU Architecture via
/// Microbenchmarking: https://arxiv.org/abs/1804.06826 Dissecting GPU Memory
/// Hierarchy through Microbenchmarking: https://arxiv.org/abs/1509.02308
/// Capturing the Memory Topology of GPUs: https://hgpu.org/?p=27501

/// From 1804.06826 Citadel Paper (https://arxiv.org/abs/1804.06826)
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

    double *ptr = (double*)(posArray + i);

    // every warp loads all data in l1 cache

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
