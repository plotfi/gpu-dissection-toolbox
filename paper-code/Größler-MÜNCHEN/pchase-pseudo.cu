/*

  This pchase work is from:

  Capturing the Memory Topology of GPUs

  https://mediatum.ub.tum.de/doc/1689994/1689994.pdf

*/

__global__ void anotherKernel() {


  unsigned int* ptr; unsigned int j = 0;

  for (int k = 0; k < array_length; k++) {
    ptr = my_array + j;

    asm volatile("ld.global.ca.u32 %0, [%1];"
                 : "=r"(j) : "l"(ptr) : "memory");
  }

  for (int k = 0; k < measureSize; k++) {
    ptr = my_array + j;
    asm volatile ("mov.u32 %0, %%clock;\n\t"
        "ld.global.ca.u32 %1, [%3];\n\t"
        "st.shared.u32 [smem_ptr64], %1;"
        "mov.u32 %2, %%clock;\n\t"
        "add.u64 smem_ptr64, smem_ptr64, 4;"
        : "=r"(start_time), "=r"(j), "=r"(end_time)
        : "l"(ptr) : "memory");
    s_tvalue[k] = end_time-start_time;
  }
}

