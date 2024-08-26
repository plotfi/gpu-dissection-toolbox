
// INIT
for (i=0; i < array_size; i++) {
  A[i] = (i + stride) % array_size;
}

// Classic PCHASE
start_time=clock();
for(it=0; it < iterations; it++) {
  j = A[j];
}
end_time = clock();
//calculate average memory latency
tvalue = (end_time − start_time) / iterations;


/// Fine-grained P-chase kernel (single thread, single CTA)
__global__ void KernelFunction(...) {
  // declare shared memory space
  __shared__ unsigned int s_tvalue[];
  __shared__ unsigned int s_index[];

  // preheat the data

  for (it = 0; it < iterations; it++) {
    uint32_t start_time = clock();
    j = my_array[j];

    // store the array index
    s_index[it] = j;
    uint32_t end_time = clock();

    // store the access latency
    s_tvalue[it] = end_time − start_time;
  }
}


