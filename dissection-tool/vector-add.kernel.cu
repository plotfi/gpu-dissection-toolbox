__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= numElements)
    return;
  float a = A[i];
  float b = B[i];
  printf("(gridDim.x: %d, blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d): %f "
         "+ %f\n",
         gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, a, b);
  C[i] = a + b;
}
