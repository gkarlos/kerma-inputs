#include <stdio.h>

__global__ void kernel(int __attribute__((annotate("10"))) * A)
    __attribute__((annotate("2:2"))) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  A[i] = 42;
  A[i + 1] = 42;
  A[i + 2] = 42;
  A[i + 3] = 42;
}

int main(int argc, char const *argv[]) {
  int *A;
  kernel<<<2, 1000>>>(A);
  cudaDeviceSynchronize();
  return 0;
}
