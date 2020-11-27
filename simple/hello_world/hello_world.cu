#include <stdio.h>

__global__ void kernel(int *A, int x) {
  A[x + x] = 42;
}

int main(int argc, char const *argv[])
{
  kernel<<<1,2>>>(0, 42);
  cudaDeviceSynchronize();
  return 0;
}
