/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a) {
  if (a < 0) {
    return (a * -1);
  } else {
    return a;
  }
}

float percentDiff(double val1, double val2) {
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) {
    return 0.0f;
  }

  else {
    return 100.0f *
           (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
  }
}

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size */
#define NI 256
#define NJ 256
#define NK 256

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv3D(DATA_TYPE *A, DATA_TYPE *B) {
  int i, j, k;
  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

  for (i = 1; i < NI - 1; ++i) // 0
  {
    for (j = 1; j < NJ - 1; ++j) // 1
    {
      for (k = 1; k < NK - 1; ++k) // 2
      {
        // printf("i:%d\nj:%d\nk:%d\n", i, j, k);
        B[i * (NK * NJ) + j * NK + k] =
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
            c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
            c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
            c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
            c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
            c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
            c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
            c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
  }
}

void init(DATA_TYPE *A) {
  int i, j, k;

  for (i = 0; i < NI; ++i) {
    for (j = 0; j < NJ; ++j) {
      for (k = 0; k < NK; ++k) {
        A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
      }
    }
  }
}

void compareResults(DATA_TYPE *B, DATA_TYPE *B_outputFromGpu) {
  int i, j, k, fail;
  fail = 0;

  // Compare result from cpu and gpu...
  for (i = 1; i < NI - 1; ++i) // 0
  {
    for (j = 1; j < NJ - 1; ++j) // 1
    {
      for (k = 1; k < NK - 1; ++k) // 2
      {
        if (percentDiff(B[i * (NK * NJ) + j * NK + k],
                        B_outputFromGpu[i * (NK * NJ) + j * NK + k]) >
            PERCENT_DIFF_ERROR_THRESHOLD) {
          fail++;
        }
      }
    }
  }

  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
  cudaSetDevice(GPU_DEVICE);
}

__global__ void
convolution3D_kernel(DATA_TYPE __attribute__((annotate("256,256,256"))) * A,
                     DATA_TYPE __attribute__((annotate("256,256,256"))) * B,
                     int __attribute__((annotate("4"))) i)
    __attribute__((annotate("32,8:8,32"))) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

  c11 = +2;
  c21 = +5;
  c31 = -8;
  c12 = -3;
  c22 = +6;
  c32 = -9;
  c13 = +4;
  c23 = +7;
  c33 = +10;

  if ((i < (NI - 1)) && (j < (NJ - 1)) && (k < (NK - 1)) && (i > 0) &&
      (j > 0) && (k > 0)) {
    B[i * (NK * NJ) + j * NK + k] =
        c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
        c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
        c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
        c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
        c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
        c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
        c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
        c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
        c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
        c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
        c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
        c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
        c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
        c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
        c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
  }
}

void convolution3DCuda(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *B_outputFromGpu) {
  double t_start, t_end;

  DATA_TYPE *A_gpu;
  DATA_TYPE *B_gpu;

  cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
  cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
  cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK,
             cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ * NK,
             cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)(ceil(((float)NK) / ((float)block.x))),
            (size_t)(ceil(((float)NJ) / ((float)block.y))));

  t_start = rtclock();
  int i;
  for (i = 1; i < NI - 1; ++i) // 0
  {
    convolution3D_kernel<<<grid, block>>>(A_gpu, B_gpu, i);
  }

  cudaDeviceSynchronize();
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK,
             cudaMemcpyDeviceToHost);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
}

int main(int argc, char *argv[]) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *B_outputFromGpu;

  A = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));
  B_outputFromGpu = (DATA_TYPE *)malloc(NI * NJ * NK * sizeof(DATA_TYPE));

  init(A);

  GPU_argv_init();

  convolution3DCuda(A, B, B_outputFromGpu);

  t_start = rtclock();
  conv3D(A, B);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(B, B_outputFromGpu);

  free(A);
  free(B);
  free(B_outputFromGpu);

  return 0;
}