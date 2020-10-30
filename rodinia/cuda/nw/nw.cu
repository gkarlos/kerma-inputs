
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef RD_WG_SIZE_0_0
	#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE RD_WG_SIZE
#else
	#define BLOCK_SIZE 16
#endif

#define LIMIT -999

// includes, kernels
// #include "needle_kernel.cu"

#define SDATA(index) CUT_BANK_CHECKER(sdata, index)

__device__ __host__ int maximum(int a, int b, int c) {

  int k;
  if (a <= b)
    k = b;
  else
    k = a;

  if (k <= c)
    return (c);
  else
    return (k);
}

__global__ void needle_cuda_shared_1(int *referrence, int *matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  int index =
      cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);
  int index_n =
      cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1);
  int index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols);
  int index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
  __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];

  if (tx == 0)
    temp[tx][0] = matrix_cuda[index_nw];

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
    ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];

  __syncthreads();

  for (int m = 0; m < BLOCK_SIZE; m++) {

    if (tx <= m) {

      int t_index_x = tx + 1;
      int t_index_y = m - tx + 1;

      temp[t_index_y][t_index_x] =
          maximum(temp[t_index_y - 1][t_index_x - 1] +
                      ref[t_index_y - 1][t_index_x - 1],
                  temp[t_index_y][t_index_x - 1] - penalty,
                  temp[t_index_y - 1][t_index_x] - penalty);
    }

    __syncthreads();
  }

  for (int m = BLOCK_SIZE - 2; m >= 0; m--) {

    if (tx <= m) {

      int t_index_x = tx + BLOCK_SIZE - m;
      int t_index_y = BLOCK_SIZE - tx;

      temp[t_index_y][t_index_x] =
          maximum(temp[t_index_y - 1][t_index_x - 1] +
                      ref[t_index_y - 1][t_index_x - 1],
                  temp[t_index_y][t_index_x - 1] - penalty,
                  temp[t_index_y - 1][t_index_x] - penalty);
    }

    __syncthreads();
  }

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
    matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
}

__global__ void needle_cuda_shared_2(int *referrence, int *matrix_cuda,
                                     int cols, int penalty, int i,
                                     int block_width) {

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx + block_width - i;
  int b_index_y = block_width - bx - 1;

  int index =
      cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);
  int index_n =
      cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1);
  int index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols);
  int index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
  __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
    ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

  if (tx == 0)
    temp[tx][0] = matrix_cuda[index_nw];

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];

  __syncthreads();

  for (int m = 0; m < BLOCK_SIZE; m++) {

    if (tx <= m) {

      int t_index_x = tx + 1;
      int t_index_y = m - tx + 1;

      temp[t_index_y][t_index_x] =
          maximum(temp[t_index_y - 1][t_index_x - 1] +
                      ref[t_index_y - 1][t_index_x - 1],
                  temp[t_index_y][t_index_x - 1] - penalty,
                  temp[t_index_y - 1][t_index_x] - penalty);
    }

    __syncthreads();
  }

  for (int m = BLOCK_SIZE - 2; m >= 0; m--) {

    if (tx <= m) {

      int t_index_x = tx + BLOCK_SIZE - m;
      int t_index_y = BLOCK_SIZE - tx;

      temp[t_index_y][t_index_x] =
          maximum(temp[t_index_y - 1][t_index_x - 1] +
                      ref[t_index_y - 1][t_index_x - 1],
                  temp[t_index_y][t_index_x - 1] - penalty,
                  temp[t_index_y - 1][t_index_x] - penalty);
    }

    __syncthreads();
  }

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
    matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

int blosum62[24][24] = {{4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
                         -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4},
                        {-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
                         -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4},
                        {-2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
                         -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4},
                        {-2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
                         -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4},
                        {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
                         -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
                         0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4},
                        {-1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
                         -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
                         -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
                         -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
                         1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
                         2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},
                        {-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
                         -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4},
                        {-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
                         5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
                         0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
                         -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
                         -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4},
                        {0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
                         -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
                         -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
                         -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
                         1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4},
                        {-2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
                         -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4},
                        {-1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
                         -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  printf("WG size of kernel = %d \n", BLOCK_SIZE);

  runTest(argc, argv);

  return EXIT_SUCCESS;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
  exit(1);
}

void runTest(int argc, char **argv) {
  int max_rows, max_cols, penalty;
  int *input_itemsets, *output_itemsets, *referrence;
  int *matrix_cuda, *referrence_cuda;
  int size;

  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  if (argc == 3) {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
  } else {
    usage(argc, argv);
  }

  if (atoi(argv[1]) % 16 != 0) {
    fprintf(stderr, "The dimension values must be a multiple of 16\n");
    exit(1);
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
  input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
  output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

  if (!input_itemsets)
    fprintf(stderr, "error: can not allocate memory");

  srand(7);

  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  printf("Start Needleman-Wunsch\n");

  for (int i = 1; i < max_rows; i++) { // please define your own sequence.
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  }
  for (int j = 1; j < max_cols; j++) { // please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      referrence[i * max_cols + j] =
          blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (int i = 1; i < max_rows; i++)
    input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; j++)
    input_itemsets[j] = -j * penalty;

  size = max_cols * max_rows;
  cudaMalloc((void **)&referrence_cuda, sizeof(int) * size);
  cudaMalloc((void **)&matrix_cuda, sizeof(int) * size);

  cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size,
             cudaMemcpyHostToDevice);

  dim3 dimGrid;
  dim3 dimBlock(BLOCK_SIZE, 1);
  int block_width = (max_cols - 1) / BLOCK_SIZE;

  printf("Processing top-left matrix\n");
  // process top-left matrix
  for (int i = 1; i <= block_width; i++) {
    dimGrid.x = i;
    dimGrid.y = 1;
    needle_cuda_shared_1<<<dimGrid, dimBlock>>>(
        referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
  }
  printf("Processing bottom-right matrix\n");
  // process bottom-right matrix
  for (int i = block_width - 1; i >= 1; i--) {
    dimGrid.x = i;
    dimGrid.y = 1;
    needle_cuda_shared_2<<<dimGrid, dimBlock>>>(
        referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
  }

  cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size,
             cudaMemcpyDeviceToHost);

//#define TRACEBACK
#ifdef TRACEBACK

  FILE *fpo = fopen("result.txt", "w");
  fprintf(fpo, "print traceback value GPU:\n");

  for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;) {
    int nw, n, w, traceback;
    if (i == max_rows - 2 && j == max_rows - 2)
      fprintf(fpo, "%d ",
              output_itemsets[i * max_cols + j]); // print the first element
    if (i == 0 && j == 0)
      break;
    if (i > 0 && j > 0) {
      nw = output_itemsets[(i - 1) * max_cols + j - 1];
      w = output_itemsets[i * max_cols + j - 1];
      n = output_itemsets[(i - 1) * max_cols + j];
    } else if (i == 0) {
      nw = n = LIMIT;
      w = output_itemsets[i * max_cols + j - 1];
    } else if (j == 0) {
      nw = w = LIMIT;
      n = output_itemsets[(i - 1) * max_cols + j];
    } else {
    }

    // traceback = maximum(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + referrence[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if (traceback == new_nw)
      traceback = nw;
    if (traceback == new_w)
      traceback = w;
    if (traceback == new_n)
      traceback = n;

    fprintf(fpo, "%d ", traceback);

    if (traceback == nw) {
      i--;
      j--;
      continue;
    }

    else if (traceback == w) {
      j--;
      continue;
    }

    else if (traceback == n) {
      i--;
      continue;
    }

    else
      ;
  }

  fclose(fpo);

#endif

  cudaFree(referrence_cuda);
  cudaFree(matrix_cuda);

  free(referrence);
  free(input_itemsets);
  free(output_itemsets);
}
