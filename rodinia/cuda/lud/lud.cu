/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <assert.h>
#include <cuda.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
// Common
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
#define GET_RAND_FP ((float)rand() / ((float)(RAND_MAX) + (float)(1)))

#define MIN(i, j) ((i) < (j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE { RET_SUCCESS, RET_FAILURE } func_ret_t;

typedef struct __stopwatch_t{
    struct timeval begin;
    struct timeval end;
}stopwatch;

void stopwatch_start(stopwatch *sw) {
  if (sw == NULL)
    return;

  bzero(&sw->begin, sizeof(struct timeval));
  bzero(&sw->end, sizeof(struct timeval));

  gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw) {
  if (sw == NULL)
    return;

  gettimeofday(&sw->end, NULL);
}

double get_interval_by_sec(stopwatch *sw) {
  if (sw == NULL)
    return 0;
  return ((double)(sw->end.tv_sec - sw->begin.tv_sec) +
          (double)(sw->end.tv_usec - sw->begin.tv_usec) / 1000000);
}

int get_interval_by_usec(stopwatch *sw) {
  if (sw == NULL)
    return 0;
  return ((sw->end.tv_sec - sw->begin.tv_sec) * 1000000 +
          (sw->end.tv_usec - sw->begin.tv_usec));
}

func_ret_t create_matrix_from_file(float **mp, const char *filename,
                                   int *size_p) {
  int i, j, size;
  float *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if (fp == NULL) {
    return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

  m = (float *)malloc(sizeof(float) * size * size);
  if (m == NULL) {
    fclose(fp);
    return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      fscanf(fp, "%f ", m + i * size + j);
    }
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}

func_ret_t create_matrix_from_random(float **mp, int size) {
  float *l, *u, *m;
  int i, j, k;

  srand(time(NULL));

  l = (float *)malloc(size * size * sizeof(float));
  if (l == NULL)
    return RET_FAILURE;

  u = (float *)malloc(size * size * sizeof(float));
  if (u == NULL) {
    free(l);
    return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (i > j) {
        l[i * size + j] = GET_RAND_FP;
      } else if (i == j) {
        l[i * size + j] = 1;
      } else {
        l[i * size + j] = 0;
      }
    }
  }

  for (j = 0; j < size; j++) {
    for (i = 0; i < size; i++) {
      if (i > j) {
        u[j * size + i] = 0;
      } else {
        u[j * size + i] = GET_RAND_FP;
      }
    }
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      for (k = 0; k <= MIN(i, j); k++)
        m[i * size + j] = l[i * size + k] * u[j * size + k];
    }
  }

  free(l);
  free(u);

  *mp = m;

  return RET_SUCCESS;
}

void matrix_multiply(float *inputa, float *inputb, float *output, int size) {
  int i, j, k;

  for (i = 0; i < size; i++)
    for (k = 0; k < size; k++)
      for (j = 0; j < size; j++)
        output[i * size + j] = inputa[i * size + k] * inputb[k * size + j];
}

func_ret_t lud_verify(float *m, float *lu, int matrix_dim) {
  int i, j, k;
  float *tmp = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));

  for (i = 0; i < matrix_dim; i++)
    for (j = 0; j < matrix_dim; j++) {
      float sum = 0;
      float l, u;
      for (k = 0; k <= MIN(i, j); k++) {
        if (i == k)
          l = 1;
        else
          l = lu[i * matrix_dim + k];
        u = lu[k * matrix_dim + j];
        sum += l * u;
      }
      tmp[i * matrix_dim + j] = sum;
    }

  func_ret_t ret = RET_SUCCESS;

  for (i = 0; i < matrix_dim; i++) {
    for (j = 0; j < matrix_dim; j++) {
      if (fabs(m[i * matrix_dim + j] - tmp[i * matrix_dim + j]) > 0.0001) {
        ret = RET_FAILURE;
        printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j,
               m[i * matrix_dim + j], tmp[i * matrix_dim + j]);
      }
    }
  }
  free(tmp);

  return ret;
}

void matrix_duplicate(float *src, float **dst, int matrix_dim) {
  int s = matrix_dim * matrix_dim * sizeof(float);
  float *p = (float *)malloc(s);
  memcpy(p, src, s);
  *dst = p;
}

void print_matrix(float *m, int matrix_dim) {
  int i, j;
  for (i = 0; i < matrix_dim; i++) {
    for (j = 0; j < matrix_dim; j++)
      printf("%f ", m[i * matrix_dim + j]);
    printf("\n");
  }
}

// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t create_matrix(float **mp, int size) {
  float *m;
  int i, j;
  float lamda = -0.001;
  float coe[2 * size - 1];
  float coe_i = 0.0;

  for (i = 0; i < size; i++) {
    coe_i = 10 * exp(lamda * i);
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }

  m = (float *)malloc(sizeof(float) * size * size);
  if (m == NULL) {
    return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      m[i * size + j] = coe[size - 1 - i + j];
    }
  }

  *mp = m;

  return RET_SUCCESS;
}
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

static int do_verify = 0;

static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"input", 1, NULL, 'i'},
    {"size", 1, NULL, 's'},
    {"verify", 0, NULL, 'v'},
    {0, 0, 0, 0}};

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

__global__ void lud_diagonal(float *m, int matrix_dim, int offset) {
  int i, j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset * matrix_dim + offset;
  for (i = 0; i < BLOCK_SIZE; i++) {
    shadow[i][threadIdx.x] = m[array_offset + threadIdx.x];
    array_offset += matrix_dim;
  }
  __syncthreads();
  for (i = 0; i < BLOCK_SIZE - 1; i++) {

    if (threadIdx.x > i) {
      for (j = 0; j < i; j++)
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j] * shadow[j][i];
      shadow[threadIdx.x][i] /= shadow[i][i];
    }

    __syncthreads();
    if (threadIdx.x > i) {

      for (j = 0; j < i + 1; j++)
        shadow[i + 1][threadIdx.x] -= shadow[i + 1][j] * shadow[j][threadIdx.x];
    }
    __syncthreads();
  }

  /*
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset + 1) * matrix_dim + offset;
  for (i = 1; i < BLOCK_SIZE; i++) {
    m[array_offset + threadIdx.x] = shadow[i][threadIdx.x];
    array_offset += matrix_dim;
  }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset) {
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i, j, array_offset;
  int idx;

  if (threadIdx.x < BLOCK_SIZE) {
    idx = threadIdx.x;

    array_offset = offset * matrix_dim + offset;
    for (i = 0; i < BLOCK_SIZE / 2; i++) {
      dia[i][idx] = m[array_offset + idx];
      array_offset += matrix_dim;
    }

    array_offset = offset * matrix_dim + offset;
    for (i = 0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx] = m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = threadIdx.x - BLOCK_SIZE;

    array_offset = (offset + BLOCK_SIZE / 2) * matrix_dim + offset;
    for (i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
      dia[i][idx] = m[array_offset + idx];
      array_offset += matrix_dim;
    }

    array_offset =
        (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
    for (i = 0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset + idx];
      array_offset += matrix_dim;
    }
  }
  __syncthreads();

  if (threadIdx.x < BLOCK_SIZE) { // peri-row
    idx = threadIdx.x;
    for (i = 1; i < BLOCK_SIZE; i++) {
      for (j = 0; j < i; j++)
        peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
    }
  } else { // peri-col
    idx = threadIdx.x - BLOCK_SIZE;
    for (i = 0; i < BLOCK_SIZE; i++) {
      for (j = 0; j < i; j++)
        peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();

  if (threadIdx.x < BLOCK_SIZE) { // peri-row
    idx = threadIdx.x;
    array_offset = (offset + 1) * matrix_dim + offset;
    for (i = 1; i < BLOCK_SIZE; i++) {
      m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { // peri-col
    idx = threadIdx.x - BLOCK_SIZE;
    array_offset =
        (offset + (blockIdx.x + 1) * BLOCK_SIZE) * matrix_dim + offset;
    for (i = 0; i < BLOCK_SIZE; i++) {
      m[array_offset + idx] = peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
}

__global__ void lud_internal(float *m, int matrix_dim, int offset) {
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] =
      m[(offset + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] =
      m[(global_row_id + threadIdx.y) * matrix_dim + offset + threadIdx.x];

  __syncthreads();

  sum = 0;
  for (i = 0; i < BLOCK_SIZE; i++)
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -=
      sum;
}

void lud_cuda(float *m, int matrix_dim) {
  int i = 0;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float *)malloc(matrix_dim * matrix_dim * sizeof(float));

  for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
    lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
    lud_perimeter<<<(matrix_dim - i) / BLOCK_SIZE - 1, BLOCK_SIZE * 2>>>(
        m, matrix_dim, i);
    dim3 dimGrid((matrix_dim - i) / BLOCK_SIZE - 1,
                 (matrix_dim - i) / BLOCK_SIZE - 1);
    lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i);
  }
  lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
}

int main(int argc, char *argv[]) {
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index = 0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *d_m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
      // argv[0]); exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
              argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if ((optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }

  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify) {
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  cudaMalloc((void **)&d_m, matrix_dim * matrix_dim * sizeof(float));

  /* beginning of timing point */
  stopwatch_start(&sw);
  cudaMemcpy(d_m, m, matrix_dim * matrix_dim * sizeof(float),
             cudaMemcpyHostToDevice);

  lud_cuda(d_m, matrix_dim);

  cudaMemcpy(m, d_m, matrix_dim * matrix_dim * sizeof(float),
             cudaMemcpyDeviceToHost);

  /* end of timing point */
  stopwatch_stop(&sw);
  printf("Time consumed(ms): %lf\n", 1000 * get_interval_by_sec(&sw));

  cudaFree(d_m);

  if (do_verify) {
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim);
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
} /* ----------  end of function main  ---------- */
