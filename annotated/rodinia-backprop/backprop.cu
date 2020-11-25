#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <sys/time.h>

#define BIGRND 0x7fffffff

#define GPU
#define THREADS 256
#define WIDTH 16  // shared memory width
#define HEIGHT 16 // shared memory height

#define ETA 0.3      // eta value
#define MOMENTUM 0.3 // momentum value
#define NUM_THREAD 4 // OpenMP threads

#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

////////////////////////////////////////////////////////////////////////////////

typedef struct {
  int input_n;  /* number of input units */
  int hidden_n; /* number of hidden units */
  int output_n; /* number of output units */

  float *input_units;  /* the input units */
  float *hidden_units; /* the hidden units */
  float *output_units; /* the output units */

  float *hidden_delta; /* storage for hidden unit error */
  float *output_delta; /* storage for output unit error */

  float *target; /* storage for target vector */

  float **input_weights;  /* weights from input to hidden layer */
  float **hidden_weights; /* weights from hidden to output layer */

  /*** The next two are for momentum ***/
  float **input_prev_weights;  /* previous change on input to hidden wgt */
  float **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;

int layer_size = 0;
unsigned int num_threads = 0;
unsigned int num_blocks = 0;

extern "C" __global__ void bpnn_layerforward_CUDA(
    __attribute__((annotate("640001"))) float *input_cuda,
    __attribute__((annotate("17"))) float *output_hidden_cuda,
    __attribute__((annotate("17,640001"))) float *input_hidden_cuda,
    __attribute__((annotate("16,40000"))) float *hidden_partial_sum,
    __attribute__((annotate("640000"))) int in,
    __attribute__((annotate("16"))) int hid)
    __attribute__((annotate("1,40000:16,16"))) {
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_in = HEIGHT * by + ty + 1;

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT][WIDTH];

  if (tx == 0)
    input_node[ty] = input_cuda[index_in];

  __syncthreads();

  weight_matrix[ty][tx] = input_hidden_cuda[index];

  __syncthreads();

  weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

  __syncthreads();

  for (int i = 1; i <= __log2f(HEIGHT); i++) {
    int power_two = __powf(2, i);
    if (ty % power_two == 0)
      weight_matrix[ty][tx] =
          weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];
    __syncthreads();
  }
  //__syncthreads();

  input_hidden_cuda[index] = weight_matrix[ty][tx];

  /*
    for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
      unsigned int power_two = i - 1;
      if( (ty & power_two) == 0 ) {
        weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty +
    power_two/2][tx];
      }
     }
  */

  __syncthreads();

  if (tx == 0) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
  }
}

extern "C" __global__ void
bpnn_adjust_weights_cuda(__attribute__((annotate("17"))) float *delta,
                         __attribute__((annotate("16"))) int hid,
                         __attribute__((annotate("640001"))) float *ly,
                         __attribute__((annotate("640000"))) int in,
                         __attribute__((annotate("17,640001"))) float *w,
                         __attribute__((annotate("17,640001"))) float *oldw)
    __attribute__((annotate("640001"))) {
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  oldw[index] =
      ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  __syncthreads();

  if (ty == 0 && by == 0) {
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}

extern "C" float squash(float x) {
  // float m;
  // x = -x;
  // m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  // return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}

extern "C" void bpnn_layerforward(float *l1, float *l2, float **conn, int n1,
                                  int n2) {
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(+: sum) schedule(static)
#endif
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {

    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }
}

extern "C" void bpnn_output_error(float *delta, float *target, float *output,
                                  int nj, float *err) {
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

extern "C" void bpnn_hidden_error(float *delta_h, int nh, float *delta_o,
                                  int no, float **who, float *hidden,
                                  float *err) {
  int j, k;
  float h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

extern "C" void bpnn_adjust_weights(float *delta, int ndelta, float *ly,
                                    int nly, float **w, float **oldw) {
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  // eta = 0.3;
  // momentum = 0.3;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
#pragma omp parallel for shared(oldw, w, delta) private(j, k, new_dw)          \
    firstprivate(ndelta, nly, momentum)
#endif
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}

extern "C" float *alloc_1d_dbl(int n) {
  float *p;
  p = (float *)malloc((unsigned)(n * sizeof(float)));
  if (p == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return p;
}

extern "C" float **alloc_2d_dbl(int m, int n) {
  int i;
  float **p;
  p = (float **)malloc((unsigned)(m * sizeof(float *)));
  if (p == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }
  for (i = 0; i < m; i++) {
    p[i] = alloc_1d_dbl(n);
  }
  return p;
}

extern "C" void bpnn_initialize(int seed) {
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}

extern "C" BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out) {
  BPNN *newnet;

  newnet = (BPNN *)malloc(sizeof(BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}

extern "C" void bpnn_randomize_weights(float **w, int m, int n) {
  int i, j;
  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

extern "C" void bpnn_randomize_row(float *w, int m) {
  int i;
  for (i = 0; i <= m; i++) {
    // w[i] = (float) rand()/RAND_MAX;
    w[i] = 0.1;
  }
}

extern "C" void bpnn_zero_weights(float **w, int m, int n) {
  int i, j;
  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}

extern "C" BPNN *bpnn_create(int n_in, int n_hidden, int n_out) {
  BPNN *newnet;
  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}

extern "C" void load(BPNN *net) {
  int nr = layer_size;

  // int nc = ???           // KERMA: was not initialized
  // int imgsize = nr * nc; // KERMA: never used

  float *units = net->input_units;

  int k = 1;
  for (int i = 0; i < nr; i++) {
    units[k] = (float)rand() / (float)RAND_MAX;
    k++;
  }
}

extern "C" void bpnn_train_cuda(BPNN *net, float *eo, float *eh) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

#ifdef GPU
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;

  dim3 grid(1, num_blocks);
  dim3 threads(16, 16);

  input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim =
      (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy
  // using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

  cudaMalloc((void **)&input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void **)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
#endif

#ifdef CPU
  printf("Performing CPU computation\n");
  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,
                    hid);
#endif

#ifdef GPU
  printf("Performing GPU computation\n");

  cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum, in, hid);

  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(partial_sum, hidden_partial_sum,
             num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);
#endif

#ifdef GPU

  cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_prev_weights_cuda,
             (in + 1) * (hid + 1) * sizeof(float));

  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);

  bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda, hid,
                                              input_cuda, in, input_hidden_cuda,
                                              input_prev_weights_cuda);

  cudaDeviceSynchronize();

  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);
#endif
}

void bpnn_free(BPNN *net) {
  int n1 = net->input_n;
  int n2 = net->hidden_n;

  free((char *)net->input_units);
  free((char *)net->hidden_units);
  free((char *)net->output_units);

  free((char *)net->hidden_delta);
  free((char *)net->output_delta);
  free((char *)net->target);

  for (int i = 0; i <= n1; i++) {
    free((char *)net->input_weights[i]);
    free((char *)net->input_prev_weights[i]);
  }
  free((char *)net->input_weights);
  free((char *)net->input_prev_weights);

  for (int i = 0; i <= n2; i++) {
    free((char *)net->hidden_weights[i]);
    free((char *)net->hidden_prev_weights[i]);
  }

  free((char *)net->hidden_weights);
  free((char *)net->hidden_prev_weights);

  free(net);
}

extern "C" void backprop_face() {
  BPNN *net;
  // int i;  // KERMA: never used
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  printf("Input layer size : %d\n", layer_size);
  load(net);
  // entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(int argc, char *argv[]) {
  int seed;

  if (argc != 2) {
    fprintf(stderr, "usage: backprop <num of input elements>\n");
    exit(0);
  }

  layer_size = atoi(argv[1]);

  if (layer_size % 16 != 0) {
    fprintf(stderr, "The number of input points must be divided by 16\n");
    exit(0);
  }

  seed = 7;
  bpnn_initialize(seed);
  backprop_face();

  return 0;
}

double gettime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) { return setup(argc, argv); }
