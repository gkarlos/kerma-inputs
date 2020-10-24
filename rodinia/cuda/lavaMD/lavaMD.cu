//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	UPDATE
//========================================================================================================================================================================================================200

//	14 APR 2011 Lukasz G. Szafaryn

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150
#include <stdbool.h> // (in path known to compiler)			needed by true/false
#include <stdio.h>   // (in path known to compiler)			needed by printf
#include <stdlib.h>  // (in path known to compiler)			needed by malloc
#include <time.h>
#include <sys/time.h>

#define fp double

#define NUMBER_PAR_PER_BOX 100							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

/* #define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance */
// Parameterized work group size
#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 128
#endif

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE

//===============================================================================================================================================================================================================200
//	STRUCTURES
//===============================================================================================================================================================================================================200
typedef struct
{
	fp x, y, z;
} THREE_VECTOR;

typedef struct
{
	fp v, x, y, z;
} FOUR_VECTOR;

typedef struct nei_str
{
	// neighbor box
	int x, y, z;
	int number;
	long offset;
} nei_str;

typedef struct box_str
{
	// home box
	int x, y, z;
	int number;
	long offset;
	// neighbor boxes
	int nn;
	nei_str nei[26];
} box_str;

typedef struct par_str
{
	fp alpha;
} par_str;

typedef struct dim_str
{
	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;
} dim_str;

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

int isInteger(char *str) {
  if (*str == '\0') {
    return 0;
  }

  for (; *str != '\0'; str++) {
    if (*str < 48 ||
        *str >
            57) { // digit characters (need to include . if checking for float)
      return 0;
    }
  }
  return 1;
}

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		fflush(NULL);
		exit(EXIT_FAILURE);
	}
}

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  //	THREAD PARAMETERS
  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  int bx = blockIdx.x;  // get current horizontal block index (0-n)
  int tx = threadIdx.x; // get current horizontal thread index (0-n)
  // int ax = bx*NUMBER_THREADS+tx;
  // int wbx = bx;
  int wtx = tx;

  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  //	DO FOR THE NUMBER OF BOXES
  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  if (bx < d_dim_gpu.number_boxes) {
    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //	Extract input parameters
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // parameters
    fp a2 = 2.0 * d_par_gpu.alpha * d_par_gpu.alpha;

    // home box
    int first_i;
    FOUR_VECTOR *rA;
    FOUR_VECTOR *fA;
    __shared__ FOUR_VECTOR rA_shared[100];

    // nei box
    int pointer;
    int k = 0;
    int first_j;
    FOUR_VECTOR *rB;
    fp *qB;
    int j = 0;
    __shared__ FOUR_VECTOR rB_shared[100];
    __shared__ double qB_shared[100];

    // common
    fp r2;
    fp u2;
    fp vij;
    fp fs;
    fp fxij;
    fp fyij;
    fp fzij;
    THREE_VECTOR d;

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //	Home box
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    //----------------------------------------------------------------------------------------------------------------------------------140
    //	Setup parameters
    //----------------------------------------------------------------------------------------------------------------------------------140

    // home box - box parameters
    first_i = d_box_gpu[bx].offset;

    // home box - distance, force, charge and type parameters
    rA = &d_rv_gpu[first_i];
    fA = &d_fv_gpu[first_i];

    //----------------------------------------------------------------------------------------------------------------------------------140
    //	Copy to shared memory
    //----------------------------------------------------------------------------------------------------------------------------------140

    // home box - shared memory
    while (wtx < NUMBER_PAR_PER_BOX) {
      rA_shared[wtx] = rA[wtx];
      wtx = wtx + NUMBER_THREADS;
    }
    wtx = tx;

    // synchronize threads  - not needed, but just to be safe
    __syncthreads();

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //	nei box loop
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // loop over neiing boxes of home box
    for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {

      //----------------------------------------50
      //	nei box - get pointer to the right box
      //----------------------------------------50

      if (k == 0) {
        pointer = bx; // set first box to be processed to home box
      } else {
        pointer =
            d_box_gpu[bx].nei[k - 1].number; // remaining boxes are nei boxes
      }

      //----------------------------------------------------------------------------------------------------------------------------------140
      //	Setup parameters
      //----------------------------------------------------------------------------------------------------------------------------------140

      // nei box - box parameters
      first_j = d_box_gpu[pointer].offset;

      // nei box - distance, (force), charge and (type) parameters
      rB = &d_rv_gpu[first_j];
      qB = &d_qv_gpu[first_j];

      //----------------------------------------------------------------------------------------------------------------------------------140
      //	Setup parameters
      //----------------------------------------------------------------------------------------------------------------------------------140

      // nei box - shared memory
      while (wtx < NUMBER_PAR_PER_BOX) {
        rB_shared[wtx] = rB[wtx];
        qB_shared[wtx] = qB[wtx];
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;

      // synchronize threads because in next section each thread accesses data
      // brought in by different threads here
      __syncthreads();

      //----------------------------------------------------------------------------------------------------------------------------------140
      //	Calculation
      //----------------------------------------------------------------------------------------------------------------------------------140

      // loop for the number of particles in the home box
      // for (int i=0; i<nTotal_i; i++){
      while (wtx < NUMBER_PAR_PER_BOX) {
        // loop for the number of particles in the current nei box
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
          // r2 = rA[wtx].v + rB[j].v - DOT(rA[wtx],rB[j]);
          // u2 = a2*r2;
          // vij= exp(-u2);
          // fs = 2.*vij;

          // d.x = rA[wtx].x  - rB[j].x;
          // fxij=fs*d.x;
          // d.y = rA[wtx].y  - rB[j].y;
          // fyij=fs*d.y;
          // d.z = rA[wtx].z  - rB[j].z;
          // fzij=fs*d.z;

          // fA[wtx].v +=  qB[j]*vij;
          // fA[wtx].x +=  qB[j]*fxij;
          // fA[wtx].y +=  qB[j]*fyij;
          // fA[wtx].z +=  qB[j]*fzij;
          r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v -
               DOT((fp)rA_shared[wtx], (fp)rB_shared[j]);
          u2 = a2 * r2;
          vij = exp(-u2);
          fs = 2 * vij;

          d.x = (fp)rA_shared[wtx].x - (fp)rB_shared[j].x;
          fxij = fs * d.x;
          d.y = (fp)rA_shared[wtx].y - (fp)rB_shared[j].y;
          fyij = fs * d.y;
          d.z = (fp)rA_shared[wtx].z - (fp)rB_shared[j].z;
          fzij = fs * d.z;

          fA[wtx].v += (double)((fp)qB_shared[j] * vij);
          fA[wtx].x += (double)((fp)qB_shared[j] * fxij);
          fA[wtx].y += (double)((fp)qB_shared[j] * fyij);
          fA[wtx].z += (double)((fp)qB_shared[j] * fzij);
        }

        // increment work thread index
        wtx = wtx + NUMBER_THREADS;
      }

      // reset work index
      wtx = tx;

      // synchronize after finishing force contributions from current nei box
      // not to cause conflicts when starting next box
      __syncthreads();

      //----------------------------------------------------------------------------------------------------------------------------------140
      //	Calculation END
      //----------------------------------------------------------------------------------------------------------------------------------140
    }
    // // increment work block index
    // wbx = wbx + NUMBER_BLOCKS;

    // // synchronize - because next iteration will overwrite current shared
    // memory
    // __syncthreads();

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //	nei box loop END
    //------------------------------------------------------------------------------------------------------------------------------------------------------160
  }
}

void kernel_gpu_cuda_wrapper(par_str par_cpu, dim_str dim_cpu, box_str *box_cpu,
                             FOUR_VECTOR *rv_cpu, fp *qv_cpu,
                             FOUR_VECTOR *fv_cpu) {
  // timer
  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	GPU SETUP
  //======================================================================================================================================================150

  //====================================================================================================100
  //	INITIAL DRIVER OVERHEAD
  //====================================================================================================100

  cudaDeviceSynchronize();

  //====================================================================================================100
  //	VARIABLES
  //====================================================================================================100

  box_str *d_box_gpu;
  FOUR_VECTOR *d_rv_gpu;
  fp *d_qv_gpu;
  FOUR_VECTOR *d_fv_gpu;

  dim3 threads;
  dim3 blocks;

  //====================================================================================================100
  //	EXECUTION PARAMETERS
  //====================================================================================================100

  blocks.x = dim_cpu.number_boxes;
  blocks.y = 1;
  threads.x = NUMBER_THREADS; // define the number of threads in the block
  threads.y = 1;

  time1 = get_time();

  //======================================================================================================================================================150
  //	GPU MEMORY				(MALLOC)
  //======================================================================================================================================================150

  //====================================================================================================100
  //	GPU MEMORY				(MALLOC) COPY IN
  //====================================================================================================100

  //==================================================50
  //	boxes
  //==================================================50
  cudaMalloc((void **)&d_box_gpu, dim_cpu.box_mem);

  //==================================================50
  //	rv
  //==================================================50
  cudaMalloc((void **)&d_rv_gpu, dim_cpu.space_mem);

  //==================================================50
  //	qv
  //==================================================50
  cudaMalloc((void **)&d_qv_gpu, dim_cpu.space_mem2);

  //====================================================================================================100
  //	GPU MEMORY				(MALLOC) COPY
  //====================================================================================================100

  //==================================================50
  //	fv
  //==================================================50
  cudaMalloc((void **)&d_fv_gpu, dim_cpu.space_mem);

  time2 = get_time();

  //==================================================50
  //	boxes
  //==================================================50
  cudaMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);

  //==================================================50
  //	rv
  //==================================================50
  cudaMemcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);

  //==================================================50
  //	qv
  //==================================================50
  cudaMemcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice);

  //==================================================50
  //	fv
  //==================================================50
  cudaMemcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);

  time3 = get_time();

  //======================================================================================================================================================150
  //	KERNEL
  //======================================================================================================================================================150

  // launch kernel - all boxes
  kernel_gpu_cuda<<<blocks, threads>>>(par_cpu, dim_cpu, d_box_gpu, d_rv_gpu,
                                       d_qv_gpu, d_fv_gpu);

  checkCUDAError("Start");
  cudaDeviceSynchronize();

  time4 = get_time();

  //======================================================================================================================================================150
  //	GPU MEMORY			COPY (CONTD.)
  //======================================================================================================================================================150
  cudaMemcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem, cudaMemcpyDeviceToHost);

  time5 = get_time();

  //======================================================================================================================================================150
  //	GPU MEMORY DEALLOCATION
  //======================================================================================================================================================150

  cudaFree(d_rv_gpu);
  cudaFree(d_qv_gpu);
  cudaFree(d_fv_gpu);
  cudaFree(d_box_gpu);

  time6 = get_time();

  //======================================================================================================================================================150
  //	DISPLAY TIMING
  //======================================================================================================================================================150

  printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

  printf("%15.12f s, %15.12f %% : GPU: SET DEVICE / DRIVER INIT\n",
         (float)(time1 - time0) / 1000000,
         (float)(time1 - time0) / (float)(time6 - time0) * 100);
  printf("%15.12f s, %15.12f %% : GPU MEM: ALO\n",
         (float)(time2 - time1) / 1000000,
         (float)(time2 - time1) / (float)(time6 - time0) * 100);
  printf("%15.12f s, %15.12f %% : GPU MEM: COPY IN\n",
         (float)(time3 - time2) / 1000000,
         (float)(time3 - time2) / (float)(time6 - time0) * 100);

  printf("%15.12f s, %15.12f %% : GPU: KERNEL\n",
         (float)(time4 - time3) / 1000000,
         (float)(time4 - time3) / (float)(time6 - time0) * 100);

  printf("%15.12f s, %15.12f %% : GPU MEM: COPY OUT\n",
         (float)(time5 - time4) / 1000000,
         (float)(time5 - time4) / (float)(time6 - time0) * 100);
  printf("%15.12f s, %15.12f %% : GPU MEM: FRE\n",
         (float)(time6 - time5) / 1000000,
         (float)(time6 - time5) / (float)(time6 - time0) * 100);

  printf("Total time:\n");
  printf("%.12f s\n", (float)(time6 - time0) / 1000000);
}

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int main(int argc, char *argv[]) {

  printf("thread block size of kernel = %d \n", NUMBER_THREADS);
  //======================================================================================================================================================150
  //	CPU/MCPU VARIABLES
  //======================================================================================================================================================150

  // timer
  // long long time0 =  get_time(); // KERMA: unused
  // timer
  // long long time1;
  // long long time2;
  // long long time3;
  // long long time4;
  // long long time5;
  // long long time6;
  // long long time7;

  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str *box_cpu;
  FOUR_VECTOR *rv_cpu;
  fp *qv_cpu;
  FOUR_VECTOR *fv_cpu;
  int nh;

  // time1 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	CHECK INPUT ARGUMENTS
  //======================================================================================================================================================150

  // assing default values
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  for (dim_cpu.cur_arg = 1; dim_cpu.cur_arg < argc; dim_cpu.cur_arg++) {
    // check if -boxes1d
    if (strcmp(argv[dim_cpu.cur_arg], "-boxes1d") == 0) {
      // check if value provided
      if (argc >= dim_cpu.cur_arg + 1) {
        // check if value is a number
        if (isInteger(argv[dim_cpu.cur_arg + 1]) == 1) {
          dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg + 1]);
          if (dim_cpu.boxes1d_arg < 0) {
            printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
            return 0;
          }
          dim_cpu.cur_arg = dim_cpu.cur_arg + 1;
        }
        // value is not a number
        else {
          printf("ERROR: Value to -boxes1d parameter in not a number\n");
          return 0;
        }
      }
      // value not provided
      else {
        printf("ERROR: Missing value to -boxes1d parameter\n");
        return 0;
      }
    }
    // unknown
    else {
      printf("ERROR: Unknown parameter\n");
      return 0;
    }
  }

  // Print configuration
  printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

  // time2 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	INPUTS
  //======================================================================================================================================================150

  par_cpu.alpha = 0.5;

  // time3 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	DIMENSIONS
  //======================================================================================================================================================150

  // total number of boxes
  dim_cpu.number_boxes =
      dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  // time4 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	SYSTEM MEMORY
  //======================================================================================================================================================150

  //====================================================================================================100
  //	BOX
  //====================================================================================================100

  // allocate boxes
  box_cpu = (box_str *)malloc(dim_cpu.box_mem);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for (i = 0; i < dim_cpu.boxes1d_arg; i++) {
    // home boxes in y direction
    for (j = 0; j < dim_cpu.boxes1d_arg; j++) {
      // home boxes in x direction
      for (k = 0; k < dim_cpu.boxes1d_arg; k++) {

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for (l = -1; l < 2; l++) {
          // neighbor boxes in y direction
          for (m = -1; m < 2; m++) {
            // neighbor boxes in x direction
            for (n = -1; n < 2; n++) {

              // check if (this neighbor exists) and (it is not the same as home
              // box)
              if ((((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0) == true &&
                   ((i + l) < dim_cpu.boxes1d_arg &&
                    (j + m) < dim_cpu.boxes1d_arg &&
                    (k + n) < dim_cpu.boxes1d_arg) == true) &&
                  (l == 0 && m == 0 && n == 0) == false) {

                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
                box_cpu[nh].nei[box_cpu[nh].nn].number =
                    (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg *
                     dim_cpu.boxes1d_arg) +
                    (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                    box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset =
                    box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;
              }

            } // neighbor boxes in x direction
          }   // neighbor boxes in y direction
        }     // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    }   // home boxes in y direction
  }     // home boxes in z direction

  //====================================================================================================100
  //	PARAMETERS, DISTANCE, CHARGE AND FORCE
  //====================================================================================================100

  // random generator seed set to random value - time in this case
  srand(time(NULL));

  // input (distances)
  rv_cpu = (FOUR_VECTOR *)malloc(dim_cpu.space_mem);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    rv_cpu[i].v =
        (rand() % 10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
    rv_cpu[i].x =
        (rand() % 10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
    rv_cpu[i].y =
        (rand() % 10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
    rv_cpu[i].z =
        (rand() % 10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
  }

  // input (charge)
  qv_cpu = (fp *)malloc(dim_cpu.space_mem2);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    qv_cpu[i] = (rand() % 10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
  }

  // output (forces)
  fv_cpu = (FOUR_VECTOR *)malloc(dim_cpu.space_mem);
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fv_cpu[i].v = 0; // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].x = 0; // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].y = 0; // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].z = 0; // set to 0, because kernels keeps adding to initial value
  }

  // time5 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	KERNEL
  //======================================================================================================================================================150

  //====================================================================================================100
  //	GPU_CUDA
  //====================================================================================================100

  kernel_gpu_cuda_wrapper(par_cpu, dim_cpu, box_cpu, rv_cpu, qv_cpu, fv_cpu);

  // time6 = get_time(); // KERMA: unused

  //======================================================================================================================================================150
  //	SYSTEM MEMORY DEALLOCATION
  //======================================================================================================================================================150

  // dump results
#ifdef OUTPUT
  FILE *fptr;
  fptr = fopen("result.txt", "w");
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y,
            fv_cpu[i].z);
  }
  fclose(fptr);
#endif

  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  // time7 = get_time(); // KERMA: unused
  return 0.0; // always returns 0.0
}
