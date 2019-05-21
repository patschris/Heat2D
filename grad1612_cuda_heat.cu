#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NXPROB 10                       /* x dimension of problem grid */
#define NYPROB 10                       /* y dimension of problem grid */
#define STEPS 100                       /* number of time steps */
#define CX 0.1                          /* Old struct parms */
#define CY 0.1
#define MAXTHREADS 32                   /* Threads per block gpu */
#define DEBUG  0                        /* Some extra messages  1: On, 0: Off */


 /******************* Prototypes *******************/
 __global__ void inidat(float *);
 __global__ void update(const float *, float *);
 __global__ void print (float *);
/**************************************************/


int main (void) {
    float *u0, *u1;
    cudaError_t err;

    if ((err = cudaMalloc((void**)&u0,  NXPROB * NYPROB * sizeof(float))) != cudaSuccess) {
        printf("cudaMalloc failed u0: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    if ((err = cudaMalloc((void**)&u1,  NXPROB * NYPROB * sizeof(float))) != cudaSuccess) {
        printf("cudaMalloc failed u0: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    #if DEBUG
        int devCount;
        const int kb = 1024;
        const int mb = kb * kb;
        cudaGetDeviceCount(&devCount);
        for (int i = 0; i < devCount; i++) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            printf("%s:   %d.%d\nGlobal memory:   %zd mb\n", props.name , props.major, props.minor, props.totalGlobalMem/mb);
            printf("Shared memory:   %zd kb\nConstant memory: %zd kb\nBlock registers: %d\n", props.sharedMemPerBlock/kb, props.totalConstMem/kb, props.regsPerBlock);
            printf("Warp size:         %d\nThreads per block: %d\n", props.warpSize, props.maxThreadsPerBlock);
            printf("Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
            printf("Max grid dimensions: [%d, %d, %d]\n\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        }
    #endif

    dim3 dimBlock((NXPROB+MAXTHREADS-1)/MAXTHREADS, (NXPROB+MAXTHREADS-1)/MAXTHREADS);
	dim3 dimGrid (MAXTHREADS , MAXTHREADS);
    inidat<<<dimBlock,dimGrid>>> (u0);
    print<<<1,1>>> (u0);

    if (u0 && (err = cudaFree(u0)) != cudaSuccess) 
        printf("cudaFree failed u0: %s\n", cudaGetErrorString(err));
    if (u1 && (err = cudaFree(u1)) != cudaSuccess) 
        printf("cudaFree failed u0: %s\n", cudaGetErrorString(err));
    if (cudaDeviceReset() != cudaSuccess) 
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(err));

    return 0;
}

/* Array initialization */
__global__ void inidat(float  *u) {
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix>=0 && ix<NXPROB && iy>=0 && iy<NYPROB)
        u[ix*NYPROB+iy] = (float)(ix * (NXPROB - ix - 1) * iy * (NYPROB - iy - 1));
}

__global__ void print (float *u) {
    for (int i =0; i<NXPROB; i++) {
        for (int j=0; j<NYPROB; j++)
            printf("%.2f ", u[i*NYPROB+j]);
        printf("\n");
    }
}


/* Array update */
__global__ void update(const float *init, float *dest){

	const int ix = blockIdx.x * blockDim.x + threadIdx.x ;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y ;
	
    if (ix>0 && ix<NXPROB-1 && iy>0 && iy<NYPROB-1) {
        dest[ix*NYPROB+iy] = 
            init[ix*NYPROB+iy]  + 
            CX* (init[(ix+1)*NYPROB+iy] + init[(ix-1)*NYPROB+iy] - 2.0 * init[ix*NYPROB+iy]) +
            CY * (init[ix*NYPROB+iy+1] + init[ix*NYPROB+iy-1] - 2.0 * init[ix*NYPROB+iy]);
    }
}
