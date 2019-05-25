#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NXPROB 640                       /* x dimension of problem grid */
#define NYPROB 512                       /* y dimension of problem grid */
#define STEPS 100000                    /* number of time steps */
#define CX 0.1                          /* Old struct parms */
#define CY 0.1
#define DEBUG  0                        /* Some extra messages  1: On, 0: Off */
#define BLOCK_SIZE_X 8                  /* Block size (x-dimension) */
#define BLOCK_SIZE_Y 8                  /* Block size (y-dimension)  */


#define SIZE (NXPROB*NYPROB)

#define CUDA_SAFE_CALL(call) {                                    \
    cudaError err = call;                                                    \
    if( err != cudaSuccess) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);} }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)


/* Useful GPU */
void detailsGPU () {
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
}

__global__ void print (float *u, int iz) {
    for (int i =0; i<NXPROB; i++) {
        for (int j=0; j<NYPROB; j++)
            printf("%6.2f ", u[iz*SIZE + i*NYPROB + j]);
        printf("\n");
    }
}

/* Array initialization */
__global__ void inidat(float  *u, int iz) {
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix>=0 && ix<NXPROB && iy>=0 && iy<NYPROB)
        u[iz*SIZE + ix*NYPROB + iy] =  (float)(ix * (NXPROB - ix - 1) * iy * (NYPROB - iy - 1));
}

__global__ void update (float * __restrict__ u, int iz) {
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;	
	if (ix>0 && ix<NXPROB-1 && iy>0 && iy<NYPROB-1)
        u[(1-iz)*SIZE + ix*NYPROB+iy] = u[iz*SIZE + ix*NYPROB + iy]  + 
            CX * (u[iz*SIZE + (ix+1)*NYPROB + iy] + u[iz*SIZE + (ix-1)*NYPROB + iy] - 2.0 * u[iz*SIZE + ix*NYPROB + iy]) +
            CY * (u[iz*SIZE + ix*NYPROB + iy+1] + u[iz*SIZE + ix*NYPROB + iy-1] - 2.0 * u[iz*SIZE + ix*NYPROB + iy]);
}

int main(void) {
    int k, iz;
    float *u, t;
    cudaEvent_t start, stop;
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid (FRACTION_CEILING(NXPROB, BLOCK_SIZE_X), FRACTION_CEILING(NYPROB, BLOCK_SIZE_Y));
    #if DEBUG
        detailsGPU ();
    #endif
    printf("Problem size: %dx%d\nAmount of iterations: %d\n", NXPROB, NYPROB, STEPS);
    CUDA_SAFE_CALL(cudaMalloc((void**)&u,  2 * NXPROB * NYPROB * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(u, 0,  2 * NXPROB * NYPROB * sizeof(float)));
    inidat<<<dimGrid, dimBlock>>>(u,0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    iz = 0;
    for (k=0; k<STEPS; k++) {
        update<<<dimGrid, dimBlock>>>(u, iz);
        cudaDeviceSynchronize();
        iz = 1-iz;
    }     
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    printf("Elapsed time: %e sec\n", t/1000);
    CUDA_SAFE_CALL(cudaFree(u));
    return 0;
}