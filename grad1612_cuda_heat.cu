#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NXPROB 10                       /* x dimension of problem grid */
#define NYPROB 10                       /* y dimension of problem grid */
#define STEPS 1//00                       /* number of time steps */
#define CX 0.1                          /* Old struct parms */
#define CY 0.1
#define MAXTHREADS 32                   /* Threads per block gpu */
#define DEBUG  0                       /* Some extra messages  1: On, 0: Off */


/* Array initialization */
__global__ void inidat(float  *u, int iz, int pitch) {
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix>=0 && ix<NXPROB && iy>=0 && iy<NYPROB)
        u[iz*pitch + ix*NYPROB+iy] = /*(ix*NYPROB+iy)*(iz+1);*/ (float)(ix * (NXPROB - ix - 1) * iy * (NYPROB - iy - 1));
}

__global__ void print (float *u, int iz, int pitch) {
    for (int i =0; i<NXPROB; i++) {
        for (int j=0; j<NYPROB; j++)
            printf("%.2f ", u[iz*pitch + i*NYPROB+j]);
        printf("\n");
    }
}


/* Array update */
__global__ void update(float *u, int iz, int pitch){

	const int ix = blockIdx.x * blockDim.x + threadIdx.x ;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y ;

    if (ix>0 && ix<NXPROB-1 && iy>0 && iy<NYPROB-1)
        u[(1-iz)*pitch + ix*NYPROB + iy] = u[iz*pitch + ix*NYPROB + iy]  + 
                CX* (u[iz*pitch + (ix+1)*NYPROB + iy] + u[iz*pitch + (ix-1)*NYPROB + iy] - 2.0 * u[iz*pitch + ix*NYPROB + iy]) + 
                CY * (u[iz*pitch + ix*NYPROB + iy+1] + u[iz*pitch + ix*NYPROB + iy-1] - 2.0 * u[iz*pitch + ix*NYPROB + iy]);
    
}


int main (void) {
    float *u;
    int iz, k;
    size_t pitch;
    cudaError_t err;

   /* http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html */
    if ((err = cudaMallocPitch((void**)&u, &pitch, NXPROB * NYPROB * sizeof(float), 2))!= cudaSuccess) {
        printf("cudaMallocPitch failed u: %s\n", cudaGetErrorString(err));
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
    inidat<<<dimBlock,dimGrid>>> (u, 0, pitch);
    // inidat<<<dimBlock,dimGrid>>> (u, 1, pitch);
    printf("Initial\nu0\n");
    print<<<1,1>>>(u,0,pitch);
    cudaDeviceSynchronize();
    printf("\n\nu1\n");
    print<<<1,1>>>(u,1,pitch); 
    cudaDeviceSynchronize();
    printf("----------------------------------------------------------------\n");
    iz = 0;
    for (k = 0; k < STEPS; k++) {
        update<<<dimBlock,dimGrid>>> (u, iz, pitch);
        cudaDeviceSynchronize();
        iz = 1-iz;
    }
    printf("Final\nu0\n");
    print<<<1,1>>>(u,0,pitch);
    cudaDeviceSynchronize();
    printf("\n\nu1\n");
    print<<<1,1>>>(u,1,pitch); 
    cudaDeviceSynchronize();

    if (u && (err = cudaFree(u)) != cudaSuccess) 
        printf("cudaFree failed u: %s\n", cudaGetErrorString(err));
    if (cudaDeviceReset() != cudaSuccess) 
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(err));

    return 0;
}