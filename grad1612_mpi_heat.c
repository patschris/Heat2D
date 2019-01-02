#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define NXPROB      20                 /* x dimension of problem grid */
#define NYPROB      20                 /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */
#define SOUTH       0
#define EAST        1
#define NORTH       2
#define WEST        3

struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};

int main (void) {

    int comm_sz,    /* number of processes */
        my_rank,    /* my process unique id */
        neighBor[4],
        dims[2],
        periods[2];
    MPI_Comm comm2d;
    /* First, find out my taskid and how many tasks are running */
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    if (!my_rank) printf ("%d processes are running\n", comm_sz);
    printf("Hello, i am %d process\n", my_rank);
    /* Create 2D cartesian grid */
    periods[0] = 0;
    periods[1] = 0;
    /* Invert (Ox,Oy) classic convention */
    dims[0] = dims[1] = sqrt(comm_sz);
    MPI_Cart_create(MPI_COMM_WORLD, 2 /*ndims*/, dims, periods, 1/*reorganisation*/, &comm2d);
    /* Identify neighBors */
    neighBor[SOUTH] = MPI_PROC_NULL;
    neighBor[EAST] = MPI_PROC_NULL;
    neighBor[NORTH] = MPI_PROC_NULL;
    neighBor[WEST] = MPI_PROC_NULL;
  
    /* Left/West and Right/East neighBors */
    MPI_Cart_shift(comm2d, 0, 1, &neighBor[WEST], &neighBor[EAST]);

    /* Bottom/South and Upper/North neighBors */
    MPI_Cart_shift(comm2d, 1, 1, &neighBor[SOUTH], &neighBor[NORTH]);
    printf("I am %d and my neighbors are North=%d, South=%d, East =%d, West=%d\n", my_rank, neighBor[NORTH], neighBor[SOUTH], neighBor[EAST], neighBor[WEST]);
    

    MPI_Finalize();
    return 0;
}