#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define NXPROB          10                 /* x dimension of problem grid */
#define NYPROB          10                 /* y dimension of problem grid */
#define STEPS           1000               /* number of time steps */
#define MAXWORKER       12                 /* maximum number of worker tasks */
#define MINWORKER       1                  /* minimum number of worker tasks */
#define BEGIN           1                  /* message tag */
#define LTAG            2                  /* message tag */
#define RTAG            3                  /* message tag */
#define NONE            0                  /* indicates no neighbor */
#define DONE            4                  /* message tag */
#define MASTER          0                  /* taskid of first process */

#define REORGANISATION  1
#define GRIDX           2
#define GRIDY           2

#define SOUTH           0
#define EAST            1
#define NORTH           2
#define WEST            3

#define CONVERGENCE     0
#define INTERVAL	      200		            /* After how many rounds are we checking for convergence */
#define SENSITIVITY	    0.1		            /* Convergence's sensitivity (EPSILON) */

struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};

/**************************************************************************
* Prototypes
****************************************************************************/
void update(int, int, int, float *, float *);
void inidat(int, int, float **);
void prtdat(int, int, float **, char *);
void processToMap(int *, int *, int *, int *, int, int, int, int);
/****************************************************************************/

int main (void) {
    
   int comm_sz, my_rank, neighBor[4], dims[2], periods[2], rc, xcell, ycell, *xs, *ys, *xe, *ye, i, size_total_x, size_total_y;
   float  **u[2]; /* array for grid */
   MPI_Comm comm2d;
   /* Variables for clock */
   double start_time, end_time, elapsed_time;

    /* First, find out my taskid and how many tasks are running */
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    /* Create 2D cartesian grid */
    periods[0] = periods[1] = 0;
    /* Invert (Ox,Oy) classic convention */
    dims[0] = GRIDX;
    dims[1] = GRIDY;
    MPI_Cart_create(MPI_COMM_WORLD, 2 , dims, periods, REORGANISATION, &comm2d);
    
    /* Identify neighBors */
    neighBor[SOUTH] = neighBor[EAST] = neighBor[NORTH] = neighBor[WEST] = MPI_PROC_NULL;
    /* Find Left/West and Right/East neighBors */
    MPI_Cart_shift(comm2d, 0, 1, &neighBor[WEST], &neighBor[EAST]);
    /* Find Bottom/South and Upper/North neighBors */
    MPI_Cart_shift(comm2d, 1, 1, &neighBor[SOUTH], &neighBor[NORTH]);

    /* Size of each cell */
    xcell = NXPROB/GRIDX;
    ycell = NYPROB/GRIDX;

   /* Size with extra rows and columns */
    size_total_x = NXPROB + 2*GRIDX;
    size_total_y = NYPROB + 2*GRIDY;

   /* Allocate 2D contiguous arrays u[0] and u[1] (3d u) */
   /* Allocate size_total_x rows */
   u[0] = malloc(size_total_x*sizeof(*u[0]));
   u[1] = malloc(size_total_x*sizeof(*u[1]));
   /* Allocate u[0][0] and u[1][0] for contiguous arrays */
   u[0][0] = malloc(size_total_x*size_total_y*sizeof(**u[0]));
   u[1][0] = malloc(size_total_x*size_total_y*sizeof(**u[1]));
   /* Loop on rows */
   for (i=1;i<size_total_x;i++) {
      /* Increment size_total_x block on u[0][i] and u[1][i] address */
      u[0][i] = u[0][0] + i*size_total_y;
      u[1][i] = u[0][0] + i*size_total_y;
   }

    /* Allocate coordinates of processes */
    xs = malloc(comm_sz*sizeof(int));
    xe = malloc(comm_sz*sizeof(int));
    ys = malloc(comm_sz*sizeof(int));
    ye = malloc(comm_sz*sizeof(int));

    if (my_rank == MASTER) {
      if ((comm_sz > MAXWORKER || comm_sz < MINWORKER) && comm_sz != GRIDX*GRIDY) {
        printf("ERROR: the number of tasks must be between %d and %d.\n Quiting...\n", MINWORKER+1,MAXWORKER+1);
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
      }
      else {
          printf("Starting with %d processes\nProblem size:%dx%d\nEach process will take: %dx%d\n", comm_sz, NXPROB, NYPROB, xcell, ycell);
      }
      inidat(size_total_x, size_total_y, u[0]);
      prtdat(size_total_x, size_total_y, u[0], "initial.dat");
   
      processToMap(xs, ys, xe, ye, xcell, ycell, GRIDX, GRIDY);
      printf("xs: "); for (i=0; i<comm_sz; i++) printf("%d ", xs[i]); printf("\n");
      printf("ys: "); for (i=0; i<comm_sz; i++) printf("%d ", ys[i]); printf("\n");
      printf("xe: "); for (i=0; i<comm_sz; i++) printf("%d ", xe[i]); printf("\n");
      printf("ye: "); for (i=0; i<comm_sz; i++) printf("%d ", xe[i]); printf("\n");
    }
   /* Starting time */
   start_time = MPI_Wtime();




   /* Ending time */
   end_time = MPI_Wtime();
   /* Elapsed time */
   elapsed_time = end_time - start_time;


    /* Free all arrays */
    free(xs);
    free(ys);
    free(xe);
    free(ye);
    free(u[0][0]);
    free(u[1][0]);
    free(u[0]);
    free(u[1]);

    MPI_Finalize();
    return 0;
}

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int start, int end, int ny, float *u1, float *u2) {
   for (int ix = start; ix <= end; ix++) 
      for (int iy = 1; iy <= ny-2; iy++) 
         *(u2+ix*ny+iy) = *(u1+ix*ny+iy) + parms.cx * (*(u1+(ix+1)*ny+iy) + *(u1+(ix-1)*ny+iy) - 
                          2.0 * *(u1+ix*ny+iy)) + parms.cy * (*(u1+ix*ny+iy+1) + *(u1+ix*ny+iy-1) - 2.0 * *(u1+ix*ny+iy));
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float **u) {
   int ix, iy;

   for (ix = 0; ix < nx; ix++) 
      for (iy = 0; iy < ny; iy++)
         u[ix][iy] = ix * (nx - ix - 1) * iy * (ny - iy - 1);
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float **u, char *fnam) {
  FILE *fp = fopen(fnam, "w");
  
   for (int ix = 0; ix < nx; ix++) {
      for (int iy = 0; iy < ny; iy++) {
         fprintf(fp, "%6.1f ", u[ix][iy]);
      }
      fprintf(fp, "\n");
   }
   fclose(fp);
}


/************************************************************/
/*                                                          */
/* This subroutine computes the coordinates xs, xe, ys, ye, */
/* for each cell on the grid, respecting processes topology */
/*                                                          */
/************************************************************/

void processToMap(int *xs, int *ys, int *xe, int *ye, int xcell, int ycell, int x_domains, int y_domains) {

   /* Index variables */
   int i, j;

   /* Computation of starting ys,ye on (Ox) standard axis
      for the first column of global domain,
      Convention x(i,j) with i row and j column */
   for (i=0;i<x_domains;i++) {
      ys[i] = 1;
      /* Here, ye(0:(x_domains-1)) = 2+ycell-1 */
      ye[i] = ys[i]+ycell-1;
   }

   /* Computation of ys,ye on (Ox) standard axis
      for all other cells of global domain */
   for (i=1;i<y_domains;i++)
      for (j=0;j<x_domains;j++) {
         ys[i*x_domains+j] = ys[(i-1)*x_domains+j]+ycell+2;
         ye[i*x_domains+j] = ys[i*x_domains+j]+ycell-1;
      }

   /* Computation of starting xs,xe on (Oy) standard axis
      for the first row of global domain,
      Convention x(i,j) with i row and j column */
   for (i=0;i<y_domains;i++) {
      xs[i*x_domains] = 1;
      /* Here, xe(i*x_domains) = 2+xcell-1 */
      xe[i*x_domains] = xs[i*x_domains]+xcell-1;
   }

   /* Computation of xs,xe on (Oy) standard axis
      for all other cells of global domain */
   for (i=1;i<=y_domains;i++) {
      for (j=1;j<x_domains;j++) {
         xs[(i-1)*x_domains+j] = xs[(i-1)*x_domains+(j-1)]+xcell+2;
         xe[(i-1)*x_domains+j] = xs[(i-1)*x_domains+j]+xcell-1;
      }
   }
}