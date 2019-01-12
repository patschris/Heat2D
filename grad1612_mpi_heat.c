#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define NXPROB          10                 /* x dimension of problem grid */
#define NYPROB          10                 /* y dimension of problem grid */
#define STEPS           1                  /* number of time steps */
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

#define CONVERGENCE     0                    /* 1: On, 0: Off */
#define INTERVAL	      200		            /* After how many rounds are we checking for convergence */
#define SENSITIVITY	   0.1		            /* Convergence's sensitivity (EPSILON) */

#define CX              0.1                  /* Old struct parms */
#define CY              0.1

/**************************************************************************
* Prototypes
****************************************************************************/
void update(int, int, int, float **, float **);
void prtdat(int, int, float **, char *);
/****************************************************************************/

enum coordinates {SOUTH=0, EAST, NORTH, WEST};

int main (void) {
    
   int comm_sz, my_rank, neighBor[4], dims[2], periods[2], *xs, *ys, *xe, *ye, i, j;
   float  **u[2]; /* array for grid */
   MPI_Comm comm2d;
   MPI_Datatype column, row;
   MPI_Request recvRequest[4], sendRequest[4];
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
   //printf("I am %d and my neighbors are North=%d, South=%d, East =%d, West=%d\n", my_rank, neighBor[NORTH], neighBor[SOUTH], neighBor[EAST], neighBor[WEST]);

   /* Size of each cell */
   int xcell = NXPROB/GRIDX;
   int ycell = NYPROB/GRIDY;

   /* Size with extra rows and columns */
   int size_total_x = NXPROB + 2*GRIDX;
   int size_total_y = NYPROB + 2*GRIDY;

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
      u[1][i] = u[1][0] + i*size_total_y;
   }

    /* Allocate coordinates of processes */
    xs = malloc(comm_sz*sizeof(int));
    xe = malloc(comm_sz*sizeof(int));
    ys = malloc(comm_sz*sizeof(int));
    ye = malloc(comm_sz*sizeof(int));

   /* Create column data type to communicate with East and West neighBors */
   MPI_Type_vector(xcell, 1, size_total_y, MPI_FLOAT, &column);
   MPI_Type_commit(&column);
   /* Create row data type to communicate with North and South neighBors */
   MPI_Type_contiguous(ycell, MPI_FLOAT, &row);
   MPI_Type_commit(&row);

   if (my_rank == MASTER) {
      if ((comm_sz > MAXWORKER || comm_sz < MINWORKER) && comm_sz != GRIDX*GRIDY) {
        printf("ERROR: the number of tasks must be between %d and %d.\n Quiting...\n", MINWORKER+1,MAXWORKER+1);
        MPI_Abort(comm2d, 1);
        exit(1);
      }
      else {
         printf("Starting with %d processes\nProblem size:%dx%d\nEach process will take: %dx%d\n", comm_sz, NXPROB, NYPROB, xcell, ycell);
      }                                                          
      /* This subroutine computes the coordinates xs, xe, ys, ye,  for each cell on the grid, respecting processes topology */
      /* Computation of starting ys,ye on (Ox) standard axis for the first column of global domain */
      for (i=0;i<GRIDX;i++) {
         ys[i] = 1;
         ye[i] = ys[i]+ycell-1; /* Here, ye(0:(GRIDX-1)) = 2+ycell-1 */
      }
      /* Computation of ys,ye on (Ox) standard axis for all other cells of global domain */
      for (i=1;i<GRIDY;i++) {
         for (j=0;j<GRIDX;j++) {
            ys[i*GRIDX+j] = ys[(i-1)*GRIDX+j]+ycell+2;
            ye[i*GRIDX+j] = ys[i*GRIDX+j]+ycell-1;
         }
      }
      /* Computation of starting xs,xe on (Oy) standard axis for the first row of global domain, Convention x(i,j) with i row and j column */
      for (i=0;i<GRIDY;i++) {
         xs[i*GRIDX] = 1;
         xe[i*GRIDX] = xs[i*GRIDX]+xcell-1; /* Here, xe(i*GRIDX) = 2+xcell-1 */
      }
      /* Computation of xs,xe on (Oy) standard axis for all other cells of global domain */
      for (i=1;i<=GRIDY;i++) {
         for (j=1;j<GRIDX;j++) {
            xs[(i-1)*GRIDX+j] = xs[(i-1)*GRIDX+(j-1)]+xcell+2;
            xe[(i-1)*GRIDX+j] = xs[(i-1)*GRIDX+j]+xcell-1;
         }
      }
   }

   MPI_Bcast(xs, comm_sz, MPI_INT, 0, comm2d);
   MPI_Bcast(ys, comm_sz, MPI_INT, 0, comm2d);
   MPI_Bcast(xe, comm_sz, MPI_INT, 0, comm2d);
   MPI_Bcast(ye, comm_sz, MPI_INT, 0, comm2d);

   if (my_rank == MASTER) {
      printf("Initializing grid and writing initial.dat file...\n");
      for (i = 0; i < size_total_x; i++) { 
         for (j = 0; j < size_total_y; j++) {
            u[0][i][j] = i * (size_total_x - i - 1) * j * (size_total_y - j - 1);
            u[1][i][j] = 0;
         }
      }
      prtdat(size_total_x, size_total_y, u[0], "initial.dat");
   }
   else {
      for (i = 0; i < size_total_x; i++) {
         for (j = 0; j < size_total_y; j++) {
            if (i>=xs[my_rank] && i<=xe[my_rank]  && j >=ys[my_rank] && j<=ye[my_rank]) u[0][i][j] = i * (size_total_x - i - 1) * j * (size_total_y - j - 1);
            else u[0][i][j] = 0.0;
            u[1][i][j] = 0.0;
         }
      }
   }
   MPI_Barrier(comm2d);
   start_time = MPI_Wtime();


   end_time = MPI_Wtime();
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

   /* Free datatypes */
   MPI_Type_free(&column);
   MPI_Type_free(&row);

   MPI_Finalize();
   return 0;
}

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int start, int end, int ny, float **uold, float **unew) {
   for (int i = start; i <= end; i++) 
      for (int j = 1; j <= ny-2; j++)
         unew[i][j] = uold[i][j] + CX * (uold[i+1][j] + uold[i-1][j]  - 2.0 * uold[i][j]) + CY * (uold[i][j+1] + uold[i][j-1] - 2.0 * uold[i][j]);
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float **u, char *fnam) {
   FILE *fp = fopen(fnam, "w");
   for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
         fprintf(fp, "%6.1f ", u[i][j]);
      }
      fprintf(fp, "\n");
   }
   fclose(fp);
}