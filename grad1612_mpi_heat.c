#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define NXPROB 10                      /* x dimension of problem grid */
#define NYPROB 10                      /* y dimension of problem grid */
#define STEPS 1                        /* number of time steps */
#define MAXWORKER 12                   /* maximum number of worker tasks */
#define MINWORKER 1                    /* minimum number of worker tasks */
#define MASTER 0                       /* taskid of first process */

#define REORGANISATION 1
#define GRIDX 2
#define GRIDY 2

#define CONVERGENCE 0                  /* 1: On, 0: Off */
#define INTERVAL 200                   /* After how many rounds are we checking for convergence */
#define SENSITIVITY 0.1                /* Convergence's sensitivity (EPSILON) */

#define CX 0.1                         /* Old struct parms */
#define CY 0.1

/**************************************************************************
* Prototypes
****************************************************************************/
void update(int, int, int, float **, float **);
void prtdat(int, int, float **, char *);
/****************************************************************************/

enum coordinates {SOUTH = 0, EAST, NORTH, WEST};

int main(void) {
   
   float **u[2];
   int i, j, k, iz, *xs, *ys, comm_sz, my_rank, neighBor[4], dims[2], periods[2];
   MPI_Comm comm2d;
   MPI_Datatype column, row;
   MPI_Status status[4];
   MPI_Request recvRequest[4], sendRequest[4];
   /* Variables for clock */
   double start_time, end_time, elapsed_time;

   /* First, find out my rank and how many tasks are running */
   MPI_Init(NULL, NULL);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Create 2D cartesian grid */
   periods[0] = periods[1] = 0;
   dims[0] = GRIDX;
   dims[1] = GRIDY;
   MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, REORGANISATION, &comm2d);

   /* Find Left/West and Right/East neighBors */
   MPI_Cart_shift(comm2d, 0, 1, &neighBor[WEST], &neighBor[EAST]);
   /* Find Bottom/South and Upper/North neighBors */
   MPI_Cart_shift(comm2d, 1, 1, &neighBor[NORTH], &neighBor[SOUTH]);

   /* Size of each cell */
   int xcell = NXPROB / GRIDX;
   int ycell = NYPROB / GRIDY;

   /* Size with extra rows and columns */
   int size_total_x = NXPROB + 2*GRIDX;
   int size_total_y = NYPROB + 2*GRIDY;

   /* Allocate 2D contiguous arrays u[0] and u[1] (3d u) */
   /* Allocate size_total_x rows */
   u[0] = malloc(size_total_x * sizeof(*u[0]));
   u[1] = malloc(size_total_x * sizeof(*u[1]));
   /* Allocate u[0][0] and u[1][0] for contiguous arrays */
   u[0][0] = malloc(size_total_x * size_total_y * sizeof(**u[0]));
   u[1][0] = malloc(size_total_x * size_total_y * sizeof(**u[1]));
   /* Loop on rows */
   for (i = 1; i < size_total_x; i++) {
      /* Increment size_total_x block on u[0][i] and u[1][i] address */
      u[0][i] = u[0][0] + i * size_total_y;
      u[1][i] = u[1][0] + i * size_total_y;
   }

   /* Allocate coordinates of processes */
   xs = malloc(comm_sz * sizeof(int));
   ys = malloc(comm_sz * sizeof(int));

   /* Create column data type to communicate with East and West neighBors */
   MPI_Type_vector(xcell, 1, size_total_y, MPI_FLOAT, &column);
   MPI_Type_commit(&column);
   /* Create row data type to communicate with North and South neighBors */
   MPI_Type_contiguous(ycell, MPI_FLOAT, &row);
   MPI_Type_commit(&row);

   if (my_rank == MASTER) {
      if ((comm_sz > MAXWORKER || comm_sz < MINWORKER) && comm_sz != GRIDX * GRIDY) {
         printf("ERROR: the number of tasks must be between %d and %d.\n Quiting...\n", MINWORKER + 1, MAXWORKER + 1);
         MPI_Abort(comm2d, 1);
         exit(1);
      }
      else {
         printf("Starting with %d processes\nProblem size:%dx%d\nEach process will take: %dx%d\n", comm_sz, NXPROB, NYPROB, xcell, ycell);
      }

      /* Compute the coordinates of the top left cell of the array that takes each worker */
      for (i = 0; i < GRIDX; i++) ys[i] = 1;
      for (i = 1; i < GRIDY; i++)
         for (j = 0; j < GRIDX; j++)
            ys[i * GRIDX + j] = ys[(i - 1) * GRIDX + j] + ycell + 2;
      for (i = 0; i < GRIDY; i++) xs[i * GRIDX] = 1;
      for (i = 1; i <= GRIDY; i++)
         for (j = 1; j < GRIDX; j++)
            xs[(i - 1) * GRIDX + j] = xs[(i - 1) * GRIDX + (j - 1)] + xcell + 2;
   }

   MPI_Bcast(xs, comm_sz, MPI_INT, 0, comm2d);
   MPI_Bcast(ys, comm_sz, MPI_INT, 0, comm2d);

   if (my_rank == MASTER) {
      printf("Initializing grid and writing initial.dat file...\n");
      for (i = 0; i < size_total_x; i++) {
         for (j = 0; j < size_total_y; j++) {
            u[0][i][j] = i * (size_total_x - i - 1) * j * (size_total_y - j - 1);
            u[1][i][j] = 0;
         }
      }
      prtdat(size_total_x, size_total_y, u[0], "./initial.dat");
   }
   else {
      for (i = 0; i < size_total_x; i++) {
         for (j = 0; j < size_total_y; j++) {
            if (i >= xs[my_rank] && i <= xs[my_rank]+xcell-1 && j >= ys[my_rank] && j <= ys[my_rank]+ycell-1)
               u[0][i][j] = i * (size_total_x - i - 1) * j * (size_total_y - j - 1);
            else
               u[0][i][j] = 0.0;
            u[1][i][j] = 0.0;
         }
      }
      char str[10];
      sprintf(str, "%d.txt", my_rank);
      prtdat(size_total_x, size_total_y, u[0], str);
   }
   
   MPI_Barrier(comm2d);
   start_time = MPI_Wtime();
   
   iz = 0;
   for (k = 0; k < STEPS; k++) {
      
      /* Receive */
      MPI_Irecv(&u[iz][xs[my_rank]-1][ys[my_rank]], 1, row, neighBor[NORTH], 1, comm2d, &recvRequest[NORTH]); // receive a row from north
      MPI_Irecv(&u[iz][xs[my_rank]+xcell][ys[my_rank]], 1, row, neighBor[SOUTH], 2, comm2d, &recvRequest[SOUTH]); //receive a row from south
      MPI_Irecv(&u[iz][xs[my_rank]][ys[my_rank]-1], 1, column, neighBor[WEST], 3, comm2d, &recvRequest[WEST]); //receive a column from west
      MPI_Irecv(&u[iz][xs[my_rank]][ys[my_rank]+ycell], 1, column, neighBor[EAST], 4, comm2d, &recvRequest[EAST]); // receive a column from east
      
      /* Send */
      MPI_Isend(&u[iz][xs[my_rank]+xcell-1][ys[my_rank]], 1, row, neighBor[SOUTH], 1, comm2d, &sendRequest[SOUTH]); //send a row to south
      MPI_Isend(&u[iz][xs[my_rank]][ys[my_rank]], 1, row, neighBor[NORTH], 2, comm2d, &sendRequest[NORTH]); // send a row to north
      MPI_Isend(&u[iz][xs[my_rank]][ys[my_rank]+ycell-1], 1, column, neighBor[EAST], 3, comm2d, &sendRequest[EAST]); // send a column to east
      MPI_Isend(&u[iz][xs[my_rank]][ys[my_rank]], 1, column, neighBor[WEST], 4, comm2d, &sendRequest[WEST]); // send a column to west
     
      /* Update inner elements */
      for (i = xs[my_rank]+1; i < xs[my_rank]+xcell-1; i++)
         for (j = ys[my_rank]+1; j < ys[my_rank]+ycell-1; j++)
            u[1-iz][i][j] = u[iz][i][j] + CX*(u[iz][i+1][j] + u[iz][i-1][j] - 2.0*u[iz][i][j]) + CY*(u[iz][i][j+1] + u[iz][i][j-1] - 2.0*u[iz][i][j]);

      MPI_Waitall(4, recvRequest, status); // wait to receive everything
      
      /* Update boundary elements */

      /* First and last row update */
      for (j=ys[my_rank]; j<ys[my_rank]+ycell; j++) {
         u[1-iz][xs[my_rank]][j] = u[iz][xs[my_rank]][j] + CX*(u[iz][xs[my_rank]+1][j] + u[iz][xs[my_rank]-1][j] - 2.0*u[iz][xs[my_rank]][j]) + CY*(u[iz][xs[my_rank]][j+1] + u[iz][xs[my_rank]][j-1] - 2.0*u[iz][xs[my_rank]][j]);
         u[1-iz][xs[my_rank]+xcell-1][j] = u[iz][xs[my_rank]+xcell-1][j] + CX*(u[iz][xs[my_rank]+xcell][j] + u[iz][xs[my_rank]+xcell-2][j] - 2.0*u[iz][xs[my_rank]+xcell-1][j]) + CY*(u[iz][xs[my_rank]+xcell-1][j+1] + u[iz][xs[my_rank]+xcell-1][j-1] - 2.0*u[iz][xs[my_rank]+xcell-1][j]);
      }
      
      /* First and last column update */
      for (j=xs[my_rank]+1; j<xs[my_rank]+xcell-1; j++) {
         u[1-iz][j][ys[my_rank]] = u[iz][j][ys[my_rank]] + CX*(u[iz][j+1][ys[my_rank]] + u[iz][j-1][ys[my_rank]] - 2.0*u[iz][j][ys[my_rank]]) + CY*(u[iz][j][ys[my_rank]+1] + u[iz][j][ys[my_rank]-1] - 2.0*u[iz][j][ys[my_rank]]);
         u[1-iz][j][ys[my_rank]+ycell-1] = u[iz][j][ys[my_rank]+ycell-1] + CX*(u[iz][j+1][ys[my_rank]+ycell-1] + u[iz][j-1][ys[my_rank]+ycell-1] - 2.0*u[iz][j][ys[my_rank]+ycell-1]) + CY*(u[iz][j][ys[my_rank]+ycell] + u[iz][j][ys[my_rank]+ycell-2] - 2.0*u[iz][j][ys[my_rank]+ycell-1]);
      }
      
      #if CONVERGENCE
         /* Reduction */
      #endif
      
      iz = 1-iz; // swap arrays
      MPI_Waitall(4, sendRequest, status); //wait to send everything

      char str[10];
      sprintf(str, "After%d.txt", my_rank);
      prtdat(size_total_x, size_total_y, u[iz], str);
   }

   end_time = MPI_Wtime();
   elapsed_time = end_time - start_time;
   if (my_rank == MASTER) printf("Time: %.6f sec\n", elapsed_time);

   /* Free all arrays */
   free(xs);
   free(ys);
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
      for (int j = 1; j <= ny - 2; j++)
         unew[i][j] = uold[i][j] + CX*(uold[i+1][j] + uold[i-1][j] - 2.0*uold[i][j]) + CY*(uold[i][j+1] + uold[i][j-1] - 2.0*uold[i][j]);
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