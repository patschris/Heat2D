#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <omp.h>

#define NXPROB 10                      /* x dimension of problem grid */
#define NYPROB 10                      /* y dimension of problem grid */
#define STEPS 100                      /* number of time steps */
#define MASTER 0                       /* taskid of first process */

#define REORGANISATION 1               /* Reorganization of processes for cartesian grid (1: Enable, 0: Disable) */
#define GRIDX 2
#define GRIDY 2

#define CONVERGENCE 1                  /* 1: On, 0: Off */
#define INTERVAL 10                    /* After how many rounds are we checking for convergence */
#define SENSITIVITY 0.1                /* Convergence's sensitivity (EPSILON) */

#define CX 0.1                         /* Old struct parms */
#define CY 0.1

#define DEBUG  0                       /* Some extra messages  1: On, 0: Off */

#define NUMTHREADS   10

enum coordinates {SOUTH = 0, EAST, NORTH, WEST};

int main(void) {
   float **u[2];
   int i, j, k, iz, *xs, *ys, comm_sz, my_rank, neighBor[4], dims[2], periods[2];
   MPI_Comm comm2d;
   MPI_Datatype column, row;
   MPI_Status status[4];
   MPI_Request recvRequest[4], sendRequest[4];
   /* Variables for clock */
   double start_time, end_time, local_elapsed_time, elapsed_time;
   #if CONVERGENCE
      int tobreak = 0;
      float locdiff, totdiff;
   #endif

   /* First, find out my rank and how many tasks are running */
   MPI_Init(NULL, NULL);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Create 2D cartesian grid */
   periods[0] = periods[1] = 0;
   dims[0] = GRIDY;
   dims[1] = GRIDX;
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
   if ((u[0] = malloc(size_total_x * sizeof(*u[0]))) == NULL) {
      perror ("u[0] malloc failed");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
   }
   if ((u[1] = malloc(size_total_x * sizeof(*u[1]))) == NULL) {
      perror ("u[1] malloc failed");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
   }
   /* Allocate u[0][0] and u[1][0] for contiguous arrays */
   if ((u[0][0] = malloc(size_total_x * size_total_y * sizeof(**u[0]))) == NULL) {
      perror ("u[0][0] malloc failed");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
   }
   if ((u[1][0] = malloc(size_total_x * size_total_y * sizeof(**u[1]))) == NULL) {
      perror ("u[1][0] malloc failed");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
   }
   /* Loop on rows */
   #pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(i) shared(u, size_total_x, size_total_y)
   for (i = 1; i < size_total_x; i++) {
      /* Increment size_total_x block on u[0][i] and u[1][i] address */
      u[0][i] = u[0][0] + i * size_total_y;
      u[1][i] = u[1][0] + i * size_total_y;
   }

   /* Allocate coordinates of processes */
   if ((xs = malloc(comm_sz * sizeof(int))) == NULL) {
      perror ("xs malloc failed");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
   }
   if ((ys = malloc(comm_sz * sizeof(int))) == NULL) {
      perror ("ys malloc failed");
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
   }

   if (my_rank == MASTER) {
      if (comm_sz != GRIDX * GRIDY) {
         printf("ERROR: the number of tasks must be equal to %d.\nQuiting...\n", GRIDX*GRIDY);
         MPI_Abort(MPI_COMM_WORLD, 1);
         exit(1);
      }
      else if (NXPROB % GRIDX || NYPROB % GRIDY) {
         printf("ERROR: (%d/%d) or (%d/%d) is not an integer\nQuiting...\n", NXPROB, GRIDX, NYPROB, GRIDY);
         MPI_Abort(MPI_COMM_WORLD, 1);
         exit(1);
      }
      else {
         printf("Starting with %d processes\nProblem size:%dx%d\nEach process will take: %dx%d\n", comm_sz, NXPROB, NYPROB, xcell, ycell);
      }

      /* Compute the coordinates of the top left cell of the array that takes each worker */
      #pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(i) shared(ys)
      for (i = 0; i < GRIDX; i++) ys[i] = 1;

      #pragma omp parallel for num_threads(NUMTHREADS) collapse(2) schedule (static,1) default(none) private(i,j) shared(ys, ycell)
      for (i = 1; i < GRIDY; i++)
         for (j = 0; j < GRIDX; j++)
            ys[i * GRIDX + j] = ys[(i - 1) * GRIDX + j] + ycell + 2;

      #pragma omp parallel for num_threads(NUMTHREADS) schedule (static,1) default(none) private(i) shared(xs)
      for (i = 0; i < GRIDY; i++) xs[i * GRIDX] = 1;
      
      #pragma omp parallel for num_threads(NUMTHREADS) collapse(2) schedule (static,1) default(none) private(i,j) shared(xs, xcell) 
      for (i = 1; i <= GRIDY; i++)
         for (j = 1; j < GRIDX; j++)
            xs[(i - 1) * GRIDX + j] = xs[(i - 1) * GRIDX + (j - 1)] + xcell + 2;
   }

   /* Create row data type to communicate with North and South neighbors */
   MPI_Type_contiguous(ycell, MPI_FLOAT, &row);
   MPI_Type_commit(&row);
   /* Create column data type to communicate with East and West neighbors */
   MPI_Type_vector(xcell, 1, size_total_y, MPI_FLOAT, &column);
   MPI_Type_commit(&column);

   MPI_Bcast(xs, comm_sz, MPI_INT, 0, comm2d);
   MPI_Bcast(ys, comm_sz, MPI_INT, 0, comm2d);

   /* Each process initializes it's 2D subarray */
   #pragma omp parallel for num_threads(NUMTHREADS) collapse(2) schedule (static,1) default(none) private(i,j) shared(u, xs, ys, xcell, ycell, my_rank, size_total_x, size_total_y) 
   for (i = 0; i < size_total_x; i++) {
      for (j = 0; j < size_total_y; j++) {
         if (i >= xs[my_rank] && i < xs[my_rank]+xcell && j >= ys[my_rank] && j < ys[my_rank]+ycell)
            u[0][i][j] = i * (size_total_x - i - 1) * j * (size_total_y - j - 1);
         else
            u[0][i][j] = 0.0;
         u[1][i][j] = 0.0;
      }
   }

   #if DEBUG
      int len;
      char processor[MPI_MAX_PROCESSOR_NAME];
      MPI_Get_processor_name(processor, &len);
      printf("I am %d and my neighbors are North=%d, South=%d, East=%d, West=%d (Running on %s)\n", my_rank, neighBor[NORTH], neighBor[SOUTH], neighBor[EAST], neighBor[WEST], processor);
   #endif

   MPI_Barrier(comm2d);
   start_time = MPI_Wtime();

   iz = 0;
   
   /* Persistent communication */
   MPI_Send_init(&u[iz][xs[my_rank]+xcell-1][ys[my_rank]], 1, row, neighBor[SOUTH], 1, comm2d, &sendRequest[SOUTH]); //send a row to south
   MPI_Send_init(&u[iz][xs[my_rank]][ys[my_rank]], 1, row, neighBor[NORTH], 2, comm2d, &sendRequest[NORTH]); // send a row to north
   MPI_Send_init(&u[iz][xs[my_rank]][ys[my_rank]+ycell-1], 1, column, neighBor[EAST], 3, comm2d, &sendRequest[EAST]); // send a column to east
   MPI_Send_init(&u[iz][xs[my_rank]][ys[my_rank]], 1, column, neighBor[WEST], 4, comm2d, &sendRequest[WEST]); // send a column to west
   
   MPI_Recv_init(&u[iz][xs[my_rank]-1][ys[my_rank]], 1, row, neighBor[NORTH], 1, comm2d, &recvRequest[NORTH]); // receive a row from north
   MPI_Recv_init(&u[iz][xs[my_rank]+xcell][ys[my_rank]], 1, row, neighBor[SOUTH], 2, comm2d, &recvRequest[SOUTH]); //receive a row from south
   MPI_Recv_init(&u[iz][xs[my_rank]][ys[my_rank]-1], 1, column, neighBor[WEST], 3, comm2d, &recvRequest[WEST]); //receive a column from west
   MPI_Recv_init(&u[iz][xs[my_rank]][ys[my_rank]+ycell], 1, column, neighBor[EAST], 4, comm2d, &recvRequest[EAST]); // receive a column from east

   #pragma omp parallel num_threads(NUMTHREADS) private(k)
   for (k = 0; k < STEPS; k++) {
      #pragma omp master
      {
         /* Receive */
         MPI_Startall(4, sendRequest);
         /* Send */
         MPI_Startall(4, recvRequest);
      }
            
      /* Update inner elements */
      #pragma omp for schedule(static,1) collapse(2)
      for (i = xs[my_rank]+1; i < xs[my_rank]+xcell-1; i++)
         for (j = ys[my_rank]+1; j < ys[my_rank]+ycell-1; j++)
            u[1-iz][i][j] = u[iz][i][j] + CX*(u[iz][i+1][j] + u[iz][i-1][j] - 2.0*u[iz][i][j]) + CY*(u[iz][i][j+1] + u[iz][i][j-1] - 2.0*u[iz][i][j]);
      
      
      #pragma omp master
      MPI_Waitall(4, recvRequest, status); // wait to receive everything
      

      #pragma omp barrier                 // wait the master who waits

      /* Update boundary elements */

      /* First and last row update */
      #pragma omp for schedule(static,1)
      for (j=ys[my_rank]; j<ys[my_rank]+ycell; j++) {
         u[1-iz][xs[my_rank]][j] = u[iz][xs[my_rank]][j] + CX*(u[iz][xs[my_rank]+1][j] + u[iz][xs[my_rank]-1][j] - 2.0*u[iz][xs[my_rank]][j]) + CY*(u[iz][xs[my_rank]][j+1] + u[iz][xs[my_rank]][j-1] - 2.0*u[iz][xs[my_rank]][j]);
         u[1-iz][xs[my_rank]+xcell-1][j] = u[iz][xs[my_rank]+xcell-1][j] + CX*(u[iz][xs[my_rank]+xcell][j] + u[iz][xs[my_rank]+xcell-2][j] - 2.0*u[iz][xs[my_rank]+xcell-1][j]) + CY*(u[iz][xs[my_rank]+xcell-1][j+1] + u[iz][xs[my_rank]+xcell-1][j-1] - 2.0*u[iz][xs[my_rank]+xcell-1][j]);
      }
      
      /* First and last column update */
      #pragma omp for schedule(static,1)
      for (j=xs[my_rank]+1; j<xs[my_rank]+xcell-1; j++) {
         u[1-iz][j][ys[my_rank]] = u[iz][j][ys[my_rank]] + CX*(u[iz][j+1][ys[my_rank]] + u[iz][j-1][ys[my_rank]] - 2.0*u[iz][j][ys[my_rank]]) + CY*(u[iz][j][ys[my_rank]+1] + u[iz][j][ys[my_rank]-1] - 2.0*u[iz][j][ys[my_rank]]);
         u[1-iz][j][ys[my_rank]+ycell-1] = u[iz][j][ys[my_rank]+ycell-1] + CX*(u[iz][j+1][ys[my_rank]+ycell-1] + u[iz][j-1][ys[my_rank]+ycell-1] - 2.0*u[iz][j][ys[my_rank]+ycell-1]) + CY*(u[iz][j][ys[my_rank]+ycell] + u[iz][j][ys[my_rank]+ycell-2] - 2.0*u[iz][j][ys[my_rank]+ycell-1]);
      }
      
      /* Convergence check every INTERVAL iterations */
      #if CONVERGENCE
         if ((i+1) % INTERVAL == 0) {
            locdiff = 0.0;
            #pragma omp for schedule(static,1) reduction(+:locdiff)
            for (i = xs[my_rank]; i < xs[my_rank]+xcell; i++)
               for (j = ys[my_rank]; j < ys[my_rank]+ycell; j++)
                  locdiff += (u[iz][i][j] - u[1-iz][i][j])*(u[iz][i][j] - u[1-iz][i][j]); // square distance
                        
            #pragma omp master
            {
               MPI_Allreduce(&locdiff, &totdiff, 1, MPI_FLOAT, MPI_SUM, comm2d);
               if (totdiff < SENSITIVITY) tobreak=1;
            }
            if (tobreak) break;
         }
      #endif
      #pragma omp master
      {
         iz = 1-iz; // swap arrays
         MPI_Waitall(4, sendRequest, status); //wait to send everything
      }
   }
   end_time = MPI_Wtime();
   
   local_elapsed_time = end_time - start_time;
   MPI_Reduce(&local_elapsed_time, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, comm2d);
   if (my_rank == MASTER) printf("Elapsed time: %e sec\n", elapsed_time);

   /* Free all arrays */
   free(xs);
   free(ys);
   free(u[0][0]);
   free(u[1][0]);
   free(u[0]);
   free(u[1]);

   /* Free datatypes */
   MPI_Type_free(&row);
   MPI_Type_free(&column);

   MPI_Finalize();
   return 0;
}