mpicc -g -Wall -o grad1612_mpi_heat grad1612_mpi_heat.c  
mpiexec -n 4 ./grad1612_mpi_heat

mpicc -g -Wall -fopenmp -o grad1612_hybrid_heat grad1612_hybrid_heat.c
mpiexec -n 4 ./grad1612_hybrid_heat
