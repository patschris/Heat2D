mpicc -g -Wall -o grad1612_mpi_heat grad1612_mpi_heat.c
mpiexec -n 9 ./grad1612_mpi_heat