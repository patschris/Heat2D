# Heat 2D

## Problem description
<p align="justify">The purpose of the program is to simulate heat transfer to a surface. We have an 𝑀×𝑁 table, where each element of the table is a point in space and has a specific temperature. The initial temperature is high in the center and zero at the outer elements. The system changes state over time. This is because each point influences and is influenced by its neighbors. Neighboring elements are those that are above, below, to the right and left of an element. The outer elements don't change values as they are considered elements that absorb or emit heat in the system.</p>
<p align="justify">The program given as a model for the requirements of the exercise is available <a href="https://github.com/patschris/Heat2D/blob/master/mpi_heat2Dn.c">here</a> and the goal is to evaluate, redesign and improve it in order to better scale. We will compare the given program in time, in acceleration and efficiency with those that will design in MPI, in MPI + OpenMP (hybrid) and in Cuda.</p>
  
## Compilation

### MPI
mpicc -g -Wall -o grad1612_mpi_heat grad1612_mpi_heat.c
mpiexec -n 4 ./grad1612_mpi_heat

### MPI + OpenMP (Hybrid)
mpicc -g -Wall -fopenmp -o grad1612_hybrid_heat grad1612_hybrid_heat.c
mpiexec -n 4 ./grad1612_hybrid_heat
