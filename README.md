## Under construction!
This code implements a naive Krylov subspace expansion method for time-evolving
a wavefunction, and is parallelized using openMP and native Eigen implementation.

## Usage
### Compile
Compile using 'make'
### Multithreading
Set the OMP_NUM_THREADS environments variable to the number of threads you want to use.
For example, on Linux/Mac: export OMP_NUM_THREADS=4
