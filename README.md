## Under construction!
This code implements a naive Krylov subspace expansion method for time-evolving
a wavefunction, and is parallelized using openMP.

## Usage
### Compile
Compile using 'make'
### Multithreading
Set the OMP_NUM_THREADS environment variable to the number of threads you want to use.
For example, on Linux/Mac: export OMP_NUM_THREADS=4
