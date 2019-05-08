[![DOI](https://data.caltech.edu/badge/147138220.svg)](https://data.caltech.edu/badge/latestdoi/147138220)

## krylov-cpp
This code implements a Krylov-subspace expansion method for time-evolving
a wavefunction, and is parallelized using openMP. The current release uses re-orthogonalized Lanczos to build 
the orthonormal set of basis vectors from the Krylov-subspace. 

## Usage
### Compile
Compile using 'make'
### Multithreading
Set the OMP_NUM_THREADS environment variable to the number of threads you want to use.
For example, on Linux/Mac: export OMP_NUM_THREADS=4

## Disclaimer
Whereas this code was checked, benchmarked and tested, I can not guarantee that it is entirely bug-free. Please use it
with a healthy dose of skepticism.
