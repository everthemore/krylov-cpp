## Under construction!
This code implements a naive Krylov subspace expansion method for time-evolving
a wavefunction. It is a work in progress still :).

This code uses openMP to parallelize the construction of the Hamiltonian. The matrix multiplication
is probably not done optimally (i.e. in parallel) yet.
Set the OMP_NUM_THREADS environments variable to specify the number of threads!
