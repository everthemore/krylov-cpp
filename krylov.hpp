#ifndef __KRYLOV_HPP__
#define __KRYLOV_HPP__

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unsupported/Eigen/MatrixFunctions>

/**
  For convenience, we introduce some new types
*/
using basis = std::unordered_map<std::string, int>;
using inversebasis = std::unordered_map<int, std::string>;
using observable = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
using state = Eigen::VectorXcd;
using complex = std::complex<double>;
typedef Eigen::Triplet<complex> T;

/**
  Converts an integer to a binary string representation, and
  pads it until it has length L. This was done initially with
  bitsets, but they can't be initialized with dynamically known
  lengths.
*/
std::string int_to_bin(int i, int L) {
  std::string binary_representation = "";

  while (i > 0) {
    binary_representation = std::to_string(i % 2) + binary_representation;
    i /= 2;
  }

  while (binary_representation.length() < L)
    binary_representation = "0" + binary_representation;

  return binary_representation;
}

/**
  Count the number of non-zero entries in the bitstring s
*/
int count_ones(std::string bitstring) {
  int count = 0;
  for (int i = 0; i < bitstring.length(); ++i)
    if (bitstring[i] == '1') count++;
  return count;
}

/**
  Builds the allowed basis states for L sites and k particles
*/
basis build_basis(const int L, int k, inversebasis &inversebasis) {
  int counter = 0;
  basis basis;

  for (int i = 0; i < pow(2, L); ++i) {
    std::string binrep = int_to_bin(i, L);
    if (count_ones(binrep) == k) {
      inversebasis.insert({counter, binrep});
      basis.insert({binrep, counter++});
    }
  }

  return basis;
}

/**
  Constructs the Hamiltonian matrix in sparse format
*/
observable build_hamiltonian(basis basis, inversebasis inversebasis, complex J,
                             complex F, complex U, complex W, int seed) {
  int num_basis = (int)basis.size();

  // Initialize the random number generator
  std::default_random_engine generator(seed);
  // And create a uniform distribution we can sample from
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  std::vector<double> onsitedisorder;
  int systemsize = inversebasis[0].length();
  for (int i = 0; i < systemsize; ++i)
    onsitedisorder.push_back(distribution(generator));

  // Vector storing all the triplet pairs for setting the Sparse
  // Hamiltonian later on.
  std::vector<T> total;

// For each basis state..
#pragma omp parallel
  {
    // Each thread has its own local triplet list
    std::vector<T> tripletList;

#pragma omp for nowait
    for (int i = 0; i < num_basis; ++i) {
      // Extract the string describing this state
      std::string this_state = inversebasis[i];

      //
      // .. check for hopping
      //
      for (int site = 0; site < systemsize - 1; ++site) {
        // Make a copy of the state
        std::string newstate(this_state);

        if (this_state[site] == '0' && this_state[site + 1] == '1') {
          newstate[site] = '1';
          newstate[site + 1] = '0';

          basis::const_iterator got = basis.find(newstate);
          int j = (int)got->second;

          // Add forward and backward hopping entries
          tripletList.push_back(T(i, j, J));
          tripletList.push_back(T(j, i, std::conj(J)));
        }
      }

      //
      // .. then the local field and disorder
      //
      for (int site = 0; site < systemsize; ++site) {
        complex localspin = (int)(this_state[site] - '0');

        tripletList.push_back(T(i, i,
                                onsitedisorder.at(site) * localspin * W +
                                    localspin * F * complex(site, 0)));
      }  // Local field and disorder

      //
      // .. and then the interactions
      //
      for (int site = 0; site < systemsize - 1; ++site) {
        complex localspin = (int)(this_state[site] - '0');
        complex nextspin = (int)(this_state[site + 1] - '0');
        tripletList.push_back(T(i, i, localspin * nextspin * U));
      }  // interactions

    }  // basis
#pragma omp critical
    total.insert(total.end(), tripletList.begin(), tripletList.end());
  }

  // Create the sparse Hamiltonian matrix and populate it with the triplets
  observable H(num_basis, num_basis);
  H.setFromTriplets(total.begin(), total.end());
  return H;
}

/**
  Constructs the density operator on a given site
*/
observable build_density_operator(basis basis, inversebasis inversebasis,
                                  int site) {
  int num_basis = (int)basis.size();
  std::vector<T> total;

// For each basis state..
#pragma omp parallel
  {
    std::vector<T> tripletList;
#pragma omp for nowait
    for (int i = 0; i < num_basis; ++i) {
      //  for (std::pair<std::string, int> element : basis) {
      std::string this_state = inversebasis[i];
      complex localspin = (int)(this_state[site] - '0');

      tripletList.push_back(T(i, i, localspin));
    }
#pragma omp critical
    total.insert(total.end(), tripletList.begin(), tripletList.end());
  }

  // Create a new sparse matrix and populate it with the triplets
  observable N(num_basis, num_basis);
  N.setFromTriplets(total.begin(), total.end());
  return N;
}

/**
  Constructs the initial wavefunction. It is set to the
  statestring specified. If that statestring does not exist,
  the program will crash.
*/
state build_initial_state(basis basis, std::string statestring) {
  int num_basis = (int)basis.size();
  state Psi(num_basis);
  Psi = Eigen::VectorXcd::Zero(num_basis);

  basis::const_iterator got = basis.find(statestring);
  int that_index = (int)got->second;

  std::cout << "  [*] State found at index: " << that_index << std::endl;

  Psi.coeffRef(that_index) += complex(1, 0);
  return Psi;
}

/**
 * Creates an orthonormal subspace the size of Q's columns.
 */
void build_orthogonal_basis_Arnoldi(observable H, state Psi,
                                    Eigen::MatrixXcd &Q, Eigen::MatrixXcd &h) {
  Q.col(0) = Psi.normalized();
  for (int i = 1; i < Q.cols(); ++i) {
    // Obtain the next q vector
    state q = H * Q.col(i - 1);

    // Orthonormalize it w.r.t. all the others
    for (int k = 0; k < i; ++k) {
      h(k, i - 1) = Q.col(k).dot(q);
      q = q - h(k, i - 1) * Q.col(k);
    }

    // Normalize it and store that in our upper Hessenberg matrix h
    h(i, i - 1) = q.norm();

    // Check if we found a subspace
    if (h(i, i - 1).real() <= 1e-14) {
      std::cerr << "Invariant subspace found before finishing Arnoldi"
                << std::endl;
    }

    // And set it as the next column of Q
    Q.col(i) = q / h(i, i - 1);
  }

  // Complete h by setting the last column manually
  int i = Q.cols() - 1;
  h(i, i) = Q.col(i).dot(H * Q.col(i));
  h(i - 1, i) = h(i, i - 1);
}

/**
 * Creates an orthonormal subspace the size of Q's columns.
 */
void build_orthogonal_basis_Lanczos(observable H, state Psi,
                                    Eigen::MatrixXcd &Q, Eigen::MatrixXcd &h) {
  Eigen::VectorXcd alpha(Q.cols());
  Eigen::VectorXcd beta(Q.cols());

  beta(0) = 0;

  Q.col(0) = Psi.normalized();  // u1; U1 = [u1]
  for (int j = 1; j < Q.cols(); ++j) {
    // Obtain the next q vector
    state q = H * Q.col(j - 1);

    alpha(j - 1) = Q.col(j - 1).dot(q);

    q = q - alpha(j - 1) * Q.col(j - 1);
    if (j > 1) {
      q = q - beta(j - 2) * Q.col(j - 2);
    }

    // Re-orthogonalize
    auto delta = Q.col(j - 1).dot(q);
    q = q - Q.col(j - 1) * delta;
    alpha(j - 1) = alpha(j - 1) + delta;

    // Compute the norm
    beta(j - 1) = q.norm();

    // Check if we found a subspace
    if (beta(j - 1).real() <= 1e-14) {
      std::cerr << "Invariant subspace found before finishing Arnoldi"
                << std::endl;
    }

    // And set it as the next column of Q
    Q.col(j) = q / beta(j - 1);
  }

  // Set the projected Hamiltonian
  h.diagonal() = alpha.tail(Q.cols());
  h.diagonal(-1) = beta.head(Q.cols() - 1);
  h.diagonal(+1) = beta.head(Q.cols() - 1);

  // The last diagonal has to be set separately still
  int i = Q.cols() - 1;
  h(i, i) = Q.col(i).dot(H * Q.col(i));
}

/**
  Propagates Psi for one dt under H using a Krylov subspace
  of dimension m.
*/
state krylov_propagate(observable H, state Psi, double dt, int m) {
  // Krylov subspace orthogonal basis
  Eigen::MatrixXcd Q(Psi.size(), m);
  Q = Eigen::MatrixXcd::Zero(Psi.size(), m);

  // H projected into the Krylov subspace. It is an upper Hessenberg
  // matrix if doing Arnoldi, but we never need the extra bottom row
  // if we're not implementing an adaptive routine that automatically
  // sets m.
  Eigen::MatrixXcd h(m, m);
  h = Eigen::MatrixXcd::Zero(m, m);

  /**
   * Orthogonalize using Gram-Schmidt (sometimes called Arnoldi, since the
   *   to-be-orthogonalized vectors are computed on-the-fly instead of being
   *   stored as a matrix).
   * Includes a re-orthogonalization step
   *   (https://link.springer.com/article/10.1007/s00211-005-0615-4)
   *
   * TODO: A better thing to do here might be to use householder to
   *       compute a QR decomposition and use that! After this first
   *       round of GramSchmidt-ing, the resulting vectors are not
   *       very orthogonal for large m (about 10).
   *
   * TODO: We are building a Krylov subspace of the Hamiltonian, i.e.
   *       of a Hermitian matrix. So we should be using Lanczos instead of
   *       Arnoldi!
   */

  /*
  // Use Arnoldi to compute Q and h
  build_orthogonal_basis_Arnoldi(H, Psi, Q, h);
  std::cout << "Arnoldi: " << std::endl << h << std::endl;
  std::cout << "QdagQ:" << std::endl << Q.adjoint() * Q << std::endl;
  */

  // Use the Lanczos algorithm with re-orthogonalization to
  // compute an orthonormal Krylov subspace basis, and use it
  // to obtain the projection matrix Q and the projected Hamiltonian h.
  build_orthogonal_basis_Lanczos(H, Psi, Q, h);

  // Convert to represent -iHdt
  h *= complex(0, -dt);
  // And return the first column of the back-transformed H
  return (Q * h.exp()).col(0);
}

#endif  //__KRYLOV_HPP__
