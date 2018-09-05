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
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

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
      for (int site = 0; site < this_state.length() - 1; ++site) {
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
      for (int site = 0; site < this_state.length(); ++site) {
        complex localspin = (int)(this_state[site] - '0');

        tripletList.push_back(
            T(i, i, distribution(generator) * localspin * W + localspin * F));
      }  // Local field and disorder

      //
      // .. and then the interactions
      //
      for (int site = 0; site < this_state.length() - 1; ++site) {
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
  Propagates Psi for one dt under H using a Krylov subspace
  of dimension m.
*/
state krylov_propagate(observable H, state Psi, double dt, int m) {
  // Krylov subspace orthogonal basis
  Eigen::MatrixXcd Q(Psi.size(), m);
  Q = Eigen::MatrixXcd::Zero(Psi.size(), m);
  // H projected into the Krylov subspace
  Eigen::MatrixXcd h(m, m);
  h = Eigen::MatrixXcd::Zero(m, m);

  // Set the first column
  Q.col(0) = Psi.normalized();
  for (int i = 1; i < m; ++i) {
    // Obtain the next q vector
    state qi = H * Q.col(i - 1);

    // Orthonormalize it w.r.t. all the others
    for (int k = 0; k < i; ++k) {
      qi = qi - Q.col(k).dot(qi) * Q.col(k);
    }

    // And set the next column
    Q.col(i) = qi.normalized();
  }

  // Now we can project the Hamiltonian into the Krylov subspace
  h = Q.adjoint() * H * Q;  // Is now an m-by-m matrix
  // Convert to represent -iHdt
  h *= complex(0, -dt);

  // This is a basis-by-basis times basis multiplication; this should
  // be sparse and hence not cost much memory; but it does..
  // return (Q * h.exp() * Q.adjoint()) * Q.col(0);
  return (Q * h.exp()).col(0);
}

#endif
