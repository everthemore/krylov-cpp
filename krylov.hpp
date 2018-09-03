#ifndef __KRYLOV_HPP__
#define __KRYLOV_HPP__

using basis = std::unordered_map<std::string, int>;
using observable = Eigen::SparseMatrix<std::complex<double>>;
using state = Eigen::VectorXcd;
using complex = std::complex<double>;

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

int count_ones(std::string s) {
  int count = 0;
  for (int i = 0; i < s.length(); ++i)
    if (s[i] == '1') count++;
  return count;
}

/**
  Builds the allowed basis states for L sites and k particles
*/
basis build_basis(const int L, int k) {
  int counter = 0;
  basis basis;

  for (int i = 0; i < pow(2, L); ++i) {
    std::string binrep = int_to_bin(i, L);
    if (count_ones(binrep) == k) basis.insert({binrep, counter++});
  }

  return basis;
}

/**
  Constructs the Hamiltonian matrix in sparse format
*/
observable build_hamiltonian(basis basis, complex J, complex F, complex U,
                             complex W, int seed) {
  int num_basis = (int)basis.size();
  observable H(num_basis, num_basis);

  // For each basis state..
  for (std::pair<std::string, int> element : basis) {
    int this_index = element.second;

    // .. check for hopping
    for (int site = 0; site < element.first.length() - 1; ++site) {
      // Make a copy of the state
      std::string newstate(element.first);

      if (element.first[site] == '0' && element.first[site + 1] == '1') {
        newstate[site] = '1';
        newstate[site + 1] = '0';

        basis::const_iterator got = basis.find(newstate);
        int that_index = (int)got->second;

        // Add forward and backward hopping entries
        H.coeffRef(this_index, that_index) += J * std::complex<double>(1, 0);
        H.coeffRef(that_index, this_index) += J * std::complex<double>(1, 0);
      }
    }

    // .. then the local field and disorder
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int site = 0; site < element.first.length(); ++site) {
      // complex localspin = 2 * ((int)(element.first[site] - '0') - 0.5);
      complex localspin = (int)(element.first[site] - '0');

      // Add on-site disorder
      H.coeffRef(this_index, this_index) +=
          distribution(generator) * localspin * W;

      // Add local field
      H.coeffRef(this_index, this_index) += localspin * complex(site, 0) * F;
    }  // Local field and disorder

    // .. and then the interactions
    for (int site = 0; site < element.first.length() - 1; ++site) {
      // complex localspin = 2 * ((int)(element.first[site] - '0') - 0.5);
      complex localspin = (int)(element.first[site] - '0');
      // complex nextspin = 2 * ((int)(element.first[site + 1] - '0') - 0.5);
      complex nextspin = (int)(element.first[site + 1] - '0');
      H.coeffRef(this_index, this_index) += localspin * nextspin * U;
    }  // interactions

  }  // basis

  return H;
}

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
  Constructs the density operator on a given site
*/
observable build_density_operator(basis basis, int site) {
  int num_basis = (int)basis.size();
  observable N(num_basis, num_basis);

  // For each basis state..
  for (std::pair<std::string, int> element : basis) {
    int this_index = element.second;
    complex localspin = (int)(element.first[site] - '0');
    N.coeffRef(this_index, this_index) += localspin;
  }
  return N;
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
