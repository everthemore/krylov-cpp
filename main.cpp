#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <bitset>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unsupported/Eigen/MatrixFunctions>
#include "krylov.hpp"

/*
static const int L = 10;
static const int k = 5;

static const complex J = 0.5;
static const complex F = 0;
static const complex U = 1;
static const double W = 0;
const int seed = 0;
*/

int main(int argc, char **argv) {
  /*
    Parse the arguments
  */
  const int L = std::stoi(argv[1]);
  const int k = std::stoi(argv[2]);
  const complex J = complex(std::stof(argv[3]), 0);
  const complex F = complex(std::stof(argv[4]), 0);
  const complex U = complex(std::stof(argv[5]), 0);
  const complex W = complex(std::stof(argv[6]), 0);
  const int seed = std::stoi(argv[7]);

  std::cout << "Starting Krylov time-evolution for " << std::to_string(L)
            << " sites and " << std::to_string(k) << " particles" << std::endl;

  std::cout << "[*] Creating basis states" << std::endl;
  // Create basis map
  basis basis = build_basis(L, k);
  std::cout << "[*] Done" << std::endl;

  std::cout << "[*] Building Hamiltonian with field " << F
            << " and interactions " << U << std::endl;
  observable H = build_hamiltonian(basis, J, F, U, W, seed);
  std::cout << "[*] Done" << std::endl;

  int countit = 0;
  for (int k = 0; k < H.outerSize(); ++k)
    for (observable::InnerIterator it(H, k); it; ++it) {
      it.value();
      it.row();    // row index
      it.col();    // col index (here it is equal to k)
      it.index();  // inner index, here it is equal to it.row()
      countit++;
    }

  std::cout << "Non-zero entries: " << countit << std::endl;

  std::string statestring = "";
  for (int i = 0; i < L; i += 2) statestring += "01";
  std::cout << "[*] Setting up initial state " << std::endl;
  std::cout << "  [*] " << statestring << std::endl;
  auto Psi = build_initial_state(basis, statestring);
  std::cout << "[*] Done" << std::endl;

  std::cout << "[*] Constructing the density operators" << std::endl;
  std::vector<observable> N;
  for (int i = 0; i < L; ++i) N.push_back(build_density_operator(basis, i));
  std::cout << "[*] Done" << std::endl;

  /*
  Eigen::SelfAdjointEigenSolver<observable> es(H);
  std::cout << "The eigenvalues of H are:" << std::endl
            << es.eigenvalues() << std::endl;
  */

  double dt = 0.005;

  std::ofstream outputfile;
  char filename[128];
  std::sprintf(filename, "imbalance-L-%d-F-%.2f-U-%.2f-W-%.2f-seed-%d.txt", L,
               F.real(), U.real(), W.real(), seed);
  outputfile.open(filename);

  state Psit = Psi;
  std::cout << "[*] Running Krylov propagation..." << std::endl;
  for (int i = 0; i < 10000; ++i) {
    double density_even = 0;
    for (int site = 0; site < L; site += 2)
      density_even += Psit.dot(N[site] * Psit).real();

    double density_odd = 0;
    for (int site = 1; site < L; site += 2)
      density_odd += Psit.dot(N[site] * Psit).real();

    double imbalance =
        (density_odd - density_even) / (density_odd + density_even);

    if (i % 100 == 0) outputfile << i * dt << "\t" << imbalance << std::endl;
    Psit = krylov_propagate(H, Psit, dt, 3);
  }
  std::cout << "[*] Done" << std::endl;
  outputfile.close();

  return 0;
}