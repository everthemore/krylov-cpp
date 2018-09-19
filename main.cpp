#include "krylov.hpp"

int main(int argc, char **argv) {
  /*
    Parse the arguments
  */
  const int L = std::stoi(argv[1]);
  const int k = (int)(L / 2);  // std::stoi(argv[2]);
  const complex J = complex(std::stof(argv[2]), 0);
  const complex F = complex(std::stof(argv[3]), 0);
  const complex U = complex(std::stof(argv[4]), 0);
  const complex W = complex(std::stof(argv[5]), 0);
  const int m = std::stoi(argv[6]);
  const int seed = std::stoi(argv[7]);

  double dt = 0.025;
  int num_krylov_steps = 40000;
  int compute_every = 100;
  int checkpoint_every = 5000;

  // Wavefunction output filename
  char wf_filename[128];
  std::sprintf(wf_filename,
               "wavefunction-L-%d-F-%.2f-U-%.2f-W-%.2f-m-%d-seed-%d.txt", L,
               F.real(), U.real(), W.real(), m, seed);

  // Imbalance output filename
  char imbalance_filename[128];
  std::sprintf(imbalance_filename,
               "imbalance-L-%d-F-%.2f-U-%.2f-W-%.2f-m-%d-seed-%d.txt", L,
               F.real(), U.real(), W.real(), m, seed);

  std::cout << "Starting Krylov time-evolution for " << std::to_string(L)
            << " sites and " << std::to_string(k) << " particles" << std::endl;

  std::cout << "[*] Creating basis states" << std::endl;
  // Create basis map
  inversebasis inversebasis;
  basis basis = build_basis(L, k, inversebasis);
  std::cout << "[*] Done" << std::endl;

  std::cout << "[*] Building Hamiltonian with field " << F
            << " and interactions " << U << std::endl;
  observable H = build_hamiltonian(basis, inversebasis, J, F, U, W, seed);
  std::cout << "[*] Done" << std::endl;

  std::string statestring = "";
  for (int i = 0; i < L; i += 2) statestring += "01";
  std::cout << "[*] Setting up initial state " << std::endl;
  std::cout << "  [*] " << statestring << std::endl;
  auto Psi = build_initial_state(basis, statestring);
  std::cout << "[*] Done" << std::endl;

  std::cout << "[*] Constructing the density operators" << std::endl;
  int num_basis = basis.size();
  observable evenDensity(num_basis, num_basis);
  observable oddDensity(num_basis, num_basis);
  for (int i = 0; i < L; ++i) {
    if (i % 2 == 0) {
      evenDensity =
          evenDensity + build_density_operator(basis, inversebasis, i);
    } else {
      oddDensity = oddDensity + build_density_operator(basis, inversebasis, i);
    }
  }
  std::cout << "[*] Done" << std::endl;

  std::ofstream outputfile;
  outputfile.open(imbalance_filename);

  std::cout << "[*] Running Krylov propagation..." << std::endl;
  state Psit = Psi;
  for (int i = 0; i < num_krylov_steps; ++i) {
    if (i % compute_every == 0) {
      double density_even = Psit.dot(evenDensity * Psit).real();
      double density_odd = Psit.dot(oddDensity * Psit).real();
      double imbalance =
          (density_odd - density_even) / (density_odd + density_even);

      outputfile << i * dt << "\t" << imbalance << std::endl;
    }

    Psit = krylov_propagate(H, Psit, dt, m);

    // Checkpoint the wavefunction
    if (i % checkpoint_every == 0) {
      std::ofstream wfoutfile;
      wfoutfile.open(wf_filename);
      wfoutfile << Psit << std::endl;
      wfoutfile.close();
    }
  }

  std::cout << "[*] Done" << std::endl;
  outputfile.close();

  return 0;
}
