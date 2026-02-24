#include "FSI_parallel.hpp"
#include <chrono>

int main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      auto t_start = std::chrono::high_resolution_clock::now();

      FluidStructureProblem<2> flow_problem(1, 1);
      flow_problem.run(5, 1e-4f);

      auto t_end = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(t_end - t_start).count();

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "[Timer] Total wall time: " << elapsed << " s" << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << "\n----------------------------------------------------\n"
                << "Exception on processing:\n"
                << exc.what() << "\nAborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...)
    {
      std::cerr << "\n----------------------------------------------------\n"
                << "Unknown exception!\nAborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }

  return 0;
}