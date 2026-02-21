#include "FSI_parallel.hpp"

int main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      FluidStructureProblem<2> flow_problem(1, 1);
      flow_problem.run(5, 1e-4f);
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