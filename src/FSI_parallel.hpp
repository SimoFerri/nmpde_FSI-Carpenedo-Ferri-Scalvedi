#ifndef FSI_PARALLEL_HPP
#define FSI_PARALLEL_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <iostream>
#include <fstream>

using namespace dealii;

// -----------------------------------------------------------------------
// FSIPreconditioner
// -----------------------------------------------------------------------
class FSIPreconditioner : public Subscriptor
{
public:
void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                const TrilinosWrappers::SparseMatrix &pressure_mass_,
                const TrilinosWrappers::SparseMatrix &displacement_stiffness_,
                const TrilinosWrappers::SparseMatrix &B_,
                const TrilinosWrappers::SparseMatrix &Cu_,
                const TrilinosWrappers::SparseMatrix &Cp_,
                const std::vector<std::vector<bool>> &displacement_constant_modes = {});

void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
            const TrilinosWrappers::MPI::BlockVector &src) const;

protected:
const TrilinosWrappers::SparseMatrix *velocity_stiffness;
const TrilinosWrappers::SparseMatrix *displacement_stiffness;
const TrilinosWrappers::SparseMatrix *pressure_mass;
const TrilinosWrappers::SparseMatrix *B, *Cu, *Cp;

TrilinosWrappers::PreconditionAMG preconditioner_velocity;
TrilinosWrappers::PreconditionAMG preconditioner_displacement;
TrilinosWrappers::PreconditionILU preconditioner_pressure;

mutable TrilinosWrappers::MPI::Vector tmp_p, tmp_d, intermediate_tmp;
};

// -----------------------------------------------------------------------
// StokesBoundaryValues
// -----------------------------------------------------------------------
template <int dim>
class StokesBoundaryValues : public Function<dim>
{
public:
StokesBoundaryValues()
    : Function<dim>(dim + 1 + dim)
{}

virtual double value(const Point<dim>  &p,
                        const unsigned int component = 0) const override;

virtual void vector_value(const Point<dim> &p,
                            Vector<double>   &value) const override;
};

// -----------------------------------------------------------------------
// FluidStructureProblem
// -----------------------------------------------------------------------
template <int dim>
class FluidStructureProblem
{
public:
FluidStructureProblem(const unsigned int stokes_degree,
                        const unsigned int elasticity_degree);
void run(const unsigned int max_cycles, const float tol);

private:
enum
{
    fluid_domain_id,
    solid_domain_id
};

static bool cell_is_in_fluid_domain(
    const typename DoFHandler<dim>::cell_iterator &cell);

static bool cell_is_in_solid_domain(
    const typename DoFHandler<dim>::cell_iterator &cell);

void make_grid();
void set_active_fe_indices();
void setup_dofs();
void assemble_system();
void assemble_interface_term(
    const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
    const FEFaceValuesBase<dim>          &stokes_fe_face_values,
    std::vector<Tensor<1, dim>>          &elasticity_phi,
    std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
    std::vector<double>                  &stokes_phi_p,
    FullMatrix<double>                   &local_interface_matrix) const;
void  solve();
void  output_results(const unsigned int refinement_cycle) const;
float refine_mesh(float tol);

const unsigned int stokes_degree;
const unsigned int elasticity_degree;

ConditionalOStream pcout;

parallel::distributed::Triangulation<dim> triangulation;
FESystem<dim>         stokes_fe;
FESystem<dim>         elasticity_fe;
hp::FECollection<dim> fe_collection;
DoFHandler<dim>       dof_handler;

IndexSet              locally_owned_dofs;
IndexSet              locally_relevant_dofs;
std::vector<IndexSet> block_owned_dofs;
std::vector<IndexSet> block_relevant_dofs;

AffineConstraints<double> constraints;

TrilinosWrappers::MPI::BlockVector  solution;
TrilinosWrappers::MPI::BlockVector  solution_owned;
TrilinosWrappers::MPI::BlockVector  system_rhs;
TrilinosWrappers::BlockSparseMatrix system_matrix;
TrilinosWrappers::BlockSparseMatrix mass_matrix;

const double viscosity;
const double lambda;
const double mu;
};

#endif // FSI_PARALLEL_HPP