#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
 
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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
 
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
 
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/lac/sparsity_tools.h>

#include <iostream>
#include <fstream>
 
 

namespace Step46
{
  using namespace dealii;
 
  class FSIPreconditioner : public Subscriptor // EDITED
  {
  public:
    void initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                    const TrilinosWrappers::SparseMatrix &pressure_mass_,
                    const TrilinosWrappers::SparseMatrix &displacement_stiffness_,
                    const TrilinosWrappers::SparseMatrix &B_,
                    const TrilinosWrappers::SparseMatrix &Cu_,
                    const TrilinosWrappers::SparseMatrix &Cp_,
                    const std::vector<std::vector<bool>> &displacement_constant_modes = {})
    {
      velocity_stiffness     = &velocity_stiffness_;
      pressure_mass          = &pressure_mass_;
      displacement_stiffness = &displacement_stiffness_;
      B = &B_; 
      Cu = &Cu_; 
      Cp = &Cp_;

      // VELOCITY AMG
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_vel;
      amg_data_vel.elliptic              = true; // AMG ASSUMES ELLIPTICITY, THIS IS AN APPROXIMATION FOR THE VELOCITY BLOCK
      amg_data_vel.higher_order_elements = true; // WE ARE USING Q2 FOR VELOCITY
      amg_data_vel.smoother_sweeps       = 2; // FEW SMOOTHER SWEEPS, AMG IS JUST AN APPROXIMATION
      amg_data_vel.aggregation_threshold = 1e-3; // LOW THRESHOLD TO GET MORE AGGREGATES, AMG IS JUST AN APPROXIMATION
      preconditioner_velocity.initialize(velocity_stiffness_, amg_data_vel);

      // ILU FOR PRESSURE MASS MATRIX
      preconditioner_pressure.initialize(pressure_mass_); 

      // DISPLACEMENT AMG
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_disp;
      amg_data_disp.constant_modes        = displacement_constant_modes; // PASS CONSTANT MODES TO AMG TO IMPROVE COARSENING, ESPECIALLY FOR LOW-ORDER DISPLACEMENT SPACES
      amg_data_disp.elliptic              = true; // AMG ASSUMES ELLIPTICITY, THIS IS AN APPROXIMATION FOR THE DISPLACEMENT BLOCK
      amg_data_disp.higher_order_elements = false; // WE ARE USING Q1 FOR DISPLACEMENT, SETTING THIS TO TRUE CAUSES AMG TO USE A MORE EXPENSIVE SMOOTHER, WHICH WE DON'T NEED FOR LOW-ORDER DISPLACEMENT BLOCK
      amg_data_disp.smoother_sweeps       = 3; // A FEW MORE SMOOTHER SWEEPS FOR THE DISPLACEMENT BLOCK, WHICH CAN BE MORE DIFFICULT TO SOLVE THAN THE VELOCITY BLOCK
      amg_data_disp.aggregation_threshold = 1e-3; // LOW THRESHOLD TO GET MORE AGGREGATES, AMG IS JUST AN APPROXIMATION
      preconditioner_displacement.initialize(displacement_stiffness_, amg_data_disp); 

      tmp_p.reinit(pressure_mass_.locally_owned_domain_indices(), pressure_mass_.get_mpi_communicator());
      tmp_d.reinit(displacement_stiffness_.locally_owned_domain_indices(), displacement_stiffness_.get_mpi_communicator());
      intermediate_tmp.reinit(displacement_stiffness_.locally_owned_domain_indices(), displacement_stiffness_.get_mpi_communicator());
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // SOLVE FLUID VELOCITY
      {
        SolverControl solver_control_vel(2000, 1e-2 * src.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_vel(solver_control_vel);
        dst.block(0) = 0;
        solver_gmres_vel.solve(*velocity_stiffness, dst.block(0), src.block(0), preconditioner_velocity);
      }

      // SOLVE FLUID PRESSURE
      B->vmult(tmp_p, dst.block(0)); // tmp_p = B * dst.block(0)
      tmp_p.sadd(-1.0, 1.0, src.block(1)); // tmp_p = - B * dst.block(0) + src.block(1)
      {
        SolverControl solver_control_pres(2000, 1e-2 * tmp_p.l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pres(solver_control_pres);
        dst.block(1) = 0;
        solver_cg_pres.solve(*pressure_mass, dst.block(1), tmp_p, preconditioner_pressure);
      }

      // SOLVE DISPLACEMENT
      Cu->vmult(tmp_d, dst.block(0)); // tmp_d = Cu * dst.block(0)
      Cp->vmult(intermediate_tmp, dst.block(1)); // intermediate_tmp = Cp * dst.block(1)
      tmp_d.add(1.0, intermediate_tmp); // tmp_d = Cu * dst.block(0) + Cp * dst.block(1)
      tmp_d.sadd(-1.0, 1.0, src.block(2)); // tmp_d = - Cu * dst.block(0) - Cp * dst.block(1) + src.block(2)
      {
        SolverControl solver_control_disp(2000, 1e-2 * tmp_d.l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_disp(solver_control_disp);
        dst.block(2) = 0;
        solver_gmres_disp.solve(*displacement_stiffness, dst.block(2), tmp_d, preconditioner_displacement);
      }
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness, *displacement_stiffness, *pressure_mass;
    const TrilinosWrappers::SparseMatrix *B, *Cu, *Cp;

    TrilinosWrappers::PreconditionAMG preconditioner_velocity, preconditioner_displacement;
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    mutable TrilinosWrappers::MPI::Vector tmp_p, tmp_d, intermediate_tmp; // MUTABLE BECAUSE WE MODIFY THEM
  };



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
      const FEFaceValuesBase<dim> &         elasticity_fe_face_values,
      const FEFaceValuesBase<dim> &         stokes_fe_face_values,
      std::vector<Tensor<1, dim>> &         elasticity_phi,
      std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
      std::vector<double> &                 stokes_phi_p,
      FullMatrix<double> &                  local_interface_matrix) const;
    void solve();
    void output_results(const unsigned int refinement_cycle) const;
    float refine_mesh(float tol);
 
    const unsigned int stokes_degree;
    const unsigned int elasticity_degree;

    MPI_Comm mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation; //al posto di Triangulation<dim>
    ConditionalOStream pcout;
 
    FESystem<dim>         stokes_fe;
    FESystem<dim>         elasticity_fe;
    hp::FECollection<dim> fe_collection;
    DoFHandler<dim>       dof_handler;

    AffineConstraints<double> constraints;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    
    // System matrix and vectors are block objects
    TrilinosWrappers::BlockSparseMatrix system_matrix;
    TrilinosWrappers::BlockSparseMatrix mass_matrix; // PRESSURE MASS MATRIX
 
    TrilinosWrappers::MPI::BlockVector solution;
    TrilinosWrappers::MPI::BlockVector system_rhs;
 
    const double viscosity;
    const double lambda;
    const double mu;
  };
 

 
  template <int dim>
  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues()
      : Function<dim>(dim + 1 + dim)
    {}
 
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
 
    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };
 
 

  template <int dim>
  double StokesBoundaryValues<dim>::value(const Point<dim> & p,
                                          const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
 
    if (component == dim - 1)
      switch (dim)
        {
          case 2:
            return std::sin(numbers::PI * p[0]);
          case 3:
            return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
          default:
            Assert(false, ExcNotImplemented());
        }
 
    return 0;
  }
 

 
  template <int dim>
  void StokesBoundaryValues<dim>::vector_value(const Point<dim> &p,
                                               Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = StokesBoundaryValues<dim>::value(p, c);
  }
 
 

  template <int dim>
  FluidStructureProblem<dim>::FluidStructureProblem(
    const unsigned int stokes_degree,
    const unsigned int elasticity_degree)
    : stokes_degree(stokes_degree)
    , elasticity_degree(elasticity_degree)
    , mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator, typename Triangulation<dim>::MeshSmoothing(
                                 Triangulation<dim>::maximum_smoothing))
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , stokes_fe(FE_Q<dim>(stokes_degree + 1),
                dim,
                FE_Q<dim>(stokes_degree),
                1,
                FE_Nothing<dim>(),
                dim)
    , elasticity_fe(FE_Nothing<dim>(),
                    dim,
                    FE_Nothing<dim>(),
                    1,
                    FE_Q<dim>(elasticity_degree),
                    dim)
    , dof_handler(triangulation)
    , viscosity(2)
    , lambda(1)
    , mu(1)
  {
    fe_collection.push_back(stokes_fe);
    fe_collection.push_back(elasticity_fe);
  }
 

 
  template <int dim>
  bool FluidStructureProblem<dim>::cell_is_in_fluid_domain(
    const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fluid_domain_id);
  }
 
 

  template <int dim>
  bool FluidStructureProblem<dim>::cell_is_in_solid_domain(
    const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_domain_id);
  }
 

 
  template <int dim>
  void FluidStructureProblem<dim>::make_grid()
  {
    GridGenerator::subdivided_hyper_cube(triangulation, 8, -1, 1);
 
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && (face->center()[dim - 1] == 1))
          face->set_all_boundary_ids(1);
 
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (((std::fabs(cell->center()[0]) < 0.25) &&
           (cell->center()[dim - 1] > 0.5)) ||
          ((std::fabs(cell->center()[0]) >= 0.25) &&
           (cell->center()[dim - 1] > -0.5)))
        cell->set_material_id(fluid_domain_id);
      else
        cell->set_material_id(solid_domain_id);
  }
 

  template <int dim>
  void FluidStructureProblem<dim>::set_active_fe_indices()
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index(0);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index(1);
        else
          Assert(false, ExcNotImplemented());
      }
  }

 

  template <int dim>
  void FluidStructureProblem<dim>::setup_dofs() // EDITED
  {
    set_active_fe_indices();
    dof_handler.distribute_dofs(fe_collection);

    // RENUMBERING DOFS: VELOCITY (0), PRESSURE (1), DISPLACEMENT (2)
    std::vector<unsigned int> block_component(dim + 1 + dim);
    for (unsigned int d = 0; d < dim; ++d)
      block_component[d] = 0;
    block_component[dim] = 1;
    for (unsigned int d = 0; d < dim; ++d)
      block_component[dim + 1 + d] = 2;

    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // partitioning blocks
    std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];
    const unsigned int n_d = dofs_per_block[2];

    std::vector<IndexSet> owned_partitioning(3);
    owned_partitioning[0] = locally_owned_dofs.get_view(0, n_u);
    owned_partitioning[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    owned_partitioning[2] = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

    std::vector<IndexSet> relevant_partitioning(3);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
    relevant_partitioning[2] = locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

    pcout << " DoFs: velocity = " << n_u << ", pressure = " << n_p << ", displacement = " << n_d << std::endl;

    // vincoli con i dofs rilevanti
    {
      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      
      const FEValuesExtractors::Vector velocities(0);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               StokesBoundaryValues<dim>(),
                                               constraints,
                                               fe_collection.component_mask(
                                                 velocities));
 
      const FEValuesExtractors::Vector displacements(dim + 1);
      VectorTools::interpolate_boundary_values(
        dof_handler,
        0,
        Functions::ZeroFunction<dim>(dim + 1 + dim),
        constraints,
        fe_collection.component_mask(displacements));

      std::vector<types::global_dof_index> local_face_dof_indices(
        stokes_fe.n_dofs_per_face());
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if(cell->is_artificial())
          continue;

        if (cell_is_in_fluid_domain(cell))
          for (const auto face_no : cell->face_indices())
            if (cell->face(face_no)->at_boundary() == false)
              {
                bool face_is_on_interface = false;
 
                if ((cell->neighbor(face_no)->has_children() == false) &&
                    (cell_is_in_solid_domain(cell->neighbor(face_no))))
                  face_is_on_interface = true;
                else if (cell->neighbor(face_no)->has_children() == true)
                  {
                    for (unsigned int sf = 0;
                         sf < cell->face(face_no)->n_children();
                         ++sf)
                      if (cell_is_in_solid_domain(
                            cell->neighbor_child_on_subface(face_no, sf)))
                        {
                          face_is_on_interface = true;
                          break;
                        }
                  }
 
                if (face_is_on_interface)
                  {
                    cell->face(face_no)->get_dof_indices(local_face_dof_indices,
                                                         0);
                    for (unsigned int i = 0; i < local_face_dof_indices.size();
                         ++i)
                      if (stokes_fe.face_system_to_component_index(i).first <
                          dim)
                          {
                            if (!constraints.is_constrained(local_face_dof_indices[i]))
                              constraints.add_line(local_face_dof_indices[i]);
                          }
                  }
              }

      constraints.close();

    }
  }

    {
      // SETUP 3X3 BLOCK SPARSITY PATTERN
      BlockDynamicSparsityPattern dsp(3, 3);
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          dsp.block(i, j).reinit(dofs_per_block[i], dofs_per_block[j]);
      dsp.collect_sizes();

      Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                                 fe_collection.n_components());
      Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                                 fe_collection.n_components());
  
      for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
        for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
          {
            if (((c < dim + 1) && (d < dim + 1) /*&& !((c == dim) && (d == dim))*/) ||
                ((c >= dim + 1) && (d >= dim + 1)))
              cell_coupling[c][d] = DoFTools::always;
 
            if ((c >= dim + 1) && (d < dim + 1))
              face_coupling[c][d] = DoFTools::always;
          }
 
      DoFTools::make_flux_sparsity_pattern(dof_handler,
                                           dsp,
                                           cell_coupling,
                                           face_coupling);
      constraints.condense(dsp);

      // Initialize system matrix and vectors (trilinos block objects)
      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
      mass_matrix.reinit(owned_partitioning, dsp, mpi_communicator); // WE NEED MASS_MATRIX.BLOCK(1,1) FOR THE PRECONDITIONER
 
    }
 
    solution.reinit(owned_partitioning, mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);
    /*
    solution.reinit(3); // 3 INSTEAD OF 2
    for (unsigned int i = 0; i < 3; ++i)
      solution.block(i).reinit(dofs_per_block[i]);
    solution.collect_sizes();

    system_rhs.reinit(3); // 3 INSTEAD OF 2
    for (unsigned int i = 0; i < 3; ++i)
      system_rhs.block(i).reinit(dofs_per_block[i]);
    system_rhs.collect_sizes();
    */
  }


 
  template <int dim>
  void FluidStructureProblem<dim>::assemble_system() // EDITED
  {
    system_matrix = 0;
    mass_matrix = 0;
    system_rhs = 0;
 
    const QGauss<dim> stokes_quadrature(stokes_degree + 2);
    const QGauss<dim> elasticity_quadrature(elasticity_degree + 2);
 
    hp::QCollection<dim> q_collection;
    q_collection.push_back(stokes_quadrature);
    q_collection.push_back(elasticity_quadrature);
 
    hp::FEValues<dim> hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_gradients);
 
    const QGauss<dim - 1> common_face_quadrature(
      std::max(stokes_degree + 2, elasticity_degree + 2));
 
    FEFaceValues<dim>    stokes_fe_face_values(stokes_fe,
                                            common_face_quadrature,
                                            update_JxW_values |
                                              update_gradients | update_values);
    FEFaceValues<dim>    elasticity_fe_face_values(elasticity_fe,
                                                common_face_quadrature,
                                                update_normal_vectors |
                                                  update_values);
    FESubfaceValues<dim> stokes_fe_subface_values(stokes_fe,
                                                  common_face_quadrature,
                                                  update_JxW_values |
                                                    update_gradients |
                                                    update_values);
    FESubfaceValues<dim> elasticity_fe_subface_values(elasticity_fe,
                                                      common_face_quadrature,
                                                      update_normal_vectors |
                                                        update_values);
 
    const unsigned int stokes_dofs_per_cell = stokes_fe.n_dofs_per_cell();
    const unsigned int elasticity_dofs_per_cell =
      elasticity_fe.n_dofs_per_cell();
 
    FullMatrix<double> local_matrix;
    FullMatrix<double> local_interface_matrix(elasticity_dofs_per_cell,
                                              stokes_dofs_per_cell);
    FullMatrix<double> local_pressure_mass(stokes_dofs_per_cell, 
                                           stokes_dofs_per_cell); // LOCAL PRESSURE MASS MATRIX
    Vector<double>     local_rhs;
 
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_dof_indices(
      stokes_dofs_per_cell);
 
    const Functions::ZeroFunction<dim> right_hand_side(dim + 1);
 
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacements(dim + 1);
 
    std::vector<SymmetricTensor<2, dim>> stokes_symgrad_phi_u(
      stokes_dofs_per_cell);
    std::vector<double> stokes_div_phi_u(stokes_dofs_per_cell);
    std::vector<double> stokes_phi_p(stokes_dofs_per_cell);
 
    std::vector<Tensor<2, dim>> elasticity_grad_phi(elasticity_dofs_per_cell);
    std::vector<double>         elasticity_div_phi(elasticity_dofs_per_cell);
    std::vector<Tensor<1, dim>> elasticity_phi(elasticity_dofs_per_cell);
 
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if(!cell->is_locally_owned())
          continue;

        hp_fe_values.reinit(cell);
 
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
 
        local_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                            cell->get_fe().n_dofs_per_cell());
        local_rhs.reinit(cell->get_fe().n_dofs_per_cell());
        local_pressure_mass.reinit(cell->get_fe().n_dofs_per_cell(), cell->get_fe().n_dofs_per_cell());
 
        if (cell_is_in_fluid_domain(cell))
          {
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            Assert(dofs_per_cell == stokes_dofs_per_cell, ExcInternalError());
 
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    stokes_symgrad_phi_u[k] =
                      fe_values[velocities].symmetric_gradient(k, q);
                    stokes_div_phi_u[k] =
                      fe_values[velocities].divergence(k, q);
                    stokes_phi_p[k] = fe_values[pressure].value(k, q);
                  }
 
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const unsigned int comp_i = fe_values.get_fe().system_to_component_index(i).first;
                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const unsigned int comp_j = fe_values.get_fe().system_to_component_index(j).first;
                    local_matrix(i, j) +=
                      (2 * viscosity * stokes_symgrad_phi_u[i] *
                         stokes_symgrad_phi_u[j] -
                       stokes_div_phi_u[i] * stokes_phi_p[j] -
                       stokes_phi_p[i] * stokes_div_phi_u[j]) *
                      fe_values.JxW(q);
                    
                    if (comp_i == dim && comp_j == dim) {
                      local_pressure_mass(i, j) += (1.0 / viscosity) * 
                                                    stokes_phi_p[i] * stokes_phi_p[j] *
                                                    fe_values.JxW(q);
                    }
                  }
                }
              }
          }
        else
          {
            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            Assert(dofs_per_cell == elasticity_dofs_per_cell,
                   ExcInternalError());
 
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    elasticity_grad_phi[k] =
                      fe_values[displacements].gradient(k, q);
                    elasticity_div_phi[k] =
                      fe_values[displacements].divergence(k, q);
                  }
 
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) +=
                        (lambda * elasticity_div_phi[i] *
                           elasticity_div_phi[j] +
                         mu * scalar_product(elasticity_grad_phi[i],
                                             elasticity_grad_phi[j]) +
                         mu *
                           scalar_product(elasticity_grad_phi[i],
                                          transpose(elasticity_grad_phi[j]))) *
                        fe_values.JxW(q);
                    }
              }
          }
 
        local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);

        if (cell_is_in_fluid_domain(cell)) { // NEW
            constraints.distribute_local_to_global(local_pressure_mass,
                                               local_dof_indices,
                                               mass_matrix);
        }
 
        if (cell_is_in_solid_domain(cell))
          for (const auto f : cell->face_indices())
            if (cell->face(f)->at_boundary() == false)
              {
                if ((cell->neighbor(f)->level() == cell->level()) &&
                    (cell->neighbor(f)->has_children() == false) &&
                    cell_is_in_fluid_domain(cell->neighbor(f)))
                  {
                    elasticity_fe_face_values.reinit(cell, f);
                    stokes_fe_face_values.reinit(cell->neighbor(f),
                                                 cell->neighbor_of_neighbor(f));
 
                    assemble_interface_term(elasticity_fe_face_values,
                                            stokes_fe_face_values,
                                            elasticity_phi,
                                            stokes_symgrad_phi_u,
                                            stokes_phi_p,
                                            local_interface_matrix);
 
                    cell->neighbor(f)->get_dof_indices(neighbor_dof_indices);
                    constraints.distribute_local_to_global(
                      local_interface_matrix,
                      local_dof_indices,
                      neighbor_dof_indices,
                      system_matrix);
                  }
 
                else if ((cell->neighbor(f)->level() == cell->level()) &&
                         (cell->neighbor(f)->has_children() == true))
                  {
                    for (unsigned int subface = 0;
                         subface < cell->face(f)->n_children();
                         ++subface)
                      if (cell_is_in_fluid_domain(
                            cell->neighbor_child_on_subface(f, subface)))
                        {
                          elasticity_fe_subface_values.reinit(cell, f, subface);
                          stokes_fe_face_values.reinit(
                            cell->neighbor_child_on_subface(f, subface),
                            cell->neighbor_of_neighbor(f));
 
                          assemble_interface_term(elasticity_fe_subface_values,
                                                  stokes_fe_face_values,
                                                  elasticity_phi,
                                                  stokes_symgrad_phi_u,
                                                  stokes_phi_p,
                                                  local_interface_matrix);
 
                          cell->neighbor_child_on_subface(f, subface)
                            ->get_dof_indices(neighbor_dof_indices);
                          constraints.distribute_local_to_global(
                            local_interface_matrix,
                            local_dof_indices,
                            neighbor_dof_indices,
                            system_matrix);
                        }
                  }
 
                else if (cell->neighbor_is_coarser(f) &&
                         cell_is_in_fluid_domain(cell->neighbor(f)))
                  {
                    elasticity_fe_face_values.reinit(cell, f);
                    stokes_fe_subface_values.reinit(
                      cell->neighbor(f),
                      cell->neighbor_of_coarser_neighbor(f).first,
                      cell->neighbor_of_coarser_neighbor(f).second);
 
                    assemble_interface_term(elasticity_fe_face_values,
                                            stokes_fe_subface_values,
                                            elasticity_phi,
                                            stokes_symgrad_phi_u,
                                            stokes_phi_p,
                                            local_interface_matrix);
 
                    cell->neighbor(f)->get_dof_indices(neighbor_dof_indices);
                    constraints.distribute_local_to_global(
                      local_interface_matrix,
                      local_dof_indices,
                      neighbor_dof_indices,
                      system_matrix);
                  }
              }
      }

      system_matrix.compress(VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
  }
 
 
 
  template <int dim>
  void FluidStructureProblem<dim>::assemble_interface_term(
    const FEFaceValuesBase<dim> &         elasticity_fe_face_values,
    const FEFaceValuesBase<dim> &         stokes_fe_face_values,
    std::vector<Tensor<1, dim>> &         elasticity_phi,
    std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
    std::vector<double> &                 stokes_phi_p,
    FullMatrix<double> &                  local_interface_matrix) const
  {
    Assert(stokes_fe_face_values.n_quadrature_points ==
             elasticity_fe_face_values.n_quadrature_points,
           ExcInternalError());
    const unsigned int n_face_quadrature_points =
      elasticity_fe_face_values.n_quadrature_points;
 
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacements(dim + 1);
 
    local_interface_matrix = 0;
    for (unsigned int q = 0; q < n_face_quadrature_points; ++q)
      {
        const Tensor<1, dim> normal_vector =
          elasticity_fe_face_values.normal_vector(q);
 
        for (unsigned int k = 0; k < stokes_fe_face_values.dofs_per_cell; ++k)
          {
            stokes_symgrad_phi_u[k] =
              stokes_fe_face_values[velocities].symmetric_gradient(k, q);
            stokes_phi_p[k] = stokes_fe_face_values[pressure].value(k, q);
          }
        for (unsigned int k = 0; k < elasticity_fe_face_values.dofs_per_cell;
             ++k)
          elasticity_phi[k] =
            elasticity_fe_face_values[displacements].value(k, q);
 
        for (unsigned int i = 0; i < elasticity_fe_face_values.dofs_per_cell;
             ++i)
          for (unsigned int j = 0; j < stokes_fe_face_values.dofs_per_cell; ++j)
            local_interface_matrix(i, j) +=
              -((2 * viscosity * (stokes_symgrad_phi_u[j] * normal_vector) -
                 stokes_phi_p[j] * normal_vector) *
                elasticity_phi[i] * stokes_fe_face_values.JxW(q));
      }
  }
 
 
 
  template <int dim>
  void FluidStructureProblem<dim>::solve() // EDITED
  {
    FSIPreconditioner preconditioner; 

    preconditioner.initialize(system_matrix.block(0, 0), // VELOCITY STIFFNESS
                              mass_matrix.block(1, 1), // PRESSURE MASS MATRIX
                              system_matrix.block(2, 2), // DISPLACEMENT STIFFNESS
                              system_matrix.block(1, 0), // B (PRESSURE-VELOCITY COUPLING)
                              system_matrix.block(2, 0), // Cu (VELOCITY-DISPLACEMENT COUPLING)
                              system_matrix.block(2, 1), // Cp (PRESSURE-DISPLACEMENT COUPLING)
                              {});
    
    SolverControl solver_control(5000, 1e-8 * system_rhs.l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control); // WE USE GMRES FOR THE FULL BLOCK SYSTEM

    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    std::cout << "   Converged in " << solver_control.last_step() << " iterations." << std::endl;
  }

 
 
  template <int dim>
  void FluidStructureProblem<dim>::output_results(
    const unsigned int refinement_cycle) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.emplace_back("displacement");
 
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    TrilinosWrappers::MPI::BlockVector relevant_solution;

    const unsigned int n_u = solution.block(0).size();
    const unsigned int n_p = solution.block(1).size();
    const unsigned int n_d = solution.block(2).size();

    std::vector<IndexSet> owned_partitioning(3);
    owned_partitioning[0] = locally_owned_dofs.get_view(0, n_u);
    owned_partitioning[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    owned_partitioning[2] = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

    std::vector<IndexSet> relevant_partitioning(3);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
    relevant_partitioning[2] = locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);
    
    relevant_solution.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
 
    relevant_solution = solution;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
 
    data_out.add_data_vector(relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();
 
    const std::string filename = "solution-" + Utilities::int_to_string(refinement_cycle, 2);
    data_out.write_vtu_with_pvtu_record("./", filename, refinement_cycle, mpi_communicator, 4);
  }
 

 
  template <int dim>
  float FluidStructureProblem<dim>::refine_mesh(float tol)
  {
    Vector<float> stokes_estimated_error_per_cell(
      triangulation.n_active_cells());
    Vector<float> elasticity_estimated_error_per_cell(
      triangulation.n_active_cells());
 
    const QGauss<dim - 1> stokes_face_quadrature(stokes_degree + 2);
    const QGauss<dim - 1> elasticity_face_quadrature(elasticity_degree + 2);
 
    hp::QCollection<dim - 1> face_q_collection;
    face_q_collection.push_back(stokes_face_quadrature);
    face_q_collection.push_back(elasticity_face_quadrature);

    TrilinosWrappers::MPI::BlockVector relevant_solution;

    const unsigned int n_u = solution.block(0).size();
    const unsigned int n_p = solution.block(1).size();
    const unsigned int n_d = solution.block(2).size();

    std::vector<IndexSet> owned_partitioning(3);
    owned_partitioning[0] = locally_owned_dofs.get_view(0, n_u);
    owned_partitioning[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
    owned_partitioning[2] = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

    std::vector<IndexSet> relevant_partitioning(3);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
    relevant_partitioning[2] = locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);
    
    relevant_solution.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
 
    relevant_solution = solution;
 
    const FEValuesExtractors::Vector velocities(0);
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      face_q_collection,
      std::map<types::boundary_id, const Function<dim> *>(),
      relevant_solution,
      stokes_estimated_error_per_cell,
      fe_collection.component_mask(velocities));
 
    const FEValuesExtractors::Vector displacements(dim + 1);
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      face_q_collection,
      std::map<types::boundary_id, const Function<dim> *>(),
      relevant_solution,
      elasticity_estimated_error_per_cell,
      fe_collection.component_mask(displacements));
 
    stokes_estimated_error_per_cell *=
      4.0f / stokes_estimated_error_per_cell.l2_norm();
    elasticity_estimated_error_per_cell *=
      1.0f / elasticity_estimated_error_per_cell.l2_norm();

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
 
    estimated_error_per_cell += stokes_estimated_error_per_cell;
    estimated_error_per_cell += elasticity_estimated_error_per_cell;

    double local_error_norm = estimated_error_per_cell.l2_norm();

    local_error_norm = local_error_norm * local_error_norm; 
    double global_error_norm = Utilities::MPI::sum(local_error_norm, mpi_communicator);
    global_error_norm = std::sqrt(global_error_norm);

    if(global_error_norm > tol)
    {
      // usavo solo GridRefinement::refine_and_coarsen_fixed_number, ma ogni processore sceglie il suo 30% di celle da raffinare
      // -> ingestibile, quindi lo rendo parallelo con parallel::distributed
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                      estimated_error_per_cell,
                                                      0.3,
                                                      0.0);
      triangulation.execute_coarsening_and_refinement();
    }

    return global_error_norm;
  }
 
  template <int dim>
  void FluidStructureProblem<dim>::run(const unsigned int max_cycles, const float tol)
  {
    double estimated_error_norm = 0.0f;
    make_grid();
 
    for (unsigned int refinement_cycle = 0; refinement_cycle < max_cycles;
         ++refinement_cycle)
      {
        pcout << "Refinement cycle " << refinement_cycle << std::endl;
 
        if (refinement_cycle > 0)
          estimated_error_norm = refine_mesh(tol);
 
        setup_dofs();
 
        pcout << "   Assembling..." << std::endl;
        assemble_system();
 
        pcout << "   Solving..." << std::endl;
        solve();
        pcout << "   Writing output..." << std::endl;
        output_results(refinement_cycle);

        const unsigned int n_u = solution.block(0).size();
        const unsigned int n_p = solution.block(1).size();
        const unsigned int n_d = solution.block(2).size();

        std::vector<IndexSet> owned_partitioning(3);
        owned_partitioning[0] = locally_owned_dofs.get_view(0, n_u);
        owned_partitioning[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
        owned_partitioning[2] = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

        std::vector<IndexSet> relevant_partitioning(3);
        relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
        relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
        relevant_partitioning[2] = locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

        TrilinosWrappers::MPI::BlockVector relevant_solution;
        relevant_solution.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
        relevant_solution = solution;

        Point<dim> upper_right_solid_corner(0.25 - 1e-12, 0.5 - 1e-12);
        Vector<double> local_value(fe_collection.n_components());
        local_value = 0.0;

        try {
          VectorTools::point_value(dof_handler, relevant_solution, upper_right_solid_corner, local_value);
        } catch (const VectorTools::ExcPointNotAvailableHere &) {
        }

        Vector<double> global_value(fe_collection.n_components());
        global_value = 0.0;

        for (unsigned int i = 0; i < fe_collection.n_components(); ++i) {
          global_value[i] = Utilities::MPI::sum(local_value[i], mpi_communicator);
        }

        pcout << "   [Result logs] Displacement at (0.25, 0.5), upper right corner of the solid: ";
        for(unsigned int i = dim + 1; i < fe_collection.n_components(); ++i) {
          pcout << global_value[i] << " ";
        }
        pcout << std::endl;
      }
  }
} // namespace Step46
 
 
int main(int argc, char *argv[])
{
  try
    {
      using namespace Step46;

      // Initialize MPI (required for Trilinos)
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1); // 1 thread per MPI process
 
      FluidStructureProblem<2> flow_problem(1, 1);
      flow_problem.run(5, 1e-4f);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}