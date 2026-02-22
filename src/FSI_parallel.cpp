#include "FSI_parallel.hpp"

// -----------------------------------------------------------------------
// FSIPreconditioner
// -----------------------------------------------------------------------
void FSIPreconditioner::initialize(
const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
const TrilinosWrappers::SparseMatrix &pressure_mass_,
const TrilinosWrappers::SparseMatrix &displacement_stiffness_,
const TrilinosWrappers::SparseMatrix &B_,
const TrilinosWrappers::SparseMatrix &Cu_,
const TrilinosWrappers::SparseMatrix &Cp_,
const std::vector<std::vector<bool>> &displacement_constant_modes)
{
velocity_stiffness     = &velocity_stiffness_;
pressure_mass          = &pressure_mass_;
displacement_stiffness = &displacement_stiffness_;
B  = &B_;
Cu = &Cu_;
Cp = &Cp_;

TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_vel;
amg_data_vel.elliptic              = true;
amg_data_vel.higher_order_elements = true;
amg_data_vel.smoother_sweeps       = 2;
amg_data_vel.aggregation_threshold = 1e-3;
preconditioner_velocity.initialize(velocity_stiffness_, amg_data_vel);

TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
preconditioner_pressure.initialize(pressure_mass_, ilu_data);

TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_disp;
amg_data_disp.constant_modes        = displacement_constant_modes;
amg_data_disp.elliptic              = true;
amg_data_disp.higher_order_elements = false;
amg_data_disp.smoother_sweeps       = 3;
amg_data_disp.aggregation_threshold = 1e-3;
preconditioner_displacement.initialize(displacement_stiffness_, amg_data_disp);

tmp_p.reinit(pressure_mass_.locally_owned_domain_indices(), MPI_COMM_WORLD);
tmp_d.reinit(displacement_stiffness_.locally_owned_domain_indices(), MPI_COMM_WORLD);
intermediate_tmp.reinit(displacement_stiffness_.locally_owned_domain_indices(), MPI_COMM_WORLD);
}

void FSIPreconditioner::vmult(TrilinosWrappers::MPI::BlockVector       &dst,
                            const TrilinosWrappers::MPI::BlockVector &src) const
{
// Solve fluid velocity
{
    SolverControl solver_control(2000, 1e-2 * src.block(0).l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
    dst.block(0) = 0;
    solver.solve(*velocity_stiffness, dst.block(0), src.block(0),
                preconditioner_velocity);
}

// Solve fluid pressure
B->vmult(tmp_p, dst.block(0));
tmp_p.sadd(-1.0, 1.0, src.block(1));
{
    SolverControl solver_control(2000, 1e-2 * tmp_p.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    dst.block(1) = 0;
    solver.solve(*pressure_mass, dst.block(1), tmp_p, preconditioner_pressure);
}

// Solve displacement
Cu->vmult(tmp_d, dst.block(0));
Cp->vmult(intermediate_tmp, dst.block(1));
tmp_d.add(1.0, intermediate_tmp);
tmp_d.sadd(-1.0, 1.0, src.block(2));
{
    SolverControl solver_control(2000, 1e-2 * tmp_d.l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
    dst.block(2) = 0;
    solver.solve(*displacement_stiffness, dst.block(2), tmp_d,
                preconditioner_displacement);
}
}

// -----------------------------------------------------------------------
// StokesBoundaryValues
// -----------------------------------------------------------------------
template <int dim>
double StokesBoundaryValues<dim>::value(const Point<dim>  &p,
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
                                            Vector<double>   &values) const
{
for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = StokesBoundaryValues<dim>::value(p, c);
}

// -----------------------------------------------------------------------
// FluidStructureProblem
// -----------------------------------------------------------------------
template <int dim>
FluidStructureProblem<dim>::FluidStructureProblem(
const unsigned int stokes_degree,
const unsigned int elasticity_degree)
: stokes_degree(stokes_degree)
, elasticity_degree(elasticity_degree)
, pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
, triangulation(MPI_COMM_WORLD,
                Triangulation<dim>::limit_level_difference_at_vertices)
, stokes_fe(FE_Q<dim>(stokes_degree + 1), dim,
            FE_Q<dim>(stokes_degree),     1,
            FE_Nothing<dim>(),            dim)
, elasticity_fe(FE_Nothing<dim>(),          dim,
                FE_Nothing<dim>(),          1,
                FE_Q<dim>(elasticity_degree), dim)
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
    if (!cell->is_locally_owned()) continue;
    if (cell_is_in_fluid_domain(cell))
        cell->set_active_fe_index(0);
    else if (cell_is_in_solid_domain(cell))
        cell->set_active_fe_index(1);
    else
        Assert(false, ExcNotImplemented());
    }
}

template <int dim>
void FluidStructureProblem<dim>::setup_dofs()
{
set_active_fe_indices();
dof_handler.distribute_dofs(fe_collection);

std::vector<unsigned int> block_component(dim + 1 + dim);
for (unsigned int d = 0; d < dim; ++d)
    block_component[d] = 0;
block_component[dim] = 1;
for (unsigned int d = 0; d < dim; ++d)
    block_component[dim + 1 + d] = 2;

DoFRenumbering::component_wise(dof_handler, block_component);

locally_owned_dofs    = dof_handler.locally_owned_dofs();
locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
const unsigned int n_u = dofs_per_block[0];
const unsigned int n_p = dofs_per_block[1];
const unsigned int n_d = dofs_per_block[2];

block_owned_dofs.resize(3);
block_relevant_dofs.resize(3);
block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
block_owned_dofs[2]    = locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_d);
block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
block_relevant_dofs[2] = locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_d);

pcout << "   DoFs: velocity=" << dofs_per_block[0]
        << ", pressure=" << dofs_per_block[1]
        << ", solid=" << dofs_per_block[2] << std::endl;

// --- Constraints ---
{
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    const FEValuesExtractors::Vector velocities(0);
    VectorTools::interpolate_boundary_values(
    dof_handler, 1, StokesBoundaryValues<dim>(), constraints,
    fe_collection.component_mask(velocities));

    const FEValuesExtractors::Vector displacements(dim + 1);
    VectorTools::interpolate_boundary_values(
    dof_handler, 0,
    Functions::ZeroFunction<dim>(dim + 1 + dim), constraints,
    fe_collection.component_mask(displacements));
}

// --- Interface constraints ---
{
    std::vector<types::global_dof_index> local_face_dof_indices(
    stokes_fe.n_dofs_per_face());
    for (const auto &cell : dof_handler.active_cell_iterators())
    if (!cell->is_artificial() && cell_is_in_fluid_domain(cell))
        for (const auto face_no : cell->face_indices())
        if (!cell->face(face_no)->at_boundary())
            {
            bool face_is_on_interface = false;

            if (!cell->neighbor(face_no)->has_children() &&
                cell_is_in_solid_domain(cell->neighbor(face_no)))
                face_is_on_interface = true;
            else if (cell->neighbor(face_no)->has_children())
                for (unsigned int sf = 0;
                    sf < cell->face(face_no)->n_children(); ++sf)
                if (cell_is_in_solid_domain(
                        cell->neighbor_child_on_subface(face_no, sf)))
                    {
                    face_is_on_interface = true;
                    break;
                    }

            if (face_is_on_interface)
                {
                cell->face(face_no)->get_dof_indices(
                    local_face_dof_indices, 0);
                for (unsigned int i = 0; i < local_face_dof_indices.size(); ++i)
                    if (stokes_fe.face_system_to_component_index(i).first < dim)
                    constraints.add_line(local_face_dof_indices[i]);
                }
            }
}

constraints.close();

pcout << "   Number of active cells: "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

// --- Sparsity patterns ---
{
    TrilinosWrappers::BlockSparsityPattern dsp(
    block_owned_dofs, block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);

    Table<2, DoFTools::Coupling> cell_coupling(fe_collection.n_components(),
                                                fe_collection.n_components());
    Table<2, DoFTools::Coupling> face_coupling(fe_collection.n_components(),
                                                fe_collection.n_components());

    for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
    for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
        {
        if (((c < dim + 1) && (d < dim + 1) &&
                !((c == dim) && (d == dim))) ||
            ((c >= dim + 1) && (d >= dim + 1)))
            cell_coupling[c][d] = DoFTools::always;
        if ((c >= dim + 1) && (d < dim + 1))
            face_coupling[c][d] = DoFTools::always;
        }
    cell_coupling[dim][dim] = DoFTools::always;

    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp, constraints, true,
                                        cell_coupling, face_coupling,
                                        numbers::invalid_subdomain_id);

    // Manually add interface entries for solid->fluid off-diagonal blocks
    {
    std::vector<types::global_dof_index> local_dofs, neighbor_dofs;
    for (const auto &cell : dof_handler.active_cell_iterators())
        {
        if (!cell->is_locally_owned()) continue;
        if (!cell_is_in_solid_domain(cell)) continue;

        local_dofs.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);

        for (const auto f : cell->face_indices())
            {
            if (cell->face(f)->at_boundary()) continue;
            if (cell->neighbor(f)->has_children())
                {
                for (unsigned int sub = 0;
                        sub < cell->face(f)->n_children(); ++sub)
                    {
                    auto child = cell->neighbor_child_on_subface(f, sub);
                    if (cell_is_in_fluid_domain(child))
                        {
                        neighbor_dofs.resize(child->get_fe().n_dofs_per_cell());
                        child->get_dof_indices(neighbor_dofs);
                        constraints.add_entries_local_to_global(
                            local_dofs, neighbor_dofs, dsp);
                        }
                    }
                }
            else if (cell_is_in_fluid_domain(cell->neighbor(f)) ||
                        (cell->neighbor_is_coarser(f) &&
                        cell_is_in_fluid_domain(cell->neighbor(f))))
                {
                neighbor_dofs.resize(
                    cell->neighbor(f)->get_fe().n_dofs_per_cell());
                cell->neighbor(f)->get_dof_indices(neighbor_dofs);
                constraints.add_entries_local_to_global(
                    local_dofs, neighbor_dofs, dsp);
                }
            }
        }
    }
    dsp.compress();

    // Pressure mass matrix sparsity
    Table<2, DoFTools::Coupling> mass_coupling(dim + 1 + dim, dim + 1 + dim);
    for (unsigned int c = 0; c < dim + 1 + dim; ++c)
    for (unsigned int d = 0; d < dim + 1 + dim; ++d)
        mass_coupling[c][d] = (c == dim && d == dim) ? DoFTools::always
                                                    : DoFTools::none;

    TrilinosWrappers::BlockSparsityPattern dsp_mass(
    block_owned_dofs, block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, mass_coupling, dsp_mass,
                                    constraints, true,
                                    numbers::invalid_subdomain_id);
    dsp_mass.compress();

    system_matrix.reinit(dsp);
    mass_matrix.reinit(dsp_mass);
}

system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
}

template <int dim>
void FluidStructureProblem<dim>::assemble_system()
{
system_matrix = 0;
mass_matrix   = 0;
system_rhs    = 0;

const QGauss<dim> stokes_quadrature(stokes_degree + 2);
const QGauss<dim> elasticity_quadrature(elasticity_degree + 2);

hp::QCollection<dim> q_collection;
q_collection.push_back(stokes_quadrature);
q_collection.push_back(elasticity_quadrature);

hp::FEValues<dim> hp_fe_values(fe_collection, q_collection,
                                update_values | update_quadrature_points |
                                    update_JxW_values | update_gradients);

const QGauss<dim - 1> common_face_quadrature(
    std::max(stokes_degree + 2, elasticity_degree + 2));

FEFaceValues<dim> stokes_fe_face_values(stokes_fe, common_face_quadrature,
                                        update_JxW_values |
                                            update_gradients | update_values);
FEFaceValues<dim> elasticity_fe_face_values(elasticity_fe,
                                            common_face_quadrature,
                                            update_normal_vectors |
                                                update_values);
FESubfaceValues<dim> stokes_fe_subface_values(stokes_fe,
                                                common_face_quadrature,
                                                update_JxW_values |
                                                update_gradients |
                                                update_values);
FESubfaceValues<dim> elasticity_fe_subface_values(
    elasticity_fe, common_face_quadrature,
    update_normal_vectors | update_values);

const unsigned int stokes_dofs_per_cell     = stokes_fe.n_dofs_per_cell();
const unsigned int elasticity_dofs_per_cell = elasticity_fe.n_dofs_per_cell();

FullMatrix<double> local_matrix;
FullMatrix<double> local_interface_matrix(elasticity_dofs_per_cell,
                                            stokes_dofs_per_cell);
FullMatrix<double> local_pressure_mass(stokes_dofs_per_cell,
                                        stokes_dofs_per_cell);
Vector<double> local_rhs;

std::vector<types::global_dof_index> local_dof_indices;
std::vector<types::global_dof_index> neighbor_dof_indices(stokes_dofs_per_cell);

const Functions::ZeroFunction<dim> right_hand_side(dim + 1);

const FEValuesExtractors::Vector velocities(0);
const FEValuesExtractors::Scalar pressure(dim);
const FEValuesExtractors::Vector displacements(dim + 1);

std::vector<SymmetricTensor<2, dim>> stokes_symgrad_phi_u(stokes_dofs_per_cell);
std::vector<double>                  stokes_div_phi_u(stokes_dofs_per_cell);
std::vector<double>                  stokes_phi_p(stokes_dofs_per_cell);

std::vector<Tensor<2, dim>> elasticity_grad_phi(elasticity_dofs_per_cell);
std::vector<double>         elasticity_div_phi(elasticity_dofs_per_cell);
std::vector<Tensor<1, dim>> elasticity_phi(elasticity_dofs_per_cell);

for (const auto &cell : dof_handler.active_cell_iterators())
    {
    if (!cell->is_locally_owned()) continue;
    hp_fe_values.reinit(cell);

    const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

    local_matrix.reinit(cell->get_fe().n_dofs_per_cell(),
                        cell->get_fe().n_dofs_per_cell());
    local_rhs.reinit(cell->get_fe().n_dofs_per_cell());
    local_pressure_mass.reinit(cell->get_fe().n_dofs_per_cell(),
                                cell->get_fe().n_dofs_per_cell());
    local_matrix        = 0;
    local_rhs           = 0;
    local_pressure_mass = 0;

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

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                const unsigned int comp_i =
                    fe_values.get_fe().system_to_component_index(i).first;
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                    const unsigned int comp_j =
                        fe_values.get_fe().system_to_component_index(j).first;
                    local_matrix(i, j) +=
                        (2 * viscosity * stokes_symgrad_phi_u[i] *
                            stokes_symgrad_phi_u[j] -
                        stokes_div_phi_u[i] * stokes_phi_p[j] -
                        stokes_phi_p[i] * stokes_div_phi_u[j]) *
                        fe_values.JxW(q);

                    if (comp_i == dim && comp_j == dim)
                        local_pressure_mass(i, j) +=
                        (1.0 / viscosity) * stokes_phi_p[i] *
                        stokes_phi_p[j] * fe_values.JxW(q);
                    }
                }
            }
        }
    else
        {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
        Assert(dofs_per_cell == elasticity_dofs_per_cell, ExcInternalError());

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
                local_matrix(i, j) +=
                    (lambda * elasticity_div_phi[i] * elasticity_div_phi[j] +
                    mu * scalar_product(elasticity_grad_phi[i],
                                        elasticity_grad_phi[j]) +
                    mu * scalar_product(elasticity_grad_phi[i],
                                        transpose(elasticity_grad_phi[j]))) *
                    fe_values.JxW(q);
            }
        }

    local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(local_matrix, local_rhs,
                                            local_dof_indices, system_matrix,
                                            system_rhs);

    if (cell_is_in_fluid_domain(cell))
        {
        std::vector<types::global_dof_index> pressure_global_indices;
        std::vector<unsigned int>            pressure_local_indices;
        for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
            if (cell->get_fe().system_to_component_index(i).first == dim)
            {
                pressure_local_indices.push_back(i);
                pressure_global_indices.push_back(local_dof_indices[i]);
            }
        FullMatrix<double> pressure_submatrix(pressure_local_indices.size(),
                                                pressure_local_indices.size());
        for (unsigned int i = 0; i < pressure_local_indices.size(); ++i)
            for (unsigned int j = 0; j < pressure_local_indices.size(); ++j)
            pressure_submatrix(i, j) =
                local_pressure_mass(pressure_local_indices[i],
                                    pressure_local_indices[j]);
        constraints.distribute_local_to_global(
            pressure_submatrix, pressure_global_indices, mass_matrix);
        }

    if (cell_is_in_solid_domain(cell))
        for (const auto f : cell->face_indices())
        if (!cell->face(f)->at_boundary())
            {
            if ((cell->neighbor(f)->level() == cell->level()) &&
                !cell->neighbor(f)->has_children() &&
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
                    local_interface_matrix, local_dof_indices,
                    neighbor_dof_indices, system_matrix);
                }
            else if ((cell->neighbor(f)->level() == cell->level()) &&
                        cell->neighbor(f)->has_children())
                {
                for (unsigned int subface = 0;
                        subface < cell->face(f)->n_children(); ++subface)
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
                        local_interface_matrix, local_dof_indices,
                        neighbor_dof_indices, system_matrix);
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
                    local_interface_matrix, local_dof_indices,
                    neighbor_dof_indices, system_matrix);
                }
            }
    }

system_matrix.compress(VectorOperation::add);
system_rhs.compress(VectorOperation::add);
mass_matrix.compress(VectorOperation::add);
}

template <int dim>
void FluidStructureProblem<dim>::assemble_interface_term(
const FEFaceValuesBase<dim>          &elasticity_fe_face_values,
const FEFaceValuesBase<dim>          &stokes_fe_face_values,
std::vector<Tensor<1, dim>>          &elasticity_phi,
std::vector<SymmetricTensor<2, dim>> &stokes_symgrad_phi_u,
std::vector<double>                  &stokes_phi_p,
FullMatrix<double>                   &local_interface_matrix) const
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
    for (unsigned int k = 0; k < elasticity_fe_face_values.dofs_per_cell; ++k)
        elasticity_phi[k] =
        elasticity_fe_face_values[displacements].value(k, q);

    for (unsigned int i = 0; i < elasticity_fe_face_values.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < stokes_fe_face_values.dofs_per_cell; ++j)
        local_interface_matrix(i, j) +=
            -((2 * viscosity *
                (stokes_symgrad_phi_u[j] * normal_vector) -
                stokes_phi_p[j] * normal_vector) *
            elasticity_phi[i] * stokes_fe_face_values.JxW(q));
    }
}

template <int dim>
void FluidStructureProblem<dim>::solve()
{
FSIPreconditioner preconditioner;
preconditioner.initialize(system_matrix.block(0, 0),
                            mass_matrix.block(1, 1),
                            system_matrix.block(2, 2),
                            system_matrix.block(1, 0),
                            system_matrix.block(2, 0),
                            system_matrix.block(2, 1),
                            {});

SolverControl solver_control(5000, 1e-8 * system_rhs.l2_norm());
SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
constraints.distribute(solution_owned);
solution = solution_owned;

pcout << "   Converged in " << solver_control.last_step()
        << " iterations." << std::endl;
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

DataOut<dim> data_out;
data_out.attach_dof_handler(dof_handler);
data_out.add_data_vector(solution, solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);

std::vector<unsigned int> partition_int(triangulation.n_active_cells());
GridTools::get_subdomain_association(triangulation, partition_int);
const Vector<double> partitioning(partition_int.begin(), partition_int.end());
data_out.add_data_vector(partitioning, "partitioning");

data_out.build_patches();
data_out.write_vtu_with_pvtu_record("./", "solution-fsi", refinement_cycle,
                                    MPI_COMM_WORLD);
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

const FEValuesExtractors::Vector velocities(0);
KellyErrorEstimator<dim>::estimate(
    dof_handler, face_q_collection,
    std::map<types::boundary_id, const Function<dim> *>(), solution,
    stokes_estimated_error_per_cell,
    fe_collection.component_mask(velocities));

const FEValuesExtractors::Vector displacements(dim + 1);
KellyErrorEstimator<dim>::estimate(
    dof_handler, face_q_collection,
    std::map<types::boundary_id, const Function<dim> *>(), solution,
    elasticity_estimated_error_per_cell,
    fe_collection.component_mask(displacements));

// Scale both error vectors by their global norms
auto scale_globally = [](Vector<float> &v, float factor) {
    float local_sq  = v.norm_sqr();
    float global_sq = 0.0f;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    v *= factor / std::sqrt(global_sq);
};
scale_globally(stokes_estimated_error_per_cell, 4.0f);
scale_globally(elasticity_estimated_error_per_cell, 1.0f);

Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
estimated_error_per_cell += stokes_estimated_error_per_cell;
estimated_error_per_cell += elasticity_estimated_error_per_cell;

// Zero out interface cells
for (const auto &cell : dof_handler.active_cell_iterators())
    {
    if (!cell->is_locally_owned()) continue;
    for (const auto f : cell->face_indices())
        {
        if (cell->at_boundary(f)) continue;
        auto check_neighbor = [&](bool this_is_solid) {
            auto is_other = [&](const typename DoFHandler<dim>::cell_iterator &c) {
            return this_is_solid ? cell_is_in_fluid_domain(c)
                                    : cell_is_in_solid_domain(c);
            };
            if ((cell->neighbor(f)->level() == cell->level() &&
                !cell->neighbor(f)->has_children() &&
                is_other(cell->neighbor(f))) ||
                (cell->neighbor(f)->level() == cell->level() &&
                cell->neighbor(f)->has_children() &&
                is_other(cell->neighbor_child_on_subface(f, 0))) ||
                (cell->neighbor_is_coarser(f) && is_other(cell->neighbor(f))))
            estimated_error_per_cell(cell->active_cell_index()) = 0;
        };
        if (cell_is_in_solid_domain(cell))
            check_neighbor(true);
        else
            check_neighbor(false);
        }
    }

float local_norm_sq  = estimated_error_per_cell.norm_sqr();
float global_norm_sq = 0.0f;
MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
float estimated_error_norm = std::sqrt(global_norm_sq);

if (estimated_error_norm > tol)
    {
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.0);
    triangulation.execute_coarsening_and_refinement();
    }

return estimated_error_norm;
}

template <int dim>
void FluidStructureProblem<dim>::run(const unsigned int max_cycles,
                                    const float        tol)
{
float estimated_error_norm = 0.0f;
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

    Point<dim>     upper_right_solid_corner(0.25 - 1e-12, 0.5 - 1e-12);
    Vector<double> local_value(fe_collection.n_components());
    bool           found_locally = false;
    try
        {
        VectorTools::point_value(dof_handler, solution,
                                    upper_right_solid_corner, local_value);
        found_locally = true;
        }
    catch (const VectorTools::ExcPointNotAvailableHere &)
        {}

    Vector<double> global_value(fe_collection.n_components());
    for (unsigned int i = 0; i < fe_collection.n_components(); ++i)
        {
        double local_entry  = found_locally ? local_value[i] : 0.0;
        double global_entry = 0.0;
        MPI_Allreduce(&local_entry, &global_entry, 1, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD);
        global_value[i] = global_entry;
        }

    pcout << "   [Result logs] Displacement at (0.25, 0.5), "
            << "upper right corner of the solid: [";
    for (unsigned int i = dim + 1; i < fe_collection.n_components(); ++i)
        pcout << global_value[i]
            << (i < fe_collection.n_components() - 1 ? ", " : "");
    pcout << "]" << std::endl;

    if (refinement_cycle > 0)
        {
        pcout << "   [Result logs] Estimated error per cell norm: "
                << estimated_error_norm << std::endl;
        if (estimated_error_norm <= tol)
            break;
        }
    pcout << std::endl;
    }
}

// Explicit instantiation for dim=2
template class StokesBoundaryValues<2>;
template class FluidStructureProblem<2>;
// template class StokesBoundaryValues<3>;
// template class FluidStructureProblem<3>;