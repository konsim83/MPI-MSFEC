#include "helmholtz_basis.h"

namespace HelmholtzProblem
{

using namespace dealii;

NedRTBasis::Parameters::Parameters(const NedRTMultiscale::Parameters &parameters)
:
verbose (parameters.verbose_basis),
use_direct_solver (parameters.use_direct_solver_basis),
renumber_dofs (parameters.renumber_dofs),
output_flag(false),
n_refine_global (parameters.n_refine_global),
n_refine_local (parameters.n_refine_local),
filename_global (parameters.filename_output)
{
}



NedRTBasis::Parameters::Parameters(const Parameters &other)
:
verbose (other.verbose),
use_direct_solver (other.use_direct_solver),
renumber_dofs (other.renumber_dofs),
output_flag(other.output_flag),
n_refine_global (other.n_refine_global),
n_refine_local (other.n_refine_local),
filename_global (other.filename_global)
{
}



NedRTBasis::NedRTBasis (const NedRTMultiscale::Parameters &parameters_,
		typename Triangulation<3>::active_cell_iterator& global_cell,
		unsigned int local_subdomain,
		MPI_Comm mpi_communicator)
:
mpi_communicator(mpi_communicator),
parameters(parameters_),
triangulation(),
fe (FE_Nedelec<3>(parameters.degree), 1,
		FE_RaviartThomas<3>(parameters.degree), 1),
dof_handler (triangulation),
constraints_curl_v(GeometryInfo<3>::lines_per_cell),
constraints_div_v(GeometryInfo<3>::faces_per_cell),
sparsity_pattern_curl(),
sparsity_pattern_div(),
basis_curl_v(GeometryInfo<3>::lines_per_cell),
basis_div_v(GeometryInfo<3>::faces_per_cell),
system_rhs_curl_v(GeometryInfo<3>::lines_per_cell),
system_rhs_div_v(GeometryInfo<3>::faces_per_cell),
global_element_matrix(fe.dofs_per_cell,
		fe.dofs_per_cell),
global_element_rhs(fe.dofs_per_cell),
global_weights(fe.dofs_per_cell, 0),
global_cell_id (global_cell->id()),
local_subdomain(local_subdomain),
volume_measure(0),
face_measure(GeometryInfo<3>::faces_per_cell, 0),
edge_measure(GeometryInfo<3>::lines_per_cell, 0),
corner_points(GeometryInfo<3>::vertices_per_cell,
		Point<3>()),
length_system_basis(GeometryInfo<3>::lines_per_cell
		+ GeometryInfo<3>::faces_per_cell),
is_built_global_element_matrix(false),
is_set_global_weights(false),
is_set_cell_data(false),
is_copyable (true)
{
	global_cell_ptr = std::make_shared<typename Triangulation<3>::active_cell_iterator> (&global_cell);
	set_cell_data ();
}



NedRTBasis::NedRTBasis(const NedRTBasis &other)
:
// guard against copying non copyable objects
//Assert (is_copyable,
//		ExcMessage ("Object can not be copied after triangulation and other parts are initialized.")),
mpi_communicator (other.mpi_communicator),
parameters (other.parameters),
triangulation (), // must be constructed deliberately, but is empty on copying anyway
fe (other.fe),
dof_handler (triangulation),
constraints_curl_v (other.constraints_curl_v),
constraints_div_v (other.constraints_div_v),
sparsity_pattern_curl (other.sparsity_pattern_curl), // only possible if object is empty
sparsity_pattern_div (other.sparsity_pattern_div), // only possible if object is empty
assembled_matrix (other.assembled_matrix), // only possible if object is empty
system_matrix (other.system_matrix), // only possible if object is empty
basis_curl_v (other.basis_curl_v),
basis_div_v (other.basis_div_v),
system_rhs_curl_v (other.system_rhs_curl_v),
system_rhs_div_v (other.system_rhs_div_v),
global_rhs (other.global_rhs),
global_element_matrix (other.global_element_matrix),
global_element_rhs (other.global_element_rhs),
global_weights (other.global_weights),
global_solution (other.global_solution),
inner_schur_preconditioner (other.inner_schur_preconditioner),
global_cell_id (other.global_cell_id),
global_cell_ptr(other.global_cell_ptr),
local_subdomain (other.local_subdomain),
volume_measure (other.volume_measure),
face_measure (other.face_measure),
edge_measure (other.edge_measure),
corner_points (other.corner_points),
length_system_basis (other.length_system_basis),
is_built_global_element_matrix (other.is_built_global_element_matrix),
is_set_global_weights (other.is_set_global_weights),
is_set_cell_data (other.is_set_cell_data),
is_copyable (other.is_copyable)
{
	set_cell_data ();
}



NedRTBasis::~NedRTBasis ()
{
	system_matrix.clear();

	for (unsigned int n_basis=0; n_basis<basis_curl_v.size(); ++n_basis)
	{
		constraints_curl_v[n_basis].clear();
	}

	for (unsigned int n_basis=0; n_basis<basis_div_v.size(); ++n_basis)
	{
		constraints_div_v[n_basis].clear();
	}

	dof_handler.clear ();
}



void
NedRTBasis::setup_grid ()
{
	Assert (is_set_cell_data,
			ExcMessage ("Cell data must be set first."));

	GridGenerator::general_cell(triangulation, corner_points, /* colorize faces */false);

	triangulation.refine_global (parameters.n_refine_local);

	is_copyable = false;
}



void
NedRTBasis::setup_system_matrix ()
{
	dof_handler.distribute_dofs (fe);

	if (parameters.renumber_dofs)
	{
		DoFRenumbering::Cuthill_McKee (dof_handler);
	}

	DoFRenumbering::block_wise (dof_handler);

	std::vector<types::global_dof_index> dofs_per_block (2);
	DoFTools::count_dofs_per_block (dof_handler, dofs_per_block);
	const unsigned int n_sigma = dofs_per_block[0],
					   n_u = dofs_per_block[1];

	if (parameters.verbose)
	{
	  std::cout << "Number of active cells: "
				  << triangulation.n_active_cells()
				  << std::endl
				  << "Total number of cells: "
				  << triangulation.n_cells()
				  << std::endl
				  << "Number of degrees of freedom: "
				  << dof_handler.n_dofs()
				  << " (" << n_sigma << '+' << n_u << ')'
				  << std::endl;
	}

	BlockSparsityPattern     sparsity_pattern;

	{
		// Allocate memory
		BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

		// Initialize the system matrix for global assembly
		sparsity_pattern.copy_from(dsp);
	}

	assembled_matrix.reinit (sparsity_pattern);

	global_solution.reinit (dofs_per_block);
}



void
NedRTBasis::setup_basis_dofs_curl ()
{
	Assert (is_set_cell_data,
					ExcMessage ("Cell data must be set first."));

	Timer timer;

	if (parameters.verbose)
	{
		std::cout << "Setting up dofs for H(curl) part....." << std::endl;

		timer.start ();
	}

	ShapeFun::ShapeFunctionVector<3>
			std_shape_function (fe.base_element(0),
					*global_cell_ptr,
					/*verbose =*/ false);
	ShapeFun::ShapeFunctionVectorCurl<3>
			std_shape_function_curl (fe.base_element(0),
					*global_cell_ptr,
					/*verbose =*/ false);

	std::vector<types::global_dof_index> dofs_per_block (2);
	DoFTools::count_dofs_per_block (dof_handler, dofs_per_block);
	const unsigned int n_sigma = dofs_per_block[0],
					   n_u = dofs_per_block[1];

	// Allocate memory
	BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

	// set constraints (first hanging nodes, then boundary conditions)
	for (unsigned int n_basis=0; n_basis<basis_curl_v.size(); ++n_basis)
	{
		std_shape_function.set_shape_fun_index(n_basis);
		std_shape_function_curl.set_shape_fun_index(n_basis);

		constraints_curl_v[n_basis].clear ();

		DoFTools::make_hanging_node_constraints (dof_handler, constraints_curl_v[n_basis]);

		VectorTools::project_boundary_values_curl_conforming(dof_handler,
					/*first vector component */ 0,
					std_shape_function,
					/*boundary id*/ 0,
					constraints_curl_v[n_basis]);
		VectorTools::project_boundary_values_div_conforming(dof_handler,
					/*first vector component */ 3,
//								ZeroFunction<3>(3),
					std_shape_function_curl,
					/*boundary id*/ 0,
					constraints_curl_v[n_basis]);

		constraints_curl_v[n_basis].close ();
	}

	DoFTools::make_sparsity_pattern(dof_handler,
									  dsp,
									  constraints_curl_v[0], // do not write into constraint dofs (same dofs for all problems)
									  /*keep_constrained_dofs = */ true); // must condense constraints later

	sparsity_pattern_curl.copy_from(dsp);

	for (unsigned int n_basis=0; n_basis<basis_curl_v.size(); ++n_basis)
	{
		basis_curl_v[n_basis].reinit (dofs_per_block);
		system_rhs_curl_v[n_basis].reinit (dofs_per_block);
	}

	if (parameters.verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
}



void
NedRTBasis::setup_basis_dofs_div ()
{
	Assert (is_set_cell_data,
					ExcMessage ("Cell data must be set first."));

	Timer timer;

	if (parameters.verbose)
	{
		std::cout << "Setting up dofs for H(div) part....." << std::endl;

		timer.start ();
	}

	ShapeFun::ShapeFunctionVector<3>
			std_shape_function (fe.base_element(1),
					*global_cell_ptr,
					/*verbose =*/ false);
//	ShapeFun::ShapeFunctionVectorCurl<3>
//			std_shape_function_curl (fe.base_element(1),
//					*global_cell_ptr,
//					/*verbose =*/ false);

	std::vector<types::global_dof_index> dofs_per_block (2);
	DoFTools::count_dofs_per_block (dof_handler, dofs_per_block);
	const unsigned int n_sigma = dofs_per_block[0],
					   n_u = dofs_per_block[1];

	// Allocate memory
	BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

	for (unsigned int n_basis=0; n_basis<basis_div_v.size(); ++n_basis)
	{
		std_shape_function.set_shape_fun_index(n_basis);
//		std_shape_function_curl.set_shape_fun_index(n_basis);

		// set constraints (first hanging nodes, then flux)
		constraints_div_v[n_basis].clear ();

		DoFTools::make_hanging_node_constraints (dof_handler, constraints_div_v[n_basis]);

//		VectorTools::project_boundary_values_curl_conforming(dof_handler,
//					/*first vector component */ 0,
//					std_shape_function_curl,
//					/*boundary id*/ 0,
//					constraints_div_v[n_basis]);
		VectorTools::project_boundary_values_div_conforming(dof_handler,
					/*first vector component */ 3,
					std_shape_function,
					/*boundary id*/ 0,
					constraints_div_v[n_basis]);

		constraints_div_v[n_basis].close ();
	}

	DoFTools::make_sparsity_pattern(dof_handler,
									  dsp,
									  constraints_div_v[0], // do not write into constraint dofs (same dofs for all problems)
									  /*keep_constrained_dofs = */ true); // must condense constraints later

	sparsity_pattern_div.copy_from(dsp);

	for (unsigned int n_basis=0; n_basis<basis_div_v.size(); ++n_basis)
	{
		basis_div_v[n_basis].reinit (dofs_per_block);
		system_rhs_div_v[n_basis].reinit (dofs_per_block);
	}

	if (parameters.verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
}



void
NedRTBasis::assemble_system ()
{
	Timer timer;
	if (parameters.verbose)
	{
		std::cout << "Assembling local linear system in cell   "
			<< global_cell_id.to_string()
			<< "....." << std::endl;

		timer.start ();
	}
	// Choose appropriate quadrature rules
	QGauss<3>   quadrature_formula(parameters.degree + 2);
    QGauss<2> face_quadrature_formula(parameters.degree + 2);

	// Get relevant quantities to be updated from finite element
	FEValues<3> fe_values (fe, quadrature_formula,
							 update_values    | update_gradients |
							 update_quadrature_points  | update_JxW_values);

    FEFaceValues<3> fe_face_values (fe, face_quadrature_formula,
                                      update_values    | update_normal_vectors |
                                      update_quadrature_points  | update_JxW_values);

	// Define some abbreviations
	const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
	const unsigned int   n_q_points      = quadrature_formula.size();
	const unsigned int   n_face_q_points = face_quadrature_formula.size();

	// Declare local contributions and reserve memory
	FullMatrix<double>   		local_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       		local_rhs (dofs_per_cell);
	std::vector<Vector<double>> local_rhs_v(GeometryInfo<3>::lines_per_cell
											+ GeometryInfo<3>::faces_per_cell,
											Vector<double> (dofs_per_cell));

	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	// Equation data
	const RightHandSide				right_hand_side;
	const DiffusionInverse_A		diffusion_inverse_a;
	const Diffusion_B				diffusion_b;
	const ReactionRate				reaction_rate;

	// Equation data for right-hand side
//	ShapeFun::ShapeFunctionVectorCurl<3>
//				std_shape_function_sigma_curl (fe.base_element(0),
//						*global_cell_ptr,
//						/*verbose =*/ false);
	ShapeFun::ShapeFunctionVectorDiv<3>
				std_shape_function_u_div (fe.base_element(1),
						*global_cell_ptr,
						/*verbose =*/ false);

	// allocate
	std::vector<Tensor<1,3>> 	rhs_values (n_q_points);
	std::vector<double> 		reaction_rate_values (n_q_points);
	std::vector<Tensor<2,3> > 	diffusion_inverse_a_values (n_q_points);
	std::vector<double>		 	diffusion_b_values (n_q_points);
	std::vector<double>		 	diffusion_b_face_values (n_face_q_points);
	std::vector<Tensor<1,3>> 	std_shape_function_sigma_curl_values (n_face_q_points);
	std::vector<double>			std_shape_function_u_div_values (n_face_q_points);

	const FEValuesExtractors::Vector curl (/* first_vector_component */ 0);
	const FEValuesExtractors::Vector flux (/* first_vector_component */ 3);

	// ------------------------------------------------------------------
	// loop over cells
	typename DoFHandler<3>::active_cell_iterator
								cell = dof_handler.begin_active(),
								endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);

		local_matrix = 0;
		local_rhs = 0;

		for (unsigned int n_basis=0;
				n_basis<length_system_basis;
				++n_basis)
		{
			local_rhs_v[n_basis] = 0;
		}

		right_hand_side.value_list (fe_values.get_quadrature_points(),
									rhs_values);
		reaction_rate.value_list(fe_values.get_quadrature_points(),
									reaction_rate_values);
		diffusion_inverse_a.value_list (fe_values.get_quadrature_points(),
							  diffusion_inverse_a_values);
		diffusion_b.value_list (fe_values.get_quadrature_points(),
									  diffusion_b_values);

		// loop over quad points
		for (unsigned int q=0; q<n_q_points; ++q)
		{
			for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				// Test functions
				const Tensor<1,3> 		tau_i = fe_values[curl].value (i, q);
				const Tensor<1,3>      	curl_tau_i = fe_values[curl].curl (i, q);
				const double 	     	div_v_i = fe_values[flux].divergence (i, q);
				const Tensor<1,3>      	v_i = fe_values[flux].value (i, q);

				for (unsigned int j=0; j<dofs_per_cell; ++j)
				{
					// trial functions
					const Tensor<1,3> 		sigma_j = fe_values[curl].value (j, q);
					const Tensor<1,3>      	curl_sigma_j = fe_values[curl].curl (j, q);
					const double 	     	div_u_j = fe_values[flux].divergence (j, q);
					const Tensor<1,3>      	u_j = fe_values[flux].value (j, q);

					/*
					 * Discretize
					 * A^{-1}sigma - curl(u) = 0
					 * curl(sigma) - grad(B*div(u)) + alpha u = f , where alpha>0.
					 */
					local_matrix(i,j) += (tau_i * diffusion_inverse_a_values[q] * sigma_j /* block (0,0) */
											- curl_tau_i * u_j /* block (0,1) */
											+ v_i * curl_sigma_j /* block (1,0) */
											+ div_v_i * diffusion_b_values[q] * div_u_j /* block (1,1) */
											+ v_i * reaction_rate_values[q] * u_j) /* block (1,1) */
											* fe_values.JxW(q);
				} // end for ++j

				// Only for use in global assembly
				local_rhs(i) += v_i *
								rhs_values[q] *
								fe_values.JxW(q);

				// Only for use in local solving.
				for (unsigned int n_basis=0; n_basis<length_system_basis; ++n_basis)
				{
					if (n_basis<GeometryInfo<3>::lines_per_cell)
					{
						// This is rhs for curl.
						local_rhs_v[n_basis](i) += 0;
					}
					else
						// This is rhs for div.
						local_rhs_v[n_basis](i) += 0;
				}
			} // end for ++i
		} // end for ++q

		for (unsigned int face_number=0;
				face_number<GeometryInfo<3>::faces_per_cell;
				++face_number)
		{
			if (cell->face(face_number)->at_boundary()
//				&&
//				(cell->face(face_number)->boundary_id() == 1)
				)
			{
				fe_face_values.reinit (cell, face_number);

				diffusion_b.value_list (fe_face_values.get_quadrature_points(),
										diffusion_b_face_values);

				for (unsigned int n_basis=0; n_basis<length_system_basis; ++n_basis)
				{
					if (n_basis<GeometryInfo<3>::lines_per_cell)
					{
//						// The curl of sigma has natural BCs -(curl sigma)xn
//						std_shape_function_sigma_curl.set_shape_fun_index(n_basis);
//
//						std_shape_function_sigma_curl.tensor_value_list (fe_face_values.get_quadrature_points(),
//																	std_shape_function_sigma_curl_values);
//
//						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
//						{
//							const Tensor<1,3> 	sigma_curl_cross_n = cross_product_3d ( std_shape_function_sigma_curl_values[q_point],
//													           	   	   	   	   fe_face_values.normal_vector(q_point));
//
//							for (unsigned int i=0; i<dofs_per_cell; ++i)
//							{
//								// Note the minus.
//								local_rhs_v[n_basis](i) -= ( sigma_curl_cross_n *
//																fe_face_values[curl].value (i, q_point) *
//																fe_face_values.JxW(q_point));
//								std::cout << local_rhs_v[n_basis](i) << std::endl;
//							}
//						}
					}
					else
					{
						const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;

						std_shape_function_u_div.set_shape_fun_index(offset_index);//
						std_shape_function_u_div.value_list (fe_face_values.get_quadrature_points(),
																	std_shape_function_u_div_values);

						for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
						{
							for (unsigned int i=0; i<dofs_per_cell; ++i)
							{
								// Note the minus.
								local_rhs_v[n_basis](i) += ( fe_face_values.normal_vector(q_point) *
																fe_face_values[flux].value (i, q_point) *
																diffusion_b_face_values[q_point] *
																std_shape_function_u_div_values[q_point] *
																fe_face_values.JxW(q_point));
							}
						}
					}
				} // end n_basis++
			} // end if cell->at_boundary()
		} // end face_number++

		// Only for use in global assembly
		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			global_rhs(local_dof_indices[i]) += local_rhs(i);
		}

		// Add to global matrix
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j=0; j<dofs_per_cell; ++j)
			{
				assembled_matrix.add (local_dof_indices[i],
								   local_dof_indices[j],
								   local_matrix(i,j));
			}

			for (unsigned int n_basis=0;
							n_basis<length_system_basis;
							++n_basis)
			{
				if (n_basis<GeometryInfo<3>::lines_per_cell)
				{
					// This is for curl.
					system_rhs_curl_v[n_basis](local_dof_indices[i]) += local_rhs_v[n_basis];
				}
				else
				{
					// This is for curl.
					const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
					system_rhs_div_v[offset_index](local_dof_indices[i]) += local_rhs_v[n_basis];
				}
			}
		}
		// ------------------------------------------
	}// end for ++cell

	if (parameters.verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
} // end assemble()



void
NedRTBasis::solve_direct (unsigned int n_basis)
{
	Timer timer;
	if (parameters.verbose)
	{
		std::cout << "Solving linear system (directly) in cell   "
					<< global_cell_id.to_string()
					<< "for basis   "
					<< n_basis
					<< "....." << std::endl;

		timer.start ();
	}

	BlockVector<double> *system_rhs_ptr = NULL;
	BlockVector<double> *solution_ptr = NULL;
	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		system_rhs_ptr = &(system_rhs_curl_v[n_basis]);
		solution_ptr = &(basis_curl_v[n_basis]);
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
		system_rhs_ptr = &(system_rhs_div_v[offset_index]);
		solution_ptr = &(basis_div_v[offset_index]);
	}

	// for convenience
	const BlockVector<double> &system_rhs = *system_rhs_ptr;
	BlockVector<double> &solution = *solution_ptr;

	//use direct solver
	SparseDirectUMFPACK A_inv;
	A_inv.initialize(system_matrix);

	A_inv.vmult(solution, system_rhs);

	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		constraints_curl_v[n_basis].distribute(solution);
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;

		constraints_div_v[offset_index].distribute(solution);
	}

	if (parameters.verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
}



void
NedRTBasis::solve_iterative (unsigned int n_basis)
{
	Timer timer;

	// ------------------------------------------
	// Make a preconditioner for each system matrix
	if (parameters.verbose)
	{
		std::cout << "Computing preconditioner in cell   "
			<< global_cell_id.to_string()
			<< "for basis   "
			<< n_basis
			<< "....." << std::endl;

		timer.start ();
	}

	BlockVector<double> *system_rhs_ptr = NULL;
	BlockVector<double> *solution_ptr = NULL;
	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		system_rhs_ptr = &(system_rhs_curl_v[n_basis]);
		solution_ptr = &(basis_curl_v[n_basis]);
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
		system_rhs_ptr = &(system_rhs_div_v[offset_index]);
		solution_ptr = &(basis_div_v[offset_index]);
	}

	// for convenience
	const BlockVector<double> &system_rhs = *system_rhs_ptr;
	BlockVector<double> &solution = *solution_ptr;

	inner_schur_preconditioner = std::make_shared<typename LinearSolvers::LocalInnerPreconditioner<3>::type>();

	typename LinearSolvers::InnerPreconditioner<3>::type::AdditionalData data;
	inner_schur_preconditioner->initialize (system_matrix.block(0,0), data);

	if (parameters.verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
	// ------------------------------------------

	// Now solve.
	if (parameters.verbose)
	{
		std::cout << "Solving linear system (iteratively, with preconditioner) in cell   "
					<< global_cell_id.to_string()
					<< "for basis   "
					<< n_basis
					<< "....." << std::endl;

		timer.start ();
	}

	// Construct inverse of upper left block
	const LinearSolvers::InverseMatrix<SparseMatrix<double>, typename LinearSolvers::LocalInnerPreconditioner<3>::type>
						block_inverse ( system_matrix.block(0,0), *inner_schur_preconditioner );

	Vector<double> tmp (system_rhs.block(0).size());
	{
		// Set up Schur complement
		LinearSolvers::SchurComplement<BlockSparseMatrix<double>,
						BlockVector<double>,
						typename LinearSolvers::InnerPreconditioner<3>::type>
				schur_complement (system_matrix, block_inverse);

		// Compute schur_rhs = -g + C*A^{-1}*f
		Vector<double> schur_rhs (system_rhs.block(1).size());

		block_inverse.vmult (tmp, system_rhs.block(0));
		system_matrix.block(1,0).vmult (schur_rhs, tmp);
		schur_rhs -= system_rhs.block(1);

		{
			SolverControl solver_control (system_matrix.m(),
												1e-6*schur_rhs.l2_norm());
//			SolverCG<BlockVector<double>> schur_solver (solver_control);
			SolverMinRes<BlockVector<double>> schur_solver (solver_control);

			schur_solver.solve (schur_complement,
						solution.block(1),
						schur_rhs,
						PreconditionIdentity());

			if (parameters.verbose)
				std::cout
					<< std::endl
					<< "   Iterative Schur complement solver converged in"
					<< solver_control.last_step()
					<< " iterations."
					<< std::endl;
		}

		{
			// use computed u to solve for sigma
			system_matrix.block(0,1).vmult (tmp, solution.block(1));
			tmp *= -1;
			tmp += system_rhs.block(0);

			// Solve for sigma
			block_inverse.vmult (solution.block(0), tmp);

			if (parameters.verbose)
				std::cout << "   Outer solver completed." << std::endl;
		}
	}

	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		constraints_curl_v[n_basis].distribute(solution);
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;

		constraints_div_v[offset_index].distribute(solution);
	}


	if (parameters.verbose)
	{
		timer.stop ();
		printf("....done (%gs)\n",timer());
	}
}



void
NedRTBasis::assemble_global_element_matrix()
{
	// First, reset.
	global_element_matrix = 0;

	// Get lengths of tmp vectors for assembly
	std::vector<types::global_dof_index> dofs_per_component (3+3);
		DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
		const unsigned int 	n_sigma = dofs_per_component[0],
							n_u = dofs_per_component[3];

	Vector<double>			tmp_u (n_u), tmp_sigma (n_sigma);

	// This assembles the local contribution to the global global matrix
	// with an algebraic trick. It uses the local system matrix stored in
	// the respective basis object.
	unsigned int block_row, block_col;


	BlockVector<double> *test_vec_ptr, *trial_vec_ptr;
	unsigned int offset_index = GeometryInfo<3>::lines_per_cell;

	for (unsigned int i_test=0;
			i_test < length_system_basis;
			++i_test)
	{
		if (i_test<GeometryInfo<3>::lines_per_cell)
		{
			block_row = 0;
			test_vec_ptr = &(basis_curl_v.at(i_test));
		}
		else
		{
			block_row = 1;
			test_vec_ptr = &(basis_div_v.at(i_test-offset_index));
		}

		for (unsigned int i_trial=0;
				i_trial<length_system_basis;
				++i_trial)
		{
			if (i_trial<GeometryInfo<3>::lines_per_cell)
			{
				block_col = 0;
				trial_vec_ptr = &(basis_curl_v.at(i_trial));
			}
			else
			{
				block_col = 1;
				trial_vec_ptr = &(basis_div_v.at(i_trial-offset_index));
			}

			if (block_row==0) /* This means we are testing with sigma. */
			{
				if (block_col==0) /* This means trial function is sigma. */
				{
					assembled_matrix.block(block_row, block_col).vmult(tmp_sigma, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += (test_vec_ptr->block(block_row) * tmp_sigma);
					tmp_sigma = 0;
				}
				if (block_col==1) /* This means trial function is u. */
				{
					assembled_matrix.block(block_row, block_col).vmult(tmp_sigma, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += (test_vec_ptr->block(block_row) * tmp_sigma);
					tmp_sigma = 0;
				}
			} // end if
			else /* This means we are testing with u. */
			{
				if (block_col==0) /* This means trial function is sigma. */
				{
					assembled_matrix.block(block_row, block_col).vmult(tmp_u, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += (test_vec_ptr->block(block_row) * tmp_u);
					tmp_u = 0;
				}
				if (block_col==1) /* This means trial function is u. */
				{
					assembled_matrix.block(block_row, block_col).vmult(tmp_u, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += test_vec_ptr->block(block_row) * tmp_u;
					tmp_u = 0;
				}
			} // end else
		} // end for i_trial

		if (i_test>=GeometryInfo<3>::lines_per_cell)
		{
			block_row = 1;
			// If we are testing with u we possibly have a right-hand side.
			global_element_rhs(i_test) += test_vec_ptr->block(block_row) * global_rhs.block(block_row);
		}
	} // end for i_test

	is_built_global_element_matrix = true;
}



void
NedRTBasis::output_basis (unsigned int n_basis)
{
	Timer timer;
	if (parameters.verbose)
	{
		std::cout << "Writing local solution in cell   "
			<< global_cell_id.to_string()
			<< "for basis   "
			<< n_basis
			<< "....." << std::endl;

		timer.start ();
	}

	BlockVector<double> *basis_ptr = NULL;
	if (n_basis<GeometryInfo<3>::lines_per_cell)
		basis_ptr = &(basis_curl_v[n_basis]);
	else
		basis_ptr = &(basis_div_v.at(n_basis - GeometryInfo<3>::lines_per_cell));

	std::vector<std::string> solution_names(3, "sigma");
	solution_names.push_back ("u");
	solution_names.push_back ("u");
	solution_names.push_back ("u");

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	interpretation (3,
					DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);

	DataOut<3> data_out;
	data_out.add_data_vector (dof_handler,
								*basis_ptr,
								solution_names,
								interpretation);

	data_out.build_patches (parameters.degree + 1);

	std::string filename = "basis_";
	if (n_basis<GeometryInfo<3>::lines_per_cell)
	{
		filename += "curl";
		filename += "." + Utilities::int_to_string(local_subdomain, 5);
		filename += ".cell-" + global_cell_id.to_string();
		filename += ".index-";
		filename += Utilities::int_to_string (n_basis, 2);
	}
	else
	{
		filename += "div-";
		filename += "." + Utilities::int_to_string(local_subdomain, 5);
		filename += ".cell-" + global_cell_id.to_string();
		filename += ".index-";
		filename += Utilities::int_to_string (n_basis - GeometryInfo<3>::lines_per_cell, 2);
	}
	filename += ".vtu";

	std::ofstream output (filename);
	data_out.write_vtu (output);

	if (parameters.verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
}



void
NedRTBasis::output_global_solution_in_cell () const
{
	// Names of solution components
	std::vector<std::string> solution_names(3, "sigma");
	solution_names.push_back ("u");
	solution_names.push_back ("u");
	solution_names.push_back ("u");

	// Interpretation of solution components
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	interpretation (3,
					DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);

	// Build the data out object and add the data
	DataOut<3> data_out;
	data_out.add_data_vector (dof_handler,
			global_solution,
			solution_names,
			interpretation);

	data_out.build_patches ();

	std::ofstream output (parameters.filename_global.c_str());
	data_out.write_vtu (output);
}



void
NedRTBasis::set_output_flag(bool flag)
{
	parameters.output_flag = flag;
}



void
NedRTBasis::set_cell_data ()
{

	global_cell_id = (*global_cell_ptr)->id();

	for (unsigned int vertex_n=0;
		 vertex_n<GeometryInfo<3>::vertices_per_cell;
		 ++vertex_n)
	{
		corner_points.at(vertex_n) = (*global_cell_ptr)->vertex(vertex_n);
	}

	volume_measure = (*global_cell_ptr)->measure ();

	for (unsigned int j_face=0;
			j_face<GeometryInfo<3>::faces_per_cell;
			++j_face)
	{
		face_measure.at(j_face) = (*global_cell_ptr)->face(j_face)->measure ();
	}

	for (unsigned int j_egde=0;
			j_egde<GeometryInfo<3>::lines_per_cell;
			++j_egde)
	{
		edge_measure.at(j_egde) = (*global_cell_ptr)->line(j_egde)->measure ();
	}

	is_set_cell_data = true;
}



void
NedRTBasis::set_global_weights (const std::vector<double> &weights)
{
	// Copy assignment of global weights
	global_weights = weights;

	// reinitialize the global solution on this cell
	global_solution = 0;

	const unsigned int dofs_per_cell_sigma	= fe.base_element(0).n_dofs_per_cell();
	const unsigned int dofs_per_cell_u		= fe.base_element(1).n_dofs_per_cell();

	// First set block 0
	for (unsigned int i=0;
			i<dofs_per_cell_sigma;
			++i)
		global_solution.block(0).sadd (1, global_weights[i], basis_curl_v[i].block(0));

	// Then set block 1
	for (unsigned int i=0;
			i<dofs_per_cell_u;
			++i)
		global_solution.block(1).sadd (1, global_weights[i+dofs_per_cell_sigma], basis_div_v[i].block(1));

	is_set_global_weights = true;
}



void
NedRTBasis::set_sigma_to_std ()
{
	// Quadrature used for projection
	QGauss<3> 	quad_rule (3);

	// Set up vector shape function from finite element on current cell
	ShapeFun::ShapeFunctionVector<3>
		std_shape_function_curl (fe.base_element(0),
				*global_cell_ptr,
				/*verbose =*/ false);

	DoFHandler<3>	dof_handler_fake (triangulation);
	dof_handler_fake.distribute_dofs (fe.base_element(0));

	if (parameters.renumber_dofs)
	{
		DoFRenumbering::Cuthill_McKee (dof_handler_fake);
	}

	AffineConstraints<double>	constraints;
	constraints.clear ();
	DoFTools::make_hanging_node_constraints (dof_handler_fake, constraints);
	constraints.close();

	for (unsigned int i=0; i<basis_curl_v.size(); ++i)
	{
		basis_curl_v[i].block(0).reinit (dof_handler_fake.n_dofs());
		basis_curl_v[i].block(1) = 0;

		std_shape_function_curl.set_shape_fun_index (i);

		VectorTools::project (dof_handler_fake,
				constraints,
				quad_rule,
				std_shape_function_curl,
				basis_curl_v[i].block(0));
	}

	dof_handler_fake.clear ();
}



void
NedRTBasis::set_u_to_std ()
{
	// Quadrature used for projection
	QGauss<3> 	quad_rule (3);

	// Set up vector shape function from finite element on current cell
	ShapeFun::ShapeFunctionVector<3>
			std_shape_function_div (fe.base_element(1),
					*global_cell_ptr,
					/*verbose =*/ false);

	DoFHandler<3>	dof_handler_fake (triangulation);
	dof_handler_fake.distribute_dofs (fe.base_element(1));

	if (parameters.renumber_dofs)
	{
		DoFRenumbering::Cuthill_McKee (dof_handler_fake);
	}

	AffineConstraints<double>	constraints;
	constraints.clear ();
	DoFTools::make_hanging_node_constraints (dof_handler_fake, constraints);
	constraints.close();

	for (unsigned int i=0; i<basis_div_v.size(); ++i)
	{
		basis_div_v[i].block(0) = 0;
		basis_div_v[i].block(1).reinit (dof_handler_fake.n_dofs());

		std_shape_function_div.set_shape_fun_index (i);

		VectorTools::project (dof_handler_fake,
				constraints,
				quad_rule,
				std_shape_function_div,
				basis_div_v[i].block(1));
	}

	dof_handler_fake.clear ();
}



void
NedRTBasis::set_filename_global ()
{
	parameters.filename_global += (parameters.filename_global
			+ "." + Utilities::int_to_string(local_subdomain, 5)
			+ ".cell-" + global_cell_id.to_string()
			+ ".vtu");
}



const FullMatrix<double>&
NedRTBasis::get_global_element_matrix () const
{
	return global_element_matrix;
}



const Vector<double>&
NedRTBasis::get_global_element_rhs () const
{
	return global_element_rhs;
}



const std::string&
NedRTBasis::get_filename_global () const
{
	return parameters.filename_global;
}



void NedRTBasis::run ()
{
	if (parameters.verbose)
	{
		printf("\n------------------------------------------------------------\n");
	}

	// Create grid
	setup_grid ();

	// Reserve space for system matrices
	setup_system_matrix ();

	// Set up boundary conditions and other constraints
	setup_basis_dofs_curl ();
	setup_basis_dofs_div ();

	// Assemble
	assemble_system ();

	if (parameters.set_to_std)
	{
		set_sigma_to_std (); /* This is only a sanity check. */
		set_u_to_std (); /* This is only a sanity check. */
	}
	else // in this case solve
	{
		for (unsigned int n_basis=0;
						n_basis<length_system_basis;
						++n_basis)
		{
			if (n_basis<GeometryInfo<3>::lines_per_cell)
			{
				// This is for curl.
				system_matrix.reinit (sparsity_pattern_curl);

				system_matrix.copy_from(assembled_matrix);

				// Now take care of constraints
				constraints_curl_v[n_basis].condense(system_matrix, system_rhs_curl_v[n_basis]);

				// Now solve
				if (parameters.use_direct_solver)
					solve_direct (n_basis);
				else
				{
					solve_iterative (n_basis);
				}
			}
			else
			{
				// This is for div.
				const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;

				system_matrix.reinit (sparsity_pattern_curl);

				system_matrix.copy_from(assembled_matrix);

				// Now take care of constraints
				constraints_div_v[offset_index].condense(system_matrix, system_rhs_div_v[offset_index]);

				// Now solve
				if (parameters.use_direct_solver)
					solve_direct (n_basis);
				else
				{
					solve_iterative (n_basis);
				}
			}
		}
	}

	assemble_global_element_matrix ();

	{
		// Free memory as much as possible
		system_matrix.clear ();
		for (unsigned int i=0; i<basis_curl_v.size(); ++i)
		{
			sparsity_pattern_curl.reinit (0,0);
			constraints_curl_v[i].clear ();
		}
		for (unsigned int i=0; i<basis_div_v.size(); ++i)
		{
			sparsity_pattern_div.reinit (0,0);
			constraints_div_v[i].clear ();
		}
	}

	// We need to set a filename for the global solution on the current cell
	set_filename_global ();

	// Write basis output only if desired
	if (parameters.output_flag)
		for (unsigned int n_basis=0;
				n_basis<length_system_basis;
				++n_basis)
			output_basis (n_basis);

	if (parameters.verbose)
	{
		printf("------------------------------------------------------------\n");
	}
}

} // end namespace HelmholtzProblem
