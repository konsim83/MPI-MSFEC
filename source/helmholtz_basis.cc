#include "helmholtz_basis.h"

namespace HelmholtzProblem
{
using namespace dealii;

NedRTBasis::NedRTBasis ()
:
degree(0),
n_refine_global(1),
n_refine(1),
cell_number(0),
length_system_basis(GeometryInfo<3>::lines_per_cell
		+ GeometryInfo<3>::faces_per_cell),
output_flag(false),
is_built_global_element_matrix(false),
is_set_global_weights(false),
is_set_cell_data(false),
filename_global_solution(""),
fe (FE_Nedelec<3>(degree), 1,
		FE_RaviartThomas<3>(degree), 1),
dof_handler (triangulation),
volume_measure(0),
face_measure(GeometryInfo<3>::faces_per_cell, 0),
edge_measure(GeometryInfo<3>::lines_per_cell, 0),
corner_points(GeometryInfo<3>::vertices_per_cell,
		Point<3>()),
constraints_curl_v(GeometryInfo<3>::lines_per_cell),
constraints_div_v(GeometryInfo<3>::faces_per_cell),
sparsity_pattern_curl_v(GeometryInfo<3>::lines_per_cell),
sparsity_pattern_div_v(GeometryInfo<3>::faces_per_cell),
system_matrix_curl_v(GeometryInfo<3>::lines_per_cell),
system_matrix_div_v(GeometryInfo<3>::faces_per_cell),
system_rhs_curl_v(GeometryInfo<3>::lines_per_cell),
system_rhs_div_v(GeometryInfo<3>::faces_per_cell),
basis_curl_v(GeometryInfo<3>::lines_per_cell),
basis_div_v(GeometryInfo<3>::faces_per_cell),
global_element_matrix(fe.dofs_per_cell,
		fe.dofs_per_cell),
global_element_rhs(fe.dofs_per_cell),
global_weights(fe.dofs_per_cell, 0),
Preconditioner_v(GeometryInfo<3>::lines_per_cell
		+ GeometryInfo<3>::faces_per_cell)
{
}


NedRTBasis::~NedRTBasis ()
{
	for (unsigned int n_basis=0; n_basis<basis_curl_v.size(); ++n_basis)
	{
		system_matrix_curl_v.at(n_basis).clear();
		constraints_curl_v.at(n_basis).clear();
	}

	for (unsigned int n_basis=0; n_basis<basis_div_v.size(); ++n_basis)
	{
		system_matrix_div_v.at(n_basis).clear();
		constraints_div_v.at(n_basis).clear();
	}

	dof_handler.clear ();
}


void
NedRTBasis::setup_grid ()
{
	Assert (is_set_cell_data,
			ExcMessage ("Cell data must be set first."));

	for (unsigned int i=0; i<length_system_basis; ++i)
		Preconditioner_v.at(i).reset();

	GridGenerator::general_cell(triangulation, corner_points, /* colorize faces */true);

	triangulation.refine_global (n_refine);
}


void
NedRTBasis::setup_system_matrix ()
{
	dof_handler.distribute_dofs (fe);

	if (renumber_dofs)
	{
		DoFRenumbering::Cuthill_McKee (dof_handler);
		std::vector<unsigned int> block_component (3+3,0);
		block_component[3] = 1;
		block_component[4] = 1;
		block_component[5] = 1;

		// Make 2x2 blocks (first fluxes, then concentration)
		DoFRenumbering::component_wise (dof_handler, block_component);
	}
	else
	{
		DoFRenumbering::component_wise (dof_handler);
	}

	std::vector<types::global_dof_index> dofs_per_component (3+3);
	DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
	const unsigned int n_sigma = dofs_per_component[0],
					   n_u = dofs_per_component[3];
	if (verbose)
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

	{
		// Allocate memory
		BlockDynamicSparsityPattern dsp(2, 2);
		dsp.block(0, 0).reinit (n_sigma, n_sigma);
		dsp.block(1, 0).reinit (n_u, n_sigma);
		dsp.block(0, 1).reinit (n_sigma, n_u);
		dsp.block(1, 1).reinit (n_u, n_u);
		dsp.collect_sizes ();

		DoFTools::setup_sparsity_pattern (dof_handler, dsp);

		// Initialize the system matrix for global assembly
		sparsity_pattern.copy_from(dsp);
	}

	system_matrix.reinit (sparsity_pattern);

	// Initialize the rhs for global assembly
	system_rhs.reinit (2);
	system_rhs.block(0).reinit (n_sigma);
	system_rhs.block(1).reinit (n_u);
	system_rhs.collect_sizes ();

	// Initialize global solution
	global_solution.reinit (2);
	global_solution.block(0).reinit (n_sigma);
	global_solution.block(1).reinit (n_u);
	global_solution.collect_sizes ();
}


void
NedRTBasis::setup_basis_dofs_curl ()
{
	Assert (is_set_cell_data,
					ExcMessage ("Cell number must be set first."));

	ShapeFun::ShapeFunctionVector<3>
			std_shape_function (fe.base_element(0),
					global_cell,
					/*verbose =*/ false);
	ShapeFun::ShapeFunctionVectorCurl<3>
			std_shape_function_curl (fe.base_element(0),
					global_cell,
					/*verbose =*/ false);

	// Some size info.
	std::vector<types::global_dof_index> dofs_per_component (3+3);
	DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
	const unsigned int n_sigma = dofs_per_component[0],
					   n_u = dofs_per_component[3];

	// Allocate memory
	BlockDynamicSparsityPattern dsp(2, 2);
	dsp.block(0, 0).reinit (n_sigma, n_sigma);
	dsp.block(1, 0).reinit (n_u, n_sigma);
	dsp.block(0, 1).reinit (n_sigma, n_u);
	dsp.block(1, 1).reinit (n_u, n_u);
	dsp.collect_sizes ();

	////////////////////////////////////////////////////
	// set constraints (first hanging nodes, then curl)
	for (unsigned int n_basis=0; n_basis<basis_curl_v.size(); ++n_basis)
	{
		constraints_curl_v.at(n_basis).clear ();

		DoFTools::setup_hanging_node_constraints (dof_handler, constraints_curl_v.at(n_basis));

		std_shape_function.set_shape_fun_index(n_basis);
		std_shape_function_curl.set_shape_fun_index(n_basis);

		for (unsigned int i=0;
				i<GeometryInfo<3>::faces_per_cell;
				++i)
		{
			VectorTools::project_boundary_values_curl_conforming(dof_handler,
						/*first vector component */ 0,
						std_shape_function,
						/*boundary id*/ i,
						constraints_curl_v.at(n_basis));
			VectorTools::project_boundary_values_div_conforming(dof_handler,
								/*first vector component */ 3,
//								ZeroFunction<3>(3),
								std_shape_function_curl,
								/*boundary id*/ i,
								constraints_curl_v.at(n_basis));
		}

		constraints_curl_v.at(n_basis).close ();

		// Initialize sizes
		basis_curl_v.at(n_basis).reinit (2);
		basis_curl_v.at(n_basis).block(0).reinit (n_sigma);
		basis_curl_v.at(n_basis).block(1).reinit (n_u);
		basis_curl_v.at(n_basis).collect_sizes ();

		system_rhs_curl_v.at(n_basis).reinit (2);
		system_rhs_curl_v.at(n_basis).block(0).reinit (n_sigma);
		system_rhs_curl_v.at(n_basis).block(1).reinit (n_u);
		system_rhs_curl_v.at(n_basis).collect_sizes ();

		DoFTools::setup_sparsity_pattern(dof_handler,
										  dsp,
										  constraints_curl_v.at(n_basis),
										  /*keep_constrained_dofs = */ false);

		constraints_curl_v.at(n_basis).condense (dsp);

		sparsity_pattern_curl_v.at(n_basis).copy_from(dsp);

		system_matrix_curl_v.at(n_basis).reinit (sparsity_pattern_curl_v.at(n_basis));
	}
	////////////////////////////////////////////////////

	if (plot_sparsity_pattern)
	{
		std::cout
			<< std::endl
			<< "   Plotting sparsity pattern for first basis..."
			<< std::endl;
		std::ofstream out ("sparsity_pattern.gpl");
		sparsity_pattern_curl_v.at(/*n_basis */ 0).print_gnuplot(out);
	}
}


void
NedRTBasis::setup_basis_dofs_div ()
{
	Assert (is_set_cell_data,
					ExcMessage ("Cell number must be set first."));

	ShapeFun::ShapeFunctionVector<3>
			std_shape_function (fe.base_element(1),
					global_cell,
					/*verbose =*/ false);
//	ShapeFun::ShapeFunctionVectorCurl<3>
//			std_shape_function_curl (fe.base_element(1),
//					global_cell,
//					/*verbose =*/ false);

	// Some size info
	std::vector<types::global_dof_index> dofs_per_component (3+3);
		DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
		const unsigned int n_sigma = dofs_per_component[0],
						   n_u = dofs_per_component[3];

	// Allocate memory
	BlockDynamicSparsityPattern dsp(2, 2);
	dsp.block(0, 0).reinit (n_sigma, n_sigma);
	dsp.block(1, 0).reinit (n_u, n_sigma);
	dsp.block(0, 1).reinit (n_sigma, n_u);
	dsp.block(1, 1).reinit (n_u, n_u);
	dsp.collect_sizes ();

	////////////////////////////////////////////////////
	for (unsigned int n_basis=0; n_basis<basis_div_v.size(); ++n_basis)
	{
		// set constraints (first hanging nodes, then flux)
		constraints_div_v.at(n_basis).clear ();

		DoFTools::setup_hanging_node_constraints (dof_handler, constraints_div_v.at(n_basis));

		std_shape_function.set_shape_fun_index(n_basis);
//		std_shape_function_curl.set_shape_fun_index(n_basis);

		for (unsigned int i=0;
				i<GeometryInfo<3>::faces_per_cell;
				++i)
		{
//			VectorTools::project_boundary_values_curl_conforming(dof_handler,
//						/*first vector component */ 0,
//						std_shape_function_curl,
//						/*boundary id*/ i,
//						constraints_div_v.at(n_basis));
			VectorTools::project_boundary_values_div_conforming(dof_handler,
						/*first vector component */ 3,
						std_shape_function,
						/*boundary id*/ i,
						constraints_div_v.at(n_basis));
		}

		constraints_div_v.at(n_basis).close ();

		basis_div_v.at(n_basis).reinit (2);
		basis_div_v.at(n_basis).block(0).reinit (n_sigma);
		basis_div_v.at(n_basis).block(1).reinit (n_u);
		basis_div_v.at(n_basis).collect_sizes ();

		system_rhs_div_v.at(n_basis).reinit (2);
		system_rhs_div_v.at(n_basis).block(0).reinit (n_sigma);
		system_rhs_div_v.at(n_basis).block(1).reinit (n_u);
		system_rhs_div_v.at(n_basis).collect_sizes ();

		DoFTools::setup_sparsity_pattern(dof_handler,
										  dsp,
										  constraints_div_v.at(n_basis),
										  /*keep_constrained_dofs = */ false);

		constraints_div_v.at(n_basis).condense (dsp);

		sparsity_pattern_div_v.at(n_basis).copy_from(dsp);

		system_matrix_div_v.at(n_basis).reinit (sparsity_pattern_div_v.at(n_basis));
	}
	////////////////////////////////////////////////////

	if (plot_sparsity_pattern)
	{
		std::cout
			<< std::endl
			<< "   Plotting sparsity pattern for first basis..."
			<< std::endl;
		std::ofstream out ("sparsity_pattern.gpl");
		sparsity_pattern_div_v.at(/* n_basis */ 0).print_gnuplot(out);
	}
}


void
NedRTBasis::assemble_system ()
{
	Timer timer;
	if (verbose)
	{
		printf("Assembling local linear system in   cell %6d .......", cell_number);

		timer.start ();
	}
	// Choose appropriate quadrature rules
	QGauss<3>   quadrature_formula(degree+2);
    QGauss<2> face_quadrature_formula(degree+2);

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
//						global_cell,
//						/*verbose =*/ false);
	ShapeFun::ShapeFunctionVectorDiv<3>
				std_shape_function_u_div (fe.base_element(1),
						global_cell,
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

	// -----------------------------------------
	// need this to keep dofs apart from each other
	std::vector<typename DoFHandler<3>::active_cell_iterator>
		cell_v = std::vector<typename DoFHandler<3>::active_cell_iterator>(length_system_basis,
																			dof_handler.begin_active()),
		endc_v = std::vector<typename DoFHandler<3>::active_cell_iterator>(length_system_basis,
																			dof_handler.end());

	std::vector<std::vector<types::global_dof_index>>
		local_dof_indices_v
			= std::vector<std::vector<types::global_dof_index>>(length_system_basis,
																std::vector<types::global_dof_index>(dofs_per_cell));
	// -----------------------------------------


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
			local_rhs_v.at(n_basis) = 0;
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
						local_rhs_v.at(n_basis)(i) += 0;
					}
					else
						// This is rhs for div.
						local_rhs_v.at(n_basis)(i) += 0;
				}
			} // end for ++i
		} // end for ++q


		// Boundary integrals
		for (unsigned int face_number=0; face_number<GeometryInfo<3>::faces_per_cell; ++face_number)
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
//								local_rhs_v.at(n_basis)(i) -= ( sigma_curl_cross_n *
//																fe_face_values[curl].value (i, q_point) *
//																fe_face_values.JxW(q_point));
//								std::cout << local_rhs_v.at(n_basis)(i) << std::endl;
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
								local_rhs_v.at(n_basis)(i) += ( fe_face_values.normal_vector(q_point) *
																fe_face_values[flux].value (i, q_point) *
																diffusion_b_face_values[q_point] *
																std_shape_function_u_div_values[q_point] *
																fe_face_values.JxW(q_point));
//								std::cout << local_rhs_v.at(n_basis)(i) << std::endl;
							}
						}
					}
				} // end n_basis++
			} // end face_number++


		// ------------------------------------------
		// Only for use in global assembly
		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			system_rhs(local_dof_indices[i]) += local_rhs(i);
		}

		// Add to global matrix
		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				system_matrix.add (local_dof_indices[i],
								   local_dof_indices[j],
								   local_matrix(i,j));
		}
		// ------------------------------------------


		// ------------------------------------------
		// Only for use in local solving
		for (unsigned int n_basis=0;
				n_basis<length_system_basis;
				++n_basis)
		{
			cell_v.at(n_basis)->get_dof_indices(local_dof_indices_v.at(n_basis));

			if (n_basis<GeometryInfo<3>::lines_per_cell)
			{
				// This is for curl.
				constraints_curl_v.at(n_basis).distribute_local_to_global(local_matrix,
																local_rhs_v.at(n_basis),
																local_dof_indices_v.at(n_basis),
																system_matrix_curl_v.at(n_basis),
																system_rhs_curl_v.at(n_basis),
																/* use inhomogeneities for rhs */ true);
			}
			else
			{
				// This is for curl.
				const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
				constraints_div_v.at(offset_index).distribute_local_to_global(local_matrix,
																local_rhs_v.at(n_basis),
																local_dof_indices_v.at(n_basis),
																system_matrix_div_v.at(offset_index),
																system_rhs_div_v.at(offset_index),
																/* use inhomogeneities for rhs */ true);
			}
			++cell_v.at(n_basis);
		}
		// ------------------------------------------


	}// end for ++cell
	// ------------------------------------------------------------------

	if (verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
} // end assemble()



void
NedRTBasis::solve_direct (unsigned int n_basis)
{
	Timer timer;
	if (verbose)
	{
		printf("Solving linear system (directly) in   cell %6d   for basis   %d .......", cell_number, n_basis);

		timer.start ();
	}

	// Better work with references here but this should also be ok.
	ConstraintMatrix *constraints_ptr = NULL;
	BlockVector<double> *basis_ptr = NULL;
	BlockSparseMatrix<double> *system_matrix_ptr = NULL;
	BlockVector<double> *system_rhs_ptr = NULL;

	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		constraints_ptr = &(constraints_curl_v.at(n_basis));
		basis_ptr = &(basis_curl_v.at(n_basis));
		system_matrix_ptr = &(system_matrix_curl_v.at(n_basis));
		system_rhs_ptr = &(system_rhs_curl_v.at(n_basis));
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
		constraints_ptr = &(constraints_div_v.at(offset_index));
		basis_ptr = &(basis_div_v.at(offset_index));
		system_matrix_ptr = &(system_matrix_div_v.at(offset_index));
		system_rhs_ptr = &(system_rhs_div_v.at(offset_index));
	}

	constraints_ptr->distribute(*basis_ptr);

	//use direct solver
	SparseDirectUMFPACK A_direct;
	A_direct.initialize(*system_matrix_ptr);

	A_direct.vmult(*basis_ptr, *system_rhs_ptr);
	constraints_ptr->distribute(*basis_ptr);

	if (verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
}


void
NedRTBasis::solve_iterative_preconditioned (unsigned int n_basis)
{
	Timer timer;

	// ------------------------------------------
	// Make a preconditioner for each system matrix
	if (verbose)
	{
		printf("Computing preconditioner in   cell %6d   for basis %6d.......", cell_number, n_basis);

		timer.start ();
	}

	// Better work with references here but this should also be ok.
	ConstraintMatrix *constraints_ptr = NULL;
	BlockVector<double> *basis_ptr = NULL;
	BlockSparseMatrix<double> *system_matrix_ptr = NULL;
	BlockVector<double> *system_rhs_ptr = NULL;

	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		constraints_ptr = &(constraints_curl_v.at(n_basis));
		basis_ptr = &(basis_curl_v.at(n_basis));
		system_matrix_ptr = &(system_matrix_curl_v.at(n_basis));
		system_rhs_ptr = &(system_rhs_curl_v.at(n_basis));
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
		constraints_ptr = &(constraints_div_v.at(offset_index));
		basis_ptr = &(basis_div_v.at(offset_index));
		system_matrix_ptr = &(system_matrix_div_v.at(offset_index));
		system_rhs_ptr = &(system_rhs_div_v.at(offset_index));
	}

	constraints_ptr->distribute(*basis_ptr);

	Preconditioner_v.at(n_basis)
				= std_cxx11::shared_ptr<typename InnerPreconditioner<3>::type>
															( new typename InnerPreconditioner<3>::type() );
	Preconditioner_v.at(n_basis)->initialize (system_matrix_ptr->block(0,0),
								typename InnerPreconditioner<3>::type::AdditionalData());

	if (verbose)
	{
		timer.stop ();
		printf("done (%gs)\n",timer());
	}
	// ------------------------------------------

	// Now solve.
	if (verbose)
	{
		printf("Solving linear system (iteratively, with preconditioner) in   cell %6d   for basis   %d .......", cell_number, n_basis);

		timer.start ();
	}

	// Construct inverse of upper left block
	const InverseMatrixPrecon<
						SparseMatrix<double>,
						typename InnerPreconditioner<3>::type
						>
						A_inverse ( system_matrix_ptr->block(0,0), *(Preconditioner_v.at(n_basis)) );

	Vector<double> tmp (basis_ptr->block(0).size());
	{
		// Set up Schur complement
		SchurComplementPrecon<typename InnerPreconditioner<3>::type>
			schur_complement (*system_matrix_ptr, A_inverse);

		// Compute schur_rhs = -g + C*A^{-1}*f
		Vector<double> schur_rhs (basis_ptr->block(1).size());
		A_inverse.vmult (tmp, system_rhs_ptr->block(0));
		system_matrix_ptr->block(1,0).vmult (schur_rhs, tmp);
		schur_rhs -= system_rhs_ptr->block(1);

		// Set Solver parameters for solving for u
		SolverControl solver_control (basis_ptr->block(1).size(),
									1e-6*schur_rhs.l2_norm());
		SolverCG<> cg (solver_control);

//		ApproximateSchurComplement approximate_schur (*system_matrix_ptr);
//		InverseMatrix<ApproximateSchurComplement> preconditioner (approximate_schur);
//
//		// Solve for u
//		cg.solve (schur_complement,
//				basis_ptr->block(1),
//				schur_rhs,
//				preconditioner);

		// This is faster
		cg.solve (schur_complement,
					basis_ptr->block(1),
					schur_rhs,
					PreconditionIdentity());

		constraints_ptr->distribute(*basis_ptr);

		if (verbose)
			std::cout
				<< std::endl
				<< "       "
				<< solver_control.last_step()
				<< " CG Schur complement iterations to obtain convergence."
				<< std::endl;
	}

	{
		// use computed u to solve for sigma
		system_matrix_ptr->block(0,1).vmult (tmp, basis_ptr->block(1));
		tmp *= -1;
		tmp += system_rhs_ptr->block(0);

		// Solve for sigma
		A_inverse.vmult (basis_ptr->block(0), tmp);
	}

	constraints_ptr->distribute(*basis_ptr);

	if (verbose)
	{
		timer.stop ();
		printf("....done (%gs)\n",timer());
	}
}


void
NedRTBasis::solve_iterative(unsigned int n_basis)
{
	Timer timer;
	if (verbose)
	{
		printf("Solving linear system (iteratively, invert flux, no preconditioner) in   cell %6d   for basis   %d .......", cell_number, n_basis);

		timer.start ();
	}

	// Better work with references here but this should also be ok.
	ConstraintMatrix *constraints_ptr = NULL;
	BlockVector<double> *basis_ptr = NULL;
	BlockSparseMatrix<double> *system_matrix_ptr = NULL;
	BlockVector<double> *system_rhs_ptr = NULL;

	if (n_basis < GeometryInfo<3>::lines_per_cell)
	{
		constraints_ptr = &(constraints_curl_v.at(n_basis));
		basis_ptr = &(basis_curl_v.at(n_basis));
		system_matrix_ptr = &(system_matrix_curl_v.at(n_basis));
		system_rhs_ptr = &(system_rhs_curl_v.at(n_basis));
	}
	else
	{
		const unsigned int offset_index = n_basis - GeometryInfo<3>::lines_per_cell;
		constraints_ptr = &(constraints_div_v.at(offset_index));
		basis_ptr = &(basis_div_v.at(offset_index));
		system_matrix_ptr = &(system_matrix_div_v.at(offset_index));
		system_rhs_ptr = &(system_rhs_div_v.at(offset_index));
	}

	constraints_ptr->distribute(*basis_ptr);

	// Construct inverse of upper left block
	InverseMatrix<SparseMatrix<double> > inverse_mass (system_matrix_ptr->block(0,0));

	Vector<double> tmp (basis_ptr->block(0).size());
	{
		// Set up Schur complement
		SchurComplement schur_complement (*system_matrix_ptr, inverse_mass);

		// Compute schur_rhs = -g + C*A^{-1}*f
		Vector<double> schur_rhs (basis_ptr->block(1).size());
		inverse_mass.vmult (tmp, system_rhs_ptr->block(0));
		system_matrix_ptr->block(1,0).vmult (schur_rhs, tmp);
		schur_rhs -= system_rhs_ptr->block(1);

		// Set Solver parameters for solving for u
		SolverControl solver_control (basis_ptr->block(1).size(),
									1e-12*schur_rhs.l2_norm());
		SolverCG<> cg (solver_control);

		// Set up approximate Schur complement as preconditioner
		ApproximateSchurComplement approximate_schur (*system_matrix_ptr);
		InverseMatrix<ApproximateSchurComplement> approximate_inverse (approximate_schur);

		// Solve for u
		cg.solve (schur_complement,
				basis_ptr->block(1),
				schur_rhs,
				approximate_inverse);

		constraints_ptr->distribute(*basis_ptr);

		if (verbose)
			std::cout
				<< std::endl
				<< "       "
				<< solver_control.last_step()
				<< " CG Schur complement iterations to obtain convergence."
				<< std::endl;
	}

	{
		// use computed u to solve for sigma
		system_matrix_ptr->block(0,1).vmult (tmp, basis_ptr->block(1));
		tmp *= -1;
		tmp += system_rhs_ptr->block(0);

		// Solve for sigma
		inverse_mass.vmult (basis_ptr->block(0), tmp);
	}

	constraints_ptr->distribute(*basis_ptr);

	if (verbose)
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
					system_matrix.block(block_row, block_col).vmult(tmp_sigma, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += (test_vec_ptr->block(block_row) * tmp_sigma);
					tmp_sigma = 0;
				}
				if (block_col==1) /* This means trial function is u. */
				{
					system_matrix.block(block_row, block_col).vmult(tmp_sigma, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += (test_vec_ptr->block(block_row) * tmp_sigma);
					tmp_sigma = 0;
				}
			} // end if
			else /* This means we are testing with u. */
			{
				if (block_col==0) /* This means trial function is sigma. */
				{
					system_matrix.block(block_row, block_col).vmult(tmp_u, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += (test_vec_ptr->block(block_row) * tmp_u);
					tmp_u = 0;
				}
				if (block_col==1) /* This means trial function is u. */
				{
					system_matrix.block(block_row, block_col).vmult(tmp_u, trial_vec_ptr->block(block_col));
					global_element_matrix(i_test,i_trial) += test_vec_ptr->block(block_row) * tmp_u;
					tmp_u = 0;
				}
			} // end else
		} // end for i_trial

		if (i_test>=GeometryInfo<3>::lines_per_cell)
		{
			block_row = 1;
			// If we are testing with u we possibly have a right-hand side.
			global_element_rhs(i_test) += test_vec_ptr->block(block_row) * system_rhs.block(block_row);
		}
	} // end for i_test

	if (debug_verbose)
	{
		std::cout << "\n\nMS_MATRIX\n";
		for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
		{
			for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
			{
				std::cout << global_element_matrix(i,j) << "   " ;
			}
			std::cout << "\n";
		}
		std::cout << "\n" << std::endl;
	}


	is_built_global_element_matrix = true;
}


void
NedRTBasis::output_basis (unsigned int n_basis)
{
	Timer timer;
	if (verbose)
	{
		printf("Writing local solution in   cell %6d   for basis   %d .......", cell_number, n_basis);

		timer.start ();
	}

	BlockVector<double> *basis_ptr = NULL;
	if (n_basis<GeometryInfo<3>::lines_per_cell)
		basis_ptr = &(basis_curl_v.at(n_basis));
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

	data_out.build_patches (degree+1);

	std::string filename = "basis_3d_cell-";
	if (n_basis<GeometryInfo<3>::lines_per_cell)
	{
		filename += "curl";
		filename += Utilities::int_to_string (cell_number, 3);
		filename += "_index-";
		filename += Utilities::int_to_string (n_basis, 2);
	}
	else
	{
		filename += "div";
		filename += Utilities::int_to_string (cell_number, 3);
		filename += "_index-";
		filename += Utilities::int_to_string (n_basis - GeometryInfo<3>::lines_per_cell, 2);
	}
	filename += ".vtu";

	std::ofstream output (filename);
	data_out.write_vtu (output);

	if (verbose)
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

	std::ofstream output (filename_global_solution.c_str());
	data_out.write_vtu (output);
}


void
NedRTBasis::set_output_flag(bool flag)
{
	output_flag = flag;
}


void
NedRTBasis::set_cell_data(typename Triangulation<3>::active_cell_iterator &cell,
									unsigned int n_cell)
{
	global_cell = cell;
	cell_number = n_cell;

	for (unsigned int vertex_n=0;
		 vertex_n<GeometryInfo<3>::vertices_per_cell;
		 ++vertex_n)
	{
		corner_points.at(vertex_n) = cell->vertex(vertex_n);
	}

	volume_measure = cell->measure ();

	for (unsigned int j_face=0;
			j_face<GeometryInfo<3>::faces_per_cell;
			++j_face)
	{
		face_measure.at(j_face) = cell->face(j_face)->measure ();
	}

	for (unsigned int j_egde=0;
			j_egde<GeometryInfo<3>::lines_per_cell;
			++j_egde)
	{
		edge_measure.at(j_egde) = cell->line(j_egde)->measure ();
	}

	is_set_cell_data = true;
}


void
NedRTBasis::set_global_weights (std::vector<double> &weights)
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
		global_solution.block(0).sadd (1, global_weights.at(i), basis_curl_v.at(i).block(0));

	// Then set block 1
	for (unsigned int i=0;
			i<dofs_per_cell_u;
			++i)
		global_solution.block(1).sadd (1, global_weights.at(i+dofs_per_cell_sigma), basis_div_v.at(i).block(1));

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
				global_cell,
				/*verbose =*/ false);

	DoFHandler<3>	dof_handler_fake (triangulation);
	dof_handler_fake.distribute_dofs (fe.base_element(0));

	if (renumber_dofs)
	{
		DoFRenumbering::Cuthill_McKee (dof_handler_fake);
	}

	ConstraintMatrix	constraints;
	constraints.clear ();
	DoFTools::setup_hanging_node_constraints (dof_handler_fake, constraints);
	constraints.close();

	for (unsigned int i=0; i<basis_curl_v.size(); ++i)
	{
		basis_curl_v.at(i).block(0).reinit (dof_handler_fake.n_dofs());
		basis_curl_v.at(i).block(1) = 0;

		std_shape_function_curl.set_shape_fun_index (i);

		VectorTools::project (dof_handler_fake,
				constraints,
				quad_rule,
				std_shape_function_curl,
				basis_curl_v.at(i).block(0));
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
					global_cell,
					/*verbose =*/ false);

	DoFHandler<3>	dof_handler_fake (triangulation);
	dof_handler_fake.distribute_dofs (fe.base_element(1));

	if (renumber_dofs)
	{
		DoFRenumbering::Cuthill_McKee (dof_handler_fake);
	}

	ConstraintMatrix	constraints;
	constraints.clear ();
	DoFTools::setup_hanging_node_constraints (dof_handler_fake, constraints);
	constraints.close();

	for (unsigned int i=0; i<basis_div_v.size(); ++i)
	{
		basis_div_v.at(i).block(0) = 0;
		basis_div_v.at(i).block(1).reinit (dof_handler_fake.n_dofs());

		std_shape_function_div.set_shape_fun_index (i);

		VectorTools::project (dof_handler_fake,
				constraints,
				quad_rule,
				std_shape_function_div,
				basis_div_v.at(i).block(1));
	}

	dof_handler_fake.clear ();
}


void
NedRTBasis::set_filename_global_solution ()
{
	filename_global_solution += ("solution_ms_3d_fine-"
								  + Utilities::int_to_string (cell_number, 3)
								  + ".vtu");
}


unsigned int
NedRTBasis::get_cell_number () const
{
	Assert (is_set_cell_data,
				ExcMessage ("Cell number must be set first."));

	return cell_number;
}


unsigned int
NedRTBasis::get_n_sigma () const
{
	std::vector<types::global_dof_index> dofs_per_component (3+1);
	DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

	return dofs_per_component[0];
}


unsigned int
NedRTBasis::get_n_u () const
{
	std::vector<types::global_dof_index> dofs_per_component (3+1);
	DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

	return dofs_per_component[3];
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
NedRTBasis::get_filename_global_solution () const
{
	return filename_global_solution;
}


void NedRTBasis::run ()
{
	if (verbose)
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

	set_filename_global ();

	for (unsigned int index_basis=0;
			index_basis<GeometryInfo<3>::vertices_per_cell;
			++index_basis)
	{
		// reset everything
		system_rhs.reinit(solution_vector[index_basis].size());
		system_matrix.reinit (sparsity_pattern);

		system_matrix.copy_from(diffusion_matrix);

		// Now take care of constraints
		constraints_vector[index_basis].condense(system_matrix, system_rhs);

		// Now solve
		if (parameters.use_direct_solver)
			solve_direct (n_basis);
		else
		{
			solve_iterative(n_basis);
		}
	}

	if (parameters.set_to_std)
	{
		if (parameters.renumber_dofs)
		{
			throw std::runtime_error("DoF renumbering must be disabled when setting multiscale basis to standard setting.");
		}
		else
		{
			set_sigma_to_std (); /* This is only a sanity check. */
			set_u_to_std (); /* This is only a sanity check. */
		}
	}

	assemble_global_element_matrix ();

	if (output_flag)
		output_basis ();





	// Free memory as much as possible
	for (unsigned int i=0; i<basis_curl_v.size(); ++i)
	{
		sparsity_pattern_curl_v.at(i).reinit (0,0);
		system_matrix_curl_v.at(i).clear ();
		system_rhs_curl_v.at(i).reinit (2);
		system_rhs_curl_v.at(i).collect_sizes ();
		constraints_curl_v.at(i).clear ();
	}
	for (unsigned int i=0; i<basis_div_v.size(); ++i)
	{
		sparsity_pattern_div_v.at(i).reinit (0,0);
		system_matrix_div_v.at(i).clear ();
		system_rhs_div_v.at(i).reinit (2);
		system_rhs_div_v.at(i).collect_sizes ();
		constraints_div_v.at(i).clear ();
	}


	// We need to set a filename for the global solution on the current cell
	set_filename_global_solution ();

	// Write basis output only if desired
	if (output_flag)
		for (unsigned int n_basis=0;
				n_basis<basis_curl_v.size()+basis_div_v.size();
				++n_basis)
			output_basis (n_basis);

	if (verbose)
	{
		printf("------------------------------------------------------------\n");
	}
}

} // end namespace HelmholtzProblem
