#include "helmholtz_ref.h"

namespace HelmholtzProblem
{

using namespace dealii;


NedRTStd::NedRTStd (Parameters::NedRT::ParametersStd &parameters_)
:
mpi_communicator(MPI_COMM_WORLD),
parameters(parameters_),
triangulation(mpi_communicator,
			  typename Triangulation<3>::MeshSmoothing(
				Triangulation<3>::smoothing_on_refinement |
				Triangulation<3>::smoothing_on_coarsening)),
fe (FE_Nedelec<3>(0), 1,
	FE_RaviartThomas<3>(0), 1),
dof_handler (triangulation),
pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
{
}


NedRTStd::~NedRTStd ()
{
	system_matrix.clear();
	constraints.clear();
	dof_handler.clear ();
}



void NedRTStd::setup_grid ()
{
	TimerOutput::Scope t(computing_timer, "mesh generation");

	GridGenerator::hyper_cube (triangulation, 0.0, 1.0, true);

	triangulation.refine_global (parameters.n_refine);
}



void NedRTStd::setup_system_matrix ()
{
	TimerOutput::Scope t(computing_timer, "system and constraint setup");

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

	pcout << "Number of active cells: "
			<< triangulation.n_global_active_cells()
			<< std::endl
			<< "Total number of cells: "
			<< triangulation.n_cells()
			<< " (on " << triangulation.n_levels() << " levels)"
			<< std::endl
			<< "Number of degrees of freedom: "
			<< dof_handler.n_dofs()
			<< " (" << n_sigma << '+' << n_u << ')'
			<< std::endl;

	owned_partitioning.resize(2);
	owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_sigma);
	owned_partitioning[1] = dof_handler.locally_owned_dofs().get_view(n_sigma, n_sigma + n_u);

	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
	relevant_partitioning.resize(2);
	relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_sigma);
	relevant_partitioning[1] = locally_relevant_dofs.get_view(n_sigma, n_sigma + n_u);

	setup_constraints ();

	{
		// Allocate memory
		BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

		DoFTools::make_sparsity_pattern( dof_handler, dsp, constraints, false);
		SparsityTools::distribute_sparsity_pattern(dsp,
													dof_handler.locally_owned_dofs_per_processor(),
													mpi_communicator,
													locally_relevant_dofs);

		system_matrix.clear();
		system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

//		preconditioner_matrix.clear();
//		preconditioner_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
	}

	locally_relevant_solution.reinit(owned_partitioning,
								 relevant_partitioning,
								 mpi_communicator);

	system_rhs.reinit(owned_partitioning, mpi_communicator);
}



void NedRTStd::setup_constraints ()
{
	// set constraints (first hanging nodes, then flux)
	constraints.clear ();
	constraints.reinit(locally_relevant_dofs);

	DoFTools::make_hanging_node_constraints (dof_handler, constraints);

//	for (unsigned int i=0;
//			i<GeometryInfo<3>::faces_per_cell;
//			++i)
//	{
//		VectorTools::project_boundary_values_curl_conforming(dof_handler,
//					/*first vector component */ 0,
//					ZeroFunction<3>(6),
//					/*boundary id*/ i,
//					constraints);
//		VectorTools::project_boundary_values_div_conforming(dof_handler,
//							/*first vector component */ 3,
//							ZeroFunction<3>(6),
//							/*boundary id*/ i,
//							constraints);
//	}

	constraints.close();
}



void NedRTStd::assemble_system ()
{
	TimerOutput::Scope t(computing_timer, "assembly");

	system_matrix         = 0;
	system_rhs            = 0;

	QGauss<3>   quadrature_formula(parameters.degree + 2);
	QGauss<2> 	face_quadrature_formula(parameters.degree + 2);

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
	FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
	Vector<double>       local_rhs (dofs_per_cell);


	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


	// equation data
	const RightHandSide          		right_hand_side;
	const BoundaryDivergenceValues_u	boundary_divergence_values_u;
	const DiffusionInverse_A     		a_inverse;
	const Diffusion_B     				b;
	const ReactionRate           		reaction_rate;


	// allocate
	std::vector<Tensor<1,3>> 	rhs_values (n_q_points);
	std::vector<double> 		reaction_rate_values (n_q_points);
	std::vector<double> 		boundary_divergence_values_u_values (n_face_q_points);
	std::vector<Tensor<2,3>> 	a_inverse_values (n_q_points);
	std::vector<double>		 	b_values (n_q_points);

	// define extractors
	const FEValuesExtractors::Vector curl (/* first_vector_component */ 0);
	const FEValuesExtractors::Vector flux (/* first_vector_component */ 3);


	// ------------------------------------------------------------------
	// loop over cells
	typename DoFHandler<3>::active_cell_iterator
								cell = dof_handler.begin_active(),
								endc = dof_handler.end();
	for (; cell!=endc; ++cell)
	{
		if (cell->is_locally_owned())
		{
			fe_values.reinit (cell);
			local_matrix = 0;
			local_rhs = 0;

			right_hand_side.value_list (fe_values.get_quadrature_points(),
										rhs_values);
			reaction_rate.value_list(fe_values.get_quadrature_points(),
												reaction_rate_values);
			a_inverse.value_list (fe_values.get_quadrature_points(),
								  a_inverse_values);
			b.value_list(fe_values.get_quadrature_points(),
												b_values);

			// loop over quad points
			for (unsigned int q=0; q<n_q_points; ++q)
			{
				// loop over rows
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				{
					// test functions
					const Tensor<1,3> 		tau_i = fe_values[curl].value (i, q);
					const Tensor<1,3>      	curl_tau_i = fe_values[curl].curl (i, q);
					const double 	     	div_v_i = fe_values[flux].divergence (i, q);
					const Tensor<1,3>      	v_i = fe_values[flux].value (i, q);

					// loop over columns
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
						local_matrix(i,j) += (tau_i * a_inverse_values[q] * sigma_j /* block (0,0) */
																- curl_tau_i * u_j /* block (0,1) */
																+ v_i * curl_sigma_j /* block (1,0) */
																+ div_v_i * b_values[q] * div_u_j /* block (1,1) */
																+ v_i * reaction_rate_values[q] * u_j) /* block (1,1) */
																* fe_values.JxW(q);
					} // end for ++j

					local_rhs(i) += v_i *
									rhs_values[q] *
									fe_values.JxW(q);
				} // end for ++i
			} // end for ++q


			// line integral over boundary faces for for natural conditions on u
	//		for (unsigned int face_n=0;
	//					 face_n<GeometryInfo<3>::faces_per_cell;
	//					 ++face_n)
	//		{
	//			if (cell->at_boundary(face_n)
	//		//					&& cell->face(face_n)->boundary_id()!=0 /* Select only certain faces. */
	//		//					&& cell->face(face_n)->boundary_id()!=2 /* Select only certain faces. */
	//					)
	//			{
	//				fe_face_values.reinit (cell, face_n);
	//
	//				boundary_values_u.value_list (fe_face_values.get_quadrature_points(),
	//					  boundary_values_u_values);
	//
	//				for (unsigned int q=0; q<n_face_q_points; ++q)
	//					for (unsigned int i=0; i<dofs_per_cell; ++i)
	//						local_rhs(i) += -(fe_face_values[flux].value (i, q) *
	//										fe_face_values.normal_vector(q) *
	//										boundary_values_u_values[q] *
	//										fe_face_values.JxW(q));
	//			}
	//		}

			// Add to global matrix, include constraints
			cell->get_dof_indices(local_dof_indices);
			constraints.distribute_local_to_global(local_matrix,
													local_rhs,
													local_dof_indices,
													system_matrix,
													system_rhs,
													/* use inhomogeneities for rhs */ true);
		}
	}// end for ++cell

	system_matrix.compress(VectorOperation::add);
	system_rhs.compress(VectorOperation::add);
}


void NedRTStd::solve_direct ()
{
	TimerOutput::Scope              t(computing_timer, " direct solver (MUMPS)");

#ifndef USE_PETSC_LA
	throw std::runtime_error("You must use deal.ii with PetSc to use MUMPS.");
#endif

	throw std::runtime_error("Solver not implemented: MUMPS does not work on xyzWrapper::MPI::BlockSparseMatrix classes.");

//	LA::MPI::BlockVector completely_distributed_solution(owned_partitioning, mpi_communicator);
//
//	SolverControl                    solver_control;
//	PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
//	solver.set_symmetric_mode(true);
//	solver.solve(system_matrix, completely_distributed_solution, system_rhs);
//
//	pcout << "   Solved in "
//			<< solver_control.last_step()
//			<< " iterations."
//			<< std::endl;
//
//	constraints.distribute(completely_distributed_solution);
//	locally_relevant_solution = completely_distributed_solution;
}


void
NedRTStd::solve_iterative ()
{
	inner_schur_preconditioner = std::make_shared<typename LinearSolvers::InnerPreconditioner<3>::type>();

	typename LinearSolvers::InnerPreconditioner<3>::type::AdditionalData data;
#ifdef USE_PETSC_LA
	data.symmetric_operator = true; // Only for AMG
#endif

	inner_schur_preconditioner->initialize(system_matrix.block(0, 0), data);

	const LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
		typename LinearSolvers::InnerPreconditioner<3>::type>
			block_inverse (system_matrix.block(0,0),
					*inner_schur_preconditioner );

	// Vector for solution
	LA::MPI::BlockVector distributed_solution(owned_partitioning,
												mpi_communicator);

	// tmp of size block(0)
	LA::MPI::Vector	tmp(owned_partitioning[0],
							mpi_communicator);

	// Set up Schur complement
	LinearSolvers::SchurComplementMPI<LA::MPI::BlockSparseMatrix,
				LA::MPI::Vector,
				typename LinearSolvers::InnerPreconditioner<3>::type>
		schur_complement (system_matrix, block_inverse,
							owned_partitioning,
							mpi_communicator);

	// Compute schur_rhs = -g + C*A^{-1}*f
	LA::MPI::Vector schur_rhs (owned_partitioning[1],
//								relevant_partitioning[1],
								mpi_communicator);
	block_inverse.vmult (tmp, system_rhs.block(0));
	system_matrix.block(1,0).vmult (schur_rhs, tmp);
	schur_rhs -= system_rhs.block(1);
	{
		TimerOutput::Scope t(computing_timer, "Schur complement solver (for u)");

		// Set Solver parameters for solving for u
		SolverControl solver_control (system_matrix.m(),
									1e-6*schur_rhs.l2_norm());
		SolverCG<LA::MPI::Vector> schur_solver (solver_control);

//		PreconditionIdentity preconditioner;

		/*
		 * Precondition the Schur complement with
		 * the approximate inverse of the
		 * Schur complement.
		 */
//		LinearSolvers::ApproximateInverseMatrix<LinearSolvers::SchurComplementMPI<LA::MPI::BlockSparseMatrix,
//																					LA::MPI::Vector,
//																					typename LinearSolvers::InnerPreconditioner<3>::type>,
//									PreconditionIdentity>
//									preconditioner (schur_complement,
//												PreconditionIdentity() );

		/*
		 * Precondition the Schur complement with
		 * the (approximate) inverse of an approximate
		 * Schur complement.
		 */
		LinearSolvers::ApproximateSchurComplementMPI<LA::MPI::BlockSparseMatrix,
													LA::MPI::Vector,
													LA::MPI::PreconditionAMG>
													approx_schur (system_matrix, owned_partitioning, mpi_communicator);

		LinearSolvers::ApproximateInverseMatrix<LinearSolvers::ApproximateSchurComplementMPI<LA::MPI::BlockSparseMatrix,
																					LA::MPI::Vector,
																					LA::MPI::PreconditionAMG>,
												PreconditionIdentity>
												preconditioner (approx_schur,
															PreconditionIdentity() );

		/*
		 * Precondition the Schur complement with a preconditioner of block(1,1).
		 */
//		LA::MPI::PreconditionAMG preconditioner;
//		preconditioner.initialize(system_matrix.block(1, 1), data);

		schur_solver.solve (schur_complement,
					distributed_solution.block(1),
					schur_rhs,
					preconditioner);

		pcout << "   Iterative Schur complement solver converged in " << solver_control.last_step() << " iterations."
					  << std::endl;
	}

	{
		TimerOutput::Scope t(computing_timer, "outer CG solver (for sigma)");

	//	SolverControl                    outer_solver_control;
	//	PETScWrappers::SparseDirectMUMPS outer_solver(outer_solver_control, mpi_communicator);
	//	outer_solver.set_symmetric_mode(true);

		// use computed u to solve for sigma
		system_matrix.block(0,1).vmult (tmp, distributed_solution.block(1));
		tmp *= -1;
		tmp += system_rhs.block(0);

		// Solve for sigma
	//	outer_solver.solve(system_matrix.block(0,0), distributed_solution.block(0), tmp);
		block_inverse.vmult (distributed_solution.block(0), tmp);

		pcout << "   Outer solver completed."
						  << std::endl;
	}

	constraints.distribute(distributed_solution);

	locally_relevant_solution = distributed_solution;
}


void NedRTStd::output_results () const
{
//
//	{
//		const ComponentSelectFunction<3> u_mask(std::make_pair(3, 6), 3 + 3);
//		const ComponentSelectFunction<3> sigma_mask(std::make_pair(0, 3), 3 + 3);
//	}

//		LA::MPI::BlockVector interpolated;
//		interpolated.reinit(owned_partitioning, MPI_COMM_WORLD);
//		VectorTools::interpolate(dof_handler, ExactSolution<3>(), interpolated);
//
//		LA::MPI::BlockVector interpolated_relevant(owned_partitioning,
//											   relevant_partitioning,
//											   MPI_COMM_WORLD);
//		interpolated_relevant = interpolated;
//		{
//			std::vector<std::string> solution_names(dim, "ref_u");
//			solution_names.emplace_back("ref_p");
//			data_out.add_data_vector(interpolated_relevant,
//								   solution_names,
//								   DataOut<dim>::type_dof_data,
//								   data_component_interpretation);
//		}

	std::vector<std::string> solution_names(3, "sigma");
	solution_names.emplace_back("u");
	solution_names.emplace_back("u");
	solution_names.emplace_back("u");

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation(3+3, DataComponentInterpretation::component_is_part_of_vector);

	DataOut<3> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(locally_relevant_solution,
						 solution_names,
						 DataOut<3>::type_dof_data,
						 data_component_interpretation);

	Vector<float> subdomain(triangulation.n_active_cells());
	for (unsigned int i = 0; i < subdomain.size(); ++i)
		subdomain(i) = triangulation.locally_owned_subdomain();

	data_out.add_data_vector(subdomain, "subdomain_id");
	data_out.build_patches();

	std::string filename(parameters.filename_output);
	filename += "_n_refine-" + Utilities::int_to_string(parameters.n_refine,2);
	filename += "." + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
	filename += ".vtu";

	std::ofstream output(filename);
	data_out.write_vtu(output);

	// pvtu-record for all local outputs
	if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
	{
		std::vector<std::string> local_filenames(Utilities::MPI::n_mpi_processes(mpi_communicator),
													parameters.filename_output);
		for (unsigned int i = 0;
			 i < Utilities::MPI::n_mpi_processes(mpi_communicator);
			 ++i)
		{
			local_filenames[i] += "_n_refine-" + Utilities::int_to_string(parameters.n_refine, 2)
									+ "." + Utilities::int_to_string(i, 4)
									+ ".vtu";
		}

		std::string master_file = parameters.filename_output
								+ "_n_refine-" + Utilities::int_to_string(parameters.n_refine,2)
								+ ".pvtu";
		std::ofstream master_output(master_file.c_str());
		data_out.write_pvtu_record(master_output, local_filenames);
	}
}


/*!
 * Solve standard mixed problem with Nedelec-Raviart-Thomas element pairing.
 */
void NedRTStd::run (){

	if (parameters.compute_solution == false)
	{
		deallog << "Run of standard problem is explicitly disabled in parameter file. " << std::endl;
		return;
	}

#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif


	setup_grid();

	setup_system_matrix ();

	assemble_system ();

	if (parameters.use_direct_solver)
		solve_direct (); // SparseDirectMUMPS
	else
	{
		solve_iterative (); // Schur complement for A
	}

	{
		TimerOutput::Scope t(computing_timer, "vtu output");
		output_results ();
	}


	if (parameters.verbose)
	{
		computing_timer.print_summary();
		computing_timer.reset();
	}
}

} // end namespace HelmholtzProblem