#include <RT_DQ/rt_dq_ref.h>

namespace RTDQ
{
  using namespace dealii;

  RTDQStd::RTDQStd(ParametersStd &    parameters_,
                   const std::string &parameter_filename_)
    : mpi_communicator(MPI_COMM_WORLD)
    , parameters(parameters_)
    , parameter_filename(parameter_filename_)
    , triangulation(mpi_communicator,
                    typename Triangulation<3>::MeshSmoothing(
                      Triangulation<3>::smoothing_on_refinement |
                      Triangulation<3>::smoothing_on_coarsening))
    , fe(FE_RaviartThomas<3>(parameters.degree),
         1,
         FE_DGQ<3>(parameters.degree),
         1)
    , dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}

  RTDQStd::~RTDQStd()
  {
    system_matrix.clear();
    constraints.clear();
    dof_handler.clear();
  }

  void
    RTDQStd::setup_grid()
  {
    TimerOutput::Scope t(computing_timer, "mesh generation");

    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);

    triangulation.refine_global(parameters.n_refine);
  }

  void
    RTDQStd::setup_system_matrix()
  {
    TimerOutput::Scope t(computing_timer, "system (incl. constraint) setup");

    dof_handler.distribute_dofs(fe);

    if (parameters.renumber_dofs)
      {
        DoFRenumbering::Cuthill_McKee(dof_handler);
      }

    DoFRenumbering::block_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);
    const unsigned int n_sigma = dofs_per_block[0], n_u = dofs_per_block[1];

    pcout << "Number of active cells: " << triangulation.n_global_active_cells()
          << std::endl
          << "Total number of cells: " << triangulation.n_cells() << " (on "
          << triangulation.n_levels() << " levels)" << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << n_sigma << '+' << n_u << ')' << std::endl;

    owned_partitioning.resize(2);
    owned_partitioning[0] =
      dof_handler.locally_owned_dofs().get_view(0, n_sigma);
    owned_partitioning[1] =
      dof_handler.locally_owned_dofs().get_view(n_sigma, n_sigma + n_u);

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_sigma);
    relevant_partitioning[1] =
      locally_relevant_dofs.get_view(n_sigma, n_sigma + n_u);

    setup_constraints();

    {
      // Allocate memory
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                   dof_handler.locally_owned_dofs()),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.clear();
      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    locally_relevant_solution.reinit(owned_partitioning,
                                     relevant_partitioning,
                                     mpi_communicator);

    system_rhs.reinit(owned_partitioning, mpi_communicator);
  }

  void
    RTDQStd::setup_constraints()
  {
    TimerOutput::Scope t(computing_timer, "constraint setup");

    // set constraints (first hanging nodes, then flux)
    constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    /*
     * Set true for Neumann BCs (hard constraints on sigma).
     */
    if (parameters.is_pure_neumann)
      {
        EquationData::Boundary_A_grad_u     tensor_boundary_A_grad_u;
        VectorFunctionFromTensorFunction<3> boundary_A_grad_u(
          tensor_boundary_A_grad_u);

        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
          VectorTools::project_boundary_values_div_conforming(
            dof_handler,
            /*first vector component */ 0,
            boundary_A_grad_u,
            /*boundary id*/ i,
            constraints);
      }

    /*
     * If we have a Laplace problem (not Helmholtz) and a pure
     * Neumann problem then we need to make sure that u is unique.
     * We therefore add a constraint on dofs
     */
    if (parameters.is_laplace && parameters.is_pure_neumann)
      {
        IndexSet locally_relevant_dofs_u(relevant_partitioning[1]);

        // initially set a non-admissible value
        unsigned int first_local_dof_u = dof_handler.n_dofs() + 1;
        if (locally_relevant_dofs_u.n_elements() > 0)
          first_local_dof_u = locally_relevant_dofs_u.nth_index_in_set(0);

        // first boundary dof is minimum of all
        const unsigned int first_dof_u =
          dealii::Utilities::MPI::min(first_local_dof_u, mpi_communicator);

        /*
         * This constrains only the first dof on the first processor. We set it
         * to zero. Note that setting a point value may be problematic for an
         * H1-Function but in parallel adding mean value constraints should not
         * be done.
         */
        if (first_dof_u == first_local_dof_u)
          constraints.add_line(first_dof_u);
      }

    constraints.close();
  }

  void
    RTDQStd::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    QGauss<3> quadrature_formula(parameters.degree + 2);
    QGauss<2> face_quadrature_formula(parameters.degree + 3);

    // Get relevant quantities to be updated from finite element
    FEValues<3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<3> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

    // Define some abbreviations
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // Declare local contributions and reserve memory
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // equation data
    const EquationData::RightHandSideParsed right_hand_side(
      parameter_filename, /* n_components */ 1);
    const EquationData::BoundaryValues_u   boundary_values_u;
    const EquationData::DiffusionInverse_A a_inverse(parameter_filename);
    const EquationData::ReactionRate       reaction_rate;

    // allocate
    std::vector<double>       rhs_values(n_q_points);
    std::vector<double>       reaction_rate_values(n_q_points);
    std::vector<double>       boundary_values_u_values(n_face_q_points);
    std::vector<Tensor<2, 3>> a_inverse_values(n_q_points);

    // define extractors
    const FEValuesExtractors::Vector flux(0);
    const FEValuesExtractors::Scalar concentration(3);

    // ------------------------------------------------------------------
    // loop over cells
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            local_matrix = 0;
            local_rhs    = 0;

            right_hand_side.value_list(fe_values.get_quadrature_points(),
                                       rhs_values);
            reaction_rate.value_list(fe_values.get_quadrature_points(),
                                     reaction_rate_values);
            a_inverse.value_list(fe_values.get_quadrature_points(),
                                 a_inverse_values);

            // loop over quad points
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // loop over rows
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // test functions
                    const Tensor<1, 3> tau_i = fe_values[flux].value(i, q);
                    const double div_tau_i   = fe_values[flux].divergence(i, q);
                    const double v_i = fe_values[concentration].value(i, q);

                    // loop over columns
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        // trial functions
                        const Tensor<1, 3> sigma_j =
                          fe_values[flux].value(j, q);
                        const double div_sigma_j =
                          fe_values[flux].divergence(j, q);
                        const double u_j = fe_values[concentration].value(j, q);

                        /*
                         * Discretize
                         * K^{-1}sigma + grad(u) = 0
                         * div(sigma) + alpha*u = f ,
                         * where alpha>=0 (this is
                         * important) This is the
                         * simplest form of a
                         * diffusion-reaction equation
                         * where an anisotropic
                         * diffusion and reaction are in
                         * balance in a heterogeneous
                         * medium.
                         */
                        local_matrix(i, j) +=
                          (tau_i * a_inverse_values[q] * sigma_j -
                           div_tau_i * u_j + v_i * div_sigma_j +
                           v_i * reaction_rate_values[q] * u_j) *
                          fe_values.JxW(q);
                      } // end for ++j

                    local_rhs(i) += v_i * rhs_values[q] * fe_values.JxW(q);
                  } // end for ++i
              }     // end for ++q

            // line integral over boundary faces for for natural
            // conditions on u
            if (!parameters.is_pure_neumann)
              {
                for (unsigned int face_n = 0;
                     face_n < GeometryInfo<3>::faces_per_cell;
                     ++face_n)
                  {
                    if (cell->at_boundary(face_n)
                        //                &&
                        //                cell->face(face_n)->boundary_id()!=0
                        //                /* Select only certain
                        //                faces. */
                        //                &&
                        //                cell->face(face_n)->boundary_id()!=2
                        //                /* Select only certain
                        //                faces. */
                    )
                      {
                        fe_face_values.reinit(cell, face_n);

                        boundary_values_u.value_list(
                          fe_face_values.get_quadrature_points(),
                          boundary_values_u_values);

                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            local_rhs(i) += -(fe_face_values[flux].value(i, q) *
                                              fe_face_values.normal_vector(q) *
                                              boundary_values_u_values[q]) *
                                            fe_face_values.JxW(q);
                      }
                  }
              }


            // Add to global matrix, include constraints
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(
              local_matrix,
              local_rhs,
              local_dof_indices,
              system_matrix,
              system_rhs,
              /* use inhomogeneities for rhs */ true);
          }
      } // end for ++cell
        // ------------------------------------------------------------------
  }     // end assemble()

  void
    RTDQStd::solve_direct()
  {
    TimerOutput::Scope t(computing_timer, " direct solver (MUMPS)");

    throw std::runtime_error(
      "Solver not implemented: MUMPS does not work on "
      "TrilinosWrappers::MPI::BlockSparseMatrix classes.");
  }

  void
    RTDQStd::solve_iterative()
  {
    inner_schur_preconditioner =
      std::make_shared<typename LinearSolvers::InnerPreconditioner<3>::type>();

    typename LinearSolvers::InnerPreconditioner<3>::type::AdditionalData data;

    inner_schur_preconditioner->initialize(system_matrix.block(0, 0), data);

    /*
     * Define the inverse of the first
     * block by its action.
     */
    const LinearSolvers::InverseMatrix<
      LA::MPI::SparseMatrix,
      typename LinearSolvers::InnerPreconditioner<3>::type>
      block_inverse(system_matrix.block(0, 0), *inner_schur_preconditioner);

    // Vector for solution
    LA::MPI::BlockVector distributed_solution(owned_partitioning,
                                              mpi_communicator);

    // tmp of size block(0)
    LA::MPI::Vector tmp(owned_partitioning[0], mpi_communicator);

    // Set up Schur complement
    LinearSolvers::SchurComplementMPI<
      LA::MPI::BlockSparseMatrix,
      LA::MPI::Vector,
      typename LinearSolvers::InnerPreconditioner<3>::type>
      schur_complement(system_matrix,
                       block_inverse,
                       owned_partitioning,
                       mpi_communicator);

    // Compute schur_rhs = -g + C*A^{-1}*f
    LA::MPI::Vector schur_rhs(owned_partitioning[1],
                              //    						  relevant_partitioning[1],
                              mpi_communicator);
    block_inverse.vmult(tmp, system_rhs.block(0));
    system_matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= system_rhs.block(1);
    {
      TimerOutput::Scope t(computing_timer, "Schur complement solver (for u)");

      // Set Solver parameters for solving for u
      SolverControl             solver_control(system_matrix.m(),
                                   1e-6 * schur_rhs.l2_norm());
      SolverCG<LA::MPI::Vector> schur_solver(solver_control);

      /////////////////////////////////////////////////////////////////////
      /*
       * Use no preconditioner
       */
      // PreconditionIdentity preconditioner;

      /*
       * Precondition the Schur complement with
       * the approximate inverse of the
       * Schur complement.
       */
      //            LinearSolvers::ApproximateInverseMatrix<
      //              LinearSolvers::SchurComplementMPI<
      //                LA::MPI::BlockSparseMatrix,
      //                LA::MPI::Vector,
      //                typename
      //                LinearSolvers::InnerPreconditioner<3>::type>,
      //              PreconditionIdentity>
      //              preconditioner(schur_complement,
      //            		  PreconditionIdentity(),
      //#ifdef DEBUG
      //                       /* n_iter */ 1000);
      //#else
      //                       /* n_iter */ 20);
      //#endif

      /*
       * Precondition the Schur complement with
       * the approximate inverse of an approximate
       * Schur complement.
       */
      LinearSolvers::ApproximateSchurComplementMPI<LA::MPI::BlockSparseMatrix,
                                                   LA::MPI::Vector,
                                                   LA::MPI::PreconditionILU>
        approx_schur(system_matrix, owned_partitioning, mpi_communicator);

      LinearSolvers::ApproximateInverseMatrix<
        LinearSolvers::ApproximateSchurComplementMPI<LA::MPI::BlockSparseMatrix,
                                                     LA::MPI::Vector,
                                                     LA::MPI::PreconditionILU>,
        PreconditionIdentity>
        preconditioner(approx_schur,
                       PreconditionIdentity(),
#ifdef DEBUG
                       /* n_iter */ 1000);
#else
                       /* n_iter */ 20);
#endif
      /////////////////////////////////////////////////////////////////////

      schur_solver.solve(schur_complement,
                         distributed_solution.block(1),
                         schur_rhs,
                         preconditioner);

      pcout << "   Iterative Schur complement solver converged in "
            << solver_control.last_step() << " iterations." << std::endl;

      constraints.distribute(distributed_solution);
    }

    {
      TimerOutput::Scope t(computing_timer, "outer CG solver (for sigma)");

      //	SolverControl                    outer_solver_control;
      //	PETScWrappers::SparseDirectMUMPS
      // outer_solver(outer_solver_control,
      // mpi_communicator); 	outer_solver.set_symmetric_mode(true);

      // use computed u to solve for sigma
      system_matrix.block(0, 1).vmult(tmp, distributed_solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      // Solve for sigma
      //	outer_solver.solve(system_matrix.block(0,0),
      // distributed_solution.block(0), tmp);
      block_inverse.vmult(distributed_solution.block(0), tmp);

      pcout << "   Outer solver completed." << std::endl;

      constraints.distribute(distributed_solution);
    }

    locally_relevant_solution = distributed_solution;
  }

  void
    RTDQStd::transfer_solution()
  {
    TimerOutput::Scope t(computing_timer, "solution transfer");

    /*
     * Refine everything.
     */
    {
      for (typename Triangulation<3>::active_cell_iterator cell =
             triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        if (cell->is_locally_owned())
          cell->set_refine_flag();
    }

    /*
     * Prepare the triangulation for refinement.
     */
    triangulation.prepare_coarsening_and_refinement();

    /*
     * Prepare the refinement in the transfer object,
     * locally_relevant_old_solution is the source.
     */
    parallel::distributed::SolutionTransfer<3, LA::MPI::BlockVector>
      solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(
      locally_relevant_solution);

    /*
     * Now actually refine the mesh
     */
    triangulation.execute_coarsening_and_refinement();

    { /*
       * Setup new dofs and constraints.
       */
      dof_handler.distribute_dofs(fe);

      if (parameters.renumber_dofs)
        {
          DoFRenumbering::Cuthill_McKee(dof_handler);
        }

      DoFRenumbering::block_wise(dof_handler);

      std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler);
      const unsigned int n_sigma = dofs_per_block[0], n_u = dofs_per_block[1];

      owned_partitioning.resize(2);
      owned_partitioning[0] =
        dof_handler.locally_owned_dofs().get_view(0, n_sigma);
      owned_partitioning[1] =
        dof_handler.locally_owned_dofs().get_view(n_sigma, n_sigma + n_u);

      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      relevant_partitioning.resize(2);
      relevant_partitioning[0].clear();
      relevant_partitioning[1].clear();
      relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_sigma);
      relevant_partitioning[1] =
        locally_relevant_dofs.get_view(n_sigma, n_sigma + n_u);

      setup_constraints();

      locally_relevant_solution.reinit(owned_partitioning,
                                       relevant_partitioning,
                                       mpi_communicator);
    }


    /*
     * New locally_owned_solution from new dofs.
     */
    TrilinosWrappers::MPI::BlockVector locally_owned_solution;
    locally_owned_solution.reinit(owned_partitioning, mpi_communicator);

    /*
     * Now interpolate to new mesh.
     */
    solution_transfer.interpolate(locally_owned_solution);

    /*
     * Take care of constraints.
     */
    constraints.distribute(locally_owned_solution);

    locally_relevant_solution = locally_owned_solution;
  }

  void
    RTDQStd::output_results() const
  {
    std::vector<std::string> solution_names(3, "sigma");
    solution_names.push_back("u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        3, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

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

    // Postprocess
    std::unique_ptr<RTDQ_PostProcessor> postprocessor(
      new RTDQ_PostProcessor(parameter_filename));
    data_out.add_data_vector(locally_relevant_solution, *postprocessor);

    data_out.build_patches();

    std::string filename(parameters.filename_output);
    filename += "_n_refine-" + Utilities::int_to_string(parameters.n_refine, 2);
    filename +=
      "." +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
    filename += ".vtu";

    std::ofstream output(parameters.dirname_output + "/" + filename);
    data_out.write_vtu(output);

    // pvtu-record for all local outputs
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> local_filenames(
          Utilities::MPI::n_mpi_processes(mpi_communicator),
          parameters.filename_output);
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          {
            local_filenames[i] +=
              "_n_refine-" + Utilities::int_to_string(parameters.n_refine, 2) +
              "." + Utilities::int_to_string(i, 4) + ".vtu";
          }

        std::string master_file =
          parameters.filename_output + "_n_refine-" +
          Utilities::int_to_string(parameters.n_refine, 2) + ".pvtu";
        std::ofstream master_output(parameters.dirname_output + "/" +
                                    master_file);
        data_out.write_pvtu_record(master_output, local_filenames);
      }
  }

  /**
   * Solve problem with all input parameters.
   */
  void
    RTDQStd::run()
  {
    if (parameters.compute_solution == false)
      {
        deallog << "Run of standard problem is explicitly disabled in "
                   "parameter file. "
                << std::endl;
        return;
      }

    pcout << std::endl
          << "===========================================" << std::endl
          << "Solving >> RT-DQ STANDARD << problem in 3D." << std::endl;

#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif

    setup_grid();

    setup_system_matrix();

    assemble_system();

    if (parameters.use_direct_solver)
      solve_direct(); // SparseDirectMUMPS
    else
      {
        solve_iterative(); // Schur complement for A
      }

    const int n_transfer = parameters.transfer_to_level - parameters.n_refine;
    if (n_transfer > 0)
      {
        pcout << std::endl
              << "INFO: Transfer to finer grid by   " << n_transfer
              << "   levels to global refinement level   "
              << parameters.transfer_to_level << std::endl
              << std::endl;

        for (int i = 0; i < n_transfer; ++i)
          transfer_solution();
      }
    else
      {
        pcout
          << "INFO: Transfer to coarser or same grid requested. This is being ignored so that the solution is not being transferred at all."
          << std::endl;
      }

    {
      TimerOutput::Scope t(computing_timer, "vtu output");
      try
        {
          Tools::create_data_directory(parameters.dirname_output);
        }
      catch (std::runtime_error &e)
        {
          // No exception handling here.
        }
      output_results();
    }

    if (parameters.verbose)
      {
        computing_timer.print_summary();
        computing_timer.reset();
      }

    pcout << std::endl
          << "===========================================" << std::endl;
  }

} // end namespace RTDQ
