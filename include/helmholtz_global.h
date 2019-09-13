#ifndef HELMHOLTZ_GLOBAL_H_
#define HELMHOLTZ_GLOBAL_H_

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
// For distributing the sparsity pattern.
#include <deal.II/lac/sparsity_tools.h>

// Distributed triangulation
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/grid/cell_id.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

// std library
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <memory>

// my headers
#include "config.h"
#include "helmholtz_eqn_data.h"

#include "inverse_matrix.tpp"
#include "schur_complement.tpp"
#include "approximate_schur_complement.tpp"
#include "preconditioner.h"
#include "helmholtz_basis.h"


namespace HelmholtzProblem
{
using namespace dealii;


class NedRTMultiscale
{
public:
	struct Parameters
	{
		Parameters(const std::string &parameter_filename);

		static void declare_parameters(ParameterHandler &prm);
		void        parse_parameters(ParameterHandler &prm);

		bool compute_solution;
		bool verbose;
		bool verbose_basis;
		bool use_direct_solver; /* This is often better for 2D problems. */
		bool use_direct_solver_basis; /* This is often better for 2D problems. */
		bool renumber_dofs; /* Reduce bandwidth in either system component */

		unsigned int n_refine_global;
		unsigned int n_refine_local;

		std::string filename_output;
	};
	NedRTMultiscale (Parameters &parameters_);
	~NedRTMultiscale ();

	void run ();

private:
	void setup_grid ();
	void initialize_and_compute_basis ();
	void setup_system_matrix ();
	void setup_constraints ();
	void assemble_system ();
	void solve_direct ();
	void solve_iterative ();
	void send_global_weights_to_cell ();

	std::vector<std::string> collect_filenames_on_mpi_process ();
	void output_results_coarse () const;
	void output_results_fine ();

	MPI_Comm mpi_communicator;

	Parameters &parameters;

	parallel::distributed::Triangulation<3> triangulation;

	// Modified finite element
	FESystem<3>        fe;

	// Modified DoFHandler
	DoFHandler<3>      dof_handler;

	IndexSet locally_relevant_dofs;
	std::vector<IndexSet> owned_partitioning;
	std::vector<IndexSet> relevant_partitioning;

	// Constraint matrix holds boundary conditions
	AffineConstraints<double> 		constraints;

	/*!
	 * Distributed system matrix.
	 */
	LA::MPI::BlockSparseMatrix 		system_matrix;

	/*!
	 * Solution vector containing weights at the dofs.
	 */
	LA::MPI::BlockVector       		locally_relevant_solution;

	/*!
	 * Contains all parts of the right-hand side needed to
	 * solve the linear system.
	 */
	LA::MPI::BlockVector       		system_rhs;

	ConditionalOStream 		pcout;
	TimerOutput        		computing_timer;

	std::shared_ptr<typename LinearSolvers::InnerPreconditioner<3>::type> 	inner_schur_preconditioner;

	/*!
	 * STL Vector holding basis functions for each coarse cell.
	 */
	using BasisMap = std::map<CellId, NedRTBasis>;
	BasisMap cell_basis_map;
};

} // end namespace HelmholtzProblem


#endif /* HELMHOLTZ_GLOBAL_H_ */
