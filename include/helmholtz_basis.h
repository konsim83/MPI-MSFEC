#ifndef HELMHOLTZ_BASIS_H_
#define HELMHOLTZ_BASIS_H_

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

// triangulation
#include <deal.II/grid/tria.h>
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
//#include "shape_fun_vector.tpp"
//#include "shape_fun_vector_curl.tpp"
//#include "shape_fun_vector_div.tpp"
//#include "shape_fun_concatinate_functions.tpp"


namespace HelmholtzProblem
{

using namespace dealii;

class NedRTBasis
{
	public:
		struct Parameters
		{
			Parameters();

			bool verbose;
			bool use_direct_solver; /* This is often better for 2D problems. */
			bool renumber_dofs; /* Reduce bandwidth in either system component */

			bool output_flag;

			unsigned int n_refine_global;
			unsigned int n_refine_local;

			std::string filename_global;
		};
		NedRTBasis () = delete;
		NedRTBasis (unsigned int n_refine_local,
					typename Triangulation<3>::active_cell_iterator& global_cell,
					unsigned int local_subdomain,
					MPI_Comm mpi_communicator);
		NedRTBasis (const NedRTBasis &other);
		~NedRTBasis ();

		void run ();
		void output_global_solution_in_cell () const;

		// Getter

		const FullMatrix<double>& get_global_element_matrix () const;
		const Vector<double>& get_global_element_rhs () const;
		const std::string& get_filename_global () const;

		// Setter
		void set_global_weights (const std::vector<double> &global_weights);
		void set_output_flag (bool flag);

	private:
		void setup_grid ();
		void setup_system_matrix ();

		void setup_basis_dofs_curl ();
		void setup_basis_dofs_div ();

		void assemble_system ();
		void assemble_global_element_matrix ();

		// Private setters
		void set_u_to_std ();
		void set_sigma_to_std ();
		void set_filename_global_solution ();

		// Solver routines
		void solve_direct (unsigned int n_basis);
		void solve_iterative_preconditioned (unsigned int n_basis);

		void output_basis (unsigned int n_basis);

		Parameters &parameters;

		Triangulation<3>   triangulation;

		FESystem<3>        fe;

		DoFHandler<3>      dof_handler;

		// Constraints for each basis
		std::vector<AffineConstraints> 		  	constraints_curl_v;
		std::vector<AffineConstraints> 		  	constraints_div_v;

		// Sparsity patterns and system matrices for each basis
		BlockSparsityPattern     sparsity_pattern_curl;
		BlockSparsityPattern     sparsity_pattern_div;

		BlockSparseMatrix<double> 	assembled_matrix;
		BlockSparseMatrix<double> 	system_matrix;


		std::vector<BlockVector<double>>       basis_v;

		BlockVector<double>       system_rhs;
		BlockVector<double>       global_rhs;


		// These are only the sparsity pattern and system_matrix for later use

		FullMatrix<double>   		global_element_matrix;
		Vector<double>   			global_element_rhs;
		std::vector<double> 		global_weights;

		BlockVector<double>			global_solution;

		// Shared pointer to preconditioner type for each system matrix
		std::shared_ptr<typename InnerPreconditioner<3>::type> InnerPreconditioner;

		typename Triangulation<3>::active_cell_iterator global_cell;


		/*!
		 * Global cell number.
		 */
		const CellId global_cell_id;

		/*!
		 * Global subdomain number.
		 */
		const unsigned int local_subdomain;


		// Geometry info
		double volume_measure;
		std::vector<double> face_measure;
		std::vector<double> edge_measure;

		std::vector<Point<3>> corner_points;

		unsigned int length_system_basis;

		bool is_built_global_element_matrix;
		bool is_set_global_weights;
		bool is_set_cell_data;
};

} // end namespace HelmholtzProblem

#endif /* HELMHOLTZ_BASIS_H_ */
