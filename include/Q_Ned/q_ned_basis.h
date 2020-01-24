#ifndef Q_NED_BASIS_H_
#define Q_NED_BASIS_H_

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// STL
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

// my headers
#include <Q_Ned/q_ned_parameters.h>
#include <Q_Ned/q_ned_post_processor.h>
#include <config.h>
#include <equation_data/eqn_boundary_vals.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>
#include <equation_data/eqn_coeff_R.h>
#include <equation_data/eqn_exact_solution_lin.h>
#include <equation_data/eqn_rhs.h>
#include <functions/concatinate_functions.h>
#include <functions/scalar_shape_function.h>
#include <functions/scalar_shape_function_grad.h>
#include <functions/vector_shape_function.h>
#include <linear_algebra/approximate_inverse.h>
#include <linear_algebra/approximate_schur_complement.tpp>
#include <linear_algebra/inverse_matrix.h>
#include <linear_algebra/preconditioner.h>
#include <linear_algebra/schur_complement.tpp>
#include <vector_tools/my_vector_tools.h>

namespace QNed
{
  using namespace dealii;

  class QNedBasis
  {
  public:
    QNedBasis() = delete;
    QNedBasis(const ParametersMs &parameters_ms,
              const std::string & parameter_filename,
              typename Triangulation<3>::active_cell_iterator &global_cell,
              CellId                                           first_cell,
              unsigned int                                     local_subdomain,
              MPI_Comm mpi_communicator);
    QNedBasis(const QNedBasis &other);
    ~QNedBasis();

    void
      run();
    void
      output_global_solution_in_cell() const;

    // Getter

    const FullMatrix<double> &
      get_global_element_matrix() const;
    const Vector<double> &
      get_global_element_rhs() const;
    const std::string &
      get_filename_global() const;

    // Setter
    void
      set_global_weights(const std::vector<double> &global_weights);

  private:
    void
      setup_grid();
    void
      setup_system_matrix();

    void
      setup_basis_dofs_curl();
    void
      setup_basis_dofs_h1();

    void
      assemble_system();
    void
      assemble_global_element_matrix();

    // Private setters
    void
      set_output_flag();
    void
      set_u_to_std();
    void
      set_sigma_to_std();
    void
      set_filename_global();
    void
      set_cell_data();

    // Solver routines
    void
      solve_direct(unsigned int n_basis);

    /**
     * Schur complement solver with inner and outer preconditioner.
     *
     * @param n_basis
     */
    void
      solve_iterative(unsigned int n_basis);

    /**
     * Project the exact solution onto the local fe space.
     */
    void
      write_exact_solution_in_cell();

    void
      output_basis();

    MPI_Comm mpi_communicator;

    ParametersBasis    parameters;
    const std::string &parameter_filename;

    Triangulation<3> triangulation;

    FESystem<3> fe;

    DoFHandler<3> dof_handler;

    // Constraints for each basis
    std::vector<AffineConstraints<double>> constraints_curl_v;
    std::vector<AffineConstraints<double>> constraints_h1_v;

    // Sparsity patterns and system matrices for each basis
    BlockSparsityPattern sparsity_pattern;
    //		BlockSparsityPattern     sparsity_pattern_curl;
    //		BlockSparsityPattern     sparsity_pattern_div;

    BlockSparseMatrix<double> assembled_matrix;
    BlockSparseMatrix<double> system_matrix;

    std::vector<BlockVector<double>> basis_curl_v;
    std::vector<BlockVector<double>> basis_h1_v;

    std::vector<BlockVector<double>> system_rhs_curl_v;
    std::vector<BlockVector<double>> system_rhs_h1_v;
    BlockVector<double>              global_rhs;

    // These are only the sparsity pattern and system_matrix for later use

    FullMatrix<double>  global_element_matrix;
    Vector<double>      global_element_rhs;
    std::vector<double> global_weights;

    BlockVector<double> global_solution;
    BlockVector<double> exact_solution_in_cell;

    // Shared pointer to preconditioner type for each system matrix
    std::shared_ptr<typename LinearSolvers::LocalInnerPreconditioner<3>::type>
      inner_schur_preconditioner;

    /*!
     * Global cell number.
     */
    CellId global_cell_id;

    /*!
     * Global cell number of first cell.
     */
    CellId first_cell;

    /*!
     * Global cell iterator.
     */
    typename Triangulation<3>::active_cell_iterator global_cell_it;

    /*!
     * Global subdomain number.
     */
    const unsigned int local_subdomain;

    // Geometry info
    double              volume_measure;
    std::vector<double> face_measure;
    std::vector<double> edge_measure;

    std::vector<Point<3>> corner_points;

    unsigned int length_system_basis;

    bool is_built_global_element_matrix;
    bool is_set_global_weights;
    bool is_set_cell_data;

    bool is_copyable;
  };

} // end namespace QNed

#endif /* Q_NED_BASIS_H_ */
