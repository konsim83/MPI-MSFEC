#ifndef NED_RT_BASIS_H_
#define NED_RT_BASIS_H_

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
#include <vector>

// my headers
#include <Ned_RT/ned_rt_parameters.h>
#include <Ned_RT/ned_rt_post_processor.h>
#include <equation_data/eqn_boundary_vals.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>
#include <equation_data/eqn_coeff_R.h>
#include <equation_data/eqn_exact_solution_lin.h>
#include <equation_data/eqn_rhs.h>
#include <functions/concatinate_functions.h>
#include <functions/vector_shape_function.h>
#include <functions/vector_shape_function_curl.h>
#include <functions/vector_shape_function_div.h>
#include <linear_algebra/approximate_inverse.h>
#include <linear_algebra/approximate_schur_complement.tpp>
#include <linear_algebra/inverse_matrix.h>
#include <linear_algebra/preconditioner.h>
#include <linear_algebra/schur_complement.tpp>
#include <vector_tools/my_vector_tools.h>

namespace NedRT
{
  using namespace dealii;

  class NedRTBasis
  {
  public:
    /*
     * Constructor.
     */
    NedRTBasis() = delete;

    /**
     * Constructor.
     *
     * @param parameters_ms
     * @param parameter_filename
     * @param global_cell
     * @param first_cell
     * @param local_subdomain
     * @param mpi_communicator
     */
    NedRTBasis(const ParametersMs &parameters_ms,
               const std::string & parameter_filename,
               typename Triangulation<3>::active_cell_iterator &global_cell,
               CellId                                           first_cell,
               unsigned int                                     local_subdomain,
               MPI_Comm mpi_communicator);

    /**
     * Copy constructor.
     *
     * @param other
     */
    NedRTBasis(const NedRTBasis &other);
    ~NedRTBasis();

    /**
     * Compute the basis.
     */
    void
      run();

    /**
     * Write vtu file for solution in cell.
     */
    void
      output_global_solution_in_cell();

    /**
     * Get reference to global multiscale element matrix.
     *
     * @return
     */
    const FullMatrix<double> &
      get_global_element_matrix() const;

    /**
     * * Get reference to global multiscale element rhs.
     *
     * @return
     */
    const Vector<double> &
      get_global_element_rhs() const;

    /**
     * Get global filename.
     * @return
     */
    const std::string &
      get_filename_global() const;

    /**
     * Set the global (coarse) weight after coarse solution is
     * computed.
     *
     * @param global_weights
     */
    void
      set_global_weights(const std::vector<double> &global_weights);

  private:
    void
      setup_grid();
    void
      setup_system_matrix();

    /**
     * Setup the constraints for H(curl)-basis.
     */
    void
      setup_basis_dofs_curl();

    /**
     * Setup the constraints for H(div)-basis.
     */
    void
      setup_basis_dofs_div();

    /**
     * Assemble local system.
     */
    void
      assemble_system();

    /**
     * Build the global multiscale element matrix.
     */
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

    /**
     * Can not be used yet since ther is no TrilinosWrapper for the direct
     * solver for block matrices. Does not scale well anyway for large
     * problems.
     *
     * @param n_basis
     */
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

    /**
     * Write the multiscale basis as vtu.
     */
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
    std::vector<AffineConstraints<double>> constraints_div_v;

    // Sparsity patterns and system matrices for each basis
    BlockSparsityPattern sparsity_pattern;

    BlockSparseMatrix<double> assembled_matrix;
    BlockSparseMatrix<double> system_matrix;

    std::vector<BlockVector<double>> basis_curl_v;
    std::vector<BlockVector<double>> basis_div_v;

    std::vector<BlockVector<double>> system_rhs_curl_v;
    std::vector<BlockVector<double>> system_rhs_div_v;
    BlockVector<double>              global_rhs;

    FullMatrix<double>  global_element_matrix;
    Vector<double>      global_element_rhs;
    std::vector<double> global_weights;

    BlockVector<double> global_solution;
    BlockVector<double> exact_solution_in_cell;

    // Shared pointer to preconditioner type for each system matrix
    std::shared_ptr<typename LinearSolvers::LocalInnerPreconditioner<3>::type>
      inner_schur_preconditioner;

    /**
     * Global cell identifier.
     */
    CellId global_cell_id;

    /**
     * Global cell identifier of first cell.
     */
    CellId first_cell;

    /**
     * Global cell iterator.
     */
    typename Triangulation<3>::active_cell_iterator global_cell_it;

    /**
     * Global subdomain number.
     */
    const unsigned int local_subdomain;

    // Geometry info
    double                volume_measure;
    std::vector<double>   face_measure;
    std::vector<double>   edge_measure;
    std::vector<Point<3>> corner_points;

    unsigned int length_system_basis;

    bool is_built_global_element_matrix;
    bool is_set_global_weights;
    bool is_set_cell_data;

    bool is_copyable;
  };

} // end namespace NedRT

#endif /* NED_RT_BASIS_H_ */