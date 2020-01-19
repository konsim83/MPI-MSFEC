#ifndef INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_H_
#define INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_H_

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

// STL
#include <memory>

// my headers
#include <config.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <typename MatrixType, typename PreconditionerType>
  class ApproximateInverseMatrix : public Subscriptor
  {
  public:
    ApproximateInverseMatrix(const MatrixType &        m,
                             const PreconditionerType &preconditioner,
                             const unsigned int        n_iter);

    template <typename VectorType>
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;
    const PreconditionerType &           preconditioner;
    const unsigned int                   max_iter;
  };

} // end namespace LinearSolvers

#include <linear_algebra/approximate_inverse.tpp>

#endif /* INCLUDE_LINEAR_ALGEBRA_APPROXIMATE_INVERSE_H_ */
