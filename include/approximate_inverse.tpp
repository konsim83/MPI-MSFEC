#ifndef INCLUDE_APPROXIMATE_INVERSE_TPP_
#define INCLUDE_APPROXIMATE_INVERSE_TPP_

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <memory>

#include "config.h"


namespace LinearSolvers
{
using namespace dealii;

template <typename MatrixType, typename PreconditionerType>
class ApproximateInverseMatrix : public Subscriptor
{
public:
	ApproximateInverseMatrix (const MatrixType &m,
			 const PreconditionerType &preconditioner,
			 const unsigned int n_iter);

	template <typename VectorType>
	void vmult(VectorType &dst, const VectorType &src) const;

private:
	const SmartPointer<const MatrixType> matrix;
	const PreconditionerType& preconditioner;
	const unsigned int max_iter;
};


template <typename MatrixType, typename PreconditionerType>
ApproximateInverseMatrix<MatrixType, PreconditionerType>::ApproximateInverseMatrix(
		const MatrixType &m,
		const PreconditionerType &preconditioner,
		const unsigned int n_iter)
:
matrix (&m),
preconditioner (preconditioner),
max_iter(n_iter)
{
}


template <typename MatrixType, typename PreconditionerType>
template <typename VectorType>
void
ApproximateInverseMatrix<MatrixType, PreconditionerType>::vmult(
		VectorType       &dst,
		const VectorType &src) const
{
	SolverControl solver_control (/* max_iter */ max_iter,
									1e-6*src.l2_norm());
	SolverCG<VectorType> local_solver(solver_control);

	dst = 0;

	try
	{
		local_solver.solve(*matrix, dst, src, preconditioner);
	}
	catch (std::exception &e)
	{
		Assert(false, ExcMessage(e.what()));
	}
}

} // end namespace LinearSolvers

#endif /* INCLUDE_APPROXIMATE_INVERSE_TPP_ */
