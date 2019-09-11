#ifndef INCLUDE_INVERSE_MATRIX_TPP_
#define INCLUDE_INVERSE_MATRIX_TPP_

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
class InverseMatrix : public Subscriptor
{
public:
	InverseMatrix (const MatrixType &m,
			 const PreconditionerType &preconditioner);

	template <typename VectorType>
	void vmult(VectorType &dst, const VectorType &src) const;

private:
	const SmartPointer<const MatrixType> matrix;
	const PreconditionerType& preconditioner;
};


template <typename MatrixType, typename PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
		const MatrixType &m,
		const PreconditionerType &preconditioner)
:
matrix (&m),
preconditioner (preconditioner)
{
}


template <typename MatrixType, typename PreconditionerType>
template <typename VectorType>
void
InverseMatrix<MatrixType, PreconditionerType>::vmult(
		VectorType       &dst,
		const VectorType &src) const
{
	SolverControl solver_control (std::max(static_cast<std::size_t> (src.size()),
											static_cast<std::size_t> (1000)),
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

#endif /* INCLUDE_INVERSE_MATRIX_TPP_ */
