#ifndef HELMHOLTZ_PRECON_H_
#define HELMHOLTZ_PRECON_H_

#include "config.h"
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <int dim>
  class InnerPreconditioner
  {
  public:
    // Parallell, generic
    //	using type = LA::MPI::PreconditionAMG;
    using type = LA::MPI::PreconditionILU; // Turns out to be the best
                                           //	using type = PreconditionIdentity;

    // Parallel, Petsc
    //	tyename PETScWrappers::PreconditionNone type;

    // Parallel, Petsc
    //	typename TrilinosWrappers::PreconditionIdentity type;

    // Serial
    //  typedef SparseDirectUMFPACK type;
    //	typedef SparseILU<double> type;
    //	typedef PreconditionIdentity type;
  };

  template <int dim>
  class LocalInnerPreconditioner;

  template <>
  class LocalInnerPreconditioner<2>
  {
  public:
    using type = SparseDirectUMFPACK;
  };

  template <>
  class LocalInnerPreconditioner<3>
  {
  public:
    using type = SparseILU<double>;
  };

} // namespace LinearSolvers

#endif /* HELMHOLTZ_PRECON_H_ */
