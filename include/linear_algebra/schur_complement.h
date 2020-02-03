#ifndef INCLUDE_LINEAR_ALGEBRA_SCHUR_COMPLEMENT_H_
#define INCLUDE_LINEAR_ALGEBRA_SCHUR_COMPLEMENT_H_

#include "config.h"
#include <deal.II/base/subscriptor.h>
#include <linear_algebra/inverse_matrix.h>

#include <memory>
#include <vector>

namespace LinearSolvers
{
  using namespace dealii;

  /*!
   * @class SchurComplement
   *
   * @brief Implements a serial Schur complement
   *
   * Implements a serial Schur complement through the use of an inner inverse
   * matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & C
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		\sigma \\
   *		u
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		0 \\
   *		u
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we first define the inverse and
   *define the Schur complement as \f{eqnarray}{ \tilde S = C - BP_A^{-1}B^T \f}
   *to solve for \f$u\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class SchurComplement : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left block of the system matrix.
     *
     * @param system_matrix
     * 	Block Matrix
     * @param relevant_inverse_matrix
     * 	Inverse of upper left block of the system matrix.
     */
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &relevant_inverse_matrix);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Smart pointer to system matrix.
     */
    const SmartPointer<const BlockMatrixType> system_matrix;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
      relevant_inverse_matrix;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2, tmp3;
  };

  /*
   * Now the MPI version
   */

  /*!
   * @class SchurComplementMPI
   *
   * @brief Implements a MPI parallel Schur complement
   *
   * Like the SchurComplement class just MPI parallel.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class SchurComplementMPI : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left block of the system matrix.
     *
     * @param system_matrix
     * 	Block Matrix
     * @param relevant_inverse_matrix
     * 	Inverse of upper left block of the system matrix.
     * @param owned_partitioning
     * @param mpi_communicator
     */
    SchurComplementMPI(const BlockMatrixType &system_matrix,
                       const InverseMatrix<BlockType, PreconditionerType>
                         &                          relevant_inverse_matrix,
                       const std::vector<IndexSet> &owned_partitioning,
                       MPI_Comm                     mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
      vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Smart pointer to system matrix.
     */
    const SmartPointer<const BlockMatrixType> system_matrix;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
      relevant_inverse_matrix;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2, tmp3;
  };

} // end namespace LinearSolvers

#include <linear_algebra/schur_complement.tpp>

#endif /* INCLUDE_LINEAR_ALGEBRA_SCHUR_COMPLEMENT_H_ */
