#ifndef INCLUDE_SCHUR_COMPLEMENT_TPP_
#define INCLUDE_SCHUR_COMPLEMENT_TPP_

#include <deal.II/base/subscriptor.h>

#include <memory>
#include <vector>

#include "config.h"
#include "inverse_matrix.tpp"

namespace LinearSolvers
{
  using namespace dealii;

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class SchurComplement : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &relevant_inverse_matrix);

    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    const SmartPointer<const BlockMatrixType> system_matrix;
    const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
      relevant_inverse_matrix;

    mutable VectorType tmp1, tmp2, tmp3;
  };


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  SchurComplement<BlockMatrixType, VectorType, PreconditionerType>::
    SchurComplement(const BlockMatrixType &system_matrix,
                    const InverseMatrix<BlockType, PreconditionerType>
                      &relevant_inverse_matrix)
    : system_matrix(&system_matrix)
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
    , tmp3(system_matrix.block(1, 1).m())
  {}


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
  SchurComplement<BlockMatrixType, VectorType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;
  }


  //
  //
  //


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class SchurComplementMPI : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    SchurComplementMPI(const BlockMatrixType &system_matrix,
                       const InverseMatrix<BlockType, PreconditionerType>
                         &                          relevant_inverse_matrix,
                       const std::vector<IndexSet> &owned_partitioning,
                       MPI_Comm                     mpi_communicator);

    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    const SmartPointer<const BlockMatrixType> system_matrix;
    const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
      relevant_inverse_matrix;

    const std::vector<IndexSet> &owned_partitioning;

    MPI_Comm mpi_communicator;

    mutable VectorType tmp1, tmp2, tmp3;
  };


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  SchurComplementMPI<BlockMatrixType, VectorType, PreconditionerType>::
    SchurComplementMPI(const BlockMatrixType &system_matrix,
                       const InverseMatrix<BlockType, PreconditionerType>
                         &                          relevant_inverse_matrix,
                       const std::vector<IndexSet> &owned_partitioning,
                       MPI_Comm                     mpi_communicator)
    : system_matrix(&system_matrix)
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
    , tmp3(owned_partitioning[1], mpi_communicator)
  {}


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
  SchurComplementMPI<BlockMatrixType, VectorType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    system_matrix->block(1, 1).vmult(tmp3, src);
    dst -= tmp3;
  }


  ///////////////////////////////////////////////////
  ///////////////////////////////////////////////////



  // template <typename BlockMatrixType, typename VectorType, typename
  // PreconditionerType> class SchurComplement : public Subscriptor
  //{
  // private:
  //	using BlockType = typename BlockMatrixType::BlockType;
  //
  // public:
  //	SchurComplement (const BlockMatrixType &system_matrix,
  //				   const InverseMatrix<BlockType, PreconditionerType>
  //&relevant_inverse_matrix);
  //
  //	void vmult (VectorType       &dst,
  //			  const VectorType &src) const;
  //
  // private:
  //	const SmartPointer<const BlockMatrixType > system_matrix;
  //	const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
  //relevant_inverse_matrix;
  //
  //	mutable VectorType tmp1, tmp2, tmp3;
  //};
  //
  //
  // template <typename BlockMatrixType, typename VectorType, typename
  // PreconditionerType> SchurComplement<BlockMatrixType, VectorType,
  // PreconditionerType>::SchurComplement( 		const BlockMatrixType	&system_matrix,
  //		const InverseMatrix<BlockType, PreconditionerType>
  //&relevant_inverse_matrix)
  //:
  // system_matrix (&system_matrix),
  // relevant_inverse_matrix (&relevant_inverse_matrix),
  // tmp1 (system_matrix.block(1,1).m()),
  // tmp2 (system_matrix.block(1,1).m()),
  // tmp3 (system_matrix.block(0,0).m())
  //{}
  //
  //
  // template <typename BlockMatrixType, typename VectorType, typename
  // PreconditionerType> void SchurComplement<BlockMatrixType, VectorType,
  // PreconditionerType>::vmult( 		VectorType       &dst, 		const VectorType &src)
  //const
  //{
  //	system_matrix->block(1,0).vmult (tmp1, src);
  //	relevant_inverse_matrix->vmult (tmp2, tmp1);
  //	system_matrix->block(0,1).vmult (dst, tmp2);
  //	system_matrix->block(1,1).vmult(tmp3, src);
  //	dst -= tmp3;
  //}
  //
  //
  ////
  ////
  ////
  //
  //
  // template <typename BlockMatrixType, typename VectorType, typename
  // PreconditionerType> class SchurComplementMPI : public Subscriptor
  //{
  // private:
  //	using BlockType = typename BlockMatrixType::BlockType;
  //
  // public:
  //	SchurComplementMPI (const BlockMatrixType &system_matrix,
  //				   const InverseMatrix<BlockType, PreconditionerType>
  //&relevant_inverse_matrix, 				   const std::vector<IndexSet> &owned_partitioning,
  //				   MPI_Comm mpi_communicator);
  //
  //	void vmult (VectorType       &dst,
  //			  const VectorType &src) const;
  //
  // private:
  //	const SmartPointer<const BlockMatrixType > system_matrix;
  //	const SmartPointer<const InverseMatrix<BlockType, PreconditionerType>>
  //relevant_inverse_matrix;
  //
  //	const std::vector<IndexSet>& owned_partitioning;
  //
  //	MPI_Comm mpi_communicator;
  //
  //	mutable VectorType tmp1, tmp2, tmp3;
  //};


  // template <typename BlockMatrixType, typename VectorType, typename
  // PreconditionerType> SchurComplementMPI<BlockMatrixType, VectorType,
  // PreconditionerType>::SchurComplementMPI( 		const BlockMatrixType
  //&system_matrix, 		const InverseMatrix<BlockType, PreconditionerType>
  //&relevant_inverse_matrix, 		const std::vector<IndexSet> &owned_partitioning,
  //		MPI_Comm mpi_communicator)
  //:
  // system_matrix (&system_matrix),
  // relevant_inverse_matrix (&relevant_inverse_matrix),
  // owned_partitioning(owned_partitioning),
  // mpi_communicator(mpi_communicator),
  // tmp1 (owned_partitioning[1],
  //		mpi_communicator),
  // tmp2 (owned_partitioning[1],
  //		mpi_communicator),
  // tmp3 (owned_partitioning[0],
  //		mpi_communicator)
  //{}
  //
  //
  // template <typename BlockMatrixType, typename VectorType, typename
  // PreconditionerType> void SchurComplementMPI<BlockMatrixType, VectorType,
  // PreconditionerType>::vmult( 		VectorType       &dst, 		const VectorType &src)
  //const
  //{
  //	system_matrix->block(1,0).vmult (tmp1, src);
  //	relevant_inverse_matrix->vmult (tmp2, tmp1);
  //	system_matrix->block(0,1).vmult (dst, tmp2);
  //	system_matrix->block(1,1).vmult(tmp3, src);
  //	dst -= tmp3;
  //}


} // end namespace LinearSolvers

#endif /* INCLUDE_SCHUR_COMPLEMENT_TPP_ */
