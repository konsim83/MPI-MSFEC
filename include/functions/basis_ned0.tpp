#ifndef INCLUDE_BASIS_NED0_TPP_
#define INCLUDE_BASIS_NED0_TPP_

#include <functions/basis_q1_grad.h>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  BasisNed0<dim>::BasisNed0(const BasisNed0<dim> &basis)
    : Function<dim>(dim)
    , index_basis(0)
    , coeff_matrix(basis.coeff_matrix)
  {}


  template <>
  BasisNed0<2>::BasisNed0(
    const typename Triangulation<2>::active_cell_iterator &cell)
    : Function<2>(2)
    , index_basis(0)
    , coeff_matrix(4, 4)
  {
    FullMatrix<double> point_matrix(4, 4);

    for (unsigned int i = 0; i < 4; ++i)
      {
        const Point<2> &p = cell->vertex(i);

        point_matrix(i, 0) = 1;
        point_matrix(i, 1) = p(0);
        point_matrix(i, 2) = p(1);
        point_matrix(i, 3) = p(0) * p(1);
      }

    // Columns of coeff_matrix are the coefficients of the polynomial
    coeff_matrix.invert(point_matrix);
  }


  template <>
  BasisNed0<3>::BasisNed0(
    const typename Triangulation<3>::active_cell_iterator &cell)
    : Function<3>(3)
    , index_basis(0)
    , coeff_matrix(8, 8)
  {
    FullMatrix<double> point_matrix(8, 8);

    for (unsigned int i = 0; i < 8; ++i)
      {
        const Point<3> &p = cell->vertex(i);

        point_matrix(i, 0) = 1;
        point_matrix(i, 1) = p(0);
        point_matrix(i, 2) = p(1);
        point_matrix(i, 3) = p(2);
        point_matrix(i, 4) = p(0) * p(1);
        point_matrix(i, 5) = p(1) * p(2);
        point_matrix(i, 6) = p(0) * p(2);
        point_matrix(i, 7) = p(0) * p(1) * p(2);
      }

    // Columns of coeff_matrix are the coefficients of the polynomial
    coeff_matrix.invert(point_matrix);
  }


  template <int dim>
  void
    BasisNed0<dim>::set_index(unsigned int index)
  {
    index_basis = index;
  }


  template <>
  void
    BasisNed0<2>::vector_value(const Point<2> &p, Vector<double> &value) const
  {
    value(0) =
      coeff_matrix(1, index_basis) + coeff_matrix(3, index_basis) * p(1);
    value(1) =
      coeff_matrix(2, index_basis) + coeff_matrix(3, index_basis) * p(0);
  }


  template <>
  void
    BasisNed0<3>::vector_value(const Point<3> &p, Vector<double> &value) const
  {
    value(0) = coeff_matrix(1, index_basis) +
               coeff_matrix(4, index_basis) * p(1) +
               coeff_matrix(6, index_basis) * p(2) +
               coeff_matrix(7, index_basis) * p(1) * p(2);
    value(1) = coeff_matrix(2, index_basis) +
               coeff_matrix(4, index_basis) * p(0) +
               coeff_matrix(5, index_basis) * p(2) +
               coeff_matrix(7, index_basis) * p(0) * p(2);
    value(2) = coeff_matrix(3, index_basis) +
               coeff_matrix(5, index_basis) * p(1) +
               coeff_matrix(6, index_basis) * p(0) +
               coeff_matrix(7, index_basis) * p(0) * p(1);
  }


  template <>
  void
    BasisNed0<2>::vector_value_list(const std::vector<Point<2>> &points,
                                    std::vector<Vector<double>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        vector_value(points[p], values[p]);
      } // end ++p
  }


  template <>
  void
    BasisNed0<3>::vector_value_list(const std::vector<Point<3>> &points,
                                    std::vector<Vector<double>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        vector_value(points[p], values[p]);
      } // end ++p
  }

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NED0_TPP_ */
