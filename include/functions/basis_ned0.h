#ifndef INCLUDE_BASIS_NED0_H_
#define INCLUDE_BASIS_NED0_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// STL
#include <cmath>
#include <fstream>

// My Headers

namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class BasisNed0
   *
   * Class implements \f$Ned_0\f$-basis functions for a given
   * quadrilateral.
   */
  template <int dim>
  class BasisNed0 : public Function<dim>
  {
  public:
    BasisNed0() = delete;

    /*!
     * Constructor.
     * @param cell
     */
    BasisNed0(const typename Triangulation<dim>::active_cell_iterator &cell);

    /*!
     * Copy constructor.
     *
     * @param basis
     */
    BasisNed0(const BasisNed0<dim> &);

    /*!
     * Set the index of the basis function to be evaluated.
     *
     * @param index
     */
    void
      set_index(unsigned int index);

    /*!
     * Evaluate a basis function with a preset index at one given point in 2D or
     * 3D.
     *
     * @param p
     * @param component
     */
    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Evaluate a basis function with a preset index at given point list in 2D
     * and 3D.
     *
     * @param p
     * @param component
     */
    virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &  values) const override;

  private:
    /*!
     * Index of current basis function to be evaluated.
     */
    unsigned int index_basis;

    /*!
     * Matrix columns hold coefficients of basis functions.
     */
    FullMatrix<double> coeff_matrix;
  };

  // declare specializations
  template <>
  BasisNed0<2>::BasisNed0(
    const typename Triangulation<2>::active_cell_iterator &cell);

  template <>
  BasisNed0<3>::BasisNed0(
    const typename Triangulation<3>::active_cell_iterator &cell);

  template <>
  void
    BasisNed0<2>::vector_value(const Point<2> &p, Vector<double> &value) const;

  template <>
  void
    BasisNed0<3>::vector_value(const Point<3> &p, Vector<double> &value) const;

  template <>
  void
    BasisNed0<2>::vector_value_list(const std::vector<Point<2>> &points,
                                    std::vector<Vector<double>> &values) const;

  template <>
  void
    BasisNed0<3>::vector_value_list(const std::vector<Point<3>> &points,
                                    std::vector<Vector<double>> &values) const;

  // exernal template instantiations
  extern template class BasisNed0<2>;
  extern template class BasisNed0<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NED0_H_ */
