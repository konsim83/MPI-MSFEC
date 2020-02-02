#ifndef INCLUDE_BASIS_Q1_GRAD_H_
#define INCLUDE_BASIS_Q1_GRAD_H_

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


  template <int dim>
  class BasisQ1Grad : public Function<dim>
  {
  public:
    BasisQ1Grad() = delete;

    /*!
     * Constructor.
     * @param cell
     */
    BasisQ1Grad(const typename Triangulation<dim>::active_cell_iterator &cell);

    /*!
     * Copy constructor.
     *
     * @param basis
     */
    BasisQ1Grad(const BasisQ1Grad<dim> &);

    /*!
     * Set the index of the basis function to be evaluated.
     *
     * @param index
     */
    void
      set_index(unsigned int index);


    virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;


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
  BasisQ1Grad<2>::BasisQ1Grad(
    const typename Triangulation<2>::active_cell_iterator &cell);

  template <>
  BasisQ1Grad<3>::BasisQ1Grad(
    const typename Triangulation<3>::active_cell_iterator &cell);

  template <>
  void
    BasisQ1Grad<2>::vector_value(const Point<2> &p,
                                 Vector<double> &value) const;

  template <>
  void
    BasisQ1Grad<3>::vector_value(const Point<3> &p,
                                 Vector<double> &value) const;

  template <>
  void
    BasisQ1Grad<2>::vector_value_list(
      const std::vector<Point<2>> &points,
      std::vector<Vector<double>> &values) const;

  template <>
  void
    BasisQ1Grad<3>::vector_value_list(
      const std::vector<Point<3>> &points,
      std::vector<Vector<double>> &values) const;

  // exernal template instantiations
  extern template class BasisQ1Grad<2>;
  extern template class BasisQ1Grad<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_Q1_GRAD_H_ */
