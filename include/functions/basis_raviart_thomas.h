#ifndef INCLUDE_BASIS_RAVIART_THOMAS_H_
#define INCLUDE_BASIS_RAVIART_THOMAS_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include <functions/my_mapping_q1.h>

namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class BasisRaviartThomas
   *
   * Class implements gradients of scalar \f$Q_1\f$-basis functions for a given
   * quadrilateral.
   */
  template <int dim>
  class BasisRaviartThomas : public Function<dim>
  {
  public:
    BasisRaviartThomas() = delete;

    /*!
     * Constructor.
     * @param cell
     */
    BasisRaviartThomas(const typename Triangulation<dim>::active_cell_iterator &cell,
                 unsigned int degree = 0);

    /*!
     * Copy constructor.
     */
    BasisRaviartThomas(BasisRaviartThomas<dim> &basis);

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
    MyMappingQ1<dim> mapping;

    FE_RaviartThomas<dim> fe;

    unsigned int index_basis;
  };

  // exernal template instantiations
  extern template class BasisRaviartThomas<2>;
  extern template class BasisRaviartThomas<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_RAVIART_THOMAS_H_ */
