#ifndef INCLUDE_BASIS_NEDELEC_H_
#define INCLUDE_BASIS_NEDELEC_H_

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_nedelec.h>
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
   * @class BasisNedelec
   *
   * Class implements gradients of scalar \f$Q_1\f$-basis functions for a given
   * quadrilateral.
   */
  template <int dim>
  class BasisNedelec : public Function<dim>, public MyMappingQ1<dim>
  {
  public:
    BasisNedelec() = delete;

    /*!
     * Constructor.
     * @param cell
     */
    BasisNedelec(const typename Triangulation<dim>::active_cell_iterator &cell,
                 unsigned int degree = 0);

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
    FE_Nedelec<dim> fe;

    unsigned int index_basis;
  };

  // exernal template instantiations
  extern template class BasisNedelec<2>;
  extern template class BasisNedelec<3>;

} // namespace ShapeFun

#endif /* INCLUDE_BASIS_NEDELEC_H_ */
