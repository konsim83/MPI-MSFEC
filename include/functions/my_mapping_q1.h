#ifndef INCLUDE_FUNCTIONS_MY_MAPPING_Q1_H_
#define INCLUDE_FUNCTIONS_MY_MAPPING_Q1_H_

// Deal.ii
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// STL
#include <vector>

// My Headers

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  class MyMappingQ1
  {
  public:
    MyMappingQ1() = delete;

    MyMappingQ1(const typename Triangulation<dim>::active_cell_iterator &cell);

    MyMappingQ1(const MyMappingQ1<dim> &);

    Point<dim>
      map_real_to_unit_cell(const Point<dim> &p) const;

    void
      map_real_to_unit_cell(const std::vector<Point<dim>> &points_in,
                            std::vector<Point<dim>> &points_out) const;

    Point<dim>
      map_unit_cell_to_real(const Point<dim> &p) const;

    void
      map_unit_cell_to_real(const std::vector<Point<dim>> &points_in,
                            std::vector<Point<dim>> &points_out) const;

  private:
    /*!
     * Matrix holds coefficients of Q1 mapping (unit cell to physical cell).
     */
    FullMatrix<double> coeff_matrix;

    /*!
     * Matrix holds coefficients of inverse Q1 mapping (physical cell to unit
     * cell).
     */
    FullMatrix<double> coeff_matrix_inverse;
  };


  /*
   * 2D declarations of specializations
   */
  template <>
  MyMappingQ1<2>::MyMappingQ1(
    const typename Triangulation<2>::active_cell_iterator &cell);

  template <>
  Point<2>
    MyMappingQ1<2>::map_real_to_unit_cell(const Point<2> &p) const;

  template <>
  void
    MyMappingQ1<2>::map_real_to_unit_cell(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &points_out) const;

  template <>
  Point<2>
    MyMappingQ1<2>::map_unit_cell_to_real(const Point<2> &p) const;

  template <>
  void
    MyMappingQ1<2>::map_unit_cell_to_real(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &points_out) const;

  /*
   * 3D declarations of specializations
   */
  template <>
  MyMappingQ1<3>::MyMappingQ1(
    const typename Triangulation<3>::active_cell_iterator &cell);

  template <>
  Point<3>
    MyMappingQ1<3>::map_real_to_unit_cell(const Point<3> &p) const;

  template <>
  void
    MyMappingQ1<3>::map_real_to_unit_cell(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &points_out) const;

  template <>
  Point<3>
    MyMappingQ1<3>::map_unit_cell_to_real(const Point<3> &p) const;

  template <>
  void
    MyMappingQ1<3>::map_unit_cell_to_real(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &points_out) const;


  /*
   * exernal template instantiations
   */
  extern template class MyMappingQ1<2>;
  extern template class MyMappingQ1<3>;

} // namespace ShapeFun



#endif /* INCLUDE_FUNCTIONS_MY_MAPPING_Q1_H_ */
