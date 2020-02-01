#ifndef INCLUDE_FUNCTIONS_MY_MAPPING_Q1_TPP_
#define INCLUDE_FUNCTIONS_MY_MAPPING_Q1_TPP_

#include <functions/my_mapping_q1.h>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  MyMappingQ1<dim>::MyMappingQ1(const MyMappingQ1<dim> &mapping)
    : coeff_matrix(mapping.coeff_matrix)
    , coeff_matrix_inverse(mapping.coeff_matrix_inverse)
  {}


  /*
   * 2D implementations
   */
  template <>
  MyMappingQ1<2>::MyMappingQ1(
    const typename Triangulation<2>::active_cell_iterator &cell)
    : coeff_matrix(2, 4)
    , coeff_matrix_inverse(2, 4)
  {
    FullMatrix<double> point_matrix(4, 4);
    FullMatrix<double> point_matrix_inverse(4, 4);
    FullMatrix<double> rhs_matrix(2, 4);

    for (unsigned int alpha = 0; alpha < 4; ++alpha)
      {
        const Point<2> &p     = cell->vertex(alpha);
        const Point<2> &p_ref = GeometryInfo<2>::unit_cell_vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = p(0);
        point_matrix(2, alpha) = p(1);
        point_matrix(3, alpha) = p(0) * p(1);

        // this is rhs if we want mapping from ref cell to real cell
        rhs_matrix(0, alpha) = p_ref(0);
        rhs_matrix(1, alpha) = p_ref(1);
      }

    // Columns of coeff_matrix are the coefficients of the polynomial
    point_matrix_inverse.invert(point_matrix);
    rhs_matrix.mmult(/* destination */ coeff_matrix_inverse,
                     point_matrix_inverse);

    // clear matrixces
    point_matrix         = 0;
    point_matrix_inverse = 0;
    rhs_matrix           = 0;

    // Now with roles swtiched for inverse mapping
    for (unsigned int alpha = 0; alpha < 4; ++alpha)
      {
        const Point<2> &p     = cell->vertex(alpha);
        const Point<2> &p_ref = GeometryInfo<2>::unit_cell_vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = p_ref(0);
        point_matrix(2, alpha) = p_ref(1);
        point_matrix(3, alpha) = p_ref(0) * p_ref(1);

        // this is rhs if we want mapping from ref cell to real cell
        rhs_matrix(0, alpha) = p(0);
        rhs_matrix(1, alpha) = p(1);
      }

    // Columns of coeff_matrix are the coefficients of the polynomial
    point_matrix_inverse.invert(point_matrix);
    rhs_matrix.mmult(/* destination */ coeff_matrix, point_matrix_inverse);
  }


  template <>
  Point<2>
    MyMappingQ1<2>::map_real_to_unit_cell(const Point<2> &p) const
  {
    Point<2> p_out;

    for (unsigned d = 0; d < 2; ++d)
      p_out(d) = coeff_matrix_inverse(d, 0) +
                 coeff_matrix_inverse(d, 1) * p(0) +
                 coeff_matrix_inverse(d, 2) * p(1) +
                 coeff_matrix_inverse(d, 3) * p(0) * p(1);

    return p_out;
  }


  template <>
  Point<2>
    MyMappingQ1<2>::map_unit_cell_to_real(const Point<2> &p) const
  {
    Point<2> p_out;

    for (unsigned d = 0; d < 2; ++d)
      p_out(d) = coeff_matrix(d, 0) + coeff_matrix(d, 1) * p(0) +
                 coeff_matrix(d, 2) * p(1) + coeff_matrix(d, 3) * p(0) * p(1);

    return p_out;
  }


  template <>
  void
    MyMappingQ1<2>::map_real_to_unit_cell(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        for (unsigned d = 0; d < 2; ++d)
          points_out[i](d) =
            coeff_matrix_inverse(d, 0) +
            coeff_matrix_inverse(d, 1) * points_in[i](0) +
            coeff_matrix_inverse(d, 2) * points_in[i](1) +
            coeff_matrix_inverse(d, 3) * points_in[i](0) * points_in[i](1);
      }
  }


  template <>
  void
    MyMappingQ1<2>::map_unit_cell_to_real(
      const std::vector<Point<2>> &points_in,
      std::vector<Point<2>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        for (unsigned d = 0; d < 2; ++d)
          points_out[i](d) =
            coeff_matrix(d, 0) + coeff_matrix(d, 1) * points_in[i](0) +
            coeff_matrix(d, 2) * points_in[i](1) +
            coeff_matrix(d, 3) * points_in[i](0) * points_in[i](1);
      }
  }


  template <>
  FullMatrix<double>
    MyMappingQ1<2>::jacobian_map_real_to_unit_cell(const Point<2> &p) const
  {
    FullMatrix<double> jacobian(2, 2);

    jacobian(0, 0) =
      coeff_matrix_inverse(0, 1) + coeff_matrix_inverse(0, 3) * p(1);
    jacobian(0, 1) =
      coeff_matrix_inverse(0, 2) + coeff_matrix_inverse(0, 3) * p(0);
    jacobian(1, 0) =
      coeff_matrix_inverse(1, 1) + coeff_matrix_inverse(1, 3) * p(1);
    jacobian(1, 1) =
      coeff_matrix_inverse(1, 2) + coeff_matrix_inverse(1, 3) * p(0);

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<2>::jacobian_map_real_to_unit_cell(
      const std::vector<Point<2>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        jacobian_out[i](0, 0) = coeff_matrix_inverse(0, 1) +
                                coeff_matrix_inverse(0, 3) * points_in[i](1);
        jacobian_out[i](0, 1) = coeff_matrix_inverse(0, 2) +
                                coeff_matrix_inverse(0, 3) * points_in[i](0);
        jacobian_out[i](1, 0) = coeff_matrix_inverse(1, 1) +
                                coeff_matrix_inverse(1, 3) * points_in[i](1);
        jacobian_out[i](1, 1) = coeff_matrix_inverse(1, 2) +
                                coeff_matrix_inverse(1, 3) * points_in[i](0);
      }
  }

  template <>
  FullMatrix<double>
    MyMappingQ1<2>::jacobian_map_unit_cell_to_real(const Point<2> &p) const
  {
    FullMatrix<double> jacobian(2, 2);

    jacobian(0, 0) = coeff_matrix(0, 1) + coeff_matrix(0, 3) * p(1);
    jacobian(0, 1) = coeff_matrix(0, 2) + coeff_matrix(0, 3) * p(0);
    jacobian(1, 0) = coeff_matrix(1, 1) + coeff_matrix(1, 3) * p(1);
    jacobian(1, 1) = coeff_matrix(1, 2) + coeff_matrix(1, 3) * p(0);

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<2>::jacobian_map_unit_cell_to_real(
      const std::vector<Point<2>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        jacobian_out[i](0, 0) =
          coeff_matrix(0, 1) + coeff_matrix(0, 3) * points_in[i](1);
        jacobian_out[i](0, 1) =
          coeff_matrix(0, 2) + coeff_matrix(0, 3) * points_in[i](0);
        jacobian_out[i](1, 0) =
          coeff_matrix(1, 1) + coeff_matrix(1, 3) * points_in[i](1);
        jacobian_out[i](1, 1) =
          coeff_matrix(1, 2) + coeff_matrix(1, 3) * points_in[i](0);
      }
  }


  /*
   * 3D implementations
   */
  template <>
  MyMappingQ1<3>::MyMappingQ1(
    const typename Triangulation<3>::active_cell_iterator &cell)
    : coeff_matrix(3, 8)
    , coeff_matrix_inverse(3, 8)
  {
    FullMatrix<double> point_matrix(8, 8);
    FullMatrix<double> point_matrix_inverse(8, 8);
    FullMatrix<double> rhs_matrix(3, 8);

    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        const Point<3> &p     = cell->vertex(alpha);
        const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = p(0);
        point_matrix(2, alpha) = p(1);
        point_matrix(3, alpha) = p(2);
        point_matrix(4, alpha) = p(0) * p(1);
        point_matrix(5, alpha) = p(1) * p(2);
        point_matrix(6, alpha) = p(0) * p(2);
        point_matrix(7, alpha) = p(0) * p(1) * p(2);

        // this is rhs if we want mapping from ref cell to real cell
        rhs_matrix(0, alpha) = p_ref(0);
        rhs_matrix(1, alpha) = p_ref(1);
        rhs_matrix(3, alpha) = p_ref(2);
      }

    // Columns of coeff_matrix are the coefficients of the polynomial
    point_matrix_inverse.invert(point_matrix);
    rhs_matrix.mmult(/* destination */ coeff_matrix_inverse,
                     point_matrix_inverse);

    // clear matrixces
    point_matrix         = 0;
    point_matrix_inverse = 0;
    rhs_matrix           = 0;

    // Now with roles swtiched for inverse mapping
    for (unsigned int alpha = 0; alpha < 8; ++alpha)
      {
        const Point<3> &p     = cell->vertex(alpha);
        const Point<3> &p_ref = GeometryInfo<3>::unit_cell_vertex(alpha);

        // point matrix to be inverted
        point_matrix(0, alpha) = 1;
        point_matrix(1, alpha) = p_ref(0);
        point_matrix(2, alpha) = p_ref(1);
        point_matrix(3, alpha) = p_ref(2);
        point_matrix(4, alpha) = p_ref(0) * p_ref(1);
        point_matrix(5, alpha) = p_ref(1) * p_ref(2);
        point_matrix(6, alpha) = p_ref(0) * p_ref(2);
        point_matrix(7, alpha) = p_ref(0) * p_ref(1) * p_ref(2);

        // this is rhs if we want mapping from ref cell to real cell
        rhs_matrix(0, alpha) = p(0);
        rhs_matrix(1, alpha) = p(1);
        rhs_matrix(3, alpha) = p(2);
      }

    // Columns of coeff_matrix are the coefficients of the polynomial
    point_matrix_inverse.invert(point_matrix);
    rhs_matrix.mmult(/* destination */ coeff_matrix, point_matrix_inverse);
  }


  template <>
  Point<3>
    MyMappingQ1<3>::map_real_to_unit_cell(const Point<3> &p) const
  {
    Point<3> p_out;

    for (unsigned d = 0; d < 3; ++d)
      p_out(d) =
        coeff_matrix_inverse(d, 0) + coeff_matrix_inverse(d, 1) * p(0) +
        coeff_matrix_inverse(d, 2) * p(1) + coeff_matrix_inverse(d, 3) * p(2) +
        coeff_matrix_inverse(d, 4) * p(0) * p(1) +
        coeff_matrix_inverse(d, 5) * p(1) * p(2) +
        coeff_matrix_inverse(d, 6) * p(0) * p(2) +
        coeff_matrix_inverse(d, 7) * p(0) * p(1) * p(2);

    return p_out;
  }


  template <>
  Point<3>
    MyMappingQ1<3>::map_unit_cell_to_real(const Point<3> &p) const
  {
    Point<3> p_out;

    for (unsigned d = 0; d < 3; ++d)
      p_out(d) = coeff_matrix(d, 0) + coeff_matrix(d, 1) * p(0) +
                 coeff_matrix(d, 2) * p(1) + coeff_matrix(d, 3) * p(2) +
                 coeff_matrix(d, 4) * p(0) * p(1) +
                 coeff_matrix(d, 5) * p(1) * p(2) +
                 coeff_matrix(d, 6) * p(0) * p(2) +
                 coeff_matrix(d, 7) * p(0) * p(1) * p(2);

    return p_out;
  }


  template <>
  void
    MyMappingQ1<3>::map_real_to_unit_cell(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        for (unsigned d = 0; d < 3; ++d)
          points_out[i](d) =
            coeff_matrix_inverse(d, 0) +
            coeff_matrix_inverse(d, 1) * points_in[i](0) +
            coeff_matrix_inverse(d, 2) * points_in[i](1) +
            coeff_matrix_inverse(d, 3) * points_in[i](2) +
            coeff_matrix_inverse(d, 4) * points_in[i](0) * points_in[i](1) +
            coeff_matrix_inverse(d, 5) * points_in[i](1) * points_in[i](2) +
            coeff_matrix_inverse(d, 6) * points_in[i](0) * points_in[i](2) +
            coeff_matrix_inverse(d, 7) * points_in[i](0) * points_in[i](1) *
              points_in[i](2);
      }
  }


  template <>
  void
    MyMappingQ1<3>::map_unit_cell_to_real(
      const std::vector<Point<3>> &points_in,
      std::vector<Point<3>> &      points_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), points_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        for (unsigned d = 0; d < 3; ++d)
          points_out[i](d) =
            coeff_matrix(d, 0) + coeff_matrix(d, 1) * points_in[i](0) +
            coeff_matrix(d, 2) * points_in[i](1) +
            coeff_matrix(d, 3) * points_in[i](2) +
            coeff_matrix(d, 4) * points_in[i](0) * points_in[i](1) +
            coeff_matrix(d, 5) * points_in[i](1) * points_in[i](2) +
            coeff_matrix(d, 6) * points_in[i](0) * points_in[i](2) +
            coeff_matrix(d, 7) * points_in[i](0) * points_in[i](1) *
              points_in[i](2);
      }
  }


  template <>
  FullMatrix<double>
    MyMappingQ1<3>::jacobian_map_real_to_unit_cell(const Point<3> &p) const
  {
    FullMatrix<double> jacobian(3, 3);

    for (unsigned int d = 0; d < 3; ++d)
      {
        jacobian(d, 0) = coeff_matrix_inverse(d, 1) +
                         coeff_matrix_inverse(d, 4) * p(1) +
                         coeff_matrix_inverse(d, 6) * p(2) +
                         coeff_matrix_inverse(d, 7) * p(1) * p(2);
        jacobian(d, 1) = coeff_matrix_inverse(d, 2) +
                         coeff_matrix_inverse(d, 4) * p(0) +
                         coeff_matrix_inverse(d, 5) * p(2) +
                         coeff_matrix_inverse(d, 7) * p(0) * p(2);
        jacobian(d, 2) = coeff_matrix_inverse(d, 3) +
                         coeff_matrix_inverse(d, 5) * p(2) +
                         coeff_matrix_inverse(d, 6) * p(1) +
                         coeff_matrix_inverse(d, 7) * p(1) * p(2);
      }

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<3>::jacobian_map_real_to_unit_cell(
      const std::vector<Point<3>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        for (unsigned int d = 0; d < 3; ++d)
          {
            jacobian_out[i](d, 0) =
              coeff_matrix_inverse(d, 1) +
              coeff_matrix_inverse(d, 4) * points_in[i](1) +
              coeff_matrix_inverse(d, 6) * points_in[i](2) +
              coeff_matrix_inverse(d, 7) * points_in[i](1) * points_in[i](2);
            jacobian_out[i](d, 1) =
              coeff_matrix_inverse(d, 2) +
              coeff_matrix_inverse(d, 4) * points_in[i](0) +
              coeff_matrix_inverse(d, 5) * points_in[i](2) +
              coeff_matrix_inverse(d, 7) * points_in[i](0) * points_in[i](2);
            jacobian_out[i](d, 2) =
              coeff_matrix_inverse(d, 3) +
              coeff_matrix_inverse(d, 5) * points_in[i](2) +
              coeff_matrix_inverse(d, 6) * points_in[i](1) +
              coeff_matrix_inverse(d, 7) * points_in[i](1) * points_in[i](2);
          }
      }
  }

  template <>
  FullMatrix<double>
    MyMappingQ1<3>::jacobian_map_unit_cell_to_real(const Point<3> &p) const
  {
    FullMatrix<double> jacobian(3, 3);

    for (unsigned int d = 0; d < 3; ++d)
      {
        jacobian(d, 0) = coeff_matrix(d, 1) + coeff_matrix(d, 4) * p(1) +
                         coeff_matrix(d, 6) * p(2) +
                         coeff_matrix(d, 7) * p(1) * p(2);
        jacobian(d, 1) = coeff_matrix(d, 2) + coeff_matrix(d, 4) * p(0) +
                         coeff_matrix(d, 5) * p(2) +
                         coeff_matrix(d, 7) * p(0) * p(2);
        jacobian(d, 2) = coeff_matrix(d, 3) + coeff_matrix(d, 5) * p(2) +
                         coeff_matrix(d, 6) * p(1) +
                         coeff_matrix(d, 7) * p(1) * p(2);
      }

    return jacobian;

    return jacobian;
  }

  template <>
  void
    MyMappingQ1<3>::jacobian_map_unit_cell_to_real(
      const std::vector<Point<3>> &    points_in,
      std::vector<FullMatrix<double>> &jacobian_out) const
  {
    Assert(points_in.size() == points_out.size(),
           ExcDimensionMismatch(points_in.size(), jacobian_out.size()));

    for (unsigned int i = 0; i < points_in.size(); ++i)
      {
        for (unsigned int d = 0; d < 3; ++d)
          {
            jacobian_out[i](d, 0) =
              coeff_matrix(d, 1) + coeff_matrix(d, 4) * points_in[i](1) +
              coeff_matrix(d, 6) * points_in[i](2) +
              coeff_matrix(d, 7) * points_in[i](1) * points_in[i](2);
            jacobian_out[i](d, 1) =
              coeff_matrix(d, 2) + coeff_matrix(d, 4) * points_in[i](0) +
              coeff_matrix(d, 5) * points_in[i](2) +
              coeff_matrix(d, 7) * points_in[i](0) * points_in[i](2);
            jacobian_out[i](d, 2) =
              coeff_matrix(d, 3) + coeff_matrix(d, 5) * points_in[i](2) +
              coeff_matrix(d, 6) * points_in[i](1) +
              coeff_matrix(d, 7) * points_in[i](1) * points_in[i](2);
          }
      }
  }

} // namespace ShapeFun



#endif /* INCLUDE_FUNCTIONS_MY_MAPPING_Q1_TPP_ */
