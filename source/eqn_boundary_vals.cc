#include "eqn_boundary_vals.h"

namespace LaplaceProblem
{
  using namespace dealii;

  void
  Boundary_B_div_u::value_list(const std::vector<Point<3>> &points,
                               std::vector<double> &        values,
                               const unsigned int /* component = 0 */) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p] = 0.0;
      }
  }


  void
  Boundary_A_curl_u::value_list(const std::vector<Point<3>> &points,
                                std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p].clear();

        // Set values of components
        for (unsigned int d = 0; d < 3; ++d)
          values[p][d] = 0;
      }
  }

} // end namespace LaplaceProblem
