#ifndef EQN_BOUNDARY_VALS_H_
#define EQN_BOUNDARY_VALS_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// std library
#include <cmath>
#include <cstdlib>
#include <vector>

namespace LaplaceProblem
{
  using namespace dealii;


  /**
   * Boundary values for B*div u.
   */
  class Boundary_B_div_u : public Function<3>
  {
  public:
    /**
     * Constructor.
     */
    Boundary_B_div_u()
      : Function<3>(1)
    {}

    /**
     * Implementation of boundary values for B*div u.
     *
     * @param p
     * @param component = 0
     */
    virtual void
    value_list(const std::vector<Point<3>> &points,
               std::vector<double> &        values,
               const unsigned int /* component = 0 */) const override;
  };


  /**
   * Boundary values for A*curl u.
   */
  class Boundary_A_curl_u : public TensorFunction<1, 3>
  {
  public:
    /**
     * Constructor.
     */
    Boundary_A_curl_u()
      : TensorFunction<1, 3>()
    {}

    /**
     * Implementation of boundary values for A*curl u.
     *
     * @param p
     */
    virtual void
    value_list(const std::vector<Point<3>> &points,
               std::vector<Tensor<1, 3>> &  values) const override;
  };

} // end namespace LaplaceProblem

#endif /* EQN_BOUNDARY_VALS_H_ */
