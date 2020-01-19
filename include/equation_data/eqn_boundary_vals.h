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

namespace EquationData
{
  using namespace dealii;

  /**
   * Scalar boundary values for u in the
   * first and last part of the de Rham complex.
   */
  class BoundaryValues_u : public Function<3>
  {
  public:
    /**
     * Constructor.
     */
    BoundaryValues_u()
      : Function<3>(1)
    {}

    /**
     * Implementation of scalar boundary values for u.
     *
     * @param p
     * @param component = 0
     */
    virtual void
    value_list(const std::vector<Point<3>> &points,
               std::vector<double> &        values,
               const unsigned int           component = 0) const override;
  };


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

} // end namespace EquationData

#endif /* EQN_BOUNDARY_VALS_H_ */
