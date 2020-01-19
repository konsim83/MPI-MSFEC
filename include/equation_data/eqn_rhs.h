#ifndef EQN_RHS_H_
#define EQN_RHS_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>
#include <equation_data/eqn_exact_solution_lin.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /**
   * Right hand side of equation is a
   * vectorial function. Directly implemented or parsed.
   */
  class RightHandSideParsed : public TensorFunction<1, 3>
  {
  public:
    /**
     * Constructor.
     */
    RightHandSideParsed()
      : TensorFunction<1, 3>()
    {}

    /**
     * Implementation of right hand side.
     *
     * @param p
     * @param component = 0
     */
    void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const override;
  };

  /**
   * Right hand side of equation in first and last part of
   * the de Rham complex is a scalar function.
   * Directly implemented or parsed.
   */
  class RightHandSideScalarParsed : public Function<3>
  {
  public:
    /**
     * Constructor.
     */
    RightHandSideScalarParsed()
      : Function<3>()
    {}

    /**
     * Implementation of right hand side.
     *
     * @param p
     * @param component = 0
     */
    void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 unsigned int                 component = 0) const override;
  };

  /**
   * Right hand side of equation is a vectorial function. This class
   * is derived from the class of an abstract solution class.
   */
  class RightHandSideExactLin : public TensorFunction<1, 3>,
                                public ExactSolutionLin_Data,
                                public Diffusion_A_Data,
                                public Diffusion_B_Data
  {
  public:
    /**
     * Constructor.
     *
     * @param parameter_filename
     */
    RightHandSideExactLin(const std::string &parameter_filename);

    /**
     * Implementation of right hand side of exact solution.
     *
     * @param p
     * @param component = 0
     */
    void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<Tensor<1, 3>> &  values) const override;

  private:
    const double pi = numbers::PI;

    /**
     * Expression for 'rot*curl(u)'.
     */
    Tensor<1, 3> R_trans_curl_u;
  };

} // end namespace EquationData

#endif /* EQN_RHS_H_ */
