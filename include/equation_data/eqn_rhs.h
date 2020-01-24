#ifndef EQN_RHS_H_
#define EQN_RHS_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

// my headers
#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>
#include <equation_data/eqn_exact_solution_lin.h>

namespace EquationData
{
  using namespace dealii;


  class RightHandSide : public Functions::ParsedFunction<3>
  {
  public:
    RightHandSide(unsigned int n_components)
      : Functions::ParsedFunction<3>(n_components)
    {}

    virtual void
      tensor_value_list(const std::vector<Point<3>> & /*points*/,
                        std::vector<Tensor<1, 3>> & /*values*/) const {};
  };



  /**
   * Right hand side of equation is a
   * vectorial function. Directly implemented or parsed.
   */
  class RightHandSideParsed : public RightHandSide
  {
  public:
    /**
     * Constructor.
     *
     * @param parameter_filename
     * @param n_components
     */
    RightHandSideParsed(const std::string &parameter_filename,
                        unsigned int       n_components);

    /**
     * Implementation of right hand side.
     *
     * @param p
     * @param component = 0
     */
    virtual void
      tensor_value_list(const std::vector<Point<3>> &points,
                        std::vector<Tensor<1, 3>> &  values) const override;
  };



  /**
   * Right hand side of equation is a vectorial function. This class
   * is derived from the class of an abstract solution class.
   */
  class RightHandSideExactLin : public RightHandSide,
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
    virtual void
      tensor_value_list(const std::vector<Point<3>> &points,
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
