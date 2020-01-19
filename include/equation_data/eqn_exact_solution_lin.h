#ifndef INCLUDE_EQUATION_DATA_EQN_EXACT_SOLUTION_LIN_H_
#define INCLUDE_EQUATION_DATA_EQN_EXACT_SOLUTION_LIN_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>

#include <equation_data/eqn_coeff_A.h>
#include <equation_data/eqn_coeff_B.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace EquationData
{
  using namespace dealii;


  /**
   * This class contains only data members read from a parameter file.
   */
  class ExactSolutionLin_Data
  {
  public:
    /**
     * Constructor
     */
    ExactSolutionLin_Data(const std::string &parameter_filename);

    /**
     * Declare all parameters for tensor valued diffusion.
     *
     * @param prm
     */
    static void
    declare_parameters(ParameterHandler &prm);

    /**
     * Parse all delaced parameters in file.
     *
     * @param prm
     */
    void
    parse_parameters(ParameterHandler &prm);

    /**
     * Matrix coefficient 'A' of 'u=Ax+b'.
     */
    Tensor<2, 3> A;

    /**
     * Vector coefficient 'b' of 'u=Ax+b'.
     */
    Tensor<1, 3> b;

    /**
     * Divergence of u is simply the trace of 'A'.
     */
    double div_u;

    /**
     * The curl of u if simply the representation vector
     * of the anti-symmetric part of the linear map.
     */
    Tensor<1, 3> curl_u;
  };


  /**
   * Exact solution (linear function).
   */
  class ExactSolutionLin : public TensorFunction<1, 3>,
                           public ExactSolutionLin_Data
  {
  public:
    /**
     * Constructor
     */
    ExactSolutionLin(const std::string &parameter_filename);

    /**
     * Tensor value function. u = A*x+b.
     */
    Tensor<1, 3>
    value(const Point<3> &p) const override;

    /**
     * Tensor value list function. u = A*x+b.
     */
    void
    value_list(const std::vector<Point<3>> &points,
               std::vector<Tensor<1, 3>> &  values) const;
  };


  /**
   * Auxiliary variable for H1-H(curl) problem.
   */
  class ExactSolutionLin_B_div : public Function<3>,
                                 public ExactSolutionLin_Data
  {
  public:
    /**
     * Constructor
     */
    ExactSolutionLin_B_div(const std::string &parameter_filename);

    /**
     * Value function for 'sigma = -B*div_u'
     */
    double
    value(const Point<3> &   point,
          const unsigned int component = 0) const override;

    /**
     * Value list function for 'sigma = -B*div_u'.
     */
    void
    value_list(const std::vector<Point<3>> &points,
               std::vector<double> &        values,
               const unsigned int           component = 0) const override;

  private:
    const Diffusion_B b;
  };


  /**
   * Auxiliary variable for H(curl)-H(div) problem.
   */
  class ExactSolutionLin_A_curl : public TensorFunction<1, 3>,
                                  public ExactSolutionLin_Data
  {
  public:
    /**
     * Constructor
     */
    ExactSolutionLin_A_curl(const std::string &parameter_filename);

    /**
     * Value function for 'sigma = A*curl_u'.
     */
    virtual Tensor<1, 3>
    value(const Point<3> &point) const override;

    /**
     * Value list function for 'sigma = A*curl_u'.
     */
    void
    value_list(const std::vector<Point<3>> &points,
               std::vector<Tensor<1, 3>> &  values) const override;

  private:
    const Diffusion_A a;
  };

} // end namespace EquationData

#endif /* INCLUDE_EQUATION_DATA_EQN_EXACT_SOLUTION_LIN_H_ */
