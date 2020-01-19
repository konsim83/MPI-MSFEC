#ifndef EQN_COEFF_B_H_
#define EQN_COEFF_B_H_

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

// std library
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /**
   * Class containing data for tensor valued
   * diffusivity.
   */
  class Diffusion_B_Data
  {
  public:
    /**
     * Constructor
     */
    Diffusion_B_Data(const std::string &parameter_filename);

    static void
      declare_parameters(ParameterHandler &prm);
    void
      parse_parameters(ParameterHandler &prm);

    /**
     * Frequency of oscillations
     */
    unsigned int k;

    /**
     * Scaling factor
     */
    double scale;

    /**
     * Scaling factor for oscillations
     */
    double alpha;

    /**
     * Function expression in muParser format.
     */
    std::string expression;
  };

  /**
   * Second (scalar) coefficient function. Must be positive definite and
   * uniformly bounded from below and above.
   */
  class Diffusion_B : public FunctionParser<3>, public Diffusion_B_Data
  {
  public:
    /**
     * Constructor
     */
    Diffusion_B(const std::string &parameter_filename,
                bool               use_exact_solution = false);

  private:
    const std::string             variables = "x,y,z";
    std::string                   fnc_expression;
    std::map<std::string, double> constants;
  };

  /**
   * Inverse of second diffusion tensor. Must be positive definite and
   * uniformly bounded from below and above.
   */
  class DiffusionInverse_B : public FunctionParser<3>, public Diffusion_B_Data
  {
  public:
    /**
     * Constructor
     */
    DiffusionInverse_B(const std::string &parameter_filename,
                       bool               use_exact_solution = false);

  private:
    const std::string             variables = "x,y,z";
    std::string                   inverse_fnc_expression;
    std::map<std::string, double> constants;
  };

} // end namespace EquationData

#endif /* EQN_COEFF_B_H_ */
