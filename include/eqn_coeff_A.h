#ifndef EQN_COEFF_A_H_
#define EQN_COEFF_A_H_

#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

// std library
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace LaplaceProblem
{
  using namespace dealii;

  /**
   * Class containing data for tensor valued diffusivity of a positive tensor.
   */
  class Diffusion_A_Data
  {
  public:
    Diffusion_A_Data(const std::string &parameter_filename);

    static void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);

    /**
     * Frequency of oscillations in x, y and z.
     */
    unsigned int k_x, k_y, k_z;

    /**
     * Scaling factor in x, y and z..
     */
    double scale_x, scale_y, scale_z;

    /**
     * Scaling factor for oscillations in x, y and z.
     */
    double alpha_x, alpha_y, alpha_z;

    /**
     * Three Euler angles.
     */
    const double alpha_, beta_, gamma_;

    /**
     * True if tensor coefficient should be rotated in space.
     */
    bool rotate;

    /**
     * Description of rotation with Euler angles. This rotates the
     * tensor coefficients in space and allows for the construction
     * of more general symmetric positive definite data. If 'rotate=false'
     * then 'rot' is just the identity.
     */
    Tensor<2, 3> rot;
  };


  /**
   * Diffusion tensor. Must be positive definite and
   * uniformly bounded from below and above.
   */
  class Diffusion_A : public TensorFunction<2, 3>, public Diffusion_A_Data
  {
  public:
    /**
     * Constructor.
     */
    Diffusion_A(const std::string &parameter_filename)
      : TensorFunction<2, 3>()
      , Diffusion_A_Data(parameter_filename)
    {}

    /**
     * Implementation of the diffusion tensor.
     * Must be positive definite and uniformly bounded.
     *
     * @param points
     * @param values
     */
    virtual Tensor<2, 3>
    value(const Point<3> &point) const override;


    /**
     * Implementation of the diffusion tensor.
     * Must be positive definite and uniformly bounded.
     *
     * @param points
     * @param values
     */
    virtual void
    value_list(const std::vector<Point<3>> &points,
               std::vector<Tensor<2, 3>> &  values) const override;
  };


  /**
   * Inverse of diffusion tensor. Must be positive
   * definite and uniformly bounded from below and above.
   */
  class DiffusionInverse_A : public TensorFunction<2, 3>,
                             public Diffusion_A_Data
  {
  public:
    /**
     * Constructor.
     */
    DiffusionInverse_A(const std::string &parameter_filename)
      : TensorFunction<2, 3>()
      , Diffusion_A_Data(parameter_filename)
    {}

    /**
     * Implementation of inverse of the diffusion tensor.
     * Must be positive definite and uniformly bounded.
     *
     * @param points
     * @param values
     */
    virtual Tensor<2, 3>
    value(const Point<3> &point) const override;

    /**
     * Implementation of inverse of the diffusion tensor.
     * Must be positive definite and uniformly bounded.
     *
     * @param points
     * @param values
     */
    virtual void
    value_list(const std::vector<Point<3>> &points,
               std::vector<Tensor<2, 3>> &  values) const override;
  };

} // end namespace LaplaceProblem

#endif /* EQN_COEFF_A_H_ */
