#ifndef EQN_COEFF_A_H_
#define EQN_COEFF_A_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>

// std library
#include <cmath>
#include <vector>
#include <cstdlib>

namespace LaplaceProblem
{

using namespace dealii;


/**
 * Class containing data for tensor valued diffusivity of a positive tensor.
 */
class Diffusion_A_Data
{
public:
	Diffusion_A_Data (const std::string &parameter_filename);

	static void declare_parameters(ParameterHandler &prm);
	void parse_parameters(ParameterHandler &prm);

	/**
	 * Frequency of oscillations
	 */
	unsigned int k_x;
	unsigned int k_y;
	unsigned int k_z;

	/**
	 * Scaling factor
	 */
	double scale_x;
	double scale_y;
	double scale_z;

	/**
	 * Three Euler angles.
	 */
	const double alpha_, beta_, gamma_;

	bool rotate;

	/**
	 * Description of rotation with Euler angles.
	 */
	Tensor<2,3> rot;
};


/**
 * Inverse of diffusion tensor. Must be positive definite and uniformly bounded from below and above.
 */
class Diffusion_A : public TensorFunction<2,3>, public Diffusion_A_Data
{
public:
	/**
	 * Constructor.
	 */
	Diffusion_A (const std::string &parameter_filename)
	:
	TensorFunction<2,3>(),
	Diffusion_A_Data(parameter_filename)
	{}

	/**
	 * Function to compute the tensor values of A.
	 */
	virtual void value_list (const std::vector<Point<3> > &points,
                           std::vector<Tensor<2,3> >    &values) const override;
};


/**
 * Inverse of diffusion tensor. Must be positive definite and uniformly bounded from below and above.
 */
class DiffusionInverse_A : public TensorFunction<2,3>, public Diffusion_A_Data
{
public:
	/**
	 * Constructor.
	 */
	DiffusionInverse_A (const std::string &parameter_filename)
	:
	TensorFunction<2,3>(),
	Diffusion_A_Data(parameter_filename)
	{}

	/**
	 * Function to compute the inverse of the tensor values of A.
	 */
	virtual void value_list (const std::vector<Point<3> > &points,
                           std::vector<Tensor<2,3> >    &values) const override;
};
  
} // end namespace LaplaceProblem

#endif /* EQN_COEFF_A_H_ */
