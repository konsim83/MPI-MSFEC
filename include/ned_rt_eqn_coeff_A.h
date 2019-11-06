#ifndef HELMHOLTZ_EQN_COEFF_A_H_
#define HELMHOLTZ_EQN_COEFF_A_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/numbers.h>

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
	Diffusion_A_Data ()
	: k(0),
	  scale(1.0),
	  alpha(numbers::PI/3),
	  beta(numbers::PI/6),
	  gamma(numbers::PI/4),
	  rotate(false)
	{
		rot[0][0] = cos(alpha)*cos(gamma) - sin(alpha)*cos(beta)*sin(gamma);
		rot[0][1] = -cos(alpha)*sin(gamma) - sin(alpha)*cos(beta)*cos(gamma);
		rot[0][2] = sin(alpha)*sin(beta);
		rot[1][0] = sin(alpha)*cos(gamma) + cos(alpha)*cos(beta)*sin(gamma);
		rot[1][1] = -sin(alpha)*sin(gamma) + cos(alpha)*cos(beta)*cos(gamma);
		rot[1][2] = -cos(alpha)*sin(beta);
		rot[2][0] = sin(beta)*sin(gamma);
		rot[2][1] = sin(beta)*cos(gamma);
		rot[2][2] = cos(beta);
	}

	/**
	 * Frequency of oscillations
	 */
	const int k;

	/**
	 * Scaling factor
	 */
	const double scale;

	/**
	 * Three Euler angles.
	 */
	const double alpha, beta, gamma;

	const bool rotate;

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
	Diffusion_A ()
	:
	TensorFunction<2,3>(),
	Diffusion_A_Data()
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
	DiffusionInverse_A ()
	:
	TensorFunction<2,3>()
	{}

	/**
	 * Function to compute the inverse of the tensor values of A.
	 */
	virtual void value_list (const std::vector<Point<3> > &points,
                           std::vector<Tensor<2,3> >    &values) const override;
};
  
} // end namespace LaplaceProblem

#endif /* HELMHOLTZ_EQN_COEFF_A_H_ */
