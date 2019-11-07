#ifndef HELMHOLTZ_EQN_COEFF_B_H_
#define HELMHOLTZ_EQN_COEFF_B_H_

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

// std library
#include <cmath>
#include <vector>
#include <cstdlib>

namespace LaplaceProblem
{

using namespace dealii;


/**
 * Class containing data for diffusivity of a positive scalar.
 */
class Diffusion_B_Data
{
public:
	Diffusion_B_Data ()
	: k (14),
	  scale (0.1)
	{
	}

	/**
	 * Frequency of oscillations
	 */
	const int k;

	/**
	 * Scaling factor
	 */
	const double scale;
};


/**
 * Second diffusion tensor. Must be positive definite and uniformly
 * bounded from below and above.
 */
class Diffusion_B : public Function<3>, public Diffusion_B_Data
{
public:
	Diffusion_B ()
	:
	Function<3>(),
	Diffusion_B_Data ()
	{}

	virtual void value_list (const std::vector<Point<3>> &points,
                           std::vector<double>    &values,
						   const unsigned int  component = 0) const override;
};


/**
 * Inverse of second diffusion tensor. Must be positive definite and
 * uniformly bounded from below and above.
 */
class DiffusionInverse_B : public Function<3>, public Diffusion_B_Data
{
public:
	DiffusionInverse_B ()
	:
	Function<3>(),
	Diffusion_B_Data ()
	{}

	virtual void value_list (const std::vector<Point<3>> &points,
                           std::vector<double>    &values,
						   const unsigned int  component = 0) const override;
};
  
} // end namespace LaplaceProblem

#endif /* HELMHOLTZ_EQN_COEFF_B_H_ */
