#ifndef HELMHOLTZ_EQN_COEFF_B_H_
#define HELMHOLTZ_EQN_COEFF_B_H_

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/parameter_handler.h>

// std library
#include <cmath>
#include <vector>
#include <cstdlib>

namespace LaplaceProblem
{

using namespace dealii;


/**
 * Second diffusion tensor. Must be positive definite and uniformly
 * bounded from below and above.
 */
class Diffusion_B : public Functions::ParsedFunction<3>
{
public:
	Diffusion_B (const std::string &parameter_filename);
};


/**
 * Inverse of second diffusion tensor. Must be positive definite and
 * uniformly bounded from below and above.
 */
class DiffusionInverse_B : public Functions::ParsedFunction<3>
{
public:
	DiffusionInverse_B (const std::string &parameter_filename);
};
  
} // end namespace LaplaceProblem

#endif /* HELMHOLTZ_EQN_COEFF_B_H_ */
