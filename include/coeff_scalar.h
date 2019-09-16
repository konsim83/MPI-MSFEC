#ifndef INCLUDE_COEFF_SCALAR_H_
#define INCLUDE_COEFF_SCALAR_H_

#include "coefficients.h"


namespace Coefficients
{

using namespace dealii;

/**
 * Second diffusion tensor. Must be positive definite and uniformly bounded from below and above.
 */
class DiffusionScalar : public Function<3>
{
public:
	DiffusionScalar () : Function<3>() {}

  virtual void value_list (const std::vector<Point<3>> &points,
                           std::vector<double>    &values,
						   const unsigned int  component = 0) const override;
};

}

#endif /* INCLUDE_COEFF_SCALAR_H_ */
