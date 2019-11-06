#include "ned_rt_eqn_coeff_B.h"

namespace LaplaceProblem
{

using namespace dealii;

/**
 * Implementation of second diffusion tensor. Must be positive definite and uniformly bounded.
 *
 * @param points
 * @param values
 */
void
Diffusion_B::value_list (const std::vector<Point<3> > &points,
						 std::vector<double>    &values,
						 const unsigned int  /* component = 0 */) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()));

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = scale * (1.0 - 0.99 * (
									  sin(2*numbers::PI*k*(  points.at(p)(0)   ))
//									+ cos(2*numbers::PI*k*(  points.at(p)(1)  ))
//									+ sin(2*numbers::PI*k*(  points.at(p)(2)  ))
									)
					);
	}
}



/**
 * Implementation of inverse of second diffusion tensor. Must
 * be positive definite and uniformly bounded.
 *
 * @param points
 * @param values
 */
void
DiffusionInverse_B::value_list (const std::vector<Point<3> > &points,
						 std::vector<double>    &values,
						 const unsigned int  /* component = 0 */) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()));

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = 1 / (scale * (1.0 - 0.99 * (
									  sin(2*numbers::PI*k*(  points.at(p)(0)   ))
//									+ cos(2*numbers::PI*k*(  points.at(p)(1)  ))
//									+ sin(2*numbers::PI*k*(  points.at(p)(2)  ))
									)
					));
	}
}

} // end namespace LaplaceProblem
