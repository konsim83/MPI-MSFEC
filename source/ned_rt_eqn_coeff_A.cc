#include "ned_rt_eqn_coeff_A.h"

namespace LaplaceProblem
{

using namespace dealii;

/**
 * Implementation of inverse of the diffusion tensor. Must be positive definite and uniformly bounded.
 *
 * @param points
 * @param values
 */
void
Diffusion_A::value_list (const std::vector<Point<3> > &points,
						 std::vector<Tensor<2,3> >    &values) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()));

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p].clear ();

		//This is just diagonal
		values[p][0][0] = (1.0 - 0.99 * sin( 2 * numbers::PI * k * points.at(p)(0) ));
		values[p][1][1] = (1.0 - 0.99 * sin( 2 * numbers::PI * k * points.at(p)(1) ));
		values[p][2][2] = (1.0 - 0.99 * sin( 2 * numbers::PI * k * points.at(p)(2) ));
	}

	if (rotate)
		for (unsigned int p=0; p<points.size(); ++p)
		{
			// Rotation leads to anisotropy
			values[p] = rot * values[p] * transpose (rot);
		}
}



/**
 * Implementation of inverse of the diffusion tensor. Must be positive definite and uniformly bounded.
 *
 * @param points
 * @param values
 */
void
DiffusionInverse_A::value_list (const std::vector<Point<3> > &points,
						 std::vector<Tensor<2,3> >    &values) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()));

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p].clear ();

		//This is just diagonal
		values[p][0][0] = 1 / (1.0 - 0.99 * sin( 2 * numbers::PI * k * points.at(p)(0) ));
		values[p][1][1] = 1 / (1.0 - 0.99 * sin( 2 * numbers::PI * k * points.at(p)(1) ));
		values[p][2][2] = 1 / (1.0 - 0.99 * sin( 2 * numbers::PI * k * points.at(p)(2) ));
	}

	if (rotate)
		for (unsigned int p=0; p<points.size(); ++p)
		{
			// Rotation leads to anisotropy
			values[p] = rot * values[p] * transpose (rot);
		}
}

} // end namespace LaplaceProblem
