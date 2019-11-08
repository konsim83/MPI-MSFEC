#include "ned_rt_eqn_coeff_R.h"

namespace LaplaceProblem
{

using namespace dealii;

/**
 * Implementation of lower order term.
 *
 * @param p
 * @param component = 0
 */
void
ReactionRate::value_list (const std::vector<Point<3> > &points,
							 std::vector<double>    &values,
							 const unsigned int  /* component = 0 */) const
{
	for (unsigned int p=0; p<points.size(); ++p)
		values[p] = 0.0;

}

} // end namespace LaplaceProblem
