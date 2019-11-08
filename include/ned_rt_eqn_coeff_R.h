#ifndef HELMHOLTZ_EQN_COEFF_R_H_
#define HELMHOLTZ_EQN_COEFF_R_H_

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
 * This lower order term can regularize the weak form. If it vanishes we have a
 * Darcy (last element of the de Rham complex).
 */
class ReactionRate : public Function<3>
{
public:
	ReactionRate ()
	:
	Function<3>()
	{}

	virtual void value_list (const std::vector<Point<3>> &points,
	                           std::vector<double>    &values,
							   const unsigned int  component = 0) const override;
};
  
} // end namespace LaplaceProblem

#endif /* HELMHOLTZ_EQN_COEFF_R_H_ */
