#ifndef EQN_RHS_H_
#define EQN_RHS_H_

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
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
 * Right hand side of equation is a vectorial function.
 */
class RightHandSide : public TensorFunction<1,3>
{
public:
	RightHandSide () : TensorFunction<1,3>() {}

	virtual void value_list (const std::vector<Point<3> > &points,
			  	  	  	  	   std::vector<Tensor<1,3> >    &values) const override;
};
  
} // end namespace LaplaceProblem

#endif /* EQN_RHS_H_ */
