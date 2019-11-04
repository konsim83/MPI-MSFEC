#ifndef HELMHOLTZ_EQN_DATA_H_
#define HELMHOLTZ_EQN_DATA_H_

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

// std library
#include <cmath>
#include <vector>
#include <cstdlib>

namespace LaplaceProblem
{

const double PI_D = 3.141592653589793238463;
const float  PI_F = 3.14159265358979f;

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


/**
 * Boundary values for u. In the weak form they are natural.
 */
class BoundaryDivergenceValues_u : public Function<3>
{
public:
	BoundaryDivergenceValues_u () : Function<3>(1) {}

	virtual double value (const Point<3>   &p,
                        const unsigned int  component = 0) const override;
};




/**
 * Boundary values for u. In the weak form they are essential.
 */
class BoundaryValues_sigma : public TensorFunction<1,3>
{
public:
  BoundaryValues_sigma () : TensorFunction<1,3>() {}

  virtual void value_list (const std::vector<Point<3> > &points,
		  	  	  	  	   std::vector<Tensor<1,3> >    &values) const override;
};




/**
 * Inverse of diffusion tensor. Must be positive definite and uniformly bounded from below and above.
 */
class DiffusionInverse_A : public TensorFunction<2,3>
{
public:
  DiffusionInverse_A () : TensorFunction<2,3>() {}

  virtual void value_list (const std::vector<Point<3> > &points,
                           std::vector<Tensor<2,3> >    &values) const override;
};


/**
 * Second diffusion tensor. Must be positive definite and uniformly bounded from below and above.
 */
class Diffusion_B : public Function<3>
{
public:
	Diffusion_B () : Function<3>() {}

  virtual void value_list (const std::vector<Point<3>> &points,
                           std::vector<double>    &values,
						   const unsigned int  component = 0) const override;
};



/**
 * Implementation of reaction rate. This term regularizes the weak form. If it vanishes we have a Darcy problem.
 */
class ReactionRate : public Function<3>
{
public:
	ReactionRate () : Function<3>() {}

	virtual double value (const Point<3>   &p,
	                        const unsigned int  component = 0) const override;

//	virtual void value_list (const std::vector<Point<3> > &points,
//	                             std::vector<double>    &values) const;
};
///
  
} // end namespace LaplaceProblem

#endif /* HELMHOLTZ_EQN_DATA_H_ */
