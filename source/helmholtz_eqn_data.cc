#include "helmholtz_eqn_data.h"

namespace HelmholtzProblem
{

using namespace dealii;

/**
 * Implementation of right hand side.
 *
 * @param p
 * @param component = 0
 * @return
 */
void
RightHandSide::value_list (const std::vector<Point<3> > &points,
	  	  	   std::vector<Tensor<1,3> >    &values) const
{
	Assert (points.size() == values.size(),
				ExcDimensionMismatch (points.size(), values.size()));

	for (unsigned int i=0; i<values.size(); ++i)
	{
		values.at(i).clear ();

		// Gradient_0 of V = sin(2*PI*j*x)*sin(2*PI*k*y)*sin(2*PI*l*z)
//		const double a = 1;
//		const unsigned int j = 2, k = 2, l = 2;
//		values.at(i)[0] = a*2*PI_D*j*cos(2*PI_D*j*points[i](0))*sin(2*PI_D*k*points[i](1))*sin(2*PI_D*l*points[i](2));
//		values.at(i)[1] = a*2*PI_D*k*sin(2*PI_D*j*points[i](0))*cos(2*PI_D*k*points[i](1))*sin(2*PI_D*l*points[i](2));
//		values.at(i)[2] = a*2*PI_D*l*sin(2*PI_D*j*points[i](0))*sin(2*PI_D*k*points[i](1))*cos(2*PI_D*l*points[i](2));

		// Gradient_0 of x(x-1)y(y-1)z(z-1)
//		double x2 = points[i](0)*points[i](0) - points[i](0);
//		double y2 = points[i](1)*points[i](1) - points[i](1);
//		double z2 = points[i](2)*points[i](2) - points[i](2);
//		values.at(i)[0] = (2*points[i](0) - 1) * y2 * z2;
//		values.at(i)[1] = (2*points[i](1) - 1) * x2 * z2;
//		values.at(i)[2] = (2*points[i](2) - 1) * x2 * y2;

		// Curl_0
//		const double a = 1, b = 0.7, c = 0.5;
//		const unsigned int j = 1, k = 1, l = 1;
//		values.at(i)[0] += c*2*PI_D*k*sin(2*PI_D*j*points[i](0))*cos(2*PI_D*k*points[i](1))*sin(2*PI_D*l*points[i](2))
//						- b*2*PI_D*k*sin(2*PI_D*j*points[i](0))*sin(2*PI_D*k*points[i](1))*cos(2*PI_D*l*points[i](2));
//		values.at(i)[1] += a*2*PI_D*k*sin(2*PI_D*j*points[i](0))*sin(2*PI_D*k*points[i](1))*cos(2*PI_D*l*points[i](2))
//						- c*2*PI_D*k*cos(2*PI_D*j*points[i](0))*sin(2*PI_D*k*points[i](1))*sin(2*PI_D*l*points[i](2));
//		values.at(i)[2] += b*2*PI_D*k*cos(2*PI_D*j*points[i](0))*sin(2*PI_D*k*points[i](1))*sin(2*PI_D*l*points[i](2))
//						- a*2*PI_D*k*sin(2*PI_D*j*points[i](0))*cos(2*PI_D*k*points[i](1))*sin(2*PI_D*l*points[i](2));






		// Gradient
//		values.at(i)[0] = points[i](1) * points[i](2);
//		values.at(i)[1] = points[i](0) * points[i](2);
//		values.at(i)[2] = points[i](0) * points[i](1);

		// Curl
//		const unsigned int k = 1;
//		values.at(i)[0] = - 2 * PI_D * k * cos(2 * PI_D * k * points[i](1) * points[i](2)) * points[i](1);
//		values.at(i)[1] = - 2 * PI_D * k * cos(2 * PI_D * k * points[i](0) * points[i](2)) * points[i](2);
//		values.at(i)[2] = - 2 * PI_D * k * cos(2 * PI_D * k * points[i](0) * points[i](1)) * points[i](0);

		// Curl
//		const unsigned int k = 1;
//		values.at(i)[0] = 2 * PI_D * k * (cos(2 * PI_D * k * points[i](0) * points[i](1))
//						- cos(2 * PI_D * k * points[i](0) * points[i](2))) * points[i](0);
//		values.at(i)[1] = 2 * PI_D * k * (cos(2 * PI_D * k * points[i](1) * points[i](2))
//						- cos(2 * PI_D * k * points[i](0) * points[i](1))) * points[i](1);
//		values.at(i)[2] = 2 * PI_D * k * (cos(2 * PI_D * k * points[i](0) * points[i](2))
//						- cos(2 * PI_D * k * points[i](1) * points[i](2))) * points[i](2);

		// Curl
		values.at(i)[0] += - points[i](1);
		values.at(i)[1] += - points[i](2);
		values.at(i)[2] += - points[i](0);





//		values.at(i)[0] = 0;
//		values.at(i)[1] = std::pow(sin(PI_D*points[i](0))
//										* sin(PI_D*points[i](1))
//										* sin(PI_D*points[i](2)), 2);;
//		values.at(i)[2] = 0;

//		values.at(i)[0] += 1;
//		values.at(i)[1] += 1;
//		values.at(i)[2] += 1;
	}
}

/**
 * Implementation of (natural) boundary values for u.
 *
 * @param p
 * @param component = 0
 * @return
 */
double
BoundaryDivergenceValues_u::value (const Point<3>  &p,
						  const unsigned int /*component*/) const
{
	return 0.0;
}


/**
 * Implementation of (essential) boundary values for sigma.
 *
 * @param p
 * @param component = 0
 * @return
 */
void
BoundaryValues_sigma::value_list (const std::vector<Point<3> > &points,
		 	 	 	 	 	 	 std::vector<Tensor<1,3> >    &values) const
{
	Assert (points.size() == values.size(),
			ExcDimensionMismatch (points.size(), values.size()));

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p].clear ();

		// Set values of components
		for (unsigned int d=0; d<3; ++d)
			values[p][d] = 0;
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

	const int k = 19;

	const double alpha = PI_D/3,
				beta = PI_D/6,
				gamma = PI_D/4;

	// Description with Euler angles
	Tensor<2,3> rot;
	rot[0][0] = cos(alpha)*cos(gamma) - sin(alpha)*cos(beta)*sin(gamma);
	rot[0][1] = -cos(alpha)*sin(gamma) - sin(alpha)*cos(beta)*cos(gamma);
	rot[0][2] = sin(alpha)*sin(beta);
	rot[1][0] = sin(alpha)*cos(gamma) + cos(alpha)*cos(beta)*sin(gamma);
	rot[1][1] = -sin(alpha)*sin(gamma) + cos(alpha)*cos(beta)*cos(gamma);
	rot[1][2] = -cos(alpha)*sin(beta);
	rot[2][0] = sin(beta)*sin(gamma);
	rot[2][1] = sin(beta)*cos(gamma);
	rot[2][2] = cos(beta);

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p].clear ();

		//This is just diagonal
		values[p][0][0] = 1 / (1.0 - 0.999 * sin( 2 * PI_D * k * points.at(p)(1) ));
		values[p][1][1] = 0.01 / (1.0 - 0.999 * sin( 2 * PI_D * k * points.at(p)(2) ));
		values[p][2][2] = 0.01 / (1.0 - 0.999 * sin( 2 * PI_D * k * points.at(p)(0) ));

		// Now rotation leads to anisotropy
//		values[p] = rot * values[p] * transpose (rot);
	}

//	for (unsigned int p=0; p<points.size(); ++p)
//		for (unsigned int d=0; d<3; ++d)
//			values[p][d][d] = 1;
}



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

	const int k = 0;

	for (unsigned int p=0; p<points.size(); ++p)
	{
		values[p] = 1.0 * (1.0 - 0.99 * (
									  sin(2*PI_D*k*(  points.at(p)(0)*points.at(p)(2)  ))
//									+ cos(2*PI_D*k*(  points.at(p)(1)  ))
//									+ sin(2*PI_D*k*(  points.at(p)(2)  ))
									)
					);
	}

//	for (unsigned int p=0; p<points.size(); ++p)
//		values[p] = 1;
}


/**
 * Implementation of reaction rate.
 *
 * @param p
 * @param component = 0
 * @return
 */
double
ReactionRate::value (const Point<3>  &p,
								  const unsigned int /*component*/) const
{
	return 0.0;
}

} // end namespace HelmholtzProblem
