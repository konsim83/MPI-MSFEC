#include "ned_rt_eqn_coeff_A.h"

namespace LaplaceProblem
{

using namespace dealii;

Diffusion_A_Data::Diffusion_A_Data(const std::string &parameter_filename)
:
alpha_(numbers::PI/3),
beta_(numbers::PI/6),
gamma_(numbers::PI/4)
{
	ParameterHandler prm;

	declare_parameters(prm);

	// open file
	std::ifstream parameter_file(parameter_filename);

	prm.parse_input(parameter_file,
			/* filename = */ "generated_parameter.in",
			/* last_line = */ "",
			/* skip_undefined = */ true);
	parse_parameters(prm);

	rot[0][0] = cos(alpha_)*cos(gamma_) - sin(alpha_)*cos(beta_)*sin(gamma_);
	rot[0][1] = -cos(alpha_)*sin(gamma_) - sin(alpha_)*cos(beta_)*cos(gamma_);
	rot[0][2] = sin(alpha_)*sin(beta_);
	rot[1][0] = sin(alpha_)*cos(gamma_) + cos(alpha_)*cos(beta_)*sin(gamma_);
	rot[1][1] = -sin(alpha_)*sin(gamma_) + cos(alpha_)*cos(beta_)*cos(gamma_);
	rot[1][2] = -cos(alpha_)*sin(beta_);
	rot[2][0] = sin(beta_)*sin(gamma_);
	rot[2][1] = sin(beta_)*cos(gamma_);
	rot[2][2] = cos(beta_);
}


void
Diffusion_A_Data::declare_parameters(
		ParameterHandler &prm)
{
	prm.enter_subsection("Equation parameters");
	{
		prm.enter_subsection("Diffusion A");
		{
			prm.declare_entry(
				"frequency x",
				"0",
				Patterns::Integer(0,100),
				"Frequency in first principal direction.");
			prm.declare_entry(
				"frequency y",
				"0",
				Patterns::Integer(0,100),
				"Frequency in second principal direction.");
			prm.declare_entry(
				"frequency z",
				"0",
				Patterns::Integer(0,100),
				"Frequency in third principal direction.");
			prm.declare_entry(
				"scale x",
				"1",
				Patterns::Double(0.0001,10000),
				"Scaling in first principal direction.");
			prm.declare_entry(
				"scale y",
				"1",
				Patterns::Double(0.0001,10000),
				"Scaling in second principal direction.");
			prm.declare_entry(
				"scale z",
				"1",
				Patterns::Double(0.0001,10000),
				"Scaling in third principal direction.");
			prm.declare_entry(
				"rotate",
				"true",
				Patterns::Bool(),
				"Choose whether to rotate the tensor or not.");
		}
		prm.leave_subsection();
	}
	prm.leave_subsection();
}


void
Diffusion_A_Data::parse_parameters(
		ParameterHandler &prm)
{
	prm.enter_subsection("Equation parameters");
	{
		prm.enter_subsection("Diffusion A");
		{
			k_x = prm.get_integer("frequency x");
			k_y = prm.get_integer("frequency y");
			k_z = prm.get_integer("frequency z");

			scale_x = prm.get_double("scale x");
			scale_y = prm.get_double("scale y");
			scale_z = prm.get_double("scale z");

			rotate = prm.get_bool("rotate");
		}
		prm.leave_subsection();
	}
	prm.leave_subsection();
}


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
		values[p][0][0] = scale_x * (1.0 - 0.99 * sin( 2 * numbers::PI * k_x * points.at(p)(0) ));
		values[p][1][1] = scale_y * (1.0 - 0.99 * sin( 2 * numbers::PI * k_y * points.at(p)(1) ));
		values[p][2][2] = scale_z * (1.0 - 0.99 * sin( 2 * numbers::PI * k_z * points.at(p)(2) ));
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
		values[p][0][0] = 1 / (scale_x * (1.0 - 0.99 * sin( 2 * numbers::PI * k_x * points.at(p)(0) )));
		values[p][1][1] = 1 / (scale_y * (1.0 - 0.99 * sin( 2 * numbers::PI * k_y * points.at(p)(1) )));
		values[p][2][2] = 1 / (scale_z * (1.0 - 0.99 * sin( 2 * numbers::PI * k_z * points.at(p)(2) )));
	}

	if (rotate)
		for (unsigned int p=0; p<points.size(); ++p)
		{
			// Rotation leads to anisotropy
			values[p] = rot * values[p] * transpose (rot);
		}
}

} // end namespace LaplaceProblem
