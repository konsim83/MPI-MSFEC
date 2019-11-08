#include "ned_rt_eqn_coeff_B.h"

namespace LaplaceProblem
{

using namespace dealii;

/**
 * Constructor
 */
Diffusion_B::Diffusion_B (const std::string &parameter_filename)
:
Functions::ParsedFunction<3>()
{
	// A parameter handler
	ParameterHandler prm;

	// Declare a section for the function we need
	prm.enter_subsection("Equation parameters");
		prm.enter_subsection("Diffusion B");
			Functions::ParsedFunction<3>::declare_parameters(prm, 1);
		prm.leave_subsection();
	prm.leave_subsection();

	// open file
	std::ifstream parameter_file(parameter_filename);

	// Parse an input file.
	prm.parse_input(parameter_file,
			/* filename = */ "generated_parameter.in",
			/* last_line = */ "",
			/* skip_undefined = */ true);

	// Initialize the ParsedFunction object with the given file
	prm.enter_subsection("Equation parameters");
		prm.enter_subsection("Diffusion B");
			this->parse_parameters(prm);
		prm.leave_subsection();
	prm.leave_subsection();
}


/**
 * Constructor
 */
DiffusionInverse_B::DiffusionInverse_B (const std::string &parameter_filename)
:
Functions::ParsedFunction<3>()
{
	// A parameter handler
	ParameterHandler prm;

	// Declare a section for the function we need
	prm.enter_subsection("Equation parameters");
		prm.enter_subsection("Diffusion B inverse");
			Functions::ParsedFunction<3>::declare_parameters(prm, 1);
		prm.leave_subsection();
	prm.leave_subsection();

	// open file
	std::ifstream parameter_file(parameter_filename);

	// Parse an input file.
	prm.parse_input(parameter_file,
			/* filename = */ "generated_parameter.in",
			/* last_line = */ "",
			/* skip_undefined = */ true);

	// Initialize the ParsedFunction object with the given file
	prm.enter_subsection("Equation parameters");
		prm.enter_subsection("Diffusion B inverse");
			this->parse_parameters(prm);
		prm.leave_subsection();
	prm.leave_subsection();
}

} // end namespace LaplaceProblem
