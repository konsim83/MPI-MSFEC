#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace Parameters
{

using namespace dealii;

namespace NedRT
{

struct ParametersMs
{
	ParametersMs(const std::string &parameter_filename);

	static void declare_parameters(ParameterHandler &prm);
	void        parse_parameters(ParameterHandler &prm);

	bool compute_solution;
	bool verbose;
	bool verbose_basis;
	bool use_direct_solver; /* This is often better for 2D problems. */
	bool use_direct_solver_basis; /* This is often better for 2D problems. */
	bool renumber_dofs; /* Reduce bandwidth in either system component */

	unsigned int n_refine_global;
	unsigned int n_refine_local;

	std::string filename_output;
};



struct ParametersBasis
{
	ParametersBasis(const ParametersMs &param_ms);
	ParametersBasis(const ParametersBasis &other); // This the the copy constructor

	const unsigned int degree = 0;
	const bool set_to_std = false;

	bool verbose;
	bool use_direct_solver; /* This is often better for 2D problems. */
	bool renumber_dofs; /* Reduce bandwidth in either system component */

	bool output_flag;

	unsigned int n_refine_global;
	unsigned int n_refine_local;

	std::string filename_global;
};


}  // namespace NedRT

} // namespace Parameters

#endif /* INCLUDE_PARAMETERS_H_ */
