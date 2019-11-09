#include "q_ned_post_processor.h"

namespace LaplaceProblem
{

using namespace dealii;

/**
 * Constructor
 */
QNed_PostProcessor::QNed_PostProcessor(const std::string &parameter_filename)
:
a(parameter_filename),
b_inverse(parameter_filename)
{}


std::vector<std::string>
QNed_PostProcessor::get_names() const
{
	std::vector<std::string> solution_names(1, "div_u");
	solution_names.emplace_back("curl_u");
	solution_names.emplace_back("curl_u");
	solution_names.emplace_back("curl_u");
	solution_names.emplace_back("A_curl_u");
	solution_names.emplace_back("A_curl_u");
	solution_names.emplace_back("A_curl_u");

	return solution_names;
}


std::vector<DataComponentInterpretation::DataComponentInterpretation>
QNed_PostProcessor::get_data_component_interpretation() const
{
	// div u = -B_inv*sigma
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		interpretation(1, DataComponentInterpretation::component_is_scalar);

	// curl u
	interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

	// A*curl u
	interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
	interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

	return interpretation;
}


UpdateFlags
QNed_PostProcessor::get_needed_update_flags() const
{
  return update_values | update_gradients | update_quadrature_points;
}


void
QNed_PostProcessor::evaluate_vector_field(
	const DataPostprocessorInputs::Vector<3> &inputs,
	std::vector<Vector<double>> &computed_quantities) const
{
	const unsigned int n_quadrature_points = inputs.solution_values.size();

	Assert(inputs.solution_gradients.size() == n_quadrature_points,
		 ExcInternalError());

	Assert(computed_quantities.size() == n_quadrature_points,
		 ExcInternalError());

	Assert(inputs.solution_values[0].size() == 4, ExcInternalError());

	std::vector<Tensor<2,3>> 	a_values (n_quadrature_points);
	std::vector<double>		 	b_inverse_values (n_quadrature_points);

	// Evaluate A and B at quadrature points
	a.value_list (inputs.evaluation_points,
			a_values);
	b_inverse.value_list(inputs.evaluation_points,
			b_inverse_values);

	for (unsigned int q = 0; q < n_quadrature_points; ++q)
	{
		// div u = -B*sigma
		computed_quantities[q](0) = - b_inverse_values[q] * inputs.solution_values[q][0];

		// curl u
		computed_quantities[q](1) = inputs.solution_gradients[q][3][2] - inputs.solution_gradients[q][2][3];
		computed_quantities[q](2) = inputs.solution_gradients[q][3][1] - inputs.solution_gradients[q][1][3];
		computed_quantities[q](3) = inputs.solution_gradients[q][2][1] - inputs.solution_gradients[q][1][2];

		// A*curl u
		for (unsigned int d = 4; d < 7; ++d)
		{
			computed_quantities[q](d) = 0; // erase old stuff
			for (unsigned int i = 0; i < 3; ++i)
				computed_quantities[q](d) += a_values[q][d-4][i] * computed_quantities[q](i+1);
		}
	}
}

} // end namespace LaplaceProblem
