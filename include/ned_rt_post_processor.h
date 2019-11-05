#ifndef INCLUDE_NED_RT_POST_PROCESSOR_H_
#define INCLUDE_NED_RT_POST_PROCESSOR_H_

#include <deal.II/numerics/data_postprocessor.h>

// my headers
#include "config.h"
#include "parameters.h"

#include "ned_rt_eqn_data.h"

namespace LaplaceProblem
{
using namespace dealii;

class NedRT_PostProcessor : public DataPostprocessor<3>
{
public:
	// Constructor
	NedRT_PostProcessor(unsigned int partition);

	virtual void evaluate_vector_field(
			const DataPostprocessorInputs::Vector<3> &inputs,
			std::vector<Vector<double>> &computed_quantities) const override;

	virtual std::vector<std::string> get_names() const override;

	virtual std::vector<
			DataComponentInterpretation::DataComponentInterpretation>
		get_data_component_interpretation() const override;

	virtual UpdateFlags get_needed_update_flags() const override;

private:
	const unsigned int partition;
};

} // end namespace LaplaceProblem

#endif /* INCLUDE_NED_RT_POST_PROCESSOR_H_ */
