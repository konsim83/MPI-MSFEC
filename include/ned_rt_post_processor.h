#ifndef INCLUDE_NED_RT_POST_PROCESSOR_H_
#define INCLUDE_NED_RT_POST_PROCESSOR_H_

// deal.ii
#include <deal.II/numerics/data_postprocessor.h>

// C++
#include <vector>

// my headers
#include "config.h"
#include "ned_rt_eqn_coeff_A.h"
#include "ned_rt_eqn_coeff_B.h"


namespace LaplaceProblem
{
using namespace dealii;

class NedRT_PostProcessor : public DataPostprocessor<3>
{
public:
	/**
	 * Constructor.
	 */
	NedRT_PostProcessor ();

	/**
	 * This is the actual evaluation routine of the  post processor.
	 */
	virtual void evaluate_vector_field(
			const DataPostprocessorInputs::Vector<3> &inputs,
			std::vector<Vector<double>> &computed_quantities) const override;

	/**
	 * Define all names of solution and post processed quantities.
	 */
	virtual std::vector<std::string> get_names() const override;

	/**
	 * Define all interpretations of solution and post processed quantities.
	 */
	virtual std::vector<
			DataComponentInterpretation::DataComponentInterpretation>
	get_data_component_interpretation() const override;

	/**
	 * Define all necessary update flags when looping over cells to be post processed.
	 */
	virtual UpdateFlags get_needed_update_flags() const override;

private:
	const DiffusionInverse_A	a_inverse;
	const Diffusion_B     		b;
};

} // end namespace LaplaceProblem

#endif /* INCLUDE_NED_RT_POST_PROCESSOR_H_ */
