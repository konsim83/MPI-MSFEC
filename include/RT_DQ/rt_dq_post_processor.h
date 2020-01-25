#ifndef INCLUDE_RT_DQ_POSTPROCESSOR_H_
#define INCLUDE_RT_DQ_POSTPROCESSOR_H_

// deal.ii
#include <deal.II/numerics/data_postprocessor.h>
#include <equation_data/eqn_coeff_A.h>

#include <vector>

// my headers
#include <config.h>

namespace RTDQ
{
  using namespace dealii;

  class RTDQ_PostProcessor : public DataPostprocessor<3>
  {
  public:
    /**
     * Constructor.
     */
    RTDQ_PostProcessor(const std::string &parameter_filename);

    /**
     * This is the actual evaluation routine of the  post processor.
     */
    virtual void
      evaluate_vector_field(
        const DataPostprocessorInputs::Vector<3> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

    /**
     * Define all names of solution and post processed quantities.
     */
    virtual std::vector<std::string>
      get_names() const override;

    /**
     * Define all interpretations of solution and post processed quantities.
     */
    virtual std::vector<
      DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation() const override;

    /**
     * Define all necessary update flags when looping over cells to be post
     * processed.
     */
    virtual UpdateFlags
      get_needed_update_flags() const override;

  private:
    const EquationData::DiffusionInverse_A a_inverse;
  };

} // end namespace RTDQ

#endif /* INCLUDE_RT_DQ_POSTPROCESSOR_H_ */
