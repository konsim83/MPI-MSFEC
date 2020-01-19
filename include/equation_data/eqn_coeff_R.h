#ifndef EQN_COEFF_R_H_
#define EQN_COEFF_R_H_

#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

// std library
#include <cmath>
#include <cstdlib>
#include <vector>

namespace EquationData
{
  using namespace dealii;

  /**
   * This lower order term can regularize the weak form. If it vanishes
   * we have a Darcy problem (in the last element of the de Rham complex).
   */
  class ReactionRate : public Function<3>
  {
  public:
    /**
     * Constructor.
     */
    ReactionRate()
      : Function<3>()
    {}

    /**
     * Implementation of lower order term.
     *
     * @param p
     * @param component = 0
     */
    virtual void
      value_list(const std::vector<Point<3>> &points,
                 std::vector<double> &        values,
                 const unsigned int           component = 0) const override;
  };

} // end namespace EquationData

#endif /* EQN_COEFF_R_H_ */
