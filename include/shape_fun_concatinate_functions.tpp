#ifndef INCLUDE_SHAPE_FUN_CONCATINATE_FUNCTIONS_TPP_
#define INCLUDE_SHAPE_FUN_CONCATINATE_FUNCTIONS_TPP_


#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  class ShapeFunctionConcatinateVector : public Function<dim>
  {
  public:
    ShapeFunctionConcatinateVector(const Function<dim> &function1,
                                   const Function<dim> &function2);

    virtual double
    value(const Point<dim> &p, const unsigned int component) const override;

    //	void value_list (const std::vector<Point<dim> > &points,
    //						 std::vector<double>    &values,
    //						 const unsigned int component) const;

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  values) const override;

  private:
    SmartPointer<const Function<dim>> function_ptr1;
    SmartPointer<const Function<dim>> function_ptr2;
  };


  template <int dim>
  ShapeFunctionConcatinateVector<dim>::ShapeFunctionConcatinateVector(
    const Function<dim> &function1,
    const Function<dim> &function2)
    : Function<dim>(function1.n_components + function2.n_components)
    , function_ptr1(&function1)
    , function_ptr2(&function2)
  {}


  template <int dim>
  inline double
  ShapeFunctionConcatinateVector<dim>::value(const Point<dim> & p,
                                             const unsigned int component) const
  {
    if (component < function_ptr1->n_components)
      {
        Vector<double> value1(function_ptr1->n_components);
        function_ptr1->vector_value(p, value1);
        return value1(component);
      }
    else
      {
        Vector<double> value2(function_ptr2->n_components);
        function_ptr2->vector_value(p, value2);
        return value2(component - function_ptr1->n_components);
      }
  }


  template <int dim>
  inline void
  ShapeFunctionConcatinateVector<dim>::vector_value(const Point<dim> &p,
                                                    Vector<double> &value) const
  {
    Vector<double> value1(function_ptr1->n_components);
    function_ptr1->vector_value(p, value1);

    Vector<double> value2(function_ptr2->n_components);
    function_ptr2->vector_value(p, value2);

    for (unsigned int j = 0; j < function_ptr1->n_components; ++j)
      value(j) = value1(j);
    for (unsigned int j = 0; j < function_ptr2->n_components; ++j)
      value(function_ptr1->n_components + j) = value2(j);
  }


  template <int dim>
  inline void
  ShapeFunctionConcatinateVector<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    Vector<double> value1(function_ptr1->n_components);
    Vector<double> value2(function_ptr2->n_components);

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        value1 = 0;
        value2 = 0;
        function_ptr1->vector_value(points[i], value1);
        function_ptr2->vector_value(points[i], value2);

        for (unsigned int j = 0; j < function_ptr1->n_components; ++j)
          {
            values[i](j) = value1(j);
          }
        for (unsigned int j = 0; j < function_ptr2->n_components; ++j)
          {
            values[i](function_ptr1->n_components + j) = value2(j);
          }
      }
  }

} // namespace ShapeFun

#endif /* INCLUDE_SHAPE_FUN_CONCATINATE_FUNCTIONS_TPP_ */
