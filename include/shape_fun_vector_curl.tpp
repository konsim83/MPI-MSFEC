#ifndef INCLUDE_SHAPE_FUN_VECTOR_CURL_TPP_
#define INCLUDE_SHAPE_FUN_VECTOR_CURL_TPP_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <vector>


namespace ShapeFun
{

using namespace dealii;

/**
 * Class for evaluations of vector valued shape functions.
 *
 * @author Konrad Simon, 2019
 */
template <int dim>
class ShapeFunctionVectorCurl : public Function<dim>
{
public:
	/**
	 * Constructor takes a vector finite element like <code>BDM<dim> <\code>
	 * or <code> RaviartThomas<dim> <\code> and a cell iterator pointing to
	 * a certain cell in a triangulation.
	 *
	 * @param fe
	 * @param cell
	 * @param verbose = false
	 */
	ShapeFunctionVectorCurl (const FiniteElement<dim> &fe,
						typename Triangulation<dim>::cell_iterator& cell,
						bool verbose = false);

	/**
	 * Evaluate shape function at point <code> p<\code>
	 *
	 * @param[in] p
	 * @param[out] value
	 */
	void vector_value (const Point<dim> &p, Vector<double> &value) const;

	/**
	 * Evaluate shape function at point list <code> points <\code>
	 *
	 * @param[in] points
	 * @param[out] values
	 */
	void vector_value_list (const std::vector<Point<dim> > &points,
								 	 std::vector<Vector<double>>    &values) const;

	/**
	 * Evaluate shape function at point list <code> points <\code>
	 *
	 * @param[in] points
	 * @param[out] values
	 */
	void tensor_value_list (const std::vector<Point<dim> > &points,
									 std::vector<Tensor<1,dim>>    &values) const;

	/**
	 * Set pointer to current cell (actually and iterator).
	 *
	 * @param cell
	 */
	void set_current_cell (const typename Triangulation<dim>::cell_iterator &cell);

	/**
	 * Set shape function index.
	 *
	 * @param index
	 */
	void set_shape_fun_index (unsigned int index);

private:
	SmartPointer<const FiniteElement<dim> >		fe_ptr;
	const unsigned int	dofs_per_cell;
	unsigned int		shape_fun_index;

	const MappingQ<dim>  	mapping;

	typename Triangulation<dim>::cell_iterator	*current_cell_ptr;

	const FEValuesExtractors::Vector curl;

	const bool verbose;
};



template <int dim>
inline
ShapeFunctionVectorCurl<dim>::ShapeFunctionVectorCurl (const FiniteElement<dim> &fe,
			typename Triangulation<dim>::cell_iterator &cell,
			bool verbose)
:
Function<dim> (dim),
fe_ptr (&fe),
dofs_per_cell (fe_ptr->dofs_per_cell),
shape_fun_index (0),
mapping(1),
current_cell_ptr(&cell),
curl(0),
verbose(verbose)
{
	// If element is primitive it is invalid.
	// Also there must not be more than one block.
	// This excludes FE_Systems.
	Assert( (!fe_ptr->is_primitive()),
			FETools::ExcInvalidFE ());
	Assert (fe_ptr->n_blocks() == 1,
					ExcDimensionMismatch (1, fe_ptr->n_blocks()));
	if (verbose)
	{
		std::cout << "\n		Constructed vector shape function for   "
				<< fe_ptr->get_name ()
				<< "   on cell   [";
				for (unsigned int i=0; i<(std::pow(2,dim)-1); ++i)
				{
					std::cout
					<< cell->vertex (i) << ", \n";

				}
				std::cout
				<< cell->vertex (std::pow(2,dim)-1) << "]\n"
				<< std::endl;
	}
}


template <int dim>
inline void
ShapeFunctionVectorCurl<dim>::set_current_cell (const typename Triangulation<dim>::cell_iterator &cell)
{
	current_cell_ptr = &cell;
}


template <int dim>
inline void
ShapeFunctionVectorCurl<dim>::set_shape_fun_index(unsigned int index)
{
	shape_fun_index = index;
}


template <int dim>
inline void
ShapeFunctionVectorCurl<dim>::vector_value(const Point<dim>   &p, Vector<double> &value) const
{
	// Map physical points to reference cell
	Point<dim> point_on_ref_cell (mapping.transform_real_to_unit_cell(*current_cell_ptr, p));

	// Copy-assign a fake quadrature rule form mapped point
	Quadrature<dim> fake_quadrature (point_on_ref_cell);

	// Update he fe_values object
	FEValues<dim>		fe_values(*fe_ptr, fake_quadrature,
			update_values  | update_gradients |  update_quadrature_points);

	fe_values.reinit(*current_cell_ptr);

	(fe_values[curl].curl (shape_fun_index, /* q_index */ 0)).unroll(value);
}


template <int dim>
inline void
ShapeFunctionVectorCurl<dim>::vector_value_list(const std::vector<Point<dim> > &points,
		 	 	 	 	 	 	 std::vector<Vector<double>>    &values) const
{
	Assert (points.size() == values.size(),
				ExcDimensionMismatch (points.size(), values.size()));

	const unsigned int   n_q_points = points.size();

	// Map physical points to reference cell
	std::vector<Point<dim>> points_on_ref_cell (n_q_points);
	for (unsigned int i = 0; i < n_q_points; ++i)
	{
		points_on_ref_cell.at(i) = mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
	}

	// Copy-assign a fake quadrature rule form mapped point
	Quadrature<dim> fake_quadrature (points_on_ref_cell);

	// Update he fe_values object
	FEValues<dim>		fe_values(*fe_ptr, fake_quadrature,
			update_values  | update_gradients |  update_quadrature_points);

	fe_values.reinit(*current_cell_ptr);

	for (unsigned int i = 0; i < n_q_points; ++i)
	{
		(fe_values[curl].curl (shape_fun_index,
												/* q_index */ i)).unroll(values.at(i));
	}
}

template <int dim>
inline void
ShapeFunctionVectorCurl<dim>::tensor_value_list(const std::vector<Point<dim> > &points,
		 	 	 	 	 	 	 std::vector<Tensor<1,dim>>    &values) const
{
	Assert (points.size() == values.size(),
				ExcDimensionMismatch (points.size(), values.size()));

	const unsigned int   n_q_points = points.size();

	// Map physical points to reference cell
	std::vector<Point<dim>> points_on_ref_cell (n_q_points);
	for (unsigned int i = 0; i < n_q_points; ++i)
	{
		points_on_ref_cell.at(i) = mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
	}

	// Copy-assign a fake quadrature rule form mapped point
	Quadrature<dim> fake_quadrature (points_on_ref_cell);

	// Update he fe_values object
	FEValues<dim>		fe_values(*fe_ptr, fake_quadrature,
			update_values  | update_gradients |  update_quadrature_points);

	fe_values.reinit(*current_cell_ptr);

	for (unsigned int i = 0; i < n_q_points; ++i)
	{
		values.at(i) = fe_values[curl].curl (shape_fun_index,
									/* q_index */ i);
	}
}

} // close namespace

#endif /* INCLUDE_SHAPE_FUN_VECTOR_CURL_TPP_ */
