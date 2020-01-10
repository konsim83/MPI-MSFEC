#include "eqn_rhs.h"

namespace LaplaceProblem
{
  using namespace dealii;

  RightHandSideExactLin::RightHandSideExactLin(
    const std::string &parameter_filename)
    : TensorFunction<1, 3>()
    , ExactSolutionLin_Data(parameter_filename)
    , Diffusion_A_Data(parameter_filename)
    , Diffusion_B_Data(parameter_filename)
  {
    R_trans_curl_u = transpose(rot) * curl_u;
  }


  void
  RightHandSideExactLin::value_list(const std::vector<Point<3>> &points,
                                    std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i].clear();

        // -grad (B div_u), works only with a certain
        values[i][0] =
          2 * pi * k * div_u * scale * alpha * cos(2 * pi * k * points[i](0));
        values[i][1] = 0;
        values[i][2] = 0;

        values[i][0] +=
          2 * pi *
          (-rot[2][1] * scale_y * alpha_y * k_y *
             cos(2 * pi * k_y * points[i](1)) * R_trans_curl_u[1] +
           rot[1][2] * scale_z * alpha_z * k_z *
             cos(2 * pi * k_z * points[i](2)) * R_trans_curl_u[2]);

        values[i][1] +=
          2 * pi *
          (-rot[0][2] * scale_z * alpha_z * k_z *
             cos(2 * pi * k_z * points[i](2)) * R_trans_curl_u[2] +
           rot[2][0] * scale_x * alpha_x * k_x *
             cos(2 * pi * k_x * points[i](0)) * R_trans_curl_u[0]);

        values[i][2] +=
          2 * pi *
          (-rot[1][0] * scale_x * alpha_x * k_x *
             cos(2 * pi * k_x * points[i](0)) * R_trans_curl_u[0] +
           rot[0][1] * scale_y * alpha_y * k_y *
             cos(2 * pi * k_y * points[i](1)) * R_trans_curl_u[1]);
      }
  }


  void
  RightHandSideParsed::value_list(const std::vector<Point<3>> &points,
                                  std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values.at(i).clear();

        //////////////////////////////////////////////////////////
        ////   Zero BCs   ////////////////////////////////////////
        //////////////////////////////////////////////////////////

        /*
         * Gradient_0 of
         * V = sin(2*PI*j*x)*sin(2*PI*k*y)*sin(2*PI*l*z)
         */
        //		const double a = 1;
        //		const unsigned int j = 2, k = 2, l = 2;
        //		values.at(i)[0] =
        //a*2*numbers::PI*j*cos(2*numbers::PI*j*points[i](0))*sin(2*numbers::PI*k*points[i](1))*sin(2*numbers::PI*l*points[i](2));
        //		values.at(i)[1] =
        //a*2*numbers::PI*k*sin(2*numbers::PI*j*points[i](0))*cos(2*numbers::PI*k*points[i](1))*sin(2*numbers::PI*l*points[i](2));
        //		values.at(i)[2] =
        //a*2*numbers::PI*l*sin(2*numbers::PI*j*points[i](0))*sin(2*numbers::PI*k*points[i](1))*cos(2*numbers::PI*l*points[i](2));

        /*
         * Gradient_0 of
         * V = x(x-1)y(y-1)z(z-1)
         */
        double x2       = points[i](0) * points[i](0) - points[i](0);
        double y2       = points[i](1) * points[i](1) - points[i](1);
        double z2       = points[i](2) * points[i](2) - points[i](2);
        values.at(i)[0] = 10 * (2 * points[i](0) - 1) * y2 * z2;
        values.at(i)[1] = 10 * (2 * points[i](1) - 1) * x2 * z2;
        values.at(i)[2] = 10 * (2 * points[i](2) - 1) * x2 * y2;

        /*
         * Curl_0
         */
        //		const double a = 1, b = 0.7, c = 0.5;
        //		const unsigned int j = 1, k = 1, l = 1;
        //		values.at(i)[0] +=
        //c*2*numbers::PI*k*sin(2*numbers::PI*j*points[i](0))*cos(2*numbers::PI*k*points[i](1))*sin(2*numbers::PI*l*points[i](2))
        //						-
        //b*2*numbers::PI*k*sin(2*numbers::PI*j*points[i](0))*sin(2*numbers::PI*k*points[i](1))*cos(2*numbers::PI*l*points[i](2));
        //		values.at(i)[1] +=
        //a*2*numbers::PI*k*sin(2*numbers::PI*j*points[i](0))*sin(2*numbers::PI*k*points[i](1))*cos(2*numbers::PI*l*points[i](2))
        //						-
        //c*2*numbers::PI*k*cos(2*numbers::PI*j*points[i](0))*sin(2*numbers::PI*k*points[i](1))*sin(2*numbers::PI*l*points[i](2));
        //		values.at(i)[2] +=
        //b*2*numbers::PI*k*cos(2*numbers::PI*j*points[i](0))*sin(2*numbers::PI*k*points[i](1))*sin(2*numbers::PI*l*points[i](2))
        //						-
        //a*2*numbers::PI*k*sin(2*numbers::PI*j*points[i](0))*cos(2*numbers::PI*k*points[i](1))*sin(2*numbers::PI*l*points[i](2));
        //////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////



        //////////////////////////////////////////////////////////
        ////   No BCs   //////////////////////////////////////////
        //////////////////////////////////////////////////////////

        // Gradient
        //		values.at(i)[0] = points[i](1) * points[i](2);
        //		values.at(i)[1] = points[i](0) * points[i](2);
        //		values.at(i)[2] = points[i](0) * points[i](1);

        // Curl
        //		const unsigned int k = 1;
        //		values.at(i)[0] = - 2 * numbers::PI * k * cos(2 * numbers::PI * k
        //* points[i](1) * points[i](2)) * points[i](1); 		values.at(i)[1] = - 2 *
        //numbers::PI * k * cos(2 * numbers::PI * k * points[i](0) *
        //points[i](2)) * points[i](2); 		values.at(i)[2] = - 2 * numbers::PI * k
        //* cos(2 * numbers::PI * k * points[i](0) * points[i](1)) *
        //points[i](0);

        // Curl
        //		const unsigned int k = 1;
        //		values.at(i)[0] = 2 * numbers::PI * k * (cos(2 * numbers::PI * k *
        //points[i](0) * points[i](1))
        //						- cos(2 * numbers::PI * k * points[i](0) * points[i](2)))
        //* points[i](0); 		values.at(i)[1] = 2 * numbers::PI * k * (cos(2 *
        //numbers::PI * k * points[i](1) * points[i](2))
        //						- cos(2 * numbers::PI * k * points[i](0) * points[i](1)))
        //* points[i](1); 		values.at(i)[2] = 2 * numbers::PI * k * (cos(2 *
        //numbers::PI * k * points[i](0) * points[i](2))
        //						- cos(2 * numbers::PI * k * points[i](1) * points[i](2)))
        //* points[i](2);

        // Curl
        values.at(i)[0] += -10 * points[i](1);
        values.at(i)[1] += -10 * points[i](2);
        values.at(i)[2] += -10 * points[i](0);
        //////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////



        //////////////////////////////////////////////////////////
        ////   Anything   ////////////////////////////////////////
        //////////////////////////////////////////////////////////
        //		values.at(i)[0] = 0;
        //		values.at(i)[1] = std::pow(sin(numbers::PI*points[i](0))
        //										* sin(numbers::PI*points[i](1))
        //										* sin(numbers::PI*points[i](2)), 2);;
        //		values.at(i)[2] = 0;

        //		values.at(i)[0] += 1;
        //		values.at(i)[1] += 1;
        //		values.at(i)[2] += 1;
        //////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////
        //////////////////////////////////////////////////////////
      }
  }

} // end namespace LaplaceProblem
