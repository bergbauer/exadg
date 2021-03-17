/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SOLVER_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SOLVER_H_

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

// ExaDG

// driver
#include <exadg/convection_diffusion/driver.h>

// utilities
#include <exadg/utilities/general_parameters.h>
#include <exadg/utilities/resolution_parameters.h>

namespace ExaDG
{
// forward declarations
template<int dim, typename Number>
std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>
get_application(std::string input_file);

void
create_input_file(std::string const & input_file)
{
  dealii::ParameterHandler prm;

  GeneralParameters general;
  general.add_parameters(prm);

  SpatialResolutionParameters spatial;
  spatial.add_parameters(prm);

  TemporalResolutionParameters temporal;
  temporal.add_parameters(prm);

  try
  {
    // we have to assume a default dimension and default Number type
    // for the automatic generation of a default input file
    unsigned int const Dim = 2;
    typedef double     Number;
    get_application<Dim, Number>(input_file)->add_parameters(prm);
  }
  catch(...)
  {
  }

  prm.print_parameters(input_file,
                       dealii::ParameterHandler::Short |
                         dealii::ParameterHandler::KeepDeclarationOrder);
}

template<int dim, typename Number>
void
run(std::string const & input_file,
    unsigned int const  degree,
    unsigned int const  refine_space,
    unsigned int const  refine_time,
    MPI_Comm const &    mpi_comm,
    bool const          is_test)
{
  Timer timer;
  timer.restart();

  std::shared_ptr<ConvDiff::Driver<dim, Number>> solver;
  solver.reset(new ConvDiff::Driver<dim, Number>(mpi_comm, is_test));

  std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>> application =
    get_application<dim, Number>(input_file);

  solver->setup(application, degree, refine_space, refine_time, false);

  solver->solve();

  if(not(is_test))
    solver->print_performance_results(timer.wall_time());
}

} // namespace ExaDG

int
main(int argc, char ** argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);

  std::string input_file;

  if(argc == 1)
  {
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // clang-format off
      std::cout << "To run the program, use:      ./solver input_file" << std::endl
                << "To setup the input file, use: ./solver input_file --help" << std::endl;
      // clang-format on
    }

    return 0;
  }
  else if(argc >= 2)
  {
    input_file = std::string(argv[1]);

    if(argc == 3 and std::string(argv[2]) == "--help")
    {
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
        ExaDG::create_input_file(input_file);

      return 0;
    }
  }

  ExaDG::GeneralParameters            general(input_file);
  ExaDG::SpatialResolutionParameters  spatial(input_file);
  ExaDG::TemporalResolutionParameters temporal(input_file);

  // k-refinement
  for(unsigned int degree = spatial.degree_min; degree <= spatial.degree_max; ++degree)
  {
    // h-refinement
    for(unsigned int refine_space = spatial.refine_space_min;
        refine_space <= spatial.refine_space_max;
        ++refine_space)
    {
      // dt-refinement
      for(unsigned int refine_time = temporal.refine_time_min;
          refine_time <= temporal.refine_time_max;
          ++refine_time)
      {
        // run the simulation
        if(general.dim == 2 && general.precision == "float")
          ExaDG::run<2, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else if(general.dim == 2 && general.precision == "double")
          ExaDG::run<2, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else if(general.dim == 3 && general.precision == "float")
          ExaDG::run<3, float>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else if(general.dim == 3 && general.precision == "double")
          ExaDG::run<3, double>(
            input_file, degree, refine_space, refine_time, mpi_comm, general.is_test);
        else
          AssertThrow(false,
                      dealii::ExcMessage("Only dim = 2|3 and precision=float|double implemented."));
      }
    }
  }

  return 0;
}

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SOLVER_H_ */
