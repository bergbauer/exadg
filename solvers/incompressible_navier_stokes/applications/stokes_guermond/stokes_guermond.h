/*
 * stokes_guermond.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_GUERMOND_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_GUERMOND_H_

// ExaDG
#include <exadg/grid/one_sided_cylindrical_manifold.h>

namespace ExaDG
{
namespace IncNS
{
namespace StokesGuermond
{
using namespace dealii;

enum class MeshType
{
  UniformCartesian,
  Complex
};

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi    = numbers::PI;
    double       sint  = std::sin(t);
    double       sinx  = std::sin(pi * p[0]);
    double       siny  = std::sin(pi * p[1]);
    double       sin2x = std::sin(2. * pi * p[0]);
    double       sin2y = std::sin(2. * pi * p[1]);
    if(component == 0)
      result = pi * sint * sin2y * std::pow(sinx, 2.);
    else if(component == 1)
      result = -pi * sint * sin2x * std::pow(siny, 2.);

    return result;
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure() : Function<dim>(1, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int /*component*/) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi   = numbers::PI;
    double       sint = std::sin(t);
    double       siny = std::sin(pi * p[1]);
    double       cosx = std::cos(pi * p[0]);
    result            = cosx * siny * sint;

    return result;
  }
};

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt() : Function<dim>(dim, 0.0)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi    = numbers::PI;
    double       cost  = std::cos(t);
    double       sinx  = std::sin(pi * p[0]);
    double       siny  = std::sin(pi * p[1]);
    double       sin2x = std::sin(2. * pi * p[0]);
    double       sin2y = std::sin(2. * pi * p[1]);
    if(component == 0)
      result = pi * cost * sin2y * std::pow(sinx, 2.);
    else if(component == 1)
      result = -pi * cost * sin2x * std::pow(siny, 2.);

    return result;
  }
};

/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */
template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(double const nu) : Function<dim>(dim, 0.0), viscosity(nu)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double t      = this->get_time();
    double result = 0.0;

    const double pi    = numbers::PI;
    double       sint  = std::sin(t);
    double       cost  = std::cos(t);
    double       sinx  = std::sin(pi * p[0]);
    double       siny  = std::sin(pi * p[1]);
    double       cosx  = std::cos(pi * p[0]);
    double       cosy  = std::cos(pi * p[1]);
    double       sin2x = std::sin(2. * pi * p[0]);
    double       sin2y = std::sin(2. * pi * p[1]);
    if(component == 0)
    {
      result = +pi * cost * sin2y * std::pow(sinx, 2.) -
               2. * std::pow(pi, 3.) * sint * sin2y * (1. - 4. * std::pow(sinx, 2.)) * viscosity -
               pi * sint * sinx * siny;
    }
    else if(component == 1)
    {
      result = -pi * cost * sin2x * std::pow(siny, 2.) +
               2. * std::pow(pi, 3.) * sint * sin2x * (1. - 4. * std::pow(siny, 2.)) * viscosity +
               pi * sint * cosx * cosy;
    }

    return result;
  }

private:
  double const viscosity;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const viscosity = 1.0e-6;

  MeshType const mesh_type = MeshType::UniformCartesian;

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type             = ProblemType::Unsteady;
    param.equation_type            = EquationType::Stokes;
    param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    param.right_hand_side          = true;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                   = SolverType::Unsteady;
    param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    param.time_step_size                = 1.e-2;
    param.max_number_of_time_steps      = 1e8;
    param.order_time_integrator         = 3;     // 1; // 2; // 3;
    param.start_with_low_order          = false; // true; // false;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // pressure level is undefined
    param.adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

    // divergence and continuity penalty terms
    param.use_divergence_penalty = false;
    param.use_continuity_penalty = false;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-20, 1.e-12);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-8);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; // Multigrid;


    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver

    // linear solver
    param.solver_momentum                = SolverMomentum::GMRES;
    param.solver_data_momentum           = SolverData(1e4, 1.e-12, 1.e-8, 100);
    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = false;

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    if(mesh_type == MeshType::UniformCartesian)
    {
      // Uniform Cartesian grid
      const double left = 0.0, right = 1.0;
      GridGenerator::hyper_cube(*triangulation, left, right);

      /****** test one-sided spherical manifold *********/
      if(false)
      {
        Point<dim> center                               = Point<dim>();
        center[0]                                       = 1.15;
        center[1]                                       = 0.5;
        typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
        cell->set_all_manifold_ids(10);
        //    cell->vertex(0)[1] = -1.0;
        //    cell->vertex(2)[1] = 2.0;
        //    cell->vertex(4)[1] = -1.0;
        //    cell->vertex(6)[1] = 2.0;

        static std::shared_ptr<Manifold<dim>> my_manifold = std::shared_ptr<Manifold<dim>>(
          static_cast<Manifold<dim> *>(new OneSidedCylindricalManifold<dim>(cell, 1, center)));
        triangulation->set_manifold(10, *my_manifold);
      }
      /****** test one-sided spherical manifold *********/

      triangulation->refine_global(n_refine_space);
    }
    else if(mesh_type == MeshType::Complex)
    {
      // Complex Geometry
      const double       left = -1.0, right = 1.0;
      Triangulation<dim> tria1, tria2;
      const double       radius = (right - left) * 0.25;
      const double       width  = right - left;
      GridGenerator::hyper_shell(
        tria1, Point<dim>(), radius, 0.5 * width * std::sqrt(dim), 2 * dim);
      tria1.reset_all_manifolds();
      if(dim == 2)
      {
        GridTools::rotate(numbers::PI / 4, tria1);
      }
      GridGenerator::hyper_ball(tria2, Point<dim>(), radius);
      GridGenerator::merge_triangulations(tria1, tria2, *triangulation);
      triangulation->set_all_manifold_ids(0);
      for(typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
          cell != triangulation->end();
          ++cell)
      {
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for(unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
          {
            if(std::abs(cell->face(f)->vertex(v).norm() - radius) > 1e-12)
              face_at_sphere_boundary = false;
          }
          if(face_at_sphere_boundary)
          {
            cell->face(f)->set_all_manifold_ids(1);
          }
        }
      }
      static const SphericalManifold<dim> spherical_manifold;
      triangulation->set_manifold(1, spherical_manifold);

      triangulation->refine_global(n_refine_space);
    }
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    // test case with pure Dirichlet boundary conditions for velocity
    // all boundaries have ID = 0 by default

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new AnalyticalSolutionVelocity<dim>()));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new PressureBC_dudt<dim>()));
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
    field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
    field_functions->right_hand_side.reset(new RightHandSide<dim>(viscosity));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 10;
    pp_data.output_data.write_divergence     = false;
    pp_data.output_data.degree               = degree;

    // calculation of velocity error
    pp_data.error_data_u.analytical_solution_available = true;
    pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
    pp_data.error_data_u.calculate_relative_errors = false;
    pp_data.error_data_u.error_calc_start_time     = start_time;
    pp_data.error_data_u.error_calc_interval_time  = (end_time - start_time) / 10;
    pp_data.error_data_u.name                      = "velocity";

    // ... pressure error
    pp_data.error_data_p.analytical_solution_available = true;
    pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
    pp_data.error_data_p.calculate_relative_errors = false;
    pp_data.error_data_p.error_calc_start_time     = start_time;
    pp_data.error_data_p.error_calc_interval_time  = (end_time - start_time) / 10;
    pp_data.error_data_p.name                      = "pressure";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace StokesGuermond
} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_GUERMOND_H_ */
