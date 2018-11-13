/*
 * time_int_bdf.h
 *
 *  Created on: Aug 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_

#include <deal.II/lac/vector_view.h>

#include <deal.II/lac/la_parallel_vector.h>

#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/push_back_vectors.h"
#include "time_integration/time_int_bdf_base.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<int dim, int fe_degree, typename value_type>
class TimeIntBDF : public TimeIntBDFBase
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  TimeIntBDF(std::shared_ptr<DGOperation<dim, fe_degree, value_type>> conv_diff_operation_in,
             InputParameters const &                                  param_in,
             std::shared_ptr<Function<dim>>                           velocity_in,
             unsigned int const                                       n_refine_time_in)
    : TimeIntBDFBase(param_in.start_time,
                     param_in.end_time,
                     param_in.max_number_of_time_steps,
                     param_in.order_time_integrator,
                     param_in.start_with_low_order,
                     false /* TODO */,
                     param_in.restart_data),
      conv_diff_operation(conv_diff_operation_in),
      param(param_in),
      velocity(velocity_in),
      n_refine_time(n_refine_time_in),
      cfl_number(param.cfl_number / std::pow(2.0, n_refine_time_in)),
      solution(param_in.order_time_integrator),
      vec_convective_term(param_in.order_time_integrator),
      N_iter_average(0.0),
      solver_time_average(0.0),
      cfl_oif(param.cfl_oif / std::pow(2.0, n_refine_time_in))
  {
  }

private:
  void
  allocate_vectors();

  void
  initialize_current_solution();

  void
  initialize_former_solutions();

  void
  initialize_vec_convective_term();

  void
  calculate_time_step_size();

  void
  prepare_vectors_for_next_timestep();

  void
  solve_timestep();

  void
  initialize_oif();

  void
  setup_derived();

  void
  initialize_solution_oif_substepping(unsigned int i);

  void
  update_sum_alphai_ui_oif_substepping(unsigned int i);

  void
  do_timestep_oif_substepping_and_update_vectors(double const start_time,
                                                 double const time_step_size);

  void
  output_solver_info_header() const;

  void
  output_remaining_time() const;

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia);

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  double
  recalculate_adaptive_time_step();

  void
  postprocessing() const;

  void
  analyze_computing_times() const;

  std::shared_ptr<DGOperation<dim, fe_degree, value_type>> conv_diff_operation;

  InputParameters const & param;

  std::shared_ptr<Function<dim>> velocity;

  unsigned int const n_refine_time;
  double const       cfl_number;

  // solution vectors
  VectorType              solution_np;
  std::vector<VectorType> solution;
  std::vector<VectorType> vec_convective_term;

  VectorType sum_alphai_ui;
  VectorType rhs_vector;

  // iteration counts and solver time
  double N_iter_average;
  double solver_time_average;

  // Operator-integration-factor (OIF) splitting

  // cfl number for OIF splitting
  double const cfl_oif;

  std::shared_ptr<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>>
    convective_operator_OIF;

  std::shared_ptr<
    ExplicitTimeIntegrator<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>, VectorType>>
    time_integrator_OIF;

  // solution vectors needed for OIF substepping of convective term
  VectorType solution_tilde_m;
  VectorType solution_tilde_mp;
};

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::setup_derived()
{
  // Initialize vec_convective_term: Note that this function has to be called
  // after the solution has been initialized because the solution is evaluated in this function.
  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit &&
     (param.equation_type == ConvDiff::EquationType::Convection ||
      param.equation_type == ConvDiff::EquationType::ConvectionDiffusion) &&
     start_with_low_order == false)
  {
    initialize_vec_convective_term();
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(
      new ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>(conv_diff_operation));

    if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK1Stage1)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(1, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK2Stage2)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(2, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK3Stage3)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(3, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage4)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(4, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK3Stage4Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK3Stage4Reg2C<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage5Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg2C<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage5Reg3C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg3C<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK5Stage9Reg2S)
    {
      time_integrator_OIF.reset(
        new LowStorageRK5Stage9Reg2S<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK3Stage7Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>, VectorType>(
          convective_operator_OIF, 3, 7));
    }
    else if(param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage8Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>, VectorType>(
          convective_operator_OIF, 4, 8));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::allocate_vectors()
{
  for(unsigned int i = 0; i < solution.size(); ++i)
    conv_diff_operation->initialize_dof_vector(solution[i]);

  conv_diff_operation->initialize_dof_vector(solution_np);

  conv_diff_operation->initialize_dof_vector(sum_alphai_ui);
  conv_diff_operation->initialize_dof_vector(rhs_vector);

  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit &&
     (param.equation_type == ConvDiff::EquationType::Convection ||
      param.equation_type == ConvDiff::EquationType::ConvectionDiffusion))
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      conv_diff_operation->initialize_dof_vector(vec_convective_term[i]);
  }

  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    conv_diff_operation->initialize_dof_vector(solution_tilde_m);
    conv_diff_operation->initialize_dof_vector(solution_tilde_mp);
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_current_solution()
{
  conv_diff_operation->prescribe_initial_conditions(solution[0], this->get_time());
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_former_solutions()
{
  // Start with i=1 since we only want to initialize the solution at former instants of time.
  for(unsigned int i = 1; i < solution.size(); ++i)
  {
    conv_diff_operation->prescribe_initial_conditions(solution[i], this->get_previous_time(i));
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    conv_diff_operation->evaluate_convective_term(vec_convective_term[i],
                                                  solution[i],
                                                  this->get_previous_time(i));
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::calculate_time_step_size()
{
  pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  if(param.calculation_of_time_step_size ==
     ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    double const time_step = calculate_const_time_step(param.time_step_size, n_refine_time);
    this->set_time_step_size(time_step);

    print_parameter(pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::Convection ||
                  param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepCFL does not make sense!"));

    // calculate minimum vertex distance
    double const global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout, "h_min", global_min_cell_diameter);

    double const max_velocity =
      calculate_max_velocity(conv_diff_operation->get_data().get_dof_handler().get_triangulation(),
                             velocity,
                             this->get_time());

    print_parameter(pcout, "U_max", max_velocity);
    print_parameter(pcout, "CFL", cfl_number);
    print_parameter(pcout, "Exponent fe_degree (convection)", param.exponent_fe_degree_convection);

    double time_step_conv = calculate_const_time_step_cfl(cfl_number,
                                                          max_velocity,
                                                          global_min_cell_diameter,
                                                          fe_degree,
                                                          param.exponent_fe_degree_convection);

    // decrease time_step in order to exactly hit end_time
    time_step_conv = (param.end_time - param.start_time) /
                     (1 + int((param.end_time - param.start_time) / time_step_conv));

    this->set_time_step_size(time_step_conv);

    print_parameter(pcout, "Time step size (convection)", time_step_conv);
  }
  else if(param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    // calculate minimum vertex distance
    double const global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    double time_step = calculate_time_step_max_efficiency(
      param.c_eff, global_min_cell_diameter, fe_degree, order, n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_step = (param.end_time - param.start_time) /
                (1 + int((param.end_time - param.start_time) / time_step));

    this->set_time_step_size(time_step);

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
    print_parameter(pcout, "Time step size", time_step);
  }
  else
  {
    AssertThrow(
      param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified ||
        param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL ||
        param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency,
      ExcMessage(
        "Specified calculation of time step size not implemented for BDF time integrator!"));
  }

  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // make sure that CFL condition is used for the calculation of the time step size (the aim
    // of the OIF splitting approach is to overcome limitations of the CFL condition)
    AssertThrow(
      param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL,
      ExcMessage(
        "Specified calculation of time step size not compatible with OIF splitting approach!"));

    pcout << std::endl << "OIF substepping for convective term:" << std::endl << std::endl;

    print_parameter(pcout, "CFL (OIF)", cfl_oif);
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::prepare_vectors_for_next_timestep()
{
  push_back(solution);

  solution[0].swap(solution_np);

  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit &&
     (param.equation_type == ConvDiff::EquationType::Convection ||
      param.equation_type == ConvDiff::EquationType::ConvectionDiffusion))
  {
    push_back(vec_convective_term);
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::output_solver_info_header() const
{
  // write output
  if(get_time_step_number() % param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << this->get_time_step_number()
          << "t_n = " << std::scientific << std::setprecision(4) << this->get_time()
          << " -> t_n+1 = " << this->get_next_time() << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl;
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::output_remaining_time() const
{
  // write output
  if(get_time_step_number() % param.output_solver_info_every_timesteps == 0)
  {
    if(this->get_time() > param.start_time)
    {
      double const remaining_time = global_timer.wall_time() * (param.end_time - this->get_time()) /
                                    (this->get_time() - param.start_time);
      pcout << std::endl
            << "Estimated time until completion is " << remaining_time << " s / "
            << remaining_time / 3600. << " h." << std::endl;
    }
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for(unsigned int i = 0; i < this->order; i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), solution[i].begin());
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorView<value_type> vector_view(solution[i].local_size(), solution[i].begin());
    oa << vector_view;
  }

#pragma GCC diagnostic pop
}

template<int dim, int fe_degree, typename value_type>
double
TimeIntBDF<dim, fe_degree, value_type>::recalculate_adaptive_time_step()
{
  // currently no adaptive time stepping implemented, do nothing
  return this->get_time_step_size();
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::solve_timestep()
{
  Timer timer;
  timer.restart();

  // calculate rhs (rhs-vector f and inhomogeneous boundary face integrals)
  conv_diff_operation->rhs(rhs_vector, this->get_next_time());

  // add the convective term to the right-hand side of the equations
  // if the convective term is treated explicitly (additive decomposition)
  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit)
  {
    // only if convective term is really involved
    if(param.equation_type == ConvDiff::EquationType::Convection ||
       param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      conv_diff_operation->evaluate_convective_term(vec_convective_term[0],
                                                    solution[0],
                                                    this->get_time());

      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        rhs_vector.add(-extra.get_beta(i), vec_convective_term[i]);
    }
  }

  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor splitting
  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    calculate_sum_alphai_ui_oif_substepping(cfl_number, cfl_oif);
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    sum_alphai_ui.equ(bdf.get_alpha(0) / this->get_time_step_size(), solution[0]);
    for(unsigned int i = 1; i < solution.size(); ++i)
      sum_alphai_ui.add(bdf.get_alpha(i) / this->get_time_step_size(), solution[i]);
  }

  // apply mass matrix to sum_alphai_ui and add to rhs vector
  conv_diff_operation->apply_mass_matrix_add(rhs_vector, sum_alphai_ui);


  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(extra.get_beta(0), solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    solution_np.add(extra.get_beta(i), solution[i]);

  // solve the linear system of equations
  unsigned int iterations = conv_diff_operation->solve(
    solution_np, rhs_vector, bdf.get_gamma0() / this->get_time_step_size(), this->get_next_time());

  N_iter_average += iterations;
  solver_time_average += timer.wall_time();

  // write output
  if(get_time_step_number() % param.output_solver_info_every_timesteps == 0)
  {
    pcout << "Solve scalar convection-diffusion problem:" << std::endl
          << "  Iterations: " << std::setw(6) << std::right << iterations
          << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_solution_oif_substepping(unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = solution[i];
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::update_sum_alphai_ui_oif_substepping(unsigned int i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
  else // i>0
    sum_alphai_ui.add(bdf.get_alpha(i) / this->get_time_step_size(), solution_tilde_m);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::do_timestep_oif_substepping_and_update_vectors(
  double const start_time,
  double const time_step_size)
{
  // solve sub-step
  time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                      solution_tilde_m,
                                      start_time,
                                      time_step_size);

  solution_tilde_mp.swap(solution_tilde_m);
}


template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::postprocessing() const
{
  conv_diff_operation->do_postprocessing(solution[0],
                                         this->get_time(),
                                         this->get_time_step_number());
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::analyze_computing_times() const
{
  pcout << std::endl
        << "Number of time steps = " << (get_time_step_number() - 1) << std::endl
        << "Average number of iterations = " << std::scientific << std::setprecision(3)
        << N_iter_average / (get_time_step_number() - 1) << std::endl
        << "Average wall time per time step = " << std::scientific << std::setprecision(3)
        << solver_time_average / (get_time_step_number() - 1) << std::endl;

  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max"
        << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(total_time, MPI_COMM_WORLD);
  pcout << "  Time loop:           " << std::scientific << std::setprecision(4) << std::setw(10)
        << data.min << " " << std::setprecision(4) << std::setw(10) << data.avg << " "
        << std::setprecision(4) << std::setw(10) << data.max << " "
        << "          "
        << "  " << std::setw(6) << std::left << data.min_index << " " << std::setw(6) << std::left
        << data.max_index << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
