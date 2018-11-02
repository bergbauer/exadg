#include "mass_operator.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
void
MassMatrixOperator<dim, degree, Number>::initialize(
  MatrixFree<dim, Number> const &     mf_data,
  MassMatrixOperatorData<dim> const & mass_matrix_operator_data,
  unsigned int                        level_mg_handler)
{
  ConstraintMatrix constraint_matrix;
  Parent::reinit(mf_data, constraint_matrix, mass_matrix_operator_data, level_mg_handler);
}

template<int dim, int degree, typename Number>
void
MassMatrixOperator<dim, degree, Number>::initialize(
  MatrixFree<dim, Number> const &     mf_data,
  ConstraintMatrix const &            constraint_matrix,
  MassMatrixOperatorData<dim> const & mass_matrix_operator_data,
  unsigned int                        level_mg_handler)
{
  Parent::reinit(mf_data, constraint_matrix, mass_matrix_operator_data, level_mg_handler);
}

template<int dim, int degree, typename Number>
void
MassMatrixOperator<dim, degree, Number>::do_cell_integral(FEEvalCell & fe_eval) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_value(fe_eval.get_value(q), q);
}
} // namespace ConvDiff

#include "mass_operator.hpp"
