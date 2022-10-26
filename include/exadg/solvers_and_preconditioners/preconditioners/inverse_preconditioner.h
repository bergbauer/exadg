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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEPRECONDITIONER_H_

#include <deal.II/lac/sparse_matrix.h>

#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
template<typename Operator>
class InversePreconditioner : public PreconditionerBase<typename Operator::value_type>
{
public:
  typedef typename PreconditionerBase<typename Operator::value_type>::VectorType VectorType;
  typedef typename PreconditionerBase<dealii::TrilinosScalar>::VectorType        VectorTypeDouble;

  // distributed sparse system matrix
  dealii::TrilinosWrappers::SparseMatrix system_matrix;

  InversePreconditioner(Operator const & underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {
    underlying_operator.init_system_matrix(system_matrix, underlying_operator.get_matrix_free().get_dof_handler().get_communicator());
    underlying_operator.calculate_system_matrix(system_matrix);
  }

  void
  update()
  {
    underlying_operator.calculate_system_matrix(system_matrix);
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    VectorTypeDouble dst_double;
    dst_double.reinit(dst, false);
    VectorTypeDouble src_double;
    src_double.reinit(src, true);
    src_double = src;

    dealii::FullMatrix<double> full_matrix(system_matrix.m());

    full_matrix.copy_from(system_matrix);
    full_matrix.invert(full_matrix);

    dealii::TrilinosWrappers::SparseMatrix inverse_matrix(system_matrix.m(), system_matrix.m(), system_matrix.m());

    for(unsigned int i = 0; i < system_matrix.m(); ++i)
      for(unsigned int j = 0; j < system_matrix.m(); ++j)
        inverse_matrix.set(i, j, full_matrix(i, j));

    inverse_matrix.compress(dealii::VectorOperation::insert);

    inverse_matrix.vmult(dst_double, src_double);

    // convert: double -> Number
    dst.copy_locally_owned_data_from(dst_double);
  }

private:
  Operator const & underlying_operator;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_INVERSEPRECONDITIONER_H_ */
