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

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_ADDITIVESCHWARZPRECONDITIONER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_ADDITIVESCHWARZPRECONDITIONER_H_

#include <deal.II/lac/sparse_matrix.h>

#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
template<typename Operator>
class AdditiveSchwarzPreconditioner : public PreconditionerBase<typename Operator::value_type>
{
public:
  typedef typename PreconditionerBase<typename Operator::value_type>::VectorType VectorType;
  typedef typename PreconditionerBase<dealii::TrilinosScalar>::VectorType        VectorTypeDouble;

  // distributed sparse system matrix
  dealii::TrilinosWrappers::SparseMatrix system_matrix;

  AdditiveSchwarzPreconditioner(Operator const & underlying_operator_in)
    : underlying_operator(underlying_operator_in)
  {
    underlying_operator.assemble_as_matrix(system_matrix);
  }

  /*
   *  This function updates the block Jacobi preconditioner.
   *  Make sure that the underlying operator has been updated
   *  when calling this function.
   */
  void
  update()
  {
    underlying_operator.assemble_as_matrix(system_matrix);
  }

  /*
   *  This function applies the block Jacobi preconditioner.
   *  Make sure that the block Jacobi preconditioner has been
   *  updated when calling this function.
   */
  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    dst.reinit(src, false);
    underlying_operator.apply_inverse_as_blocks(dst, src);

    const bool check_vmult = false;

    if(check_vmult)
    {
      VectorType dst2;
      dst2.reinit(dst, false);

      VectorTypeDouble dst_double;
      dst_double.reinit(dst, false);
      VectorTypeDouble src_double;
      src_double.reinit(src, true);
      src_double = src;

      system_matrix.vmult(dst_double, src_double);

      // convert: double -> Number
      dst2.copy_locally_owned_data_from(dst_double);

      std::cout << "dst l2:  " << dst.l2_norm() << std::endl;
      std::cout << "dst2 l2: " << dst2.l2_norm() << std::endl;

      dst2.add(-1., dst);

      std::cout << "difference between dst and dst2 is: " << dst2.l2_norm() << std::endl;
    }
  }

private:
  Operator const & underlying_operator;
};

} // namespace ExaDG


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ADDITIVESCHWARZPRECONDITIONER_H_ */
