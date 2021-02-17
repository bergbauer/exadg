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

#ifndef DG_CG_TRANSFER
#define DG_CG_TRANSFER

// deal.II
#include <deal.II/matrix_free/matrix_free.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer.h>

namespace ExaDG
{
using namespace dealii;

template<int dim,
         typename Number,
         typename VectorType = LinearAlgebra::distributed::Vector<Number>,
         int components      = 1>
class MGTransferC : virtual public MGTransfer<VectorType>
{
public:
  MGTransferC(const Mapping<dim> &              mapping,
              const MatrixFree<dim, Number> &   matrixfree_dg,
              const MatrixFree<dim, Number> &   matrixfree_cg,
              const AffineConstraints<Number> & constraints_dg,
              const AffineConstraints<Number> & constraints_cg,
              const unsigned int                level,
              const unsigned int                fe_degree,
              const unsigned int                dof_handler_index = 0);

  virtual ~MGTransferC();

  void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const;

  void
  restrict_and_add(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

  void
  prolongate(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

private:
  template<int degree>
  void
  do_interpolate(VectorType & dst, const VectorType & src) const;

  template<int degree>
  void
  do_restrict_and_add(VectorType & dst, const VectorType & src) const;

  template<int degree>
  void
  do_prolongate(VectorType & dst, const VectorType & src) const;

  const unsigned int      fe_degree;
  MatrixFree<dim, Number> data_composite;
};

} // namespace ExaDG

#endif
