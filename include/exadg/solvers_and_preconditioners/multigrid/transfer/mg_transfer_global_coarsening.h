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

#ifndef EXADG_MG_TRANSFER_GLOBAL_COARSENING_H
#define EXADG_MG_TRANSFER_GLOBAL_COARSENING_H

// deal.II
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/levels_hybrid_multigrid.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename VectorType>
class MGTransferGlobalCoarsening : virtual public MGTransfer<VectorType>
{
public:
  virtual ~MGTransferGlobalCoarsening()
  {
  }

  void
  reinit(const Mapping<dim> &                                        mapping,
         MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> &   mg_matrixfree,
         MGLevelObject<std::shared_ptr<AffineConstraints<Number>>> & mg_constraints,
         MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &         mg_constrained_dofs,
         const unsigned int                                          dof_handler_index = 0);

  void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const;

  void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const;

  void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const;

private:
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;

  std::unique_ptr<dealii::MGTransferGlobalCoarsening<dim, VectorType>>
    mg_transfer_global_coarsening;
};
} // namespace ExaDG

#endif // EXADG_MG_TRANSFER_GLOBAL_COARSENING_H
