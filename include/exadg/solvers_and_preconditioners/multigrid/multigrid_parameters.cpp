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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_parameters.h>

namespace ExaDG
{
std::string
enum_to_string(MultigridType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridType::Undefined:
      string_type = "Undefined";
      break;
    case MultigridType::hMG:
      string_type = "h-MG";
      break;
    case MultigridType::chMG:
      string_type = "ch-MG";
      break;
    case MultigridType::hcMG:
      string_type = "hc-MG";
      break;
    case MultigridType::cMG:
      string_type = "c-MG";
      break;
    case MultigridType::pMG:
      string_type = "p-MG";
      break;
    case MultigridType::cpMG:
      string_type = "cp-MG";
      break;
    case MultigridType::pcMG:
      string_type = "pc-MG";
      break;
    case MultigridType::hpMG:
      string_type = "hp-MG";
      break;
    case MultigridType::chpMG:
      string_type = "chp-MG";
      break;
    case MultigridType::hcpMG:
      string_type = "hcp-MG";
      break;
    case MultigridType::hpcMG:
      string_type = "hpc-MG";
      break;
    case MultigridType::phMG:
      string_type = "ph-MG";
      break;
    case MultigridType::cphMG:
      string_type = "cph-MG";
      break;
    case MultigridType::pchMG:
      string_type = "pch-MG";
      break;
    case MultigridType::phcMG:
      string_type = "phc-MG";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PSequenceType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PSequenceType::GoToOne:
      string_type = "GoToOne";
      break;
    case PSequenceType::DecreaseByOne:
      string_type = "DecreaseByOne";
      break;
    case PSequenceType::Bisect:
      string_type = "Bisect";
      break;
    case PSequenceType::Manual:
      string_type = "Manual";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MultigridSmoother const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridSmoother::Chebyshev:
      string_type = "Chebyshev";
      break;
    case MultigridSmoother::GMRES:
      string_type = "GMRES";
      break;
    case MultigridSmoother::CG:
      string_type = "CG";
      break;
    case MultigridSmoother::Jacobi:
      string_type = "Jacobi";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(AMGType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case AMGType::ML:
      string_type = "ML";
      break;
    case AMGType::BoomerAMG:
      string_type = "BoomerAMG";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(MultigridCoarseGridSolver const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridCoarseGridSolver::Chebyshev:
      string_type = "Chebyshev";
      break;
    case MultigridCoarseGridSolver::CG:
      string_type = "CG";
      break;
    case MultigridCoarseGridSolver::GMRES:
      string_type = "GMRES";
      break;
    case MultigridCoarseGridSolver::AMG:
      string_type = "AMG";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}


std::string
enum_to_string(MultigridCoarseGridPreconditioner const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case MultigridCoarseGridPreconditioner::None:
      string_type = "None";
      break;
    case MultigridCoarseGridPreconditioner::PointJacobi:
      string_type = "PointJacobi";
      break;
    case MultigridCoarseGridPreconditioner::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case MultigridCoarseGridPreconditioner::AMG:
      string_type = "AMG";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

std::string
enum_to_string(PreconditionerSmoother const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    case PreconditionerSmoother::None:
      string_type = "None";
      break;
    case PreconditionerSmoother::PointJacobi:
      string_type = "PointJacobi";
      break;
    case PreconditionerSmoother::BlockJacobi:
      string_type = "BlockJacobi";
      break;
    case PreconditionerSmoother::AdditiveSchwarz:
      string_type = "AdditiveSchwarz";
      break;
    default:
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
      break;
  }

  return string_type;
}

bool
MultigridData::involves_h_transfer() const
{
  if(type != MultigridType::pMG && type != MultigridType::cpMG && type != MultigridType::pcMG)
    return true;
  else
    return false;
}

bool
MultigridData::involves_c_transfer() const
{
  if(type == MultigridType::hMG || type == MultigridType::pMG || type == MultigridType::hpMG ||
     type == MultigridType::phMG)
    return false;
  else
    return true;
}

bool
MultigridData::involves_p_transfer() const
{
  if(type != MultigridType::hMG && type != MultigridType::hcMG && type != MultigridType::chMG)
    return true;
  else
    return false;
}

} // namespace ExaDG
