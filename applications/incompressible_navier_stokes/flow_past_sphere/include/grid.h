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

#ifndef APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_SPHERE_H_
#define APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_SPHERE_H_

#include <deal.II/grid/grid_out.h>

namespace ExaDG
{
namespace FlowPastSphere
{
double const       radius        = 0.05;
double const       radius_next   = 2. * radius;
double const       outer         = 4. * radius;
unsigned int const length_factor = 5;

template<int dim, int spacedim = dim>
class SphericalManifoldBoundaryLayer : public dealii::Manifold<dim, spacedim>
{
public:
  SphericalManifoldBoundaryLayer(const dealii::Point<dim> & center,
                                 double const               radius,
                                 double const               radius_next)
    : spherical(dealii::Point<spacedim>()),
      center(center),
      radius(radius),
      radius_next(radius_next),
      stretch_factor(0.4)
  {
  }

  std::unique_ptr<dealii::Manifold<dim, spacedim>>
  clone() const override
  {
    return std::make_unique<SphericalManifoldBoundaryLayer<dim, spacedim>>(center,
                                                                           radius,
                                                                           radius_next);
  }

  dealii::Point<spacedim>
  get_intermediate_point(const dealii::Point<spacedim> & p1,
                         const dealii::Point<spacedim> & p2,
                         const double                    w) const override
  {
    return push_forward(spherical.get_intermediate_point(pull_back(p1), pull_back(p2), w));
  }

  dealii::Point<spacedim>
  get_new_point(const dealii::ArrayView<const dealii::Point<spacedim>> & vertices,
                const dealii::ArrayView<const double> &                  weights) const override
  {
    boost::container::small_vector<dealii::Point<spacedim>, 100> pulled_back_vertices;
    for(dealii::Point<spacedim> const & vertex : vertices)
      pulled_back_vertices.push_back(pull_back(vertex));
    return push_forward(spherical.get_new_point(dealii::ArrayView<const dealii::Point<spacedim>>(
                                                  pulled_back_vertices.data(), vertices.size()),
                                                weights));
  }

  void
  get_new_points(const dealii::ArrayView<const dealii::Point<spacedim>> & surrounding_points,
                 const dealii::Table<2, double> &                         weights,
                 dealii::ArrayView<dealii::Point<spacedim>>               new_points) const override
  {
    boost::container::small_vector<dealii::Point<spacedim>, 100> pulled_back_vertices;
    for(dealii::Point<spacedim> const & vertex : surrounding_points)
      pulled_back_vertices.push_back(pull_back(vertex));
    spherical.get_new_points(dealii::ArrayView<const dealii::Point<spacedim>>(
                               pulled_back_vertices.data(), surrounding_points.size()),
                             weights,
                             new_points);
    for(dealii::Point<spacedim> & point : new_points)
      point = push_forward(point);
  }

private:
  dealii::Point<spacedim>
  push_forward(dealii::Point<spacedim> const & p) const
  {
    // apply transformation function
    double const R = p.norm();
    double const r = radius + (R - stretch_factor * radius) /
                                (radius_next - stretch_factor * radius) * (R - radius);
    return p * (r / R) + center;
  }

  dealii::Point<spacedim>
  pull_back(dealii::Point<spacedim> const & p) const
  {
    // inverse of the function above, using the positive root
    double const r = (p - center).norm();
    double const c =
      stretch_factor * radius * radius - (r - radius) * (radius_next - stretch_factor * radius);
    double const b = (1 + stretch_factor) * radius;
    double const R = 0.5 * b + std::sqrt(0.25 * b * b - c);
    return dealii::Point<dim>(p - center) * (R / r);
  }

  dealii::SphericalManifold<spacedim> const spherical;
  dealii::Point<dim>                        center;
  double const                              radius;
  double const                              radius_next;
  double const                              stretch_factor;
};



template<int dim>
void
create_sphere_grid(dealii::Triangulation<dim> & tria,
                   unsigned int const           n_refinements,
                   TriangulationType const &    triangulation_type)
{
  AssertThrow(
    triangulation_type != TriangulationType::FullyDistributed,
    dealii::ExcMessage(
      "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
      "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

  dealii::Point<dim> center_sphere;
  center_sphere[0] = 0.5;
  center_sphere[1] = 0.2;
  center_sphere[2] = 0.2;

  // Create inner mesh
  dealii::Triangulation<dim> tria1, tria2, tria3, tria4, tria_ser;
  // hyper shell for high order boundary
  dealii::GridGenerator::hyper_shell(tria1, center_sphere, radius, radius_next, 6);
  // hyper shell for transition sphere to cube
  dealii::GridGenerator::hyper_shell(tria2, center_sphere, radius_next, std::sqrt(dim) * outer, 6);

  dealii::Point<dim> lower_left, upper_right;
  // wake behind sphere
  lower_left[0] = center_sphere[0] + outer;
  for(unsigned int d = 1; d < dim; ++d)
    lower_left[d] = center_sphere[d] - outer;
  upper_right[0] = center_sphere[0] + (1 + 2 * length_factor) * outer;
  for(unsigned int d = 1; d < dim; ++d)
    upper_right[d] = center_sphere[d] + outer;
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria3, {length_factor, 1, 1}, lower_left, upper_right, false);
  // inflow before sphere
  lower_left[0]  = center_sphere[0] - 3 * outer;
  upper_right[0] = center_sphere[0] - outer;
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria4, {1, 1, 1}, lower_left, upper_right, false);

  dealii::GridGenerator::merge_triangulations({&tria1, &tria2, &tria3, &tria4}, tria_ser);

  // Set cells near sphere to spherical manifold for first round of refinement
  for(auto const & cell : tria_ser.active_cell_iterators())
  {
    if(cell->index() < 6)
      cell->set_all_manifold_ids(1);
  }
  const SphericalManifoldBoundaryLayer<dim> spherical_manifold(center_sphere, radius, radius_next);
  tria_ser.set_manifold(1, spherical_manifold);

  tria_ser.refine_global(1);

  dealii::GridOut           grid_out;
  std::ofstream             stream("grid_inner.vtk");
  dealii::GridOutFlags::Vtk flags;
  flags.write_higher_order_cells = true;
  grid_out.set_flags(flags);
  grid_out.write_vtk(tria_ser, stream);

  // Shift two points somewhat to generate a better balance in points
  //  for(auto const & cell : tria_ser.active_cell_iterators())
  //  {
  //    if(std::abs(cell->vertex(0)[0] - outer) < 1e-10 and std::abs(cell->vertex(0)[1]) < 1e-10 and
  //       std::abs(cell->vertex(0)[2]) < 1e-10)
  //      cell->vertex(0)[0] += 0.3 * (outer - radius_next);
  //    else if(std::abs(cell->vertex(0)[0] - 0.5 * (outer + radius_next)) < 1e-10 and
  //            std::abs(cell->vertex(0)[1]) < 1e-10 and std::abs(cell->vertex(0)[2]) < 1e-10)
  //      cell->vertex(0)[0] += 0.15 * (outer - radius_next);
  //    else if(std::abs(cell->vertex(1)[0] + outer) < 1e-10 and
  //            std::abs(cell->vertex(1)[1]) < 1e-10 and std::abs(cell->vertex(1)[2]) < 1e-10)
  //      cell->vertex(1)[0] -= 0.3 * (outer - radius_next);
  //    else if(std::abs(cell->vertex(1)[0] + 0.5 * (outer + radius_next)) < 1e-10 and
  //            std::abs(cell->vertex(1)[1]) < 1e-10 and std::abs(cell->vertex(1)[2]) < 1e-10)
  //      cell->vertex(1)[0] -= 0.15 * (outer - radius_next);
  //  }

  // Remove refinement to make sure all MPI ranks agree on this mesh
  dealii::Triangulation<dim> tria_inner;
  dealii::GridGenerator::flatten_triangulation(tria_ser, tria_inner);

  // Create outer layer of cube cells
  for(unsigned int d = 1; d < dim; ++d)
    lower_left[d] = center_sphere[d] - 2. * outer;
  for(unsigned int d = 1; d < dim; ++d)
    upper_right[d] = center_sphere[d] + 2. * outer;
  upper_right[0] = center_sphere[0] + (1 + 2 * length_factor) * outer;
  dealii::Triangulation<dim> tria_rectangle;
  std::vector<unsigned int>  refinements(dim, 4);
  refinements[0] = 2 * length_factor + 4;
  dealii::GridGenerator::subdivided_hyper_rectangle(tria_rectangle,
                                                    refinements,
                                                    lower_left,
                                                    upper_right);

  // Remove cells present in inner mesh
  std::set<typename dealii::Triangulation<dim>::active_cell_iterator> cells_to_remove;
  for(auto const & cell : tria_rectangle.active_cell_iterators())
    if(outer - std::abs(cell->center()[1] - center_sphere[1]) > 0. and
       (dim < 3 or (outer - std::abs(cell->center()[2] - center_sphere[2]) > 0.)))
      cells_to_remove.insert(cell);

  dealii::Triangulation<dim> tria_outer;
  dealii::GridGenerator::create_triangulation_with_removed_cells(tria_rectangle,
                                                                 cells_to_remove,
                                                                 tria_outer);

  dealii::GridGenerator::merge_triangulations({&tria_inner, &tria_outer}, tria);

  // Set manifold ids again on the final triangulation
  tria.reset_all_manifolds();
  for(auto const & cell : tria.active_cell_iterators())
  {
    if(cell->index() < 6 * 8)
    {
      cell->set_all_manifold_ids(1);
      for(unsigned int v : cell->vertex_indices())
        AssertThrow((cell->vertex(v) - center_sphere).norm() < radius_next + 1e-10,
                    dealii::ExcInternalError());
    }
  }
  tria.set_manifold(1, spherical_manifold);

  //  dealii::GridOut grid_out;
  std::ofstream stream2("grid.vtk");
  //  dealii::GridOutFlags::Vtk flags;
  flags.write_higher_order_cells = true;
  grid_out.set_flags(flags);
  grid_out.write_vtk(tria, stream2);

  // Set boundary ids
  for(auto const & cell : tria.cell_iterators())
    for(unsigned int f = 0; f < cell->n_faces(); ++f)
      if(cell->at_boundary(f))
      {
        // sphere -> id 3
        if(std::abs((cell->face(f)->vertex(0) - center_sphere).norm() - radius) < 1e-10)
          cell->face(f)->set_boundary_id(3);
        // inflow -> id 1
        else if(std::abs((cell->face(f)->center()[0] - center_sphere[0]) + 3. * outer) < 1e-10)
          cell->face(f)->set_boundary_id(1);
        // symmetry -> id 0
        else if(2. * outer - std::abs((cell->face(f)->center()[1] - center_sphere[1])) < 1e-10 or
                (dim == 3 and
                 (2. * outer - std::abs(cell->face(f)->center()[2] - center_sphere[2]) < 1e-10)))
          cell->face(f)->set_boundary_id(0);
        // outflow -> id 2
        else
        {
          AssertThrow(std::abs(cell->face(f)->center()[0] - center_sphere[0] -
                               (1 + 2 * length_factor) * outer) < 1e-10,
                      dealii::ExcInternalError());
          cell->face(f)->set_boundary_id(2);
        }
      }

  if(n_refinements > 0)
  {
    // In the first round, only refine cells in the inner part of the duct as
    // the outer cells carry little information
    for(auto const & cell : tria.active_cell_iterators())
      if(cell->is_locally_owned())
      {
        if(outer - std::abs(cell->center()[1] - center_sphere[1]) > 0. and
           (dim < 3 or (outer - std::abs(cell->center()[2] - center_sphere[2]) > 0.)))
          cell->set_refine_flag();
      }
    tria.execute_coarsening_and_refinement();

    tria.refine_global(n_refinements - 1);
  }

  // Refine mesh adaptively once again in region around sphere and in the
  // immediate wake
  //  for(auto const & cell : tria.active_cell_iterators())
  //    if(cell->is_locally_owned())
  //    {
  //      dealii::Point<dim> center = cell->center();
  //      if(center[0] > 0 and center[0] < 4.5 * outer)
  //      {
  //        // Check radius of 2.5 * radius for (y,z) coordinates
  //        const double radius_factor = center[0] < 0 ? 2.3 : 2.8;
  //        center[0]                  = 0;
  //        if(center.norm() < radius_factor * radius)
  //          cell->set_refine_flag();
  //      }
  //      else if(center.norm() < radius_next)
  //        cell->set_refine_flag();
  //    }
  //  tria.execute_coarsening_and_refinement();


  // dealii::GridTools::shift(center, tria);
}


} // namespace FlowPastSphere
} // namespace ExaDG

#endif /* APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_SPHERE_H_ */
