#include "mfem.hpp"
using namespace mfem;

int main()
{
   // Box size and grid
   const double Lx = 2.0, Ly = 1.0, H = 1.0;
   const int nx = 4, ny = 2, nz = 2;

   // Cartesian HEX mesh
   Mesh base = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, Lx, Ly, H);

   // x-direction periodic
   Vector tx({Lx, 0.0, 0.0});
   Mesh mesh = Mesh::MakePeriodic(base, base.CreatePeriodicVertexMapping({tx}));

   // mark boundary attributes: bottom=1, top=2, ymin=3, ymax=4
   Vector bbmin, bbmax; 
   mesh.GetBoundingBox(bbmin, bbmax);
   const double zmin = bbmin(2), zmax = bbmax(2);
   const double ymin = bbmin(1), ymax = bbmax(1);
   const double tol = 1e-12 * (zmax - zmin + ymax - ymin + Lx);

   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      ElementTransformation *Tb = mesh.GetBdrElementTransformation(be);
      IntegrationPoint ip; 
      ip.Set2(0.5, 0.5); // face center
      Vector x(3); 
      Tb->Transform(ip, x);

      int attr = 0;
      if (fabs(x(2) - zmin) < tol) attr = 1;        // bottom
      else if (fabs(x(2) - zmax) < tol) attr = 2;   // top (free surface)
      else if (fabs(x(1) - ymin) < tol) attr = 3;   // y = min side
      else if (fabs(x(1) - ymax) < tol) attr = 4;   // y = max side
      // x-faces are gone here (periodic), so no attributes for them.

      // Set attributes to boundary elements
      if (attr) { mesh.GetBdrElement(be)->SetAttribute(attr); }
   }

   mesh.Save("/Users/hirsch/mfem/mfem-4.8/thesis/wave-tank.mesh");
}
