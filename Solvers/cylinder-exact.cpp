#include "mfem.hpp"
#include <cmath>
#include <complex>
#include <iostream>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/hankel.hpp>

using namespace mfem;
using namespace std;

int main(int argc, char *argv[])
{
    // ---------------- user parameters ----------------
    const char *mesh_file = "../Meshes/cylinder_mesh_exact.msh";
    const int order = 4;

    const double a = 0.5;      // cylinder radius
    const double eps_runup = 1e-3 * a;   // run-up evaluation offset

    const double H = 0.01;     // wave height
    const double inv_H2 = 2.0 / H;

    const double lambda = 1.0;
    const double k = 2.0 * M_PI;
    const double A = 0.005;     // should be H/2 for standard normalization

    const double Etol = 1e-10;
    const int MaxIter = 400;
    // -------------------------------------------------

    // Load mesh
    Mesh mesh(mesh_file, 1, 1);
    mesh.EnsureNodes();
    mesh.SetCurvature(order);

    const int dim = mesh.Dimension();

    // FE space
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes(&mesh, &fec);

    // Cylinder center from bounding box
    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);
    const double cx = 2.0;
    const double cy = 1.5;
    // const double cx = 0.5 * (bbmin(0) + bbmax(0));
    // const double cy = 0.5 * (bbmin(1) + bbmax(1));


    // -------- McCamy–Fuchs envelope coefficient --------
    FunctionCoefficient eta_env_exact([&](const Vector &X)
    {
        const double xc = X(0) - cx;
        const double yc = X(1) - cy;

        double r = sqrt(xc*xc + yc*yc);
        //r = std::max(r, a + eps_runup);

        const double phi =
            (yc >= 0.0) ? acos(xc/r) : (-acos(xc/r) + 2.0*M_PI);

        const double ka = k * a;
        const double kr = k * r;

        // m = 0
        const double J0P = -boost::math::cyl_bessel_j(1, ka);
        complex<double> H0P(-boost::math::cyl_bessel_j(1, ka),
                            -boost::math::cyl_neumann(1, ka));

        complex<double> H0r(boost::math::cyl_bessel_j(0, kr),
                            boost::math::cyl_neumann(0, kr));

        complex<double> E =
            boost::math::cyl_bessel_j(0, kr) - H0r * (J0P / H0P);

        double oldterm = 0.0;
        for (int m = 1; m <= MaxIter; m++)
        {
            const double JmP = 0.5 *
                (boost::math::cyl_bessel_j(m-1, ka) -
                 boost::math::cyl_bessel_j(m+1, ka));

            complex<double> HmP(
                0.5 * (boost::math::cyl_bessel_j(m-1, ka) -
                       boost::math::cyl_bessel_j(m+1, ka)),
                0.5 * (boost::math::cyl_neumann(m-1, ka) -
                       boost::math::cyl_neumann(m+1, ka)));
            

            if (std::abs(HmP) < 1e-14) continue;


            const double Jmr = boost::math::cyl_bessel_j(m, kr);
            complex<double> Hmr(boost::math::cyl_bessel_j(m, kr),
                                boost::math::cyl_neumann(m, kr));

            const double phase = m * M_PI / 2.0;
            const complex<double> im(cos(phase), sin(phase));

            complex<double> term =
                2.0 * im * (Jmr - Hmr * (JmP / HmP)) * cos(m * phi);

            const double nextterm = term.real();
            if (std::isnan(nextterm)) { break; }

            E += term;

            if (abs(nextterm) < Etol && abs(oldterm) < Etol) { break; }
            oldterm = nextterm;
        }

        return (A * abs(E)) * inv_H2;
    });

    // ======== Extract cylinder boundary on free surface ========
    Mesh mesh2(mesh_file, 1, 1);
    mesh2.EnsureNodes();
    mesh2.SetCurvature(order);
    int dim2 = mesh2.Dimension();
    FiniteElementCollection *fec2 = new H1_FECollection(order, dim2);
    FiniteElementSpace fespace2(&mesh2, fec2);
    GridFunction phi(&fespace2);
    phi = 0.0;
    
    Array<int> bdr_attr;
    bdr_attr.Append(2);
    SubMesh mesh2_fs = SubMesh::CreateFromBoundary(mesh2, bdr_attr);
    int dim2_fs = mesh2_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec2_fs = new H1_FECollection(order, dim2_fs);
    FiniteElementSpace fespace2_fs(&mesh2_fs, fec2_fs);
    GridFunction eta(&fespace2_fs);
    eta.ProjectCoefficient(eta_env_exact);
    mesh2_fs.Transfer(eta, phi);

    Array<int> bdr_attr2;
    bdr_attr2.Append(3);
    SubMesh mesh2_cyl = SubMesh::CreateFromBoundary(mesh2, bdr_attr2);
    int dim_cyl = mesh2_cyl.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec2_cyl = new H1_FECollection(order, dim_cyl);
    FiniteElementSpace fespace2_cyl(&mesh2_cyl, fec2_cyl);
    GridFunction cyl(&fespace2_cyl);
    mesh2_cyl.Transfer(phi, cyl);

    // -------- GLVis --------
    socketstream sol_sock("localhost", 19916);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh2_cyl << cyl << flush;

    cout << "Envelope field sent to GLVis.\n";    



const double tol = 5e-3;

std::vector<std::pair<double,double>> cyl_eta;
Array<int> vdofs;

for (int v = 0; v < mesh2_cyl.GetNV(); v++)
{
    const double *X = mesh2_cyl.GetVertex(v);

    const double dx = X[0] - cx;
    const double dy = X[1] - cy;
    const double r  = std::sqrt(dx*dx + dy*dy);

    if (abs(r - a) > tol) continue;

    double theta = std::atan2(dy, dx);
    if (theta < 0.0) theta += 2.0*M_PI;

    fespace2_cyl.GetVertexDofs(v, vdofs);
    if (vdofs.Size() == 0) continue;

    const double val = cyl(vdofs[0]);
    if (val == 0.0) continue;   // <-- filter out zero values

    cyl_eta.emplace_back(theta, cyl(vdofs[0]));
}

// sort by angle
std::sort(cyl_eta.begin(), cyl_eta.end(),
          [](const auto &a, const auto &b)
          { return a.first < b.first; });



// write to file
std::ofstream fout("data/cylinder_boundary.txt");
fout << "# theta(rad)  eta\n";
for (auto &p : cyl_eta)
{
    fout << p.first << " " << p.second << "\n";
}
fout.close();

std::cout << "Extracted " << cyl_eta.size()
          << " points on cylinder rim\n";


              // Project envelope to grid function
    GridFunction eta_env(&fespace2_fs);
    eta_env.ProjectCoefficient(eta_env_exact);

    // // -------- GLVis --------
    // socketstream sol_sock("localhost", 19916);
    // sol_sock.precision(8);
    // sol_sock << "solution\n" << mesh << eta_env << flush;

    // cout << "Envelope field sent to GLVis.\n";


ParaViewDataCollection pvdc("cylinder_exact", &mesh2_fs);
pvdc.SetPrefixPath("./ParaView");
pvdc.SetDataFormat(VTKFormat::BINARY);
pvdc.SetHighOrderOutput(false);
pvdc.SetLevelsOfDetail(1);
pvdc.RegisterField("eta_env", &eta_env);
pvdc.SetCycle(0);
pvdc.SetTime(0.0);
pvdc.Save();


    return 0;
}




//     // Project envelope to grid function
//     GridFunction eta_env(&fes);
//     eta_env.ProjectCoefficient(eta_env_exact);

//     // ===== Create FREE SURFACE mesh (top boundary, attr=2) =====
//     Array<int> fs_attr(1);
//     fs_attr[0] = 2;
//     Mesh mesh_fs = SubMesh::CreateFromBoundary(mesh, fs_attr);

//     // FE space on free surface
//     H1_FECollection fec_fs(order, mesh_fs.Dimension());
//     FiniteElementSpace fes_fs(&mesh_fs, &fec_fs);
    
//     // Transfer eta_env to free surface mesh
//     GridFunction eta_env_fs(&fes_fs);
//     eta_env_fs.ProjectCoefficient(eta_env_exact);