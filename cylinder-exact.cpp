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
    const char *mesh_file = "mesh_cylinder.msh";
    const int order = 4;

    const double a = 0.1;      // cylinder radius
    const double eps_runup = 1e-3 * a;   // run-up evaluation offset

    const double H = 0.05;     // wave height
    const double inv_H2 = 2.0 / H;

    const double lambda = 0.5;
    const double k = 2.0 * M_PI / lambda;
    const double A = 0.025;     // should be H/2 for standard normalization

    const double Etol = 1e-6;
    const int MaxIter = 100;
    // -------------------------------------------------

    // Load mesh
    Mesh mesh(mesh_file, 1, 1);
    mesh.EnsureNodes();
    mesh.SetCurvature(order);

    const int dim = mesh.Dimension();

    // FE space
    H1_FECollection fec(order, dim);
    FiniteElementSpace fes(&mesh, &fec);

    // Cylinder center from bounding box (matches your earlier logic)
    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);
    const double cx = 0.5 * (bbmin(0) + bbmax(0));
    const double cy = 0.5 * (bbmin(1) + bbmax(1));

    // -------- McCamy–Fuchs envelope coefficient --------
    FunctionCoefficient eta_env_exact([&](const Vector &X)
    {
        const double xc = X(0) - cx;
        const double yc = X(1) - cy;

        double r = sqrt(xc*xc + yc*yc);

        // // // Evaluate slightly outside the cylinder wall
        // if (r < a + eps_runup)
        // {
        //     r = a + eps_runup;
        // }
        //if (r < a) { return 0.0; }
        r = std::max(r, a + eps_runup);



        const double phi =
            (yc >= 0.0) ? acos(xc/r) : (-acos(xc/r) + 2.0*M_PI);

        const double ka = k * a;
        const double kr = k * r;

        // m = 0
        const double J0P = -boost::math::cyl_bessel_j(1, ka);
        complex<double> H0P(boost::math::cyl_bessel_j(1, ka),
                            boost::math::cyl_neumann(1, ka));

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

        // Envelope (max over time), normalized by (H/2)
        return (A * abs(E)) * inv_H2;
    });

    // Project envelope to grid function
    GridFunction eta_env(&fes);
    eta_env.ProjectCoefficient(eta_env_exact);

    // -------- GLVis --------
    socketstream sol_sock("localhost", 19916);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << eta_env << flush;

    cout << "Envelope field sent to GLVis.\n";
    cout << "Keys: m (mesh), s (smooth), l (lighting), r (reset)\n";

    return 0;
}
