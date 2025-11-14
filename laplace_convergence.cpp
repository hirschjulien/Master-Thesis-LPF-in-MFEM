#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector> 

using namespace std;
using namespace mfem;

struct Dispersion {double kh; double omega;};
Dispersion SolveDispersion(const double g, const double h, const double T, int n){
    double omega = 2*M_PI / T;
    double kh = omega*omega * h / g;  //initial kh
    for (int i = 0; i < n; i++){
        kh = sqrt(kh * omega*omega * h/g * cosh(kh)/sinh(kh));
    };
    return {kh, omega};
};

int main (){

    int order = 2;
    const int max_ref_levels = 4;

    const char *mesh_file = "wave-tank.mesh";

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();


    //================= Initialize parameters =================
    // Wave in x direction
    const double H = 0.05;   // wave height
    const double ph = 0.0;   // phase
    const double g = 9.81;
    
    // --- Compute wavelength and depth from the mesh ---
    Vector bbmin, bbmax; 
    mesh.GetBoundingBox(bbmin, bbmax);
    const double Lx = bbmax(0) - bbmin(0);
    const double Ly = bbmax(1) - bbmin(1);
    const double h  = bbmax(2) - bbmin(2);   // since top is 0, bottom is -h
    const double zmax = bbmax(2);


    const double m = 2;     // number of wave periods in domain
    const double k  = m * 2.0*M_PI / Lx;         
    const double kh = k * h;    // by definition
    const double omega = k * sqrt((g/k) * tanh(kh));
    //const double T = 2.0 * M_PI / omega;     // find T that matches the mesh, one whole has to fit into the x direction
    const double cs = sqrt((g/k) * tanh(kh));
    cout << (cs - omega/k) << "\n";     // check dispersion relation



    // --- Boundary phi at the free surface (Dirichlet)
    FunctionCoefficient phi_fs_init([&](const Vector& x){
    return -0.5*H*cs * (cosh(kh)/sinh(kh)) * sin(-k*x(0));
    });

    //================= CONVERGENCE STORAGE =================
    std::vector<double> ndofs;
    std::vector<double> L2errs;

    //================= REFINEMENT/CONVERGENCE LOOP =========
    for (int lev = 0; lev < max_ref_levels; ++lev)
    {
        // Build FE space at this refinement level
        FiniteElementCollection *fec = new H1_FECollection(order, dim);              
        FiniteElementSpace fespace(&mesh, fec);  
        
        // SubMesh
        Array<int> bdr_attr;    // array with boundary attributes
        bdr_attr.Append(2);     // add the boundary attributes that I want to extract, 2 is top surface in "wave-tank.mesh"

        SubMesh mesh_fs = SubMesh::CreateFromBoundary(mesh, bdr_attr);  // create free surface mesh "mesh_fs" from parent mesh and selected boundary

        int dim_fs = mesh_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
        FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);   //need to define new collection and space for free surface
        FiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

        //Initialize phi
        GridFunction phi(&fespace);
        phi = 0.0;

        GridFunction phi_fs(&fespace_fs);
        phi_fs.ProjectCoefficient(phi_fs_init);
        mesh_fs.Transfer(phi_fs, phi);

        // Build Laplace system Δphi = 0 with Dirichlet data at the free surface
        Array<int> ess_bdr(mesh.bdr_attributes.Max()); 
        ess_bdr = 0;
        ess_bdr[2-1] = 1;                          // only attribute 2 is essential (free surface)

        // Impose free-surface Dirichlet φ
        //phi.ProjectBdrCoefficient(phi_fs_init, ess_bdr);

        BilinearForm a(&fespace);
        a.AddDomainIntegrator(new DiffusionIntegrator);
        a.Assemble();

        LinearForm b(&fespace); // RHS = 0
        b.Assemble();

        SparseMatrix A; Vector X, B;
        Array<int> ess_tdof; fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);
        a.FormLinearSystem(ess_tdof, phi, b, A, X, B);

        // Minimal solver
        GSSmoother M(A);
        PCG(A, M, B, X, 0, 400, 1e-12, 0.0);
        a.RecoverFEMSolution(X, b, phi);

        // --- Analytic phi in the volume ---
        FunctionCoefficient phi_exact_coef([&](const Vector &X){
            const double z_rel = X(2) - zmax; // alligning vertical domain from mesh and analytical solution z[-h, 0]
            const double vertical = cosh(k*(z_rel + h)) / sinh(kh);
            return -0.5*H*cs * vertical * sin(-k*X(0));
        });

        const double L2_err = phi.ComputeL2Error(phi_exact_coef);
        const int true_vsize = fespace.GetTrueVSize();

        ndofs.push_back(true_vsize);
        L2errs.push_back(L2_err);

        std::cout << "Level " << lev
                  << "  NDOFs = " << true_vsize
                  << "  L2 error = " << L2_err << std::endl;

        // Visualize only the last level
        if (lev == max_ref_levels - 1)
        {
            socketstream vol("localhost", 19916);
            vol << "solution\n" << mesh << phi
                << "window_title 'phi, refinement level "<< lev <<")'\nkeys mm"
                << flush;
        }

        // Clean up FEC for this level
        delete fec;
        delete fec_fs;

        // Refine for next level (skip after the last)
        if (lev < max_ref_levels - 1)
        {
            mesh.UniformRefinement();
        }
    }

    //================= WRITE CONVERGENCE DATA ===============
    {
        ofstream out("convergence_data.txt");
        out << "# DOFs   L2_error\n";
        for (size_t i = 0; i < ndofs.size(); ++i)
        {
            out << ndofs[i] << "   " << L2errs[i] << "\n";
        }
        out.close();
        cout << "\nConvergence data written to convergence_data.txt\n";
    }

    return 0;
}




    // ----- This only works if I know T beforehand -----
    // // Wave in x direction
    // const double H = 0.005;   // wave height
    // const double ph = 0.0;   // phase
    // const double T = 5;
    // const double g = 9.81;

    // // --- Compute wavelength and depth from the mesh ---
    // Vector bbmin, bbmax; 
    // mesh.GetBoundingBox(bbmin, bbmax);
    // const double Lx = bbmax(0) - bbmin(0);
    // const double Ly = bbmax(1) - bbmin(1);
    // const double h  = bbmax(2) - bbmin(2);   // since top is 0, bottom is -h
    // const double zmax = bbmax(2);

    // // --- Compute derived wave numbers ---
    // const double k = 2.0 * M_PI / Lx;       // one full period fits domain
    // const double ky = 0.0;
    // //const double k  = hypot(kx, ky);
    // int n = 30;
    // Dispersion d = SolveDispersion(g, h, T, n);  // Ensure dispersion relation
    // const double cs = sqrt(g/k * tanh(d.kh));

    // cout << cs - d.omega/k << endl;



    // --- Compute derived wave numbers ---
    // const double k = 2.0 * M_PI / Lx;       // one full period fits domain
    // const double ky = 0.0;
    // //const double k  = hypot(kx, ky);
    // const double kh = k * h;                 // consistent with geometry
    // const double cs = sqrt((g / k) * tanh(kh));
    // const double omega = 2*M_PI / T;