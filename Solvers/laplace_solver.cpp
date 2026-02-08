#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// SERIAL LAPLACE SOLVER

int main (){
    int order = 4;
    int ref_levels = 2; // 0 --> 16 Elements

    const char *mesh_file = "../Meshes/wave-tank.mesh";

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim); // H1 Lagrange elements, pointer "fec" resides on heap, has to be deallocated again in the end
    FiniteElementSpace fespace(&mesh, fec);

    
    for (int i=0; i < ref_levels; i++){
        mesh.UniformRefinement();       // refining the mesh Mesh. Repeats uniform h-refinement ref_levels times
        fespace.Update();
    }

    Array<int> bdr_attr;    // array with boundary attributes
    bdr_attr.Append(2);     // add the boundary attributes that I want to extract, 2 is top surface in "wave-tank.mesh"

    
    SubMesh mesh_fs = SubMesh::CreateFromBoundary(mesh, bdr_attr);  // create free surface mesh "mesh_fs" from parent mesh and selected boundary
    int dim_fs = mesh_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);   //need to define new collection and space for free surface
    FiniteElementSpace fespace_fs(&mesh_fs, fec_fs);
    
        //================= Initialize parameters =================
    // Wave in x direction
    const double H = 0.005;   // wave height
    const double ph = 0.0;   // phase
    const double T = 2*M_PI;
    const double g = 9.81;

    // --- Compute wavelength and depth from the mesh ---
    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);   // [PARALLEL] Bounding box of the global domain (same values on all ranks)
    const double Lx = bbmax(0) - bbmin(0);
    const double Ly = bbmax(1) - bbmin(1);
    const double h  = bbmax(2) - bbmin(2);   // since top is 0, bottom is -h, still water depth
    const double zmax = bbmax(2);

    // --- Compute derived wave numbers ---
    const double k = 2.0 * M_PI / Lx;       // one full period fits domain
    const double ky = 0.0;
    double kh = k*h;
    double omega = k * sqrt((g/k) * tanh(kh));
    double U = omega / k;
    double t = 0.0;
    double theta = 0.0;   // wave direction angle
    double kx_dir = cos(theta);
    double ky_dir = sin(theta);

    auto phase = [&](const Vector &X)
    {
        double x = X(0);
        double y = (X.Size() >= 2 ? X(1) : 0.0);
        return omega * t - k * (kx_dir * x + ky_dir * y);
    };

    FunctionCoefficient eta_init([&](const Vector& x){
        return H/2.0 * cos(omega*t - k*x(0) + ph);   
    });

    FunctionCoefficient phi_exact([&](const Vector &X)
    {
        double z_rel    = X(2) - zmax;       // z + h
        double vertical = cosh(k * (z_rel + h)) / sinh(kh);
        double arg      = omega * t - k * (kx_dir * X(0) + ky_dir * X(1));

        return -0.5 * H * U * vertical * sin(arg);
    });
    
    //Initialize phi
    GridFunction phi(&fespace); // GridFunction "borrows" adress of fespace, an existing stack allocated Mesh
    phi = 0.0;

    GridFunction phi_fs(&fespace_fs);
    phi_fs.ProjectCoefficient(phi_exact);
    mesh_fs.Transfer(phi_fs, phi);


    // Build Laplace system Δphi = 0 with Dirichlet data at the free surface
    Array<int> ess_bdr(mesh.bdr_attributes.Max()); 
    ess_bdr = 0;
    ess_bdr[2-1] = 1;                          // only attribute 2 is essential
    //phi.ProjectBdrCoefficient(phi_fs_init, ess_bdr);   // impose phi_eta as Dirichlet boundary condition
                                                // Here it is assumed that phi_eta = eta

    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator);
    a.Assemble();

    LinearForm b(&fespace); // RHS = 0
    b.Assemble();

    SparseMatrix A; 
    Vector X, B;
    Array<int> ess_tdof; fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);
    a.FormLinearSystem(ess_tdof, phi, b, A, X, B);

    // Minimal solver
    GSSmoother M(A);
    PCG(A, M, B, X, 0, 500, 1e-24, 0.0);
    a.RecoverFEMSolution(X, b, phi);

    // Transfer trace of phi onto the free-surface submesh
    // GridFunction phi_fs(&fespace_fs);
    // SubMesh::CreateTransferMap(phi, phi_fs).Transfer(phi, phi_fs);
    
    // socketstream vol("localhost", 19916);
    // vol << "solution\n" << mesh << phi << "window_title 'Initial phi without time stepping'\nkeys mm" << flush;
    // socketstream surf("localhost", 19916);
    // surf << "solution\n" << mesh_fs << phi_fs << "window_title 'Free surface trace'\nkeys mm" << std::flush;

    // ------ GET w AND COMPARE TO ANLYTICAL SOLUTION -------
    GridFunction w(&fespace);   // w on parent mesh
    phi.GetDerivative(1, 2, w); // Derivative in z direction for whole parent mesh
    GridFunction w_tilde(&fespace_fs); // w_tilde on Submesh
    mesh_fs.Transfer(w, w_tilde);   // Transfer the vertical derivative 

    FunctionCoefficient w_analytical([&](const Vector& X){
        double z_rel    = X(2) - zmax;
        return -H * U * 0.5 * k * (sinh(k*(z_rel+h)) / sinh(k*h)) * sin(omega*t - k*(kx_dir * X(0) + ky_dir * X(1)));
    });

    double l2_w = w.ComputeL2Error(w_analytical);
    double l2 = phi.ComputeL2Error(phi_exact);
    cout << "w Error:" << l2_w << "  phi Error:" << l2 << endl;

    delete fec_fs;
    delete fec;

return 0;
}
      


// I don't need this function right now because k is predefined from my mesh to ensure periodicity on my mesh
const double SolveDispersion(const double g, const double h, const double T, int n){
    double omega = 2*M_PI / T;
    double kh = omega*omega * h / g;  //initial kh
    for (int i; i < n; i++){
        kh = sqrt(kh * omega*omega * h/g * cosh(kh));
    };
    return kh;
};



    // Move columns by eta and plot the deformed mesh

    // function<double(const Vector&)> eta_deform = [&](const Vector& x){
    //     return -0.5*H*cs * cosh(kh)/sinh(kh) * sin(-k*x(0));
    // };

    // double z_cut = 1.0;

    // mfem::VectorFunctionCoefficient lift(3,
    //     [&](const mfem::Vector &x, mfem::Vector &y)
    //     {
    //         y = x;
    //         if (x(2) >= z_cut) { y(2) += eta_deform(x); } // lift only above the threshold
    //     });
    // mesh.Transform(lift);
    // mesh_fs.Transform(lift);


    // mesh.FinalizeTopology();
    // mesh.Finalize();

    // socketstream mesh_sock("localhost", 19916);
    // mesh_sock << "mesh\n" << mesh << "window_title 'Deformed mesh'\n" << std::flush;