#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main (){

    int order = 4;
    int ref_levels = 0;

    const char *mesh_file = "../Meshes/wave-tank.mesh";

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim); // H1 Lagrange elements, pointer "fec" resides on heap, has to be deallocated again in the end
    FiniteElementSpace fespace(&mesh, fec);

    
    for (int i=0; i < ref_levels; i++){
        mesh.UniformRefinement();       // refining the mesh Mesh. Repeats uniform h-refinement ref_levels times
        fespace.Update();
    }
    
    //Initialize phi
    GridFunction phi(&fespace); // GridFunction "borrows" adress of fespace, an existing stack allocated Mesh
    phi = 0.0;

    Array<int> bdr_attr;    // array with boundary attributes
    bdr_attr.Append(2);     // add the boundary attributes that I want to extract, 2 is top surface in "wave-tank.mesh"

    
    SubMesh mesh_fs = SubMesh::CreateFromBoundary(mesh, bdr_attr);  // create free surface mesh "mesh_fs" from parent mesh and selected boundary
    
    
    // NC refinement missing
    
    int dim_fs = mesh_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);   //need to define new collection and space for free surface
    FiniteElementSpace fespace_fs(&mesh_fs, fec_fs);
    

    // GridFunction eta_fs(&fespace_fs);     // Here I imposed eta on the submesh, but I want to impose it on the top boundary nodes of the volum mesh
                                            //for free surface boundary conditions ->ex34.cpp line 454

    //================= Initialize eta
    //================= Initialize parameters =================
    // Wave in x direction
    const double H = 0.005;   // wave height
    const double ph = 0.0;   // phase
    const double T = 2*M_PI;
    const double g = 9.81;
    

    // --- Compute wavelength and depth from the mesh ---
    Vector bbmin, bbmax; 
    mesh.GetBoundingBox(bbmin, bbmax);
    const double Lx = bbmax(0) - bbmin(0);
    const double Ly = bbmax(1) - bbmin(1);
    const double h  = bbmax(2) - bbmin(2);   // since top is 0, bottom is -h, still water depth
    const double zmax = bbmax(2);

    // --- Compute derived wave numbers ---
    const double k = 2.0 * M_PI / Lx;       // one full period fits domain
    const double ky = 0.0;
    //const double k  = hypot(kx, ky);
    int n = 10;
    //const double kh = SolveDispersion(g, h, T, n);  // Ensure dispersion rrelation
    double kh = k*h;
    const double cs = sqrt(g/k * tanh(kh));
    double omega = k * sqrt((g/k) * tanh(kh));
    double t = 0.0;

    // FunctionCoefficient eta([&](const Vector& x){
    //     return H/2.0 * cos(k*x(0) + ky*x(1) + ph);
    // });

    // FunctionCoefficient phi_fs_init([&](const Vector& x){
    //     return 0.5*H*cs * cosh(kh)/sinh(kh) * sin(k*x(0) + ky*x(1) + ph);
    // });


    FunctionCoefficient eta_init([&](const Vector& x){
        return H/2.0 * cos(omega*t - k*x(0) + ph);   
    });

    FunctionCoefficient phi_fs_init([&](const Vector& x){
        return -0.5*H*cs * cosh(kh)/sinh(kh) * sin(omega*t -k*x(0) + ph);
    });
    
    GridFunction phi_fs(&fespace_fs);
    phi_fs.ProjectCoefficient(phi_fs_init);
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

    SparseMatrix A; Vector X, B;
    Array<int> ess_tdof; fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);
    a.FormLinearSystem(ess_tdof, phi, b, A, X, B);

    // Minimal solver
    GSSmoother M(A);
    PCG(A, M, B, X, 0, 400, 1e-12, 0.0);
    a.RecoverFEMSolution(X, b, phi);

    // Transfer trace of phi onto the free-surface submesh
    // GridFunction phi_fs(&fespace_fs);
    //SubMesh::CreateTransferMap(phi, phi_fs).Transfer(phi, phi_fs);

    // analytical w_tilde:
    // FunctionCoefficient w_exact([&](const Vector &x){
    //     return -k*k*k * H * cwave * cosh(k*h) *
    //     sin(k*x(0) - omega*t) / (2.0 * sinh(k*h));
    //     });

    // GridFunction w(&fespace);
    // w.GetDerivative(1, 2, phi);

    // GridFunction w_tilde(&fespace_fs);
    // mesh_fs.Transfer(w, w_tilde);

    // double l2err; 
    // w.ComputeL2Error(w_exact)

    socketstream vol("localhost", 19916);
    vol << "solution\n" << mesh << phi << "window_title 'Initial phi without time stepping'\nkeys mm" << flush;
    // socketstream surf("localhost", 19916);
    // surf << "solution\n" << mesh_fs << phi_fs << "window_title 'Free surface trace'\nkeys mm" << std::flush;

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