#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main (int argc, char *argv[])
{
    Mpi::Init(argc, argv);                  // Initialize MPI (must be first MFEM call in parallel)
    Hypre::Init();                          // Initialize Hypre

    int num_procs = Mpi::WorldSize();       //  Total number of MPI ranks
    int myid      = Mpi::WorldRank();       //  This rank's id

    int order = 2;
    int ref_levels = 0;

    const char *mesh_file = "../Meshes/wave-tank.mesh";

    Mesh *mesh_serial = new Mesh(mesh_file, 1, 1);  // Serial mesh read on all ranks
    int dim = mesh_serial->Dimension();


    for (int i = 0; i < ref_levels; i++)
    {
        mesh_serial->UniformRefinement();   // refining the mesh. Repeats uniform h-refinement ref_levels times
    }


    //  Build ParMesh by partitioning serial mesh
    ParMesh *mesh = new ParMesh(MPI_COMM_WORLD, *mesh_serial); // Partition serial mesh across MPI ranks
    delete mesh_serial;                                        // Serial mesh no longer needed


    FiniteElementCollection *fec = new H1_FECollection(order, dim); // H1 Lagrange elements, pointer "fec" resides on heap, has to be deallocated again in the end

    ParFiniteElementSpace fespace(mesh, fec);  //  FE space is now parallel and lives on ParMesh

    //Initialize phi
    ParGridFunction phi(&fespace); //  Parallel GridFunction "borrows" address of fespace on ParMesh
    phi = 0.0;

    Array<int> bdr_attr;    // array with boundary attributes
    bdr_attr.Append(2);     // add the boundary attributes that I want to extract, 2 is top surface in "wave-tank.mesh"


    ParSubMesh mesh_fs = ParSubMesh::CreateFromBoundary(*mesh, bdr_attr);
    // create free surface mesh "mesh_fs" from parent parallel mesh and selected boundary


    int dim_fs = mesh_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);   //need to define new collection and space for free surface

    //  Parallel FE space on the free-surface ParSubMesh
    ParFiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

    //================= Initialize eta
    //================= Initialize parameters =================
    // Wave in x direction
    const double H = 0.005;   // wave height
    const double ph = 0.0;   // phase
    const double T = 2*M_PI;
    const double g = 9.81;

    // --- Compute wavelength and depth from the mesh ---
    Vector bbmin, bbmax;
    mesh->GetBoundingBox(bbmin, bbmax);   //  Bounding box of the global domain (same values on all ranks)
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
        return 0.5*H*cs * cosh(kh)/sinh(kh) * sin(omega*t -k*x(0) + ph);
    });
    
    ParGridFunction phi_fs(&fespace_fs);
    phi_fs.ProjectCoefficient(phi_fs_init);   //  Project free-surface potential on ParSubMesh

    //  Transfer from free-surface ParSubMesh to parent ParMesh trace
    mesh_fs.Transfer(phi_fs, phi);


    Array<int> ess_bdr(mesh->bdr_attributes.Max()); 
    ess_bdr = 0;
    ess_bdr[2-1] = 1;                          // only attribute 2 is essential

    ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator);  // same Laplace operator, now assembled in parallel
    a.Assemble();

    ParLinearForm b(&fespace); // RHS = 0, defined in parallel
    b.Assemble();


    HypreParMatrix A;          //  Global parallel sparse matrix
    Vector X, B;               //  True-dof vectors
    Array<int> ess_tdof;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof);   //  Essential true DOFs on the parallel FE space

    a.FormLinearSystem(ess_tdof, phi, b, A, X, B);     //  Assemble global system A X = B

    // AMG preconditioner on the parallel matrix
    HypreBoomerAMG amg(A);
    amg.SetPrintLevel(0);  // 0 = silent, >0 prints "Setup phase times:" etc.

    // Standard MFEM parallel CG solver
    CGSolver pcg(MPI_COMM_WORLD);
    pcg.SetOperator(A);
    pcg.SetRelTol(1e-12);                      // relative tolerance
    pcg.SetAbsTol(0.0);                        // absolute tolerance
    pcg.SetMaxIter(400);                       // max iterations
    pcg.SetPrintLevel(myid == 0 ? 1 : 0);      // only rank 0 prints residual history
    pcg.SetPreconditioner(amg);

    // Debug prints to see if we get stuck
    if (myid == 0) { cout << "Before CG solve" << endl; }
    pcg.Mult(B, X);
    if (myid == 0) { cout << "After CG solve" << endl; }


    a.RecoverFEMSolution(X, b, phi); //  Recover parallel FE solution phi from X


    socketstream vol;
    vol.open("localhost", 19916);    //  Connect each rank to GLVis server
    if (vol)
    {
        vol << "parallel " << num_procs << " " << myid << "\n"; // tell GLVis we're in parallel
        vol << "solution\n" << *mesh << phi
            << "window_title 'Initial phi without time stepping (parallel)'\n"
            << "keys mm" << flush;
    }
    else if (myid == 0)
    {
        cout << "Unable to connect to GLVis server at localhost:19916\n";
    }

    delete fec_fs;
    delete fec;
    delete mesh;

    return 0;
}

// Example command line :
// mpirun -np 4 ./laplace_solver_p
