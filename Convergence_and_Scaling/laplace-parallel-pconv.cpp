#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main (int argc, char *argv[])
{
    Mpi::Init(argc, argv);                  // Initialize MPI (must be first MFEM call in parallel)
    Hypre::Init();                          // Initialize Hypre (parallel linear algebra backend)
    int num_procs = Mpi::WorldSize();       // Total number of MPI ranks
    int myid = Mpi::WorldRank();            // This rank's id

        ofstream fout;
    if (myid == 0){
        fout.open("data/laplace-pconv-eta.txt");
        fout << "# order dofs eta-inf-error";
    }

for(int order=1; order<=10; order++){
    //int order = 10;
    int ref_levels = 0;
    int par_ref_levels = 0;

    const char *mesh_file = "../Meshes/wave-tank.mesh";

    Mesh mesh_serial(mesh_file, 1, 1);  // Serial mesh read on all ranks
    int dim = mesh_serial.Dimension();

    for (int i = 0; i < ref_levels; i++)
    {
        mesh_serial.UniformRefinement();   // refining the mesh. Repeats uniform h-refinement ref_levels times
    }

        //================= Initialize parameters =================
    // Wave in x direction
    const double H = 0.01;   // wave height
    const double ph = 0.0;   // phase
    //const double T = 2*M_PI;
    const double g = 9.81;

    // --- Compute wavelength and depth from the mesh ---
    Vector bbmin, bbmax;
    mesh_serial.GetBoundingBox(bbmin, bbmax);   // [PARALLEL] Bounding box of the global domain (same values on all ranks)
    const double Lx = bbmax(0) - bbmin(0);
    const double Ly = bbmax(1) - bbmin(1);
    const double h  = bbmax(2) - bbmin(2);   // since top is 0, bottom is -h, still water depth
    const double zmax = bbmax(2);

    // --- Compute derived wave numbers ---
    const double lambda = 1.0;      // L = 1
    const double k      = 2.0*M_PI; // k = 2π
    //const double h      = 1.0/(2.0*M_PI);
    const double kh     = 1.0;
    //const double k = 2.0 * M_PI / Lx;       // one full period fits domain
    const double ky = 0.0;
    //double kh = k*h;
    // double omega = k * sqrt((g/k) * tanh(kh));
    // double U = omega / k;
    const double U = sqrt((g / k) * tanh(kh));
    const double T     = lambda / U;
    const double omega = 2.0 * M_PI / T;
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


    // ----- PARTITIONING OF THE SERIAL MESH ------
    ParMesh mesh_parallel(MPI_COMM_WORLD, mesh_serial); // Partition serial mesh across MPI ranks
    mesh_serial.Clear();                                        // Serial mesh no longer needed
    {
        for (int i = 0; i < par_ref_levels; i++)
        {
            mesh_parallel.UniformRefinement();
        }
    }

    FiniteElementCollection *fec = new H1_FECollection(order, dim); // H1 Lagrange elements, pointer "fec" resides on heap, has to be deallocated again in the end
    ParFiniteElementSpace fespace(&mesh_parallel, fec);  // FE space is now parallel and lives on ParMesh

    //Initialize phi
    ParGridFunction phi(&fespace); // Parallel GridFunction "borrows" address of fespace on ParMesh
    phi = 0.0;

    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh_parallel.bdr_attributes.Max()); // MArker Array for boundaries
    ess_bdr = 0;
    ess_bdr[2-1] = 1;
    //mesh_parallel.MarkExternalBoundaries(ess_bdr);
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);   // Needed for forming linear system
    

    // ------- FREE SURFACE SUBMESH -------
    Array<int> bdr_attr;
    bdr_attr.Append(2);
    ParSubMesh mesh_fs = ParSubMesh::CreateFromBoundary(mesh_parallel, bdr_attr);
    // create free surface mesh "mesh_fs" from parent parallel mesh and selected boundary

    int dim_fs = mesh_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);   //need to define new collection and space for free surface

    // [PARALLEL] Parallel FE space on the free-surface ParSubMesh
    ParFiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

    
    ParGridFunction phi_fs(&fespace_fs);
    phi_fs.ProjectCoefficient(phi_exact);   // Project free-surface potential on ParSubMesh    
    mesh_fs.Transfer(phi_fs, phi);  // Transfer from free-surface ParSubMesh to parent ParMesh trace


    ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator);
    a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    a.Assemble();

    ParLinearForm b(&fespace); // RHS = 0
    b.Assemble();


    OperatorPtr A;          // Global parallel sparse matrix. OperatorPtr object contains HypreParMatrix A
    Vector X, B;               // True-dof vectors (distributed)

    a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);     // Assemble global system A X = B


    // ------ LINEAR SOLVER ------
    // HypreBoomerAMG *prec = new HypreBoomerAMG;
    // prec->SetPrintLevel(0);
    OperatorJacobiSmoother *jacobi = NULL;
    jacobi = new OperatorJacobiSmoother(a, ess_tdof_list);


    // Standard MFEM parallel CG solver
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetPreconditioner(*jacobi);
    cg.SetOperator(*A);
    cg.SetRelTol(1e-24);         // relative tolerance
    cg.SetAbsTol(0.0);           // absolute tolerance
    cg.SetPrintLevel(0);         // 0 = silent, >0 prints "Setup phase times:" etc.
    cg.SetMaxIter(1000);          // max iterations
    cg.Mult(B, X);               // Solve
    
    a.RecoverFEMSolution(X, b, phi); // Recover parallel FE solution phi from X

        // ----- PARAVIEW -----
    // {
    // ParaViewDataCollection *pd = NULL;
    // pd = new ParaViewDataCollection("Laplace_parallel", &mesh_parallel);
    // pd->SetPrefixPath("ParaView");
    // pd->RegisterField("solution", &phi);
    // pd->SetLevelsOfDetail(order);
    // pd->SetDataFormat(VTKFormat::BINARY);
    // pd->SetHighOrderOutput(true);
    // pd->SetCycle(0);
    // pd->SetTime(0.0);
    // pd->Save();
    // delete pd;
    // }

    // ----- GLVIS -----
    // socketstream vol;
    // vol.open("localhost", 19916);
    // vol << "parallel " << num_procs << " " << myid << "\n"; // tell GLVis we're in parallel
    // vol << "solution\n" << mesh_parallel << phi
    //     << "window_title 'Initial phi without time stepping (parallel)'\n"
    //     << "keys mm" << flush;


    // double l2 = phi.ComputeMaxError(phi_exact);
    // if(myid==0)
    // cout << "Inf Phi Error: " << l2 << endl;


        // // ------ eta ----
        double local_err = phi.ComputeMaxError(phi_exact);
        double global_err = 0.0;
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        int local_ndofs = fespace.GetTrueVSize();
        int global_ndofs = 0;

        MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (myid == 0)
        {
            cout << "order = " << order
                << "  dofs = " << global_ndofs
                << "  ||phi - phi_exact||_inf = "
                << global_err << endl;

            fout << order << " " << global_ndofs << " " << global_err << "\n";
        }

    delete fec_fs;
    delete fec;
    delete jacobi;
    }
return 0;
}

// Example command line (4 MPI processes):
// mpirun -np 4 ./laplace_solver_p
