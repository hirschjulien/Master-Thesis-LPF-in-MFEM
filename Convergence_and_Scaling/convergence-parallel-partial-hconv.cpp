#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Solver for Linear Boundary Conditions on periodic domain:
// d_eta / dt = w_tilde
// d_phi_fs / dt = -g * eta

class rhs_linear : public TimeDependentOperator
{
private:
    ParGridFunction &phi;
    double g;
    Array<int> ess_tdof;
    ParFiniteElementSpace &fespace_fs;
    ParFiniteElementSpace &fespace;
    ParSubMesh &mesh_fs;

    mutable ParGridFunction eta_stage_gf;
    mutable ParGridFunction phi_fs_stage_gf;
    mutable ParGridFunction w;
    mutable ParGridFunction w_tilde;

    //HypreBoomerAMG *prec;
    
    //cache the partially assembled bilinear form (operator) once
    mutable ParBilinearForm *a_loc_cached = nullptr; 
    mutable OperatorJacobiSmoother *jacobi;

public:
    rhs_linear(ParFiniteElementSpace *fes_fs,
               ParFiniteElementSpace *fes_vol,
               ParGridFunction &phi_in,
               double g_in,
               const Array<int> &ess_tdof_in,
               ParSubMesh &mesh_fs_in)
        : TimeDependentOperator(2 * fes_fs->GetTrueVSize()),
          phi(phi_in),
          g(g_in),
          ess_tdof(ess_tdof_in),
          fespace_fs(*fes_fs),
          fespace(*fes_vol),
          mesh_fs(mesh_fs_in),
          eta_stage_gf(&fespace_fs),
          phi_fs_stage_gf(&fespace_fs),
          w(&fespace),
          w_tilde(&fespace_fs)
    {
        // prec = new HypreBoomerAMG;
        // prec->SetPrintLevel(0);
        
        //assemble PARTIAL bilinear form once here
        a_loc_cached = new ParBilinearForm(&fespace);               
        a_loc_cached->AddDomainIntegrator(new DiffusionIntegrator); 
        a_loc_cached->SetAssemblyLevel(AssemblyLevel::PARTIAL);     
        a_loc_cached->Assemble();                   
        
        jacobi = new OperatorJacobiSmoother(*a_loc_cached, ess_tdof); //boundaries have to be included in the jacobi
    }
    const ParGridFunction &GetWTilde() const { return w_tilde; }

    void Mult(const Vector &eta_phifs_true,
              Vector &d_eta_phifs_true_dt) const override
    {
        const int Nt = fespace_fs.GetTrueVSize();
        d_eta_phifs_true_dt.SetSize(2 * Nt);

        Vector eta_true(const_cast<double*>(eta_phifs_true.GetData()), Nt);
        Vector phi_fs_true(const_cast<double*>(eta_phifs_true.GetData()) + Nt, Nt);

        Vector deta_true_dt(d_eta_phifs_true_dt.GetData(), Nt);
        Vector dphi_fs_true_dt(d_eta_phifs_true_dt.GetData() + Nt, Nt);

        eta_stage_gf.SetFromTrueDofs(eta_true);
        phi_fs_stage_gf.SetFromTrueDofs(phi_fs_true);

        mesh_fs.Transfer(phi_fs_stage_gf, phi);

        ParLinearForm b_loc(&fespace);
        b_loc.Assemble();

        OperatorPtr A_loc;
        Vector X_loc, B_loc;
        
        a_loc_cached->FormLinearSystem(ess_tdof, phi, b_loc, A_loc, X_loc, B_loc);

        // OperatorJacobiSmoother jacobi;  // jacobi preconditioner
        // jacobi.SetOperator(*A_loc);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetPreconditioner(*jacobi);
        cg.SetOperator(*A_loc);
        cg.SetRelTol(1e-24);
        cg.SetAbsTol(0.0);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(5000);
        cg.Mult(B_loc, X_loc);

        a_loc_cached->RecoverFEMSolution(X_loc, b_loc, phi);

        phi.ExchangeFaceNbrData();
        phi.GetDerivative(1, 2, w);
        w.ExchangeFaceNbrData();
        mesh_fs.Transfer(w, w_tilde);

        w_tilde.GetTrueDofs(deta_true_dt);

        dphi_fs_true_dt = eta_true;
        dphi_fs_true_dt *= -g;
    }
};

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    ofstream fout;
    if (myid == 0){
        fout.open("data/pf-parallel-hconv-w4.txt");
        fout << "order dofs phifs_inf_error \n";
    }

    int ref_levels = 0;
    int par_ref_levels = 0;

    const char *mesh_file = "../Meshes/wave-tank.mesh";

    // if (myid == 0)
    // {
    //     cout << "order  diff(||phi_fs_init||_inf - ||phi||_inf)\n";
    //     cout << "-------------------------------------------\n";
    // }

    // *** CHANGED: one switch to choose p- or h-convergence quickly
    const bool h_convergence = true; // *** CHANGED: set true for h-convergence, false for p-convergence

    // ========= START CONVERGENCE LOOP =========
    for (int it = 0; it < 3; it++) // *** CHANGED: unified loop counter
    {
        // *** CHANGED: choose refinement/order based on test type
        const int order = h_convergence ? 4 : (it + 1);          // *** CHANGED
        const int ref_levels_it = h_convergence ? it : par_ref_levels; // *** CHANGED

        Mesh mesh_serial(mesh_file, 1, 1);
        int dim = mesh_serial.Dimension();
        for (int i = 0; i < ref_levels_it; i++) { mesh_serial.UniformRefinement(); } // *** CHANGED

        ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
        mesh_serial.Clear();

        for (int i = 0; i < par_ref_levels; i++) { mesh.UniformRefinement(); }

        FiniteElementCollection *fec = new H1_FECollection(order, dim);
        ParFiniteElementSpace fespace(&mesh, fec);

        mesh.EnsureNodes();

        Array<int> bdr_attr;
        bdr_attr.Append(2);
        ParSubMesh mesh_fs = ParSubMesh::CreateFromBoundary(mesh, bdr_attr);
        int dim_fs = mesh_fs.Dimension();

        FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);
        ParFiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

        int fe_true = fespace_fs.GetTrueVSize();
        Array<int> fe_offset(3);
        fe_offset[0] = 0;
        fe_offset[1] = fe_true;
        fe_offset[2] = 2 * fe_true;
        BlockVector eta_phi_fs(fe_offset);

        ParGridFunction eta(&fespace_fs);
        ParGridFunction phi_fs(&fespace_fs);

        double H = 0.01;
        double g = 9.81;

        Vector bbmin, bbmax;
        mesh.GetBoundingBox(bbmin, bbmax);

        double Lx = bbmax(0) - bbmin(0);
        //double h  = bbmax(2) - bbmin(2);
        double h = 1.0/(2.0*M_PI);      

        const double lambda = 1.0;      // L = 1
        const double k      = 2.0*M_PI / lambda;
        const double kh     = 1.0;

        const double cwave = sqrt((g / k) * tanh(kh));
        const double T     = lambda / cwave;
        const double omega = 2.0 * M_PI / T;
        
        if(it == 0 && myid==0){
        cout << "k=" << k << ", kh=" << kh << ", omega=" << omega << ", T=" << T << ", cwave=" << cwave << endl;
        }

        ODESolver *ode_solver = new RK4Solver();
        double t = 0.0;
        double t_final = T;

        double theta = 0.0;
        double kx_dir = cos(theta);
        double ky_dir = sin(theta);

        auto phase = [&](const Vector &X)
        {
            return omega * t - k * (kx_dir * X(0) + ky_dir * X(1));
        };

        FunctionCoefficient eta_init([&](const Vector &X)
        {
            return 0.5 * H * cos(phase(X));
        });

        FunctionCoefficient phi_fs_init([&](const Vector &X)
        {
            return -0.5 * H * cwave * cosh(kh)/sinh(kh) * sin(phase(X));
        });

        eta.ProjectCoefficient(eta_init);
        phi_fs.ProjectCoefficient(phi_fs_init);

        eta.GetTrueDofs(eta_phi_fs.GetBlock(0));
        phi_fs.GetTrueDofs(eta_phi_fs.GetBlock(1));

        ParGridFunction phi(&fespace);
        phi = 0.0;
        mesh_fs.Transfer(phi_fs, phi);

        Array<int> essential_bdr(mesh.bdr_attributes.Max());
        essential_bdr = 0;
        essential_bdr[2-1] = 1;

        Array<int> ess_tdof;
        fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);

        rhs_linear surface(&fespace_fs, &fespace, phi, g, ess_tdof, mesh_fs);
        ode_solver->Init(surface);

        int nsteps = 150;
        double dt = t_final / nsteps;
        if(it==0 && myid==0){
        cout << "dt=" << dt << endl;
        }

        for (int step = 0; step < nsteps + 1; step++)
        {
            ode_solver->Step(eta_phi_fs, t, dt);
            eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));
            phi_fs.SetFromTrueDofs(eta_phi_fs.GetBlock(1));
        }


        // ======= COMPUTE ERROR AT CURRENT t AFTER TIME INTEGRATION =========
        double zmax = bbmax(2);
        // ---- Compute L_infinity error in the VOLUME and print order + DOFs ----
        FunctionCoefficient phi_exact_coef([&](const Vector &X)
        {
            const double z_rel    = X(2) - zmax;                  // align z so free surface is at 0
            const double vertical = cosh(k*(z_rel + h)) / sinh(kh);
            const double arg      = omega * t - k * (kx_dir * X(0) + ky_dir * X(1));
            return -0.5 * H * cwave * vertical * sin(arg);
        });

FunctionCoefficient w_exact([&](const Vector &X)
{
    // X is 2D on the free surface: X(0)=x, X(1)=y
    const double arg = omega*t - k*(kx_dir*X(0) + ky_dir*X(1));
    // for linear waves: w(z=0) = dphi/dz at free surface
    // using your volume expression at z_rel=0:
    return -0.5 * H * cwave * k * (sinh(k*h)/sinh(k*h)) * sin(arg); // simplifies to -0.5 H cwave k sin(arg)
});

        FunctionCoefficient phi_fs_exact([&](const Vector &X)
        {
            const double arg = omega * t - k * (kx_dir * X(0) + ky_dir * X(1));
            return -0.5 * H * cwave * cosh(kh)/sinh(kh) * sin(arg);
        });


        
        // // ----- eta -------
        // double local_err = eta.ComputeMaxError(eta_init);
        // double global_err = 0.0;
        // MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // int local_ndofs = fespace.GetTrueVSize();
        // int global_ndofs = 0;
        // MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // if (myid == 0)
        // {
        //     cout << (h_convergence ? "h_level = " : "order = ") << it         
        //         << "  order = " << order                                     
        //         << "  dofs = " << global_ndofs
        //         << "  ||eta - eta_exact||_inf = "
        //         << global_err << endl;
        //     fout << order << " " << global_ndofs << " " << global_err << "\n";
        // }

        // // -------- phi ---------
        // double local_err = phi.ComputeMaxError(phi_exact_coef);
        // double global_err = 0.0;
        // MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // int local_ndofs = fespace.GetTrueVSize();
        // int global_ndofs = 0;
        // MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // if (myid == 0)
        // {
        //     cout << (h_convergence ? "h_level = " : "order = ") << it        
        //         << "  order = " << order                                     
        //         << "  dofs = " << global_ndofs
        //         << "  ||phi - phi_exact||_inf = "
        //         << global_err << endl;

        //     fout << order << " " << global_ndofs << " " << global_err << "\n";
        // }


        // // ---- w ------
        const ParGridFunction &w_num = surface.GetWTilde();


        double local_err = w_num.ComputeMaxError(w_exact);
        double global_err = 0.0;
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        int local_ndofs = fespace.GetTrueVSize();
        int global_ndofs = 0;
        MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (myid == 0)
        {
            cout << "order = " << order
                << "  dofs = " << global_ndofs
                << "  ||w - w_exact||_inf = "
                << global_err << endl;

        //     // NOTE: your file header currently says "phi_inf_error".
        //     // If you want to keep *minimal* changes, just write to the same file anyway.
            fout << order << " " << global_ndofs << " " << global_err << "\n";
        }

                        // ------ phi_fs ----
        // double local_err = phi_fs.ComputeMaxError(phi_fs_exact);
        // double global_err = 0.0;
        // MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // int local_ndofs = fespace.GetTrueVSize();
        // int global_ndofs = 0;

        // MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // if (myid == 0)
        // {
        //     cout << "order = " << order
        //         << "  dofs = " << global_ndofs
        //         << "  ||phi_fs - phi_fs_exact||_inf = "
        //         << global_err << endl;

        //     fout << order << " " << global_ndofs << " " << global_err << "\n";
        // }


        delete ode_solver;
        delete fec_fs;
        delete fec;
    }

    if (myid == 0) {fout.close();}

    return 0;
}
