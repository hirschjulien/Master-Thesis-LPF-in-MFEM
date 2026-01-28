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
    }

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

        // ParBilinearForm a_loc(&fespace);
        // a_loc.AddDomainIntegrator(new DiffusionIntegrator);
        // a_loc.SetAssemblyLevel(AssemblyLevel::PARTIAL);     // partial assembly
        // a_loc.Assemble();
        
        phi.ExchangeFaceNbrData()   //

        ParLinearForm b_loc(&fespace);
        b_loc.Assemble();

        OperatorPtr A_loc;
        Vector X_loc, B_loc;
        
        a_loc_cached->FormLinearSystem(ess_tdof, phi, b_loc, A_loc, X_loc, B_loc);

        OperatorJacobiSmoother jacobi;  // jacobi preconditioner
        jacobi.SetOperator(*A_loc);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetPreconditioner(jacobi);
        cg.SetOperator(*A_loc);
        cg.SetRelTol(1e-24);
        cg.SetAbsTol(0.0);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(2000);
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

    int ref_levels = 0;
    int par_ref_levels = 0;

    const char *mesh_file = "../Meshes/wave-tank.mesh";

    if (myid == 0)
    {
        cout << "order  diff(||phi_fs_init||_inf - ||phi||_inf)\n";
        cout << "-------------------------------------------\n";
    }

    // ========= START P-CONVERGENCE LOOP =========
    for (int order = 1; order <= 10; order++)
    {
        Mesh mesh_serial(mesh_file, 1, 1);
        int dim = mesh_serial.Dimension();
        for (int i = 0; i < ref_levels; i++) { mesh_serial.UniformRefinement(); }

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

        double H = 0.005;
        double g = 9.81;

        Vector bbmin, bbmax;
        mesh.GetBoundingBox(bbmin, bbmax);

        double Lx = bbmax(0) - bbmin(0);
        double h  = bbmax(2) - bbmin(2);

        int m = 1;
        double k  = m * 2.0 * M_PI / Lx;
        double kh = k * h;

        double omega = sqrt(g * k * tanh(kh));
        double T     = 2.0 * M_PI / omega;
        double cwave = omega / k;
        
        if(order == 1){
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

        int nsteps = 100;
        double dt = t_final / nsteps;
        if(order==1){
        cout << "dt=" << dt << endl;
        }

        for (int step = 0; step < nsteps + 1; step++)
        {
            ode_solver->Step(eta_phi_fs, t, dt);
            eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));
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

        double local_err = phi.ComputeMaxError(phi_exact_coef);
        double global_err = 0.0;
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        const int ndofs = fespace.GetTrueVSize();

        if (myid == 0)
        {
            cout << "order = " << order
                << "  dofs = " << ndofs
                << "  ||phi - phi_exact||_inf = "
                << global_err << endl;
        }



        delete ode_solver;
        delete fec_fs;
        delete fec;
    }

    return 0;
}
