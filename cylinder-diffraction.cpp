#include "mfem.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace mfem;

// ==================== RHS for linear free-surface PF (your original) ====================
class rhs_linear : public TimeDependentOperator
{
private:
    ParGridFunction &phi;
    double g;
    Array<int> ess_tdof;
    ParFiniteElementSpace &fespace_fs;
    ParFiniteElementSpace &fespace;
    ParSubMesh &mesh_fs;

    ParGridFunction &Cgen_gf;
    ParGridFunction &Cabs_gf;

    double H, omega, k, kx_dir, ky_dir, cwave, kh;
    double T;
    double tau;

    mutable ParGridFunction eta_stage_gf;
    mutable ParGridFunction phi_fs_stage_gf;
    mutable ParGridFunction w;
    mutable ParGridFunction w_tilde;

    mutable ParGridFunction eta_e;
    mutable ParGridFunction phi_fs_e;

    mutable ParGridFunction deta_v;
    mutable ParGridFunction dphi_v;

    HypreBoomerAMG *prec;

public:
    rhs_linear(ParFiniteElementSpace *fes_fs,
               ParFiniteElementSpace *fes_vol,
               ParGridFunction &phi_in,
               double g_in,
               const Array<int> &ess_tdof_in,
               ParSubMesh &mesh_fs_in,
               ParGridFunction &Cgen_in,
               ParGridFunction &Cabs_in,
               double H_in,
               double omega_in,
               double k_in,
               double kx_dir_in,
               double ky_dir_in,
               double cwave_in,
               double kh_in,
               double T_in,
               double tau_in)
        : TimeDependentOperator(2 * fes_fs->GetTrueVSize()),
          phi(phi_in),
          g(g_in),
          ess_tdof(ess_tdof_in),
          fespace_fs(*fes_fs),
          fespace(*fes_vol),
          mesh_fs(mesh_fs_in),
          Cgen_gf(Cgen_in),
          Cabs_gf(Cabs_in),
          H(H_in),
          omega(omega_in),
          k(k_in),
          kx_dir(kx_dir_in),
          ky_dir(ky_dir_in),
          cwave(cwave_in),
          kh(kh_in),
          T(T_in),
          tau(tau_in),
          eta_stage_gf(&fespace_fs),
          phi_fs_stage_gf(&fespace_fs),
          w(&fespace),
          w_tilde(&fespace_fs),
          eta_e(&fespace_fs),
          phi_fs_e(&fespace_fs),
          deta_v(&fespace_fs),
          dphi_v(&fespace_fs)
    {
        prec = new HypreBoomerAMG;
        prec->SetPrintLevel(0);
    }

    ~rhs_linear() { delete prec; }

    void Mult(const Vector &eta_phifs_true,
              Vector &d_eta_phifs_true_dt) const override
    {
        const int Nt = fespace_fs.GetTrueVSize();
        d_eta_phifs_true_dt.SetSize(2 * Nt);

        Vector eta_true (const_cast<double*>(eta_phifs_true.GetData()), Nt);
        Vector phfs_true(const_cast<double*>(eta_phifs_true.GetData()) + Nt, Nt);

        Vector deta_true_dt (d_eta_phifs_true_dt.GetData(), Nt);
        Vector dphfs_true_dt(d_eta_phifs_true_dt.GetData() + Nt, Nt);

        eta_stage_gf.SetFromTrueDofs(eta_true);
        phi_fs_stage_gf.SetFromTrueDofs(phfs_true);

        mesh_fs.Transfer(phi_fs_stage_gf, phi);

        ParBilinearForm a_loc(&fespace);
        a_loc.AddDomainIntegrator(new DiffusionIntegrator);
        a_loc.Assemble();

        ParLinearForm b_loc(&fespace);
        b_loc.Assemble();

        OperatorPtr A_loc;
        Vector X_loc, B_loc;

        a_loc.FormLinearSystem(ess_tdof, phi, b_loc, A_loc, X_loc, B_loc);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetPreconditioner(*prec);
        cg.SetOperator(*A_loc);
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(0.0);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(400);
        cg.Mult(B_loc, X_loc);

        a_loc.RecoverFEMSolution(X_loc, b_loc, phi);

        phi.GetDerivative(1, 2, w);
        mesh_fs.Transfer(w, w_tilde);
        w_tilde.GetTrueDofs(deta_true_dt);

        dphfs_true_dt = eta_true;
        dphfs_true_dt *= -g;

        // ============ RELAXATION ZONES ============
        const double t_stage = this->GetTime();

        auto phase = [&](const Vector &Xfs) -> double
        {
            return omega * t_stage - k * (kx_dir * Xfs(0) + ky_dir * Xfs(1));
        };

        FunctionCoefficient eta_exact([&](const Vector &Xfs)
        {
            return 0.5 * H * cos(phase(Xfs));
        });

        FunctionCoefficient phi_fs_exact([&](const Vector &Xfs)
        {
            return -0.5 * H * cwave * (cosh(kh) / sinh(kh)) * sin(phase(Xfs));
        });

        eta_e.ProjectCoefficient(eta_exact);
        phi_fs_e.ProjectCoefficient(phi_fs_exact);

        deta_v.SetFromTrueDofs(deta_true_dt);
        dphi_v.SetFromTrueDofs(dphfs_true_dt);

        const double inv_tau = 1.0 / tau;

        const double *Cgen = Cgen_gf.GetData();
        const double *Cabs = Cabs_gf.GetData();

        const double *eta_ex = eta_e.GetData();
        const double *phi_ex = phi_fs_e.GetData();

        const double *eta_vdat = eta_stage_gf.GetData();
        const double *phi_vdat = phi_fs_stage_gf.GetData();

        double *deta = deta_v.GetData();
        double *dphi = dphi_v.GetData();

        const int Nv = fespace_fs.GetVSize();

        const double n_ramp = 3.0;
        const double Tramp  = n_ramp * T;

        double alpha_gen = t_stage / Tramp;
        alpha_gen = min(1.0, max(0.0, alpha_gen));

        for (int i = 0; i < Nv; ++i)
        {
            const double gen_weight = alpha_gen * Cgen[i];

            deta[i] += (gen_weight * inv_tau) * (eta_ex[i] - eta_vdat[i]);
            dphi[i] += (gen_weight * inv_tau) * (phi_ex[i] - phi_vdat[i]);

            deta[i] += (Cabs[i] * inv_tau) * (0.0 - eta_vdat[i]);
            dphi[i] += (Cabs[i] * inv_tau) * (0.0 - phi_vdat[i]);
        }

        deta_v.GetTrueDofs(deta_true_dt);
        dphi_v.GetTrueDofs(dphfs_true_dt);
    }
};

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    const int myid = Mpi::WorldRank();
    const int num_procs = Mpi::WorldSize();

    // ---------------- minimal user settings ----------------
    const char *mesh_file = "mesh_cylinder.msh";
    const int order = 4;
    const int nsteps = 100;

    const double H = 0.05;
    const double g = 9.81;

    const double lambda = 0.22;
    const double kh = 1.0;

    // Wave params
    const double k     = 2.0 * M_PI / lambda;
    const double cwave = sqrt((g / k) * tanh(kh));
    const double T     = lambda / cwave;
    const double omega = 2.0 * M_PI / T;

    double t = 0.0;
     double t_final = 5 * T;
     double dt = t_final / nsteps;
     cout << dt << endl;
     double t_last_start = t_final - T;
    // ------------------------------------------------------

    // Mesh
    Mesh mesh_serial(mesh_file, 1, 1);
    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
    mesh_serial.Clear();
    mesh.EnsureNodes();
    mesh.SetCurvature(order);

    const int dim = mesh.Dimension();

    // Volume space for phi
    H1_FECollection *fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace fespace(&mesh, fec);

    // Free-surface submesh (boundary attribute 2)
    Array<int> bdr_attr;
    bdr_attr.Append(2);
    ParSubMesh mesh_fs = ParSubMesh::CreateFromBoundary(mesh, bdr_attr);
    const int dim_fs = mesh_fs.Dimension();
    mesh_fs.EnsureNodes();
    mesh_fs.SetCurvature(order);

    H1_FECollection *fec_fs = new H1_FECollection(order, dim_fs);
    ParFiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

    // State vector (eta, phi_fs) in true dofs
    const int fe_true = fespace_fs.GetTrueVSize();
    Array<int> fe_offset(3);
    fe_offset[0] = 0;
    fe_offset[1] = fe_true;
    fe_offset[2] = 2 * fe_true;
    BlockVector eta_phi_fs(fe_offset);

    ParGridFunction eta(&fespace_fs);
    ParGridFunction phi_fs(&fespace_fs);



    // Direction
    const double theta = 0.0;
    const double kx_dir = cos(theta);
    const double ky_dir = sin(theta);

    auto phase = [&](const Vector &X) -> double
    {
        return omega * t - k * (kx_dir * X(0) + ky_dir * X(1));
    };

    // Initial conditions
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

    // Volume potential
    ParGridFunction phi(&fespace);
    phi = 0.0;
    mesh_fs.Transfer(phi_fs, phi);

    // Essential BC for phi (boundary attribute 2)
    Array<int> essential_bdr(mesh.bdr_attributes.Max());
    essential_bdr = 0;
    essential_bdr[2 - 1] = 1;

    Array<int> ess_tdof;
    fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);

    // Relaxation weights on free-surface
    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);

    const double Ng  = 2.0;
    const double xg0 = bbmin(0);
    const double xg1 = xg0 + Ng * lambda;

    FunctionCoefficient Cgen_coef([&](const Vector &X)
    {
        const double x = X(0);
        if (x <= xg0) return 1.0;
        if (x >= xg1) return 0.0;
        const double xi = (x - xg0) / (xg1 - xg0);
        const double s = 1.0 - xi;
        return -2.0*s*s*s + 3.0*s*s;
    });

    ParGridFunction Cgen_gf(&fespace_fs);
    Cgen_gf.ProjectCoefficient(Cgen_coef);

    const double Ns = 2.0;
    const double x1 = bbmax(0);
    const double x0 = x1 - Ns * lambda;
    const double p = 5.0;

    FunctionCoefficient Cabs_coef([&](const Vector &X)
    {
        const double x = X(0);
        if (x <= x0) return 0.0;
        if (x >= x1) return 1.0;
        const double xi = (x - x0) / (x1 - x0);
        return pow(xi, p);
    });

    ParGridFunction Cabs_gf(&fespace_fs);
    Cabs_gf.ProjectCoefficient(Cabs_coef);

    // ODE operator + solver
    rhs_linear surface(&fespace_fs, &fespace,
                       phi, g, ess_tdof,
                       mesh_fs, Cgen_gf, Cabs_gf,
                       H, omega, k, kx_dir, ky_dir,
                       cwave, kh, T, dt);

    ODESolver *ode_solver = new RK4Solver();
    ode_solver->Init(surface);

    // ============ envelope as max over last period (whole mesh_fs) ==============
    ParGridFunction eta_env(&fespace_fs);
    eta_env = -1e300;

    // ====== ETA MAX ========
    // Looping over whole eta at every time step and constructing the envelope (eta_env by choosing the max eta values at each node
    for (int step = 0; step < nsteps + 1; step++)
    {
        ode_solver->Step(eta_phi_fs, t, dt);
        eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));

        if (t >= t_last_start)
        {
            double *env = eta_env.GetData();
            const double *cur = eta.GetData();
            const int n = eta_env.Size();
            for (int i = 0; i < n; i++)
            {
                env[i] = max(env[i], cur[i]);
            }
        }

        if (myid == 0 && step % 1 == 0)
        {
            cout << "step " << step << "/" << nsteps << " t=" << t << "\n";
        }
    }

    eta_env *= (2.0 / H);

    // ==================== ParaView output: envelope on free surface ====================
    ParaViewDataCollection pv_fs("eta_envelope_2", &mesh_fs);
    pv_fs.SetPrefixPath("ParaView");
    pv_fs.SetLevelsOfDetail(order);
    pv_fs.SetDataFormat(VTKFormat::BINARY);
    pv_fs.SetHighOrderOutput(true);

    // register ONLY what you want to visualize
    pv_fs.RegisterField("eta", &eta);
    pv_fs.RegisterField("eta_env", &eta_env);


    // single snapshot (envelope is time-independent after extraction)
    pv_fs.SetCycle(0);
    pv_fs.SetTime(t_final);
    pv_fs.Save();


    //     // ---- Parallel GLVis: EVERY rank must send its local partition ----
    // {
    //     const char vishost[] = "localhost";
    //     const int  visport   = 19916;

    //     socketstream sol_sock;
    //     sol_sock.open(vishost, visport);
    //     sol_sock.precision(8);

    //     // Tell GLVis this is a parallel visualization stream
    //     sol_sock << "parallel " << num_procs << " " << myid << "\n";
    //     sol_sock << "solution\n" << mesh_fs << eta_env << flush;
    // }

    // if (myid == 0)
    // {
    //     cout << "Sent eta_env (max over last period) to GLVis in parallel.\n";
    //     cout << "Run GLVis as: glvis -np " << num_procs << "\n";
    // }
    // // -------------------------------------------------------------------------------------------

    delete fec;
    delete fec_fs;
    delete ode_solver;
    return 0;
}
