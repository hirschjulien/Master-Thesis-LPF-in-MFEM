#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace mfem;


// ==================== DISPERSION SOLVER ====================
// only needed when I choose wave parameters from T
static inline double coth_safe(double x)
{
    const double eps = 1e-12;
    x = max(x, eps);
    return cosh(x) / sinh(x);
}

static inline double DispersionRelationWaves(double g, double T, double h, int n)
{
    const double w = 2.0 * M_PI / T;
    double kh = (w*w) * h / g;
    kh = max(kh, 1e-8);

    for (int i = 0; i < n; ++i)
    {
        kh = sqrt((w*w/g) * h * kh * coth_safe(kh));
        kh = max(kh, 1e-8);
    }
    return kh;
}


// ================= SETTIGN UP RHS FOR TIME INTEGRATION ==========
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

    mutable ParBilinearForm *a_loc_cach = nullptr;
    mutable OperatorJacobiSmoother *jacobi;

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
        a_loc_cach = new ParBilinearForm(&fespace);
        a_loc_cach->AddDomainIntegrator(new DiffusionIntegrator);
        a_loc_cach->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        a_loc_cach->Assemble();

        jacobi = new OperatorJacobiSmoother(*a_loc_cach, ess_tdof);
    }

    ~rhs_linear() { delete jacobi; delete a_loc_cach; }

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

        ParLinearForm b_loc(&fespace);
        b_loc.Assemble();

        OperatorPtr A_loc;
        Vector X_loc, B_loc;

        a_loc_cach->FormLinearSystem(ess_tdof, phi, b_loc, A_loc, X_loc, B_loc);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetPreconditioner(*jacobi);
        cg.SetOperator(*A_loc);
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(0.0);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(1000);
        cg.Mult(B_loc, X_loc);

        a_loc_cach->RecoverFEMSolution(X_loc, b_loc, phi);

        phi.GetDerivative(1, 2, w);
        mesh_fs.Transfer(w, w_tilde);

        w_tilde.GetTrueDofs(deta_true_dt);

        dphfs_true_dt = eta_true;
        dphfs_true_dt *= -g;

        // ========= RELAXATION ZONES =========
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
            const double cgen = Cgen[i];
            const double cabs = Cabs[i];

            const double gen_weight = alpha_gen * cgen;

            deta[i] += (gen_weight * inv_tau) * (eta_ex[i] - eta_vdat[i]);
            dphi[i] += (gen_weight * inv_tau) * (phi_ex[i] - phi_vdat[i]);

            // absorption: drive to zero
            deta[i] += (cabs * inv_tau) * (0.0 - eta_vdat[i]);
            dphi[i] += (cabs * inv_tau) * (0.0 - phi_vdat[i]);
        }

        deta_v.GetTrueDofs(deta_true_dt);
        dphi_v.GetTrueDofs(dphfs_true_dt);
    }
};

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    // ========== MESH etc. ============
    int order = 4;
    int ref_levels = 0;
    int par_ref_levels = 0;

    const char *mesh_file = "../Meshes/ushape.msh";

    Mesh mesh_serial(mesh_file, 1, 1);
    int dim = mesh_serial.Dimension();

    for (int i = 0; i < ref_levels; i++) { mesh_serial.UniformRefinement(); }

    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
    mesh_serial.Clear();

    for (int i = 0; i < par_ref_levels; i++) { mesh.UniformRefinement(); }

    mesh.SetCurvature(order);
    mesh.EnsureNodes();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace fespace(&mesh, fec);

    // ================= FREE SURFACE SUBMESH =================
    // IMPORTANT: this must include the generation + absorption top patches.
    // Attributes: 5 Zh_Straight, 6 Zh_UBend, 7 Zh_Generation, 8 Zh_Absorption
    Array<int> bdr_attr;
    bdr_attr.Append(5);
    bdr_attr.Append(6);
    bdr_attr.Append(7);
    bdr_attr.Append(8);
    ParSubMesh mesh_fs = ParSubMesh::CreateFromBoundary(mesh, bdr_attr);
    int dim_fs = mesh_fs.Dimension();

    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);
    ParFiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

    // ========== WAVE PARAMETERS =========
    double H = 0.01;
    double g = 9.81;

    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);

    double h  = bbmax(2) - bbmin(2);
    const double Lx = bbmax(0) - bbmin(0);

    const double lambda = 1.0;
    const double k      = 2.0*M_PI / lambda;
    const double kh     = 1.0;
    const double ky = 0.0;
    const double cwave = sqrt((g / k) * tanh(kh));
    const double T_input     = lambda / cwave;
    const double omega = 2.0 * M_PI / T_input;

    double t_final = 5 * T_input;

    if (myid == 0)
    {
        cout << "Wave parameters:\n";
        cout << "  Lx     = " << Lx << "\n";
        cout << "  lwave  = " << lambda << "\n";
        cout << "  kh     = " << kh << "\n";
        cout << "  k      = " << k << "\n";
        cout << "  cwave  = " << cwave << "\n";
        cout << "  T      = " << T_input << "\n";
        cout << "  omega  = " << omega << "\n";
        cout << "  H      = " << H << "\n";
    }

    // ========= TIME PARAMETERS =========
    double t = 0.0;

    int nsteps = 180;
    double dt = t_final / nsteps;
    cout << dt << endl;

    // ========== INITIAL CONDITIONS ========
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

    int fe_true = fespace_fs.GetTrueVSize();
    Array<int> fe_offset(3);
    fe_offset[0] = 0;
    fe_offset[1] = fe_true;
    fe_offset[2] = 2 * fe_true;
    BlockVector eta_phi_fs(fe_offset);

    ParGridFunction eta(&fespace_fs);
    ParGridFunction phi_fs(&fespace_fs);

    eta.ProjectCoefficient(eta_init);
    phi_fs.ProjectCoefficient(phi_fs_init);

    eta.GetTrueDofs(eta_phi_fs.GetBlock(0));
    phi_fs.GetTrueDofs(eta_phi_fs.GetBlock(1));

    ParGridFunction phi(&fespace);
    phi = 0.0;

    mesh_fs.Transfer(phi_fs, phi);

    // ================= ESSENTIAL BCs (top/free-surface boundary on volume mesh) =================
    Array<int> essential_bdr(mesh.bdr_attributes.Max());
    essential_bdr = 0;
    essential_bdr[5 - 1] = 1;
    essential_bdr[6 - 1] = 1;
    essential_bdr[7 - 1] = 1;
    essential_bdr[8 - 1] = 1;

    Array<int> ess_tdof;
    fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);

    // ==================== RELAXATION FUNCTIONS Cgen and Cabs (ON fespace_fs) ====================
    ParGridFunction Cgen_gf(&fespace_fs);
    ParGridFunction Cabs_gf(&fespace_fs);
    Cgen_gf = 0.0;
    Cabs_gf = 0.0;

    // On mesh_fs, the patch IDs are element attributes.
    int max_attr_fs = 0;
    if (mesh_fs.attributes.Size()) { max_attr_fs = mesh_fs.attributes.Max(); }
    max_attr_fs = std::max(max_attr_fs, 8);

    Vector mask_gen_vals(max_attr_fs);
    Vector mask_abs_vals(max_attr_fs);

    // --- attribute ids on mesh_fs might be preserved (5,6,7,8) OR renumbered (1,2,3,4) ---
    int gen_attr = 7;
    int abs_attr = 8;

    // If MFEM renumbered attributes on the submesh, they will typically become 1..bdr_attr.Size()
    // in the same order as bdr_attr = {5,6,7,8} -> {1,2,3,4}
    if (mesh_fs.attributes.Find(7) < 0 && mesh_fs.attributes.Find(3) >= 0) { gen_attr = 3; }
    if (mesh_fs.attributes.Find(8) < 0 && mesh_fs.attributes.Find(4) >= 0) { abs_attr = 4; }

    // (optional) print once
    if (myid == 0)
    {
        cout << "mesh_fs attributes: ";
        mesh_fs.attributes.Print(cout);
        cout << "Using gen_attr=" << gen_attr << ", abs_attr=" << abs_attr << endl;
    }

    mask_gen_vals = 0.0;
    mask_abs_vals = 0.0;

    mask_gen_vals[gen_attr - 1] = 1.0;
    mask_abs_vals[abs_attr - 1] = 1.0;

    PWConstCoefficient mask_gen(mask_gen_vals);
    PWConstCoefficient mask_abs(mask_abs_vals);

    // ---- your original generation ramp (kept) ----
    const double Ng  = 3;
    const double xg0 = bbmin(0);
    const double xg1 = xg0 + Ng * lambda;

    FunctionCoefficient Cgen_coef([&](const Vector &X)
    {
        const double x = X(0);
        if (x <= xg0) return 1.0;
        if (x >= xg1) return 0.0;
        const double xi = (x - xg0) / (xg1 - xg0);
        return 1 - (-2.0 * xi*xi*xi + 3.0 * xi*xi);
    });

    // ---- FIX 2: absorption on the ENTIRE outlet leg (not only the tiny absorption patch) ----
    // outlet leg is the y>0 straight section in this U-bend parameterization.
    const double Ns = 3.0;
    const double x1 = bbmax(0);
    const double x0 = x1 - Ns * lambda;
    const double p  = 5.0;

    FunctionCoefficient Cabs_coef([&](const Vector &X)
    {
        const double x = X(0);
        const double y = X(1);

        // Only apply absorption on outlet leg
        if (y <= 0.0) return 0.0;

        // Ramp from 0 -> 1 over the last Ns*lambda of the outlet leg
        if (x <= x0) return 0.0;
        if (x >= x1) return 1.0;

        const double xi = (x - x0) / (x1 - x0);
        return pow(xi, p);
    });

    ProductCoefficient Cgen_masked(mask_gen, Cgen_coef);

    Cgen_gf.ProjectCoefficient(Cgen_masked);
    Cabs_gf.ProjectCoefficient(Cabs_coef);

    // ==================== ParaView output (free-surface + relaxation functions) ====================
    ParaViewDataCollection pv_fs("u-shape-half", &mesh_fs);
    pv_fs.SetPrefixPath("ParaView");
    pv_fs.SetLevelsOfDetail(5*order);
    pv_fs.SetDataFormat(VTKFormat::BINARY);
    pv_fs.SetHighOrderOutput(true);
    pv_fs.RegisterField("eta", &eta);
    pv_fs.RegisterField("Cgen", &Cgen_gf);
    pv_fs.RegisterField("Cabs", &Cabs_gf);

    // ============== TIME INTEGRATION =============
    double tau = dt;

    ODESolver *ode_solver = new RK4Solver();
    rhs_linear surface(&fespace_fs, &fespace,
                       phi, g,
                       ess_tdof,
                       mesh_fs,
                       Cgen_gf, Cabs_gf,
                       H, omega, k,
                       kx_dir, ky_dir,
                       cwave, kh,
                       T_input, tau);

    ode_solver->Init(surface);

    if (myid == 0)
    {
        cout << "Running on " << num_procs << " processes" << endl;
        cout << "Starting time integration with " << nsteps << " steps" << endl;
    }

    for (int step = 0; step < nsteps + 1; step++)
    {
        ode_solver->Step(eta_phi_fs, t, dt);

        eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));

        if (myid == 0 && step % 10 == 0)
        {
            cout << "Step " << step << " / " << nsteps << ", t = " << t << endl;
        }

        pv_fs.SetCycle(step);
        pv_fs.SetTime(t);
        pv_fs.Save();
    }

    delete fec;
    delete fec_fs;
    delete ode_solver;

    return 0;
}
