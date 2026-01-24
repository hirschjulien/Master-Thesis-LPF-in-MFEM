#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace mfem;

// Solver for Linear Boundary Conditions on finite domain with relaxation zones:
// d_eta / dt = w_tilde
// d_phi_fs / dt = -g * eta

// ==================== DISPERSION SOLVER (TRANSLATED FROM MATLAB) ====================
static inline double coth_safe(double x)
{
    const double eps = 1e-12;
    x = max(x, eps);
    return cosh(x) / sinh(x);
}

static inline double DispersionRelationWaves(double g, double T, double h, int n)
{
    const double w = 2.0 * M_PI / T;
    double kh = (w*w) * h / g;         // initial guess
    kh = max(kh, 1e-8);

    for (int i = 0; i < n; ++i)
    {
        kh = sqrt((w*w/g) * h * kh * coth_safe(kh));
        kh = max(kh, 1e-8);
    }
    return kh;
}

// ==================== RHS. OPERATOR WITH EMBEDDED PENALTY FORCING ====================
//
// Implements:
//   ∂t g = N(g) + (1-Γ(x))/τ (g_e(t,x) - g(t,x)), x in relaxation zone
//
// Here g = (eta, phi_fs).
//      N(g) is the rhs operator
//
// Implement two zones:
// - Generation (left): drive to exact wave (eta_e, phi_e)
// - Absorption (right): drive to 0
//
// Using forcing weights:
//   Cgen(x) : 1 at inlet boundary -> 0 at end of gen zone
//   Cabs(x) : 0 at start of abs zone -> 1 at outlet boundary
//
// Time ramp (increases generating intensity over 5 periods as mentioned in Jens paper):
//  (Currently muted)
//   alpha(t) = min(1, t / (5T))

class rhs_linear : public TimeDependentOperator
{
private:
    GridFunction &phi;
    double g;
    Array<int> ess_tdof;
    FiniteElementSpace &fespace_fs;
    FiniteElementSpace &fespace;
    SubMesh &mesh_fs;

    // forcing weights on surface (VDOFs)
    GridFunction &Cgen_gf; // 1 at inlet -> 0 into domain
    GridFunction &Cabs_gf; // 0 interior -> 1 at outlet

    // wave params
    double H, omega, k, kx_dir, ky_dir, cwave, kh;
    double T;    // period (for ramp)
    double tau;  // penalty timescale (paper suggests ~ dt)

    // stage containers
    mutable GridFunction eta_stage_gf;
    mutable GridFunction phi_fs_stage_gf;
    mutable GridFunction w;
    mutable GridFunction w_tilde;

    // exact target wave (surface)
    mutable GridFunction eta_e;
    mutable GridFunction phi_fs_e;

    // RHS in VDOFs (for applying masks)
    mutable GridFunction deta_v;
    mutable GridFunction dphi_v;

public:
    rhs_linear(FiniteElementSpace *fes_fs,
               FiniteElementSpace *fes_vol,
               GridFunction &phi_in,
               double g_in,
               const Array<int> &ess_tdof_in,
               SubMesh &mesh_fs_in,
               GridFunction &Cgen_in,
               GridFunction &Cabs_in,
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
    }

    void Mult(const Vector &eta_phifs_true,
              Vector &d_eta_phifs_true_dt) const override
    {
        const int Nt = fespace_fs.GetTrueVSize();
        d_eta_phifs_true_dt.SetSize(2 * Nt);

        // Split TRUE-DOF state
        Vector eta_true (const_cast<double*>(eta_phifs_true.GetData()), Nt);
        Vector phfs_true(const_cast<double*>(eta_phifs_true.GetData()) + Nt, Nt);

        Vector deta_true_dt (d_eta_phifs_true_dt.GetData(), Nt);
        Vector dphfs_true_dt(d_eta_phifs_true_dt.GetData() + Nt, Nt);

        // TRUE -> stage GridFunctions (VDOFs)
        eta_stage_gf.SetFromTrueDofs(eta_true);
        phi_fs_stage_gf.SetFromTrueDofs(phfs_true);

        // impose Dirichlet on free surface
        mesh_fs.Transfer(phi_fs_stage_gf, phi);


        // =============== LAPLACE SOLVE =============
        BilinearForm a_loc(&fespace);
        a_loc.AddDomainIntegrator(new DiffusionIntegrator);
        a_loc.Assemble();

        LinearForm b_loc(&fespace);
        b_loc.Assemble();

        SparseMatrix A_loc;
        Vector X_loc, B_loc;

        a_loc.FormLinearSystem(ess_tdof, phi, b_loc, A_loc, X_loc, B_loc);

        GSSmoother M_loc(A_loc);
        PCG(A_loc, M_loc, B_loc, X_loc, 0, 400, 1e-24, 0.0);

        a_loc.RecoverFEMSolution(X_loc, b_loc, phi);

        phi.GetDerivative(1, 2, w);
        mesh_fs.Transfer(w, w_tilde);

        w_tilde.GetTrueDofs(deta_true_dt);

        dphfs_true_dt = eta_true;
        dphfs_true_dt *= -g;

 
        // ============= EMBEDDED PENALTY FORCING =========================

        const double t_stage = this->GetTime();

        // ramp on forcing over 5 periods
        //const double alpha = min(1.0, max(0.0, t_stage / (1.0 * T)));
        //const double alpha = 1.0;

        // If you want absorption ON from the start, use:
        //const double alpha_abs = 1.0;
        //const double alpha_abs = alpha;

        // Exact target wave at stage time
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

        // Convert RHS to VDOFs to add masked forcing there
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

        // Smooth time ramp for generation forcing
        // Ramps from 0 -> 1 over Tramp = n_ramp * T
        const double n_ramp = 3.0;              // number of periods to ramp over
        const double Tramp  = n_ramp * T;

        double alpha_gen = t_stage / Tramp;
        alpha_gen = min(1.0, max(0.0, alpha_gen));

        for (int i = 0; i < Nv; ++i)
        {
            const double cgen = Cgen[i]; // generation forcing weight (0..1)
            const double cabs = Cabs[i]; // absorption forcing weight (0..1)

            const double gen_weight = alpha_gen * cgen;

            deta[i] += (gen_weight * inv_tau) * (eta_ex[i] - eta_vdat[i]);
            dphi[i] += (gen_weight * inv_tau) * (phi_ex[i] - phi_vdat[i]);


            // Absorption: drive toward ZERO
            deta[i] += (cabs * inv_tau) * (0.0 - eta_vdat[i]);
            dphi[i] += (cabs * inv_tau) * (0.0 - phi_vdat[i]);
        }

        // VDOF RHS -> TRUE DOF outputs
        deta_v.GetTrueDofs(deta_true_dt);
        dphi_v.GetTrueDofs(dphfs_true_dt);
    }
};

// ==================== Main ====================

int main()
{
    int order = 5;
    int ref_levels = 1;

    const char *mesh_file = "../Meshes/wave-tank-finite.mesh";  // choose "cylinder_mesh" for cylinder case
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    FiniteElementSpace fespace(&mesh, fec);

    for (int i = 0; i < ref_levels; i++)
    {
        mesh.UniformRefinement();
        fespace.Update();
    }

    mesh.EnsureNodes();

    // ----- Free surface submesh (boundary attribute 2) -----
    Array<int> bdr_attr;
    bdr_attr.Append(2);
    SubMesh mesh_fs = SubMesh::CreateFromBoundary(mesh, bdr_attr);
    int dim_fs = mesh_fs.Dimension();

    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);
    FiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

    // ----- BlockVector state (TRUE dofs) -----
    int fe_true = fespace_fs.GetTrueVSize();
    Array<int> fe_offset(3);
    fe_offset[0] = 0;
    fe_offset[1] = fe_true;
    fe_offset[2] = 2 * fe_true;
    BlockVector eta_phi_fs(fe_offset);  // [eta_true; phi_fs_true]

    // GridFunctions for ICs + visualization
    GridFunction eta(&fespace_fs);
    GridFunction phi_fs(&fespace_fs);

    // ----- Physical parameters -----
    double H = 0.05;
    double g = 9.81;

    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);

    double Lx = bbmax(0) - bbmin(0);
    double Ly = bbmax(1) - bbmin(1);
    double h  = bbmax(2) - bbmin(2);

    // ================= Choose the wave by PERIOD T =================
    const double T_input = 1.13392/3;   // seconds
    const int    n_iter  = 40;

    const double omega = 2.0 * M_PI / T_input;
    const double kh    = DispersionRelationWaves(g, T_input, h, n_iter);
    const double k     = kh / h;
    const double cwave = omega / k;
    const double lambda = 2.0 * M_PI / k;

    cout << "Wave parameters:\n";
    cout << "  T     = " << T_input << "\n";
    cout << "  omega = " << omega << "\n";
    cout << "  h     = " << h << "\n";
    cout << "  kh    = " << kh << "\n";
    cout << "  k     = " << k << "\n";
    cout << "  lambda= " << lambda << "\n";
    cout << "  cwave = " << cwave << "\n";

    // ----- Time -----
    ODESolver *ode_solver = new RK4Solver();
    double t = 0.0;
    double t_final = 8.0 * T_input;  // run longer to observe absorption

    int nsteps = 800;
    double dt = t_final / nsteps;

    // ----- Initial conditions -----
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

    // eta = 0.0;
    // phi_fs = 0.0;

    // Put ICs into TRUE-DOF state vector
    eta.GetTrueDofs(eta_phi_fs.GetBlock(0));
    phi_fs.GetTrueDofs(eta_phi_fs.GetBlock(1));

    // ----- Volume potential -----
    GridFunction phi(&fespace);
    phi = 0.0;

    // Impose initial phi_fs on top boundary
    mesh_fs.Transfer(phi_fs, phi);

    // Essential TDofs (top boundary attr=2)
    Array<int> essential_bdr(mesh.bdr_attributes.Max());
    essential_bdr = 0;
    essential_bdr[2 - 1] = 1;

    Array<int> ess_tdof;
    fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);


    // ==================== RELAXATION FUNCTIONS Cgen and Cabs ====================

    // ============== GENERATION  ==============
    //(left) ZONE : Cgen = 1 at inlet -> 0 at xg1
    const double Ng  = 2.0;
    const double xg0 = bbmin(0);
    const double xg1 = xg0 + Ng * lambda;

    FunctionCoefficient Cgen_coef([&](const Vector &X)
    {
        const double x = X(0);
        if (x <= xg0) return 1.0;
        if (x >= xg1) return 0.0;
        const double xi = (x - xg0) / (xg1 - xg0); // 0..1 
        const double s = 1 - xi; // Change direction
        return -2.0 * s*s*s + 3.0 * s*s;
    });


    GridFunction Cgen_gf(&fespace_fs);
    Cgen_gf.ProjectCoefficient(Cgen_coef);

    // ========= ABSORPTION ZONE ======= 
    //(right) zone : Cabs = 0 at x0 -> 1 at outlet x1
    const double Ns = 2.0;
    const double x1 = bbmax(0);
    const double x0 = x1 - Ns * lambda;

      const double p = 5.0;

    FunctionCoefficient Cabs_coef([&](const Vector &X)
    {
        const double x = X(0);
        if (x <= x0) return 0.0;
        if (x >= x1) return 1.0;

        const double xi = (x - x0) / (x1 - x0);  // 0..1
        return pow(xi, p);               // 1 -> 0
    });


    GridFunction Cabs_gf(&fespace_fs);
    Cabs_gf.ProjectCoefficient(Cabs_coef);


    // TEST RELAXATION FUNCTIONS
        socketstream genw("localhost", 19916);
        socketstream absw("localhost", 19916);

        genw << "solution\n" << mesh_fs << Cgen_gf
        << "window_title 'Cgen (generation)'\n" << flush;

        absw << "solution\n" << mesh_fs << Cabs_gf
        << "window_title 'Cabs (absorption)'\n" << flush;

    
    // ============= TIME INTEGRATION ============
    // ----- Choose tau for embedded forcing -----
    double tau = dt;

    // ----- RHS operator with embedded forcing -----
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

    // ======== GLVis PLOTTING =========
    socketstream vol1("localhost", 19916);
    if (!vol1) { cerr << "Forgot to start GLVis\n"; }
    vol1.precision(8);

    socketstream fs_eta("localhost", 19916);
    if (!fs_eta) { cerr << "Forgot to start GLVis\n"; }
    fs_eta.precision(8);

    if (vol1)
    {
        vol1 << "solution\n" << mesh << phi
             << "window_title 'phi (volume)'\n"
             << "keys mm\n"   // once
             << flush;
    }
    if (fs_eta)
    {
        fs_eta << "solution\n" << mesh_fs << eta
               << "window_title 'eta (free surface elevation)'\n"
               << "keys mm\n"  // once
               << flush;
    }

    for (int step = 0; step < nsteps + 1; step++)
    {
        ode_solver->Step(eta_phi_fs, t, dt);

        // update visualization eta from state
        eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));

        // Shows last phi at each time step, even though it is updated at each stage
        if (vol1)   { vol1   << "solution\n" << mesh    << phi << flush; }
        if (fs_eta) { fs_eta << "solution\n" << mesh_fs << eta << flush; }
    }


    delete fec;
    delete fec_fs;
    delete ode_solver;

    return 0;
}
