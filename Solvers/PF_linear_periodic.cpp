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
    GridFunction &phi;
    double g;
    Array<int> ess_tdof;
    FiniteElementSpace &fespace_fs;
    FiniteElementSpace &fespace;
    SubMesh &mesh_fs;

    mutable GridFunction eta_stage_gf;
    mutable GridFunction phi_fs_stage_gf;
    mutable GridFunction w;
    mutable GridFunction w_tilde;

public:
    rhs_linear(FiniteElementSpace *fes_fs,
               FiniteElementSpace *fes_vol,
               GridFunction &phi_in,
               double g_in,
               const Array<int> &ess_tdof_in,
               SubMesh &mesh_fs_in)
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
    }

    void Mult(const Vector &eta_phifs_true,
              Vector &d_eta_phifs_true_dt) const override
    {
        const int Nt = fespace_fs.GetTrueVSize();
        d_eta_phifs_true_dt.SetSize(2 * Nt);

        // Split TRUE-DOF state
        Vector eta_true(const_cast<double*>(eta_phifs_true.GetData()), Nt);
        Vector phi_fs_true(const_cast<double*>(eta_phifs_true.GetData()) + Nt, Nt);

        Vector deta_true_dt(d_eta_phifs_true_dt.GetData(), Nt);
        Vector dphi_fs_true_dt(d_eta_phifs_true_dt.GetData() + Nt, Nt);

        // Stage TRUE DOFs -> stage GridFunctions (VDOFs internally)
        eta_stage_gf.SetFromTrueDofs(eta_true);
        phi_fs_stage_gf.SetFromTrueDofs(phi_fs_true);

        // Impose Dirichlet BC (phi_fs) on free surface
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


        // ============= FREE SURFACE BOUNDARY CONDITIONS ============

        // ===== w_tilde =====
        phi.GetDerivative(1, 2, w);  // dphi/dz into w

        mesh_fs.Transfer(w, w_tilde);

        // Return derivatives in TRUE DOFs
        w_tilde.GetTrueDofs(deta_true_dt);

        // d(phi_fs)/dt = -g * eta (TRUE DOFs)
        dphi_fs_true_dt = eta_true;
        dphi_fs_true_dt *= -g;
    }
};


int main()
{

    // ========== MESH etc. ===========
    int order = 2;
    int ref_levels = 0;

    const char *mesh_file = "../Meshes/wave-tank.mesh";
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

    // Shift mesh so that top (free surface) is at z = 0
    // Vector bbmin, bbmax;
    // mesh.GetBoundingBox(bbmin, bbmax);
    // const double z_shift = bbmax(2);

    // VectorFunctionCoefficient deform(3, [&](const Vector &x, Vector &y)
    // {
    // y = x;
    // y(2) = x(2) - z_shift;
    // });

    // mesh.Transform(deform);


    // ========== Wave Parameters ===========
    double H = 0.05;
    double g = 9.81;

    Vector bbmin, bbmax;
    mesh.GetBoundingBox(bbmin, bbmax);

    double Lx = bbmax(0) - bbmin(0);
    double Ly = bbmax(1) - bbmin(1);
    double h  = bbmax(2) - bbmin(2);

    double m = 2.0;
    double k  = m * 2.0 * M_PI / Lx;
    double kh = k * h;

    double omega = sqrt(g * k * tanh(kh));
    double T     = 2.0 * M_PI / omega;
    double cwave = omega / k;

    cout << "T = " << T << endl;
    

    // --==========--- Initial conditions ==========
    double theta = 0.0;
    double kx_dir = cos(theta);
    double ky_dir = sin(theta);

    auto phase = [&](const Vector &X)
    {
        return omega * 0 - k * (kx_dir * X(0) + ky_dir * X(1)); //t=0
    };

    FunctionCoefficient eta_init([&](const Vector &X)
    {
        return 0.5 * H * cos(phase(X));
    });

    FunctionCoefficient phi_fs_init([&](const Vector &X)
    {
        return -0.5 * H * cwave * cosh(kh)/sinh(kh) * sin(phase(X));
    });


    // ========== TIME PARAMETERS ===========
    double t = 0.0;    // start time
    double t_final = T; // final time
    int nsteps = 500;
    double dt = t_final / nsteps;


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

    // GridFunctions for ICs + visualization (VDOFs)
    GridFunction eta(&fespace_fs);
    GridFunction phi_fs(&fespace_fs);


    eta.ProjectCoefficient(eta_init);
    phi_fs.ProjectCoefficient(phi_fs_init);

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
    essential_bdr[2-1] = 1;

    Array<int> ess_tdof;
    fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);


    // ========  START PLOTTING ==========
    // ----- GLVis (volume phi) -----
    socketstream vol1("localhost", 19916);
    if (!vol1) { cerr << "Forgot to start GLVis\n"; }
    vol1.precision(8);
    vol1 << "solution\n" << mesh << phi
         << "window_title 'phi (volume)'\n"
         << "keys mm\n"
         << flush;

    // ----- GLVis (free-surface elevation eta) -----
    socketstream fs_eta("localhost", 19916);
    if (!fs_eta) { cerr << "Forgot to start GLVis\n"; }
    fs_eta.precision(8);
    fs_eta << "solution\n" << mesh_fs << eta
           << "window_title 'eta (free surface elevation)'\n"
           << "keys mm\n"
           << flush;

    
    // ============ TIME INTEGRATION ===========
    ODESolver *ode_solver = new RK4Solver();

    // ----- Free-surface ODE RHS (no reused matrices) -----
    rhs_linear surface(&fespace_fs, &fespace,
                       phi, g,
                       ess_tdof,
                       mesh_fs);

    ode_solver->Init(surface);


    for (int step = 0; step < nsteps + 1; step++)
    {
        ode_solver->Step(eta_phi_fs, t, dt);

        // Update visualization GridFunction eta from TRUE DOFs
        eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));

        if (vol1)   { vol1   << "solution\n" << mesh    << phi << flush; }
        if (fs_eta) { fs_eta << "solution\n" << mesh_fs << eta << flush; }
    }

    delete fec;
    delete fec_fs;
    delete ode_solver;

    return 0;
}
