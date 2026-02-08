#include "mfem.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace mfem;

// RUNNING THE FINAL SOLVER AND EXTRACTING THE VALUES ON THE CYLINDER BOUNDARY
// TO PLOT DIFFRACTION DIAGRAMS

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
    ParGridFunction &Cabsy_gf;

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
               ParGridFunction &Cabsy_in,
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
          Cabsy_gf(Cabsy_in),
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
        // Assemble operator matrix once
        a_loc_cach = new ParBilinearForm(&fespace);
        a_loc_cach->AddDomainIntegrator(new DiffusionIntegrator);
        a_loc_cach->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        a_loc_cach->Assemble();

        
        jacobi = new OperatorJacobiSmoother(*a_loc_cach, ess_tdof); //jacobi needs to know ess_tdof, so it can avoid those boundaries to be updated and therefore fits to the constrined operator A_loc
        
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

        // ParBilinearForm a_loc(&fespace);
        // a_loc.AddDomainIntegrator(new DiffusionIntegrator);
        // a_loc.Assemble();

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
        cg.SetMaxIter(2000);
        cg.Mult(B_loc, X_loc);

        a_loc_cach->RecoverFEMSolution(X_loc, b_loc, phi);

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
        const double *Cabsy = Cabsy_gf.GetData();

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
            deta[i] += (Cabsy[i] * inv_tau) * (0.0 - eta_vdat[i]);
            dphi[i] += (Cabsy[i] * inv_tau) * (0.0 - phi_vdat[i]);
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
    const char *mesh_file = "../Meshes/mesh_cylinder_half.msh"; //cylinder_num.msh
    const int order = 4;
    const double g = 9.81;

    const double cx = 4.0;  // center slighlyt moved to the generation zone, before 6
    const double cy = 4.0;
    const double a = 0.5;

    // const double lambda = 2.0;
    // const double kh = 1.0;

    // // Wave params
    // const double k     = 2.0 * M_PI / lambda;
    const double lambda = 1.0;      // L = 1
    const double k      = 2.0*M_PI; // k = 2π
    const double h      = 1.0/(2.0*M_PI);
    const double kh     = 1.0;

    const double H = 0.01;

    const double cwave = sqrt((g / k) * tanh(kh));
    const double T     = lambda / cwave;
    const double omega = 2.0 * M_PI / T;

    double t = 0.0;
    double t_final = 10 * T; // 35 steps per period give approx dt=0.026
    const int nsteps = 350;

     double dt = t_final / nsteps;
     cout << dt << endl;
     double t_last_start = t_final - T;
    // ------------------------------------------------------

    // Mesh
    Mesh mesh_serial(mesh_file, 1, 1);
    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
    //mesh_serial.Clear();
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

    const double Ng  = 2.5; // between 2 and 3
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

    const double Ns = 4.0;  // increase for cylinder
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

    // ---- Absorption zone in y-direction (towards y = bbmax(1)) ----
const double Ns_y = 3.0;          // thickness in multiples of lambda
const double y1   = bbmax(1);     // top boundary
const double y0   = y1 - Ns_y * lambda;
const double p_y  = 5.0;

FunctionCoefficient CabsY_coef([&](const Vector &X)
{
    const double y = X(1);
    if (y <= y0) return 0.0;
    if (y >= y1) return 1.0;
    const double xi = (y - y0) / (y1 - y0);
    return pow(xi, p_y);
});

ParGridFunction Cabsy_gf(&fespace_fs);
Cabsy_gf.ProjectCoefficient(CabsY_coef);


    ParaViewDataCollection pv_fs("cylinder_from_envelope_half-final25", &mesh_fs);
    pv_fs.SetPrefixPath("ParaView");
    pv_fs.SetLevelsOfDetail(5*order);
    pv_fs.SetDataFormat(VTKFormat::BINARY);
    pv_fs.SetHighOrderOutput(true);
    pv_fs.RegisterField("eta", &eta);

    // ODE operator + solver
    rhs_linear surface(&fespace_fs, &fespace,
                       phi, g, ess_tdof,
                       mesh_fs, Cgen_gf, Cabs_gf, Cabsy_gf,
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

        if (myid == 0 && step % 10 == 0)
        {
            cout << "step " << step << "/" << nsteps << " t=" << t << "\n";
        }

        if (step % 1 == 0)
        {
            pv_fs.SetCycle(step);
            pv_fs.SetTime(t);
            pv_fs.Save();
        }
    }

    eta_env *= (2.0 / H);



// ======== Extract cylinder boundary on free surface ========
    ParMesh mesh2(MPI_COMM_WORLD, mesh_serial);
    mesh2.EnsureNodes();
    mesh2.SetCurvature(order);
    int dim2 = mesh2.Dimension();
    FiniteElementCollection *fec2 = new H1_FECollection(order, dim2);
    ParFiniteElementSpace fespace2(&mesh2, fec2);
    ParGridFunction phi2(&fespace2);
    phi2 = 0.0;
    
    // Array<int> bdr_attr;
    // bdr_attr.Append(2);
    ParSubMesh mesh2_fs = ParSubMesh::CreateFromBoundary(mesh2, bdr_attr);
    int dim2_fs = mesh2_fs.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec2_fs = new H1_FECollection(order, dim2_fs);
    ParFiniteElementSpace fespace2_fs(&mesh2_fs, fec2_fs);
    //ParGridFunction eta2(&fespace2_fs);
    //eta.ProjectCoefficient(eta_env_exact);
    mesh2_fs.Transfer(eta_env, phi2);

    Array<int> bdr_attr2;
    bdr_attr2.Append(3);
    ParSubMesh mesh2_cyl = ParSubMesh::CreateFromBoundary(mesh2, bdr_attr2);
    int dim_cyl = mesh2_cyl.Dimension();   // free surface mesh is 2D but space dimension is still 3D
    FiniteElementCollection *fec2_cyl = new H1_FECollection(order, dim_cyl);
    ParFiniteElementSpace fespace2_cyl(&mesh2_cyl, fec2_cyl);
    ParGridFunction cyl(&fespace2_cyl);
    //cyl.ProjectCoefficient(eta_env_exact);
    mesh2_cyl.Transfer(phi2, cyl);

    // ========= Collect values for cylinder rim  only on 1 rank=======
// ========= Collect values for cylinder rim =======
    const double tol = 5e-3;
std::vector<std::pair<double,double>> cyl_eta;
Array<int> vdofs;
for (int v = 0; v < mesh2_cyl.GetNV(); v++)
{
const double *X = mesh2_cyl.GetVertex(v);
const double dx = X[0] - cx;
const double dy = X[1] - cy;
const double r  = std::sqrt(dx*dx + dy*dy);
if (std::abs(r - a) > tol) continue;
double theta = std::atan2(dy, dx);
// if (theta < 0.0) theta += 2.0*M_PI;
if (theta < 0.0) continue;
fespace2_cyl.GetVertexDofs(v, vdofs);
if (vdofs.Size() == 0) continue;
const double val = cyl(vdofs[0]);
if (val == 0.0) continue;   // <-- filter out zero values
cyl_eta.emplace_back(theta, cyl(vdofs[0]));
}

// for (int v = 0; v < mesh2_cyl.GetNV(); v++)
// {
//     const double *X = mesh2_cyl.GetVertex(v);

//     const double dx = X[0] - cx;
//     const double dy = X[1] - cy;
//     const double r  = std::sqrt(dx*dx + dy*dy);

//     if (std::abs(r - a) > tol) continue;

//     // ================= CURRENT VERSION (atan2) =================
//     // double theta = std::atan2(dy, dx);
//     // // if (theta < 0.0) theta += 2.0*M_PI;
//     // if (theta < 0.0) continue;

//     // ================= FIXED VERSION (acos -> always [0, pi]) =================
//     double c = dx / r;
//     c = std::max(-1.0, std::min(1.0, c)); // clamp to avoid NaNs from roundoff
//     double theta = std::acos(c);          // theta in [0, pi] by construction

//     // Optional: enforce upper half (kills any numerical leakage below centerline)
//     if (dy < -1e-10) continue;

//     fespace2_cyl.GetVertexDofs(v, vdofs);
//     if (vdofs.Size() == 0) continue;

//     const double val = cyl(vdofs[0]);

//     // ================= CURRENT VERSION (filters exact zeros) =================
//     // if (val == 0.0) continue;

//     // ================= SAFER FILTER (recommended) =================
//     if (!std::isfinite(val)) continue;

//     cyl_eta.emplace_back(theta, val);
// }

// ========= Gather cyl_eta from all ranks =========
{
    int local_n = (int)cyl_eta.size();
    std::vector<int> all_n(num_procs), offsets(num_procs, 0);
    MPI_Allgather(&local_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 1; i < num_procs; i++)
        offsets[i] = offsets[i-1] + all_n[i-1];
    int total_n = offsets.back() + all_n.back();

    std::vector<double> local_th(local_n), local_val(local_n);
    for (int i = 0; i < local_n; i++)
    {
        local_th[i]  = cyl_eta[i].first;
        local_val[i] = cyl_eta[i].second;
    }

    std::vector<double> all_th(total_n), all_val(total_n);
    MPI_Allgatherv(local_th.data(),  local_n, MPI_DOUBLE, all_th.data(),  all_n.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_val.data(), local_n, MPI_DOUBLE, all_val.data(), all_n.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    cyl_eta.resize(total_n);
    for (int i = 0; i < total_n; i++)
        cyl_eta[i] = {all_th[i], all_val[i]};
}

// sort by angle
std::sort(cyl_eta.begin(), cyl_eta.end(),
          [](const auto &a, const auto &b)
          { return a.first < b.first; });
// // remove single cyclic down-spike (handles seam at 0/2pi)
// for (size_t i = 0, n = cyl_eta.size(); n >= 3 && i < n; ++i)
// {
//     size_t im = (i + n - 1) % n, ip = (i + 1) % n;
//     if (cyl_eta[i].second < 0.25 * 0.5 * (cyl_eta[im].second + cyl_eta[ip].second))
//     { cyl_eta.erase(cyl_eta.begin() + i); break; }
// }
// write to file







if (myid == 0)
{
std::ofstream fout("data/cylinder-diffraction.txt");
fout << "# theta(rad)  eta\n";
double prev_th = -1.0;
for (auto &p : cyl_eta)
{
    if (p.first - prev_th < 1e-10) continue; // skip duplicates from shared vertices
    fout << p.first << " " << p.second << "\n";
    prev_th = p.first;
}
fout.close();
}


std::cout << "Extracted " << cyl_eta.size()
          << " points on cylinder rim\n";








//         const double cx  = 12.0;
// const double cy  = 8;
// const double a = 0.5;

    // ===== FE space on the FREE SURFACE (this is the key fix) =====
    // H1_FECollection fec_fs(order, dim_fs);
    // ParFiniteElementSpace fes_fs(&mesh_fs, &fec_fs);
    //ParGridFunction eta_env_fs(&fespace_fs);

    //mesh_fs.Transfer(eta_env, eta_env_fs);

    // GridFunction eta_env_fs(&fes_fs);
    // eta_env_fs.ProjectCoefficient(eta_env_exact);

    // Cylinder submesh from boundary attribute 3
    // Array<int> cyl_attr = {3};
    // Mesh cyl = SubMesh::CreateFromBoundary(mesh, cyl_attr);
    // cyl.EnsureNodes();
    // cyl.SetCurvature(order);

    // H1_FECollection fec_cyl(order, cyl.Dimension());
    // FiniteElementSpace fes_cyl(&cyl, &fec_cyl);

    // GridFunction eta_cyl(&fes_cyl);
    // SubMesh::Transfer(eta_env, cyl);

    // // Find z_max
    // double z_max = -1e300;
    // for (int v = 0; v < cyl.GetNV(); v++)
    //     z_max = max(z_max, cyl.GetVertex(v)[2]);

    // const double cx = 12.0, cy = 8.0;
    // const double ztol = 1e-6;

    // vector<pair<double,double>> rim; // (theta, eta)
    // Array<int> vdofs;

    // for (int v = 0; v < cyl.GetNV(); v++)
    // {
    //     const double *X = cyl.GetVertex(v);
    //     if (fabs(X[2] - z_max) > ztol) continue;

    //     fes_cyl.GetVertexDofs(v, vdofs);
    //     double th = atan2(X[1] - cy, X[0] - cx);
    //     if (th < 0.0) th += 2.0 * M_PI;
    //     rim.emplace_back(th, eta_cyl(vdofs[0]));
    // }

    // sort(rim.begin(), rim.end());

    // cout << "# theta(rad)  eta_env\n";
    // for (auto &[th, val] : rim)
    //     cout << th << "  " << val << "\n";

    //         // Write to file
    // std::ofstream fout("cylinder_diffraction.txt");
    // fout << "# theta(rad)  eta\n";
    // for (auto &p : rim)
    // {
    //     fout << p.first << " " << p.second << "\n";
    // }
    // fout.close();

//    // ===== Extract eta_env on cylinder rim (on FREE SURFACE) =====
// const double tol = 5e-3;
// const int Nb = 360; // angle bins (try 90/180/360)

// std::vector<double> sum(Nb, 0.0), cnt(Nb, 0.0);
// Array<int> vdofs;

// auto wrap_0_2pi = [](double th)
// {
//     const double twopi = 2.0*M_PI;
//     th = fmod(th, twopi);
//     return (th < 0.0) ? th + twopi : th;
// };

// for (int v = 0; v < mesh_fs.GetNV(); v++)
// {
//     const double *X = mesh_fs.GetVertex(v);
//     const double r = hypot(X[0]-cx, X[1]-cy);
//     if (fabs(r - a) > tol) continue;

//     double theta = wrap_0_2pi(atan2(X[1]-cy, X[0]-cx));
//     int b = (int) floor(Nb * theta / (2.0*M_PI));
//     b = std::min(std::max(b, 0), Nb-1);

//     fespace_fs.GetVertexDofs(v, vdofs);
//     const double eta = eta_env_fs(vdofs[0]);

//     sum[b] += eta;
//     cnt[b] += 1.0;
// }

// // Sorted, smooth curve (one point per bin)
// std::vector<std::pair<double,double>> cyl_eta;
// cyl_eta.reserve(Nb);

// for (int b = 0; b < Nb; b++)
// {
//     if (cnt[b] < 1) continue;
//     const double theta_c = (b + 0.5) * (2.0*M_PI / Nb);
//     cyl_eta.emplace_back(theta_c, sum[b] / cnt[b]);
// }

//     // Write to file
//     std::ofstream fout("cylinder_diffraction.txt");
//     fout << "# theta(rad)  eta\n";
//     for (auto &p : cyl_eta)
//     {
//         fout << p.first << " " << p.second << "\n";
//     }
//     fout.close();

// for (auto &p : cyl_eta)
//     std::cout << p.first << "  " << p.second << "\n";




// ========== KEEEEEEPPPP ======================================

    // ==================== ParaView output: envelope on free surface ====================
    ParaViewDataCollection pv_env("eta_envelope_final10-relax", &mesh_fs);
    pv_env.SetPrefixPath("ParaView");
    pv_env.SetLevelsOfDetail(5*order);
    pv_env.SetDataFormat(VTKFormat::BINARY);
    pv_env.SetHighOrderOutput(true);

    // register ONLY what you want to visualize
    //pv_env.RegisterField("eta", &eta);
    pv_env.RegisterField("eta_env", &eta_env);


    // single snapshot (envelope is time-independent after extraction)
    pv_env.SetCycle(0);
    pv_env.SetTime(t_final);
    pv_env.Save();


// =====================================================================



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
    delete fec2;
    delete fec2_fs;
    delete fec2_cyl;
    delete ode_solver;
    return 0;
}
