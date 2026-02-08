#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

// Strong Scaling Experiment

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

    // cache the partially assembled bilinear form (operator) once
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
        // assemble PARTIAL bilinear form once here
        a_loc_cached = new ParBilinearForm(&fespace);
        a_loc_cached->AddDomainIntegrator(new DiffusionIntegrator);
        a_loc_cached->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        a_loc_cached->Assemble();

        jacobi = new OperatorJacobiSmoother(*a_loc_cached, ess_tdof);
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

        ParLinearForm b_loc(&fespace);
        b_loc.Assemble();

        OperatorPtr A_loc;
        Vector X_loc, B_loc;

        a_loc_cached->FormLinearSystem(ess_tdof, phi, b_loc, A_loc, X_loc, B_loc);

        CGSolver cg(MPI_COMM_WORLD);
        cg.SetPreconditioner(*jacobi);
        cg.SetOperator(*A_loc);
        cg.SetRelTol(1e-8);
        cg.SetAbsTol(0.0);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(300);
        cg.Mult(B_loc, X_loc);

        a_loc_cached->RecoverFEMSolution(X_loc, b_loc, phi);

        phi.GetDerivative(1, 2, w);
        mesh_fs.Transfer(w, w_tilde);

        w_tilde.GetTrueDofs(deta_true_dt);

        dphi_fs_true_dt = eta_true;
        dphi_fs_true_dt *= -g;
    }

    ~rhs_linear() override
    {
        delete jacobi;
        delete a_loc_cached;
    }
};

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    // ============================
    // mode = 0 -> strong scaling (fixed problem size)
    // mode = 1 -> weak scaling   (grow problem with ranks)
    // ============================
    const int mode = 1; // <-- set to 0 (strong) or 1 (weak)

    // Mesh / refinement controls
    int ref_levels = 0;
    const char *mesh_file = "../Meshes/wave-tank-big.mesh";

    // Strong scaling
    const int par_ref_levels_fixed = 1;

    // Weak scaling
    const int par_ref_base = 1;     
    const int dim_ref = 3;         

    const int nsteps = 10;

    ofstream fout;
    if (myid == 0)
    {
        fout.open("data/strong-scaling.txt", ios::app);
        fout << "# mode(0=strong,1=weak)  order  par_ref_level  ranks  dofs  runtime[s]\n";
        if (mode == 0) { cout << "Strong scaling test\n"; }
        else           { cout << "Weak scaling test\n"; }
        cout << "procs  order  par_ref  dofs  time_total[s]  time_per_step[s]\n";
        cout << "------------------------------------------------------------\n";
    }

    // wave parameters
    double H = 0.01;
    double g = 9.81;

    double t = 0.0;

    double theta = 0.0;
    double kx_dir = cos(theta);
    double ky_dir = sin(theta);

    const double lambda = 1.0;
    const double k      = 2.0*M_PI / lambda;
    const double kh     = 1.0;

    const double cwave = sqrt((g / k) * tanh(kh));
    const double T     = lambda / cwave;
    const double omega = 2.0 * M_PI / T;

    ODESolver *ode_solver = new RK4Solver();
    double dt = T / nsteps;

    // Decide par_ref_levels depending on scaling mode
    int par_ref_levels = par_ref_levels_fixed;
    if (mode == 1)
    {
        // weak scaling schedule based on ranks
        const double lp = log2((double)num_procs);
        const int add = (int) llround(lp / (double)dim_ref);
        par_ref_levels = par_ref_base + add;
    }

    // Run for order = 3 and 4
    for (int order : {3, 4})
    {
        // ---------------- build mesh and spaces ----------------
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

        ParGridFunction eta(&fespace_fs);
        ParGridFunction phi_fs(&fespace_fs);

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

        // warm-up one step (not timed)
        ode_solver->Step(eta_phi_fs, t, dt);
        MPI_Barrier(MPI_COMM_WORLD);

        // Measure time stepping only. Use max over ranks.
        const double t0 = MPI_Wtime();

        for (int step = 0; step < nsteps; step++)
        {
            ode_solver->Step(eta_phi_fs, t, dt);
            eta.SetFromTrueDofs(eta_phi_fs.GetBlock(0));
            phi_fs.SetFromTrueDofs(eta_phi_fs.GetBlock(1));
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();
        const double local_time = t1 - t0;

        double max_time = 0.0;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        int local_ndofs = fespace.GetTrueVSize();
        int global_ndofs = 0;
        MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (myid == 0)
        {
            const double time_per_step = max_time / nsteps;
            cout << num_procs << "  " << order << "  " << par_ref_levels
                 << "  " << global_ndofs << "  " << max_time << "  " << time_per_step << "\n";

            fout << mode << "  " << order << "  " << par_ref_levels << "  "
                 << num_procs << "  " << global_ndofs << "  " << max_time << "\n";
        }

        delete fec_fs;
        delete fec;
    }

    if (myid == 0) { fout.close(); }
    delete ode_solver;
    return 0;
}
