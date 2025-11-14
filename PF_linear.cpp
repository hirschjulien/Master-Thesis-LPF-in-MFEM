#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// try with 2D mesh
// check all steps


// Solver for Linear Boundary Conditions:
// d_eta / dt = w_tilde
// d_phi_fs / dt = -g * eta

// Steps:
    // 1. Load Mesh
    // 2. Create SubMesh
    // 3. Define initial phi_eta and define initial eta 
    // 4. start time loop from 0 to t_end
    // 5. update eta
    // 6. impose phi_eta as Dirichlet condition on top boundary
    // 7. Use laplace_solver to solve for velocity potential phi
    // 8. visualize


// Class that defines the right hand side of the linear ode system
// describing the free surface boundary conditions

class rhs_linear : public TimeDependentOperator{
    private:
        const GridFunction &phi;
        double g;
        FiniteElementSpace &fespace_fs;
        FiniteElementSpace &fespace;
        SubMesh &mesh_fs;
        // mutable GridFunction eta_gf;

    public:
        rhs_linear(FiniteElementSpace *fes, FiniteElementSpace *fe, const GridFunction &phi, double g, SubMesh &mesh_fs) :
        TimeDependentOperator(2 * fes->GetTrueVSize()), phi(phi), g(g), fespace_fs(*fes), fespace(*fe), mesh_fs(mesh_fs) {}
        
        void Mult( const Vector &eta_phifs, Vector &d_eta_phifs_dt) const override
        {                   
            // Gridfunction for w (z direction of gradient of phi)
            GridFunction w(&fespace);   
            phi.GetDerivative(1, 2, w);

            // Transfer w_tilde (surface of gradient of phi in z-direction) from parent mesh to submesh
            GridFunction w_tilde(&fespace_fs);
            mesh_fs.Transfer(w, w_tilde);

            const int N = fespace_fs.GetTrueVSize();    // Size of the Surface ElementSpace
            d_eta_phifs_dt.SetSize(2*N);    // Ensure Size

            // Split input vector into eta and phi_fs
            Vector eta_vec(const_cast<double*>(eta_phifs.GetData()), N);    
            Vector phi_fs_vec(const_cast<double*>(eta_phifs.GetData()) + N, N);

            // Prepare output blocks
            Vector deta_dt(d_eta_phifs_dt.GetData(), N);
            Vector dphi_fs_dt(d_eta_phifs_dt.GetData() + N, N);

            //w_tilde.GetTrueDofs(deta_dt);   // I have d_eta/d_t = w_tilde 
            deta_dt = w_tilde;

            dphi_fs_dt = eta_vec;
            dphi_fs_dt *= -g;
        }
};


int main(){
    int order = 1;
    int ref_levels = 0;

    const char *mesh_file = "wave-tank.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    FiniteElementSpace fespace(&mesh, fec);

    // ----- 2. Create SubMesh -------
    Array<int> bdr_attr;
    bdr_attr.Append(2);
    SubMesh mesh_fs = SubMesh::CreateFromBoundary(mesh, bdr_attr);
    int dim_fs = mesh_fs.Dimension();

    FiniteElementCollection *fec_fs = new H1_FECollection(order, dim_fs);
    FiniteElementSpace fespace_fs(&mesh_fs, fec_fs);

    // CReate BlockVector for eta and phi_fs like in example 10 
    int fe_size = fespace_fs.GetTrueVSize();
    Array<int> fe_offset(3);
    fe_offset[0] = 0;
    fe_offset[1] = fe_size;
    fe_offset[2] = 2 * fe_size;
    BlockVector eta_phi_fs(fe_offset);
    //eta_phi_fs = 0.0;

    GridFunction eta, phi_fs;
    eta.MakeTRef(&fespace_fs, eta_phi_fs.GetBlock(0), 0);
    phi_fs.MakeTRef(&fespace_fs, eta_phi_fs.GetBlock(1), 0);

    mesh_fs.GetNodes(eta);
    mesh_fs.GetNodes(phi_fs);


    // ----- 3. Define initial phi_eta and initial eta ------
     //================= Initialize parameters =================
    // Wave in x direction
    double H = 0.05;   // wave height
    double ph = 0.0;   // phase
    double g = 9.81;
    
    // --- Compute wavelength and depth from the mesh ---
    Vector bbmin, bbmax; 
    mesh.GetBoundingBox(bbmin, bbmax);
    double Lx = bbmax(0) - bbmin(0);
    double Ly = bbmax(1) - bbmin(1);
    double h  = bbmax(2) - bbmin(2);   // since top is 0, bottom is -h
    double zmax = bbmax(2);

    double m = 1;     // number of wave periods in domain
    double k  = m * 2.0*M_PI / Lx;         
    double kh = k * h;    // by definition
    double omega = k * sqrt((g/k) * tanh(kh));
    //double T = 2.0 * M_PI / omega;
    double cs = sqrt(g/k * tanh(kh));
    
    //cout << (cs - omega/k) << "\n";     // check dispersion relation
                                            // My mesh is predefined, so k is already set, the wavelength has 
                                            // to fit the box so that periodicity stays correct

    
    // ------ 4. Start the time loop ------- https://en.ittrip.xyz/c-language/mfem-ode-c-hpc-pdes
    ODESolver *ode_solver = new RK4Solver();
    double t = 0.0;
    double t_final = 1.0;
    double dt = 0.001;

    // --- Initialize phi&eta ---
    FunctionCoefficient eta_init([&](const Vector& x){
        return H/2.0 * cos(omega*t - k*x(0) + ph);   
    });

    FunctionCoefficient phi_fs_init([&](const Vector& x){
        return 0.5*H*cs * cosh(kh)/sinh(kh) * sin(omega*t - k*x(0) + ph);
    });

    eta.ProjectCoefficient(eta_init);
    phi_fs.ProjectCoefficient(phi_fs_init);


    GridFunction phi(&fespace);
    phi = 0.0;

    //Test initial condition
        mesh_fs.Transfer(phi_fs, phi);
        // GridFunctionCoefficient phi_fs_bc(&phi); 

        Array<int> essential_bdr(mesh.bdr_attributes.Max());
        essential_bdr = 0;
        essential_bdr[2-1] = 1;
        //phi.ProjectBdrCoefficient(phi_fs_bc, essential_bdr);  // I already have boundary conditions on top from transferring the submesh over on the parentmesh!!!
                                                                // Projecting again messes up the solution in y-direction

        BilinearForm a(&fespace);
        a.AddDomainIntegrator(new DiffusionIntegrator);
        a.Assemble();

        LinearForm b(&fespace);
        b.Assemble();

        SparseMatrix A;
        Vector X, B;
        Array<int> ess_tdof;
        fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);
        a.FormLinearSystem(ess_tdof, phi, b, A, X, B);

        GSSmoother M(A);
        PCG(A, M, B, X, 0, 400, 1e-12, 0.0);
        a.RecoverFEMSolution(X, b, phi);

    // socketstream vol1("localhost", 19916);
    // vol1 << "solution\n" << mesh << phi << "window_title 'Initial Conditions imposed as Boundary Conditions on phi'\nkeys mm" << flush;

    socketstream vol1("localhost", 19916);
    if (!vol1)
    {
        cerr << "Unable to connect to GLVis server on port 19916.\n";
    }
    else
    {   
    vol1.precision(8);
    vol1 << "solution\n" << mesh << phi
         << "window_title 'Initial Conditions imposed as Boundary Conditions on phi'\n"
         << "keys mm\n"
         << flush;
    }

    // socketstream top("localhost", 19916);
    // top << "solution\n" << mesh_fs << phi_fs << "window_title 'Initial Condition for phi_fs on surface subMesh'\nkeys cm" << flush;

    // ----- Steps 4-7 ------
    rhs_linear surface(&fespace_fs, &fespace, phi, g, mesh_fs);    // Create linear Free Surface System
    ode_solver->Init(surface);  // Initialize time integration
   
    bool last_step = false;
    for(int ti = 1; !last_step; ti++){
        real_t dt_real = min(dt, t_final-t);

        ode_solver->Step(eta_phi_fs, t, dt);

        last_step = (t >= t_final - 1e-8*dt);

        // phi_fs is defined on the surface mesh. Transfer phi_fs to parent mesh
        mesh_fs.Transfer(phi_fs, phi);
        // GridFunctionCoefficient phi_fs_bc(&phi);    // Make a Coefficient out of the data on phi_fs, so I can project it as boundary condition

        Array<int> essential_bdr(mesh.bdr_attributes.Max());
        essential_bdr = 0;
        essential_bdr[2-1] = 1;
        //phi.ProjectBdrCoefficient(phi_fs_bc, essential_bdr);  // I already have boundary conditions on top from transferring the submesh over on the parentmesh!!!
                                                                // Projecting again messes up the solution in y-direction

        BilinearForm a(&fespace);
        a.AddDomainIntegrator(new DiffusionIntegrator);
        a.Assemble();

        LinearForm b(&fespace);
        b.Assemble();

        SparseMatrix A;
        Vector X, B;
        Array<int> ess_tdof;
        fespace.GetEssentialTrueDofs(essential_bdr, ess_tdof);
        a.FormLinearSystem(ess_tdof, phi, b, A, X, B);

        GSSmoother M(A);
        PCG(A, M, B, X, 0, 400, 1e-12, 0.0);
        a.RecoverFEMSolution(X, b, phi);

        vol1 << "solution\n" << mesh << phi << flush;
    }
    
    // Visualize
    // socketstream vol("localhost", 19916);
    // vol << "solution\n" << mesh << phi << "window_title 'phi after time stepping'\nkeys mm" << flush;

    delete fec;
    delete fec_fs;

    return 0;
}






// // Extract w in z direction
// GradientGridFunctionCoefficient w(&phi);    // Gradient of phi on whole phi

// VectorConstantCoefficient z(Vector({0,0,1}));   // Unit vector in z direction
// InnerProductCoefficient dz(w, z);   // Inner Product
// w_tilde.ProjectCoefficient(dz);     // d_eta/d_t = w_tilde




    // FunctionCoefficient phi_fs_init([&](const Vector& x){
    //     return 0.5*H*cs * cosh(kh)/sinh(kh) * sin(k*x(0) + ky*x(1) + ph);
    // });

    // function<double(const Vector&)> eta = [&](const Vector& x){
    //     return -0.5*H*cs * cosh(kh)/sinh(kh) * sin(-k*x(0));
    // };


        // Move columns by eta and plot the deformed mesh

    // double z_cut = 1.0;

    // mfem::VectorFunctionCoefficient lift(3,
    //     [&](const mfem::Vector &x, mfem::Vector &y)
    //     {
    //         y = x;
    //         if (x(2) >= z_cut) { y(2) += eta(x); } // lift only above the threshold
    //     });
    // mesh.Transform(lift);
    // mesh_fs.Transform(lift);

    // mesh.FinalizeTopology();
    // mesh.Finalize();