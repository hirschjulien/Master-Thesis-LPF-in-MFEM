import numpy as np
import matplotlib.pyplot as plt

def load_and_gamma(fn):
    data = np.loadtxt(fn, comments="#")     # cols: par_ref_level, dofs, runtime[s]

    par_ref = data[:, 0].astype(int)
    dofs    = data[:, 1].astype(int)
    time_s  = data[:, 2]

    par_levels = np.unique(par_ref)
    nlevels    = len(par_levels)
    nprocs     = len(data) // nlevels
    procs      = np.arange(1, nprocs + 1)

    # reshape assuming ordering: for each proc-block: par_ref_level=0,1,2,...
    time_mat = time_s.reshape(nprocs, nlevels)   # (proc, level)
    dofs_mat = dofs.reshape(nprocs, nlevels)

    # strong scaling efficiency: gamma(p) = (T1*1)/(Tp*p)
    T1 = time_mat[0, :]
    gamma = (T1[None, :] * 1.0) / (time_mat * procs[:, None])

    return procs, par_levels, dofs_mat, gamma

# --- your two files ---
fn1 = "strong_scaling_p4.txt"
fn2 = "strong_scaling_p4big.txt"   # <- change to your second filename

procs1, par_levels1, dofs_mat1, gamma1 = load_and_gamma(fn1)
procs2, par_levels2, dofs_mat2, gamma2 = load_and_gamma(fn2)

# (optional) sanity check: same par levels & procs
if not (np.array_equal(procs1, procs2) and np.array_equal(par_levels1, par_levels2)):
    print("Warning: files don't have identical proc/par_ref_level layout. Plotting anyway.")

plt.figure(figsize=(6, 4))
markers = ["o", "v", "s", "^", "D"]
linestyles = ["-", "--"]

for j, lvl in enumerate(par_levels1):
    # file 1
    plt.plot(procs1, gamma1[:, j],
             marker=markers[j % len(markers)],
             linestyle=linestyles[0],
             linewidth=1.8,
             label=f"Mesh {lvl} (DoFs={dofs_mat1[0,j]})")

    # file 2
    plt.plot(procs2, gamma2[:, j],
             marker=markers[j % len(markers)],
             linestyle=linestyles[1],
             linewidth=1.8,
             label=f"Mesh {lvl} (DoFs={dofs_mat2[0,j]})")

plt.xlabel("MPI processes")
plt.ylabel(r"$\gamma_s$")
plt.ylim(0.0, 1.05)
plt.xlim(min(procs1.min(), procs2.min()), max(procs1.max(), procs2.max()))
plt.grid(True, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # Data file (3 par_ref_levels repeated for 1,2,3,4 MPI ranks in that order)
# fn = "strong_scaling_p4big.txt"  # :contentReference[oaicite:0]{index=0}
# data = np.loadtxt(fn, comments="#")     # cols: par_ref_level, dofs, runtime[s]

# par_ref = data[:, 0].astype(int)
# dofs    = data[:, 1].astype(int)
# time_s  = data[:, 2]

# par_levels = np.unique(par_ref)
# nlevels    = len(par_levels)
# nprocs     = len(data) // nlevels
# procs      = np.arange(1, nprocs + 1)

# # reshape assuming ordering: for each proc-block: par_ref_level=0,1,2
# time_mat = time_s.reshape(nprocs, nlevels)   # (proc, level)
# dofs_mat = dofs.reshape(nprocs, nlevels)

# # strong scaling efficiency: gamma(p) = (T1*1)/(Tp*p)
# T1 = time_mat[0, :]              # baseline at 1 proc, per level
# gamma = (T1[None, :] * 1.0) / (time_mat * procs[:, None])

# plt.figure(figsize=(6, 4))
# markers = ["o", "v", "s", "^", "D"]
# for j, lvl in enumerate(par_levels):
#     plt.plot(procs, gamma[:, j], marker=markers[j % len(markers)],
#              linewidth=1.8, label=f"Mesh {lvl} (DoFs={dofs_mat[0,j]})")

# plt.xlabel("MPI processes")
# plt.ylabel(r"$\gamma_s$")
# plt.ylim(0.0, 1.05)
# plt.xlim(procs.min(), procs.max())
# plt.grid(True, which="both")
# plt.legend()
# plt.tight_layout()
# plt.show()