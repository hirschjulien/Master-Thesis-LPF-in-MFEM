import numpy as np
import matplotlib.pyplot as plt

def load_and_speedup(fn):
    data = np.loadtxt(fn, comments="#")  # cols: par_ref_level, dofs, runtime[s]

    par_ref = data[:, 0].astype(int)
    dofs    = data[:, 1].astype(int)
    time_s  = data[:, 2]

    par_levels = np.unique(par_ref)
    nlevels    = len(par_levels)
    nprocs     = len(data) // nlevels
    procs      = np.arange(1, nprocs + 1)

    # ordering assumption: for each proc-block: par_ref_level=0,1,2,...
    time_mat = time_s.reshape(nprocs, nlevels)   # (proc, level)
    dofs_mat = dofs.reshape(nprocs, nlevels)

    # Speedup: S(p)=T1/Tp
    T1 = time_mat[0, :]
    speedup = T1[None, :] / time_mat

    return procs, par_levels, dofs_mat, speedup

# --- your two files ---
fn1 = "strong_scaling_p4.txt"
fn2 = "strong_scaling_p4big.txt"   # <- change to your second filename

procs1, par_levels1, dofs1, S1 = load_and_speedup(fn1)
procs2, par_levels2, dofs2, S2 = load_and_speedup(fn2)

if not (np.array_equal(procs1, procs2) and np.array_equal(par_levels1, par_levels2)):
    print("Warning: files don't have identical proc/par_ref_level layout. Plotting anyway.")

procs = procs1  # use first file's x-axis

plt.figure(figsize=(6, 4))

# Optimal line
plt.plot(procs, procs, linestyle=":", linewidth=2.0, label="Optimal")

markers = ["o", "v", "s", "^", "D", "P", "X"]
ls1, ls2 = "-", "--"

for j, lvl in enumerate(par_levels1):
    mk = markers[j % len(markers)]

    # file 1
    plt.plot(
        procs1, S1[:, j],
        marker=mk, linestyle=ls1, linewidth=1.8,
        label=f"Mesh {lvl} (DoFs={dofs1[0, j]})"
    )

    # file 2
    plt.plot(
        procs2, S2[:, j],
        marker=mk, linestyle=ls2, linewidth=1.8,
        label=f"Mesh {lvl} (DoFs={dofs2[0, j]})"
    )

plt.xlabel(r"$N_n$")          # or: "MPI processes"
plt.ylabel("Speedup [-]")

plt.xlim(procs.min(), procs.max())
plt.ylim(0.0, max(procs.max(), np.nanmax(S1), np.nanmax(S2)) * 1.05)

plt.grid(True, which="both")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
