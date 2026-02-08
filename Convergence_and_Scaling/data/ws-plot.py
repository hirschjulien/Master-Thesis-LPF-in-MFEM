import numpy as np
import matplotlib.pyplot as plt

fn = "weak-scaling.txt"

data = np.loadtxt(fn, comments="#")

# Columns:
# 0 mode (0=strong,1=weak)
# 1 order
# 2 par_ref_level
# 3 ranks
# 4 dofs
# 5 runtime[s]
mode  = data[:, 0].astype(int)
p     = data[:, 1].astype(int)
ranks = data[:, 3].astype(int)
dofs  = data[:, 4].astype(int)
time  = data[:, 5].astype(float)

# Keep only weak scaling rows
m = (mode == 1)
p, ranks, dofs, time = p[m], ranks[m], dofs[m], time[m]

# Filtering the average of repeated runs in the bash script 
orders = np.unique(p)
rank_levels = np.unique(ranks)

time_med = {pp: [] for pp in orders}
dofs_med = {pp: [] for pp in orders}

for pp in orders:
    for rr in rank_levels:
        idx = (p == pp) & (ranks == rr)
        if np.any(idx):
            time_med[pp].append(np.median(time[idx]))
            dofs_med[pp].append(int(np.median(dofs[idx])))
        else:
            time_med[pp].append(np.nan)
            dofs_med[pp].append(np.nan)

for pp in orders:
    time_med[pp] = np.array(time_med[pp], dtype=float)
    dofs_med[pp] = np.array(dofs_med[pp], dtype=float)


# Weak scaling runtime plot
plt.figure(figsize=(6.5, 4.8))
for pp in orders:
    plt.plot(rank_levels, time_med[pp], marker="o", linewidth=2, label=fr"$p={pp}$")

plt.xlabel("MPI ranks")
plt.ylabel("Runtime [s]")
plt.xticks(rank_levels, rank_levels)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.title("Weak scaling: runtime")
plt.tight_layout()
plt.show()


# Weak scaling efficiency plot
plt.figure(figsize=(6.5, 4.8))
for pp in orders:
    T1 = time_med[pp][rank_levels == 1][0]  # runtime at 1 rank
    E  = T1 / time_med[pp]
    plt.plot(rank_levels, E, marker="o", linewidth=2, label=fr"$p={pp}$")

plt.xlabel("MPI ranks")
plt.ylabel("Weak scaling efficiency")
plt.ylim(0.0, 1.05)
plt.xticks(rank_levels, rank_levels)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.title("Weak scaling: efficiency")
plt.tight_layout()
plt.show()

