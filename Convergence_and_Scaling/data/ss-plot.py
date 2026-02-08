import numpy as np
import matplotlib.pyplot as plt

fn = "strong-scaling.txt"   # change if needed

# columns:
# mode  order  par_ref_level  ranks  dofs  runtime[s]
data = np.loadtxt(fn, comments="#")

mode  = data[:, 0].astype(int)
order = data[:, 1].astype(int)
pref  = data[:, 2].astype(int)
ranks = data[:, 3].astype(int)
dofs  = data[:, 4].astype(int)
time  = data[:, 5]

# keep only strong scaling rows
mask = (mode == 0)
order, pref, ranks, dofs, time = order[mask], pref[mask], ranks[mask], dofs[mask], time[mask]

# take MIN runtime across repeats for identical (order, pref, ranks)
key = np.stack([order, pref, ranks], axis=1)
uniq, inv = np.unique(key, axis=0, return_inverse=True)

tmin = np.full(len(uniq), np.inf)
drep = np.zeros(len(uniq), dtype=int)
for i in range(len(time)):
    j = inv[i]
    if time[i] < tmin[j]:
        tmin[j] = time[i]
        drep[j] = dofs[i]

U_order = uniq[:, 0].astype(int)
U_pref  = uniq[:, 1].astype(int)
U_ranks = uniq[:, 2].astype(int)

# ====== Speedup plot ======
plt.figure(figsize=(6.0, 4.2))

# "Optimal" line: S(p)=p
rmax = int(U_ranks.max())
rr = np.arange(1, rmax + 1)
plt.plot(rr, rr, ":", linewidth=2.0, label="Optimal")

markers = {3: "o", 4: "v", 5: "s"}

for p in sorted(np.unique(U_order)):
    for m in sorted(np.unique(U_pref)):
        sel = (U_order == p) & (U_pref == m)
        if not np.any(sel):
            continue

        r = U_ranks[sel]
        t = tmin[sel]
        d = drep[sel]

        # sort by rank
        idx = np.argsort(r)
        r, t, d = r[idx], t[idx], d[idx]

        # need a baseline at rank 1
        if 1 not in r:
            continue
        T1 = t[r == 1][0]
        S  = T1 / t

        plt.plot(
            r, S,
            marker=markers.get(p, "o"),
            linewidth=1.8,
            label=fr"$p={p}$, DoFs={d[r==1][0]})"
        )

plt.xlabel(r"$N_n$")
plt.ylabel("Speedup [-]")
plt.grid(True, which="both")
plt.xlim(1, rmax)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ====== efficiency plot ======
plt.figure(figsize=(6.0, 4.2))

for p in sorted(np.unique(U_order)):
    for m in sorted(np.unique(U_pref)):
        sel = (U_order == p) & (U_pref == m)
        if not np.any(sel):
            continue

        r = U_ranks[sel]
        t = tmin[sel]

        idx = np.argsort(r)
        r, t = r[idx], t[idx]

        if 1 not in r:
            continue
        T1 = t[r == 1][0]
        gamma = (T1 / t) / r

        plt.plot(r, gamma, marker=markers.get(p, "o"), linewidth=1.8,
                 label=fr"$p={p}$")

plt.xlabel(r"$N_n$")
plt.ylabel(r"$\gamma_s$")
plt.ylim(0.0, 1.05)
plt.grid(True, which="both")
plt.xlim(1, rmax)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
