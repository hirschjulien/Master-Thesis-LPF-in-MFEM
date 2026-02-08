import numpy as np
import matplotlib.pyplot as plt

# PLOT P CONVERGENCE
# data = np.loadtxt('laplace-pconv-phi.txt', comments='#')
# ndofs = data[:, 1]
# errors = data[:, 2]

# # --- Plot ---
# plt.figure(figsize=(6, 4))
# plt.loglog(ndofs, errors, 'o-', label=f'N=3')
# plt.xlabel('DOF')
# plt.ylabel(r"$\|\phi - \phi_h\|_{\infty}$")
# #plt.title('Runtime=T, dt=0.0061')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# PLOT H CONVERGENCE
data = np.loadtxt('laplace-hconv-phi.txt', comments='#')

orders = data[:, 0].astype(int)
ndofs  = data[:, 1]
errors = data[:, 3]

plt.figure(figsize=(6, 4))

for p in np.unique(orders):
    mask = orders == p
    plt.loglog(ndofs[mask], errors[mask], 'o-', label=f'p = {p}')

plt.xlabel('DOF')
plt.ylabel(r"$\|\phi - \phi_h\|_{\infty}$")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()