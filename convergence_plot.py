import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('convergence_data.txt', comments='#')
ndofs = data[:, 0]
errors = data[:, 1]

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.loglog(ndofs, errors, 'o-', label=f'L2 error, p-order=2')
plt.xlabel('DOF')
plt.ylabel('L2 error')
plt.title('Convergence of Laplace solver')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

