import numpy as np
import matplotlib.pyplot as plt

# Load datasets (theta [rad], eta)
diff_T25 = np.loadtxt("cylinder-diffraction-half-final10.txt", comments="#")
bnd      = np.loadtxt("cylinder_boundary.txt", comments="#")
exact    = np.loadtxt("cylinder-diffraction-half-final25.txt", comments="#")  # exact

TWOPI = 2*np.pi
PI = np.pi

def prep(arr, mirror_if_half=False, tol=1e-10):
    theta = np.mod(arr[:, 0], TWOPI)
    r = arr[:, 1]

    # sort first
    idx = np.argsort(theta)
    theta = theta[idx]
    r = r[idx]

    if mirror_if_half:
        # if it only reaches ~pi (common for half-circle outputs)
        if theta.max() <= PI + tol:
            # mirror into (pi, 2pi], avoid duplicating endpoints 0 and pi
            mask = (theta > tol) & (theta < PI - tol)
            theta_m = TWOPI - theta[mask]
            r_m     = r[mask]

            theta = np.concatenate([theta, theta_m])
            r     = np.concatenate([r, r_m])

            # re-sort after mirroring
            idx = np.argsort(theta)
            theta = theta[idx]
            r = r[idx]

    return theta, r

t1, r1 = prep(diff_T25, mirror_if_half=True)
t2, r2 = prep(bnd,      mirror_if_half=False)  # don't mirror boundary file
t3, r3 = prep(exact,    mirror_if_half=True)

plt.figure(figsize=(6.5, 6.5))
ax = plt.subplot(111, projection="polar")

ax.plot(t1, r1, marker="o", linewidth=1.5, markersize=3, label="10T")
ax.plot(t2, r2, linewidth=1.5, label="McCamy&Fuchs")
ax.plot(t3, r3, linestyle="--", linewidth=2.0, label="25T")

ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_title(r"$\eta(\theta)*(H/2)$", va="bottom")
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # Load datasets (theta [rad], eta)
# diff_T25 = np.loadtxt("cylinder-diffraction-half-final10.txt", comments="#")
# bnd      = np.loadtxt("cylinder_boundary.txt", comments="#")
# exact    = np.loadtxt("cylinder-diffraction-half-final25.txt", comments="#")  # exact

# def prep(arr):
#     theta = np.mod(arr[:, 0], 2*np.pi)
#     r = arr[:, 1]
#     idx = np.argsort(theta)
#     return theta[idx], r[idx]

# t1, r1 = prep(diff_T25)
# t2, r2 = prep(bnd)
# t3, r3 = prep(exact)

# plt.figure(figsize=(6.5, 6.5))
# ax = plt.subplot(111, projection="polar")

# ax.plot(t1, r1, marker="o", linewidth=1.5, markersize=3, label="T=10")
# ax.plot(t2, r2, linewidth=1.5, label="McCamy&Fuchs")
# ax.plot(t3, r3, linestyle="--", linewidth=2.0, label="T=25")

# ax.set_theta_zero_location("E")
# ax.set_theta_direction(1)
# ax.set_title(r"Circular phase plot: $\eta(\theta)$", va="bottom")
# ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

# plt.tight_layout()
# plt.show()
