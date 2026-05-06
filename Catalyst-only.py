"""
Four-Stroke Single-Qubit Catalyst-Only Engine

Replaces the Maxwell's demon feedback of the four-stroke engine with a
quantum catalyst interacting via an XY exchange Hamiltonian, allowing
catalysis to be assessed as an independent resource without measurement
interference.

Reproduces Figs. 11-12 of the dissertation: work extraction and catalyst
fidelity as functions of coupling strength g for ground, excited, and
superposition catalyst initial states.

All quantities are in natural units where kB = hbar = 1.
Energy and work are dimensionless, scaled by hbar*omega0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sqrtm

# Constants

hbar_omega0 = 0.1
T = 1.0
t_interaction = 1.0

# Pauli matrices

I = np.array([[1,0],[0,1]],dtype=complex)

sigma_x = np.array([[0,1],
                    [1,0]],dtype=complex)

sigma_y = np.array([[0,-1j],
                    [1j,0]],dtype=complex)

sigma_z = np.array([[1,0],
                    [0,-1]],dtype=complex)

sigma_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)

# System Hamiltonian

H_sys = hbar_omega0 * np.array([[0,0],
                                [0,1]],dtype=complex)

# Catalyst Hamiltonian

omega_c = hbar_omega0
H_cat = omega_c * np.array([[0,0],
                            [0,1]],dtype=complex)

# Initial thermal state

rho_unnorm = expm(-H_sys/T)
Z = np.trace(rho_unnorm)
rho_th = rho_unnorm/Z

print("Initial thermal state")
print(rho_th)

# Pre-measurement unitary

apply_pre_unitary = True

if apply_pre_unitary:

    prep_angle = np.pi/4
    U_prep = expm(-1j * prep_angle * sigma_x)

    rho_sys = U_prep @ rho_th @ U_prep.conj().T

else:

    rho_sys = rho_th

print("\nSystem state after preparation")
print(rho_sys)

# Catalyst initial states

cat_ground = np.array([[1],[0]],dtype=complex)
rho_cat_ground = cat_ground @ cat_ground.conj().T

cat_excited = np.array([[0],[1]],dtype=complex)
rho_cat_excited = cat_excited @ cat_excited.conj().T

theta_cat = np.pi/4
cat_super = np.array([[np.cos(theta_cat)],
                      [np.sin(theta_cat)]],dtype=complex)

rho_cat_super = cat_super @ cat_super.conj().T

cat_states = {
    "Ground":rho_cat_ground,
    "Excited":rho_cat_excited,
    "Superposition":rho_cat_super
}

# Energy
def energy(rho,H):

    return np.trace(rho @ H).real

# Fidelity

def fidelity(rho1,rho2):

    s = sqrtm(rho1)
    return np.trace(sqrtm(s @ rho2 @ s)).real**2

# Partial traces recover the reduced states of system and catalyst
def partial_trace_system(rho_joint):
    rho_sys = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for k in range(2):
            for j in range(2):
                rho_sys[i, k] += rho_joint[2*i+j, 2*k+j]
    return rho_sys

def partial_trace_catalyst(rho_joint):
    rho_cat = np.zeros((2, 2), dtype=complex)
    for j in range(2):
        for l in range(2):
            for i in range(2):
                rho_cat[j, l] += rho_joint[2*i+j, 2*i+l]
    return rho_cat

# Catalytic interaction

def catalytic_step(rho_sys, rho_cat, g):
    H_int = g * (np.kron(sigma_plus, sigma_minus) +
                 np.kron(sigma_minus, sigma_plus))
    H_tot = np.kron(H_sys, I) + np.kron(I, H_cat) + H_int
    U = expm(-1j * H_tot * t_interaction)
    rho_joint = U @ np.kron(rho_sys, rho_cat) @ U.conj().T
    return partial_trace_system(rho_joint), partial_trace_catalyst(rho_joint)

# Parameter sweep

g_range = np.linspace(0.1, 1.5, 60)

work_results = {}
fidelity_results = {}

E_initial = energy(rho_sys,H_sys)

for name,rho_cat in cat_states.items():

    W_list = []
    F_list = []

    for g in g_range:

        rho_sys_f,rho_cat_f = catalytic_step(rho_sys,rho_cat,g)

        E_final = energy(rho_sys_f,H_sys)

        W = E_initial - E_final

        F = fidelity(rho_cat,rho_cat_f)

        W_list.append(W)
        F_list.append(F)

    work_results[name] = W_list
    fidelity_results[name] = F_list

#PLOTS

linestyles = ['-', '--', '-.']
colors     = ['#08306B', '#2171B5', '#6BAED6']

plt.figure()
for i, name in enumerate(cat_states):
    plt.plot(g_range, work_results[name],
             linestyle=linestyles[i],
             color=colors[i],
             linewidth=2.5,
             label=name)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.xlabel('$g$', fontsize=14)
plt.ylabel(r'$W_{\mathrm{ext}}$', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cat_all_states_W.png', dpi=300)
plt.show()

# Find optimal catalyst

best_W = -np.inf
best_desc = ""

for name in cat_states:

    W_max = np.max(work_results[name])

    if W_max > best_W:

        best_W = W_max
        best_desc = name

print("CATALYTIC THERMODYNAMICS SUMMARY")

print(f"Best catalyst state: {best_desc}")
print(f"Maximum work extracted: {best_W:.5f}")

# PLOT

plt.plot(g_range, work_results["Ground"], color='#2171B5', linewidth=2)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlabel("$g$",fontsize=14)
plt.ylabel("$W_{\mathrm{ext}}$",fontsize=14)
plt.show()

plt.plot(g_range, fidelity_results["Ground"], linewidth=2, color='#2171B5')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlabel("$g$",fontsize=14)
plt.ylabel("$F$",fontsize=14)
plt.show()

print("\nValid regime summary (F >= 0.90, ground state):")
print(f"{'g':>6}  {'W_ext':>10}  {'F':>8}")
for g, W, F in zip(g_range,
                   work_results["Ground"],
                   fidelity_results["Ground"]):
    if F >= 0.90:
        print(f"{g:>6.3f}  {W:>10.5f}  {F:>8.4f}")

valid_points = [(g, W, F) for g, W, F in zip(
    g_range,
    work_results["Ground"],
    fidelity_results["Ground"]
) if F >= 0.90]

invalid_points = [(g, W, F) for g, W, F in zip(
    g_range,
    work_results["Ground"],
    fidelity_results["Ground"]
) if F < 0.90]

if valid_points:
    g_max, W_max, F_max = max(valid_points, key=lambda x: x[1])
    print(f"\nMax W_ext (valid): {W_max:.5f} at g = {g_max:.3f}, F = {F_max:.4f}")

if invalid_points:
    g_max_inv, W_max_inv, F_max_inv = max(invalid_points, key=lambda x: x[1])
    print(f"Max W_ext (invalid): {W_max_inv:.5f} at g = {g_max_inv:.3f}, F = {F_max_inv:.4f}")