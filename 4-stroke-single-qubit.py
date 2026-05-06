"""
Four-Stroke Single-Qubit Quantum Measurement Engine

Extends the three-stroke engine by adding a pre-measurement unitary
stroke and replaces the optimised-feedback assumption of Yanik
et al. (2022) with a demon protocol conditioned on individual
measurement outcomes.

Reproduces Figs. 8-10 of the dissertation: work extraction, efficiency,
and coefficient of performance (COP) as functions of measurement strength
kappa.

All quantities are in natural units where kB = hbar = 1.
Energy and work are dimensionless, scaled by hbar*omega0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Constants
hbar_omega0 = 0.1
T = 1.0
gamma_prime = 0.05
delta_t = 2

# Pauli matrices
I = np.array([[1, 0],[0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# System Hamiltonian
H0 = hbar_omega0 * np.array([[0, 0], [0, 1]], dtype=complex)

# Initial thermal state
rho_unnorm = expm(-H0 / T)
Z = np.trace(rho_unnorm)
rho_th = rho_unnorm / Z

print("Initial thermal state:")
print(rho_th)
print(f"Initial z0 = {np.trace(rho_th @ sigma_z).real:.5f}")

# Pre-measurement unitary: additional stroke A->B
apply_pre_unitary = True

if apply_pre_unitary:
    prep_angle = np.pi/4  # 45-degree rotation
    U_prep = expm(-1j * prep_angle * sigma_x)
    rho_before_measure = U_prep @ rho_th @ U_prep.conj().T
    print(f"\nAfter preparation unitary (θ={prep_angle:.3f} rad):")
else:
    rho_before_measure = rho_th
    print("\nNo preparation unitary:")

print(rho_before_measure)
z0_before = np.trace(rho_before_measure @ sigma_z).real
print(f"z0 before measurement = {z0_before:.5f}")

#MEASUREMENT
kappa = 0.5 - np.sqrt(2 * gamma_prime * delta_t)
print(f"\nMeasurement strength κ = {kappa:.5f}")

# Measurement operators
M_plus = 0.5 * ((np.sqrt(kappa) + np.sqrt(1 - kappa)) * I +
                (np.sqrt(kappa) - np.sqrt(1 - kappa)) * sigma_x)
M_minus = 0.5 * ((np.sqrt(kappa) + np.sqrt(1 - kappa)) * I -
                 (np.sqrt(kappa) - np.sqrt(1 - kappa)) * sigma_x)

# Measurement probabilities
p_plus = np.trace(M_plus @ rho_before_measure @ M_plus.conj().T).real
p_minus = np.trace(M_minus @ rho_before_measure @ M_minus.conj().T).real

# Post-measurement states
rho_plus = M_plus @ rho_before_measure @ M_plus.conj().T / p_plus
rho_minus = M_minus @ rho_before_measure @ M_minus.conj().T / p_minus

print(f"\nMeasurement probabilities: p(+) = {p_plus:.5f}, p(-) = {p_minus:.5f}")
print("\nPost-measurement state for '+' outcome:")
print(rho_plus)
print("\nPost-measurement state for '-' outcome:")
print(rho_minus)

#FEEDBACK CONTROL

# Optimise feedback for maximum work extraction

def optimise_feedback(state):

    # Initial energy
    E0 = np.trace(state @ H0).real

    # Try different rotation angles to find max work
    best_W = -np.inf
    best_angle = 0
    best_axis = 'x'

    # Test rotations around x-axis
    for angle in np.linspace(0, 2*np.pi, 1000):
        U = expm(-1j * angle * sigma_x/2)
        rho_final = U @ state @ U.conj().T
        E_final = np.trace(rho_final @ H0).real
        W = E0 - E_final
        if W > best_W:
            best_W = W
            best_angle = angle
            best_axis = 'x'

    # Test rotations around y-axis
    for angle in np.linspace(0, 2*np.pi, 1000):
        U = expm(-1j * angle * sigma_y/2)
        rho_final = U @ state @ U.conj().T
        E_final = np.trace(rho_final @ H0).real
        W = E0 - E_final
        if W > best_W:
            best_W = W
            best_angle = angle
            best_axis = 'y'

    if best_axis == 'x':
        U_opt = expm(-1j * best_angle * sigma_x/2)
    else:
        U_opt = expm(-1j * best_angle * sigma_y/2)

    return U_opt, best_W

# Optimise for each outcome
U_opt_plus, W_opt_plus = optimise_feedback(rho_plus)
U_opt_minus, W_opt_minus = optimise_feedback(rho_minus)

rho_plus_opt = U_opt_plus @ rho_plus @ U_opt_plus.conj().T
rho_minus_opt = U_opt_minus @ rho_minus @ U_opt_minus.conj().T

W_avg_opt = p_plus * W_opt_plus + p_minus * W_opt_minus

print(f"  '+' branch: Optimal work = {W_opt_plus:.5f}")
print(f"  '-' branch: Optimal work = {W_opt_minus:.5f}")
print(f"  Average optimal work = {W_avg_opt:.5f}")

# Demon memory erasure
TD_values = [0.001, 0.005, 0.01]  # demon temperatures

print("DEMON MEMORY ERASURE COST")
for TD in TD_values:
    Wer = TD * np.log(2)  # Landauer's principle
    net_gain_opt = W_avg_opt - Wer

    print(f"\nDemon temperature TD = {TD}")
    print(f"  Erasure work = {Wer:.5f}")
    print(f"  Optimised net gain = {net_gain_opt:.5f}")


print("COMPARISON WITH NO FEEDBACK")
E_no_feedback = p_plus * np.trace(rho_plus @ H0).real + p_minus * np.trace(rho_minus @ H0).real
E_initial = np.trace(rho_before_measure @ H0).real
print(f"  Energy before measurement: {E_initial:.5f}")
print(f"  Energy after measurement (no feedback): {E_no_feedback:.5f}")
print(f"  No feedback: ΔE = {E_no_feedback - E_initial:.5f}")

# Sweep over kappa
kappa_range = np.linspace(0.01, 0.99, 1000)
W_opt_range = []

for k in kappa_range:
    # Measurement operators for this kappa
    Mp = 0.5 * ((np.sqrt(k) + np.sqrt(1 - k)) * I +
                (np.sqrt(k) - np.sqrt(1 - k)) * sigma_x)
    Mm = 0.5 * ((np.sqrt(k) + np.sqrt(1 - k)) * I -
                 (np.sqrt(k) - np.sqrt(1 - k)) * sigma_x)

    # Measurement probabilities and states
    pp = np.trace(Mp @ rho_before_measure @ Mp.conj().T).real
    pm = np.trace(Mm @ rho_before_measure @ Mm.conj().T).real
    rho_p = Mp @ rho_before_measure @ Mp.conj().T / pp
    rho_m = Mm @ rho_before_measure @ Mm.conj().T / pm

    # Optimised feedback
    _, W_p_opt = optimise_feedback(rho_p)
    _, W_m_opt = optimise_feedback(rho_m)
    W_opt = pp * W_p_opt + pm * W_m_opt

    W_opt_range.append(W_opt)

#PLOT

plt.plot(kappa_range, W_opt_range, color='#2171B5', linewidth=2.5)

plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$W_{\mathrm{ext}}$', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, alpha=0.3)

plt.savefig('demon_work.png', dpi=300)
plt.show()

# Final summary
print("MAXWELL'S DEMON PROTOCOL SUMMARY")
print(f"Initial state preparation: {'Yes' if apply_pre_unitary else 'No'}")
print(f"Measurement strength: {kappa:.5f}")
print(f"Maximum work extracted: {W_avg_opt:.5f}")
print(f"Can beat erasure cost? {'Yes' if W_avg_opt > TD_values[0]*np.log(2) else 'No'}")

linestyles = ['-', '--', ':']
colors = ['#08306B', '#2171B5', '#6BAED6']

# Efficiency

fig1, ax1 = plt.subplots()
for i, TD in enumerate(TD_values):
    Wer = TD * np.log(2)
    eta_range = []
    for W in W_opt_range:
        if W + Wer > 0:
            eta = W / (W + Wer)
        else:
            eta = 0
        eta_range.append(eta)
    line, = ax1.plot(
        kappa_range,
        eta_range,
        linestyle=linestyles[i],
        color=colors[i],
        linewidth=2.5,
        label=f'$T_D = {TD}$'
    )

ax1.set_xlabel(r'$\kappa$', fontsize=14)
ax1.set_ylabel(r'$\eta$', fontsize=14)
ax1.tick_params(labelsize=14)
ax1.set_ylim(0,1.1)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig('demon_efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

# COP
fig2, ax2 = plt.subplots()
for i, TD in enumerate(TD_values):
    Wer = TD * np.log(2)
    cop_range = []
    for W in W_opt_range:
        # COP = Wext / Wer in the symmetric regime (Eq. 12)
        cop = W / Wer if Wer > 0 else 0
        cop_range.append(cop)
    ax2.plot(
        kappa_range,
        cop_range,
        linestyle=linestyles[i],
        color=colors[i],
        linewidth=2.5,
        label=f'$T_D = {TD}$'
    )

ax2.set_xlabel(r'$\kappa$', fontsize=14)
ax2.set_ylabel('COP', fontsize=14)
ax2.set_ylim(0,85)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, ncol=3, loc='upper center')
plt.tight_layout()
plt.savefig('demon_cop.png', dpi=300, bbox_inches='tight')
plt.show()

print("Maximum work in range =", np.max(W_opt_range))
print("Minimum work in range =", np.min(W_opt_range))

results = {}

for TD in TD_values:
    Wer = TD * np.log(2)

    eta_list = []
    cop_list = []

    for W in W_opt_range:
        # efficiency
        if W + Wer > 0:
            eta = W / (W + Wer)
        else:
            eta = 0

        # COP
        cop = W / Wer if Wer > 0 else 0

        eta_list.append(eta)
        cop_list.append(cop)

    results[TD] = {
        "eta_min": np.min(eta_list),
        "eta_max": np.max(eta_list),
        "cop_min": np.min(cop_list),
        "cop_max": np.max(cop_list),
    }


for TD, vals in results.items():
    print(f"\nT_D = {TD}")
    print(f"Efficiency: min = {vals['eta_min']:.5f}, max = {vals['eta_max']:.5f}")
    print(f"COP:        min = {vals['cop_min']:.5f}, max = {vals['cop_max']:.5f}")