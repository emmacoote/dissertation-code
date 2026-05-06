"""
Three-Stroke Single-Qubit Quantum Measurement Engine

Reproduces Figs. 5–7 of the dissertation: work extraction, efficiency,
and coefficient of performance (COP) as functions of measurement strength
kappa, validating against Yanik et al. (2022).
 
All quantities are in natural units where kB = hbar = 1.
Energy and work are dimensionless, scaled by hbar*omega0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Constants
hbar_omega0 = 0.1 # in units of kBT
T = 1.0
gamma_prime = 0.05
delta_t = 2

# Pauli matrices
I = np.array([[1, 0],[0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

kappa = 0.5 - np.sqrt(2 * gamma_prime * delta_t)

# System Hamiltonian
H0 = hbar_omega0 * np.array([[0, 0], [0, 1]], dtype=complex)

# Initial thermal state
rho_unnorm = expm(-H0 / T)
Z = np.trace(rho_unnorm)
rho_th = rho_unnorm / Z

print("ρ_th =\n", rho_th)
print("Trace =", np.trace(rho_th))

# Measurement operators
M_plus = 0.5 * ((np.sqrt(kappa) + np.sqrt(1 - kappa)) * I +
                (np.sqrt(kappa) - np.sqrt(1 - kappa)) * sigma_x)
M_minus = 0.5 * ((np.sqrt(kappa) + np.sqrt(1 - kappa)) * I -
                 (np.sqrt(kappa) - np.sqrt(1 - kappa)) * sigma_x)

print(M_plus.conj().T @ M_plus + M_minus.conj().T @ M_minus)

# Measurement probabilities
P_plus = np.trace(M_plus @ rho_th @ M_plus.conj().T).real
P_minus = np.trace(M_minus @ rho_th @ M_minus.conj().T).real

# Post-measurement states

rho_M_plus = M_plus @ rho_th @ M_plus.conj().T / P_plus
rho_M_minus = M_minus @ rho_th @ M_minus.conj().T / P_minus

print("ρ_M+ =\n", rho_M_plus)
print("\nρ_M- =\n", rho_M_minus)

# Parameters
kappa_vals = np.linspace(0.01, 0.99, 500)  # measurement strength
z0_default = -0.05     # initial Bloch z-component 
TD_values = [0.001, 0.005, 0.01]  # demon temperatures

# Definitions

def Q_from_kappa(kappa):
    return -2.0 * np.log(2.0) - np.log(kappa * (1.0 - kappa))

def Wext_from_Q_z0(Q, z0, hbarw=hbar_omega0):
    term = np.sqrt(1.0 + np.exp(-Q) * (z0**2 - 1.0))
    return 0.5 * hbarw * (z0 * np.exp(-Q/2.0) + term)

def EM_from_Q_z0(Q, z0, hbarw=hbar_omega0):
    return 0.5 * hbarw * (1.0 + z0 * np.exp(-Q/2.0))

def Ef_from_Q_z0(Q, z0, hbarw=hbar_omega0):
    term = np.sqrt(1.0 + np.exp(-Q) * (z0**2 - 1.0))
    return 0.5 * hbarw * (1.0 - term)

def Wext_minus_erasure_over_EM(kappa, z0, TD, hbarw=hbar_omega0):
    Q = Q_from_kappa(kappa)
    Wext = Wext_from_Q_z0(Q, z0, hbarw)
    Wer = TD * np.log(2.0)
    EM = EM_from_Q_z0(Q, z0, hbarw)
    return (Wext - Wer) / EM

def COP_C(kappa, z0, TD, hbarw=hbar_omega0):
    Q = Q_from_kappa(kappa)
    E0 = 0.5 * hbarw * (1.0 + z0)
    Ef = Ef_from_Q_z0(Q, z0, hbarw)
    EM = EM_from_Q_z0(Q, z0, hbarw)
    Wer = TD * np.log(2.0)
    return (E0 - Ef) / (EM - E0 + Wer)

# PLOTS

linestyles = ['-', '--', ':']
colors  = ['#08306B', '#2171B5', '#6BAED6']

# Efficiency
fig1, ax1 = plt.subplots()
for i, TD in enumerate(TD_values):
    eta_vals = [Wext_minus_erasure_over_EM(kappa, z0_default, TD) for kappa in kappa_vals]
    line, = ax1.plot(
        kappa_vals,
        eta_vals,
        linestyle=linestyles[i],
        color=colors[i],
        linewidth=2.5,
        label=f'$T_D = {TD}$'
    )

ax1.set_xlabel(r'$\kappa$', fontsize=14)
ax1.set_ylabel(r'$\eta$', fontsize=14)
ax1.tick_params(labelsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12,loc='lower right')
plt.tight_layout()
plt.savefig('efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

# COP
fig2, ax2 = plt.subplots()
for i, TD in enumerate(TD_values):
    C_vals = [COP_C(kappa, z0_default, TD) for kappa in kappa_vals]
    ax2.plot(
        kappa_vals,
        C_vals,
        linestyle=linestyles[i],
        color=colors[i],
        linewidth=2.5,
        label=f'$T_D = {TD}$'
    )

ax2.set_xlabel(r'$\kappa$', fontsize=14)
ax2.set_ylabel(r'COP', fontsize=14)
ax2.set_ylim(0,28)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, ncol=3)
plt.tight_layout()
plt.savefig('cop.png', dpi=300, bbox_inches='tight')
plt.show()

# Work Extraction

plt.figure()

for TD in TD_values:
    W_vals = [Wext_from_Q_z0(Q_from_kappa(k), z0_default) for k in kappa_vals]
    plt.plot(kappa_vals, W_vals,color='#2171B5')
plt.xlabel(r'$\kappa$', fontsize=14)
plt.ylabel(r'$W_{\mathrm{ext}}$', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('work_vs_kappa.png', dpi=300)
plt.show()

for TD in TD_values:
    eta_vals = [Wext_minus_erasure_over_EM(k, z0_default, TD) for k in kappa_vals]
    print(f"TD = {TD}, max η = {np.max(eta_vals)}")
    
W_vals = [Wext_from_Q_z0(Q_from_kappa(k), z0_default) for k in kappa_vals]
print("Maximum work in range =", np.max(W_vals))
print("Minimum work in range =", np.min(W_vals))
print("Theoretical maximum =", 0.5*hbar_omega0)

results = {}

for TD in TD_values:

    eta_vals = np.array([
        Wext_minus_erasure_over_EM(kappa, z0_default, TD)
        for kappa in kappa_vals
    ])

    cop_vals = np.array([
        COP_C(kappa, z0_default, TD)
        for kappa in kappa_vals
    ])

    results[TD] = {
        "eta_min": np.min(eta_vals),
        "eta_max": np.max(eta_vals),
        "cop_min": np.min(cop_vals),
        "cop_max": np.max(cop_vals),
    }

for TD, vals in results.items():
    print(f"\nT_D = {TD}")
    print(f"Efficiency: min = {vals['eta_min']:.5f}, max = {vals['eta_max']:.5f}")
    print(f"COP:        min = {vals['cop_min']:.5f}, max = {vals['cop_max']:.5f}")