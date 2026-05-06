"""
Four-Stroke Single-Qubit Hybrid Engine

Combines the quantum catalyst of the catalyst-only engine with the
Maxwell's demon feedback protocol of the four-stroke engine. The catalyst
interacts with the system via an XY exchange Hamiltonian prior to
measurement, and the demon applies outcome-conditioned feedback.

Reproduces Figs. 13-15 of the dissertation: work extraction, efficiency,
and COP as functions of coupling strength g, plus entropy production,
entanglement, and coherence diagnostics at the optimal configuration.

All quantities are in natural units where kB = hbar = 1.
Energy and work are dimensionless, scaled by hbar*omega0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Constants + parameters

hbar_omega0   = 0.1
T             = 1.0
gamma_prime   = 0.05
delta_t       = 2
omega_c       = hbar_omega0
t_interaction = 1.0

# Demon temperatures
TD_values = [0.001, 0.005, 0.01]

# Pauli matrices + ladder operators 

I       = np.eye(2, dtype=complex)
sigma_x = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
sigma_z = np.array([[1,  0 ], [0, -1 ]], dtype=complex)

sigma_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)

# Hamiltonians

H0    = hbar_omega0 * np.array([[0, 0], [0, 1]], dtype=complex)
H_cat = omega_c     * np.array([[0, 0], [0, 1]], dtype=complex)

# Partial traces 

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

# Definitions

def state_fidelity(rho, sigma):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0, None)
    sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    M = sqrt_rho @ sigma @ sqrt_rho
    eigvals_M = np.clip(np.linalg.eigvalsh(M), 0, None)
    return float(np.real(np.sum(np.sqrt(eigvals_M)) ** 2))


def von_neumann_entropy(rho):
    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals = np.clip(eigvals, 0, None)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log(eigvals))


def coherence_rel_entropy(rho):
    rho_diag = np.diag(np.diag(rho))
    return von_neumann_entropy(rho_diag) - von_neumann_entropy(rho)


def concurrence(rho):
    Y    = np.kron(sigma_y, sigma_y)
    R    = rho @ Y @ rho.conj() @ Y
    eigs = np.sort(np.sqrt(np.abs(np.linalg.eigvals(R))))[::-1]
    return float(np.real(max(0, eigs[0] - eigs[1] - eigs[2] - eigs[3])))


def mutual_information(rho_joint):
    rho_s = partial_trace_system(rho_joint)
    rho_c = partial_trace_catalyst(rho_joint)
    return (von_neumann_entropy(rho_s)
            + von_neumann_entropy(rho_c)
            - von_neumann_entropy(rho_joint))

def energy(rho, H):
    return np.trace(rho @ H).real


def optimize_feedback(state):
    E0     = np.trace(state @ H0).real
    best_W = -np.inf
    best_U = I.copy()
    for axis in [sigma_x, sigma_y]:
        for angle in np.linspace(0, 2 * np.pi, 1000):
            U     = expm(-1j * angle * axis / 2)
            rho_f = U @ state @ U.conj().T
            W     = E0 - np.trace(rho_f @ H0).real
            if W > best_W:
                best_W = W
                best_U = U
    return best_U, best_W

# Apply the XY exchange Hamiltonian and returns the joint state
def evolve_joint(rho_sys, rho_cat, g, t=t_interaction):
    H_int     = g * (np.kron(sigma_plus, sigma_minus) +
                     np.kron(sigma_minus, sigma_plus))
    H_tot     = np.kron(H0, I) + np.kron(I, H_cat) + H_int
    U_joint   = expm(-1j * H_tot * t)
    rho_joint = U_joint @ np.kron(rho_sys, rho_cat) @ U_joint.conj().T
    return rho_joint, partial_trace_system(rho_joint), partial_trace_catalyst(rho_joint)


# Computes delta_Ec
def catalyst_energy_cost(rho_cat_i, rho_cat_f):
    dE = np.trace(rho_cat_f @ H_cat).real - np.trace(rho_cat_i @ H_cat).real
    return -dE

# Implement the demon protocol conditioned on individual measurement
def measure_and_feedback(rho_sys):
    p_plus  = np.trace(M_plus  @ rho_sys @ M_plus.conj().T).real
    p_minus = np.trace(M_minus @ rho_sys @ M_minus.conj().T).real
    rho_plus  = M_plus  @ rho_sys @ M_plus.conj().T  / p_plus
    rho_minus = M_minus @ rho_sys @ M_minus.conj().T / p_minus
    U_opt_plus,  W_opt_plus  = optimize_feedback(rho_plus)
    U_opt_minus, W_opt_minus = optimize_feedback(rho_minus)
    W_avg = p_plus * W_opt_plus + p_minus * W_opt_minus
    return (W_avg, p_plus,
            U_opt_plus  @ rho_plus  @ U_opt_plus.conj().T,
            U_opt_minus @ rho_minus @ U_opt_minus.conj().T)

# Catalyst states

cat_ground  = np.array([[1], [0]], dtype=complex)
cat_excited = np.array([[0], [1]], dtype=complex)
cat_super   = (cat_ground + cat_excited) / np.sqrt(2)

rho_cat_initial = cat_excited @ cat_excited.conj().T  # used for all g sweeps

catalyst_states = {
    'Ground |0>':        cat_ground  @ cat_ground.conj().T,
    'Excited |1>':       cat_excited @ cat_excited.conj().T,
    'Superposition |+>': cat_super   @ cat_super.conj().T,
}

# System initialisation 

rho_unnorm = expm(-H0 / T)
rho_th     = rho_unnorm / np.trace(rho_unnorm)

print("\nInitial thermal state:")
print(rho_th)
print(f"Initial system <σ_z> = {np.trace(rho_th @ sigma_z).real:.5f}")

apply_pre_unitary = True
if apply_pre_unitary:
    prep_angle = np.pi / 4
    U_prep = expm(-1j * prep_angle * sigma_x)
    rho_sys = U_prep @ rho_th @ U_prep.conj().T
    print(f"\nAfter preparation unitary (θ = {prep_angle:.3f}):")
else:
    rho_sys = rho_th
    print("\nNo preparation unitary applied.")

print(rho_sys)
print(f"System <σ_z> = {np.trace(rho_sys @ sigma_z).real:.5f}")

coh_initial = coherence_rel_entropy(rho_sys)

# Weak measurement operators

kappa   = 0.5 - np.sqrt(2 * gamma_prime * delta_t)
print(f"\nMeasurement strength κ = {kappa:.5f}")

M_plus  = 0.5 * ((np.sqrt(kappa) + np.sqrt(1 - kappa)) * I +
                  (np.sqrt(kappa) - np.sqrt(1 - kappa)) * sigma_x)
M_minus = 0.5 * ((np.sqrt(kappa) + np.sqrt(1 - kappa)) * I -
                  (np.sqrt(kappa) - np.sqrt(1 - kappa)) * sigma_x)

# Baseline (no catalyst)

W_avg_no_cat, *_ = measure_and_feedback(rho_sys)
print(f"\nBaseline (no catalyst): W_avg = {W_avg_no_cat:.5f}")

# Coupling strength sweep for excited state catalyst

print("\nCatalytic unitary: coupling strength sweep (excited state catalyst)")

for g in [0.1, 0.5, 1.0, 2.0]:
    _, rho_sys_after, rho_cat_final = evolve_joint(rho_sys, rho_cat_initial, g)
    W_sys  = energy(rho_sys, H0) - energy(rho_sys_after, H0)
    dE_cat = energy(rho_cat_final, H_cat) - energy(rho_cat_initial, H_cat)
    F      = state_fidelity(rho_cat_initial, rho_cat_final)
    print(f"g={g:.2f} | W_sys={W_sys:.5f} | dE_cat={dE_cat:.6f} | "
          f"fidelity={F:.5f} | catalyst returned? {'Yes :)' if F > 0.99 else 'No :('}")

# g sweep: work + fidelity (excited state)

g_range        = np.linspace(0.05, 1.5, 40)
W_cat_range    = []
fidelity_range = []

for g in g_range:
    _, rho_sys_g, rho_cat_g = evolve_joint(rho_sys, rho_cat_initial, g)
    fidelity_range.append(state_fidelity(rho_cat_initial, rho_cat_g))
    W_g, *_ = measure_and_feedback(rho_sys_g)
    W_net_g = W_g - catalyst_energy_cost(rho_cat_initial, rho_cat_g)
    W_cat_range.append(W_net_g)

# Catalyst parameter optimisation for F >= 0.90 

print("\nCatalyst parameter optimisation: sweeping over all three initial states")

g_opt_range = np.linspace(0.01, 1.5, 100)
best_config  = {'W': -np.inf, 'desc': None, 'g': None, 'rho_cat': None, 'TD': None}

for desc, rho_cat_candidate in catalyst_states.items():
    state_best = {'W': -np.inf, 'g': None, 'TD': None}
    for g_test in g_opt_range:
        H_int_test = g_test * (np.kron(sigma_plus, sigma_minus) +
                               np.kron(sigma_minus, sigma_plus))
        H_tot_test = np.kron(H0, I) + np.kron(I, H_cat) + H_int_test
        U_test     = expm(-1j * H_tot_test * t_interaction)
        rho_joint_test = U_test @ np.kron(rho_sys, rho_cat_candidate) @ U_test.conj().T
        rho_sys_test   = partial_trace_system(rho_joint_test)
        rho_cat_test_f = partial_trace_catalyst(rho_joint_test)
        F = state_fidelity(rho_cat_candidate, rho_cat_test_f)
        if F > 0.9:
            W_fb, *_ = measure_and_feedback(rho_sys_test)
            dE_cat = (energy(rho_cat_test_f, H_cat) - energy(rho_cat_candidate, H_cat))
            for TD in TD_values:
                Wer   = TD * np.log(2)
                W_net = W_fb - dE_cat - Wer
                if W_net > state_best['W']:
                    state_best.update({'W': W_net, 'g': g_test, 'TD': TD})
                if W_net > best_config['W']:
                    best_config.update({'W': W_net, 'desc': desc, 'g': g_test,
                                        'rho_cat': rho_cat_candidate, 'TD': TD})
    pct = ((state_best['W'] / W_avg_no_cat) - 1) * 100 if W_avg_no_cat != 0 else 0
    print(f"  {desc}: best W_net={state_best['W']:.5f} at g={state_best['g']:.3f}, "
          f"TD={state_best['TD']} ({pct:+.2f}% vs baseline)")

print(f"\nBest overall: {best_config['desc']}, g={best_config['g']:.3f}, "
      f"TD={best_config['TD']}, W_net={best_config['W']:.5f} "
      f"({((best_config['W'] / W_avg_no_cat) - 1) * 100:+.2f}% vs baseline)")

g_opt       = best_config['g']
rho_cat_opt = best_config['rho_cat']
TD_opt      = best_config['TD']
Wer_opt     = TD_opt * np.log(2)
_, rho_sys_after_cat, _ = evolve_joint(rho_sys, rho_cat_opt, g_opt)

# Measurement + feedback at optimal g

print(f"\nMeasurement + feedback ({best_config['desc']}, g={g_opt:.3f})")

W_avg, p_plus, rho_plus_post, rho_minus_post = measure_and_feedback(rho_sys_after_cat)

_, _, rho_cat_f_opt = evolve_joint(rho_sys, rho_cat_opt, g_opt)
dE_cat_opt = energy(rho_cat_f_opt, H_cat) - energy(rho_cat_opt, H_cat)
W_net_opt  = W_avg - dE_cat_opt - Wer_opt

print(f"Measurement probabilities: p(+) = {p_plus:.5f}, p(-) = {1-p_plus:.5f}")
print(f"Average feedback work:     W_fb  = {W_avg:.5f}")
print(f"Catalyst energy change:    dE_c  = {dE_cat_opt:.5f}")
print(f"Erasure cost (TD={TD_opt}): W_er  = {Wer_opt:.5f}")
print(f"Net work:                  W_net = {W_net_opt:.5f}")

print("\nDemon memory erasure cost:")
for TD in TD_values:
    Wer      = TD * np.log(2)
    net_gain = W_avg - dE_cat_opt - Wer
    print(f"T_D = {TD}: W_er = {Wer:.5f}, net gain = {net_gain:.5f} "
          f"{'Yes :)' if net_gain > 0 else 'No :('}")

print("\nComparison with vs without catalyst:")
print(f"Without catalyst: W_fb = {W_avg_no_cat:.5f}")
print(f"With catalyst:    W_fb = {W_avg:.5f}")
print(f"W_fb improvement: {W_avg - W_avg_no_cat:.5f} "
      f"({((W_avg / W_avg_no_cat) - 1) * 100:.1f}%)")
print(f"W_net vs baseline W_fb (TD={TD_opt}): "
      f"{((W_net_opt / W_avg_no_cat) - 1) * 100:.1f}%")

# Efficiency and COP (excited state)

eta_dict = {}
cop_dict = {}
for TD in TD_values:
    Wer = TD * np.log(2)
    eta_dict[TD] = [W / (W + Wer) if (W + Wer) > 0 else 0 for W in W_cat_range]
    cop_dict[TD] = [W / Wer for W in W_cat_range]

# W_raw vs W_net + fidelity (excited state)

W_raw_list        = []
W_net_list        = []
fidelity_net_list = []

for g in g_range:
    _, rho_sys_g, rho_cat_g = evolve_joint(rho_sys, rho_cat_initial, g)
    F = state_fidelity(rho_cat_initial, rho_cat_g)
    fidelity_net_list.append(F)
    W_raw, *_ = measure_and_feedback(rho_sys_g)
    W_raw_list.append(W_raw)
    cat_cost = catalyst_energy_cost(rho_cat_initial, rho_cat_g)
    W_net_list.append(W_raw - cat_cost - Wer_opt)

# Entanglement + coherence (excited state)

entanglement_range = []
coherence_range    = []

for g in g_range:
    rho_joint_g, rho_sys_g, _ = evolve_joint(rho_sys, rho_cat_initial, g)
    entanglement_range.append(concurrence(rho_joint_g))
    coherence_range.append(coherence_rel_entropy(rho_sys_g))

# Mutual information at optimal g

rho_joint_opt, _, _ = evolve_joint(rho_sys, rho_cat_opt, g_opt)
MI = mutual_information(rho_joint_opt)
print(f"\nMutual information I(S:C) at g={g_opt:.3f}: {MI:.5f}")

# Coherence in measurement branches

p_p   = np.trace(M_plus  @ rho_sys_after_cat @ M_plus.conj().T).real
p_m   = np.trace(M_minus @ rho_sys_after_cat @ M_minus.conj().T).real
rho_p = M_plus  @ rho_sys_after_cat @ M_plus.conj().T  / p_p
rho_m = M_minus @ rho_sys_after_cat @ M_minus.conj().T / p_m

p_p_nc   = np.trace(M_plus  @ rho_sys @ M_plus.conj().T).real
p_m_nc   = np.trace(M_minus @ rho_sys @ M_minus.conj().T).real
rho_p_nc = M_plus  @ rho_sys @ M_plus.conj().T  / p_p_nc
rho_m_nc = M_minus @ rho_sys @ M_minus.conj().T / p_m_nc
rho_post_nc = p_p_nc * rho_p_nc + p_m_nc * rho_m_nc

E_prep     = energy(rho_cat_opt, H_cat)
E_maintain = max(0, energy(rho_cat_f_opt, H_cat) - E_prep)

print(f"\nCatalyst energy cost (prep):     {E_prep:.5f}")
print(f"Catalyst energy cost (maintain): {E_maintain:.5f}")

print("\nCoherence comparison:")
print(f"Initial coherence:              {coh_initial:.5f}")
print("Without catalyst:")
print(f"  After measurement (avg):      {coherence_rel_entropy(rho_post_nc):.5f}")
print(f"  (+ branch):                  {coherence_rel_entropy(rho_p_nc):.5f}")
print(f"  (- branch):                  {coherence_rel_entropy(rho_m_nc):.5f}")
print(f"With catalyst ({best_config['desc']}, g={g_opt:.3f}):")
print(f"  After catalyst:               {coherence_rel_entropy(rho_sys_after_cat):.5f}")
print(f"  (+ branch):                  {coherence_rel_entropy(rho_p):.5f}")
print(f"  (- branch):                  {coherence_rel_entropy(rho_m):.5f}")
print(f"\nSystem-catalyst concurrence at g={g_opt:.3f}: {concurrence(rho_joint_opt):.5f}")

# Entropy production (optimal g)

print("\nEntropy production (optimal g)")

S_sys_initial     = von_neumann_entropy(rho_sys)
S_sys_after_cat   = von_neumann_entropy(rho_sys_after_cat)
S_cat_initial_opt = von_neumann_entropy(rho_cat_opt)
S_cat_after       = von_neumann_entropy(rho_cat_f_opt)
S_joint_after_cat = von_neumann_entropy(rho_joint_opt)
dS_joint_unitary  = S_joint_after_cat - (S_sys_initial + S_cat_initial_opt)

print(f"Stage 1 — catalytic unitary:")
print(f"  S(sys) before:      {S_sys_initial:.5f}")
print(f"  S(sys) after:       {S_sys_after_cat:.5f}")
print(f"  S(cat) before:      {S_cat_initial_opt:.5f}")
print(f"  S(cat) after:       {S_cat_after:.5f}")
print(f"  dS_joint (should≈0): {dS_joint_unitary:.6f}")

rho_post_meas  = p_p * rho_p + p_m * rho_m
S_post_meas    = von_neumann_entropy(rho_post_meas)
dS_measurement = S_post_meas - S_sys_after_cat
print(f"Stage 2 — weak measurement:")
print(f"  S(sys) before:      {S_sys_after_cat:.5f}")
print(f"  S(sys) after (avg): {S_post_meas:.5f}")
print(f"  dS_measurement:     {dS_measurement:.5f}")

U_p, _ = optimize_feedback(rho_p)
U_m, _ = optimize_feedback(rho_m)
rho_post_fb = (p_p * (U_p @ rho_p @ U_p.conj().T) +
               p_m * (U_m @ rho_m @ U_m.conj().T))
S_post_fb   = von_neumann_entropy(rho_post_fb)
dS_feedback = S_post_fb - S_post_meas
print(f"Stage 3 — feedback unitary:")
print(f"  S(sys) before:      {S_post_meas:.5f}")
print(f"  S(sys) after:       {S_post_fb:.5f}")
print(f"  dS_feedback:        {dS_feedback:.5f}")

dS_demon_erasure = np.log(2)
print(f"Stage 4 — demon erasure:")
print(f"  dS_demon (erasure): {dS_demon_erasure:.5f}  (= ln2, independent of T_D)")

dS_sys_total = S_post_fb - S_sys_initial
dS_cat_total = S_cat_after - S_cat_initial_opt
sigma_tot    = dS_sys_total + dS_cat_total + dS_demon_erasure
print(f"Total entropy production:")
print(f"  dS_sys (full cycle): {dS_sys_total:.5f}")
print(f"  dS_cat (full cycle): {dS_cat_total:.5f}")
print(f"  dS_demon (erasure):  {dS_demon_erasure:.5f}")
print(f"  sigma_tot = {sigma_tot:.5f}  "
      f"{'... Second law satisfied' if sigma_tot >= -1e-10 else 'Second law VIOLATED'}")

# Entropy production vs g sweep

sigma_vs_g  = []
dS_sys_vs_g = []
dS_cat_vs_g = []

S_cat_initial_excited = von_neumann_entropy(rho_cat_initial)

for g in g_range:
    rho_jt, rho_sg, rho_cg = evolve_joint(rho_sys, rho_cat_initial, g)
    W_fb, p_p_g, rho_p_g, rho_m_g = measure_and_feedback(rho_sg)
    p_m_g = 1 - p_p_g
    U_pg, _ = optimize_feedback(rho_p_g)
    U_mg, _ = optimize_feedback(rho_m_g)
    rho_fb_g = (p_p_g * (U_pg @ rho_p_g @ U_pg.conj().T) +
                p_m_g * (U_mg @ rho_m_g @ U_mg.conj().T))
    dS_s = von_neumann_entropy(rho_fb_g) - S_sys_initial
    dS_c = von_neumann_entropy(rho_cg)   - S_cat_initial_excited
    dS_sys_vs_g.append(dS_s)
    dS_cat_vs_g.append(dS_c)
    sigma_vs_g.append(dS_s + dS_c + np.log(2))

print(f"\nEntropy production at g={g_opt:.3f} (from sweep): "
      f"{sigma_vs_g[np.argmin(np.abs(g_range - g_opt))]:.5f}")

# Catalyst state compariso

cat_states = {
    'Ground':        cat_ground  @ cat_ground.conj().T,
    'Excited':       cat_excited @ cat_excited.conj().T,
    'Superposition': cat_super   @ cat_super.conj().T,
}

work_results     = {}
fidelity_results = {}

for name, rho_cat in cat_states.items():
    W_list, F_list = [], []
    for g in g_range:
        _, rho_sys_f, rho_cat_f = evolve_joint(rho_sys, rho_cat, g)
        W_fb_g, *_ = measure_and_feedback(rho_sys_f)
        dE_cat_g   = energy(rho_cat_f, H_cat) - energy(rho_cat, H_cat)
        W_list.append(W_fb_g - dE_cat_g)
        F_list.append(state_fidelity(rho_cat, rho_cat_f))
    work_results[name]     = W_list
    fidelity_results[name] = F_list

# PLOTS

linestyles = ['-', '--', ':']
colors     = ['#08306B', '#2171B5', '#6BAED6']

# Work extraction
plt.figure()
plt.plot(g_range, W_cat_range, color='#2171B5', linewidth=2)
plt.xlabel('$g$', fontsize=14)
plt.ylabel(r'$W_{\mathrm{ext}}$', fontsize=14)
plt.tick_params(labelsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Efficiency
plt.figure()
for i, TD in enumerate(TD_values):
    plt.plot(g_range, eta_dict[TD],
             linestyle=linestyles[i], color=colors[i],
             linewidth=2.5, label=f'$T_D={TD}$')
plt.xlabel('$g$', fontsize=14)
plt.ylabel(r'$\eta$', fontsize=14)
plt.ylim(0.6, 1)
plt.tick_params(labelsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.show()

# COP
plt.figure()
for i, TD in enumerate(TD_values):
    plt.plot(g_range, cop_dict[TD],
             linestyle=linestyles[i], color=colors[i],
             linewidth=2.5, label=f'$T_D={TD}$')
plt.xlabel('$g$', fontsize=14)
plt.ylabel('COP', fontsize=14)
plt.ylim(0, 80)
plt.tick_params(labelsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, ncol=3, loc='upper center')
plt.tight_layout()
plt.show()

# SUMMARY

print("\nCatalytic Maxwell's demon summary:")
print(f"All work values: W_net = W_fb - dE_cat - W_er")
print(f"\nWork extraction comparison:")
print(f"  No catalyst (baseline):  W_fb  = {W_avg_no_cat:.5f}")
print(f"  Best cat. config:        W_fb  = {W_avg:.5f} "
      f"({((W_avg / W_avg_no_cat) - 1) * 100:+.1f}% in demon feedback work)")
print(f"                           W_net = {best_config['W']:.5f} "
      f"({((best_config['W'] / W_avg_no_cat) - 1) * 100:+.1f}% total) "
      f"at g={g_opt:.3f}, state={best_config['desc']}, TD={TD_opt}")
print(f"\nDemon performance (best configuration, after all costs):")
for TD in TD_values:
    Wer = TD * np.log(2)
    net = W_avg - dE_cat_opt - Wer
    print(f"  T_D={TD}: W_er={Wer:.5f}, net gain={net:.5f} "
          f"{'Yes :)' if net > 0 else 'No :('}")

print("\nCatalyst state comparison summary:")
print(f"{'State':<15}  {'Max W_ext':>10}  {'Max F':>8}  {'W at F>=0.90':>14}")
for name in cat_states:
    W_arr   = np.array(work_results[name])
    F_arr   = np.array(fidelity_results[name])
    valid_W = W_arr[F_arr >= 0.90]
    max_valid_W = max(valid_W) if len(valid_W) > 0 else float('nan')
    print(f"{name:<15}  {np.max(W_arr):>10.5f}  {np.max(F_arr):>8.4f}  {max_valid_W:>14.5f}")

F_opt = state_fidelity(rho_cat_opt, rho_cat_f_opt)
print(f"Catalyst return fidelity at optimum: {F_opt:.5f}")

print("\nEfficiency extrema for each demon temperature:")

for TD in TD_values:
    eta_arr = np.array(eta_dict[TD])
    eta_max = np.max(eta_arr)
    eta_min = np.min(eta_arr)
    g_max = g_range[np.argmax(eta_arr)]
    g_min = g_range[np.argmin(eta_arr)]
    print(f"\nT_D = {TD}:")
    print(f"  η_max = {eta_max:.5f} at g = {g_max:.3f}")
    print(f"  η_min = {eta_min:.5f} at g = {g_min:.3f}")

print("\nCOP extrema for each demon temperature:")

for TD in TD_values:
    cop_arr = np.array(cop_dict[TD])
    cop_max = np.max(cop_arr)
    cop_min = np.min(cop_arr)
    g_max = g_range[np.argmax(cop_arr)]
    g_min = g_range[np.argmin(cop_arr)]
    print(f"\nT_D = {TD}:")
    print(f"  COP_max = {cop_max:.5f} at g = {g_max:.3f}")
    print(f"  COP_min = {cop_min:.5f} at g = {g_min:.3f}")

F_threshold = 0.90
valid_mask = np.array(fidelity_range) >= F_threshold

print("\nEfficiency (valid catalytic regime):")

for TD in TD_values:
    eta_arr = np.array(eta_dict[TD])
    eta_valid = eta_arr[valid_mask]
    g_valid   = g_range[valid_mask]
    if len(eta_valid) > 0:
        eta_max = np.max(eta_valid)
        eta_min = np.min(eta_valid)
        g_max = g_valid[np.argmax(eta_valid)]
        g_min = g_valid[np.argmin(eta_valid)]
        print(f"\nT_D = {TD}:")
        print(f"  η_max(valid) = {eta_max:.5f} at g = {g_max:.3f}")
        print(f"  η_min(valid) = {eta_min:.5f} at g = {g_min:.3f}")
    else:
        print(f"\nT_D = {TD}: No valid catalytic points")

print("\nCOP (valid catalytic regime):")

for TD in TD_values:
    cop_arr = np.array(cop_dict[TD])
    cop_valid = cop_arr[valid_mask]
    g_valid   = g_range[valid_mask]
    if len(cop_valid) > 0:
        cop_max = np.max(cop_valid)
        cop_min = np.min(cop_valid)
        g_max = g_valid[np.argmax(cop_valid)]
        g_min = g_valid[np.argmin(cop_valid)]
        print(f"\nT_D = {TD}:")
        print(f"  COP_max(valid) = {cop_max:.5f} at g = {g_max:.3f}")
        print(f"  COP_min(valid) = {cop_min:.5f} at g = {g_min:.3f}")
    else:
        print(f"\nT_D = {TD}: No valid catalytic points")