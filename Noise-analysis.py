"""
Two-Qubit Uncoupled Hybrid Engine with Noise Robustness Analysis

Extends the uncoupled two-qubit hybrid engine by applying three standard
single-qubit noise channels (amplitude damping, dephasing, depolarising)
to all four qubits independently across a range of noise probabilities
p in [0, 0.20], studying the effect on work extraction, catalyst fidelity,
and system-catalyst negativity.

Reproduces the noise robustness results of the dissertation (Figs. 16-18).

All quantities are in natural units where kB = hbar = 1.
Energy and work are dimensionless, scaled by hbar*omega0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Constants + parameters

hbar_omega0   = 0.5    # System energy gap
T             = 0.2    # System temperature
gamma_prime   = 0.05   # Measurement back-action parameter
delta_t       = 2      # Measurement time step
omega_c       = hbar_omega0  # Catalyst frequency

# Demon temperatures
TD_values = [0.001, 0.005, 0.01]

TD_main   = 0.005
Wer_main  = TD_main * np.log(2)

# Pauli matrices + ladder operators

I       = np.eye(2, dtype=complex)
sigma_x = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
sigma_z = np.array([[1,  0 ], [0, -1 ]], dtype=complex)

sigma_plus  = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
proj1       = np.array([[0, 0], [0, 1]], dtype=complex)

I4 = np.eye(4, dtype=complex)

# Hamiltonians

H0    = hbar_omega0 * (np.kron(proj1, I) + np.kron(I, proj1))
H_cat = omega_c     * (np.kron(proj1, I) + np.kron(I, proj1))

# Collective sigma_z
SZ_sys = np.kron(sigma_z, I) + np.kron(I, sigma_z)

# Dimensions
dim_sys = 4
dim_cat = 4
dim_tot = dim_sys * dim_cat  # 16

# Partial traces

def partial_trace_system(rho_joint):
    rho_sys = np.zeros((dim_sys, dim_sys), dtype=complex)
    for k in range(dim_cat):
        for i in range(dim_sys):
            for j in range(dim_sys):
                rho_sys[i, j] += rho_joint[i*dim_cat+k, j*dim_cat+k]
    return rho_sys

def partial_trace_catalyst(rho_joint):
    rho_cat = np.zeros((dim_cat, dim_cat), dtype=complex)
    for s in range(dim_sys):
        for k in range(dim_cat):
            for l in range(dim_cat):
                rho_cat[k, l] += rho_joint[s*dim_cat+k, s*dim_cat+l]
    return rho_cat

# Definitions

def state_fidelity(rho, sigma):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals  = np.clip(eigvals, 0, None)
    sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    M        = sqrt_rho @ sigma @ sqrt_rho
    eigvals_M = np.clip(np.linalg.eigvalsh(M), 0, None)
    return float(np.real(np.sum(np.sqrt(eigvals_M)) ** 2))

def von_neumann_entropy(rho):
    eigvals = np.real(np.linalg.eigvalsh(rho))
    eigvals = np.clip(eigvals, 0, None)
    s = eigvals.sum()
    if s < 1e-15:
        return 0.0
    eigvals /= s
    eigvals  = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log(eigvals)))

def coherence_rel_entropy(rho):
    rho_diag = np.diag(np.diag(rho))
    return von_neumann_entropy(rho_diag) - von_neumann_entropy(rho)

def negativity(rho, dA, dB):
    pt = np.zeros_like(rho)
    for i in range(dA):
        for j in range(dA):
            pt[i*dB:(i+1)*dB, j*dB:(j+1)*dB] = \
                rho[i*dB:(i+1)*dB, j*dB:(j+1)*dB].T
    eigvals = np.real(np.linalg.eigvalsh(pt))
    return float(np.sum(np.abs(eigvals[eigvals < 0])))

def mutual_information(rho_joint):
    rho_s = partial_trace_system(rho_joint)
    rho_c = partial_trace_catalyst(rho_joint)
    return (von_neumann_entropy(rho_s)
            + von_neumann_entropy(rho_c)
            - von_neumann_entropy(rho_joint))

def concurrence_2qubit(rho):
    sy = sigma_y
    rho_tilde = np.kron(sy, sy) @ rho.conj() @ np.kron(sy, sy)
    R = rho @ rho_tilde
    eigvals = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    eigvals = np.clip(eigvals, 0, None)
    lambdas = np.sqrt(eigvals)
    return float(max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))

# Feedback optimisation

_angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
_U1     = np.array([np.cos(t/2)*I - 1j*np.sin(t/2)*ax
                    for ax in [sigma_x, sigma_y, sigma_z]
                    for t in _angles])
_UA     = np.array([np.kron(u, I) for u in _U1])   # act on qubit A
_UB     = np.array([np.kron(I, u) for u in _U1])   # act on qubit B
_HB     = _UB.conj().transpose(0,2,1) @ H0 @ _UB

def optimize_feedback(state):
    E0     = np.trace(state @ H0).real
    best_W = -np.inf
    best_U = I4.copy()
    for ua in _UA:
        tmp  = ua @ state @ ua.conj().T
        Wvec = E0 - np.einsum('kij,ji->k', _HB, tmp).real
        idx  = int(np.argmax(Wvec))
        if Wvec[idx] > best_W:
            best_W = Wvec[idx]
            best_U = _UB[idx] @ ua
    return best_U, best_W

# Hamiltonian

def _build_xy(g, p, q, n=4):
    ops_p = [I]*n; ops_m = [I]*n
    ops_p[p] = sigma_plus;  ops_p[q] = sigma_minus
    ops_m[p] = sigma_minus; ops_m[q] = sigma_plus
    def kron_chain(ops):
        out = ops[0]
        for op in ops[1:]: out = np.kron(out, op)
        return out
    return g * (kron_chain(ops_p) + kron_chain(ops_m))

def evolve_joint(rho_sys, rho_cat, g, t):
    H_free = np.kron(H0, I4) + np.kron(I4, H_cat)
    H_AC   = _build_xy(g, 0, 2)
    H_BC   = _build_xy(g, 1, 3)
    H_tot  = H_free + H_AC + H_BC
    U      = expm(-1j * H_tot * t)
    rho    = U @ np.kron(rho_sys, rho_cat) @ U.conj().T
    return rho, partial_trace_system(rho), partial_trace_catalyst(rho)

# Computes delta_Ec
def catalyst_energy_cost(rho_cat_initial, rho_cat_final):
    dE = np.trace(rho_cat_final @ H_cat).real - np.trace(rho_cat_initial @ H_cat).real
    return -dE

# Catalyst initial states

def ket2(a, b):
    v = np.zeros((4, 1), dtype=complex); v[2*a+b] = 1; return v

def dm2(a, b):
    v = ket2(a, b); return v @ v.conj().T

cat_ground   = dm2(0, 0)   # |00>
cat_excited  = dm2(1, 1)   # |11>

_v_sup     = np.kron([[np.cos(np.pi/4)], [np.sin(np.pi/4)]],
                      [[np.cos(np.pi/4)], [np.sin(np.pi/4)]])
cat_super  = _v_sup @ _v_sup.conj().T   # |+x> x |+x>

cat_states = {
    'Ground':        cat_ground,
    'Excited':       cat_excited,
    'Superposition': cat_super,
}

# System initialisation

rho_unnorm = expm(-H0 / T)
rho_th     = rho_unnorm / np.trace(rho_unnorm)

print("Initial 2-qubit thermal state:")
print(np.round(rho_th.real, 5))
print(f"Initial system <SZ> = {np.trace(rho_th @ SZ_sys).real:.5f}")

prep_angle = np.pi / 4
X_tot  = np.kron(sigma_x, I) + np.kron(I, sigma_x)
U_prep = expm(-1j * prep_angle * X_tot / 2)
rho_sys = U_prep @ rho_th @ U_prep.conj().T

print(f"\nAfter preparation unitary (theta = {prep_angle:.3f}):")
print(np.round(rho_sys.real, 5))
print(f"System <SZ> = {np.trace(rho_sys @ SZ_sys).real:.5f}")

coh_initial   = coherence_rel_entropy(rho_sys)
S_sys_initial = von_neumann_entropy(rho_sys)

# Weak measurement operators

kappa = 0.5 - np.sqrt(2 * gamma_prime * delta_t)
print(f"\nMeasurement strength kappa = {kappa:.5f}")

_ev, _ec = np.linalg.eigh(sigma_x)
_P_xp    = np.outer(_ec[:,1], _ec[:,1].conj())
_P_xm    = np.outer(_ec[:,0], _ec[:,0].conj())

M_plus  = np.kron(np.sqrt(kappa)*_P_xp + np.sqrt(1-kappa)*_P_xm, I)
M_minus = np.kron(np.sqrt(1-kappa)*_P_xp + np.sqrt(kappa)*_P_xm, I)

assert np.allclose(M_plus.conj().T @ M_plus +
                   M_minus.conj().T @ M_minus, I4, atol=1e-8), "POVM not complete!"

# Implements the demon protocol conditioned on individual measurement outcomes
def measure_and_feedback(rho_sys):
    p_plus  = np.trace(M_plus  @ rho_sys @ M_plus.conj().T).real
    p_minus = np.trace(M_minus @ rho_sys @ M_minus.conj().T).real

    rho_plus  = M_plus  @ rho_sys @ M_plus.conj().T  / max(p_plus,  1e-10)
    rho_minus = M_minus @ rho_sys @ M_minus.conj().T / max(p_minus, 1e-10)

    U_opt_plus,  W_opt_plus  = optimize_feedback(rho_plus)
    U_opt_minus, W_opt_minus = optimize_feedback(rho_minus)

    W_avg = p_plus * W_opt_plus + p_minus * W_opt_minus

    return (W_avg, p_plus,
            U_opt_plus  @ rho_plus  @ U_opt_plus.conj().T,
            U_opt_minus @ rho_minus @ U_opt_minus.conj().T)

# Baseline: no catalyst

W_avg_no_cat, *_ = measure_and_feedback(rho_sys)
print(f"\nBaseline (no catalyst): W_avg = {W_avg_no_cat:.5f}")

# Coarse scan: g in [0.01, 1.5], t = 2*pi/g (Rabi return point)

print("\nCOARSE SCAN: g sweep at t = 2*pi/g  (Rabi return point)")
print(f"{'g':>6}  {'Catalyst':>14}  {'W_fb':>8}  {'Fidelity':>9}  "
      f"{'N(S,C)':>8}  {'I(S:C)':>7}  {'C(A,B)':>7}  {'Role':>6}")

g_coarse = np.linspace(0.01, 1.5, 20)
coarse_records = []

for cat_name, rho_cat_init in cat_states.items():
    for g in g_coarse:
        t_rabi = 2 * np.pi / g
        rho_joint, rho_sg, rho_cg = evolve_joint(rho_sys, rho_cat_init, g, t_rabi)
        F  = state_fidelity(rho_cat_init, rho_cg)
        W, *_ = measure_and_feedback(rho_sg)
        N  = negativity(rho_joint, dim_sys, dim_cat)
        MI = mutual_information(rho_joint)
        C  = concurrence_2qubit(rho_sg)
        role = 'catalyst' if F >= 0.90 else ('fuel' if F >= 0.20 else 'inert')
        coarse_records.append((cat_name, g, t_rabi, W, F, N, MI, C, role))

# Print best per catalyst at Rabi return point
for cat_name in cat_states:
    recs = [(W, F, g, t, N, MI, C, role)
            for (cn, g, t, W, F, N, MI, C, role) in coarse_records if cn == cat_name]
    best = max(recs, key=lambda x: x[0])
    W, F, g, t, N, MI, C, role = best
    print(f"{g:6.3f}  {cat_name:>14}  {W:8.5f}  {F:9.5f}  "
          f"{N:8.5f}  {MI:7.5f}  {C:7.5f}  {role:>6}")

# Best g from coarse scan at Rabi return
ground_rabi_recs = [(W, F, g) for (cn, g, t, W, F, N, MI, C, role)
                    in coarse_records if cn == 'Ground']
best_ground_rabi = max(ground_rabi_recs, key=lambda x: x[0])
best_excited_rabi = max([(W, F, g) for (cn, g, t, W, F, N, MI, C, role)
                         in coarse_records if cn == 'Excited'], key=lambda x: x[0])

print(f"\nBest W_fb at Rabi return — Ground:   {best_ground_rabi[0]:.5f}  "
      f"(g={best_ground_rabi[2]:.3f}, F={best_ground_rabi[1]:.5f})")
print(f"Best W_fb at Rabi return — Excited:  {best_excited_rabi[0]:.5f}  "
      f"(g={best_excited_rabi[2]:.3f}, F={best_excited_rabi[1]:.5f})")
print(f"Improvement over baseline: "
      f"{((best_ground_rabi[0]/W_avg_no_cat)-1)*100:+.2f}% (Ground), "
      f"{((best_excited_rabi[0]/W_avg_no_cat)-1)*100:+.2f}% (Excited)")

# Optimal g at Rabi return
g_opt_rabi = best_ground_rabi[2]
t_opt_rabi = 2 * np.pi / g_opt_rabi
rho_joint_rabi, rho_sys_rabi, rho_cat_rabi = evolve_joint(rho_sys, cat_ground, g_opt_rabi, t_opt_rabi)
N_rabi  = negativity(rho_joint_rabi, dim_sys, dim_cat)
MI_rabi = mutual_information(rho_joint_rabi)
C_rabi  = concurrence_2qubit(rho_sys_rabi)
W_rabi, *_ = measure_and_feedback(rho_sys_rabi)

print(f"\nAt g_opt={g_opt_rabi:.3f}, t={t_opt_rabi:.3f} (Rabi return):")
print(f"  N(S,C)={N_rabi:.5f}, I(S:C)={MI_rabi:.5f}, C(A,B)={C_rabi:.5f}")
print(f"  W_fb={W_rabi:.5f}")

# Entropy production at Rabi return optimal
_, _, rho_cat_f_rabi = evolve_joint(rho_sys, cat_ground, g_opt_rabi, t_opt_rabi)
W_net_rabi = W_rabi - Wer_main
sigma_rabi = (von_neumann_entropy(rho_sys_rabi) - S_sys_initial +
              von_neumann_entropy(rho_cat_f_rabi) - von_neumann_entropy(cat_ground) +
              np.log(2))
print(f"  W_net (TD={TD_main}): {W_net_rabi:.5f}")
print(f"  sigma_tot: {sigma_rabi:.5f}  "
      f"{'Second law satisfied' if sigma_rabi >= -1e-10 else 'VIOLATED'}")

# Full scan: all three catalysts, full (g, t) parameter space 

g_scan = np.linspace(0.01, 1.5, 25)
full_records = []

for cat_name, rho_cat_init in cat_states.items():
    for g in g_scan:
        t_max = 2 * np.pi / g
        for t in np.linspace(0.01, t_max, 40):
            rho_joint, rho_sg, rho_cg = evolve_joint(rho_sys, rho_cat_init, g, t)
            F = state_fidelity(rho_cat_init, rho_cg)
            W, *_ = measure_and_feedback(rho_sg)
            full_records.append((cat_name, g, t, W, F))

# Find best per fidelity band across all catalysts
print(f"\n{'Fidelity band':<14}  {'W':>9}  {'Fidelity':>9}  {'g':>6}  "
      f"{'t':>7}  {'Catalyst':>14}  Role")

best_config = {'W': -np.inf, 'desc': '', 'F': 0, 'g': 0, 't': 0,
               'cat_name': '', 'rho_cat': None}

for lo, hi, label in [(0.99, 1.01, 'F > 0.99'),
                       (0.95, 0.99, 'F 0.95-0.99'),
                       (0.90, 0.95, 'F 0.90-0.95'),
                       (0.80, 0.90, 'F 0.80-0.90')]:
    cands = [(W, F, g, t, cn)
             for (cn, g, t, W, F) in full_records if lo <= F < hi]
    if cands:
        W, F, g, t, cn = max(cands, key=lambda x: x[0])
        role = 'catalyst' if F >= 0.90 else 'fuel'
        print(f"{label:<14}  {W:9.5f}  {F:9.5f}  {g:6.3f}  {t:7.4f}  {cn:>14}  {role}")
        if F >= 0.90 and W > best_config['W']:
            best_config.update({'W': W, 'F': F, 'g': g, 't': t,
                                'cat_name': cn,
                                'rho_cat': cat_states[cn],
                                'desc': f'{cn}, g={g:.3f}, t={t:.4f}'})

g_opt = best_config['g']
t_opt = best_config['t']
rho_cat_opt = best_config['rho_cat']

print(f"\nBest catalyst configuration: {best_config['desc']}")
print(f"Best work extraction:        {best_config['W']:.5f}")

# Evolve at optimal (g, t)
rho_joint_opt, rho_sys_after_cat, rho_cat_f_opt = evolve_joint(rho_sys, rho_cat_opt, g_opt, t_opt)

# Measurement + feedback at optimal 

print(f"\nMEASUREMENT & FEEDBACK (optimal: {best_config['desc']})")

W_avg, p_plus, rho_plus_post, rho_minus_post = measure_and_feedback(rho_sys_after_cat)
print(f"Measurement probabilities: p(+) = {p_plus:.5f}, p(-) = {1-p_plus:.5f}")
print(f"Average work extracted:    W_avg = {W_avg:.5f}")

# Demon memory erasure cost
print("\nDEMON MEMORY ERASURE COST  (W_er = T_D * ln2)")
for TD in TD_values:
    Wer     = TD * np.log(2)
    net_gain = W_avg - Wer
    print(f"T_D = {TD}: W_er = {Wer:.5f}, net gain = {net_gain:.5f} "
          f"{'Yes :)' if net_gain > 0 else 'No :('}")

# Comparison
print("\nCOMPARISON: WITH vs WITHOUT CATALYST")
print(f"Without catalyst: W_avg = {W_avg_no_cat:.5f}")
print(f"With catalyst:    W_avg = {W_avg:.5f}")
print(f"Improvement:      {W_avg - W_avg_no_cat:.5f} "
      f"({((W_avg / W_avg_no_cat) - 1) * 100:.1f}%)")

# Quantum correlations at optimal
N_opt  = negativity(rho_joint_opt, dim_sys, dim_cat)
MI_opt = mutual_information(rho_joint_opt)
C_opt  = concurrence_2qubit(rho_sys_after_cat)

print(f"\nSystem-catalyst negativity N(S,C):  {N_opt:.5f}")
print(f"Mutual information I(S:C):           {MI_opt:.5f}")
print(f"Within-system concurrence C(A,B):    {C_opt:.5f}")

# Coherence comparison
p_p  = np.trace(M_plus  @ rho_sys_after_cat @ M_plus.conj().T).real
p_m  = np.trace(M_minus @ rho_sys_after_cat @ M_minus.conj().T).real
rho_p = M_plus  @ rho_sys_after_cat @ M_plus.conj().T  / max(p_p,  1e-10)
rho_m = M_minus @ rho_sys_after_cat @ M_minus.conj().T / max(p_m, 1e-10)

p_p_nc   = np.trace(M_plus  @ rho_sys @ M_plus.conj().T).real
p_m_nc   = np.trace(M_minus @ rho_sys @ M_minus.conj().T).real
rho_p_nc = M_plus  @ rho_sys @ M_plus.conj().T  / max(p_p_nc, 1e-10)
rho_m_nc = M_minus @ rho_sys @ M_minus.conj().T / max(p_m_nc, 1e-10)
rho_post_nc = p_p_nc * rho_p_nc + p_m_nc * rho_m_nc

print(f"\nCOHERENCE COMPARISON")
print(f"Initial coherence:              {coh_initial:.5f}")
print(f"\nWITHOUT catalyst:")
print(f"  After measurement (avg):      {coherence_rel_entropy(rho_post_nc):.5f}")
print(f"  (+ branch):                  {coherence_rel_entropy(rho_p_nc):.5f}")
print(f"  (- branch):                  {coherence_rel_entropy(rho_m_nc):.5f}")
print(f"\nWITH catalyst ({best_config['desc']}):")
print(f"  After catalyst:               {coherence_rel_entropy(rho_sys_after_cat):.5f}")
print(f"  (+ branch):                  {coherence_rel_entropy(rho_p):.5f}")
print(f"  (- branch):                  {coherence_rel_entropy(rho_m):.5f}")

# Entropy production (optimal g) 

print(f"\nENTROPY PRODUCTION (optimal g={g_opt:.3f}, t={t_opt:.4f})")

S_sys_after_cat   = von_neumann_entropy(rho_sys_after_cat)
S_cat_initial     = von_neumann_entropy(rho_cat_opt)
S_cat_after       = von_neumann_entropy(rho_cat_f_opt)
S_joint_after_cat = von_neumann_entropy(rho_joint_opt)
dS_joint_unitary  = S_joint_after_cat - (S_sys_initial + S_cat_initial)

print(f"\nStage 1 - catalytic unitary:")
print(f"  S(sys) before:       {S_sys_initial:.5f}")
print(f"  S(sys) after:        {S_sys_after_cat:.5f}")
print(f"  S(cat) before:       {S_cat_initial:.5f}")
print(f"  S(cat) after:        {S_cat_after:.5f}")
print(f"  dS_joint (should~0): {dS_joint_unitary:.6f}")

rho_post_meas  = p_p * rho_p + p_m * rho_m
S_post_meas    = von_neumann_entropy(rho_post_meas)
dS_measurement = S_post_meas - S_sys_after_cat
print(f"\nStage 2 - weak measurement:")
print(f"  S(sys) before:       {S_sys_after_cat:.5f}")
print(f"  S(sys) after (avg):  {S_post_meas:.5f}")
print(f"  dS_measurement:      {dS_measurement:.5f}")

U_p, _ = optimize_feedback(rho_p)
U_m, _ = optimize_feedback(rho_m)
rho_p_fb    = U_p @ rho_p @ U_p.conj().T
rho_m_fb    = U_m @ rho_m @ U_m.conj().T
rho_post_fb = p_p * rho_p_fb + p_m * rho_m_fb
S_post_fb   = von_neumann_entropy(rho_post_fb)
dS_feedback = S_post_fb - S_post_meas
print(f"\nStage 3 - feedback unitary:")
print(f"  S(sys) before:       {S_post_meas:.5f}")
print(f"  S(sys) after:        {S_post_fb:.5f}")
print(f"  dS_feedback:         {dS_feedback:.5f}")

print(f"\nStage 4 - demon erasure:")
print(f"  dS_demon (erasure):  {np.log(2):.5f}  (= ln2, independent of T_D)")

dS_sys_total = S_post_fb - S_sys_initial
dS_cat_total = S_cat_after - S_cat_initial
sigma_tot    = dS_sys_total + dS_cat_total + np.log(2)
print(f"\nTotal entropy production:")
print(f"  dS_sys (full cycle): {dS_sys_total:.5f}")
print(f"  dS_cat (full cycle): {dS_cat_total:.5f}")
print(f"  dS_demon (erasure):  {np.log(2):.5f}")
print(f"  sigma_tot = {sigma_tot:.5f}  "
      f"{'... Second law satisfied' if sigma_tot >= -1e-10 else 'Second law VIOLATED'}")

# g sweep at fixed optimal t

g_range           = np.linspace(0.05, 1.5, 40)
W_cat_range       = []
fidelity_range    = []
W_raw_list        = []
W_net_list        = []
entanglement_range = []
coherence_range   = []
sigma_vs_g        = []
dS_sys_vs_g       = []
dS_cat_vs_g       = []

eta_dict = {TD: [] for TD in TD_values}
cop_dict = {TD: [] for TD in TD_values}

for g in g_range:
    rho_joint_g, rho_sys_g, rho_cat_g = evolve_joint(rho_sys, rho_cat_opt, g, t_opt)
    F = state_fidelity(rho_cat_opt, rho_cat_g)
    fidelity_range.append(F)

    W_raw, p_p_g, rho_p_g, rho_m_g = measure_and_feedback(rho_sys_g)
    W_raw_list.append(W_raw)
    cat_cost = catalyst_energy_cost(rho_cat_opt, rho_cat_g)
    W_net_g = W_raw - cat_cost - Wer_main
    W_net_list.append(W_net_g)
    W_cat_range.append(W_raw - cat_cost)

    for TD in TD_values:
        Wer = TD * np.log(2)
        W_eff = W_raw - cat_cost
        eta_dict[TD].append(W_eff / (W_eff + Wer) if (W_eff + Wer) > 0 else 0)
        cop_dict[TD].append(W_eff / Wer)

    entanglement_range.append(negativity(rho_joint_g, dim_sys, dim_cat))
    coherence_range.append(coherence_rel_entropy(rho_sys_g))

    p_m_g = 1 - p_p_g
    U_pg, _ = optimize_feedback(rho_p_g)
    U_mg, _ = optimize_feedback(rho_m_g)
    rho_fb_g = (p_p_g * (U_pg @ rho_p_g @ U_pg.conj().T) +
                p_m_g * (U_mg @ rho_m_g @ U_mg.conj().T))
    dS_s = von_neumann_entropy(rho_fb_g) - S_sys_initial
    dS_c = von_neumann_entropy(rho_cat_g) - von_neumann_entropy(rho_cat_opt)
    dS_sys_vs_g.append(dS_s)
    dS_cat_vs_g.append(dS_c)
    sigma_vs_g.append(dS_s + dS_c + np.log(2))

# Summary

print("SUMMARY:")
print(f"\nWork extraction:")
print(f"  Baseline (no catalyst):  {W_avg_no_cat:.5f}")
print(f"  Ground cat (Rabi t):     {best_ground_rabi[0]:.5f}  "
      f"({((best_ground_rabi[0]/W_avg_no_cat)-1)*100:+.2f}%)")
print(f"  Excited cat (Rabi t):    {best_excited_rabi[0]:.5f}  "
      f"({((best_excited_rabi[0]/W_avg_no_cat)-1)*100:+.2f}%)")
print(f"  Best (full scan):        {best_config['W']:.5f}  "
      f"({((best_config['W']/W_avg_no_cat)-1)*100:+.2f}%)  "
      f"<- {best_config['desc']}")
print(f"\nAt Rabi return (g={g_opt_rabi:.3f}, t={t_opt_rabi:.3f}):")
print(f"  N(S,C)={N_rabi:.5f}, I(S:C)={MI_rabi:.5f}, C(A,B)={C_rabi:.5f}")
print(f"\nDemon performance (best genuine catalyst, after erasure):")
for TD in TD_values:
    Wer = TD * np.log(2)
    net = best_config['W'] - Wer
    print(f"  T_D={TD}: W_er={Wer:.5f}, net gain={net:.5f} "
          f"{'Yes :)' if net > 0 else 'No :('}")
print(f"\nsigma_tot = {sigma_tot:.5f} (second law satisfied)")

# Noise channels

def apply_kraus(rho, kraus_ops):
    return sum(K @ rho @ K.conj().T for K in kraus_ops)

def kraus_amplitude_damping(p):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]],     dtype=complex)
    return [K0, K1]

def kraus_dephasing(p):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(p)]],     dtype=complex)
    return [K0, K1]

def kraus_depolarising(p):
    return [np.sqrt(1 - p) * I,
            np.sqrt(p / 3) * sigma_x,
            np.sqrt(p / 3) * sigma_y,
            np.sqrt(p / 3) * sigma_z]

def apply_noise_joint_16(rho_joint, kraus_sys, kraus_cat):
    sys_kraus_4  = [np.kron(Ka, Kb) for Ka in kraus_sys  for Kb in kraus_sys]
    cat_kraus_4  = [np.kron(Kc, Kd) for Kc in kraus_cat  for Kd in kraus_cat]
    joint_kraus  = [np.kron(Ks, Kc) for Ks in sys_kraus_4 for Kc in cat_kraus_4]
    return apply_kraus(rho_joint, joint_kraus)

# Noiseless baseline (fixed at optimal g, t from full scan)

G_OPT_FIXED = g_opt
T_OPT_FIXED = t_opt

rho_jt_base, rho_sys_base, rho_cat_base = evolve_joint(
    rho_sys, rho_cat_opt, G_OPT_FIXED, T_OPT_FIXED
)
W_base_noiseless, *_ = measure_and_feedback(rho_sys_base)
F_base = state_fidelity(rho_cat_opt, rho_cat_base)
N_base = negativity(rho_jt_base, dim_sys, dim_cat)
C_base = concurrence_2qubit(rho_sys_base)

print(f"\nNOISELESS BASELINE  (g={G_OPT_FIXED:.3f}, t={T_OPT_FIXED:.4f})")
print(f"  W_ext      = {W_base_noiseless:.5f}")
print(f"  Fidelity   = {F_base:.5f}")
print(f"  Negativity = {N_base:.5f}")
print(f"  Concurrence = {C_base:.5f}")

# Noise sweep

p_noise_range = np.linspace(0.0, 0.2, 60)

noise_channels = {
    'Amplitude Damping': kraus_amplitude_damping,
    'Dephasing':         kraus_dephasing,
    'Depolarising':      kraus_depolarising,
}

noise_W = {name: [] for name in noise_channels}
noise_F = {name: [] for name in noise_channels}
noise_N = {name: [] for name in noise_channels}

for name, kraus_fn in noise_channels.items():
    for p in p_noise_range:
        # 1. Noiseless unitary evolution at fixed (g_opt, t_opt)
        rho_jt, _, _ = evolve_joint(rho_sys, rho_cat_opt, G_OPT_FIXED, T_OPT_FIXED)

        # 2. Apply noise to full 16x16 joint state after catalytic unitary
        kraus_ops    = kraus_fn(p)
        rho_jt_noisy = apply_noise_joint_16(rho_jt, kraus_ops, kraus_ops)

        # 3. Marginals after noise
        rho_sys_n = partial_trace_system(rho_jt_noisy)
        rho_cat_n = partial_trace_catalyst(rho_jt_noisy)

        # 4. Metrics
        W_n = measure_and_feedback(rho_sys_n)[0]
        F_n = state_fidelity(rho_cat_opt, rho_cat_n)
        N_n = negativity(rho_jt_noisy, dim_sys, dim_cat)

        noise_W[name].append(W_n)
        noise_F[name].append(F_n)
        noise_N[name].append(N_n)

# Noise plots

channel_colors = ['#08306B', '#2171B5', '#6BAED6']
channel_ls     = ['-', '--', ':']

# Work extraction 
plt.figure(figsize=(7, 5))
plt.axhline(W_base_noiseless, color='red', linewidth=1.8,
            linestyle='dashdot', label='Noiseless baseline')
plt.axhline(W_avg_no_cat, color='gray', linewidth=1.5,
            linestyle='--', label='No-catalyst baseline')
for (name, vals), col, ls in zip(noise_W.items(), channel_colors, channel_ls):
    plt.plot(p_noise_range, vals, color=col, linewidth=2.5,
             linestyle=ls, label=name)
plt.xlabel('$p$', fontsize=14)
plt.ylabel(r'$W_{\mathrm{ext}}$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig('noise_w_ext_2q.png', dpi=300)
plt.show()

# Catalyst fidelity 
plt.figure(figsize=(7, 5))
plt.axhline(F_base, color='red', linewidth=1.8,
            linestyle='dashdot', label='Noiseless baseline')
plt.axhline(0.90, color='black', linewidth=1.5,
            linestyle=':', label='Catalyst threshold F=0.90')
for (name, vals), col, ls in zip(noise_F.items(), channel_colors, channel_ls):
    plt.plot(p_noise_range, vals, color=col, linewidth=2.5,
             linestyle=ls, label=name)
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$F$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig('noise_fidelity_2q.png', dpi=300)
plt.show()

# System-catalyst negativity
plt.figure(figsize=(7, 5))
plt.axhline(N_base, color='red', linewidth=1.8,
            linestyle='dashdot', label='Noiseless baseline')
for (name, vals), col, ls in zip(noise_N.items(), channel_colors, channel_ls):
    plt.plot(p_noise_range, vals, color=col, linewidth=2.5,
             linestyle=ls, label=name)
plt.xlabel('$p$', fontsize=14)
plt.ylabel(r'$\mathcal{N}$', fontsize=14)
plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.98, 0.95))
plt.grid(True, alpha=0.3)
plt.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig('noise_negativity_2q.png', dpi=300)
plt.show()

# Noise summary

print(f"\nNOISE SUMMARY  (g={G_OPT_FIXED:.3f}, t={T_OPT_FIXED:.4f})")
print(f"Noiseless: W={W_base_noiseless:.5f}, F={F_base:.4f}, "
      f"N={N_base:.4f}, C={C_base:.4f}")
print(f"\n{'Channel':>20}  {'p':>5}  {'W':>9}  {'F':>8}  {'N':>8}  {'W degr%':>8}")

for p_target in [0.01, 0.05, 0.1, 0.2]:
    idx = np.argmin(np.abs(p_noise_range - p_target))
    print(f"\n  p = {p_target}")
    for name in noise_channels:
        W = noise_W[name][idx]
        F = noise_F[name][idx]
        N = noise_N[name][idx]
        pct = (W / W_base_noiseless - 1) * 100
        print(f"  {name:>20}  {p_target:>5.2f}  {W:>9.5f}  {F:>8.4f}  "
              f"{N:>8.4f}  {pct:>+7.2f}%")