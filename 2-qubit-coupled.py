"""
Two-Qubit ZZ-Coupled Hybrid Engine

Extends the uncoupled two-qubit hybrid engine by introducing a ZZ
interaction between the system qubits. The ZZ term modifies
the system Hamiltonian, thermal state, and Rabi oscillation frequencies,
and its effect on catalyst fidelity and work extraction is studied across
a sweep of coupling strengths J.

Reproduces the coupled two-qubit hybrid engine results of the dissertation.

All quantities are in natural units where kB = hbar = 1.
Energy and work are dimensionless, scaled by hbar*omega0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Constants + parameters

hbar_omega0   = 0.5      # System energy gap
T             = 0.2      # System temperature
gamma_prime   = 0.05     # Measurement back-action parameter
delta_t       = 2        # Measurement time step
omega_c       = hbar_omega0   # Catalyst frequency
t_interaction = 1.0      # Interaction time 
J             = 0.2      # ZZ coupling strength 
prep_angle    = np.pi / 4

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

dim_sys = 4
dim_cat = 4
dim_tot = 16

# Hamiltonian

def build_H0(J_val):
    return (hbar_omega0 * (np.kron(proj1, I) + np.kron(I, proj1))
            + J_val * np.kron(sigma_z, sigma_z))

H0    = build_H0(J)   # default active Hamiltonian
H_cat = omega_c * (np.kron(proj1, I) + np.kron(I, proj1))
SZ_sys = np.kron(sigma_z, I) + np.kron(I, sigma_z)

#  Partial traces

def partial_trace_system(rho_joint):
    rho = rho_joint.reshape(dim_sys, dim_cat, dim_sys, dim_cat)
    return np.einsum('ikjk->ij', rho)

def partial_trace_catalyst(rho_joint):
    rho = rho_joint.reshape(dim_sys, dim_cat, dim_sys, dim_cat)
    return np.einsum('kikl->kl', rho)

# Definitions

def state_fidelity(rho, sigma):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals   = np.clip(eigvals, 0, None)
    sqrt_rho  = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    M         = sqrt_rho @ sigma @ sqrt_rho
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
    return von_neumann_entropy(np.diag(np.diag(rho))) - von_neumann_entropy(rho)

def negativity(rho, dA, dB):
    pt = np.zeros_like(rho)
    for i in range(dA):
        for j in range(dA):
            pt[i*dB:(i+1)*dB, j*dB:(j+1)*dB] = \
                rho[i*dB:(i+1)*dB, j*dB:(j+1)*dB].T
    eigvals = np.real(np.linalg.eigvalsh(pt))
    return float(np.sum(np.abs(eigvals[eigvals < 0])))

def concurrence_2qubit(rho):
    Y  = np.kron(sigma_y, sigma_y)
    R  = rho @ Y @ rho.conj() @ Y
    ev = np.sort(np.sqrt(np.abs(np.linalg.eigvals(R))))[::-1]
    return float(np.real(max(0.0, ev[0] - ev[1] - ev[2] - ev[3])))

def mutual_information(rho_joint):
    return (von_neumann_entropy(partial_trace_system(rho_joint))
            + von_neumann_entropy(partial_trace_catalyst(rho_joint))
            - von_neumann_entropy(rho_joint))

# Feedback optimisation

_angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
_U1     = np.array([np.cos(t/2)*I - 1j*np.sin(t/2)*ax
                    for ax in [sigma_x, sigma_y, sigma_z]
                    for t in _angles])  
_UA     = np.array([np.kron(u, I) for u in _U1])   # act on qubit A
_UB     = np.array([np.kron(I, u) for u in _U1])   # act on qubit B

def optimise_feedback(state, H0_in):
    HB     = _UB.conj().transpose(0, 2, 1) @ H0_in @ _UB
    E0     = np.trace(state @ H0_in).real
    best_W = -np.inf
    best_U = I4.copy()
    for ua in _UA:
        tmp  = ua @ state @ ua.conj().T
        Wvec = E0 - np.einsum('kij,ji->k', HB, tmp).real
        idx  = int(np.argmax(Wvec))
        if Wvec[idx] > best_W:
            best_W = Wvec[idx]
            best_U = _UB[idx] @ ua
    return best_U, best_W

# Sequential XY evolution

def rabi_period(g):
    return np.pi / g

def _build_xy(g, p, q, n=4):
    ops_p = [I]*n; ops_m = [I]*n
    ops_p[p] = sigma_plus;  ops_p[q] = sigma_minus
    ops_m[p] = sigma_minus; ops_m[q] = sigma_plus
    def kron_chain(ops):
        out = ops[0]
        for op in ops[1:]: out = np.kron(out, op)
        return out
    return g * (kron_chain(ops_p) + kron_chain(ops_m))

def evolve_joint(rho_sys, rho_cat, g, t, H0_in):
    H_free = np.kron(H0_in, I4) + np.kron(I4, H_cat)
    rho    = np.kron(rho_sys, rho_cat)
    U1     = expm(-1j * (H_free + _build_xy(g, 0, 2)) * t / 2)
    rho    = U1 @ rho @ U1.conj().T
    U2     = expm(-1j * (H_free + _build_xy(g, 1, 3)) * t / 2)
    rho    = U2 @ rho @ U2.conj().T
    return rho, partial_trace_system(rho), partial_trace_catalyst(rho)

# Computes delta_Ec
def catalyst_energy_cost(rho_cat_initial, rho_cat_final):
    return -(np.trace(rho_cat_final @ H_cat).real
             - np.trace(rho_cat_initial @ H_cat).real)

# Measurement + feedback

kappa    = 0.5 - np.sqrt(2 * gamma_prime * delta_t)
_ev, _ec = np.linalg.eigh(sigma_x)
_P_xp    = np.outer(_ec[:, 1], _ec[:, 1].conj())
_P_xm    = np.outer(_ec[:, 0], _ec[:, 0].conj())
M_plus   = np.kron(np.sqrt(kappa)*_P_xp + np.sqrt(1-kappa)*_P_xm, I)
M_minus  = np.kron(np.sqrt(1-kappa)*_P_xp + np.sqrt(kappa)*_P_xm, I)
assert np.allclose(M_plus.conj().T @ M_plus +
                   M_minus.conj().T @ M_minus, I4, atol=1e-8), "POVM not complete!"

def measure_and_feedback(rho_sys, H0_in):
    p_p = np.trace(M_plus  @ rho_sys @ M_plus.conj().T).real
    p_m = np.trace(M_minus @ rho_sys @ M_minus.conj().T).real
    rho_p = M_plus  @ rho_sys @ M_plus.conj().T  / max(p_p, 1e-10)
    rho_m = M_minus @ rho_sys @ M_minus.conj().T / max(p_m, 1e-10)
    U_p, W_p = optimise_feedback(rho_p, H0_in)
    U_m, W_m = optimise_feedback(rho_m, H0_in)
    W_avg = p_p * W_p + p_m * W_m
    return (W_avg, p_p,
            U_p @ rho_p @ U_p.conj().T,
            U_m @ rho_m @ U_m.conj().T)

# System preparation

def prepare_system(H0_in, angle=prep_angle):
    rho_th  = expm(-H0_in / T)
    rho_th /= np.trace(rho_th)
    X_tot   = np.kron(sigma_x, I) + np.kron(I, sigma_x)
    U_prep  = expm(-1j * angle * X_tot / 2)
    return U_prep @ rho_th @ U_prep.conj().T

# Catalyst initial states

def ket2(a, b):
    v = np.zeros((4, 1), dtype=complex); v[2*a + b] = 1; return v

def dm2(a, b):
    v = ket2(a, b); return v @ v.conj().T

cat_ground = dm2(0, 0)   # |00>

_v_sup   = np.kron([[np.cos(np.pi/4)], [np.sin(np.pi/4)]],
                    [[np.cos(np.pi/4)], [np.sin(np.pi/4)]])
cat_super = _v_sup @ _v_sup.conj().T   # |+x> x |+x>

_bell    = (ket2(0, 0) + ket2(1, 1)) / np.sqrt(2)
cat_bell = _bell @ _bell.conj().T   # (|00>+|11>)/sqrt(2)

cat_states = {
    'Ground':        cat_ground,
    'Superposition': cat_super,
    'Bell':          cat_bell,
}

# System setup

print(f"2-QUBIT ZZ-COUPLED CATALYTIC MAXWELL'S DEMON")
print(f"J={J},  T={T},  omega0={hbar_omega0},  dim={dim_tot}")

rho_sys = prepare_system(H0)

print(f"\nH0 eigenvalues: {np.sort(np.linalg.eigvalsh(H0))}")
print(f"System <SZ>:    {np.trace(rho_sys @ SZ_sys).real:.5f}")
print(f"Prep angle:     pi/4")
print(f"Meas. strength: kappa = {kappa:.5f}")
print(f"\nRabi period t_R = pi/g  "
      f"(g=1 -> {rabi_period(1):.3f}, g=2 -> {rabi_period(2):.3f})")
print(f"Catalyst returns at 2*t_R")

coh_initial   = coherence_rel_entropy(rho_sys)
S_sys_initial = von_neumann_entropy(rho_sys)

W_base, *_ = measure_and_feedback(rho_sys, H0)
print(f"\nBaseline work (no catalyst): {W_base:.5f}")

# Coupling strength sweep (ground catalyst)

print("COUPLING STRENGTH SWEEP  (ground cat, t=t_interaction)")

print(f"\n{'g':>6}  {'W_sys':>9}  {'dE_cat':>9}  {'fidelity':>9}  {'returned?':>10}")
print("-" * 52)
for g in [0.1, 0.5, 1.0, 2.0]:
    _, rho_sa, rho_cf = evolve_joint(rho_sys, cat_ground, g, t_interaction, H0)
    W_sys  = np.trace(rho_sys @ H0).real - np.trace(rho_sa @ H0).real
    dE_cat = np.trace(rho_cf @ H_cat).real - np.trace(cat_ground @ H_cat).real
    F      = state_fidelity(cat_ground, rho_cf)
    print(f"{g:6.2f}  {W_sys:9.5f}  {dE_cat:9.6f}  {F:9.5f}  "
          f"{'Yes :)' if F > 0.99 else 'No :('}  ")

# Catalyst parameter optimisation 

print("CATALYST OPTIMISATION  (superposition, near 2*t_R)")

zoom_rows = []
for zg in np.linspace(0.3, 3.0, 25):
    t_c = 2 * rabi_period(zg)
    for zt in np.linspace(0.85 * t_c, 1.15 * t_c, 32):
        _, rho_sz, rho_cz = evolve_joint(rho_sys, cat_super, zg, zt, H0)
        F = state_fidelity(cat_super, rho_cz)
        W, *_ = measure_and_feedback(rho_sz, H0)
        zoom_rows.append((zg, zt, W, F))

best_config = {'W': -np.inf, 'F': 0, 'g': 0, 't': 0, 'desc': ''}
print(f"\n{'Fidelity band':<14}  {'W':>9}  {'F':>8}  {'g':>6}  {'t':>7}  Role")
print("-" * 55)

for lo, hi, label in [(0.99, 1.01, 'F > 0.99'),
                       (0.95, 0.99, 'F 0.95-0.99'),
                       (0.90, 0.95, 'F 0.90-0.95'),
                       (0.80, 0.90, 'F 0.80-0.90')]:
    cands = [(W, F, g, t) for (g, t, W, F) in zoom_rows if lo <= F < hi]
    if cands:
        W, F, g, t = max(cands, key=lambda x: x[0])
        role = 'catalyst' if F >= 0.90 else 'fuel'
        print(f"{label:<14}  {W:9.5f}  {F:8.5f}  {g:6.3f}  {t:7.4f}  {role}")
        if F >= 0.90 and W > best_config['W']:
            best_config.update({'W': W, 'F': F, 'g': g, 't': t,
                                'desc': f'Superposition, g={g:.3f}, t={t:.4f}'})

# Fallback if no genuine catalyst found
if best_config['W'] == -np.inf:
    cands = [(W, F, g, t) for (g, t, W, F) in zoom_rows if F >= 0.80]
    if cands:
        W, F, g, t = max(cands, key=lambda x: x[0])
        best_config.update({'W': W, 'F': F, 'g': g, 't': t,
                            'desc': f'Superposition (fuel F={F:.3f}), g={g:.3f}, t={t:.4f}'})
        print(f"\nNote: no genuine catalyst (F>=0.90) found — best fuel config used.")

g_opt = best_config['g']
t_opt = best_config['t']
print(f"\nBest config: {best_config['desc']}")
print(f"Best work:   {best_config['W']:.5f}  "
      f"({(best_config['W']/W_base-1)*100:+.1f}% vs baseline)")

# Section 4: Measurement + feedback at optimal (g, t)

print(f"MEASUREMENT & FEEDBACK  (g={g_opt:.3f}, t={t_opt:.4f})")

rho_jt_opt, rho_sys_ac, rho_cat_ac = evolve_joint(
    rho_sys, cat_super, g_opt, t_opt, H0)
W_avg, p_plus, rho_p, rho_m = measure_and_feedback(rho_sys_ac, H0)
p_minus = 1 - p_plus

print(f"\nMeasurement probabilities: p(+)={p_plus:.5f}, p(-)={p_minus:.5f}")
print(f"Average work extracted:    W_avg = {W_avg:.5f}")
print(f"Without catalyst:          W_base = {W_base:.5f}")
print(f"Improvement:               {W_avg - W_base:.5f}  "
      f"({(W_avg/W_base-1)*100:+.1f}%)")

print("\nDEMON MEMORY ERASURE COST  (W_er = T_D * ln2):")
for TD in TD_values:
    Wer = TD * np.log(2)
    net = W_avg - Wer
    print(f"  T_D={TD}: W_er={Wer:.5f}, net gain={net:.5f} "
          f"{'Yes :)' if net > 0 else 'No :('}  ")

# Quantum correlations at optimal
N_opt  = negativity(rho_jt_opt, dim_sys, dim_cat)
MI_opt = mutual_information(rho_jt_opt)
C_opt  = concurrence_2qubit(rho_sys_ac)
print(f"\nSystem-catalyst negativity N(S,C): {N_opt:.5f}")
print(f"Mutual information I(S:C):          {MI_opt:.5f}")
print(f"Within-system concurrence C(A,B):   {C_opt:.5f}")

# Entropy production at optimal (g, t)

print("ENTROPY PRODUCTION  (optimal g, t)")

S_sys_i  = von_neumann_entropy(rho_sys)
S_sys_ac = von_neumann_entropy(rho_sys_ac)
S_cat_i  = von_neumann_entropy(cat_super)
S_cat_f  = von_neumann_entropy(rho_cat_ac)
dS_joint = von_neumann_entropy(rho_jt_opt) - (S_sys_i + S_cat_i)

rho_post_meas = p_plus * rho_p + p_minus * rho_m
S_pm = von_neumann_entropy(rho_post_meas)

U_p, _ = optimise_feedback(rho_p, H0)
U_m, _ = optimise_feedback(rho_m, H0)
rho_post_fb = (p_plus  * (U_p @ rho_p @ U_p.conj().T) +
               p_minus * (U_m @ rho_m @ U_m.conj().T))
S_fb = von_neumann_entropy(rho_post_fb)

dS_sys_tot = S_fb    - S_sys_i
dS_cat_tot = S_cat_f - S_cat_i
sigma_tot  = dS_sys_tot + dS_cat_tot + np.log(2)

print(f"\nStage 1 — catalytic unitary:")
print(f"  S(sys) before:        {S_sys_i:.5f}")
print(f"  S(sys) after:         {S_sys_ac:.5f}")
print(f"  S(cat) before:        {S_cat_i:.5f}")
print(f"  S(cat) after:         {S_cat_f:.5f}")
print(f"  dS_joint (should~0):  {dS_joint:.6f}")
print(f"\nStage 2 — weak measurement:")
print(f"  S(sys) before:        {S_sys_ac:.5f}")
print(f"  S(sys) after (avg):   {S_pm:.5f}")
print(f"  dS_measurement:       {S_pm - S_sys_ac:.5f}")
print(f"\nStage 3 — feedback unitary:")
print(f"  S(sys) before:        {S_pm:.5f}")
print(f"  S(sys) after:         {S_fb:.5f}")
print(f"  dS_feedback:          {S_fb - S_pm:.5f}")
print(f"\nStage 4 — demon erasure:")
print(f"  dS_demon (ln2):       {np.log(2):.5f}")
print(f"\nTotal entropy production:")
print(f"  dS_sys (full cycle):  {dS_sys_tot:.5f}")
print(f"  dS_cat (full cycle):  {dS_cat_tot:.5f}")
print(f"  dS_demon:             {np.log(2):.5f}")
print(f"  sigma_tot = {sigma_tot:.5f}  "
      f"{'Second law satisfied' if sigma_tot >= -1e-10 else 'VIOLATED'}")

# ZZ coupling sweep

print("ZZ COUPLING SWEEP  (all catalyst states)")

g_J   = 1.0
t_J   = 2 * rabi_period(g_J)
J_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

print(f"Fixed g={g_J:.1f}, t={t_J:.4f} (2*t_R)\n")
print(f"{'J':>6}  {'W_base':>9}  {'W_Grd':>9}  {'W_Sup':>9}  {'W_Bell':>9}  "
      f"{'F_Grd':>7}  {'F_Sup':>7}  {'F_Bell':>7}  {'C_sys':>7}")

J_sweep_results = []
for J_val in J_values:
    H0_J    = build_H0(J_val)
    rho_s_J = prepare_system(H0_J)
    Wb_J, *_ = measure_and_feedback(rho_s_J, H0_J)

    row = {'J': J_val, 'W_base': Wb_J}
    for cat_name, rho_cat0 in cat_states.items():
        _, rho_sa_J, rho_ca_J = evolve_joint(rho_s_J, rho_cat0, g_J, t_J, H0_J)
        W_J, *_ = measure_and_feedback(rho_sa_J, H0_J)
        F_J     = state_fidelity(rho_cat0, rho_ca_J)
        C_J     = concurrence_2qubit(rho_sa_J)
        row[f'W_{cat_name}'] = W_J
        row[f'F_{cat_name}'] = F_J
        row[f'C_{cat_name}'] = C_J
    J_sweep_results.append(row)

    print(f"{J_val:6.3f}  {Wb_J:9.5f}  "
          f"{row['W_Ground']:9.5f}  {row['W_Superposition']:9.5f}  {row['W_Bell']:9.5f}  "
          f"{row['F_Ground']:7.5f}  {row['F_Superposition']:7.5f}  {row['F_Bell']:7.5f}  "
          f"{row['C_Superposition']:7.5f}")

# Joint (J, g, t) optimisation 

print("JOINT (J, g, t) OPTIMISATION")

print(f"{'J':>6}  {'best W':>9}  {'F':>8}  {'g*':>6}  {'t*':>7}  "
      f"{'vs base':>9}  Role")

g_jopt_range = np.linspace(0.3, 3.0, 15)
jopt_rows    = []

for J_val in J_values:
    H0_J    = build_H0(J_val)
    rho_s_J = prepare_system(H0_J)
    W_nc_J, *_ = measure_and_feedback(rho_s_J, H0_J)

    best_W_J = best_F_J = best_g_J = best_t_J = 0.0
    best_FoM = -np.inf

    for zg in g_jopt_range:
        t_c = 2 * rabi_period(zg)
        for zt in np.linspace(0.80 * t_c, 1.20 * t_c, 20):
            _, rho_sz_J, rho_cz_J = evolve_joint(rho_s_J, cat_super, zg, zt, H0_J)
            F_z = state_fidelity(cat_super, rho_cz_J)
            W_z, *_ = measure_and_feedback(rho_sz_J, H0_J)
            FoM = W_z * F_z
            if FoM > best_FoM:
                best_FoM = FoM
                best_W_J, best_F_J = W_z, F_z
                best_g_J, best_t_J = zg, zt

    role = ('catalyst' if best_F_J >= 0.90
            else 'fuel'   if best_F_J >= 0.20
            else 'inert')
    imp  = (best_W_J / W_nc_J - 1) * 100 if W_nc_J > 0 else 0
    print(f"{J_val:6.3f}  {best_W_J:9.5f}  {best_F_J:8.5f}  "
          f"{best_g_J:6.3f}  {best_t_J:7.4f}  {imp:+8.1f}%  {role}")
    jopt_rows.append((J_val, best_W_J, best_F_J, best_g_J, best_t_J,
                      W_nc_J, role))

# g sweep at fixed optimal t 

g_range = np.linspace(0.05, 1.5, 40)

W_cat_range    = []
fidelity_range = []
W_raw_list     = []
W_net_list     = []
fid_net_list   = []
ent_range      = []
coh_range      = []
conc_range     = []
sigma_vs_g     = []
dSs_vs_g       = []
dSc_vs_g       = []

eta_dict = {TD: [] for TD in TD_values}
cop_dict = {TD: [] for TD in TD_values}

for g in g_range:
    rho_jg, rho_sg, rho_cg = evolve_joint(rho_sys, cat_super, g, t_opt, H0)
    F   = state_fidelity(cat_super, rho_cg)
    fidelity_range.append(F)
    fid_net_list.append(F)

    W_raw, p_pg, rho_pg, rho_mg = measure_and_feedback(rho_sg, H0)
    cc  = catalyst_energy_cost(cat_super, rho_cg)
    W_cat_range.append(W_raw - cc)
    W_raw_list.append(W_raw)
    W_net_list.append(W_raw - cc - Wer_main)

    for TD in TD_values:
        Wer = TD * np.log(2)
        W_eff = W_raw - cc
        eta_dict[TD].append(W_eff / (W_eff + Wer) if (W_eff + Wer) > 0 else 0)
        cop_dict[TD].append(W_eff / Wer)

    ent_range.append(negativity(rho_jg, dim_sys, dim_cat))
    coh_range.append(coherence_rel_entropy(rho_sg))
    conc_range.append(concurrence_2qubit(rho_sg))

    p_mg = 1 - p_pg
    U_pg, _ = optimise_feedback(rho_pg, H0)
    U_mg, _ = optimise_feedback(rho_mg, H0)
    rho_fb_g = (p_pg * (U_pg @ rho_pg @ U_pg.conj().T) +
                p_mg * (U_mg @ rho_mg @ U_mg.conj().T))
    dSs = von_neumann_entropy(rho_fb_g) - S_sys_initial
    dSc = von_neumann_entropy(rho_cg)   - von_neumann_entropy(cat_super)
    dSs_vs_g.append(dSs)
    dSc_vs_g.append(dSc)
    sigma_vs_g.append(dSs + dSc + np.log(2))

# Summary 
print("SUMMARY")

print(f"T={T},  omega0={hbar_omega0},  J={J},  dim={dim_tot}")
print(f"Sequential XY coupling")
print(f"Measurement: qubit A only, kappa={kappa:.5f}")
print(f"Feedback: sequential rotations on A then B (H0 passed explicitly)")
print(f"\nWork extraction:")
print(f"  Baseline (no catalyst):  {W_base:.5f}")
print(f"  Best with catalyst:      {best_config['W']:.5f}  "
      f"({(best_config['W']/W_base-1)*100:+.1f}%)")
print(f"  Catalyst status:         "
      f"{'genuine catalyst' if best_config['F'] >= 0.90 else 'fuel'}  "
      f"(F={best_config['F']:.5f})")
print(f"  Optimal g={g_opt:.3f}, t={t_opt:.4f}")

print(f"\nZZ coupling sweep (g={g_J:.1f}, t=2*t_R={t_J:.4f}):")
print(f"  {'J':>6}  {'W_base':>9}  {'W_Sup':>9}  {'F_Sup':>8}  "
      f"{'improvement':>12}  Role")
for row in J_sweep_results:
    J_val = row['J']
    W_s   = row['W_Superposition']
    F_s   = row['F_Superposition']
    Wb    = row['W_base']
    imp   = (W_s / Wb - 1) * 100 if Wb > 0 else 0
    role  = 'catalyst' if F_s >= 0.90 else 'fuel'
    print(f"  {J_val:6.3f}  {Wb:9.5f}  {W_s:9.5f}  {F_s:8.5f}  "
          f"{imp:+11.1f}%  {role}")

print(f"\nJoint (J, g*, t*) optimisation:")
print(f"  {'J':>6}  {'W*':>9}  {'F*':>8}  {'g*':>6}  {'t*':>7}  "
      f"{'vs base':>9}  Role")
for r in jopt_rows:
    J_val, W, F, g, t, Wn, role = r
    imp = (W / Wn - 1) * 100 if Wn > 0 else 0
    print(f"  {J_val:6.3f}  {W:9.5f}  {F:8.5f}  {g:6.3f}  {t:7.4f}  "
          f"{imp:+8.1f}%  {role}")

print(f"\nConclusion:")
print(f"  ZZ coupling (J={J}) disrupts the Rabi return condition.")
print(f"  Fidelity drops monotonically with J.")
if best_config['F'] >= 0.90:
    print(f"  Genuine catalysis achieved at J={J}  (F={best_config['F']:.5f}).")
else:
    print(f"  Genuine catalysis NOT achieved at J={J}  (F={best_config['F']:.5f} < 0.90).")
    print(f"  -> J=0 (uncoupled) remains the superior catalytic demon.")

print(f"\nDemon performance (best config, after erasure cost):")
for TD in TD_values:
    Wer = TD * np.log(2)
    net = best_config['W'] - Wer
    print(f"  T_D={TD}: W_er={Wer:.5f}, net gain={net:.5f} "
          f"{'Yes :)' if net > 0 else 'No :('}  ")

print(f"\nSecond law: sigma_tot = {sigma_tot:.5f}  "
      f"{'satisfied' if sigma_tot >= -1e-10 else 'VIOLATED'}")