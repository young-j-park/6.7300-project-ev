# sensitivity_adjoint_demo.py
# ------------------------------------------------------------
# Nonlinear equilibrium sensitivities for dx/dt=f(x,p,u), y=g(x,p,u)
# (a) FD sensitivities for x*, y* wrt p and u
# (b) step-size sweep
# (c) ranking by |dy*/d·|
# (d) adjoint sensitivities on the linearized steady-state system
# ------------------------------------------------------------

import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from vehicle_model_jax import (
    get_default_params, evalf,
    OUTPUT_POSITIONS  # <- state layout switch
)

jax.config.update("jax_enable_x64", True)

# ---------- state indexing ----------
if OUTPUT_POSITIONS:
    N_STATES = 10
    IDX_THETA, IDX_V, IDX_OMEGA = 6, 7, 8
else:
    N_STATES = 8
    IDX_THETA, IDX_V, IDX_OMEGA = 4, 5, 6

# ---------- independent parameters (exclude derived 'k', keep p_pairs fixed) ----------
PARAM_KEYS = [
    'm','I','c_d','c_r',
    'R_s','L_ds','L_qs','lambda_f',
    'G','r_w','eta_g',
    'K_p_i','K_i_i','K_p_v','K_i_v','K_p_theta','K_d_theta',
    'v_w','psi','c_wx','c_wy'
]

def refresh_k(p):
    p['k'] = jnp.asarray(p['eta_g'], dtype=jnp.float64) * \
             jnp.asarray(p['G'], dtype=jnp.float64)     / \
             jnp.asarray(p['r_w'], dtype=jnp.float64)
    return p

def params_to_vec(p):
    return jnp.array([jnp.asarray(p[k], dtype=jnp.float64) for k in PARAM_KEYS],
                     dtype=jnp.float64)

def vec_to_params(p_vec, p_template):
    q = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p_template.items()}
    for k, v in zip(PARAM_KEYS, p_vec):
        q[k] = jnp.asarray(v, dtype=jnp.float64)
    q['p_pairs'] = jnp.asarray(p_template['p_pairs'], dtype=jnp.float64)
    q['k'] = q['eta_g'] * q['G'] / q['r_w']
    return q

def pack_param_tuple(p):
    return (
        jnp.asarray(p['m'], dtype=jnp.float64),
        jnp.asarray(p['I'], dtype=jnp.float64),
        jnp.asarray(p['c_d'], dtype=jnp.float64),
        jnp.asarray(p['c_r'], dtype=jnp.float64),
        jnp.asarray(p['R_s'], dtype=jnp.float64),
        jnp.asarray(p['L_ds'], dtype=jnp.float64),
        jnp.asarray(p['L_qs'], dtype=jnp.float64),
        jnp.asarray(p['lambda_f'], dtype=jnp.float64),
        jnp.asarray(p['p_pairs'], dtype=jnp.float64),
        jnp.asarray(p['G'], dtype=jnp.float64),
        jnp.asarray(p['r_w'], dtype=jnp.float64),
        jnp.asarray(p['eta_g'], dtype=jnp.float64),
        jnp.asarray(p['k'], dtype=jnp.float64),
        jnp.asarray(p['K_p_i'], dtype=jnp.float64),
        jnp.asarray(p['K_i_i'], dtype=jnp.float64),
        jnp.asarray(p['K_p_v'], dtype=jnp.float64),
        jnp.asarray(p['K_i_v'], dtype=jnp.float64),
        jnp.asarray(p['K_p_theta'], dtype=jnp.float64),
        jnp.asarray(p['K_d_theta'], dtype=jnp.float64),
        jnp.asarray(p['v_w'], dtype=jnp.float64),
        jnp.asarray(p['psi'], dtype=jnp.float64),
        jnp.asarray(p['c_wx'], dtype=jnp.float64),
        jnp.asarray(p['c_wy'], dtype=jnp.float64),
    )

# ---------- explicit output: y = g(x,p,u) ----------
# Default: y = [v, omega]. (You can change to any metric you want.)
def g(x, p_tuple, u):
    return jnp.array([x[IDX_V], x[IDX_THETA]])

# Convenience wrappers for JAX jacobians wrt p_vec and u
def evalf_with_pvec(x, p_vec, u, p_template):
    p_now = vec_to_params(p_vec, p_template)
    return evalf(x, pack_param_tuple(p_now), u)

def evalf_with_uvec(x, p_tuple, u_vec):
    return evalf(x, p_tuple, u_vec)

def g_with_pvec(x, p_vec, u, p_template):
    p_now = vec_to_params(p_vec, p_template)
    return g(x, pack_param_tuple(p_now), u)

def g_with_uvec(x, p_tuple, u_vec):
    return g(x, p_tuple, u_vec)

# ---------- solve equilibrium f(x*,p,u)=0 ----------
def newton_equilibrium(p_dict, u, x0=None, tol=1e-10, maxit=50):
    p_tuple = pack_param_tuple(p_dict)
    if x0 is None:
        x0_np = np.zeros(N_STATES, dtype=np.float64)
        x0_np[IDX_V] = float(u[0])
        x0_np[IDX_THETA] = float(u[1])
        x = jnp.array(x0_np)
    else:
        x = jnp.array(x0)

    f_x = lambda x_: evalf(x_, p_tuple, u)
    jac_x = jax.jit(jax.jacobian(lambda xx: evalf(xx, p_tuple, u)))

    prev = np.inf
    for _ in range(maxit):
        Fx = f_x(x)
        res = float(jnp.linalg.norm(Fx))
        if res < tol: break
        A = jac_x(x)
        dx = jnp.linalg.solve(A, Fx)
        alpha = 1.0
        for _ in range(6):
            xn = x - alpha*dx
            if float(jnp.linalg.norm(f_x(xn))) <= res: x = xn; break
            alpha *= 0.5
        prev = res
    return x

# ---------- output picker y = C x (optional helper) ----------
def build_outputs_C(which=('speed','yaw_rate')):
    rows, names = [], []
    for w in which:
        c = np.zeros(N_STATES)
        if w == 'speed':      c[IDX_V] = 1.0
        elif w == 'yaw_rate': c[IDX_OMEGA] = 1.0
        elif w == 'theta':    c[IDX_THETA] = 1.0
        else: raise ValueError(f"Unknown output '{w}'")
        rows.append(c); names.append(w)
    return np.vstack(rows), names

# ---------- (a) FD sensitivities for x*, y* wrt p and u ----------
def step_rule(val, rule):
    sq = np.sqrt(np.finfo(float).eps)
    if rule == 'sqrt_eps': return sq
    if rule == 'scaled':   return 2.0*sq*max(1.0, abs(float(val)))
    raise ValueError

def fd_wrt_p(p_base, u, rule='scaled'):
    xstar = newton_equilibrium(p_base, u)
    pvec  = np.array(params_to_vec(p_base), dtype=np.float64)
    ystar = np.array(g(xstar, pack_param_tuple(p_base), u))

    n, m, q = N_STATES, len(pvec), ystar.size
    dxdP = np.zeros((n, m)); dydP = np.zeros((q, m))

    for j in range(m):
        dp = step_rule(pvec[j], rule)
        p_plus = pvec.copy(); p_plus[j] += dp
        p_plus_dict = vec_to_params(p_plus, p_base)

        x1 = newton_equilibrium(p_plus_dict, u, x0=xstar)
        y1 = np.array(g(x1, pack_param_tuple(p_plus_dict), u))

        dxdP[:, j] = (np.array(x1) - np.array(xstar)) / dp
        dydP[:, j] = (y1 - ystar) / dp

    return xstar, ystar, dxdP, dydP

def fd_wrt_u(p_base, u_base, rule='scaled'):
    xstar = newton_equilibrium(p_base, u_base)
    p_tuple = pack_param_tuple(p_base)
    ystar = np.array(g(xstar, p_tuple, u_base))

    nu = len(u_base); n = N_STATES; q = ystar.size
    dxdU = np.zeros((n, nu)); dydU = np.zeros((q, nu))

    for j in range(nu):
        du = step_rule(u_base[j], rule)
        u_plus = np.array(u_base); u_plus[j] += du

        x1 = newton_equilibrium(p_base, u_plus, x0=xstar)
        y1 = np.array(g(x1, p_tuple, u_plus))

        dxdU[:, j] = (np.array(x1) - np.array(xstar)) / du
        dydU[:, j] = (y1 - ystar) / du

    return xstar, ystar, dxdU, dydU

# ---------- (b) step-size sweep (example for one p and one u component) ----------
def sweep_param(p_base, u, pname, scales=np.logspace(-2, -12, 11)):
    idx = PARAM_KEYS.index(pname)
    x0 = newton_equilibrium(p_base, u)
    y0 = np.array(g(x0, pack_param_tuple(p_base), u))
    base_vec = np.array(params_to_vec(p_base), dtype=np.float64)
    ref = base_vec[idx] if base_vec[idx]!=0 else 1.0

    outs=[]
    for s in scales:
        dp = s*abs(ref)
        p_plus = base_vec.copy(); p_plus[idx]+=dp
        p_plus_dict = vec_to_params(p_plus, p_base)
        x1 = newton_equilibrium(p_plus_dict, u, x0=x0)
        y1 = np.array(g(x1, pack_param_tuple(p_plus_dict), u))
        outs.append((dp, (y1-y0)))
    return outs

def sweep_input(p_base, u_base, j=0, scales=np.logspace(-2, -12, 11)):
    x0 = newton_equilibrium(p_base, u_base)
    y0 = np.array(g(x0, pack_param_tuple(p_base), u_base))
    ref = u_base[j] if float(u_base[j])!=0.0 else 1.0

    outs=[]
    for s in scales:
        du = s*abs(ref)
        u_plus = np.array(u_base); u_plus[j]+=du
        x1 = newton_equilibrium(p_base, u_plus, x0=x0)
        y1 = np.array(g(x1, pack_param_tuple(p_base), u_plus))
        outs.append((du, (y1-y0)))
    return outs

# ---------- (c,d) linearization + adjoint ----------
def linearize_all(p_base, u, x_star):
    # A,E,B
    p_tuple = pack_param_tuple(p_base)
    A = jax.jacobian(lambda xx: evalf(xx, p_tuple, u))(x_star)           # (n,n)
    pv = params_to_vec(p_base)

    E = jax.jacobian(lambda pv_: evalf_with_pvec(x_star, pv_, u, p_base))(pv)   # (n,m)
    B = jax.jacobian(lambda uu: evalf_with_uvec(x_star, p_tuple, uu))(u)        # (n,nu)

    # C,Dp,Du
    C  = jax.jacobian(lambda xx: g(xx, p_tuple, u))(x_star)                    # (q,n)
    Dp = jax.jacobian(lambda pv_: g_with_pvec(x_star, pv_, u, p_base))(pv)     # (q,m)
    Du = jax.jacobian(lambda uu: g_with_uvec(x_star, p_tuple, uu))(u)          # (q,nu)

    return np.array(A), np.array(E), np.array(B), np.array(C), np.array(Dp), np.array(Du)

def adjoint_y_sensitivities(A, E, B, C, Dp, Du):
    # Solve A^T λ_k = C^T_k  and form y_sens
    AT = A.T
    q = C.shape[0]
    m = E.shape[1]; nu = B.shape[1]
    dYdP = np.zeros((q, m))
    dYdU = np.zeros((q, nu))
    for k in range(q):
        lam = np.linalg.solve(AT, C[k,:])
        dYdP[k,:] = - lam @ E + Dp[k,:]
        dYdU[k,:] = - lam @ B + Du[k,:]
    return dYdP, dYdU

def state_sensitivities(A, E, B):
    # x* sensitivities (optional)
    dXdP = - np.linalg.solve(A, E)   # (n,m)
    dXdU = - np.linalg.solve(A, B)   # (n,nu)
    return dXdP, dXdU

def rank_params(S, labels, topk=8, title=''):
    print(f"\n[Ranking] {title}")
    mags = np.abs(S)
    if S.ndim==1:
        order = np.argsort(mags)[::-1]
        for i in range(min(topk, len(order))):
            j = order[i]
            print(f"  {i+1:2d}. {labels[j]:>10s} : {S[j]: .3e} (|.|={mags[j]:.3e})")
    else:
        for k in range(S.shape[0]):
            order = np.argsort(mags[k,:])[::-1]
            print(f"  - output[{k}]")
            for i in range(min(topk, len(order))):
                j = order[i]
                print(f"    {i+1:2d}. {labels[j]:>10s} : {S[k,j]: .3e} (|.|={mags[k,j]:.3e})")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vref', type=float, default=15.0)
    ap.add_argument('--thetaref', type=float, default=0.0)
    ap.add_argument('--rule', choices=['scaled','sqrt_eps'], default='scaled')
    ap.add_argument('--sweep_param', default='c_d')
    ap.add_argument('--sweep_input_index', type=int, default=0)  # 0: v_ref, 1: theta_ref
    args = ap.parse_args()

    p = refresh_k(get_default_params())
    u = jnp.array([args.vref, args.thetaref])

    # (a) FD sensitivities wrt p and u
    print("== (a) FD sensitivities at equilibrium ==")
    t0 = time.time()
    xstar_p, ystar_p, dXdP_fd, dYdP_fd = fd_wrt_p(p, u, rule=args.rule)
    xstar_u, ystar_u, dXdU_fd, dYdU_fd = fd_wrt_u(p, u, rule=args.rule)
    print("x* =", np.array(xstar_p))
    print("y* =", np.array(ystar_p))
    print("dx*/dp (FD) shape:", dXdP_fd.shape, "\n", dXdP_fd)
    print("dy*/dp (FD) shape:", dYdP_fd.shape, "\n", dYdP_fd)
    print("dx*/du (FD) shape:", dXdU_fd.shape, "\n", dXdU_fd)
    print("dy*/du (FD) shape:", dYdU_fd.shape, "\n", dYdU_fd)
    print(f"[time] FD total = {time.time()-t0:.3f} s")

    # (b) step-size sweep for one parameter and one input
    print(f"\n== (b) Step-size sweep for param '{args.sweep_param}' ==")
    for dp, dy in sweep_param(p, u, args.sweep_param):
        print(f"  dp={dp:.2e} -> Δy = {dy}")
    print(f"\n== (b) Step-size sweep for input index {args.sweep_input_index} ==")
    for du, dy in sweep_input(p, u, j=args.sweep_input_index):
        print(f"  du={du:.2e} -> Δy = {dy}")

    # (c,d) linearization + adjoint
    print("\n== (c,d) Linearization and Adjoint ==")
    A,E,B,C,Dp,Du = linearize_all(p, u, jnp.array(xstar_p))
    dYdP_adj, dYdU_adj = adjoint_y_sensitivities(A,E,B,C,Dp,Du)
    dXdP_adj, dXdU_adj = state_sensitivities(A,E,B)
    print("dy*/dp (Adjoint):\n", dYdP_adj)
    print("dy*/du (Adjoint):\n", dYdU_adj)
    print("dx*/dp (Adjoint):\n", dXdP_adj)
    print("dx*/du (Adjoint):\n", dXdU_adj)

    # (c) ranking by |dy*/dp|
    rank_params(dYdP_adj, PARAM_KEYS, topk=8, title="|dy*/dp| (Adjoint)")

    print("\n== (d) Note ==")
    print("Adjoint solves A^T λ = C^T (q times) and forms dy*/dp = -λ^T E + Dp, dy*/du = -λ^T B + Du.\n"
          "Cost depends on #outputs q, not on #parameters m, which is ideal for design loops.")

if __name__ == "__main__":
    main()
