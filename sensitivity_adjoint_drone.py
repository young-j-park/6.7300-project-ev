# sensitivity_adjoint_drone.py
# ------------------------------------------------------------
# Steady-state sensitivities for the DRONE model:
#   dx/dt = f(x, p, u),   y = g(x, p, u)
# (a) Finite-difference (FD) sensitivities of x*, y* wrt p and u
# (b) Step-size sweep utilities
# (c) Ranking by |dy*/dÂ·|
# (d) Adjoint sensitivities on the linearized steady-state system
# ------------------------------------------------------------

import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from drone_model_jax import (
    N_STATES, IDX,
    PACK_ORDER,              # parameter key order
    get_default_params,      # -> dict
    pack_params,             # dict -> tuple
    unpack_params,           # tuple -> dict
    evalf                    # x' = f(x, p_tuple, u)
)
from utils.drone_utils import build_hover_state_and_input

jax.config.update("jax_enable_x64", True)

# ---------- outputs y = g(x,p,u) ----------
# Default: y = [y, z, theta]
def g(x, p_tuple, u):
    return jnp.array([x[IDX['y']], x[IDX['z']], x[IDX['theta']]])

# ---------- parameter helpers ----------
PARAM_KEYS = list(PACK_ORDER)

def params_to_vec(p_dict):
    """Pack selected parameters into a flat JAX vector (order = PACK_ORDER)."""
    return jnp.array([jnp.asarray(p_dict[k], dtype=jnp.float64) for k in PARAM_KEYS],
                     dtype=jnp.float64)

def vec_to_params(p_vec, p_template):
    """Unpack vector back to a full parameter dict (keeping any extra keys from template)."""
    q = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p_template.items()}
    for k, v in zip(PARAM_KEYS, p_vec):
        q[k] = jnp.asarray(v, dtype=jnp.float64)
    return q

def pack_param_tuple(p):
    """Wrapper to match drone_model_matched.pack_params signature from a dict."""
    return pack_params(p)

# ---------- convenience wrappers for JAX jacobians wrt p and u ----------
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
def newton_equilibrium(p_dict, u, x0=None, tol=1e-10, maxit=60):
    p_tuple = pack_param_tuple(p_dict)

    if x0 is None:
        x0, _ = build_hover_state_and_input(p_dict)
        # set references if provided
        x = x0
    else:
        x = jnp.array(x0)

    f_x   = lambda x_: evalf(x_, p_tuple, u)
    jac_x = jax.jit(jax.jacobian(lambda xx: evalf(xx, p_tuple, u)))
    res_prev = np.inf

    for _ in range(maxit):
        Fx = f_x(x)
        res = float(jnp.linalg.norm(Fx))
        if res < tol:
            break
        A = jac_x(x)
        # Solve A dx = Fx
        dx = jnp.linalg.solve(A, Fx)
        # Backtracking line search for robustness
        alpha = 1.0
        for _ in range(8):
            xn = x - alpha * dx
            if float(jnp.linalg.norm(f_x(xn))) <= res:
                x = xn
                break
            alpha *= 0.5
        res_prev = res
    return x

# ---------- output selector y = C x (optional helper) ----------
def build_outputs_C(which=('y','z','theta')):
    rows, names = [], []
    for w in which:
        c = np.zeros(N_STATES)
        if w == 'y':        c[IDX['y']] = 1.0
        elif w == 'z':      c[IDX['z']] = 1.0
        elif w == 'theta':  c[IDX['theta']] = 1.0
        else: raise ValueError(f"Unknown output '{w}'")
        rows.append(c); names.append(w)
    return np.vstack(rows), names

# ---------- (a) FD sensitivities wrt p and u ----------
def step_rule(val, rule):
    sq = np.sqrt(np.finfo(float).eps)
    if rule == 'sqrt_eps': return sq
    if rule == 'scaled':   return 2.0 * sq * max(1.0, abs(float(val)))
    raise ValueError

def fd_wrt_p(p_base, u, rule='scaled'):
    xstar = newton_equilibrium(p_base, u)
    pvec  = np.array(params_to_vec(p_base), dtype=np.float64)
    ystar = np.array(g(xstar, pack_param_tuple(p_base), u))

    n, m, q = N_STATES, len(pvec), ystar.size
    dxdP = np.zeros((n, m))
    dydP = np.zeros((q, m))

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
    dxdU = np.zeros((n, nu))
    dydU = np.zeros((q, nu))

    for j in range(nu):
        du = step_rule(u_base[j], rule)
        u_plus = np.array(u_base); u_plus[j] += du

        x1 = newton_equilibrium(p_base, u_plus, x0=xstar)
        y1 = np.array(g(x1, p_tuple, u_plus))

        dxdU[:, j] = (np.array(x1) - np.array(xstar)) / du
        dydU[:, j] = (y1 - ystar) / du

    return xstar, ystar, dxdU, dydU

# ---------- (b) step-size sweep utilities ----------
def sweep_param(p_base, u, pname, scales=np.logspace(-2, -12, 11)):
    idx = PARAM_KEYS.index(pname)
    x0  = newton_equilibrium(p_base, u)
    y0  = np.array(g(x0, pack_param_tuple(p_base), u))
    base_vec = np.array(params_to_vec(p_base), dtype=np.float64)
    ref = base_vec[idx] if base_vec[idx] != 0 else 1.0

    outs = []
    for s in scales:
        dp = s * abs(ref)
        p_plus = base_vec.copy(); p_plus[idx] += dp
        p_plus_dict = vec_to_params(p_plus, p_base)
        x1 = newton_equilibrium(p_plus_dict, u, x0=x0)
        y1 = np.array(g(x1, pack_param_tuple(p_plus_dict), u))
        outs.append((dp, (y1 - y0)))
    return outs

def sweep_input(p_base, u_base, j=0, scales=np.logspace(-2, -12, 11)):
    x0 = newton_equilibrium(p_base, u_base)
    y0 = np.array(g(x0, pack_param_tuple(p_base), u_base))
    ref = u_base[j] if float(u_base[j]) != 0.0 else 1.0

    outs = []
    for s in scales:
        du = s * abs(ref)
        u_plus = np.array(u_base); u_plus[j] += du
        x1 = newton_equilibrium(p_base, u_plus, x0=x0)
        y1 = np.array(g(x1, pack_param_tuple(p_base), u_plus))
        outs.append((du, (y1 - y0)))
    return outs

# ---------- (c,d) linearization + adjoint ----------
def linearize_all(p_base, u, x_star):
    p_tuple = pack_param_tuple(p_base)
    A  = jax.jacobian(lambda xx: evalf(xx, p_tuple, u))(x_star)                 # (n,n)
    pv = params_to_vec(p_base)
    E  = jax.jacobian(lambda pv_: evalf_with_pvec(x_star, pv_, u, p_base))(pv)  # (n,m)
    B  = jax.jacobian(lambda uu:  evalf_with_uvec(x_star, p_tuple, uu))(u)      # (n,nu)
    C  = jax.jacobian(lambda xx:  g(xx, p_tuple, u))(x_star)                    # (q,n)
    Dp = jax.jacobian(lambda pv_: g_with_pvec(x_star, pv_, u, p_base))(pv)      # (q,m)
    Du = jax.jacobian(lambda uu:  g_with_uvec(x_star, p_tuple, uu))(u)          # (q,nu)
    return (np.array(A), np.array(E), np.array(B),
            np.array(C), np.array(Dp), np.array(Du))

def adjoint_y_sensitivities(A, E, B, C, Dp, Du):
    """
    Solve A^T λ_k = C_k^T (k=1..q),
    dy*/dp = -λ^T E + Dp, dy*/du = -λ^T B + Du
    """
    AT = A.T
    q = C.shape[0]
    m = E.shape[1]; nu = B.shape[1]
    dYdP = np.zeros((q, m))
    dYdU = np.zeros((q, nu))
    for k in range(q):
        lam = np.linalg.solve(AT, C[k, :])
        dYdP[k, :] = - lam @ E + Dp[k, :]
        dYdU[k, :] = - lam @ B + Du[k, :]
    return dYdP, dYdU

def state_sensitivities(A, E, B):
    """Optional: x* sensitivities wrt p and u."""
    dXdP = - np.linalg.solve(A, E)   # (n,m)
    dXdU = - np.linalg.solve(A, B)   # (n,nu)
    return dXdP, dXdU

def rank_params(S, labels, topk=10, title=''):
    print(f"\n[Ranking] {title}")
    mags = np.abs(S)
    if S.ndim == 1:
        order = np.argsort(mags)[::-1]
        for i in range(min(topk, len(order))):
            j = order[i]
            print(f"  {i+1:2d}. {labels[j]:>14s} : {S[j]: .3e} (|.|={mags[j]:.3e})")
    else:
        for k in range(S.shape[0]):
            order = np.argsort(mags[k, :])[::-1]
            print(f"  - output[{k}]")
            for i in range(min(topk, len(order))):
                j = order[i]
                print(f"    {i+1:2d}. {labels[j]:>14s} : {S[k, j]: .3e} (|.|={mags[k, j]:.3e})")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yref', type=float, default=0.0)   # position y reference
    ap.add_argument('--zref', type=float, default=0.0)   # height z reference
    ap.add_argument('--rule', choices=['scaled', 'sqrt_eps'], default='scaled')
    ap.add_argument('--sweep_param', default='Kp_theta') # any key in PACK_ORDER
    ap.add_argument('--sweep_input_index', type=int, default=1)  # 0:y_ref, 1:z_ref
    args = ap.parse_args()

    # parameters & inputs
    p = get_default_params()
    u = jnp.array([args.yref, args.zref], dtype=jnp.float64)

    # (a) FD sensitivities wrt p and u
    print("== (a) FD sensitivities at equilibrium (drone) ==")
    t0 = time.time()
    xstar_p, ystar_p, dXdP_fd, dYdP_fd = fd_wrt_p(p, u, rule=args.rule)
    xstar_u, ystar_u, dXdU_fd, dYdU_fd = fd_wrt_u(p, u, rule=args.rule)
    print("x* =", np.array(xstar_p))
    print("y* =", np.array(ystar_p))
    print("dx*/dp (FD) shape:", dXdP_fd.shape)
    print("dy*/dp (FD) shape:", dYdP_fd.shape)
    print("dx*/du (FD) shape:", dXdU_fd.shape)
    print("dy*/du (FD) shape:", dYdU_fd.shape)
    print(f"[time] FD total = {time.time()-t0:.3f} s")

    # (b) step-size sweep
    print(f"\n== (b) Step-size sweep for param '{args.sweep_param}' ==")
    for dp, dy in sweep_param(p, u, args.sweep_param):
        print(f"  dp={dp:.2e} -> Δy = {dy}")
    print(f"\n== (b) Step-size sweep for input index {args.sweep_input_index} ==")
    for du, dy in sweep_input(p, u, j=args.sweep_input_index):
        print(f"  du={du:.2e} -> Δy = {dy}")

    # (c,d) linearization + adjoint at x*
    print("\n== (c,d) Linearization and Adjoint (drone) ==")
    A,E,B,C,Dp,Du = linearize_all(p, u, jnp.array(xstar_p))
    dYdP_adj, dYdU_adj = adjoint_y_sensitivities(A,E,B,C,Dp,Du)
    dXdP_adj, dXdU_adj = state_sensitivities(A,E,B)
    print("dy*/dp (Adjoint) shape:", dYdP_adj.shape)
    print("dy*/du (Adjoint) shape:", dYdU_adj.shape)
    print("dx*/dp (Adjoint) shape:", dXdP_adj.shape)
    print("dx*/du (Adjoint) shape:", dXdU_adj.shape)

    # (c) ranking by |dy*/dp|
    rank_params(dYdP_adj, PARAM_KEYS, topk=10, title="|dy*/dp| (Adjoint)")

    print("\n== (d) Note ==")
    print("Adjoint solves A^T λ = C^T (q times) and forms dy*/dp = -λ^T E + Dp, "
          "dy*/du = -λ^T B + Du. Cost scales with #outputs (q), not #parameters (m).")

if __name__ == "__main__":
    main()
