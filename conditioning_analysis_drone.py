# conditioning_analysis_drone.py
# ------------------------------------------------------------
# Conditioning Analysis for the Drone Model (robust version)
# - Build hover equilibrium
# - Jacobian & cond2 at hover
# - Sample cond2 along a trajectory (robust SVD, NaN guards)
# ------------------------------------------------------------
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from utils.drone_utils import build_hover_state_and_input
from drone_model_jax import (
    N_STATES, IDX,
    get_default_params, pack_params,
    evalf, compute_jacobian_jax, unpack_params
)

# ---------- Robust SVD-based condition number ----------
def sv_cond_2(A: np.ndarray, eps: float = 0.0):
    """
    Robust cond_2 via JAX SVD with fallbacks.
    - Adds eps*I if eps>0 (Tikhonov regularization)
    - Returns (cond2, svals)
    """
    # Finite check first
    if not np.all(np.isfinite(A)):
        return np.inf, np.array([np.nan, np.nan])

    if eps > 0.0:
        A = A + eps * np.eye(A.shape[0], dtype=A.dtype)

    try:
        # JAX SVD is often more stable on some LAPACK builds
        s = np.array(jnp.linalg.svd(jnp.asarray(A), compute_uv=False))
    except Exception:
        # Fallback: eigenvalues of A^T A
        AtA = A.T @ A
        w = np.linalg.eigvalsh(AtA)
        w = np.clip(w, 0.0, None)  # guard small negative due to roundoff
        s = np.sqrt(np.sort(w)[::-1])

    if s.size == 0 or not np.all(np.isfinite(s)):
        return np.inf, s
    smax, smin = s[0], s[-1]
    if np.isclose(smin, 0.0) or not np.isfinite(smin):
        return np.inf, s
    return (smax / smin), s


# ---------- Jacobian & cond at a single point ----------
def analyze_point(x: np.ndarray, p_tuple, u: np.ndarray, reg_eps: float = 0.0):
    """
    Returns dict(A, cond2, svals, smax, smin) at state x and input u.
    """
    A = np.asarray(compute_jacobian_jax(jnp.array(x), p_tuple, jnp.array(u)))
    cond2, svals = sv_cond_2(A, eps=reg_eps)
    return {
        "A": A,
        "cond2": float(cond2),
        "svals": svals,
        "smax": float(svals[0]),
        "smin": float(svals[-1]),
    }

# ---------- Simple RK4 integrator ----------
def rk4_step(f, x, p_tuple, u, dt):
    k1 = np.asarray(f(jnp.array(x), p_tuple, jnp.array(u)))
    k2 = np.asarray(f(jnp.array(x + 0.5*dt*k1), p_tuple, jnp.array(u)))
    k3 = np.asarray(f(jnp.array(x + 0.5*dt*k2), p_tuple, jnp.array(u)))
    k4 = np.asarray(f(jnp.array(x + dt*k3), p_tuple, jnp.array(u)))
    return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ---------- Condition number over a trajectory (robust) ----------
def sample_condition_over_traj(x0, p_tuple, u_fun, T=2.0, dt=1e-3, reg_eps: float = 1e-8):
    """
    Simulate x' = f(x,p,u(t)) for t in [0,T] and sample cond2(A(x(t))).
    Robust to NaN/Inf and SVD failures; fills NaN and continues.
    Returns: t_grid, cond2_list, smin_list, smax_list
    """
    steps = int(np.round(T / dt))
    t_grid = np.linspace(0.0, steps*dt, steps+1)
    x = x0.copy()

    cond2_list = np.empty(steps+1)
    smin_list  = np.empty(steps+1)
    smax_list  = np.empty(steps+1)

    def safe_cond_at_state(x, u):
        A = np.asarray(compute_jacobian_jax(jnp.array(x), p_tuple, jnp.array(u)))
        if not np.all(np.isfinite(A)):
            return np.nan, np.nan, np.nan
        c, svals = sv_cond_2(A, eps=reg_eps)
        if svals is None or not np.all(np.isfinite([c])):
            return np.nan, np.nan, np.nan
        return float(c), float(svals[-1]), float(svals[0])

    # t=0
    c0, smin0, smax0 = safe_cond_at_state(x, u_fun(0.0))
    cond2_list[0], smin_list[0], smax_list[0] = c0, smin0, smax0

    for k in range(steps):
        u = u_fun(t_grid[k])
        # integrate (smaller dt improves stability if needed)
        x = rk4_step(evalf, x, p_tuple, u, dt)
        c, smin, smax = safe_cond_at_state(x, u)
        cond2_list[k+1] = c
        smin_list[k+1]  = smin
        smax_list[k+1]  = smax

    return t_grid, cond2_list, smin_list, smax_list


def conditioning_analysis_swarm(p_list, reg_eps: float = 0.0):
    """Analyze conditioning for a swarm of drones.

    Args:
        p_list: sequence of packed param tuples (as returned by `pack_params`)
                or sequence of parameter dicts (as returned by `get_default_params`).
        reg_eps: regularization epsilon passed to SVD cond (optional).

    Behavior:
        - If all drones have identical parameters, runs the single-drone
          conditioning analysis at hover (using `analyze_point`) for just
          one drone and returns that result dict.
        - If drones have heterogeneous parameters, computes the condition
          number at hover for each drone and returns the maximum condition
          number found as a float in the key 'max_cond2'.

    Returns:
        If homogeneous: dict returned by `analyze_point` with extra key 'same'=True.
        If heterogeneous: dict { 'same': False, 'max_cond2': float }
    """
    # Normalize inputs to packed tuples
    normalized = []
    for p in p_list:
        if isinstance(p, dict):
            normalized.append(pack_params(p))
        else:
            # assume tuple/list/ndarray-like packed params
            normalized.append(tuple(p))

    if len(normalized) == 0:
        raise ValueError("p_list must contain at least one parameter entry")

    # Use numerical closeness to decide homogeneity (tolerant to tiny float diffs)
    first = jnp.asarray(normalized[0], dtype=jnp.float64)
    homogeneous = True
    for p in normalized[1:]:
        if not np.allclose(np.asarray(p, dtype=np.float64), np.asarray(first, dtype=np.float64)):
            homogeneous = False
            break

    if homogeneous:
        # Unpack to dict to build hover state
        p_dict = unpack_params(first)
        x_hover, u_hover = build_hover_state_and_input(p_dict)
        res = analyze_point(x_hover, first, u_hover, reg_eps=reg_eps)
        res_out = {
            'same': True,
            'cond2': res['cond2'],
            'details': res,
        }
        return res_out

    # Heterogeneous: evaluate each drone's cond2 at its hover
    max_cond = -np.inf
    max_idx = None
    for i, pt in enumerate(normalized):
        p_dict = unpack_params(pt)
        x_hover, u_hover = build_hover_state_and_input(p_dict)
        res = analyze_point(x_hover, pt, u_hover, reg_eps=reg_eps)
        try:
            c = float(res['cond2'])
        except Exception:
            c = np.inf
        if c > max_cond:
            max_cond = c
            max_idx = i

    return {
        'same': False,
        'max_cond2': float(max_cond),
        'index_of_max': max_idx,
    }

# ---------- Example run ----------
if __name__ == "__main__":
    # 1) Parameters
    p_dict  = get_default_params()
    p_tuple = pack_params(p_dict)

    # 2) Hover equilibrium
    x_hover, u_hover = build_hover_state_and_input(p_dict)

    # 3) Jacobian / cond2 at hover
    res = analyze_point(x_hover, p_tuple, u_hover, reg_eps=0.0)
    print("[Hover] cond2(A) =", res["cond2"])
    print("  smax =", res["smax"], "  smin =", res["smin"])

    # 4) Input profile: z_ref step to 0.2 at t=0.5s (y_ref=0)
    def u_profile(t):
        zref = 0.2 if t >= 0.5 else 0.0
        return np.array([0.0, zref], dtype=np.float64)

    # 5) Sample cond2 over trajectory (robust)
    t, cond2_t, smin_t, smax_t = sample_condition_over_traj(
        x0=x_hover, p_tuple=p_tuple, u_fun=u_profile,
        T=2.0, dt=2e-3, reg_eps=1e-8
    )
    print("[Trajectory] cond2 stats:",
          "min =", np.nanmin(cond2_t),
          "median =", np.nanmedian(cond2_t),
          "max =", np.nanmax(cond2_t))

    # -----------------------------
    # Swarm conditioning analysis
    # -----------------------------
    # Homogeneous swarm (5 identical drones)
    p_tuple_shared = pack_params(p_dict)
    p_list_hom = [p_tuple_shared for _ in range(5)]
    swarm_res_hom = conditioning_analysis_swarm(p_list_hom, reg_eps=1e-8)
    print("[Swarm - homogeneous] same:", swarm_res_hom.get('same'))
    print("[Swarm - homogeneous] cond2:", swarm_res_hom.get('cond2'))

    # Heterogeneous swarm example (one drone with a slightly different Dt)
    p_base = get_default_params()
    p_alt = dict(p_base)
    p_alt['Dt'] = p_base['Dt'] * 5.0  # increased translational damping for one drone
    p_list_het = [pack_params(p_base), pack_params(p_alt), pack_params(p_base)]
    swarm_res_het = conditioning_analysis_swarm(p_list_het, reg_eps=1e-8)
    print("[Swarm - heterogeneous] same:", swarm_res_het.get('same'))
    print("[Swarm - heterogeneous] max_cond2:", swarm_res_het.get('max_cond2'))
