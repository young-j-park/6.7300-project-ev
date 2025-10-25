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

from drone_model_jax import (
    N_STATES, IDX,
    get_default_params, pack_params,
    evalf, compute_jacobian_jax
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

# ---------- Build hover equilibrium ----------
def build_hover_state_and_input(p_dict: dict):
    """
    Hover assumptions:
      - theta = 0, wd = 0
      - y = z = 0, vy = vz = 0
      - Total thrust FT = m*g => each motor thrust = m*g/2
      - wm_hover = sqrt((m*g/2)/kt)
      - Electrical states: id=0, iq=0 => lamd=lamf, lamq=0
      - All PI integrators start at 0
    u = [y_ref, z_ref] = [0, 0]
    """
    m, g, kt = p_dict['m'], p_dict['g'], p_dict['kt']
    lamf = p_dict['lamf']

    wm_hover = float(np.sqrt((m * g) / (2.0 * kt)))  # equal motors
    x = np.zeros(N_STATES, dtype=np.float64)

    # Motor fluxes
    x[IDX['lamdsr_L']] = lamf
    x[IDX['lamqsr_L']] = 0.0
    x[IDX['lamdsr_R']] = lamf
    x[IDX['lamqsr_R']] = 0.0

    # PI integrators
    x[IDX['integ_id_L']] = 0.0
    x[IDX['integ_iq_L']] = 0.0
    x[IDX['integ_id_R']] = 0.0
    x[IDX['integ_iq_R']] = 0.0
    x[IDX['integ_w_L']]  = 0.0
    x[IDX['integ_w_R']]  = 0.0

    # Motor speeds (hover)
    x[IDX['wm_L']] = wm_hover
    x[IDX['wm_R']] = wm_hover

    # Body attitude/position/velocity
    x[IDX['theta']] = 0.0
    x[IDX['wd']]    = 0.0
    x[IDX['y']]     = 0.0
    x[IDX['z']]     = 0.0
    x[IDX['v_y']]   = 0.0
    x[IDX['v_z']]   = 0.0

    # Reference inputs (hover)
    u = np.array([0.0, 0.0], dtype=np.float64)
    return x, u

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
