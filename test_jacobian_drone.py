# test_jacobian_drone.py  (compatible with new drone_model_jax.py)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from model.drone_model_jax import (
    compute_jacobian_jax,
    get_default_params,
    pack_params,
    N_STATES,
    evalf,
    IDX,
)

# -------------------------------
# Finite-difference Jacobian (central)
# -------------------------------
def fd_jacobian(evalf_fun, x, p_tuple, u, h=1e-6):
    """
    Central-difference Jacobian d f / d x at (x, p_tuple, u).
    Returns an (N_STATES, N_STATES) ndarray.
    """
    x = jnp.array(x, dtype=jnp.float64)
    n = x.shape[0]
    J = jnp.zeros((n, n), dtype=jnp.float64)

    for j in range(n):
        ej = jnp.zeros(n, dtype=jnp.float64).at[j].set(1.0)
        fp = evalf_fun(x + h * ej, p_tuple, u)
        fm = evalf_fun(x - h * ej, p_tuple, u)
        J = J.at[:, j].set((fp - fm) / (2.0 * h))

    return np.asarray(J, dtype=np.float64)


def max_abs_err(A, B):
    A = np.asarray(A); B = np.asarray(B)
    return float(np.max(np.abs(A - B)))


def max_rel_err(A, B):
    A = np.asarray(A); B = np.asarray(B)
    denom = np.maximum(1.0, np.maximum(np.abs(A), np.abs(B)))
    return float(np.max(np.abs(A - B) / denom))


def random_state(rng):
    """
    Generate a physically reasonable random state vector using IDX.
    State layout (for reference):
      L: lamdsr_L, lamqsr_L, integ_id_L, integ_iq_L, wm_L, integ_w_L
      R: lamdsr_R, lamqsr_R, integ_id_R, integ_iq_R, wm_R, integ_w_R
      Body: theta, wd, y, z, v_y, v_z
    """
    x = np.zeros(N_STATES, dtype=np.float64)

    # Flux linkages around ~PM flux scale ± small
    x[IDX['lamdsr_L']] = rng.normal(0.04, 0.01)    # ~lamf ±
    x[IDX['lamqsr_L']] = rng.normal(0.0,  0.01)
    x[IDX['lamdsr_R']] = rng.normal(0.04, 0.01)
    x[IDX['lamqsr_R']] = rng.normal(0.0,  0.01)

    # Current PI integrators (small)
    x[IDX['integ_id_L']] = rng.normal(0.0, 0.2)
    x[IDX['integ_iq_L']] = rng.normal(0.0, 0.2)
    x[IDX['integ_id_R']] = rng.normal(0.0, 0.2)
    x[IDX['integ_iq_R']] = rng.normal(0.0, 0.2)

    # Motor speeds [rad/s] in a moderate range to avoid kinks at exactly 0
    x[IDX['wm_L']] = rng.uniform(50.0, 800.0)
    x[IDX['wm_R']] = rng.uniform(50.0, 800.0)

    # Speed PI integrators
    x[IDX['integ_w_L']] = rng.normal(0.0, 0.5)
    x[IDX['integ_w_R']] = rng.normal(0.0, 0.5)

    # Body states
    x[IDX['theta']] = rng.uniform(-0.4, 0.4)     # rad
    x[IDX['wd']]    = rng.uniform(-0.5, 0.5)     # rad/s
    x[IDX['y']]     = rng.uniform(-0.5, 0.5)     # m
    x[IDX['z']]     = rng.uniform(-0.5, 0.5)     # m
    x[IDX['v_y']]   = rng.uniform(-0.2, 0.2)     # m/s
    x[IDX['v_z']]   = rng.uniform(-0.2, 0.2)     # m/s
    return x


def main():
    # Do Manual Tests
    with open('./test_benchmarks/test_cases.json', 'r') as f:
        test_cases = json.load(f)

    for case in test_cases:
        J_gt = case['J']
        J_out = compute_jacobian_jax(jnp.array(case['x']), tuple(case['p']), jnp.array(case['u']))
        assert np.allclose(J_gt, J_out, atol=1e-12)
            
    # -------------------------------
    # Setup params and tuple
    # -------------------------------
    p = get_default_params()
    p_tuple = pack_params(p)

    # -------------------------------
    # Random-state tests
    # -------------------------------
    rng = np.random.default_rng(42)
    tol_abs = 1e-5           # absolute tolerance
    tol_rel = 5e-6           # relative tolerance
    n_trials = 50
    y_ref_range = (-0.5, 0.5)
    z_ref_range = (0.5, 2.0)  # avoid near-zero total-thrust commands

    print("[1] Random-state FD vs. JAX Jacobian check")
    for k in range(n_trials):
        # Random state
        x = random_state(rng)

        # Random input (position refs)
        y_ref = rng.uniform(*y_ref_range)
        z_ref = rng.uniform(*z_ref_range)
        u = jnp.array([y_ref, z_ref], dtype=jnp.float64)

        x_jnp = jnp.array(x, dtype=jnp.float64)

        # JAX Jacobian
        J_jax = compute_jacobian_jax(x_jnp, p_tuple, u)

        # FD Jacobian (central, h=1e-6)
        J_fd = fd_jacobian(evalf, x_jnp, p_tuple, u, h=1e-6)

        err_abs = max_abs_err(J_jax, J_fd)
        err_rel = max_rel_err(J_jax, J_fd)
        if not (err_abs <= tol_abs or err_rel <= tol_rel):
            print(f"  FAILED Trial {k}: max abs err = {err_abs:.3e}, max rel err = {err_rel:.3e} "
                  f"(tol_abs={tol_abs}, tol_rel={tol_rel})")
            diff = np.abs(J_jax - J_fd)
            i, j = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"     worst entry at ({i},{j}): J_jax={J_jax[i,j]:.6e}, J_fd={J_fd[i,j]:.6e}")
            return
    print("  Passed random-state Jacobian tests.")

    # -------------------------------
    # Hover-like state: total thrust ~= m g, pitch 0
    # -------------------------------
    print("[2] Hover-like state FD vs. JAX Jacobian")

    m = p["m"]; kt = p["kt"]; g = p["g"]
    w_hover = np.sqrt((m * g) / (2.0 * kt))

    x_hover = np.zeros(N_STATES, dtype=np.float64)
    x_hover[IDX['wm_L']] = w_hover
    x_hover[IDX['wm_R']] = w_hover
    x_hover[IDX['theta']] = 0.0
    x_hover[IDX['wd']] = 0.0
    x_hover[IDX['y']] = 0.0
    x_hover[IDX['z']] = 1.0
    x_hover[IDX['v_y']] = 0.0
    x_hover[IDX['v_z']] = 0.0

    u_hover = jnp.array([0.0, 1.0], dtype=jnp.float64)

    xh = jnp.array(x_hover, dtype=jnp.float64)
    J_jax = compute_jacobian_jax(xh, p_tuple, u_hover)
    J_fd  = fd_jacobian(evalf, xh, p_tuple, u_hover, h=1e-6)

    err_abs = max_abs_err(J_jax, J_fd)
    err_rel = max_rel_err(J_jax, J_fd)
    if not (err_abs <= tol_abs or err_rel <= tol_rel):
        print(f"  FAILED Hover-like: max abs err = {err_abs:.3e}, max rel err = {err_rel:.3e} "
              f"(tol_abs={tol_abs}, tol_rel={tol_rel})")
        diff = np.abs(J_jax - J_fd)
        i, j = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"     worst entry at ({i},{j}): J_jax={J_jax[i,j]:.6e}, J_fd={J_fd[i,j]:.6e}")
        return

    print("  Passed hover-like Jacobian test.")
    print("All Jacobian tests passed.")


if __name__ == "__main__":
    main()
