# steady_state_homotopy_LU.py
# ------------------------------------------------------------
# Steady-state solve: evalf(x*, p, u) = 0
# Homotopy continuation (h0 = -x) + Newton + explicit Jacobian + LU only
# ------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from model.drone_model_jax import (
    evalf as evalf_jax,          # x' = f(x, p_tuple, u)
    compute_jacobian_jax,        # J = df/dx (JAX JIT)
    get_default_params, pack_params, N_STATES, IDX
)

jax.config.update("jax_enable_x64", True)


# -------------------------------
# JAX -> NumPy wrappers
# -------------------------------
def residual_np(x: np.ndarray, p_tuple, u: np.ndarray) -> np.ndarray:
    x_j = jnp.asarray(x, dtype=jnp.float64)
    u_j = jnp.asarray(u, dtype=jnp.float64)  # u = [y_ref, z_ref]
    dxdt = evalf_jax(x_j, p_tuple, u_j)
    return np.asarray(dxdt, dtype=np.float64)

def jacobian_np(x: np.ndarray, p_tuple, u: np.ndarray) -> np.ndarray:
    J = compute_jacobian_jax(jnp.asarray(x, jnp.float64),
                             p_tuple,
                             jnp.asarray(u, jnp.float64))
    return np.asarray(J, dtype=np.float64)

# -------------------------------
# Homotopy: f_tau = τ f(x) + (1-τ)(-x)
# => J_tau = τ J_f(x) - (1-τ) I
# -------------------------------
def f_tau(x: np.ndarray, tau: float, p_tuple, u: np.ndarray) -> np.ndarray:
    return tau * residual_np(x, p_tuple, u) - (1.0 - tau) * x

def J_tau(x: np.ndarray, tau: float, p_tuple, u: np.ndarray) -> np.ndarray:
    Jf = jacobian_np(x, p_tuple, u)
    N = x.size
    return tau * Jf - (1.0 - tau) * np.eye(N)

# -------------------------------
# Newton (line search + LU only)
# -------------------------------
def newton_stepper(
    x0: np.ndarray,
    tau: float,
    p_tuple,
    u: np.ndarray,
    tol_f: float = 1e-10,
    tol_dx: float = 1e-12,
    maxiter: int = 60,
    line_search: bool = True,
    ls_c: float = 1e-4,
    ls_max_bt: int = 20,
    verbose: bool = True,
):
    x = np.asarray(x0, dtype=float).copy()
    for k in range(maxiter):
        F = f_tau(x, tau, p_tuple, u)
        fn = np.linalg.norm(F, 2)
        if verbose:
            print(f"[Newton] τ={tau:.6f}  k={k:02d}  ||F||={fn:.3e}")
        if fn < tol_f:
            return x, {"success": True, "iter": k, "fnorm": fn}

        J = J_tau(x, tau, p_tuple, u)
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return x, {"success": False, "iter": k, "fnorm": fn, "reason": "singular J_tau"}

        # Backtracking line search
        alpha = 1.0
        if line_search:
            f0 = fn
            for _ in range(ls_max_bt):
                if np.linalg.norm(f_tau(x + alpha*dx, tau, p_tuple, u), 2) <= (1 - ls_c*alpha) * f0:
                    break
                alpha *= 0.5
        if verbose:
            print(f"         step ||dx||={np.linalg.norm(dx):.3e}, alpha={alpha:.3f}")

        x_new = x + alpha * dx
        if np.linalg.norm(alpha * dx, 2) < tol_dx:
            Fn = f_tau(x_new, tau, p_tuple, u)
            if verbose:
                print(f"         small step -> ||F_new||={np.linalg.norm(Fn):.3e}")
            if np.linalg.norm(Fn, 2) < tol_f:
                return x_new, {"success": True, "iter": k+1, "fnorm": float(np.linalg.norm(Fn,2))}
            else:
                return x_new, {"success": False, "iter": k+1,
                               "fnorm": float(np.linalg.norm(Fn,2)),
                               "reason": "dx small but residual not small"}
        x = x_new

    Fn = f_tau(x, tau, p_tuple, u)
    return x, {"success": False, "iter": maxiter, "fnorm": float(np.linalg.norm(Fn,2)), "reason": "maxiter"}

# -------------------------------
# Homotopy driver (τ: 0 -> 1)
# -------------------------------
def homotopy_solve(
    x0: np.ndarray,
    p_tuple,
    u: np.ndarray,
    tau_start: float = 0.0,
    tau_end: float = 1.0,
    init_dt: float = 0.05,
    tol_f: float = 1e-10,
    tol_dx: float = 1e-12,
    verbose: bool = True,
    line_search: bool = True,
    ls_c: float = 1e-4,
    ls_max_bt: int = 20,
    max_newton_iter: int = 10,
    dt_growth: float = 1.6,
    dt_max: float = 0.10,
    dt_min: float = 1e-3,
):
    x = np.asarray(x0, dtype=float).copy()
    tau = float(tau_start)
    dt  = float(init_dt)

    path = [np.hstack(([tau], x.copy()))]

    while tau < tau_end - 1e-14:
        tau_try = min(tau + dt, tau_end)
        x_try, info = newton_stepper(
            x0=x, tau=tau_try, p_tuple=p_tuple, u=u,
            tol_f=tol_f, tol_dx=tol_dx, maxiter=max_newton_iter,
            line_search=line_search, ls_c=ls_c, ls_max_bt=ls_max_bt,
            verbose=verbose
        )
        if info.get("success", False):
            x, tau = x_try, tau_try
            path.append(np.hstack(([tau], x.copy())))
            if verbose:
                print(f"[OK] τ={tau:.6f}, it={info['iter']:2d}, ||F||={info['fnorm']:.3e}")
            dt = min(dt * dt_growth, dt_max)
        else:
            dt *= 0.5
            if verbose:
                print(f"[retry] τ={tau_try:.6f} failed ({info.get('reason','')}). dt->{dt:.3e}")
            if dt < dt_min:
                raise RuntimeError("Homotopy step too small; continuation stalled.")

    return x, np.vstack(path)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    p = get_default_params()
    p_tuple = pack_params(p)

    x0 = np.zeros(N_STATES, dtype=float)
    u  = np.array([1, 1], dtype=float)

    x_star, path = homotopy_solve(
        x0=x0, p_tuple=p_tuple, u=u,
        tau_start=0.0, tau_end=1.0,
        init_dt=0.025, tol_f=1e-10, tol_dx=1e-12, verbose=True,
        max_newton_iter=25, dt_growth=2, dt_max=0.05
    )

    np.set_printoptions(formatter={'float_kind': lambda v: f"{float(v):.3e}"})
    print("\nFinal steady-state x*:")
    print(x_star)
    print("tau steps:", path.shape[0])

    # ------------------------------------
    # Plot homotopy path: tau vs states
    # ------------------------------------
    # path has shape (n_steps, 1 + N_STATES):
    #   column 0: tau
    #   columns 1..: state vector x
    tau_vals = path[:, 0]
    X_path   = path[:, 1:]   # shape (n_steps, N_STATES)

    # Extract indices for readability
    i_lamdL = IDX['lamdsr_L']
    i_lamqL = IDX['lamqsr_L']
    i_lamdR = IDX['lamdsr_R']
    i_lamqR = IDX['lamqsr_R']
    i_wmL   = IDX['wm_L']
    i_wmR   = IDX['wm_R']
    i_y     = IDX['y']
    i_z     = IDX['z']
    i_theta = IDX['theta']

    # Extract state trajectories along the homotopy path
    lamdL_path = X_path[:, i_lamdL]
    lamqL_path = X_path[:, i_lamqL]
    lamdR_path = X_path[:, i_lamdR]
    lamqR_path = X_path[:, i_lamqR]
    wmL_path   = X_path[:, i_wmL]
    wmR_path   = X_path[:, i_wmR]
    y_path     = X_path[:, i_y]
    z_path     = X_path[:, i_z]
    theta_path = X_path[:, i_theta]

    # Compute dq-currents from fluxes (using motor parameters)
    Lds = p['Lds']
    Lqs = p['Lqs']
    lamf = p['lamf']

    idL_path = (lamdL_path - lamf) / Lds
    iqL_path = lamqL_path / Lqs
    idR_path = (lamdR_path - lamf) / Lds
    iqR_path = lamqR_path / Lqs

    # -----------------------------
    # 1) Position (y, z) vs tau
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(tau_vals, y_path, label='y position')
    plt.plot(tau_vals, z_path, label='z position')
    plt.xlabel(r'Homotopy parameter $\tau$')
    plt.ylabel('Position [m]')
    plt.title('Position states along homotopy path')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    # -----------------------------
    # 2) Motor speeds vs tau
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(tau_vals, wmL_path, label='wm_L')
    plt.plot(tau_vals, wmR_path, label='wm_R')
    plt.xlabel(r'Homotopy parameter $\tau$')
    plt.ylabel('Speed [rad/s]')
    plt.title('Motor speeds along homotopy path')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    # -----------------------------
    # 3) dq currents vs tau
    #    (electrical quantities grouped together)
    # -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # Left motor currents
    axs[0].plot(tau_vals, idL_path, label='i_dL')
    axs[0].plot(tau_vals, iqL_path, label='i_qL')
    axs[0].set_title('Left motor currents')
    axs[0].set_xlabel(r'$\tau$')
    axs[0].set_ylabel('Current [A]')
    axs[0].grid(True)
    axs[0].legend(loc='best')

    # Right motor currents
    axs[1].plot(tau_vals, idR_path, label='i_dR')
    axs[1].plot(tau_vals, iqR_path, label='i_qR')
    axs[1].set_title('Right motor currents')
    axs[1].set_xlabel(r'$\tau$')
    axs[1].grid(True)
    axs[1].legend(loc='best')

    plt.suptitle('dq currents along homotopy path')
    plt.tight_layout()

    # -----------------------------
    # 4) Pitch angle vs tau (optional)
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(tau_vals, theta_path, label='theta')
    plt.xlabel(r'Homotopy parameter $\tau$')
    plt.ylabel('Pitch angle [rad]')
    plt.title('Pitch angle along homotopy path')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()

