"""Basic Newton nonlinear solver utilities for the drone model and swarm.

Provides:
- newton: a simple Newton method with optional backtracking line-search
- solve_drone_newton: wrapper that solves f(x,p,u)=0 for a single drone using
  `model.drone_model_jax.evalf` and `compute_jacobian_jax`.
- solve_swarm_newton: wrapper that solves stacked f(X_flat)=0 for a swarm
  with shared parameters by assembling the block-diagonal Jacobian from
  per-drone Jacobians.
- solve_swarm_newton_with_params: same as above but allows per-drone (heterogeneous) param tuples.

The implementations use NumPy linear solves for the Newton step.
"""
from typing import Callable, Tuple, Optional, Sequence, Any

import numpy as np
import jax.numpy as jnp

from model.drone_model_jax import N_STATES, compute_jacobian_jax, evalf, pack_params, unpack_params, IDX
from utils.swarm_utils import swarm_evalf, swarm_evalf_with_params
from utils.drone_utils import build_hover_state_and_input


def assess_tendency(x, p_tuple, u, dt: float = 0.01):
        """Assess instantaneous tendency toward the commanded hover.

        Returns a dict with:
            - e_pos: np.array([y_ref - y, z_ref - z])
            - v: np.array([v_y, v_z])
            - a: np.array([dvy, dvz]) computed from evalf (instantaneous accelerations)
            - dot_ea: dot(e_pos, a)
            - braking: tuple(bool_y, bool_z) where True means v*a < 0 (deceleration)
            - v_pred: predicted v after dt via v + dt * a

        This is a purely local diagnostic (no time integration beyond a single small step).
        """
        x_arr = np.asarray(x)
        # extract indices
        y_idx = IDX['y']
        z_idx = IDX['z']
        vy_idx = IDX['v_y']
        vz_idx = IDX['v_z']

        # desired references
        y_ref, z_ref = u

        # Position error and current velocity
        e_pos = np.array([y_ref - float(x_arr[y_idx]), z_ref - float(x_arr[z_idx])], dtype=float)
        v = np.array([float(x_arr[vy_idx]), float(x_arr[vz_idx])], dtype=float)

        # compute instantaneous derivative via evalf -> dvy,dvz are at indices v_y,v_z
        dxdt = np.asarray(evalf(jnp.array(x_arr), p_tuple, jnp.array(u)), dtype=float)
        # Extract accelerations (derivatives of v_y, v_z)
        a = np.array([float(dxdt[IDX['v_y']]), float(dxdt[IDX['v_z']])], dtype=float)

        # dot product of position error and acceleration
        # A positive value means acceleration is (globally) in the direction of the error,
        # which is the desired corrective behavior.
        dot_ea = float(np.dot(e_pos, a))
        
        # braking flags: True if v*a < 0 (deceleration)
        braking = (float(v[0] * a[0]) < 0.0, float(v[1] * a[1]) < 0.0)
        # predicted velocity after dt
        v_pred = v + dt * a
        return {
                'e_pos': e_pos,
                'v': v,
                'a': a,
                'dot_ea': dot_ea,
                'braking': braking,
                'v_pred': v_pred,
        }

# Try to import scipy sparse solvers for large swarms; fall back to dense
try:
    import scipy.sparse as sp
    from scipy.sparse import block_diag as sp_block_diag
    from scipy.sparse.linalg import spsolve, lsqr
    SCIPY_SPARSE_AVAILABLE = True
except Exception:
    sp = None
    sp_block_diag = None
    spsolve = None
    lsqr = None
    SCIPY_SPARSE_AVAILABLE = False

# Safe wrappers around sparse solvers so static analysis (and runtime when
# SciPy isn't available) doesn't have to reason about possibly-None callables.
def _spsolve(A, b):
    """Solve A x = b using sparse solver when available, otherwise dense."""
    # If SciPy sparse is available and spsolve is present, use it.
    if SCIPY_SPARSE_AVAILABLE and spsolve is not None:
        return spsolve(A, b)
    # Otherwise convert to dense and fall back to numpy.linalg.solve
    try:
        A_dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        b_arr = np.asarray(b)
        return np.linalg.solve(A_dense, b_arr)
    except Exception:
        # propagate to caller to handle
        raise


def _lsqr(A, b):
    """Return least-squares solution for A x = b. Uses SciPy lsqr if
    available; otherwise uses numpy.linalg.lstsq on dense arrays."""
    if SCIPY_SPARSE_AVAILABLE and lsqr is not None:
        # lsqr returns a tuple (x, istop, itn, normr, normar, norma, conda, normx)
        sol = lsqr(A, b)
        return sol[0]
    # fallback: densify and use numpy lstsq
    A_dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
    b_arr = np.asarray(b)
    sol, *_ = np.linalg.lstsq(A_dense, b_arr, rcond=None)
    return sol


def newton(F: Callable[[np.ndarray], np.ndarray],
           Jfun: Callable[[np.ndarray], Any],
           x0: Optional[np.ndarray],
           tol_f: float = 1e-8,
           tol_x: float = 1e-8,
           tol_relx: float = 1e-6,
           maxiter: int = 50,
           ls_alpha: float = 0.5,
           ls_maxiter: int = 10,
           lm_lambda: Optional[float] = 1.0,
           lm_adaptive: bool = True,
           lm_maxiter: int = 50,
           lm_factor_increase: float = 50.0,
           lm_factor_decrease: float = 0.1,
           verbose: bool = False,
           use_residual: bool = False) -> Tuple[np.ndarray, dict]:
    """Newton's method with optional adaptive Levenberg-Marquardt damping.

    By default this implementation does NOT treat the residual norm
    ||F(x)|| as the convergence criterion (see `use_residual`). The
    residual is still used for line-search and damping decisions.
    Convergence (when use_residual is False) is determined by the step
    magnitude: abs(dx) <= tol_x AND rel(dx) <= tol_relx.
    """
    if x0 is None:
        raise ValueError("x0 must be provided to newton")
    x = x0.astype(np.float64).copy()
    J = None
    hist_dx = []
    hist_rel_dx = []

    # starting LM damping (can be overridden by lm_lambda). Use a more
    # conservative (larger) starting lambda by default to improve robustness
    # when Jacobians are ill-conditioned.
    lambda_lm = lm_lambda if lm_lambda is not None else 1e-2
    info_lin = {}

    for k in range(1, maxiter + 1):
        Fx = F(x)  # Compute the residual vector F(x)
        resnorm = float(np.linalg.norm(Fx)) # Compute its L2 norm
        if verbose:
            print(f"Newton iter {k}")

        # compute jacobian J = dF/dx at current state x
        J = Jfun(x)
        info_lin = {}

        # --- Solve for the Newton step dx ---
        # The standard Newton step solves the linear system J * dx = F(x)
        # The state update will be x_new = x - dx
        dx = None
        try:
            if SCIPY_SPARSE_AVAILABLE and sp is not None and sp.issparse(J):
                dx = _spsolve(J, Fx)  # Use sparse solve
            else:
                dx = np.linalg.solve(J, Fx) # Use dense solve
        except Exception:
            dx = None # Solve failed (e.g., singular Jacobian)

        dx_used = dx
        accepted = False # Flag if we accept a step (either LM or standard Newton)

        # --- Adaptive Levenberg-Marquardt (LM) Damping ---
        # If direct solve failed (dx is None) or if adaptive LM is always on
        if (dx is None or lm_adaptive) and lm_adaptive:
            n = int(J.shape[0])
            try:
                # Estimate magnitude of J.T @ J diagonal to scale lambda
                if SCIPY_SPARSE_AVAILABLE and sp is not None and sp.issparse(J):
                    JTJ_diag_max = float((J.T @ J).diagonal().max())
                else:
                    JTJ = J.T @ J
                    JTJ_diag_max = float(np.max(np.abs(np.diag(JTJ)))) if JTJ.size else 0.0
            except Exception:
                JTJ_diag_max = 0.0

            # Set initial damping parameter (lam) for this iteration
            lam = max(lambda_lm if lambda_lm is not None else 1e-6, 1e-6 * (JTJ_diag_max + 1e-12))
            
            # Inner loop: find a damping 'lam' that reduces the residual norm
            for lm_it in range(lm_maxiter):
                dx_lm = None
                try:
                    # Solve Levenberg-Marquardt normal equations:
                    # (J.T @ J + lam * I) @ dx_lm = J.T @ Fx
                    if SCIPY_SPARSE_AVAILABLE and sp is not None and sp.issparse(J):
                        JTJ = J.T @ J
                        A = JTJ + lam * sp.eye(n, format='csr')
                        rhs = J.T @ Fx
                        dx_lm = _spsolve(A, rhs)
                    else:
                        A = J.T @ J + lam * np.eye(n)
                        rhs = J.T @ Fx
                        dx_lm = np.linalg.solve(A, rhs)
                except Exception:
                    dx_lm = None # LM solve failed

                if dx_lm is None:
                    lam *= lm_factor_increase # Solve failed, increase damping (more like gradient descent)
                    continue

                # --- Line-search on LM step ---
                # Check if this LM step (with alpha=1.0) actually reduces the residual
                step_lm = -dx_lm
                alpha = 1.0
                Fx_norm0 = resnorm
                reduced = False
                for _ in range(ls_maxiter):
                    x_trial = x + alpha * step_lm
                    Fx_trial = F(x_trial)
                    # Check if LM step reduces the residual norm
                    if np.linalg.norm(Fx_trial) < Fx_norm0:
                        dx_used = dx_lm  # This is the step we will use
                        reduced = True
                        break
                    alpha *= ls_alpha # Backtrack

                if reduced:
                    # Step was good. Decrease damping (be more like Newton)
                    lam = max(lam * lm_factor_decrease, 1e-16)
                    lambda_lm = lam # Store for next main iteration
                    info_lin['lm_lambda'] = lam
                    accepted = True # Mark step as accepted
                    break
                else:
                    # Step was bad. Increase damping and retry LM solve
                    lam *= lm_factor_increase

        # --- Fallback to Least-Squares ---
        # If both direct Newton and LM failed to produce a solve
        if dx_used is None:
            try:
                # Fallback: solve J * dx = F(x) using least-squares
                if SCIPY_SPARSE_AVAILABLE and sp is not None and sp.issparse(J):
                    dx_ls = _lsqr(J, Fx)
                else:
                    dx_ls, *_ = np.linalg.lstsq(J, Fx, rcond=None)
                dx_used = dx_ls
                info_lin['fallback'] = 'lstsq'
            except Exception:
                # All solves failed (e.g., J is degenerate)
                return x, {'success': False, 'niter': k-1, 'reason': 'singular_jacobian', **info_lin}

        # norms for step-based convergence
        abs_dx = float(np.linalg.norm(np.asarray(dx_used)))
        rel_dx = float(abs_dx / (np.linalg.norm(x) + 1e-12))

        # --- Line-search on chosen step ---
        # If we *didn't* already accept an LM step, we perform a standard
        # line search on the (direct Newton or LSQ) step.
        step = -np.asarray(dx_used)
        alpha = 1.0
        Fx_norm0 = resnorm
        if not accepted:
            reduced = False
            # Perform backtracking line-search on the computed step (dx_used)
            for ls in range(ls_maxiter):
                x_trial = x + alpha * step
                Fx_trial = F(x_trial)
                # Check if the step reduces the residual norm
                if np.linalg.norm(Fx_trial) < Fx_norm0:
                    x = x_trial # Accept the trial state
                    accepted = True
                    reduced = True
                    break
                alpha *= ls_alpha # Backtrack
            if not reduced:
                # Line search failed to find a better point.
                return x, {'success': False, 'niter': k-1, 'reason': 'line_search_failed', **info_lin}

        hist_dx.append(abs_dx)
        hist_rel_dx.append(rel_dx)

        # --- Convergence Check ---
        # convergence decision: either residual+step (legacy) or step-only
        if use_residual:
            # Check based on residual AND step size
            if (resnorm <= tol_f) and (abs_dx <= tol_x) and (rel_dx <= tol_relx):
                return x, {
                    'success': True,
                    'niter': k,
                    'dx_norm': float(abs_dx),
                    'rel_dx': float(rel_dx),
                    'jacobian': J,
                    'hist_dx': hist_dx,
                    'hist_rel_dx': hist_rel_dx,
                    **info_lin,
                }
        else:
            # Check based ONLY on step size (default)
            if (abs_dx <= tol_x) and (rel_dx <= tol_relx):
                return x, {
                    'success': True,
                    'niter': k,
                    'dx_norm': float(abs_dx),
                    'rel_dx': float(rel_dx),
                    'jacobian': J,
                    'hist_dx': hist_dx,
                    'hist_rel_dx': hist_rel_dx,
                    **info_lin,
                }

    # end for loop (maxiter reached)
    return x, {'success': False, 'niter': maxiter, 'jacobian': J,
               'hist_dx': hist_dx, 'hist_rel_dx': hist_rel_dx, **info_lin}


def solve_drone_newton(p_tuple,
                       u,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-8,
                       tol_x: float = 1e-8,
                       tol_relx: float = 1e-6,
                       maxiter: int = 50,
                       lm_lambda: Optional[float] = None,
                       verbose: bool = False):
    """Solve f(x,p,u)=0 for a single drone using Newton's method.

    Returns: x_star, info
    """
    if x0 is None:
        x0 = np.zeros(N_STATES, dtype=np.float64)

    # Wrapper function for F(x) = evalf(x, p, u)
    def F_np(x_np):
        return np.asarray(evalf(jnp.array(x_np), p_tuple, jnp.array(u)), dtype=np.float64)

    # Wrapper function for J(x) = dF/dx
    def J_np(x_np):
        J = np.asarray(compute_jacobian_jax(jnp.array(x_np), p_tuple, jnp.array(u)))
        return J

    # Call the core Newton solver
    x_star, info = newton(F_np, J_np, x0, tol_f=tol, tol_x=tol_x, tol_relx=tol_relx,
                          maxiter=maxiter, lm_lambda=lm_lambda, verbose=verbose)

    # preserve solver-level diagnostics so caller can see whether Newton itself
    # converged vs. the wrapper post-check success.
    try:
        info['solver_success'] = info.get('success', False)
        info['solver_niter'] = info.get('niter', None)
    except Exception:
        info['solver_success'] = None
        info['solver_niter'] = None

    # --- Domain-Specific Post-Check ---
    # post-check: if the computed state is close to the hover state for the
    # commanded input `u`, consider the solve successful even if the generic
    # Newton criteria were conservative. This is a practical check: if the
    # drone's state matches the desired hover state (positions/velocities), it
    # is effectively holding the hover we commanded.
    try:
        # unpack params back to dict for hover builder
        p_dict = unpack_params(p_tuple)
        # Get the "true" hover state for these parameters
        x_hover, _ = build_hover_state_and_input(p_dict)
        # compute error only on physical pose/velocity components: y, z, v_y, v_z
        pos_idx = [IDX['y'], IDX['z'], IDX['v_y'], IDX['v_z']]
        xs = np.asarray(x_star)
        xh = np.asarray(x_hover)
        err = float(np.linalg.norm(xs[pos_idx] - xh[pos_idx]))
        info['error_to_hover'] = err
        
        # final success is the OR of the solver success and the practical
        # hover post-check; preserve solver_xxx fields above for clarity.
        if err <= tol:
            info['success'] = True
    except Exception:
        # if anything goes wrong here, don't mask the solver result
        info['error_to_hover'] = None

    # attach tendency diagnostic
    try:
        info['tendency'] = assess_tendency(x_star, p_tuple, u)
    except Exception:
        info['tendency'] = None
    return x_star, info


def solve_swarm_newton(p_tuple,
                       U: Sequence[Sequence[float]],
                       X0_flat: Optional[np.ndarray] = None,
                       tol: float = 1e-8,
                       tol_x: float = 1e-8,
                       tol_relx: float = 1e-6,
                       maxiter: int = 50,
                       lm_lambda: Optional[float] = None,
                       verbose: bool = False):
    """Solve stacked f(X_flat,p_shared,U)=0 for a homogeneous swarm.

    This builds the block-diagonal Jacobian from per-drone Jacobians
    (computed via compute_jacobian_jax) and performs Newton steps on
    the flattened vector.
    """
    U_arr = np.asarray(U, dtype=np.float64)
    N = U_arr.shape[0] # Number of drones
    if X0_flat is None:
        X0_flat = np.zeros(N * N_STATES, dtype=np.float64)

    # Wrapper for the stacked residual vector F(X)
    def F_np(X_flat):
        return np.asarray(swarm_evalf(jnp.array(X_flat), p_tuple, jnp.array(U_arr)), dtype=np.float64)

    # Wrapper for the stacked block-diagonal Jacobian J(X)
    def J_np(X_flat):
        # assemble block-diagonal from per-drone Jacobians
        X = np.reshape(X_flat, (N, N_STATES))
        
        # Build sparse block-diagonal Jacobian if scipy is available
        if SCIPY_SPARSE_AVAILABLE and sp is not None and sp_block_diag is not None:
            Ji_list = []
            for i in range(N):
                xi = X[i]
                ui = U_arr[i]
                # Compute single-drone Jacobian
                Ji = np.asarray(compute_jacobian_jax(jnp.array(xi), p_tuple, jnp.array(ui)))
                Ji_list.append(sp.csr_matrix(Ji))
            # Stack Jacobians along the diagonal
            Jbig = sp_block_diag(Ji_list, format='csr')
            return Jbig
        else:
            # Fallback to dense block-diagonal Jacobian
            Jbig = np.zeros((N * N_STATES, N * N_STATES), dtype=np.float64)
            for i in range(N):
                xi = X[i]
                ui = U_arr[i]
                # Compute single-drone Jacobian
                Ji = np.asarray(compute_jacobian_jax(jnp.array(xi), p_tuple, jnp.array(ui)))
                # Place it in the correct block
                i0 = i * N_STATES
                Jbig[i0:i0+N_STATES, i0:i0+N_STATES] = Ji
            return Jbig

    # Call the core Newton solver
    X_star_flat, info = newton(F_np, J_np, X0_flat, tol_f=tol, tol_x=tol_x, tol_relx=tol_relx,
                               maxiter=maxiter, lm_lambda=lm_lambda, verbose=verbose)

    # preserve solver-level diagnostics
    try:
        info['solver_success'] = info.get('success', False)
        info['solver_niter'] = info.get('niter', None)
    except Exception:
        info['solver_success'] = None
        info['solver_niter'] = None

    # --- Domain-Specific Post-Check (Swarm) ---
    # post-check: if each drone's final state is close to its hover state for
    # the commanded input, consider the swarm solve successful.
    try:
        # Since it's a homogeneous swarm, all drones share the same hover state
        x_hover, _ = build_hover_state_and_input(unpack_params(p_tuple))
        X_star = np.reshape(X_star_flat, (N, N_STATES))
        pos_idx = [IDX['y'], IDX['z'], IDX['v_y'], IDX['v_z']]
        # Calculate error for each drone
        errs = [float(np.linalg.norm(np.asarray(X_star[i])[pos_idx] - np.asarray(x_hover)[pos_idx])) for i in range(N)]
        info['error_to_hover'] = errs
        info['max_error_to_hover'] = max(errs)
        # If max error is within tolerance, mark as successful
        if info['max_error_to_hover'] <= tol:
            info['success'] = True
    except Exception:
        info['error_to_hover'] = None
        
    # attach per-drone tendency diagnostics
    try:
        X_star = np.reshape(X_star_flat, (N, N_STATES))
        tendencies = []
        for i in range(N):
            tendencies.append(assess_tendency(X_star[i], p_tuple, U_arr[i]))
        info['tendency'] = tendencies
    except Exception:
        info['tendency'] = None
    return X_star_flat, info


def solve_swarm_newton_with_params(p_tuples,
                                   U: Sequence[Sequence[float]],
                                   X0_flat: Optional[np.ndarray] = None,
                                   tol: float = 1e-8,
                                   tol_x: float = 1e-8,
                                   tol_relx: float = 1e-6,
                                   maxiter: int = 50,
                                   lm_lambda: Optional[float] = None,
                                   verbose: bool = False):
    """Solve stacked f(X_flat,p_tuples,U)=0 for heterogeneous swarm.

    p_tuples should be a sequence of packed param tuples, one per drone.
    """
    p_arr = [tuple(p) for p in p_tuples]
    U_arr = np.asarray(U, dtype=np.float64)
    N = U_arr.shape[0] # Number of drones
    if len(p_arr) != N:
        raise ValueError("Number of p_tuples must equal number of inputs U")
    if X0_flat is None:
        X0_flat = np.zeros(N * N_STATES, dtype=np.float64)

    # Wrapper for the stacked residual vector F(X)
    def F_np(X_flat):
        return np.asarray(swarm_evalf_with_params(jnp.array(X_flat), jnp.array(p_arr), jnp.array(U_arr)), dtype=np.float64)

    # Wrapper for the stacked block-diagonal Jacobian J(X)
    def J_np(X_flat):
        X = np.reshape(X_flat, (N, N_STATES))
        
        # Build sparse block-diagonal Jacobian
        if SCIPY_SPARSE_AVAILABLE and sp is not None and sp_block_diag is not None:
            Ji_list = []
            for i in range(N):
                xi = X[i]
                ui = U_arr[i]
                # Compute Jacobian using this drone's specific parameters p_arr[i]
                Ji = np.asarray(compute_jacobian_jax(jnp.array(xi), jnp.array(p_arr[i]), jnp.array(ui)))
                Ji_list.append(sp.csr_matrix(Ji))
            Jbig = sp_block_diag(Ji_list, format='csr')
            return Jbig
        else:
            # Fallback to dense block-diagonal Jacobian
            Jbig = np.zeros((N * N_STATES, N * N_STATES), dtype=np.float64)
            for i in range(N):
                xi = X[i]
                ui = U_arr[i]
                # Compute Jacobian using this drone's specific parameters p_arr[i]
                Ji = np.asarray(compute_jacobian_jax(jnp.array(xi), jnp.array(p_arr[i]), jnp.array(ui)))
                # Place it in the correct block
                i0 = i * N_STATES
                Jbig[i0:i0+N_STATES, i0:i0+N_STATES] = Ji
            return Jbig

    # Call the core Newton solver
    X_star_flat, info = newton(F_np, J_np, X0_flat, tol_f=tol, tol_x=tol_x, tol_relx=tol_relx,
                               maxiter=maxiter, lm_lambda=lm_lambda, verbose=verbose)

    # --- Domain-Specific Post-Check (Heterogeneous Swarm) ---
    # post-check: per-drone hover error
    try:
        X_star = np.reshape(X_star_flat, (N, N_STATES))
        pos_idx = [IDX['y'], IDX['z'], IDX['v_y'], IDX['v_z']]
        errs = []
        for i in range(N):
            # Must unpack this drone's specific parameters
            p_dict = unpack_params(p_arr[i])
            # Build this drone's specific hover state
            x_hover, _ = build_hover_state_and_input(p_dict)
            # Calculate its error
            errs.append(float(np.linalg.norm(np.asarray(X_star[i])[pos_idx] - np.asarray(x_hover)[pos_idx])))
        info['error_to_hover'] = errs
        info['max_error_to_hover'] = max(errs)
        # If max error is within tolerance, mark as successful
        if info['max_error_to_hover'] <= tol:
            info['success'] = True
    except Exception:
        info['error_to_hover'] = None

    # attach per-drone tendency diagnostics for heterogeneous swarm
    try:
        X_star = np.reshape(X_star_flat, (N, N_STATES))
        tendencies = []
        for i in range(N):
            # Use per-drone parameters and inputs
            tendencies.append(assess_tendency(X_star[i], p_arr[i], U_arr[i]))
        info['tendency'] = tendencies
    except Exception:
        info['tendency'] = None

    return X_star_flat, info