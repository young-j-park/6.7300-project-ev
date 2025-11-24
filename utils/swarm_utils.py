"""Utilities for assembling and evaluating a swarm of identical drones.

This module provides:
- assemble_swarm(targets): build a flattened initial state vector and U array
  for N drones given a list/array of (y,z) targets. All drone initial states
  start at zeros (y=0,z=0) as requested.
- swarm_evalf(X_flat, p_tuple, U): evaluate the stacked dynamics by
  vmap-ing the single-drone `evalf` function from `drone_model_jax`.

The design keeps parity with the single-drone API:
  - single evalf: f(x, p_tuple, u) -> dxdt (shape (N_STATES,))
  - swarm_evalf: f_swarm(X_flat, p_tuple, U) -> dX_flat (shape (N*N_STATES,))

No collision or inter-drone coupling is included.
"""
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from model.drone_model_jax import evalf, N_STATES


def assemble_swarm(targets: Sequence[Tuple[float, float]]):
    """Assemble initial flattened state vector and input array for a swarm.

    Args:
        targets: iterable of (y_ref, z_ref) for each drone. Shape (N,2).

    Returns:
        X0_flat: jnp.ndarray, shape (N * N_STATES,), initial states (all zeros)
        U: jnp.ndarray, shape (N, 2), the stacked reference inputs per drone

    Notes:
        - All drones start with zero internal states (positions/velocities zero).
        - Targets are returned as the input array (ready to pass to `swarm_evalf`).
    """
    U = jnp.asarray(targets, dtype=jnp.float64)
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError("targets must be shape (N,2) or sequence of (y,z) tuples")

    N = U.shape[0]
    X0 = jnp.zeros((N, N_STATES), dtype=jnp.float64)
    return X0.reshape((N * N_STATES,)), U


def swarm_evalf(X_flat: jnp.ndarray, p_tuple, U: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the swarm dynamics by mapping `evalf` across drones.

    Args:
        X_flat: flattened state vector shape (N * N_STATES,)
        p_tuple: packed params tuple (shared by all drones)
        U: inputs array shape (N,2) with (y_ref, z_ref) per drone

    Returns:
        dX_flat: flattened derivatives shape (N * N_STATES,)

    Implementation detail:
        This uses `jax.vmap` over the first axis of the reshaped states/inputs.
    """
    X = jnp.reshape(X_flat, (-1, N_STATES))
    if U.shape[0] != X.shape[0]:
        raise ValueError("Number of inputs (U) must match number of drones in X_flat")

    # Vectorize the single-drone evalf: map over (x, u) with shared p_tuple
    f_vmap = jax.vmap(evalf, in_axes=(0, None, 0))
    dX = f_vmap(X, p_tuple, U)  # shape (N, N_STATES)
    return jnp.reshape(dX, (-1,))


# JIT-ed convenience wrapper: compiles the batched evaluation with XLA.
swarm_jacobian_func_raw = jax.jacobian(swarm_evalf, argnums=0)
swarm_compute_jacobian_jax = jax.jit(swarm_jacobian_func_raw)
swarm_evalf_jit = jax.jit(swarm_evalf)


def swarm_evalf_with_params(X_flat: jnp.ndarray, p_tuples, U: jnp.ndarray) -> jnp.ndarray:
    """Evaluate swarm dynamics where each drone has its own params tuple.

    Args:
        X_flat: flattened state vector shape (N * N_STATES,)
        p_tuples: array-like with shape (N, P) where P is len(p_tuple)
                  (each row is a packed params tuple for the corresponding drone)
        U: inputs array shape (N,2)

    Returns:
        dX_flat: flattened derivatives shape (N * N_STATES,)

    Raises:
        ValueError if the number of per-drone param tuples doesn't match N or
        if U.shape[0] != N.

    Notes:
        - This function vmap-s across (x, p_tuple_row, u) so it supports
          heterogenous drone parameters.
    """
    X = jnp.reshape(X_flat, (-1, N_STATES))
    p_arr = jnp.asarray(p_tuples)
    if p_arr.ndim != 2:
        raise ValueError("p_tuples must be a 2D array-like with shape (N, P)")
    N = X.shape[0]
    if p_arr.shape[0] != N:
        raise ValueError("Number of per-drone param tuples must equal number of drones in X_flat")
    if U.shape[0] != N:
        raise ValueError("Number of inputs (U) must match number of drones in X_flat")

    # vmap across x (0), p_tuple row (0), and u (0)
    f_vmap = jax.vmap(evalf, in_axes=(0, 0, 0))
    dX = f_vmap(X, p_arr, U)
    return jnp.reshape(dX, (-1,))


# JIT-ed wrapper for the per-drone-params variant.
swarm_evalf_with_params_jit = jax.jit(swarm_evalf_with_params)
