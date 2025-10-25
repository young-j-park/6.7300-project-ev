import numpy as np
import jax.numpy as jnp

from drone_model_jax import get_default_params, pack_params, N_STATES, IDX
from swarm_utils import assemble_swarm, swarm_evalf


def test_assembler_drone():
    # Build params and p_tuple
    p = get_default_params()
    p_tuple = pack_params(p)

    # Three different target positions (y,z) for three drones
    targets = [(1.0, 2.0), (0.5, 1.5), (0.0, 1.0)]
    X0_flat, U = assemble_swarm(targets)

    N = len(targets)
    # Shapes
    assert X0_flat.shape == (N * N_STATES,)
    assert U.shape == (N, 2)

    # Initial positions must be zero
    for i in range(N):
        base = i * N_STATES
        assert float(X0_flat[base + IDX['y']]) == 0.0
        assert float(X0_flat[base + IDX['z']]) == 0.0

    # Inputs assembled correctly
    assert np.allclose(np.array(U), np.array(targets))

    # Evaluate swarm dynamics at t=0 (all-zero states)
    dX_flat = swarm_evalf(X0_flat, p_tuple, jnp.array(U))

    # Kinematic checks across all drones: dy == v_y and dz == v_z and dtheta == wd
    for i in range(N):
        base = i * N_STATES
        dy = float(dX_flat[base + IDX['y']])
        dz = float(dX_flat[base + IDX['z']])
        dtheta = float(dX_flat[base + IDX['theta']])

        # initial v_y, v_z, wd are zero, so derivatives should be zero
        assert np.allclose(dy, 0.0, atol=1e-12)
        assert np.allclose(dz, 0.0, atol=1e-12)
        assert np.allclose(dtheta, 0.0, atol=1e-12)

    print("OK: swarm assembler basic tests passed")


if __name__ == "__main__":
    test_assembler_drone()
