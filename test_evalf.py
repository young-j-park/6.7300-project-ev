import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from vehicle_model_jax import evalf, get_default_params, OUTPUT_POSITIONS


def compute_wind_force(p, v_state=0.0, theta=0.0):
    """Compute longitudinal wind force w_F given parameter dict p and state v,theta."""
    v_w = p['v_w']
    psi = p['psi']
    c_wx = p['c_wx']
    u_a = v_state - v_w * np.cos(psi - theta)
    w_F = -c_wx * u_a * abs(u_a)
    return w_F


def expected_longitudinal_accel(p, v_state=0.0, i_qs_r_state=0.0, theta=0.0):
    """Compute expected longitudinal acceleration (dxdt at idx_v) using params p.
    Includes electromagnetic torque term from i_qs_r_state, drag, and wind force.
    """
    m = p['m']
    k = p.get('k', p['eta_g'] * p['G'] / p['r_w'])
    p_pairs = p['p_pairs']
    lambda_f = p['lambda_f']
    c_d = p['c_d']

    # electromagnetic torque contribution: k * (3/2) * p_pairs * lambda_f * i_qs_r
    em_term = k * (3.0/2.0) * p_pairs * lambda_f * i_qs_r_state
    w_F = compute_wind_force(p, v_state=v_state, theta=theta)
    return (1.0 / m) * (em_term - c_d * v_state + w_F)


def main():
    # Determine state indices based on OUTPUT_POSITIONS flag
    if OUTPUT_POSITIONS:
        # State: [i_ds_r, i_qs_r, I_err_ds, I_err_qs, x_pos, y_pos, theta, v, omega, I_err_v]
        idx_x_pos = 4
        idx_y_pos = 5
        idx_theta = 6
        idx_v = 7
        idx_omega = 8
        idx_I_err_v = 9
        n_states = 10
    else:
        # State: [i_ds_r, i_qs_r, I_err_ds, I_err_qs, theta, v, omega, I_err_v]
        idx_x_pos = None
        idx_y_pos = None
        idx_theta = 4
        idx_v = 5
        idx_omega = 6
        idx_I_err_v = 7
        n_states = 8
    
    # 0. Manual Regression
    with open('./test_benchmarks/test_cases.json', 'r') as f:
        data = json.load(f)
    
    for test_case in data:
        x0_full = jnp.array(test_case["x"])
        
        # Remove position states if OUTPUT_POSITIONS=False
        if not OUTPUT_POSITIONS and len(x0_full) == 10:
            x0 = jnp.concatenate([x0_full[:4], x0_full[6:]])
        else:
            x0 = x0_full
            
        p_tuple = tuple(test_case["p"])
        u = jnp.array(test_case["u"])
        
        f_true_full = jnp.array(test_case["f"])
        
        # Remove position derivatives if OUTPUT_POSITIONS=False
        if not OUTPUT_POSITIONS and len(f_true_full) == 10:
            f_true = jnp.concatenate([f_true_full[:4], f_true_full[6:]])
        else:
            f_true = f_true_full
            
        f = evalf(x0, p_tuple, u)
        assert np.all(np.abs(f_true - f) < 1e-10)
    
    print("Passed the manual regression test")
        
    # Load Default Params
    p = get_default_params()
    p_tuple = tuple(p.values())
    
    # 1. Steady State
    x0 = jnp.zeros(n_states)
    u = jnp.array([0.0, 0.0])
    
    f = evalf(x0, p_tuple, u)

    expected = np.zeros(n_states, dtype=float)
    expected[idx_v] = expected_longitudinal_accel(p, v_state=0.0, i_qs_r_state=0.0, theta=0.0)

    # compare with tolerance because of floating point / JAX types
    assert np.allclose(np.array(f, dtype=float), expected, atol=1e-12)

    print("Passed the steady-state test (accounting for wind/default forcing)")
    
    # 2. Constant Speed
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        
        x0 = jnp.zeros(n_states)
        x0 = x0.at[idx_v].set(v0)  # v
        u = jnp.array([v0, 0.0])
        
        f = evalf(x0, p_tuple, u)
    
        if OUTPUT_POSITIONS:
            assert f[idx_x_pos] == x0[idx_v]  # dx/dt = v*cos(0)
            assert f[idx_y_pos] == 0.0  # dy/dt = v*sin(0) = 0
        assert f[idx_theta] == 0.0  # zero angular speed
        expected_v = expected_longitudinal_accel(p, v_state=float(v0), i_qs_r_state=0.0, theta=0.0)
        assert np.allclose(float(f[idx_v]), float(expected_v), atol=1e-12)
        assert f[idx_omega] == 0.0  # zero angular acceleration
        assert f[idx_I_err_v] == 0.0  # no speed delta needed
    
    print(f"Passed the constant-speed state test ({n_trials} trials)")
    
    # 3. Acceleration
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        
        x0 = jnp.zeros(n_states)
        u = jnp.array([v0, 0.0])
        
        f = evalf(x0, p_tuple, u)
    
        if OUTPUT_POSITIONS:
            assert f[idx_x_pos] == 0.0  # zero longitudinal speed
            assert f[idx_y_pos] == 0.0  # zero lateral speed
        assert f[idx_theta] == 0.0  # zero angular speed
        expected_v = expected_longitudinal_accel(p, v_state=0.0, i_qs_r_state=0.0, theta=0.0)
        assert np.allclose(float(f[idx_v]), float(expected_v), atol=1e-12)
        assert f[idx_omega] == 0.0  # zero angular acceleration
        assert f[idx_I_err_v] == u[0]  # speed delta
    
    print(f"Passed the acceleration test ({n_trials} trials)")
    
    # 4. Spin
    n_trials = 1000
    for _ in range(n_trials):
        w0 = np.random.rand() * np.pi/2
        
        x0 = jnp.zeros(n_states)
        u = jnp.array([0.0, w0])
        
        f = evalf(x0, p_tuple, u)
    
        assert f[idx_omega] > 0.0  # angular acceleration
        
        # Check other states match expected values given defaults (wind may produce
        # a non-zero longitudinal acceleration). Compute expected longitudinal
        # acceleration same as in the steady-state test.
        expected = np.zeros(n_states, dtype=float)
        expected[idx_v] = expected_longitudinal_accel(p, v_state=0.0, i_qs_r_state=0.0, theta=0.0)

        # compare all entries: omega should be positive but we only check sign;
        # other states should equal expected
        for i in range(n_states):
            if i == idx_omega:
                continue
            assert np.allclose(float(f[i]), float(expected[i]), atol=1e-12)
    
    print(f"Passed the spinning test ({n_trials} trials)")
    
    # 5. Level Turn
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        w0 = np.random.rand() * np.pi/2
        
        x0 = jnp.zeros(n_states)
        x0 = x0.at[idx_v].set(v0)  # v
        u = jnp.array([v0, w0])
        
        f = evalf(x0, p_tuple, u)
    
        if OUTPUT_POSITIONS:
            assert f[idx_x_pos] == x0[idx_v]  # longitudinal speed
            assert f[idx_y_pos] == 0.0  # zero lateral speed
        assert f[idx_theta] == 0.0  # zero angular speed
        # compute expected longitudinal acceleration for level turn (same as
        # constant-speed calculation: drag + wind; electromagnetic term zero for
        # zero currents)
        expected_v = expected_longitudinal_accel(p, v_state=float(v0), i_qs_r_state=0.0, theta=0.0)
        assert np.allclose(float(f[idx_v]), float(expected_v), atol=1e-12)
        assert f[idx_omega] > 0.0  # angular acceleration
        assert f[idx_I_err_v] == 0.0  # no speed delta needed
    
    print(f"Passed the level turn test ({n_trials} trials)")


if __name__ == "__main__":
    main()