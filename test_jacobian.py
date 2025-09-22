
import json
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from vehicle_model_jax import compute_jacobian_jax, get_default_params


J_analytic = {
    (0, 0): lambda p, x: (-p['R_s'] - p['K_p_i']) / p['L_ds'],
    (0, 1): lambda p, x: (p['G'] / p['r_w'] * x[7]) * p['L_qs'] / p['L_ds'],
    (0, 2): lambda p, x: p['K_i_i'] / p['L_ds'],
    
    (1, 0): lambda p, x: -(p['G'] / p['r_w'] * x[7]) * p['L_ds'] / p['L_qs'],
    (1, 1): lambda p, x: (-p['R_s'] - p['K_p_i']) / p['L_qs'],
    (1, 3): lambda p, x: p['K_i_i'] / p['L_qs'],
    
    (2, 0): lambda p, x: -1.0,
    (3, 1): lambda p, x: -1.0,
    
    (4, 6): lambda p, x: -x[7] * np.sin(x[6]),
    (4, 7): lambda p, x: np.cos(x[6]),
    
    (5, 6): lambda p, x: x[7] * np.cos(x[6]),
    (5, 7): lambda p, x: np.sin(x[6]),
    
    (6, 8): lambda p, x: 1.0,
    
    (7, 1): lambda p, x: (p['k'] * (3/2) * p['p_pairs'] * p['lambda_f']) / p['m'],
    
    (8, 6): lambda p, x: -p['K_p_theta'] / p['I'],
    (8, 8): lambda p, x: (-p['K_d_theta'] - p['c_r']) / p['I'],
    
    (9, 7): lambda p, x: -1.0,
}


def is_close(y, yhat, eps=1e-10):
    return np.abs(y - yhat) < eps


def main():

    # 0. Mannual Regression
    with open('./test_benchmarks/test_cases.json', 'r') as f:
        data = json.load(f)

    for test_case in data:
        x0 = jnp.array(test_case["x"])
        p_tuple = tuple(test_case["p"])
        u = jnp.array(test_case["u"])
        J_true = jnp.array(test_case["J"])
        J = compute_jacobian_jax(x0, p_tuple, u)

        assert np.all(np.abs(J_true - J).flatten() < 1e-10)
    
    print("Passed the manual regression test")
        

    # Load Default Params
    p = get_default_params()
    p_tuple = tuple(p.values())

    # 1. Steady State
    x0 = jnp.zeros(10)
    u = jnp.array([0.0, 0.0])
    
    J_jax = compute_jacobian_jax(x0, p_tuple, u)
    for (i, j), J_val_func in J_analytic.items():
        assert is_close(J_jax[i, j], J_val_func(p, x0), eps=1e-10)
    
    print("Passed the steady-state test")

    # 2. Constant Speed
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        
        x0 = jnp.zeros(10)
        x0 = x0.at[7].set(v0)  # v
        u = jnp.array([v0, 0.0])
        
        J_jax = compute_jacobian_jax(x0, p_tuple, u)
        for (i, j), J_val_func in J_analytic.items():
            assert is_close(J_jax[i, j], J_val_func(p, x0), eps=1e-10)
    
    print(f"Passed the constant-speed state test ({n_trials} trials)")

    # 3. Acceleration
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        
        x0 = jnp.zeros(10)
        u = jnp.array([v0, 0.0])
        
        J_jax = compute_jacobian_jax(x0, p_tuple, u)
        for (i, j), J_val_func in J_analytic.items():
            assert is_close(J_jax[i, j], J_val_func(p, x0), eps=1e-10)
    
    print(f"Passed the accelration test ({n_trials} trials)")

    # 4. Spin
    n_trials = 1000
    for _ in range(n_trials):
        w0 = np.random.rand() * np.pi/2
        
        x0 = jnp.zeros(10)
        u = jnp.array([0.0, w0])
        
        J_jax = compute_jacobian_jax(x0, p_tuple, u)
        for (i, j), J_val_func in J_analytic.items():
            assert is_close(J_jax[i, j], J_val_func(p, x0), eps=1e-10)
    
    print(f"Passed the spinning test ({n_trials} trials)")

    # 5. Level Turn
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        w0 = np.random.rand() * np.pi/2
        
        x0 = jnp.zeros(10)
        x0 = x0.at[7].set(v0)  # v
        u = jnp.array([v0, w0])
        
        J_jax = compute_jacobian_jax(x0, p_tuple, u)
        for (i, j), J_val_func in J_analytic.items():
            assert is_close(J_jax[i, j], J_val_func(p, x0), eps=1e-10)
    
    print(f"Passed the level turn test ({n_trials} trials)")


if __name__ == "__main__":
    main()
