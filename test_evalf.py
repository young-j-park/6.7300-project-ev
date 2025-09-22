
import json
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from vehicle_model_jax import evalf, get_default_params


def main():

    # 0. Mannual Regression
    with open('./test_benchmarks/test_cases.json', 'r') as f:
        data = json.load(f)

    for test_case in data:
        x0 = jnp.array(test_case["x"])
        p_tuple = tuple(test_case["p"])
        u = jnp.array(test_case["u"])
        f_true = jnp.array(test_case["f"])
        f = evalf(x0, p_tuple, u)

        assert np.all(np.abs(f_true - f) < 1e-10)
    
    print("Passed the manual regression test")
        

    # Load Default Params
    p = get_default_params()
    p_tuple = tuple(p.values())
    
    # 1. Steady State
    x0 = jnp.zeros(10)
    u = jnp.array([0.0, 0.0])
    
    f = evalf(x0, p_tuple, u)
    
    assert np.all(f == 0.0)
    
    print("Passed the steady-state test")

    # 2. Constant Speed
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        
        x0 = jnp.zeros(10)
        x0 = x0.at[7].set(v0)  # v
        u = jnp.array([v0, 0.0])
        
        f = evalf(x0, p_tuple, u)
    
        assert f[4] == x0[7]  # longitudinal speed
        assert f[5] == 0.0  # zero lateral speed
        assert f[6] == 0.0  # zero angular speed
        assert f[7] < 0.0  # drag
        assert f[8] == 0.0  # zero angular acceleration
        assert f[9] == 0.0  # no speed delta needed
    
    print(f"Passed the constant-speed state test ({n_trials} trials)")

    # 3. Acceleration
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        
        x0 = jnp.zeros(10)
        u = jnp.array([v0, 0.0])
        
        f = evalf(x0, p_tuple, u)
    
        assert f[4] == 0.0  # zero longitudinal speed
        assert f[5] == 0.0  # zero lateral speed
        assert f[6] == 0.0  # zero angular speed
        assert f[7] == 0.0  # zero longitudinal external force 
        assert f[8] == 0.0  # zero angular acceleration
        assert f[9] == u[0]  # speed delta
    
    print(f"Passed the accelration test ({n_trials} trials)")

    # 4. Spin
    n_trials = 1000
    for _ in range(n_trials):
        w0 = np.random.rand() * np.pi/2
        
        x0 = jnp.zeros(10)
        u = jnp.array([0.0, w0])
        
        f = evalf(x0, p_tuple, u)
    
        assert f[8] > 0.0  # angular acceleration
        assert np.all(f[:8] == 0.0) & np.all(f[9:] == 0.0)
    
    print(f"Passed the spinning test ({n_trials} trials)")

    # 5. Level Turn
    n_trials = 1000
    for _ in range(n_trials):
        v0 = np.random.rand() * 10.0
        w0 = np.random.rand() * np.pi/2
        
        x0 = jnp.zeros(10)
        x0 = x0.at[7].set(v0)  # v
        u = jnp.array([v0, w0])
        
        f = evalf(x0, p_tuple, u)
    
        assert f[4] == x0[7]  # longitudinal speed
        assert f[5] == 0.0  # zero lateral speed
        assert f[6] == 0.0  # zero angular speed
        assert f[7] < 0.0  # drag
        assert f[8] > 0  # angular acceleration
        assert f[9] == 0.0  # no speed delta needed
    
    print(f"Passed the level turn test ({n_trials} trials)")


if __name__ == "__main__":
    main()
