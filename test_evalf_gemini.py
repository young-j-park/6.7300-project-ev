import time
import jax.numpy as jnp
import numpy as np

# Assuming your model is in this file and is named evalf_jax
from vehicle_model_jax import evalf_jax as evalf 
from vehicle_model_jax import get_default_params

# Use a small tolerance for floating-point comparisons
TOL = 1e-6

def test_1_steady_state(p_tuple):
    """Tests the model at a dead stop with no commands."""
    print("--- 1. Testing Steady State (Do Nothing) ---")
    x0 = jnp.zeros(10)
    u = jnp.array([0.0, 0.0])
    
    f = np.asarray(evalf(x0, p_tuple, u))
    
    # EXPECTATION: If nothing moves and nothing is commanded, all derivatives must be zero.
    assert np.allclose(f, 0.0, atol=TOL), "Vehicle should not move on its own."
    print("✅ Passed: Vehicle remains stationary as expected.\n")

def test_2_constant_speed(p_tuple, n_trials=100):
    """Tests if the model correctly handles cruising at a constant speed."""
    print(f"--- 2. Testing Constant Speed ({n_trials} trials) ---")
    for _ in range(n_trials):
        v0 = np.random.uniform(1.0, 30.0) # Realistic cruising speed
        
        x0 = jnp.zeros(10).at[7].set(v0)  # Initial speed = v0
        u = jnp.array([v0, 0.0])         # Reference speed = v0
        
        f = np.asarray(evalf(x0, p_tuple, u))
        
        # EXPECTATION: At constant speed, the controller provides just enough torque
        # to counteract drag. Therefore, net acceleration and all other rates of
        # change (except position) must be zero.
        assert abs(f[0]) < TOL  # d(i_ds)/dt should be zero (steady current)
        assert abs(f[1]) < TOL  # d(i_qs)/dt should be zero (steady current)
        assert abs(f[2]) < TOL  # i_ds should match its reference
        assert abs(f[3]) < TOL  # i_qs should match its reference
        assert abs(f[4] - v0) < TOL  # d(x_pos)/dt = v*cos(0) = v0
        assert abs(f[5]) < TOL  # d(y_pos)/dt = v*sin(0) = 0
        assert abs(f[6]) < TOL  # d(theta)/dt = omega = 0
        assert abs(f[7]) < TOL  # d(v)/dt (acceleration) must be zero
        assert abs(f[8]) < TOL  # d(omega)/dt (angular accel) must be zero
        assert abs(f[9]) < TOL  # d(I_err_v)/dt = v_ref - v = 0

    print(f"✅ Passed: Maintained constant speed correctly.\n")

def test_3_acceleration(p_tuple, n_trials=100):
    """Tests the initial response to an acceleration command from a standstill."""
    print(f"--- 3. Testing Acceleration ({n_trials} trials) ---")
    for _ in range(n_trials):
        v_ref = np.random.uniform(5.0, 30.0)
        
        x0 = jnp.zeros(10)         # Start from rest (v=0)
        u = jnp.array([v_ref, 0.0]) # Command a positive speed
        
        f = np.asarray(evalf(x0, p_tuple, u))
        
        # EXPECTATION: To accelerate, the controller must ramp up the torque-producing
        # current (i_qs), while also adjusting the flux current (i_ds) for efficiency.
        # Due to inertia, the vehicle's acceleration is initially zero.
        assert f[0] < 0.0 # d(i_ds)/dt is negative for IPMSM MTPA
        assert f[1] > 0.0 # d(i_qs)/dt must be positive to increase torque
        assert f[3] > 0.0 # d(I_err_qs)/dt = i_qs_ref - i_qs > 0
        assert abs(f[4]) < TOL # d(x_pos)/dt = v = 0 initially
        assert abs(f[7]) < TOL # d(v)/dt = 0 initially, as torque needs time to build
        assert abs(f[8]) < TOL # No turning command
        assert abs(f[9] - v_ref) < TOL # d(I_err_v)/dt = v_ref - v = v_ref

    print(f"✅ Passed: Acceleration commands behave as expected.\n")

def test_4_spin(p_tuple, n_trials=100):
    """Tests the initial response to a spin-in-place command."""
    print(f"--- 4. Testing Spin-in-Place ({n_trials} trials) ---")
    for _ in range(n_trials):
        theta_ref = np.random.uniform(0.1, np.pi/2)
        
        x0 = jnp.zeros(10)
        u = jnp.array([0.0, theta_ref]) # Command a turn with zero speed
        
        f = np.asarray(evalf(x0, p_tuple, u))
        
        # EXPECTATION: The yaw controller must command a torque to start the turn,
        # resulting in angular acceleration. No forward motion should occur.
        assert abs(f[7]) < TOL  # No longitudinal acceleration
        assert f[8] > 0.0    # Must have positive angular acceleration to start turn
        assert abs(f[1]) < TOL  # No torque command from velocity controller
        
    print(f"✅ Passed: Spin commands behave as expected.\n")

def test_5_level_turn(p_tuple, n_trials=100):
    """Tests entering a turn while at cruising speed."""
    print(f"--- 5. Testing Level Turn ({n_trials} trials) ---")
    for _ in range(n_trials):
        v0 = np.random.uniform(1.0, 30.0)
        theta_ref = np.random.uniform(0.1, np.pi/2)
        
        x0 = jnp.zeros(10).at[7].set(v0)
        u = jnp.array([v0, theta_ref])
        
        f = np.asarray(evalf(x0, p_tuple, u))
        
        # EXPECTATION: The vehicle should maintain its speed (zero acceleration)
        # while simultaneously starting a turn (positive angular acceleration).
        assert abs(f[7]) < TOL  # d(v)/dt must be zero to maintain speed
        assert f[8] > 0.0    # Must have angular acceleration to initiate the turn
        assert abs(f[9]) < TOL  # v_ref = v, so velocity error is zero
        
    print(f"✅ Passed: Level turn commands behave as expected.\n")

def main():
    """Runs the full benchmark suite."""
    print("🚀 Starting Vehicle Dynamics Benchmark Suite 🚀\n")
    start_time = time.time()
    
    p_dict = get_default_params()
    p_tuple = tuple(p_dict.values())
    
    test_1_steady_state(p_tuple)
    test_2_constant_speed(p_tuple)
    test_3_acceleration(p_tuple)
    test_4_spin(p_tuple)
    test_5_level_turn(p_tuple)
    
    end_time = time.time()
    print(f"🎉 All benchmarks passed in {end_time - start_time:.4f} seconds. 🎉")

if __name__ == "__main__":
    main()