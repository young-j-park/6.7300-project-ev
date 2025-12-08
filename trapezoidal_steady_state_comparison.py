import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# 1. Model Imports
from model.drone_model_jax import get_default_params, pack_params, N_STATES, IDX

# 2. Solver Imports
# Assumes steady_state_homotopy_LU.py is in the same directory
from steady_state_homotopy_LU import homotopy_solve

# Assumes analysis_simulation_methods.py is available
# (You may need to convert the .ipynb to .py first: jupyter nbconvert --to script analysis_simulation_methods.ipynb)
from analysis_simulation_methods import step_trap

# ==============================================================================
# Main Execution
# ==============================================================================

# 1. Setup Parameters
p = get_default_params()
p_tuple = pack_params(p)

# Target: Command drone to y=1.0, z=1.0
u_target = np.array([1.0, 1.0])

# Initial guess: Zero state
x0 = np.zeros(N_STATES)

# ------------------------------------------------------------------------------
# 2. Solve Steady State (Using Homotopy from imported file)
# ------------------------------------------------------------------------------
print("Finding Steady State via Homotopy...")

# homotopy_solve expects single u input: shape (2,)
x_ss, path = homotopy_solve(
    x0,
    p_tuple,
    u_target,
    tau_start=0.0,
    tau_end=1.0,
    verbose=True
)

y_ss = x_ss[IDX['y']]
print(f"Steady State y: {y_ss:.4f}")

# ------------------------------------------------------------------------------
# 3. Dynamic Simulation (Using Step_Trap from imported file)
# ------------------------------------------------------------------------------
print("Running Dynamic Simulation via Trapezoidal Step...")

dt = 0.01
T_final = 8.0
steps = int(T_final / dt)
time_grid = np.linspace(0, T_final, steps + 1)

# step_trap in the notebook expects a SWARM input U: shape (N, 2)
# We must wrap our single u_target in a list/array
U_swarm = np.array([u_target])

# Start from rest (x=0)
x_curr = np.zeros(N_STATES)
y_traj = [x_curr[IDX['y']]]

for _ in range(steps):
    # step_trap returns the next state x_{n+1}
    x_curr = step_trap(x_curr, dt, p_tuple, U_swarm)
    y_traj.append(x_curr[IDX['y']])

# ------------------------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Plot Dynamic Trajectory
plt.plot(time_grid, y_traj, label='Trapezoidal Dynamics', linewidth=2, color='blue')

# Plot Steady State Reference
plt.axhline(y=y_ss, color='r', linestyle='--', linewidth=2, label=f'Steady State (y={y_ss:.2f})')

plt.xlabel('Time (s)')
plt.ylabel('Position y (m)')
plt.title('Verification: Trapezoidal Solution Converges to Homotopy Steady State')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()