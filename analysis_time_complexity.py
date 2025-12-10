import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model.drone_model_jax import get_default_params, pack_params, IDX
from utils.swarm_utils import assemble_swarm, swarm_evalf, swarm_compute_jacobian_jax

def get_simulation_setup():
    p = get_default_params()
    p_tuple = pack_params(p)
    targets = [(1.0, 1.0)]
    X0_flat, U = assemble_swarm(targets)
    return X0_flat, p_tuple, U

def step_euler(x, dt, p_tuple, U):
    dxdt = swarm_evalf(x, p_tuple, U)
    return x + dxdt * dt

def step_rk4(x, dt, p_tuple, U):
    k1 = swarm_evalf(x, p_tuple, U)
    k2 = swarm_evalf(x + 0.5 * dt * k1, p_tuple, U)
    k3 = swarm_evalf(x + 0.5 * dt * k2, p_tuple, U)
    k4 = swarm_evalf(x + dt * k3, p_tuple, U)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def step_trap(x, dt, p_tuple, U):
    tol = 1e-10
    max_iter = 100
    I = np.eye(len(x))
    
    f_curr = swarm_evalf(x, p_tuple, U)
    x_next = x + f_curr * dt
    
    for _ in range(max_iter):
        f_next = swarm_evalf(x_next, p_tuple, U)
        residual = x_next - x - (dt / 2.0) * (f_curr + f_next)
        
        if np.linalg.norm(residual) < tol:
            break
            
        J_f = swarm_compute_jacobian_jax(x_next, p_tuple, U)
        if np.isnan(J_f).sum() > 0:
            raise ValueError
            
        J_G = I - (dt / 2.0) * J_f
        delta = np.linalg.solve(J_G, -residual)
        x_next += delta
        
    return x_next

def step_trap_homotopy(x, dt, p_tuple, U):
    tol = 1e-10
    max_iter_newton = 20
    I = np.eye(len(x))

    f_curr = swarm_evalf(x, p_tuple, U)
    x_pred = x + f_curr * dt
    
    tau = 0.0
    d_tau = 0.1
    x_h = x_pred.copy()

    while tau < 1.0:
        tau_next = min(tau + d_tau, 1.0)
        
        x_try = x_h.copy()
        converged = False
        
        for _ in range(max_iter_newton):
            f_next_try = swarm_evalf(x_try, p_tuple, U)
            
            R_target = x_try - x - (dt / 2.0) * (f_curr + f_next_try)
            R_easy = x_try - x_pred
            
            H = tau_next * R_target + (1.0 - tau_next) * R_easy

            if np.linalg.norm(H) < tol:
                converged = True
                break

            J_f_try = swarm_compute_jacobian_jax(x_try, p_tuple, U)
            if np.isnan(J_f_try).sum() > 0:
                raise ValueError
                
            J_target = I - (dt / 2.0) * J_f_try
            JH = tau_next * J_target + (1.0 - tau_next) * I
            
            delta = np.linalg.solve(JH, -H)
            x_try += delta
        
        if converged:
            x_h = x_try
            tau = tau_next
            d_tau = min(d_tau * 1.5, 0.5) 
        else:
            d_tau *= 0.5
            if d_tau < 1e-4:
                return x_h 

    return x_h

def run_simulation(step_fn, x0, dt, T_final, p_tuple, U):
    X = [x0.copy()]
    T = [0]
    x = x0.copy()
    steps = int(T_final / dt)
    for i in range(steps):
        x = step_fn(x, dt, p_tuple, U)
        X.append(x)
        T.append((i+1)*dt)
    return x, np.array(X), np.array(T)


## 1. Reference

# Data
X0, p_tuple, U = get_simulation_setup()
T_final = 1.0
X_refs = {}
for dt_order in [2, 2.5, 3, 4, 5, 6]:
    if os.path.exists(f'./results/ref_euler_1e-{dt_order}.npz'):
        continue
        
    dt_ref = 10**(-dt_order)
    x_ref, X_ref, T_ref = run_simulation(step_euler, X0, dt_ref, T_final, p_tuple, U)
    
    np.savez(
        f'./results/ref_euler_1e-{dt_order}.npz', 
        X0=X0,
        p_tuple=p_tuple,
        U=U,
        T_final=T_final,
        dt_ref=dt_ref,
        T_ref=T_ref,
        x_ref=x_ref,
        X_ref=X_ref,
    )

    X_refs[dt_order] = X_ref

# Create figure
dt_orders = [2, 2.5, 3, 4, 5, 6]
colors = cm.viridis(np.linspace(0, 1, len(dt_orders)))
linestyles = ['-', '-', '--', '-.', ':', (0, (3, 1, 1, 1))]
plt.figure(figsize=(10, 8))
for i, dt_order in enumerate(dt_orders):
    d = np.load(f'./results/ref_euler_1e-{dt_order}.npz')
    X_ref = d['X_ref']
    
    lw = max(1.5, 4.0 - (i * 0.4))
    alpha = min(1.0, 0.4 + (i * 0.08))
    
    plt.plot(X_ref[:, IDX['y']], X_ref[:, IDX['z']],
             color=colors[i], 
             linewidth=lw, 
             alpha=alpha,
             linestyle=linestyles[i],
             label=r"$\Delta t = 10^{{-{:.2f}}}$".format(dt_order))

plt.xlabel('y (m)', fontsize=20)
plt.ylabel('z (m)', fontsize=20)
plt.title(r'Trajectory Convergence across Different Time Step Sizes ($\Delta t$)', fontsize=22)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15, loc='best', framealpha=0.9)
plt.tight_layout()
plt.show()


## 2. Trade-Off Plot

# Data
d = np.load('./results/ref_euler_1e-6.npz')
X_ref = d['X_ref']
methods = {
    'Euler': step_euler,
    'Trapezoidal': step_trap,
}
err_idx = [IDX['y'], IDX['z']] 
try:
    with open('./results/errors_euler_position.pkl', 'rb') as f:
        errors = pickle.load(f)
except:
    errors = {name: [] for name in methods}
    for dt_order in tqdm(dt_orders):
        dt = 10**(-dt_order)
        for name, func in methods.items():
            x_final, _, _ = run_simulation(func, X0, dt, T_final, p_tuple, U)
            error = np.max(np.abs(x_final - x_ref)[err_idx])
            errors[name].append(error)
    
    with open('./results/errors_euler_position.pkl', 'wb') as f:
        pickle.dump(errors, f)

# Create figure
plt.figure(figsize=(10, 6))
plt.plot([0.0485, 0.55, 5.55], np.array(errors['Euler'])[[4, 2, 1]], 'o-',  label="Euler")
plt.plot([0.420, 3.02, 23.5], np.array(errors['Trapezoidal'])[[4, 2, 1]], 'o-', label="Trapezoidal")

plt.xscale('log')
plt.yscale('log')
plt.ylabel('Position Error (vs Reference)', fontsize=20)
plt.xlabel('Execution Time (s)', fontsize=20)
plt.title('Solver Time Complexity Analysis \n (Execution Time for Simulating 1 second.)', fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15) 
plt.legend(fontsize=15)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

## 3. Larger-Scale Experiments
import matplotlib.pyplot as plt
import numpy as np

# Data
N = np.array([180, 540, 900, 1800])
forward_euler_mean = np.array([0.470, 0.484, 0.492, 0.503])
forward_euler_std = np.array([0.002, 0.004, 0.009, 0.018])
trapezoidal_mean = np.array([2.11, 5.60, 12.1, 57.8])
trapezoidal_std = np.array([0.040, 0.033, 0.146, 0.563])

# Create figure
plt.figure(figsize=(10, 6))
plt.errorbar(N, forward_euler_mean, yerr=forward_euler_std, 
             marker='o', markersize=8, capsize=5, capthick=2,
             label='Euler', linewidth=2)
plt.errorbar(N, trapezoidal_mean, yerr=trapezoidal_std, 
             marker='s', markersize=8, capsize=5, capthick=2,
             label='Trapezoidal', linewidth=2)
plt.xlabel('Number of States (N)', fontsize=20)
plt.ylabel('Per-Step Execution Time (ms)', fontsize=20)
plt.title('Solver Execution Time vs Problem Size', fontsize=25)
plt.legend(fontsize=15)
plt.grid(True, alpha=0.3)
plt.xticks(N, fontsize=15)
plt.yticks(fontsize=15) 
plt.tight_layout()
plt.show()