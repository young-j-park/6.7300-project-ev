
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def get_default_params():
    """
    Return default parameters for EV Vehicle
    """
    p = {
        # Vehicle Dynamics
        'm': 1500,         # mass (kg)
        'I': 2500,         # inertia (kg*m^2)
        'c_d': 25,         # longitudinal damping coefficient (N/(m/s))
        'c_r': 500,        # rotational damping coefficient (Nm/(rad/s))
        
        # Motor Dynamics
        'R_s': 0.1,        # resistance (ohms)
        'L_ds': 0.002,     # inductance (H)
        'L_qs': 0.003,     # inductance (H)
        'lambda_f': 0.5,   # flux linkage (Wb)
        'p_pairs': 4,      # number of pole pairs
        
        # Motor-Vehicle Torque Conversion
        'G': 10.0,         # gear ratio
        'r_w': 0.3,        # wheel radius (m)
        'eta_g': 0.95,     # gearbox efficiency
        'k': None,         # placeholder

        # Controllers
        'K_p_i': 10.0,     # current controller - proportional gain
        'K_i_i': 100.0,    # current controller - integral gain
        'K_p_v': 100.0,    # velocity controller - proportional gain
        'K_i_v': 100.0,     # velocity controller - integral gain
        'K_p_theta': 1000.0, # yaw controller - proportional gain
        'K_d_theta': 1000.0, # yaw controller - derivative gain
        
        # Wind
        'v_w': 0.0,        # wind speed (m/s)
        'psi': 0.0,        # wind direction (rad)
        'c_wx': 0.5,      # wind longitudinal aerodynamic coefficient
        'c_wy': 0.5       # wind lateral aerodynamic coefficient
    }
    
    p['k'] = p['eta_g'] * p['G'] / p['r_w']  # update torque-to-force conversion factor: k = eta_g G / r_w
    return p


@jax.jit
def evalf(x, p_tuple, u):
    """
    Compute the derivates of the states: dx/dt
    """
    # 1. Unpack x (state), p (parameter), u (input)
    i_ds_r, i_qs_r, I_err_ds, I_err_qs, x_pos, y_pos, theta, v, omega, I_err_v = x

    m, I, c_d, c_r, \
    R_s, L_ds, L_qs, lambda_f, p_pairs, \
    G, r_w, eta_g, k, \
    K_p_i, K_i_i, K_p_v, K_i_v, K_p_theta, K_d_theta, \
    v_w, psi, c_wx, c_wy = p_tuple
    
    v_ref, theta_ref = u

    # 2. Intermediate Variables (Appendix A)
    # IPMSM
    def f_MTPA(T_e_ref):
        K_t = (3/2) * p_pairs
        delta_L = L_ds - L_qs
        sqrt_term = jnp.sqrt(lambda_f**2 + 8 * delta_L**2 * (T_e_ref / K_t)**2)
        
        i_ds_ref = (lambda_f - sqrt_term) / (4 * delta_L)
        denom = K_t * (lambda_f + delta_L * i_ds_ref)
        i_qs_ref = jnp.where(jnp.isclose(denom, 0), 0.0, T_e_ref / denom)
    
        return i_ds_ref, i_qs_ref

    # SPMSM
    # def f_MTPA(T_e_ref_):
    #     torque_const = (3/2) * p_pairs * lambda_f
    #     i_ds_ref_r_ = 0.0
    #     i_qs_ref_r_ = T_e_ref_ / torque_const
    #     return i_ds_ref_r_, i_qs_ref_r_

    F_ref = K_p_v * (v_ref - v) + K_i_v * I_err_v
    T_e_ref = F_ref / k
    i_ds_ref_r, i_qs_ref_r = f_MTPA(T_e_ref)
    V_ds_r = K_p_i * (i_ds_ref_r - i_ds_r) + K_i_i * I_err_ds
    V_qs_r = K_p_i * (i_qs_ref_r - i_qs_r) + K_i_i * I_err_qs
    omega_m = (G / r_w) * v
    u_a = v - v_w * jnp.cos(psi - theta)
    v_a = -v_w * jnp.sin(psi - theta)
    w_F = -c_wx * u_a * jnp.abs(u_a)
    w_tau = -c_wy * v_a * jnp.abs(v_a)

    # 4. State Equations (dx/dt)
    dxdt = jnp.zeros(10)
    dxdt = dxdt.at[0].set((1/L_ds) * (V_ds_r - R_s*i_ds_r + omega_m*L_qs*i_qs_r))
    dxdt = dxdt.at[1].set((1/L_qs) * (V_qs_r - R_s*i_qs_r - omega_m*(L_ds*i_ds_r + lambda_f)))
    dxdt = dxdt.at[2].set(i_ds_ref_r - i_ds_r)
    dxdt = dxdt.at[3].set(i_qs_ref_r - i_qs_r)
    dxdt = dxdt.at[4].set(v * jnp.cos(theta))
    dxdt = dxdt.at[5].set(v * jnp.sin(theta))
    dxdt = dxdt.at[6].set(omega)
    dxdt = dxdt.at[7].set((1/m) * (k * (3/2) * p_pairs * lambda_f * i_qs_r - c_d*v + w_F))
    dxdt = dxdt.at[8].set((1/I) * (K_p_theta*(theta_ref - theta) - K_d_theta*omega - c_r*omega + w_tau))
    dxdt = dxdt.at[9].set(v_ref - v)
    
    return dxdt


def evalf_np(x, p_tuple, u):
    x_np = jnp.array(x[:, 0])
    u_np = jnp.array(u)
    return evalf(x_np, p_tuple, u_np)[:, None]


jacobian_func_raw = jax.jacobian(evalf, argnums=0)  # derivative w.r.t. x
compute_jacobian_jax = jax.jit(jacobian_func_raw)
