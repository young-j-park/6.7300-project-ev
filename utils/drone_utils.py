import numpy as np
import jax.numpy as jnp

from model.drone_model_jax import N_STATES, IDX


def build_hover_state_and_input(p_dict: dict):
    """
    Build hover equilibrium: theta=0, wd=0, y=z=0, v_y=v_z=0,
    wm = sqrt((m*g/2)/kt), electrical: id=0, iq=0, integrators=0.
    Returns: x (N_STATES,), u (2,)
    """
    x = np.zeros(N_STATES, dtype=np.float64)
    m = float(p_dict['m'])
    g = float(p_dict['g'])
    kt = float(p_dict['kt'])
    lamf = float(p_dict['lamf'])
    
    wm_hover = float(np.sqrt((m * g) / (2.0 * kt)))

    x[IDX['lamdsr_L']] = lamf
    x[IDX['lamqsr_L']] = 0.0
    x[IDX['lamdsr_R']] = lamf
    x[IDX['lamqsr_R']] = 0.0

    x[IDX['integ_id_L']] = 0.0
    x[IDX['integ_iq_L']] = 0.0
    x[IDX['integ_id_R']] = 0.0
    x[IDX['integ_iq_R']] = 0.0
    x[IDX['integ_w_L']] = 0.0
    x[IDX['integ_w_R']] = 0.0

    x[IDX['wm_L']] = wm_hover
    x[IDX['wm_R']] = wm_hover

    x[IDX['theta']] = 0.0
    x[IDX['wd']] = 0.0
    x[IDX['y']] = 0.0
    x[IDX['z']] = 0.0
    x[IDX['v_y']] = 0.0
    x[IDX['v_z']] = 0.0

    u = np.array([0.0, 0.0], dtype=np.float64)
    return jnp.array(x), jnp.array(u)