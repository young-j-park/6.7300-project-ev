# drone_model_matched.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ----------------------------------------------------------------------
# State layout (MATLAB-equivalent internal states)
#  Left motor : lamdsr_L, lamqsr_L, integ_id_L, integ_iq_L, wm_L, integ_w_L
#  Right motor: lamdsr_R, lamqsr_R, integ_id_R, integ_iq_R, wm_R, integ_w_R
#  Body       : theta, wd, y, z, v_y, v_z
# Total = 18
# ----------------------------------------------------------------------
N_STATES = 18
IDX = {
    'lamdsr_L': 0, 'lamqsr_L': 1, 'integ_id_L': 2, 'integ_iq_L': 3, 'wm_L': 4, 'integ_w_L': 5,
    'lamdsr_R': 6, 'lamqsr_R': 7, 'integ_id_R': 8, 'integ_iq_R': 9, 'wm_R': 10,'integ_w_R': 11,
    'theta': 12, 'wd': 13, 'y': 14, 'z': 15, 'v_y': 16, 'v_z': 17,
}

# -------------------------------
# Parameter pack (MATLAB names/values)
# -------------------------------
PACK_ORDER = [
    # Motor (shared) == param_m
    'Rs','Lds','Lqs','lamf','Jm','Bm','pp',
    # Rotation PI == param_rot
    'Kp_rot','Ki_rot','TeMax',
    # Current PI (dq) == param_cur
    'Kpd','Kid','Kpq','Kiq',
    # Drone params == D.param
    'Jd','r','kt','Dr','m','Dt','g',
    # Outer-loop ctrl (== ctrl)
    'Kp_pos_y','Ki_pos_y','Kd_pos_y',
    'Kp_pos_z','Ki_pos_z','Kd_pos_z',
    'Ay_max','Az_max',
    'Kp_theta','Kd_theta',
    'wm_max','F_sum_max','F_diff_max',
]

def pack_params(p: dict):
    return tuple(p[k] for k in PACK_ORDER)

def unpack_params(p_tuple):
    return {k: p_tuple[i] for i,k in enumerate(PACK_ORDER)}
    
def get_default_params():
    """
    Default parameters.

    Outer-loop position & attitude gains are computed from physical
    parameters and desired 2nd-order closed-loop dynamics.

    Current-loop PI gains are chosen so that the closed-loop transfer
    function becomes exactly:

        G_cl(s) = wbw / (s + wbw)

    for each axis, where wbw is the desired current-loop bandwidth.
    For plant G(s) = 1 / (L s + R), this is achieved by

        Kp = L * wbw
        Ki = R * wbw
    """

    # --- Physical parameters ---
    m  = 1.2
    g  = 9.81
    Jd = 0.05
    r  = 0.20
    kt = 5e-6
    Dr = 1e-3
    Dt = 0.05

    Rs  = 0.05
    Lds = 8e-3
    Lqs = 10e-3

    # =========================================================
    # 1) Design targets for outer-loop dynamics (same as before)
    # =========================================================
    # y-position loop
    wn_y    = 1.0        # [rad/s]
    zeta_y  = 1.0

    # z-position loop
    wn_z    = 4.0 / 3.0  # [rad/s]
    zeta_z  = 1.0

    # theta (pitch attitude) loop
    wn_th   = 8.0        # [rad/s]
    zeta_th = 1.0

    # --- Position & attitude gains ---
    Kp_pos_y = m * wn_y**2
    Kd_pos_y = 2.0 * m * zeta_y * wn_y - Dt
    Ki_pos_y = 0.0

    Kp_pos_z = m * wn_z**2
    Kd_pos_z = 2.0 * m * zeta_z * wn_z - Dt
    Ki_pos_z = 0.0

    Kp_theta = Jd * wn_th**2
    Kd_theta = 2.0 * Jd * zeta_th * wn_th - Dr

    # =========================================================
    # 2) Design targets for current-loop dynamics (1st-order LPF)
    # =========================================================
    # Desired current-loop bandwidths (tunable knobs)
    wbw_d = 100.0   # d-axis current loop bandwidth [rad/s]
    wbw_q = 100.0   # q-axis current loop bandwidth [rad/s]

    # From G_cl(s) = wbw / (s + wbw)  =>  Kp = L * wbw, Ki = R * wbw
    Kpd = Lds * wbw_d
    Kid = Rs  * wbw_d

    Kpq = Lqs * wbw_q
    Kiq = Rs  * wbw_q

    return {
        # Motor / electrical parameters (param_m)
        'Rs': Rs,
        'Lds': Lds,
        'Lqs': Lqs,
        'lamf': 0.04,
        'Jm': 1e-3,
        'Bm': 1e-4,
        'pp': 4,

        # Rotor speed PI controller (param_rot)
        'Kp_rot': 0.07,
        'Ki_rot': 0.3,
        'TeMax': 50.0,

        # dq current PI controller (param_cur)
        'Kpd': Kpd,
        'Kid': Kid,
        'Kpq': Kpq,
        'Kiq': Kiq,

        # Drone rigid-body parameters (D.param)
        'Jd': Jd,
        'r': r,
        'kt': kt,
        'Dr': Dr,
        'm': m,
        'Dt': Dt,
        'g': g,

        # Outer-loop position & attitude control gains (ctrl)
        'Kp_pos_y': Kp_pos_y,
        'Ki_pos_y': Ki_pos_y,
        'Kd_pos_y': Kd_pos_y,

        'Kp_pos_z': Kp_pos_z,
        'Ki_pos_z': Ki_pos_z,
        'Kd_pos_z': Kd_pos_z,

        'Ay_max': 4.0,
        'Az_max': 4.0,

        'Kp_theta': Kp_theta,
        'Kd_theta': Kd_theta,

        # Motor speed and thrust saturation limits
        'wm_max': 2200.0,
        'F_sum_max': 3.0 * m * g,
        'F_diff_max': 3.0 * m * g,
    }


# -------------------------------
# MATLAB-equivalent blocks (identical structure, continuous-time form)
# -------------------------------

def _controller_rotation(pp, Kp, Ki, TeMax, wm_ref, wm, integ_w):
    """
    Speed PI controller → torque reference.
    - Integration is handled outside the function (ODE: d(integ_w)/dt = wm_ref - wm)
    - Only torque saturation is applied (identical to MATLAB)
    """
    e = wm_ref - wm
    Te_cmd = pp * (Kp * e + Ki * integ_w)
    Te = jnp.clip(Te_cmd, -TeMax, TeMax)
    return Te, e  # 'e' is used as d(integ_w)/dt

def _MTPA(Ld, Lq, lamf, pp, Te):
    # == MTPA.m equivalent
    k = 1.5 * pp
    D = Lq - Ld
    tau = Te / k
    def spmsm(_):
        id_ref = 0.0
        iq_ref = jnp.where(jnp.isclose(lamf, 0.0), 0.0, Te / (k * lamf))
        return id_ref, iq_ref
    def ipmsm(_):
        # radic = jnp.maximum(lamf**2 + 8.0 * D * tau, 0.0)
        radic = jnp.maximum(lamf**2 + 8.0 * D * tau, 1e-12)
        id_ref = (lamf - jnp.sqrt(radic)) / (4.0 * D)
        denom = lamf - D * id_ref
        denom_safe = jnp.where(jnp.abs(denom) < 1e-9, 1e-9, denom)
        iq_ref = tau / denom_safe
        # iq_ref = jnp.where(jnp.isclose(denom, 0.0), 0.0, tau / denom)
        return id_ref, iq_ref
    return jax.lax.cond(jnp.abs(D) < 1e-12, spmsm, ipmsm, Te)

def _controller_motor(Rs, Ld, Lq, lamf, pp, Kpd, Kid, Kpq, Kiq,
                      id_ref, iq_ref, lamd, lamq, wm,
                      integ_d, integ_q):
    """
    Current PI + decoupling + feedforward terms.
    - No integrator update (handled externally in ODE)
    - Computes only v_d, v_q
    """
    # Measured currents
    id_mea = (lamd - lamf) / Ld
    iq_mea = lamq / Lq
    we = wm * pp
    # Errors
    ed = id_ref - id_mea
    eq = iq_ref - iq_mea
    # PI controller (integral terms use current integ_*)
    u_d = Kpd * ed + Kid * integ_d
    u_q = Kpq * eq + Kiq * integ_q
    # Decoupling + feedforward
    ff_vdsr = - we * lamq
    ff_vqsr = + we * lamd
    # ff_vdsr = Rs * id_mea - we * lamq
    # ff_vqsr = Rs * iq_mea + we * lamd
    vdsr = u_d + ff_vdsr
    vqsr = u_q + ff_vqsr
    return vdsr, vqsr, ed, eq, id_mea, iq_mea

def _dynamics_motor(Rs, Ld, Lq, lamf, Jm, Bm, pp,
                    vdsr, vqsr, lamd, lamq, wm):
    """
    Continuous-time equivalent of Dynamics_Motor.m
    """
    we = wm * pp
    # Currents
    idsr = (lamd - lamf) / Ld
    iqsr = lamq / Lq
    # Torque
    Te = 1.5 * pp * (lamf * iqsr + (Ld - Lq) * idsr * iqsr)
    # Flux derivatives
    d_lamd = vdsr - Rs * idsr + we * lamq
    d_lamq = vqsr - Rs * iqsr - we * lamd
    # Mechanical equation
    d_wm   = (Te - Bm * wm) / Jm
    return d_lamd, d_lamq, d_wm, Te, idsr, iqsr

def _controller_drone(ctrl, y_ref, z_ref, y, z, vy, vz, theta, wd):
    """
    Equivalent of MATLAB Controller_Drone:
    (y,z) → (Ay,Az) limiting → (Fy,Fz) → theta_ref, tau_ref →
    F_sum / F_diff limiting → (WL_ref, WR_ref)
    """
    Ay = jnp.clip(ctrl['Kp_pos_y'] * (y_ref - y) - ctrl['Kd_pos_y'] * vy,
                  -ctrl['Ay_max'], ctrl['Ay_max'])
    Az = jnp.clip(ctrl['Kp_pos_z'] * (z_ref - z) - ctrl['Kd_pos_z'] * vz,
                  -ctrl['Az_max'], ctrl['Az_max'])

    Fy_ref = -(ctrl['m'] * Ay + ctrl['Dt'] * vy)
    Fz_ref =  (ctrl['m'] * (Az + ctrl['g']) + ctrl['Dt'] * vz)

    theta_ref = jnp.arctan2(Fy_ref, Fz_ref)
    tau_ref   = ctrl['Kp_theta'] * (theta_ref - theta) - ctrl['Kd_theta'] * wd

    dF = jnp.clip(tau_ref / ctrl['r'], -ctrl['F_diff_max'], ctrl['F_diff_max'])
    sF = jnp.clip(jnp.sqrt(Fy_ref**2 + Fz_ref**2), 0.0, ctrl['F_sum_max'])

    FL_ref = jnp.maximum(0.0, 0.5 * (sF - dF))
    FR_ref = jnp.maximum(0.0, 0.5 * (sF + dF))

    WL_ref = jnp.minimum(jnp.sqrt(FL_ref / jnp.maximum(ctrl['kt'], 1e-12)), ctrl['wm_max'])
    WR_ref = jnp.minimum(jnp.sqrt(FR_ref / jnp.maximum(ctrl['kt'], 1e-12)), ctrl['wm_max'])

    return theta_ref, sF, WL_ref, WR_ref

def _dynamics_drone(Jd, r, kt, Dr, m, Dt, g, wmL, wmR, theta, wd, y, z, vy, vz):
    FL = kt * wmL**2
    FR = kt * wmR**2
    FT = FL + FR

    ddtheta = (r * (FR - FL) - Dr * wd) / Jd
    dwd = ddtheta
    dtheta = wd

    ay = (-FT * jnp.sin(theta) - Dt * vy) / m
    az = ( FT * jnp.cos(theta) - m * g - Dt * vz) / m

    dvy = ay
    dvz = az
    dy  = vy
    dz  = vz
    return dtheta, dwd, dy, dz, dvy, dvz

# -------------------------------
# evalf: x' = f(x, p, u),  u = [y_ref, z_ref]
# -------------------------------
@jax.jit
def evalf(x, p_tuple, u):
    p = unpack_params(p_tuple)

    # Unpack parameters
    Rs,Lds,Lqs,lamf,Jm,Bm,pp = p['Rs'],p['Lds'],p['Lqs'],p['lamf'],p['Jm'],p['Bm'],p['pp']
    Kp_rot,Ki_rot,TeMax      = p['Kp_rot'],p['Ki_rot'],p['TeMax']
    Kpd,Kid,Kpq,Kiq          = p['Kpd'],p['Kid'],p['Kpq'],p['Kiq']
    Jd,r,kt,Dr,m,Dt,g        = p['Jd'],p['r'],p['kt'],p['Dr'],p['m'],p['Dt'],p['g']

    ctrl = {
        'Kp_pos_y': p['Kp_pos_y'], 'Ki_pos_y': p['Ki_pos_y'], 'Kd_pos_y': p['Kd_pos_y'],
        'Kp_pos_z': p['Kp_pos_z'], 'Ki_pos_z': p['Ki_pos_z'], 'Kd_pos_z': p['Kd_pos_z'],
        'Ay_max': p['Ay_max'], 'Az_max': p['Az_max'],
        'Kp_theta': p['Kp_theta'], 'Kd_theta': p['Kd_theta'],
        'm': m, 'g': g, 'Dt': Dt, 'kt': kt, 'r': r,
        'wm_max': p['wm_max'], 'F_sum_max': p['F_sum_max'], 'F_diff_max': p['F_diff_max'],
    }

    # Unpack states
    (lamdL, lamqL, integ_id_L, integ_iq_L, wmL, integ_w_L,
     lamdR, lamqR, integ_id_R, integ_iq_R, wmR, integ_w_R,
     theta, wd, y, z, vy, vz) = x

    y_ref, z_ref = u

    # ===== Outer controller (y,z -> theta_ref & WL/WR refs) =====
    theta_ref, Fsum_ref, WL_ref, WR_ref = _controller_drone(
        ctrl, y_ref, z_ref, y, z, vy, vz, theta, wd
    )

    # ===== Speed PI -> torque references (per motor) =====
    Te_L_ref, e_w_L = _controller_rotation(pp, Kp_rot, Ki_rot, TeMax, WL_ref, wmL, integ_w_L)
    Te_R_ref, e_w_R = _controller_rotation(pp, Kp_rot, Ki_rot, TeMax, WR_ref, wmR, integ_w_R)
    d_integ_w_L = e_w_L
    d_integ_w_R = e_w_R

    # ===== MTPA (torque -> current references) =====
    idL_ref, iqL_ref = _MTPA(Lds, Lqs, lamf, pp, Te_L_ref)
    idR_ref, iqR_ref = _MTPA(Lds, Lqs, lamf, pp, Te_R_ref)

    # ===== Current PI (with feedforward) -> dq voltages =====
    vdsL, vqL, e_d_L, e_q_L, idL_mea, iqL_mea = _controller_motor(
        Rs,Lds,Lqs,lamf,pp, Kpd,Kid,Kpq,Kiq,
        idL_ref,iqL_ref, lamdL,lamqL, wmL, integ_id_L, integ_iq_L
    )
    vdsR, vqR, e_d_R, e_q_R, idR_mea, iqR_mea = _controller_motor(
        Rs,Lds,Lqs,lamf,pp, Kpd,Kid,Kpq,Kiq,
        idR_ref,iqR_ref, lamdR,lamqR, wmR, integ_id_R, integ_iq_R
    )
    d_integ_id_L = e_d_L
    d_integ_iq_L = e_q_L
    d_integ_id_R = e_d_R
    d_integ_iq_R = e_q_R

    # ===== Motor plant dynamics =====
    d_lamdL, d_lamqL, d_wmL, TeL, _, _ = _dynamics_motor(Rs,Lds,Lqs,lamf,Jm,Bm,pp,
                                                         vdsL,vqL, lamdL,lamqL, wmL)
    d_lamdR, d_lamqR, d_wmR, TeR, _, _ = _dynamics_motor(Rs,Lds,Lqs,lamf,Jm,Bm,pp,
                                                         vdsR,vqR, lamdR,lamqR, wmR)

    # ===== Drone rigid-body dynamics =====
    dtheta, dwd, dy, dz, dvy, dvz = _dynamics_drone(Jd, r, kt, Dr, m, Dt, g,
                                                     wmL, wmR, theta, wd, y, z, vy, vz)

    # ===== Assemble derivatives =====
    dxdt = jnp.array([
        d_lamdL, d_lamqL, d_integ_id_L, d_integ_iq_L, d_wmL, d_integ_w_L,
        d_lamdR, d_lamqR, d_integ_id_R, d_integ_iq_R, d_wmR, d_integ_w_R,
        dtheta,  dwd,     dy,           dz,           dvy,   dvz
    ])
    return dxdt

# === Jacobian (with respect to state x) ===
jacobian_func_raw = jax.jacobian(evalf, argnums=0)
compute_jacobian_jax = jax.jit(jacobian_func_raw)

# Numpy wrapper (N,1) -> (N,1)
def evalf_np(x, p_tuple, u):
    import numpy as np
    x_np = np.asarray(x[:,0], dtype=np.float64)
    u_np = np.asarray(u, dtype=np.float64)
    dx = evalf(jnp.array(x_np), p_tuple, jnp.array(u_np))
    return np.asarray(dx, dtype=np.float64)[:,None]
