# test_evalf.py  (compatible with new drone_model_jax.py)
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from model.drone_model_jax import (
    evalf, get_default_params, pack_params, N_STATES, IDX
)

# --------------------------
# helpers using SAME symbols as model
# --------------------------
def sigma_F_from_speeds(p, wL, wR):
    """ΣF = kt * (wL^2 + wR^2)."""
    return p['kt'] * (wL**2 + wR**2)

def expected_translational_accel(p, x):
    """
    ay = (-ΣF*sin(theta) - Dt*v_y)/m
    az = ( ΣF*cos(theta) - m*g - Dt*v_z)/m
    """
    m  = p['m']
    Dt = p['Dt']
    g  = p['g']

    theta = float(x[IDX['theta']])
    v_y   = float(x[IDX['v_y']])
    v_z   = float(x[IDX['v_z']])
    wL    = float(x[IDX['wm_L']])
    wR    = float(x[IDX['wm_R']])

    SigmaF = sigma_F_from_speeds(p, wL, wR)

    ay = (-SigmaF * np.sin(theta) - Dt * v_y) / m
    az = ( SigmaF * np.cos(theta) - m * g - Dt * v_z) / m
    return ay, az

def main():
    # Do Manual Tests
    with open('./test_benchmarks/test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    for case in test_cases:
        f_gt = case['f']
        f_out = evalf(jnp.array(case['x']), tuple(case['p']), jnp.array(case['u']))
        assert np.allclose(f_gt, f_out, atol=1e-12)

    # params & tuple (must match pack order)
    p = get_default_params()
    p_tuple = pack_params(p)

    u_zero = jnp.array([0.0, 0.0], dtype=jnp.float64)

    # ---------- 1) steady “ground” state ----------
    x0 = jnp.zeros(N_STATES, dtype=jnp.float64)
    f0 = evalf(x0, p_tuple, u_zero)

    # kinematics
    assert np.allclose(float(f0[IDX['y']]),     float(x0[IDX['v_y']]), atol=1e-12)
    assert np.allclose(float(f0[IDX['z']]),     float(x0[IDX['v_z']]), atol=1e-12)
    assert np.allclose(float(f0[IDX['theta']]), float(x0[IDX['wd']]),  atol=1e-12)

    # translational dynamics
    ay_exp, az_exp = expected_translational_accel(p, x0)
    assert np.allclose(float(f0[IDX['v_y']]), ay_exp, atol=1e-12)
    assert np.allclose(float(f0[IDX['v_z']]), az_exp, atol=1e-12)

    print("OK: steady-state plant consistency")

    # ---------- 2) hover test (ΣF ≈ m*g, theta=0, v=0) ----------
    x_hover = jnp.zeros(N_STATES, dtype=jnp.float64)
    SigmaF_hover = p['m'] * p['g']
    w_hover = np.sqrt(SigmaF_hover / (2.0 * p['kt']))  # equal split
    x_hover = x_hover.at[IDX['wm_L']].set(w_hover)
    x_hover = x_hover.at[IDX['wm_R']].set(w_hover)

    f_hover = evalf(x_hover, p_tuple, u_zero)
    ay_exp, az_exp = expected_translational_accel(p, x_hover)
    assert np.allclose(float(f_hover[IDX['v_y']]), ay_exp, atol=1e-10)  # ~0
    assert np.allclose(float(f_hover[IDX['v_z']]), az_exp, atol=1e-10)  # ~0

    print("OK: hover thrust consistency")

    # ---------- 3) tilted (same ΣF, small theta) ----------
    x_tilt = x_hover.copy()
    theta_small = 5.0 * np.pi / 180.0
    x_tilt = x_tilt.at[IDX['theta']].set(theta_small)
    f_tilt = evalf(x_tilt, p_tuple, u_zero)
    ay_exp, az_exp = expected_translational_accel(p, x_tilt)
    assert np.allclose(float(f_tilt[IDX['v_y']]), ay_exp, atol=1e-10)
    assert np.allclose(float(f_tilt[IDX['v_z']]), az_exp, atol=1e-10)

    print("OK: tilted-thrust consistency")

    # ---------- 4) differential thrust → sign of dot(wd) ----------
    x_spin = jnp.zeros(N_STATES, dtype=jnp.float64)
    x_spin = x_spin.at[IDX['wm_L']].set(w_hover * 0.9)
    x_spin = x_spin.at[IDX['wm_R']].set(w_hover * 1.1)
    f_spin = evalf(x_spin, p_tuple, u_zero)
    # with wd=0 and F_R>F_L, torque term r(F_R-F_L) > 0 → dot(wd) > 0
    assert float(f_spin[IDX['wd']]) > 0.0

    print("OK: differential-thrust spin sign")

    # ---------- 5) nonzero references → finite outputs ----------
    x_ref = jnp.zeros(N_STATES, dtype=jnp.float64)
    u_ref = jnp.array([1.0, 1.0], dtype=jnp.float64)  # y_ref, z_ref
    f_ref = evalf(x_ref, p_tuple, u_ref)
    assert np.all(np.isfinite(np.array(f_ref)))

    print("OK: finite-output check under nonzero references")

    # ============================
    # 6) Yaw damping only (no ΔF)
    # ============================
    x = jnp.zeros(N_STATES, dtype=jnp.float64)
    x = x.at[IDX['wd']].set(1.0)  # nonzero spin
    f_yaw = evalf(x, p_tuple, u_zero)
    # With F_R=F_L (both zero), dot(wd) = -(Dr/Jd)*wd  < 0
    assert float(f_yaw[IDX['wd']]) < 0.0
    print("OK: yaw damping only gives negative angular accel")

    # ===========================================
    # 7) Symmetric motors → no net yaw torque
    # ===========================================
    x = jnp.zeros(N_STATES, dtype=jnp.float64)
    w_eq = 500.0
    x = x.at[IDX['wm_L']].set(w_eq)
    x = x.at[IDX['wm_R']].set(w_eq)
    f_sym = evalf(x, p_tuple, u_zero)
    # Equal thrusts => r(F_R - F_L)=0 ⇒ dot(wd) = -(Dr/Jd)*wd (here wd=0)
    assert np.isclose(float(f_sym[IDX['wd']]), 0.0, atol=1e-12)
    print("OK: symmetric motors produce no yaw acceleration at wd=0")

    # ==================================================================
    # 8) Translational invariance under swapping left/right motor speeds
    # ==================================================================
    # ay, az depend on ΣF only, so swapping L/R must keep (dot(v_y), dot(v_z)) identical.
    xA = jnp.zeros(N_STATES, dtype=jnp.float64)
    xB = jnp.zeros(N_STATES, dtype=jnp.float64)
    xA = xA.at[IDX['wm_L']].set(400.0)
    xA = xA.at[IDX['wm_R']].set(700.0)
    xB = xB.at[IDX['wm_L']].set(700.0)
    xB = xB.at[IDX['wm_R']].set(400.0)
    fA = evalf(xA, p_tuple, u_zero)
    fB = evalf(xB, p_tuple, u_zero)
    assert np.allclose(float(fA[IDX['v_y']]), float(fB[IDX['v_y']]), atol=1e-12)
    assert np.allclose(float(fA[IDX['v_z']]), float(fB[IDX['v_z']]), atol=1e-12)
    print("OK: swapping L/R keeps translational accelerations identical")

    # ==================================================
    # 9) Hover, then small downward v_z adds drag term
    # ==================================================
    x = jnp.zeros(N_STATES, dtype=jnp.float64)
    SigmaF_hover = p['m'] * p['g']
    w_hover = np.sqrt(SigmaF_hover / (2.0 * p['kt']))
    x = x.at[IDX['wm_L']].set(w_hover)
    x = x.at[IDX['wm_R']].set(w_hover)
    x = x.at[IDX['v_z']].set(-0.5)  # moving downward
    f = evalf(x, p_tuple, u_zero)
    # az = (ΣF*cosθ - mg - Dt*vz)/m; with vz<0, −Dt*vz is +, so az becomes slightly positive
    assert float(f[IDX['v_z']]) > -1e-6  # near 0 or slightly positive depending on Dt
    print("OK: downward velocity adds drag and lifts vertical accel")

    # ======================================
    # 10) Zero dampings Dt=Dr=0 (pure thrust)
    # ======================================
    p0 = get_default_params()
    p0['Dt'] = 0.0
    p0['Dr'] = 0.0
    f = evalf(
        jnp.array([
            # lamdsr_L, lamqsr_L, integ_id_L, integ_iq_L, wm_L, integ_w_L,
            0.0, 0.0, 0.0, 0.0, 600.0, 0.0,
            # lamdsr_R, lamqsr_R, integ_id_R, integ_iq_R, wm_R, integ_w_R,
            0.0, 0.0, 0.0, 0.0, 600.0, 0.0,
            # theta, wd, y, z, v_y, v_z
            np.deg2rad(10.0), 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=jnp.float64),
        pack_params(p0),
        u_zero
    )
    theta = np.deg2rad(10.0)
    SigmaF = p0['kt'] * (600.0**2 + 600.0**2)
    ay_exp = (-SigmaF*np.sin(theta))/p0['m']
    az_exp = ( SigmaF*np.cos(theta) - p0['m']*p0['g'])/p0['m']
    assert np.allclose(float(f[IDX['v_y']]), ay_exp, atol=1e-10)
    assert np.allclose(float(f[IDX['v_z']]), az_exp, atol=1e-10)
    print("OK: zero damping matches pure thrust projections exactly")

    # ==========================================================
    # 11) Controller sign: y_ref>0 at θ=0 ⇒ d(integ_w_R) - d(integ_w_L) < 0
    #     (θ_ref < 0 ⇒ ΔF_ref < 0 ⇒ wR_ref < wL_ref)
    # ==========================================================
    x = jnp.zeros(N_STATES, dtype=jnp.float64)  # θ=0, wd=0, v=0
    u = jnp.array([1.0, 0.0], dtype=jnp.float64)  # y_ref positive, z_ref zero
    f = evalf(x, p_tuple, u)
    dI_wR = float(f[IDX['integ_w_R']])
    dI_wL = float(f[IDX['integ_w_L']])
    assert dI_wR - dI_wL < 0.0
    print("OK: controller makes right motor ref smaller for +y_ref at theta=0")

    # ===================================================
    # 12) Current PI integrator sign (q-axis, left motor)
    #     Positive z_ref ⇒ positive thrust ⇒ i_qs_ref > 0 ⇒ d(integ_iq_L) > 0
    # ===================================================
    x = jnp.zeros(N_STATES, dtype=jnp.float64)
    u = jnp.array([0.0, 1.0], dtype=jnp.float64)
    f = evalf(x, p_tuple, u)
    assert float(f[IDX['integ_iq_L']]) > 0.0
    print("OK: q-axis current integrator increases with positive torque ref")

    # =====================================================
    # 13) Torque → rotor accel consistency: Jm * dot(wm_L) = T_e_L (wm_L=0)
    #     Set (λd, λq) to realize desired (id, iq).
    # =====================================================
    pM = get_default_params()
    Ld, Lq, lamf, pp, Jm, Bm = pM['Lds'], pM['Lqs'], pM['lamf'], pM['pp'], pM['Jm'], pM['Bm']
    id_set, iq_set = 0.5, 2.0
    lamd = lamf + Ld*id_set
    lamq = Lq*iq_set
    x = jnp.zeros(N_STATES, dtype=jnp.float64)
    x = x.at[IDX['lamdsr_L']].set(lamd)
    x = x.at[IDX['lamqsr_L']].set(lamq)
    # keep wm_L = 0 for simplicity
    f = evalf(x, pack_params(pM), u_zero)
    TeL = 1.5 * pp * (lamf + (Ld - Lq) * id_set) * iq_set
    # Jm * d(wm_L) = TeL - Bm*wm_L  (wm_L=0)
    assert np.allclose(float(f[IDX['wm_L']]) * Jm, TeL, atol=1e-10)
    print("OK: motor torque maps to rotor angular acceleration")
    
    print("All evalf tests passed")

if __name__ == "__main__":
    main()
