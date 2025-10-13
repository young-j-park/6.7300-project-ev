# test_singularity.py
import argparse
import numpy as np
import jax.numpy as jnp

from vehicle_model_jax import (
    compute_jacobian_jax,
    get_default_params,
    OUTPUT_POSITIONS,  # True면 10상태, False면 8상태
)

# ----- 상태 인덱스(모델 정의와 동일하게) -----
if OUTPUT_POSITIONS:
    N_STATES   = 10
    IDX_THETA  = 6
    IDX_V      = 7
    IDX_OMEGA  = 8
else:
    N_STATES   = 8
    IDX_THETA  = 4
    IDX_V      = 5
    IDX_OMEGA  = 6

# ----- p_tuple 패킹(모델 evalf와 동일 순서) -----
def pack_p_tuple(p):
    return (
        p['m'], p['I'], p['c_d'], p['c_r'],
        p['R_s'], p['L_ds'], p['L_qs'], p['lambda_f'], p['p_pairs'],
        p['G'], p['r_w'], p['eta_g'], p['k'],
        p['K_p_i'], p['K_i_i'], p['K_p_v'], p['K_i_v'], p['K_p_theta'], p['K_d_theta'],
        p['v_w'], p['psi'], p['c_wx'], p['c_wy'],
    )

def singularity_metrics(A: np.ndarray, rank_tol=None):
    # SVD 기반 메트릭
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = float(s[0])
    sigma_min = float(s[-1])
    cond = sigma_max / sigma_min if sigma_min > 0 else np.inf

    if rank_tol is None:
        rank_tol = np.finfo(float).eps * max(A.shape) * sigma_max
    rank = int(np.sum(s > rank_tol))

    det = np.linalg.det(A) if A.shape[0] <= 12 else np.nan

    return dict(sigma_min=sigma_min, cond=cond, rank=rank, det=det, tol=rank_tol)

def jacobian_at(x, u, p_tuple):
    J = compute_jacobian_jax(jnp.array(x), p_tuple, jnp.array(u))
    return np.asarray(J)

def check_point(x, u, p_tuple, sigma_min_thr=1e-8, cond_thr=1e10, verbose=True):
    A = jacobian_at(x, u, p_tuple)
    met = singularity_metrics(A)
    is_singular = (met["sigma_min"] < sigma_min_thr) or (met["cond"] > cond_thr) or (met["rank"] < A.shape[0])

    if verbose:
        print(f"\n--- Operating point ---")
        print(f"x = {np.array(x)}")
        print(f"u = {np.array(u)}")
        print(f"A shape = {A.shape}")
        print(f"sigma_min = {met['sigma_min']:.3e}, cond = {met['cond']:.3e}, "
              f"rank = {met['rank']} (tol={met['tol']:.1e}), det = {met['det']:.3e}")
        print("=> singular? ", is_singular)
    return is_singular, met

def scan_grid(p_tuple, v_grid, omega_grid, theta=0.0,
              sigma_min_thr=1e-8, cond_thr=1e10):
    bad = []
    for v in v_grid:
        for w in omega_grid:
            x = np.zeros(N_STATES)
            x[IDX_THETA] = theta
            x[IDX_V]     = v
            x[IDX_OMEGA] = w
            u = np.array([v, theta], dtype=float)  # v_ref=v, theta_ref=theta
            A = jacobian_at(x, u, p_tuple)
            met = singularity_metrics(A)
            if (met["sigma_min"] < sigma_min_thr) or (met["cond"] > cond_thr) or (met["rank"] < A.shape[0]):
                bad.append(((v, w), met))
    return bad

def scan_random(p_tuple, n=100, v_max=40.0, omega_max=3.0, sigma_min_thr=1e-8, cond_thr=1e10):
    bad = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        v = float(rng.uniform(0, v_max))
        w = float(rng.uniform(-omega_max, omega_max))
        th = float(rng.uniform(-np.pi, np.pi))
        x = np.zeros(N_STATES)
        x[IDX_THETA] = th
        x[IDX_V]     = v
        x[IDX_OMEGA] = w
        u = np.array([v, th], dtype=float)
        A = jacobian_at(x, u, p_tuple)
        met = singularity_metrics(A)
        if (met["sigma_min"] < sigma_min_thr) or (met["cond"] > cond_thr) or (met["rank"] < A.shape[0]):
            bad.append(((v, w, th), met))
    return bad

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser(description="Singularity test of df/dx using JAX Jacobian")
    ap.add_argument("--mode", choices=["point", "grid", "random"], default="point")
    ap.add_argument("--v", type=float, default=10.0, help="test speed (for mode=point)")
    ap.add_argument("--omega", type=float, default=0.0, help="test yaw rate (for mode=point)")
    ap.add_argument("--theta", type=float, default=0.0, help="test heading (for mode=point)")
    ap.add_argument("--sigma_min_thr", type=float, default=1e-8)
    ap.add_argument("--cond_thr", type=float, default=1e10)
    ap.add_argument("--nrand", type=int, default=100)
    args = ap.parse_args()

    p = get_default_params()
    p_tuple = pack_p_tuple(p)

    if args.mode == "point":
        x = np.zeros(N_STATES)
        x[IDX_THETA] = args.theta
        x[IDX_V]     = args.v
        x[IDX_OMEGA] = args.omega
        u = np.array([args.v, args.theta], dtype=float)  # v_ref, theta_ref
        check_point(x, u, p_tuple, args.sigma_min_thr, args.cond_thr, verbose=True)

    elif args.mode == "grid":
        v_grid     = np.linspace(0.0, 40.0, 9)       # 0..40 m/s
        omega_grid = np.linspace(-2.0, 2.0, 9)       # -2..2 rad/s
        bad = scan_grid(p_tuple, v_grid, omega_grid, theta=0.0,
                        sigma_min_thr=args.sigma_min_thr, cond_thr=args.cond_thr)
        print(f"\n[GRID] potentially singular/ill-conditioned points: {len(bad)} / {len(v_grid)*len(omega_grid)}")
        for (v, w), met in bad[:20]:
            print(f"  v={v:.2f}, omega={w:.2f} | sigma_min={met['sigma_min']:.2e}, cond={met['cond']:.2e}, rank={met['rank']}")

    else:  # random
        bad = scan_random(p_tuple, n=args.nrand,
                          sigma_min_thr=args.sigma_min_thr, cond_thr=args.cond_thr)
        print(f"\n[RANDOM] potentially singular/ill-conditioned points: {len(bad)} / {args.nrand}")
        for (v, w, th), met in bad[:20]:
            print(f"  v={v:.2f}, omega={w:.2f}, theta={th:.2f} | sigma_min={met['sigma_min']:.2e}, cond={met['cond']:.2e}")

if __name__ == "__main__":
    main()
