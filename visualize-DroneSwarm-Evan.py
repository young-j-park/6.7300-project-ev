import numpy as np
import matplotlib.pyplot as plt
import time

# Import your custom modules
from visualize import DroneVisualizer
from model.drone_model_jax import get_default_params, pack_params
from utils.swarm_utils import assemble_swarm, swarm_evalf


# --- SHAPE GENERATION LOGIC ---

def create_circle(cy, cz, scale, n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = cy + scale * np.cos(theta)
    z = cz + scale * np.sin(theta)
    return np.column_stack((y, z))


def create_heart(cy, cz, scale, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    s = scale / 16.0
    y = cy + s * 16 * np.sin(t) ** 3
    z = cz + s * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return np.column_stack((y, z))


def create_diamond(cy, cz, scale, n):
    n_side = max(1, n // 4)
    pts = []
    pts.append(np.column_stack((np.linspace(0, scale, n_side), np.linspace(scale, 0, n_side))))
    pts.append(np.column_stack((np.linspace(scale, 0, n_side), np.linspace(0, -scale, n_side))))
    pts.append(np.column_stack((np.linspace(0, -scale, n_side), np.linspace(-scale, 0, n_side))))
    pts.append(np.column_stack((np.linspace(-scale, 0, n_side), np.linspace(0, scale, n_side))))
    arr = np.vstack(pts)
    arr[:, 0] += cy
    arr[:, 1] += cz
    return arr


def create_spiral(cy, cz, scale, n):
    theta = np.linspace(0, 6 * np.pi, n)
    r = np.linspace(0, scale, n)
    y = cy + r * np.cos(theta)
    z = cz + r * np.sin(theta)
    return np.column_stack((y, z))


def create_star(cy, cz, scale, n):
    points = []
    n_points = 5
    R = scale
    r = scale * 0.4
    for i in range(n_points * 2 + 1):
        radius = R if i % 2 == 0 else r
        angle = i * np.pi / n_points + np.pi / 2
        points.append([radius * np.cos(angle), radius * np.sin(angle)])
    points = np.array(points)
    final_pts = []
    points_per_segment = max(1, n // (n_points * 2))
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        ts = np.linspace(0, 1, points_per_segment, endpoint=False)
        for t in ts:
            final_pts.append(p1 + t * (p2 - p1))
    arr = np.array(final_pts)
    arr[:, 0] += cy
    arr[:, 1] += cz
    return arr


def create_random_box(cy, cz, scale, n):
    y = cy + (np.random.rand(n) - 0.5) * 2 * scale
    z = cz + (np.random.rand(n) - 0.5) * 2 * scale
    return np.column_stack((y, z))


def create_line(cy, cz, scale, n):
    y = np.linspace(cy - scale, cy + scale, n)
    z = np.full(n, cz)
    return np.column_stack((y, z))


def create_whale(cy, cz, scale, n, face_right=True):
    """
    Creates a whale shape.
    face_right=True  -> Head on Right, Tail on Left
    face_right=False -> Head on Left, Tail on Right
    """
    n_body = int(n * 0.7)
    n_tail = n - n_body

    # 1. BODY (Teardrop)
    t = np.linspace(0, 2 * np.pi, n_body)
    # Basic teardrop centered at 0
    y_body = 1.8 * scale * np.cos(t)
    z_body = scale * np.sin(t) * (0.8 + 0.4 * np.cos(t))

    # 2. TAIL (Flukes)
    u = np.linspace(-np.pi / 2, np.pi / 2, n_tail)
    tail_offset = -1.6 * scale
    y_tail = tail_offset - (0.6 * scale * np.cos(u))
    z_tail = (1.2 * scale * np.sin(u)) + (0.3 * scale * np.cos(u))

    # Combine body parts (centered at 0,0 relative to shape)
    y_local = np.concatenate([y_body, y_tail])
    z_local = np.concatenate([z_body, z_tail])

    # FLIP HORIZONTALLY if needed
    if not face_right:
        y_local = -y_local

    # Shift to center
    y = y_local + cy
    z = z_local + cz

    return np.column_stack((y, z))

if __name__ == "__main__":
    all_targets = []

    # ------------------------------------------
    # Change to face_right to True if you want the left tail orientation.
    print("Adding a Whale shape (Flipped)...")
    all_targets.append(create_whale(cy=0, cz=3, scale=1.5, n=60, face_right=False))

    # A Heart to the right
    # all_targets.append(create_heart(cy=5, cz=3, scale=1.2, n=40))

    # A Star to the left
    # all_targets.append(create_star(cy=-5, cz=3, scale=1.2, n=40))

    # A circle
    all_targets.append(create_circle(cy=-5, cz=3, scale = 1.2, n = 50))
    # -------------------------------------------

    if not all_targets:
        print("No shapes defined! Adding a default circle.")
        all_targets.append(create_circle(0, 2, 1.5, 30))

    targets = np.concatenate(all_targets, 0)
    N = len(targets)
    print(f"Simulation started with {N} drones.")

    # PHYSICS & VISUALIZATION
    p = get_default_params()
    p_tuple = pack_params(p)
    X0_flat, U = assemble_swarm(targets)

    dt = 0.5e-3
    T_final = 10
    frame_skip = 50

    x = X0_flat.copy()
    t = 0.0
    step = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    viz = DroneVisualizer(ax, x, U, xlim=[-6, 6], ylim=[-2, 8])

    try:
        while t < T_final:
            dxdt = swarm_evalf(x, p_tuple, U)
            x = x + dxdt * dt
            t += dt
            step += 1

            if step % frame_skip == 0:
                viz.update(x, t)
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    print("Simulation finished.")
    plt.ioff()
    plt.show()