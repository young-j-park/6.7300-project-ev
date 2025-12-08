import matplotlib.pyplot as plt
import numpy as np
from model.drone_model_jax import N_STATES, IDX


class DroneVisualizer:
    def __init__(self, ax, x_0, U=None, xlim=None, ylim=None):
        """
        Initializes the plot elements on the provided Axes (ax).
        """
        self.ax = ax
        self.N = len(x_0) // N_STATES

        # Setup Axes
        if xlim: self.ax.set_xlim(xlim)
        if ylim: self.ax.set_ylim(ylim)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Y Position (meters)')
        self.ax.set_ylabel('Z Position (meters)')

        # 1. Plot Targets (Static)
        if U is not None:
            U_arr = np.array(U)
            self.ax.scatter(U_arr[:, 0], U_arr[:, 1], c='red', marker='x',
                            s=50, alpha=0.7, label='Targets')
            for i, u in enumerate(U):
                self.ax.text(u[0] + 0.05, u[1] + 0.05, f'{i}', fontsize=8, color='red')

        # 2. Plot Drones (Dynamic)
        # We draw them initially at x_0
        ys, zs = self._get_positions(x_0)
        self.drone_scatter = self.ax.scatter(ys, zs, c='blue', marker='o',
                                             s=50, alpha=0.7, label='Drones')

        # Text labels
        self.drone_texts = [self.ax.text(ys[i], zs[i], f'{i}', fontsize=9, color='blue')
                            for i in range(self.N)]

        self.ax.legend(loc='upper right')
        self.title = self.ax.set_title("Initializing...")

    def _get_positions(self, x):
        """Helper to extract Y and Z coordinates."""
        x_reshaped = x.reshape(self.N, N_STATES)
        return x_reshaped[:, IDX['y']], x_reshaped[:, IDX['z']]

    def update(self, x, t):
        """
        Updates the positions of existing elements.
        """
        ys, zs = self._get_positions(x)

        # Update Scatter positions (efficiently)
        self.drone_scatter.set_offsets(np.c_[ys, zs])

        # Update Text positions
        for i, txt in enumerate(self.drone_texts):
            txt.set_position((ys[i] + 0.05, zs[i] + 0.05))

        # Update Title
        if t is not None:
            self.title.set_text(f'Drone Swarm ({self.N} Drones) - Time: {t:.2f} s')