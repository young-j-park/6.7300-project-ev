import matplotlib.pyplot as plt

from model.drone_model_jax import N_STATES, IDX


def visualizeNetwork(x, p, t=None, U=None, xlim=None, ylim=None, legend=False):    
    # Extract number of drones and positions
    N = len(x) // N_STATES
    
    y_positions = []
    z_positions = []
    for i in range(N):
        base = i * N_STATES
        y_pos = x[base + IDX['y']]
        z_pos = x[base + IDX['z']]
        y_positions.append(y_pos)
        z_positions.append(z_pos)
    
    # Visualization
    plt.gcf().clear()
    
    # Plot targets
    if U is not None:
        for i in range(N):
            plt.plot(U[i][0], U[i][1], 'x', c='red', 
                     markersize=10, alpha=0.7, label=f'Target {i}')
            plt.text(U[i][0] + 0.05, U[i][1] + 0.05, 
                     f'{i}', fontsize=9)
    
    # Plot individual drones
    for i in range(N):
        plt.plot(y_positions[i], z_positions[i], 'o', c='blue', 
                 markersize=10, alpha=0.7, label=f'Drone {i}')
        plt.text(y_positions[i] + 0.05, z_positions[i] + 0.05, 
                 f'{i}', fontsize=9)
    
    title = f'Drone Swarm Visualization ({N} Drones)'
    if t is not None:
        title += f' - Time: {t:.2f} s'
    plt.title(title)
    
    plt.xlabel('Y Position (meters)')
    plt.ylabel('Z Position (meters)')
    if legend:
        plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # plt.pause(0.01)
