import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# create a log directory by date
from datetime import datetime
today = datetime.today().strftime('%Y%m%d')
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)

# Create 5x5x5 velocity field
velocity_field = np.random.randn(5, 5, 5, 3)


def draw_grid_v(velocity_field, log_dir, figname="velocity_field.png"):
    """
    Draws a 3D grid with velocity vectors.
    Args:
        velocity_field (np.ndarray): A 4D array representing the velocity field.
    """
    # Check if the velocity field has the correct shape
    if velocity_field.ndim != 4 or velocity_field.shape[3] != 3:
        raise ValueError("Velocity field must be a 4D array with shape (x, y, z, 3).")

    # Generate grid
    size = velocity_field.shape[0]
    x = np.arange(size)
    y = np.arange(size)
    z = np.arange(size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # Extract velocity components
    U = velocity_field[:, :, :, 0]
    V = velocity_field[:, :, :, 1]
    W = velocity_field[:, :, :, 2]

    # Flatten for plotting
    X = grid_x.flatten()
    Y = grid_y.flatten()
    Z = grid_z.flatten()
    U = U.flatten()
    V = V.flatten()
    W = W.flatten()

    # Start figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot velocity vectors
    ax.quiver(X, Y, Z, U, V, W, length=0.3, normalize=True, color='blue')
    # Plot grid nodes
    ax.scatter(X, Y, Z, color='black', s=2)

    # Plot grid edges (wireframe cube for each cell)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if i < size - 1:
                    ax.plot([i, i+1], [j, j], [k, k], color='lightgray', linewidth=0.8)
                if j < size - 1:
                    ax.plot([i, i], [j, j+1], [k, k], color='lightgray', linewidth=0.8)
                if k < size - 1:
                    ax.plot([i, i], [j, j], [k, k+1], color='lightgray', linewidth=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Grid with Velocity Vectors')
    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.set_zlim([0, size])

    # Remove background grid panes and ticks
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)  # Remove major gridlines
    ax.set_xticks([])  # Optionally remove ticks
    ax.set_yticks([])
    ax.set_zticks([])

    plt.tight_layout()
    plt.savefig(f"{log_dir}/{figname}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    draw_grid_v(velocity_field, log_dir)
    print(f"Velocity field visualized and saved to {log_dir}/velocity_field.png")
