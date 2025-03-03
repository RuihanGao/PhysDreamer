import warp as wp
import torch
import numpy as np
from mpm_solver_diff import MPMWARPDiff
from mpm_data_structure import MPMStateStruct, MPMModelStruct
from warp_utils import MyTape, from_torch_safe
import pdb

@wp.kernel
def compute_loss_kernel(particle_x: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, particle_x[tid][0] + particle_x[tid][1] + particle_x[tid][2])


def finite_difference_grad(func, x, epsilon=1e-4):
    """Computes numerical gradients using finite differences."""
    grad = np.zeros_like(x)

    for i in range(x.size):
        x_pos = x.copy()
        x_neg = x.copy()

        x_pos.flat[i] += epsilon
        x_neg.flat[i] -= epsilon

        loss_pos = func(x_pos)
        loss_neg = func(x_neg)

        grad.flat[i] = (loss_pos - loss_neg) / (2 * epsilon)

    return grad
    
# --- COMPUTE NUMERICAL GRADIENTS ---
def loss_fn(particle_x_numpy):
    """Helper function to compute loss given input particle positions (for finite differences)."""
    mpm_state.particle_x = wp.from_numpy(particle_x_numpy, dtype=wp.vec3, device=device, requires_grad=True)
    next_state.particle_x = wp.clone(mpm_state.particle_x, requires_grad=True)

    loss_array = torch.zeros(1, dtype=torch.float32, device=device)
    loss_array = wp.from_torch(loss_array, requires_grad=True)

    with MyTape() as tape:
        solver.p2g2p_differentiable(mpm_model, mpm_state, next_state, dt=0.01, device=device)

        wp.launch(
            kernel=compute_loss_kernel,
            dim=n_particles,
            inputs=[next_state.particle_x, loss_array],
            device=device,
        )

    return loss_array.numpy()[0]


# Run the unit test
if __name__ == "__main__":

    """Unit test for verifying gradient computation in p2g2p_differentiable."""
    # Define number of particles and grid size
    n_particles = 10
    n_grid = 32
    grid_lim = 1.0

    device = "cuda:0"
    wp.init()

    # Randomly initialize particle positions and velocities
    particle_x = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)
    particle_v = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)

    mpm_state = MPMStateStruct()
    mpm_state.init(n_particles, device=device, requires_grad=True)

    next_state = MPMStateStruct()
    next_state.init(n_particles, device=device, requires_grad=True)

    # Initialize MPM Model and State
    mpm_model = MPMModelStruct()
    mpm_model.init(n_particles, device=device, requires_grad=True)
    mpm_model.init_other_params(n_grid=n_grid, grid_lim=grid_lim, device=device)

    solver = MPMWARPDiff(n_particles, n_grid, grid_lim, device=device)
    # Set material properties
    solver.set_E_nu(mpm_model, E=1000.0, nu=0.3, device=device)



    # Copy to MPM state
    mpm_state.particle_x = from_torch_safe(particle_x, dtype=wp.vec3, requires_grad=True)
    mpm_state.particle_v = from_torch_safe(particle_v, dtype=wp.vec3, requires_grad=True)

    # Allocate a loss array
    loss_array = torch.zeros(1, dtype=torch.float32, device=device)
    loss_array = wp.from_torch(loss_array, requires_grad=True)
    particle_x_np = mpm_state.particle_x.numpy()
    loss_array_np = loss_array.numpy()
    print(f"check particle_x {type(mpm_state.particle_x)}, {particle_x_np.shape}, loss_array {type(loss_array)}, {loss_array_np.shape}")
    pdb.set_trace()
    
    # --- COMPUTE ANALYTICAL GRADIENTS ---
    with MyTape() as tape:
        solver.p2g2p_differentiable(mpm_model, mpm_state, next_state, dt=0.01, device=device)

        # Launch kernel to compute sum of positions
        wp.launch(
            kernel=compute_loss_kernel,
            dim=n_particles,
            inputs=[next_state.particle_x, loss_array],
            device=device,
        )

        # Compute gradients
        tape.backward(loss_array)

    loss_value = loss_array.numpy()[0]
    print(f"Loss: {loss_value}")

    # Extract analytical gradients
    analytical_grad = mpm_state.particle_x.grad.numpy()
    print(f"Analytical Gradient Shape: {analytical_grad.shape}")

    # Compute numerical gradient using finite differences
    particle_x_numpy = particle_x.detach().cpu().numpy()
    numerical_grad = finite_difference_grad(loss_fn, particle_x_numpy)

    # --- COMPARE GRADIENTS ---
    error = np.linalg.norm(analytical_grad - numerical_grad) / (np.linalg.norm(numerical_grad) + 1e-8)
    print(f"Gradient Error: {error:.6e}")

    # Assert the gradient error is within tolerance
    assert error < 1e-4, "Gradient check failed!"

    print("Gradient verification passed!")