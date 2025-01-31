import torch

# Example MPM-like solver setup
class MPMSolver:
    def __init__(self):
        pass

    def simulate(self, mpm_model, mpm_state):
        # Simulates dynamics and updates mpm_state (placeholder logic)
        mpm_state.particle_x = mpm_state.particle_x + mpm_state.particle_v * 0.1  # Simple linear motion

class MPMState:
    def __init__(self, initial_position, initial_velocity):
        self.particle_x = initial_position.clone().detach().requires_grad_(True)  # Position
        self.particle_v = initial_velocity.clone().detach().requires_grad_(True)  # Velocity

# Define a simple loss function
def simple_loss(mpm_state):
    return torch.sum(mpm_state.particle_x)  # L = x + y + z

# Function to compute numerical gradients using finite difference
def finite_difference_loss(mpm_solver, mpm_state, mpm_model, loss_fn, param, epsilon=1e-5):
    """Compute numerical gradient using finite differences."""
    param_data = param.detach().clone()
    param.requires_grad_(False)

    grad = torch.zeros_like(param_data)
    for idx in range(param.numel()):
        param_data_flat = param_data.view(-1)

        # Perturb positively
        param_data_flat[idx] += epsilon
        perturbed_state = MPMState(
            mpm_state.particle_x.detach().clone(), param_data_flat.view_as(param)
        )  # Reset state with perturbed velocity
        mpm_solver.simulate(mpm_model, perturbed_state)
        loss_pos = loss_fn(perturbed_state)

        # Perturb negatively
        param_data_flat[idx] -= 2 * epsilon
        perturbed_state = MPMState(
            mpm_state.particle_x.detach().clone(), param_data_flat.view_as(param)
        )  # Reset state with perturbed velocity
        mpm_solver.simulate(mpm_model, perturbed_state)
        loss_neg = loss_fn(perturbed_state)

        # Restore original value
        param_data_flat[idx] += epsilon

        print(f"idx {idx}, loss_pos {loss_pos}, loss_neg {loss_neg}")
        # Compute gradient
        grad.view(-1)[idx] = (loss_pos - loss_neg) / (2 * epsilon)

    return grad

# Main script
if __name__ == "__main__":
    # Initialize MPM-like solver, state, and model
    mpm_solver = MPMSolver()
    mpm_model = None  # Placeholder for any model parameters

    # Initial position and velocity
    initial_position = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    initial_velocity = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32)
    mpm_state = MPMState(initial_position, initial_velocity)

    # Autodiff Gradient Calculation
    print(f"check mpm_state before auto diff,  {mpm_state.particle_x}, {mpm_state.particle_v}")
    mpm_solver.simulate(mpm_model, mpm_state)  # Run simulation
    loss = simple_loss(mpm_state)  # Compute loss
    loss.backward()  # Backpropagate

    autodiff_grad = mpm_state.particle_v.grad.clone().detach()
    print(f"check mpm_state after auto diff,  {mpm_state.particle_x}, {mpm_state.particle_v}")

    # Finite Difference Gradient Calculation
    numerical_grad = finite_difference_loss(
        mpm_solver, mpm_state, mpm_model, simple_loss, mpm_state.particle_v
    )

    # Compare Gradients
    print("Autodiff Gradients:", autodiff_grad)
    print("Numerical Gradients:", numerical_grad)
    print("Max Difference:", torch.abs(autodiff_grad - numerical_grad).max())
