import warp as wp
import torch
import numpy as np

from warp_utils import MyTape, from_torch_safe, CondTape
from mpm_utils import * 
from mpm_solver_diff import MPMWARPDiff
from mpm_data_structure import MPMStateStruct, MPMModelStruct
from run_gaussian_static import load_gaussians, get_volume
import torch.autograd as autograd
import pdb

"""
Test the gradient computation of Young’s modulus E.

Computation graph: 
E -> mu/lam


Returns:
    _type_: _description_
"""

def sum_stress_l2_torch(stress_tensor):
    return (stress_tensor ** 2).sum()

@wp.kernel
def compute_loss_kernel_vec3(particle_x: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, particle_x[tid][0] + particle_x[tid][1] + particle_x[tid][2])

# create a loss kernel that sums up the stress tensor (N, 3, 3) shape
@wp.kernel
def compute_loss_kernel_mat33(particle_stress: wp.array(dtype=wp.mat33), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, particle_stress[tid][0][0] + particle_stress[tid][1][1] + particle_stress[tid][2][2])


class SimulationInterface(autograd.Function):
    @staticmethod
    def forward(ctx, E_tensor, init_x, init_v, init_volume, init_cov, requires_grad=True):
        """
        Forward simulation using the **same** init_x and init_v.
        """

        n_grid = 64
        grid_lim = 1.0
        dt = 0.01

        # Reinitialize MPM state, model, solver
        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, device=device, requires_grad=requires_grad)
        # from_torch function contains init_grid and initializes particle_F_trial to identity
        mpm_state.from_torch(
            init_x,
            init_volume,
            init_cov,
            init_v,
            device=device,
            requires_grad=requires_grad,
            n_grid=n_grid,
            grid_lim=grid_lim,
        )

        next_state = MPMStateStruct()
        next_state.init(n_particles, device=device, requires_grad=requires_grad)
        next_state.from_torch(
            init_x,
            init_volume,
            init_cov,
            init_v,
            device=device,
            requires_grad=requires_grad,
            n_grid=n_grid,
            grid_lim=1.0,
        )


        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=requires_grad)
        mpm_model.init_other_params(n_grid=n_grid, grid_lim=grid_lim, device=device)

        grid_size = (
            mpm_model.grid_dim_x,
            mpm_model.grid_dim_y,
            mpm_model.grid_dim_z,
        )

        


        solver = MPMWARPDiff(n_particles, n_grid, grid_lim, device=device)

        material_params = {
            "material": "jelly",
            "g": [0.0, 0.0, 0],
            "density": 2000,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # 0.999,
        }
        solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        density_tensor = torch.ones(n_particles, dtype=torch.float32, device=device) * material_params["density"]
        mpm_state.reset_density(density_tensor.clone(), device=device, update_mass=True)
        next_state.reset_density(density_tensor.clone(), device=device, update_mass=True)
        # Initialize model parameters
        nu_tensor = torch.ones(n_particles, dtype=torch.float32, device=device) * 0.3
        solver.set_E_nu_from_torch(mpm_model, E_tensor, nu_tensor, device=device)
        # solver.prepare_mu_lam(mpm_model, mpm_state, device) # move inside the tape to obtain correct gradient for E

        # Run forward simulation
        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, True)

        with cond_tape:
            solver.prepare_mu_lam(mpm_model, mpm_state, device)

            wp.launch(
                kernel=zero_grid,  # gradient might gone
                dim=(grid_size),
                inputs=[mpm_state, mpm_model],
                device=device,
            )

            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )  # F and stress are updated

            # wp.launch(
            #     kernel=p2g_apic_with_stress,
            #     dim=n_particles,
            #     inputs=[mpm_state, mpm_model, dt],
            #     device=device,
            # )  # apply p2g'


            # wp.launch(
            #     kernel=grid_normalization_and_gravity,
            #     dim=(grid_size),
            #     inputs=[mpm_state, mpm_model, dt],
            #     device=device,
            # )

            # wp.launch(
            #     kernel=g2p_differentiable,
            #     dim=n_particles,
            #     inputs=[mpm_state, next_state, mpm_model, dt],
            #     device=device,
            # )  # x, v, C, F_trial are updated

        ctx.tape = cond_tape.tape
        ctx.mpm_solver = solver
        ctx.mpm_model = mpm_model
        ctx.mpm_state = mpm_state
        ctx.next_state = next_state
        ctx.n_particles = n_particles
        ctx.save_for_backward(E_tensor)
        particle_stress_torch = wp.to_torch(mpm_state.particle_stress).detach().clone() # shape (n_particles, 3, 3)
        return particle_stress_torch


    @staticmethod
    def backward(ctx, out_stress_grad):
        # Retrieve the tape. Backward pass through the tape
        tape = ctx.tape
        solver = ctx.mpm_solver
        mpm_model = ctx.mpm_model
        mpm_state = ctx.mpm_state
        next_state = ctx.next_state
        n_particles = ctx.n_particles
        E_tensor, = ctx.saved_tensors

        # Ensure the gradient tensor is contiguous before passing to Warp
        out_stress_grad = out_stress_grad.contiguous()
        # Convert to Warp tensor
        grad_stress_wp = from_torch_safe(out_stress_grad, dtype=wp.mat33, requires_grad=False)

        # Allocate memory for scalar loss in Warp
        loss_wp = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        # Compute loss gradient in Warp
        with tape:
            wp.launch(
                compute_stress_loss_with_grad,
                dim=n_particles,
                inputs=[mpm_state,
                        grad_stress_wp,
                        1.0,
                        loss_wp,
                ],
                device=device,
            )
        # Backpropagate through the computational graph
        print(f"check loss_wp in custom backward: {loss_wp.numpy()}")
        tape.backward(loss_wp)

        stress_grad_wp = mpm_state.particle_stress.grad
        # print(f"check stress_grad_wp: {stress_grad_wp.numpy()}") # all ones
        mu_grad_wp = mpm_model.mu.grad
        print(f"check mu_grad_wp: {mu_grad_wp.numpy()}")
        lam_grad_wp = mpm_model.lam.grad
        print(f"check lam_grad_wp: {lam_grad_wp.numpy()}") # TODO: somehow the lam_grad is all zero.

        E_grad_wp = mpm_model.E.grad
        E_grad_torch = wp.to_torch(E_grad_wp).detach().clone()
        print(f"check E_grad_wp: {E_grad_wp.numpy()}")


        return E_grad_torch, None, None, None, None  # No gradients for init_x, init_v, init_volume, init_cov

def check_autodiff(E_tensor, init_x, init_v, init_volume, init_cov, eps=1e-6):
    """
    Check gradient using finite differences while keeping init_x and init_v the same.
    """
    delta_E = torch.rand_like(E_tensor)
    delta_E /= torch.sqrt(torch.sum(delta_E ** 2))  # Normalize perturbation

    E0 = E_tensor.detach().clone() + eps * delta_E
    E1 = E_tensor.detach().clone() - eps * delta_E

    E0.requires_grad_()
    E1.requires_grad_()

    outputs0 = SimulationInterface.apply(E0, init_x, init_v, init_volume, init_cov)
    outputs1 = SimulationInterface.apply(E1, init_x, init_v, init_volume, init_cov)

    # # save output stress tensor to file
    # np.savez("example_stress_tensor.npz", outputs0=outputs0.detach().cpu().numpy(), outputs1=outputs1.detach().cpu().numpy(), E0=E0.detach().cpu().numpy(), E1=E1.detach().cpu().numpy(), init_x=init_x.detach().cpu().numpy(), init_v=init_v.detach().cpu().numpy(), init_volume=init_volume.detach().cpu().numpy(), init_cov=init_cov.detach().cpu().numpy())

    print(f"outputs0: {type(outputs0)}, shape {outputs0.shape}")

    loss0 = torch.sum(outputs0)
    print(f"E0 {E0[0]}, loss0: {loss0} type {type(loss0)}, requires_grad {loss0.requires_grad}")
    loss1 = torch.sum(outputs1)
    print(f"E1 {E1[0]}, loss1: {loss1} type {type(loss1)}, requires_grad {loss1.requires_grad}")

    print(f"Running loss backward ...")
    loss0.backward()
    loss1.backward()

    grad_finite_diff = (loss0 - loss1).item() / (2 * eps)
    grad_autodiff = torch.sum(delta_E * (E0.grad + E1.grad)/2).item()

    print(f"E0 grad: {E0.grad} \nE1 grad: {E1.grad}")

    grad_error = abs(grad_finite_diff - grad_autodiff)

    print('finite_diff:', grad_finite_diff)
    print('analytical:', grad_autodiff)
    print(f'\033[91mgrad_error: {grad_error}\033[0m')

    return grad_error

if __name__ == "__main__":    
    """Unit test for verifying gradient computation of Young’s modulus E."""
    
    n_particles = 10
    initE = 1e6

    device = "cuda:0"
    wp.init()
    # wp.config.mode = "debug"
    wp.config.verify_cuda = True

    # 🔥 Create SAME `init_x` and `init_v` for all perturbations
    init_x = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)
    init_v = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)
    # shift the mean of init_v
    init_v = init_v + 1.0
    print(f"check init_v: mean {torch.mean(init_v)}, std {torch.std(init_v)}")
    volume_array = get_volume(init_x.detach().cpu().numpy())
    init_volume = torch.tensor(volume_array, dtype=torch.float32, device=device, requires_grad=False)
    # initialize the covariance matrix with positive diagonal values
    cov = torch.tensor([0.1, 0.0, 0.0, 0.1, 0.0, 0.1], dtype=torch.float32, device=device, requires_grad=False)
    init_cov = torch.stack([cov for _ in range(n_particles)], dim=0)


    E_tensor = torch.ones(n_particles, dtype=torch.float32, device=device, requires_grad=True) * initE

    # # 🔥 Compute Autodiff and Finite Difference Gradient for E
    # grad_error = check_autodiff(E_tensor, init_x, init_v, init_volume, init_cov, eps=10)

    # # 🔥 Compare error
    # assert grad_error < 1e-7, "Gradient check failed!"
    # print("Gradient verification passed!")

    # Perform a single SGD step to verify gradient correctness
    print("\n###Performing SGD step to reduce loss...###")
    # Clone original stress to perform updates
    initE_sgd = torch.tensor(E_tensor, dtype=torch.float64).to(device).detach().clone().requires_grad_(True)

    # Compute original loss and grad
    stress = SimulationInterface.apply(initE_sgd, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), True)
    loss = sum_stress_l2_torch(stress)
    print(f"Original loss: {loss.item()}")

    loss.backward()
    grad = initE_sgd.grad.clone()
    print(f"Gradient of loss w.r.t. E: \n{grad}")

    # Perform SGD update
    lr = 1e5  # You can tune this
    initE_updated = initE_sgd - lr * grad
    initE_updated = initE_updated.detach().clone().requires_grad_(False)
    print(f"check initE_sgd {initE_sgd}, initE_updated {initE_updated}")

    # Recompute loss after update
    new_stress = SimulationInterface.apply(initE_updated, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
    new_loss = sum_stress_l2_torch(new_stress)
    print(f"New loss after SGD step: {new_loss.item()}")

    if new_loss.item() < loss.item():
        print("✅ SGD step successfully reduced the loss.")
    else:
        print("❌ SGD step did not reduce the loss. Investigate gradient correctness.")

