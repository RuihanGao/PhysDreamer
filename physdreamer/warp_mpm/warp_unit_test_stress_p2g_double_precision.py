import warp as wp
import torch
import numpy as np
from mpm_solver_diff_double import MPMWARPDiff
from mpm_data_structure_double import MPMStateStruct, MPMModelStruct
from warp_utils import MyTape, from_torch_safe, CondTape
from mpm_utils_double import * 
from run_gaussian_static import load_gaussians, get_volume
import torch.autograd as autograd
import pdb
import os.path as osp
import os

"""
Test the gradient computation of Young’s modulus E.

Computation graph: 
E -> mu/lam


Returns:
    _type_: _description_
"""

# create a log directory by date
from datetime import datetime
today = datetime.today().strftime('%Y%m%d')
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)

loss_scaling = 1
loss_in_warp = False


# define a warp kernel to scale up the loss
@wp.kernel
def scale_up_loss(
    loss: wp.array(dtype=wp.float64),
    scale: wp.float64
):
    """
    Scale up the loss by a factor `scale`.
    """
    i = wp.tid()
    loss[i] = wp.mul(loss[i], scale)


def sum_grid_vin_l2_torch(grid_v_tensor):
    return (grid_v_tensor ** 2).sum()

class SimulationInterface(autograd.Function):
    @staticmethod
    def forward(ctx, particle_stress, init_x, init_v, init_volume, init_cov, requires_grad=True):
        """
        Forward simulation using the same init_x and init_v.
        """
        n_grid = 5
        grid_lim = 1.0
        dt = 0.1

        # Reinitialize MPM state, model, solver
        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, device=device, requires_grad=requires_grad)
        # print(f"Initializing mpm_state")
        
        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=requires_grad)
        # print(f"Initializing mpm_model")

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
        mpm_model.init_other_params(n_grid=n_grid, grid_lim=grid_lim, device=device)
        # print(f"Initializing mpm_state and mpm_model params")

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

        density_tensor = torch.ones(n_particles, dtype=torch.float64, device=device) * material_params["density"]
        mpm_state.reset_density(density_tensor.clone(), device=device, update_mass=True)

        # Set perturbed particle_stress
        mpm_state.particle_stress = from_torch_safe(particle_stress, dtype=wp.mat33d, requires_grad=requires_grad)

        # Run only `p2g_apic_with_stress`
        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, True)
        if loss_in_warp:
            wp_loss = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=mpm_state.particle_stress.device)

        with cond_tape:
            wp.launch(
                kernel=zero_grid,  # gradient might gone
                dim=(grid_size),
                inputs=[mpm_state, mpm_model],
                device=device,
            )

            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

            if loss_in_warp:
                # Compute the loss in warp 
                wp.launch(
                    sum_grid_vin_l2,
                    dim=grid_size,
                    inputs=[mpm_state, wp_loss],
                    device=device,
                )

                ctx.wp_loss = wp_loss

        ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model
        ctx.tape = wp_tape
        ctx.dt = dt
        ctx.n_particles = n_particles
        # ctx.save_for_backward(particle_stress)
        
        if loss_in_warp:
            # Return the computed loss
            return wp.to_torch(wp_loss).detach().clone().requires_grad_(False)
        else:
            # Return grid_v_in for loss computation
            grid_v_in_torch = wp.to_torch(mpm_state.grid_v_in).detach().clone()
            return grid_v_in_torch
            

        

    @staticmethod
    def backward(ctx, out_grid_vin_grad):
        if loss_in_warp:
            ctx.tape.backward(ctx.wp_loss)
            stress_grad_wp = ctx.mpm_state.particle_stress.grad
            stress_grad_torch = wp.to_torch(stress_grad_wp)
            output = out_grid_vin_grad * stress_grad_torch
            ctx.tape.zero()
            return output, None, None, None, None, None  # No gradients for init_x, init_v, init_volume, init_cov
        else:
            # Retrieve the tape and state objects
            tape = ctx.tape
            mpm_state = ctx.mpm_state
            mpm_model = ctx.mpm_model

            # Convert gradients to Warp format
            out_grid_vin_grad = out_grid_vin_grad.contiguous() # derivative of loss func w.r.t. output grid_v_in, should be all ones. shape [4, 4, 4, 3], dtype float32
            grad_vin_wp = from_torch_safe(out_grid_vin_grad.to(torch.float64), dtype=wp.vec3d, requires_grad=False)
            grid_size = (mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z)
            loss_wp = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)

            with tape:
                wp.launch(
                    compute_grid_vin_loss_with_grad,
                    dim=grid_size,
                    inputs=[mpm_state, grad_vin_wp, 1.0, loss_wp],
                    device=device,
                )

                # # Manually compute the sum of grid_v_in for debugging. Equivalent to the above kernel "compute_grid_vin_loss_with_grad"
                # wp.launch(
                #     sum_grid_vin,
                #     dim=grid_size,
                #     inputs=[mpm_state, loss_wp],
                #     device=device,
                # )

                # NOTE: We don't need run "scale_up_loss" kernel since we use "loss0 = loss0 * loss_scaling" to scale up the loss
                # wp.launch(
                #     scale_up_loss,
                #     dim=1,
                #     inputs=[loss_wp, loss_scaling],
                #     device=device,
                # )

            tape.backward(loss_wp)
            print(f"loss_wp: {loss_wp}")

            stress_grad_wp = mpm_state.particle_stress.grad
            stress_grad_torch = wp.to_torch(stress_grad_wp).detach().clone()
            ctx.tape.zero()

            return stress_grad_torch, None, None, None, None, None  # No gradients for init_x, init_v, init_volume, init_cov




def check_autodiff(particle_stress, init_x, init_v, init_volume, init_cov):
    """
    Check gradient using finite differences by perturbing `particle_stress`.
    """
    # Check input
    print(f"check particle_stress")
    print(particle_stress)
    N, H, W = particle_stress.shape

    # Compute finite difference gradient
    for relative_eps in [1e-5, 1e-6, 1e-7]:
        print(f"\n--- Finite Difference with relative_eps = {relative_eps} ---")
        grad_finite_diff = torch.zeros_like(particle_stress)
        particle_stress_fd = particle_stress.detach().clone()

        for n in range(N):
            for i in range(H):
                for j in range(W):
                    perturbation = relative_eps * max(abs(particle_stress_fd[n, i, j].item()), 1.0) # perturb with relative eps
                    E = torch.zeros_like(particle_stress_fd)
                    E[n, i, j] = perturbation 
                    # print(f"n {n} i {i} j {j} perturbation {perturbation}")

                    if loss_in_warp:
                        # Directly compute loss in warp
                        loss_plus = SimulationInterface.apply(particle_stress_fd + E, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
                        loss_minus = SimulationInterface.apply(particle_stress_fd - E, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
                    else:
                        # Output grid_v_in from warp and compute loss in torch
                        output_plus = SimulationInterface.apply(particle_stress_fd + E, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
                        loss_plus = sum_grid_vin_l2_torch(output_plus)
                        output_minus = SimulationInterface.apply(particle_stress_fd - E, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
                        loss_minus = sum_grid_vin_l2_torch(output_minus)


                    # Apply loss scaling
                    loss_plus = loss_plus * loss_scaling
                    loss_minus = loss_minus * loss_scaling

                    # print(f"n {n} i {i} j {j} loss_plus {loss_plus} loss_minus {loss_minus}")
                    grad_finite_diff[n, i, j] = (loss_plus - loss_minus) / (2 * perturbation)

        print(f"relative_eps: {relative_eps} finite_diff: \n{grad_finite_diff}")


    # Compute auto-diff gradient
    particle_stress.requires_grad_()
    if loss_in_warp:
        loss_autodiff = SimulationInterface.apply(particle_stress, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), True)
    else:
        output_autodiff = SimulationInterface.apply(particle_stress, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), True)
        loss_autodiff = sum_grid_vin_l2_torch(output_autodiff)
    
    loss_autodiff = loss_autodiff * loss_scaling
    loss_autodiff.backward()
    grad_autodiff = particle_stress.grad.clone()
    print("Autograd gradient:\n", grad_autodiff)

    # compute the error
    grad_error = torch.norm(grad_finite_diff - grad_autodiff) / torch.norm(grad_finite_diff + grad_autodiff)
    print(f"grad_error: {grad_error}")

    return grad_error



if __name__ == "__main__":    
    """Unit test for verifying gradient computation of `particle_stress`."""

    n_particles = 1
    device = "cuda:0"
    wp.init()


    # Initialize test data
    # init_x = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)
    # init_v = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True) * 5.0
    # volume_array = get_volume(init_x.detach().cpu().numpy())
    # init_volume = torch.tensor(volume_array, dtype=torch.float32, device=device, requires_grad=False)

    # cov = torch.tensor([0.1, 0.0, 0.0, 0.1, 0.0, 0.1], dtype=torch.float32, device=device, requires_grad=False)
    # init_cov = torch.stack([cov for _ in range(n_particles)], dim=0)
    # particle_stress = torch.rand((n_particles, 3, 3), dtype=torch.float32, device=device, requires_grad=True)

    # Load initial data from file
    input_file_path = "example_stress_tensor.npz"
    input_data = np.load(input_file_path)
    init_x = torch.tensor(input_data["init_x"][:n_particles], dtype=torch.float64, device=device, requires_grad=True)
    init_v = torch.tensor(input_data["init_v"][:n_particles], dtype=torch.float64, device=device, requires_grad=True)
    init_volume = torch.tensor(input_data["init_volume"][:n_particles], dtype=torch.float64, device=device, requires_grad=False) * 1e6 # 20250325 scale up the volume to get larger gradient for stress
    init_cov = torch.tensor(input_data["init_cov"][:n_particles], dtype=torch.float64, device=device, requires_grad=False) 
    particle_stress = torch.tensor(input_data["outputs0"][:n_particles], dtype=torch.float64, device=device, requires_grad=True)


    print(f"check initial particle_stress: shape {particle_stress.shape}, min {particle_stress.min()}, max {particle_stress.max()}")
    print(f"Run with loss_in_warp: {loss_in_warp}")

    # Compute Autodiff and Finite Difference Gradient for particle_stress
    grad_error = check_autodiff(particle_stress, init_x, init_v, init_volume, init_cov)

    # Compare error
    assert grad_error < 1e-5, "Gradient check failed!"
    print("Gradient verification passed!")
