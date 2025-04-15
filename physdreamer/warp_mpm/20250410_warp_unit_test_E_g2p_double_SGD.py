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
from utils import draw_grid_v

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
loss_type = 2
LOSS_ON_POS = False

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

def sum_grid_vin_torch(grid_v_tensor):
    return grid_v_tensor.sum()

class SimulationInterface(autograd.Function):
    @staticmethod
    def forward(ctx, E_tensor, init_x, init_v, init_volume, init_cov, requires_grad=True):
        """
        Forward simulation using the same init_x and init_v.
        """
        n_grid = 5
        grid_lim = 1.0
        dt = 1e-4

        # Reinitialize MPM state, model, solver
        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, device=device, requires_grad=requires_grad)
        
        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=requires_grad)

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

        # Initialize E and nu
        nu_tensor = torch.ones(n_particles, dtype=torch.float64, device=device) * 0.3
        solver.set_E_nu_from_torch(mpm_model, E_tensor, nu_tensor, device=device)

        grid_counter_tensor = torch.zeros((n_grid, n_grid, n_grid), dtype=torch.int32, device=device, requires_grad=False)
        grid_counter = wp.from_torch(grid_counter_tensor, dtype=wp.int32)

        # Run forward simulation
        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, True)
        # Define impulse parameters
        impulse_mask = torch.ones(n_particles, dtype=torch.int32, device=device)
        impulse_param = Impulse_modifier()
        impulse_param.start_time = 0.0
        impulse_param.end_time = 1.0
        impulse_param.mask = wp.from_torch(impulse_mask)
        impulse_param.force = wp.vec3d(0.0, 0.0, 1.0)  # Apply Z impulse  

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
            
            wp.launch(
                kernel=p2g_apic_with_stress_debug,
                dim=n_particles,
                inputs=[mpm_state, mpm_model, dt, grid_counter],
                device=device,
            )

            wp.launch(
                kernel=p2g_apply_impulse,
                dim=n_particles,
                inputs=[0.0, dt, mpm_state, mpm_model, impulse_param],
                device=device,
            )

            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

            # 20250408 Add kernel "g2p_differentiable"
            next_state = mpm_state.partial_clone(requires_grad=requires_grad)

            wp.launch(
                kernel=g2p_differentiable,
                dim=n_particles,
                inputs=[mpm_state, next_state, mpm_model, dt],
                device=device,
            )

        ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model
        ctx.next_state = next_state
        ctx.tape = wp_tape
        ctx.dt = dt
        ctx.n_particles = n_particles
        # ctx.save_for_backward(particle_stress)

        # Return new_state.particle_x for loss computation
        if LOSS_ON_POS:
            particle_out_torch = wp.to_torch(next_state.particle_x).detach().clone()
        else:
            particle_out_torch = wp.to_torch(next_state.particle_v).detach().clone()

        return particle_out_torch
            

    @staticmethod
    def backward(ctx, out_particle_x_grad):
        # Retrieve the tape and state objects
        tape = ctx.tape
        mpm_state = ctx.mpm_state
        next_state = ctx.next_state
        mpm_model = ctx.mpm_model
        n_particles = ctx.n_particles

        # Convert gradients to Warp format
        out_particle_x_grad = out_particle_x_grad.contiguous()
        grad_px_wp = from_torch_safe(out_particle_x_grad.to(torch.float64), dtype=wp.vec3d, requires_grad=False)
        grid_size = (mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z)
        loss_wp = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)

        with tape:
            wp.launch(
                compute_px_loss_with_grad,
                dim=n_particles,
                inputs=[next_state, grad_px_wp, 1.0, loss_wp],
                device=device,
            )
      
        print(f"loss_wp: {loss_wp}")
        tape.backward(loss_wp)
        
        stress_grad_wp = mpm_state.particle_stress.grad
        stress_grad_torch = wp.to_torch(stress_grad_wp).detach().clone()
        print(f"stress_grad_torch: \n{stress_grad_torch}")
        # save stress tensor to a numpy file
        output_path = osp.join(log_dir, "stress_tensor_from_E.npz")
        np.savez(output_path, stress_grad=stress_grad_torch.detach().cpu().numpy(), stress=mpm_state.particle_stress.numpy())

        pdb.set_trace()
        E_grad_wp = mpm_model.E.grad
        E_grad_torch = wp.to_torch(E_grad_wp).detach().clone()

        ctx.tape.zero()

        return E_grad_torch, None, None, None, None, None  # No gradients for init_x, init_v, init_volume, init_cov




# TODO: update check_autodiff for input E. 
def check_autodiff(particle_stress, init_x, init_v, init_volume, init_cov):
    """
    Check gradient using finite differences by perturbing `particle_stress`.
    """
    # Check input
    print(f"check particle_stress")
    print(particle_stress)
    N, H, W = particle_stress.shape
    particle_stress_ = particle_stress.detach().clone()

    # Compute auto-diff gradient
    particle_stress.requires_grad_()
    output_autodiff = SimulationInterface.apply(particle_stress, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), True)
    if loss_type == 1:
        loss_autodiff = sum_grid_vin_torch(output_autodiff)
    elif loss_type == 2:
        loss_autodiff = sum_grid_vin_l2_torch(output_autodiff)
    else:
        raise NotImplementedError(f"loss_type {loss_type} not implemented")
    
    loss_autodiff = loss_autodiff * loss_scaling
    loss_autodiff.backward()
    grad_autodiff = particle_stress.grad.clone()
    print("Autograd gradient:\n", grad_autodiff)

    # ##### Test of the Loss Landscape (Verify that it is quadratic) #####
    # # Plot the v values for stress
    # loss_list = []
    # stress_element_list = []
    # idx_i = 0
    # idx_j = 1
    # idx_k = 0

    # list_a =  [particle_stress_[idx_i, idx_j, idx_k] * i * 10**(-5) for i in range(-10, 10)]
    # # list_b =  np.arange(1e7, 6e7, 1e7).tolist()
    # # list_b = [torch.tensor(i, dtype=torch.float64, device=device) for i in list_b]
    # list_b = []

    # stress_element_list = list_a + list_b
    # print(f"stress_element_list {stress_element_list}")
    # print(f"original stress element {particle_stress_[idx_i, idx_j, idx_k]}") 
    
    # for stress_element in stress_element_list:
    #     particle_stress_test = particle_stress_.detach().clone()
    #     # multiply the first elemnt of the stress tensor by 10^i
    #     particle_stress_test[idx_i, idx_j, idx_k] = stress_element
        
    #     # run the simulation with the perturbed stress tensor
    #     output = SimulationInterface.apply(particle_stress_test, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
    #     if loss_type == 1:
    #         loss = sum_grid_vin_torch(output)
    #     elif loss_type == 2:
    #         loss = sum_grid_vin_l2_torch(output)
        
    #     loss = loss * loss_scaling
    #     loss_list.append(loss.item())
    #     print(f"stress_element {stress_element}, loss {loss}")
    
    # # plot the loss values
    # import matplotlib.pyplot as plt
    # n_particles = particle_stress.shape[0]
    # stress_element_list = [i.item() for i in stress_element_list]
    # plt.scatter(stress_element_list, loss_list)
    # # set x axis to be in log scale
    # # plt.xscale("log")
    # plt.xlabel("Stress element")
    # plt.ylabel("Loss")
    # plt.title(f"Sensitivity of loss_type {loss_type} to stress element")
    # plt.savefig(f"{log_dir}/sensitivity_stress_element_{idx_i}_{idx_j}_{idx_k}_loss_{loss_type}_nparticles_{n_particles}.png")
    # plt.close()
    # pdb.set_trace()
    # ##### Test of the Loss Landscape Ends #####
        


    # Compute finite difference gradient
    for relative_eps in [1e-5]:
        print(f"\n--- Finite Difference with relative_eps = {relative_eps} ---")
        grad_finite_diff = torch.zeros_like(particle_stress)
        particle_stress_fd = particle_stress_.detach().clone()

        for n in range(N):
            for i in range(H):
                for j in range(W):
                    perturbation = relative_eps * max(abs(particle_stress_fd[n, i, j].item()), 1e-3) # perturb with relative eps
                    E = torch.zeros_like(particle_stress_fd)
                    E[n, i, j] = perturbation 
                    # print(f"n {n} i {i} j {j} perturbation {perturbation}")

                    # Output grid_v_in from warp and compute loss in torch
                    output_plus = SimulationInterface.apply(particle_stress_fd + E, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
                    output_minus = SimulationInterface.apply(particle_stress_fd - E, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
                    if loss_type == 1:
                        loss_plus = sum_grid_vin_torch(output_plus)
                        loss_minus = sum_grid_vin_torch(output_minus)
                    elif loss_type == 2:
                        loss_plus = sum_grid_vin_l2_torch(output_plus)
                        loss_minus = sum_grid_vin_l2_torch(output_minus)
                    else:
                        raise NotImplementedError(f"loss_type {loss_type} not implemented")

                    # Apply loss scaling
                    loss_plus = loss_plus * loss_scaling
                    loss_minus = loss_minus * loss_scaling

                    # print(f"n {n} i {i} j {j} loss_plus {loss_plus} loss_minus {loss_minus}")
                    
                    grad_finite_diff[n, i, j] = (loss_plus - loss_minus) / (2 * perturbation)

                    # print(f"perturbation {perturbation}, loss_plus {loss_plus}, loss_minus {loss_minus}, grad {grad_finite_diff[n, i, j]}")

        print(f"relative_eps: {relative_eps}, loss_plus {loss_plus}, loss_minus {loss_minus}, finite_diff: \n{grad_finite_diff}")

        # compute the relative error for the vectos
        # print(f"gradient diff")
        # print(grad_finite_diff - grad_autodiff)
        grad_error = torch.norm(grad_finite_diff - grad_autodiff) / torch.norm(grad_finite_diff + grad_autodiff)
        print(f"relative grad_error: {grad_error}")

    return grad_error, output_autodiff



if __name__ == "__main__":    
    """Unit test for verifying gradient computation of `particle_stress`."""

    n_particles = 10
    initE = 1e6
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

    # tensor([[0.0047, 0.1895, 0.8621],
    #         [0.9464, 0.9218, 0.3106],
    #         [0.1529, 0.7100, 0.1264]]

    init_v = torch.tensor(input_data["init_v"][:n_particles], dtype=torch.float64, device=device, requires_grad=True)
    init_volume = torch.tensor(input_data["init_volume"][:n_particles], dtype=torch.float64, device=device, requires_grad=False)*1e3 # 20250325 scale up the volume to get larger gradient for stress
    init_cov = torch.tensor(input_data["init_cov"][:n_particles], dtype=torch.float64, device=device, requires_grad=False) 


    # # Compute Autodiff and Finite Difference Gradient for particle_stress
    # grad_error, output_autodiff = check_autodiff(particle_stress, init_x, init_v, init_volume, init_cov)

    # # Compare error
    # if grad_error < 1e-3:
    #     print("Gradient verification passed!")
    # else:
    #     print("Gradient check failed!")
    


    # Perform a single SGD step to verify gradient correctness
    print("\n###Performing SGD step to reduce loss...###")

    # Clone original stress to perform updates
    initE_sgd = torch.tensor(initE, dtype=torch.float64).to(device).detach().clone().requires_grad_(True)

    # Compute original loss and grad
    particle_x = SimulationInterface.apply(initE_sgd, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), True)
    if loss_type == 1:
        loss = sum_grid_vin_torch(particle_x)
    elif loss_type == 2:
        loss = sum_grid_vin_l2_torch(particle_x)

    loss = loss * loss_scaling
    print(f"Original loss: {loss.item()}")

    loss.backward()
    grad = initE_sgd.grad.clone()
    print(f"Gradient of loss w.r.t. E: \n{grad}")
    # # save the gradient to a numpy array
    # output_path = osp.join(log_dir, "gradient_w_apply_impulse.npy")
    # np.save(output_path, grad.detach().cpu().numpy())
    # print(f"Save autodiff gradient to {output_path}")


    # print(f"Compare grid_v and output_autodiff")
    # grid_v_diff = grid_v - output_autodiff
    # print(f"grid_v_diff: {grid_v_diff}")
    # # draw the plot for grid_v_diff
    # draw_grid_v(grid_v_diff.detach().cpu().numpy(), log_dir, figname=f"grid_v_diff_n_particles_{n_particles}.png")
    # pdb.set_trace()


    # Perform SGD update
    lr = 1e20  # You can tune this
    initE_updated = initE_sgd - lr * grad
    initE_updated = initE_updated.detach().clone().requires_grad_(False)
    print(f"check initE_sgd {initE_sgd}, initE_updated {initE_updated}")


    # Recompute loss after update
    new_particle_x = SimulationInterface.apply(initE_updated, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
    if loss_type == 1:
        new_loss = sum_grid_vin_torch(new_particle_x)
    elif loss_type == 2:
        new_loss = sum_grid_vin_l2_torch(new_particle_x)

    new_loss = new_loss * loss_scaling
    print(f"New loss after SGD step: {new_loss.item()}")

    if new_loss.item() < loss.item():
        print("✅ SGD step successfully reduced the loss.")
    else:
        print("❌ SGD step did not reduce the loss. Investigate gradient correctness.")
