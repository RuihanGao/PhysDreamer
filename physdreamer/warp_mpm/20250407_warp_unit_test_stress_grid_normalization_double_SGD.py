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
    def forward(ctx, particle_stress, init_x, init_v, init_volume, init_cov, requires_grad=True):
        """
        Forward simulation using the same init_x and init_v.
        """
        n_grid = 5
        grid_lim = 1.0
        dt = 1e-4

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

        grid_counter_tensor = torch.zeros((n_grid, n_grid, n_grid), dtype=torch.int32, device=device, requires_grad=False)
        grid_counter = wp.from_torch(grid_counter_tensor, dtype=wp.int32)

        # Run only `p2g_apic_with_stress`
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
            wp.launch(
                kernel=zero_grid,  # gradient might gone
                dim=(grid_size),
                inputs=[mpm_state, mpm_model],
                device=device,
            )

            # print(f"check particle_mass, should be all non-negative")
            # particle_mass = wp.to_torch(mpm_state.particle_mass).detach().clone()
            # print(f"particle_mass min {particle_mass.min()}, max {particle_mass.max()}, number of non-zero elements {torch.count_nonzero(particle_mass)}") # min 7.811038813088089, max 7.811038813088089, 10
            # mass_mask = torch.where(particle_mass > 0, torch.ones_like(particle_mass), torch.zeros_like(particle_mass))
            # print(mass_mask) # all ones

            grid_m = wp.to_torch(mpm_state.grid_m).detach().clone()
            print(f"Before apic, grid_m min {grid_m.min()}, max {grid_m.max()}")
   
            wp.launch(
                kernel=p2g_apic_with_stress_debug,
                dim=n_particles,
                inputs=[mpm_state, mpm_model, dt, grid_counter],
                device=device,
            )

            grid_m = wp.to_torch(mpm_state.grid_m).detach().clone()
            print(f"After apic, grid_m min {grid_m.min()}, max {grid_m.max()}")
            
            wp.launch(
                kernel=p2g_apply_impulse,
                dim=n_particles,
                inputs=[0.0, dt, mpm_state, mpm_model, impulse_param],
                device=device,
            )

            grid_v_in_tensor = wp.to_torch(mpm_state.grid_v_in).detach().clone()

            # 20250407 Add kernel "grid_normalization_and_gravity"
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

        ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model
        ctx.tape = wp_tape
        ctx.dt = dt
        ctx.n_particles = n_particles
        # ctx.save_for_backward(particle_stress)

        # Return grid_v_out for loss computation
        grid_v_out_torch = wp.to_torch(mpm_state.grid_v_out).detach().clone()

        return grid_v_out_torch
            

    @staticmethod
    def backward(ctx, out_grid_vin_grad):
        # Retrieve the tape and state objects
        tape = ctx.tape
        mpm_state = ctx.mpm_state
        mpm_model = ctx.mpm_model

        # Convert gradients to Warp format
        out_grid_vin_grad = out_grid_vin_grad.contiguous() # derivative of loss func w.r.t. output grid_v_in, should be all ones. shape [4, 4, 4, 3], dtype float32
        grad_vin_wp = from_torch_safe(out_grid_vin_grad.to(torch.float64), dtype=wp.vec3d, requires_grad=False)
        grid_size = (mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z)
        loss_wp = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)

        # # TODO: check if we can directly assign the gradient from torch to warp
        # mpm_state.grid_v_in.grad = grad_vin_wp
        # tape.backward() # NOTE: it doesn't work -> return all zeros

        with tape:
            wp.launch(
                compute_grid_vout_loss_with_grad,
                dim=grid_size,
                inputs=[mpm_state, grad_vin_wp, 1.0, loss_wp],
                device=device,
            )
      
        print(f"loss_wp: {loss_wp}")
        tape.backward(loss_wp)
        
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
    # inspect init_x to see if particles have overlapped grid nodes for perturbation
    print("init_x")
    print(init_x)

    # tensor([[0.0047, 0.1895, 0.8621],
    #         [0.9464, 0.9218, 0.3106],
    #         [0.1529, 0.7100, 0.1264]]


    # Manually spread out particles in a grid layout
    grid_spacing = 0.3  # Adjust this to control spacing between particles
    side = int(np.ceil(n_particles ** (1/3)))  # For 3D layout if needed

    # positions = []
    # for i in range(n_particles):
    #     x = (i % side) * grid_spacing + 0.2
    #     y = ((i // side) % side) * grid_spacing + 0.2
    #     z = (i // (side * side)) * grid_spacing + 0.2
    #     positions.append([x, y, z])

    # init_x = torch.tensor(positions, dtype=torch.float64, device=device, requires_grad=True)


    # init_x = torch.tensor([
    #     [0.2, 0.4, 0.2],
    #     [0.5, 0.7, 0.5],
    #     [0.4, 0.3, 0.8],
    # ], dtype=torch.float64, device=device, requires_grad=True)
    # print("Manually spread out init_x")
    # print(init_x)

    # pdb.set_trace()    



    init_v = torch.tensor(input_data["init_v"][:n_particles], dtype=torch.float64, device=device, requires_grad=True)
    init_volume = torch.tensor(input_data["init_volume"][:n_particles], dtype=torch.float64, device=device, requires_grad=False)*1e3 # 20250325 scale up the volume to get larger gradient for stress
    init_cov = torch.tensor(input_data["init_cov"][:n_particles], dtype=torch.float64, device=device, requires_grad=False) 
    particle_stress = torch.tensor(input_data["outputs0"][:n_particles], dtype=torch.float64, device=device, requires_grad=True)

    print(f"check particle_stress")
    print(particle_stress)
    # tensor([[[ 0.0459,  0.0068, -0.0173],
    #         [ 0.0068,  0.0459, -0.0119],
    #         [-0.0173, -0.0119,  0.0459]],

    #         [[ 0.0458,  0.0068, -0.0173],
    #         [ 0.0068,  0.0458, -0.0119],
    #         [-0.0173, -0.0119,  0.0458]],

    #         [[ 0.0458,  0.0068, -0.0173],
    #         [ 0.0068,  0.0458, -0.0119],
    #         [-0.0173, -0.0119,  0.0458]]], device='cuda:0', dtype=torch.float64,
    #     requires_grad=True)



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
    particle_stress_sgd = particle_stress.detach().clone().requires_grad_(True)

    # Compute original loss and grad
    grid_v = SimulationInterface.apply(particle_stress_sgd, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), True)
    if loss_type == 1:
        loss = sum_grid_vin_torch(grid_v)
    elif loss_type == 2:
        loss = sum_grid_vin_l2_torch(grid_v)

    loss = loss * loss_scaling
    print(f"Original loss: {loss.item()}")
    loss.backward()
    grad = particle_stress_sgd.grad.clone()
    print(f"Gradient of loss w.r.t. particle_stress: \n{grad}")
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
    lr = 1e-1  # You can tune this
    particle_stress_updated = particle_stress_sgd - lr * grad
    particle_stress_updated = particle_stress_updated.detach().clone().requires_grad_(False)
    # print(f"check particle_stress_updated")
    # print(particle_stress_updated)

    # Recompute loss after update
    new_grid_v = SimulationInterface.apply(particle_stress_updated, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), False)
    if loss_type == 1:
        new_loss = sum_grid_vin_torch(new_grid_v)
    elif loss_type == 2:
        new_loss = sum_grid_vin_l2_torch(new_grid_v)

    new_loss = new_loss * loss_scaling
    print(f"New loss after SGD step: {new_loss.item()}")

    if new_loss.item() < loss.item():
        print("✅ SGD step successfully reduced the loss.")
    else:
        print("❌ SGD step did not reduce the loss. Investigate gradient correctness.")
