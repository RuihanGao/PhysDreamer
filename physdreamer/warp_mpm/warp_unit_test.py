import warp as wp
import torch
import numpy as np
from mpm_solver_diff import MPMWARPDiff
from mpm_data_structure import MPMStateStruct, MPMModelStruct
from warp_utils import MyTape, from_torch_safe, CondTape
from mpm_utils import * 
from run_gaussian_static import load_gaussians, get_volume
import torch.autograd as autograd
import pdb

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
    def forward(ctx, E_tensor):
        """
        Forward simulation and loss computation for a given Young’s modulus E.
        """

        n_grid = 32
        grid_lim = 1.0
        dt = 0.01

        # Reinitialize MPM solver, state, and model
        solver = MPMWARPDiff(n_particles, n_grid, grid_lim, device=device)

        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=n_grid, grid_lim=grid_lim, device=device)

        grid_size = (
            mpm_model.grid_dim_x,
            mpm_model.grid_dim_y,
            mpm_model.grid_dim_z,
        )

        density_tensor = torch.ones(n_particles, dtype=torch.float32, device=device) * 0.02
        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, device=device, requires_grad=True)
        mpm_state.init_grid(grid_res=n_grid, device=device, requires_grad=True)
        mpm_state.reset_density(density_tensor, device=device)

        next_state = MPMStateStruct()
        next_state.init(n_particles, device=device, requires_grad=True)
        next_state.init_grid(grid_res=n_grid, device=device, requires_grad=True)
        next_state.reset_density(density_tensor, device=device)

        # Initialize model parameters
        nu_tensor = torch.ones(n_particles, dtype=torch.float32, device=device) * 0.3
        solver.set_E_nu_from_torch(mpm_model, E_tensor, nu_tensor, device=device)
        solver.prepare_mu_lam(mpm_model, mpm_state, device)
        
        material_params = {
        "material": "jelly",
        "g": [0.0, 0.0, 0],
        }
        solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        # Initialize particle states
        init_x = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)
        init_v = torch.rand((n_particles, 3), dtype=torch.float32, device=device, requires_grad=True)
        volume_array = get_volume(init_x.detach().cpu().numpy())
        init_volume = torch.tensor(volume_array, dtype=torch.float32, device=device, requires_grad=True)


        mpm_state.particle_x = from_torch_safe(init_x, dtype=wp.vec3, requires_grad=True)
        mpm_state.particle_v = from_torch_safe(init_v, dtype=wp.vec3, requires_grad=True)
        mpm_state.particle_vol = from_torch_safe(init_volume, requires_grad=False)
        next_state.particle_vol = from_torch_safe(init_volume, requires_grad=False)

        # Allocate a loss array
        loss_array = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        # Run forward simulation
        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, True)

        with cond_tape:
            # solver.p2g2p_differentiable(mpm_model, mpm_state, next_state, dt=0.01, device=device)

            # wp.launch(
            #     kernel=compute_loss_kernel,
            #     dim=n_particles,
            #     inputs=[next_state.particle_x, loss_array],
            #     device=device,
            # )

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
                kernel=p2g_apic_with_stress,
                dim=n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )  # apply p2g'


            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

            wp.launch(
                kernel=g2p_differentiable,
                dim=n_particles,
                inputs=[mpm_state, next_state, mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated


            # grid_v_out_arr = mpm_state.grid_v_out.numpy()
            # print(f"check grid_v_in before computing loss, {grid_v_out_arr.shape}") # [32, 32, 32, 3]
            # # flatten the grid_v_in array [32, 32, 32, 3] -> [n_grids, 3]
            # grid_v_out = torch.tensor(grid_v_out_arr.reshape(-1, 3), dtype=torch.float32, device=device, requires_grad=True)
            # grid_v_out_wp = from_torch_safe(grid_v_out, dtype=wp.vec3, requires_grad=True)

            # wp.launch(
            #     kernel=compute_loss_kernel_vec3,
            #     dim=n_particles,
            #     inputs=[next_state.particle_x, loss_array],
            #     device=device,
            # )

        ctx.tape = cond_tape.tape
        ctx.mpm_solver = solver
        ctx.mpm_model = mpm_model
        ctx.mpm_state = mpm_state
        ctx.next_state = next_state
        ctx.n_particles = n_particles
        ctx.save_for_backward(E_tensor)
        particle_pos = wp.to_torch(next_state.particle_x).detach().clone()
        return particle_pos

    @staticmethod
    def backward(ctx, out_pos_grad):
        # Retrieve the tape. Backward pass through the tape
        tape = ctx.tape
        n_particles = ctx.n_particles
        last_state = ctx.next_state
        grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
        target_pos_detach = wp.clone(last_state.particle_x, device=device, requires_grad=False)
        print(f"in custom backward, check out_pos_grad: {out_pos_grad.shape}")

        with tape:
            loss_wp = torch.zeros(1, device=device)
            loss_wp = wp.from_torch(loss_wp, requires_grad=True)
            wp.launch(
                compute_posloss_with_grad, 
                dim=n_particles,
                inputs=[
                    last_state,
                    target_pos_detach,
                    grad_pos_wp,
                    0.5,
                    loss_wp,
                ],
                device=device,
            )

        print(f"check loss_wp in custom backward: {loss_wp.numpy()}")
        tape.backward(loss_wp)
        E_grad_wp = ctx.mpm_model.E.grad
        print(f"check E_grad_wp: {E_grad_wp.numpy()}")
        pdb.set_trace()
        return E_grad_wp


def check_autodiff(inputs: list, eps=1e-6):
    """
    Check the gradient of a function using finite differences.
    """
    delta_inputs = [torch.rand_like(input) for input in inputs]
    delta_inputs_norm = sum([torch.sum(delta_input ** 2) for delta_input in delta_inputs])
    delta_inputs_norm = torch.sqrt(delta_inputs_norm)
    delta_inputs = [delta_input / delta_inputs_norm for delta_input in delta_inputs]

    inputs0 = [input.detach().clone() + eps * delta_input for input, delta_input in zip(inputs, delta_inputs)]
    inputs1 = [input.detach().clone() - eps * delta_input for input, delta_input in zip(inputs, delta_inputs)]
    
    inputs0 = [input.requires_grad_() for input in inputs0]
    inputs1 = [input.requires_grad_() for input in inputs1]
    # check inputs0 <class 'list'>, [tensor([1000000., 1000000., 1000000., 1000000., 1000000., 1000000., 1000000.,
    #     1000000., 1000000., 1000000.], device='cuda:0', requires_grad=True)]

    outputs0 = SimulationInterface.apply(*inputs0)
    if isinstance(outputs0, torch.Tensor):
        outputs0 = [outputs0]
    coeffs = [torch.rand_like(output) for output in outputs0]
    loss0 = sum([(coeff * output).sum() for coeff, output in zip(coeffs, outputs0)])
    print(f"check losss0 before backward: {type(loss0)} \n {loss0}")
    pdb.set_trace()
    loss0.backward()

    outputs1 = SimulationInterface.apply(*inputs1)
    if isinstance(outputs1, torch.Tensor):
        outputs1 = [outputs1]
    loss1 = sum([(coeff * output).sum() for coeff, output in zip(coeffs, outputs1)])
    loss1.backward()

    grads0 = [input.grad for input in inputs0]
    grads1 = [input.grad for input in inputs1]

    print(f"check grads0: {grads0}")
    print(f"check grads1: {grads1}")
    pdb.set_trace()

    grad_finite_diff =  (loss0 - loss1).item() / eps / 2
    grad_autodiff = sum([torch.sum(delta_input * (grad0 + grad1)) for delta_input, grad0, grad1 in zip(delta_inputs, grads0, grads1)]).item()

    grad_error = grad_finite_diff - grad_autodiff
    
    print('finite_diff:', grad_finite_diff)
    print('analytical:', grad_autodiff)
    print(f'\033[91mgrad_error: {grad_error}\033[0m')

    return grad_error





# Run the unit test
if __name__ == "__main__":    
    """Unit test for verifying gradient computation in p2g2p_differentiable."""
    # Define number of particles and grid size
    n_particles = 10
    initE = 1e6

    device = "cuda:0"
    wp.init()

    # Initialize Young's modulus tensor
    E_tensor = torch.ones(n_particles, dtype=torch.float32, device=device, requires_grad=True) * initE

    # 🔥 Compute Autodiff and Finite Difference Gradient for E
    grad_error = check_autodiff([E_tensor], eps=10)

    # 🔥 Compare error
    assert abs(grad_error) < 1e-4, "Gradient check failed!"
    print("Gradient verification passed!")
