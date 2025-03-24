import warp as wp
import torch
import numpy as np
from warp_utils import MyTape, from_torch_safe, CondTape
from run_gaussian_static import load_gaussians, get_volume
import torch.autograd as autograd
import pdb
import os.path as osp
import os

DTYPE=64

if DTYPE == 32:
    from mpm_solver_diff import MPMWARPDiff
    from mpm_data_structure import MPMStateStruct, MPMModelStruct
    from mpm_utils import * 

    TORCH_DTYPE = torch.float32
    WARP_DTYPE = wp.float32
    WARP_VEC_DTYPE = wp.vec3f
    WARP_MAT_DTYPE = wp.mat33f
elif DTYPE == 64:
    from mpm_solver_diff_double import MPMWARPDiff
    from mpm_data_structure_double import MPMStateStruct, MPMModelStruct
    from mpm_utils_double import * 

    TORCH_DTYPE = torch.float64
    WARP_DTYPE = wp.float64
    WARP_VEC_DTYPE = wp.vec3d
    WARP_MAT_DTYPE = wp.mat33d
else:
    raise ValueError("Invalid DTYPE. Choose 32 or 64.")




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

loss_dict = {"torch": {}, "warp": {}}


class SimulationInterface(autograd.Function):
    @staticmethod
    def forward(ctx, particle_stress, init_x, init_v, init_volume, init_cov, n_grid, perturbation_idx):
        """
        Forward simulation using the same init_x and init_v.
        """
        grid_lim = 1.0
        dt = 0.1

        # Reinitialize MPM state, model, solver
        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, device=device, requires_grad=True)
        print(f"Initializing mpm_state")
        
        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=True)
        print(f"Initializing mpm_model")

        mpm_state.from_torch(
            init_x,
            init_volume,
            init_cov,
            init_v,
            device=device,
            requires_grad=True,
            n_grid=n_grid,
            grid_lim=grid_lim,
        )        
        mpm_model.init_other_params(n_grid=n_grid, grid_lim=grid_lim, device=device)

        print(f"Initializing mpm_state and mpm_model params")

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

        density_tensor = torch.ones(n_particles, dtype=TORCH_DTYPE, device=device) * material_params["density"]
        mpm_state.reset_density(density_tensor.clone(), device=device, update_mass=True)

        # Set perturbed particle_stress
        mpm_state.particle_stress = from_torch_safe(particle_stress, dtype=WARP_MAT_DTYPE, requires_grad=True)

        # Run only `p2g_apic_with_stress`
        wp_tape = MyTape()
        cond_tape: CondTape = CondTape(wp_tape, True)


        # print(f"Before running p2g_apic_with_stress")
        # print(f"check particle_C")
        # print(mpm_state.particle_C.numpy()) # all zeros
        print(f"check particle_vol, shape {mpm_state.particle_vol.numpy().shape}")
        # print(mpm_state.particle_vol.numpy())
        # print(f"check rpic_damping")
        # print(mpm_model.rpic_damping)
        # pdb.set_trace()


        # print("Before p2g_apic_with_stress_simplified:")
        # print(mpm_state.grid_v_in.numpy())

        with cond_tape:
            wp.launch(
                kernel=zero_grid,  # gradient might gone
                dim=(grid_size),
                inputs=[mpm_state, mpm_model],
                device=device,
            )

            wp.launch(
                kernel=p2g_apic_with_stress_simplified,
                dim=n_particles,
                inputs=[mpm_state, mpm_model, dt],
                device=device,
            )

        # print("After p2g_apic_with_stress_simplified:")
        # print(mpm_state.grid_v_in.numpy())

        ctx.tape = wp_tape
        ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model
        ctx.n_particles = n_particles
        ctx.save_for_backward(particle_stress)
        # save for logging purpose
        ctx.n_grid = n_grid
        ctx.perturbation_idx = perturbation_idx

        # Return grid_v_in for loss computation
        grid_v_in_torch = wp.to_torch(mpm_state.grid_v_in).detach().clone()
        return grid_v_in_torch

    @staticmethod
    def backward(ctx, out_grid_vin_grad):
        # Retrieve the tape and state objects
        tape = ctx.tape
        mpm_state = ctx.mpm_state
        mpm_model = ctx.mpm_model
        particle_stress, = ctx.saved_tensors

        # Convert gradients to Warp format
        out_grid_vin_grad = out_grid_vin_grad.contiguous() # derivative of loss func w.r.t. output grid_v_in, should be all ones. shape [4, 4, 4, 3], dtype float32
        grad_vin_wp = from_torch_safe(out_grid_vin_grad.to(TORCH_DTYPE), dtype=WARP_VEC_DTYPE, requires_grad=False)

        loss_wp = wp.zeros(1, dtype=WARP_DTYPE, device=device, requires_grad=True)

        grid_size = (mpm_model.grid_dim_x, mpm_model.grid_dim_y, mpm_model.grid_dim_z)

        with tape:
            # wp.launch(
            #     compute_grid_vin_loss_with_grad,
            #     dim=grid_size,
            #     inputs=[mpm_state, grad_vin_wp, 1.0, loss_wp],
            #     device=device,
            # )

            wp.launch(
                sum_grid_vin,
                dim=grid_size,
                inputs=[mpm_state, loss_wp],
                device=device,
            )

        tape.backward(loss_wp)
        print(f"loss_wp: {loss_wp}")
        n_grid = ctx.n_grid
        perturbation_idx = ctx.perturbation_idx
        if n_grid in loss_dict["warp"]:
            loss_dict["warp"][n_grid][perturbation_idx] = loss_wp.numpy()[0]
        else:
            loss_dict["warp"][n_grid] = {}
            loss_dict["warp"][n_grid][perturbation_idx] = loss_wp.numpy()[0]

        stress_grad_wp = mpm_state.particle_stress.grad
        stress_grad_torch = wp.to_torch(stress_grad_wp).detach().clone()
        

        return stress_grad_torch, None, None, None, None, None, None  # No gradients for init_x, init_v, init_volume, init_cov, n_grid


def check_autodiff_loss(particle_stress, init_x, init_v, init_volume, init_cov, eps=1e-6, n_grid=4):
    """
    Check gradient using finite differences by perturbing `particle_stress`.
    """
    # delta_stress = torch.rand_like(particle_stress)
    # # save delta_stress for debugging
    # np.savez("delta_stress.npz", delta_stress=delta_stress.cpu().numpy())
    n_particles = particle_stress.shape[0]
    delta_stress = np.load("delta_stress.npz")["delta_stress"][:n_particles]
    delta_stress = torch.tensor(delta_stress, dtype=TORCH_DTYPE, device=device, requires_grad=False)
    delta_stress /= torch.sqrt(torch.sum(delta_stress ** 2))  # Normalize perturbation

    stress0 = particle_stress.detach().clone() + eps * delta_stress
    stress1 = particle_stress.detach().clone() - eps * delta_stress

    stress0.requires_grad_()
    stress1.requires_grad_()
    print(f"check stress0: shape {stress0.shape}, min {stress0.min()}, max {stress0.max()}")
    print(f"check stress1: shape {stress1.shape}, min {stress1.min()}, max {stress1.max()}")

    print(f"check dtype: stress0 {stress0.dtype}, init_x {init_x.dtype}, init_v {init_v.dtype}, init_volume {init_volume.dtype}, init_cov {init_cov.dtype}")
    outputs0 = SimulationInterface.apply(stress0, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), n_grid, 0)
    print(f"outputs0: {type(outputs0)}, shape {outputs0.shape}")

    outputs1 = SimulationInterface.apply(stress1, init_x.detach().clone(), init_v.detach().clone(), init_volume.detach().clone(), init_cov.detach().clone(), n_grid, 1)
    print(f"outputs1: {type(outputs1)}, shape {outputs1.shape}")

    loss0 = torch.sum(outputs0.to(TORCH_DTYPE))
    print(f"loss0: {loss0}, requires_grad {loss0.requires_grad}")
    loss1 = torch.sum(outputs1.to(TORCH_DTYPE))
    print(f"loss1: {loss1}, requires_grad {loss1.requires_grad}")


    loss_dict["torch"][n_grid] = {0: loss0.item(), 1: loss1.item()}

    print(f"Running loss backward ...")
    loss0.backward()
    loss1.backward()

    print(f"check loss_dict for n_grid {n_grid}: \n{loss_dict}")








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
    init_x = torch.tensor(input_data["init_x"][:n_particles], dtype=TORCH_DTYPE, device=device, requires_grad=True)
    init_v = torch.tensor(input_data["init_v"][:n_particles], dtype=TORCH_DTYPE, device=device, requires_grad=True)
    print(f"check initv: \n {init_v}")
    pdb.set_trace()
    init_volume = torch.tensor(input_data["init_volume"][:n_particles], dtype=TORCH_DTYPE, device=device, requires_grad=False)
    init_cov = torch.tensor(input_data["init_cov"][:n_particles], dtype=TORCH_DTYPE, device=device, requires_grad=False)
    particle_stress = torch.tensor(input_data["outputs0"][:n_particles], dtype=TORCH_DTYPE, device=device, requires_grad=True)*1e10

    print(f"check initial particle_stress: shape {particle_stress.shape}, min {particle_stress.min()}, max {particle_stress.max()}")

    # Compute Autodiff and Finite Difference Gradient for particle_stress
    for n_grid in range(1, 6):
        relative_eps = 1e-6
        eps = relative_eps * torch.norm(particle_stress).item()

        check_autodiff_loss(particle_stress, init_x, init_v, init_volume, init_cov, eps=eps, n_grid=n_grid)
    
    print(f"check final loss_dict: \n{loss_dict}")

    # Save loss_dict to dictionary file
    
    if DTYPE == 32:
        output_name = "loss_dict_f32.pkl"
    else:
        # DTYPE == 64   
        output_name = "loss_dict_f64.pkl"
        
    with open(osp.join(log_dir, output_name), "wb") as f:
        import pickle
        pickle.dump(loss_dict, f)
        print(f"loss_dict saved to {osp.join(log_dir, output_name)}")


