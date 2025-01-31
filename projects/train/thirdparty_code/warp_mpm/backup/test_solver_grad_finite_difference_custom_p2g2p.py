import torch

import warp as wp
import numpy as np
import torch
import os
from mpm_solver_warp_diff import MPM_Simulator_WARPDiff
from run_gaussian_static import load_gaussians, get_volume
from tqdm import tqdm
from fire import Fire

from diff_warp_utils import MPMStateStruct, MPMModelStruct
from warp_rewrite import MyTape

from mpm_utils import *
import random
import pdb

@wp.kernel
def update_position(x: wp.array(dtype=wp.vec3),
                    v: wp.array(dtype=wp.vec3),
                    dt: float,):

    i = wp.tid()  # Thread index
    wp.atomic_add(x, i,  v[i]*dt)

class Custom_MPM_Simulator_WARPDiff(MPM_Simulator_WARPDiff):
    def __init__(self, n_particles, n_grid=200, grid_lim=1, device="cuda:0"):
        super().__init__(n_particles, n_grid, grid_lim, device)
    def simulate(self, mpm_model, mpm_state):
        # Simulates dynamics and updates mpm_state (placeholder logic)
        # print(f"in simulate, particle_x type {type(mpm_state.particle_x)}, shape {mpm_state.particle_x.numpy().shape}, \n particle_v type {type(mpm_state.particle_v)}, shape {mpm_state.particle_v.numpy().shape}")
        # # warp array, shape [1, 3]
        # pdb.set_trace()
        dt = float(0.1)  # Scalar
        # mpm_state.particle_x = mpm_state.particle_x + mpm_state.particle_v * dt # Simple linear motion
        # Launch the kernel
        wp.launch(kernel=update_position,
            dim=mpm_state.particle_x.shape[0],  # Process all elements in the arrays
            inputs=[mpm_state.particle_x, mpm_state.particle_v, dt])

# Define a simple loss function
def simple_loss(mpm_state):
    return torch.sum(wp.to_torch(mpm_state.particle_x))  # L = x + y + z


@wp.kernel
def compute_loss(particle_x: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, particle_x[tid][0] + particle_x[tid][1] + particle_x[tid][2])

# Function to compute numerical gradients using finite difference
def finite_difference_loss(mpm_solver, mpm_state, mpm_model, loss_fn, param, epsilon=1e-5):
    """Compute numerical gradient using finite differences."""
    
    param = param.numpy()
    param_data = param
    print(f"computing finite difference for param {type(param)}, {param}, pram_data {param_data}")
    # param_data = param.detach().clone()
    # param.requires_grad_(False)
    
    grad = torch.zeros_like(torch.from_numpy(param_data))
    for idx in range(param.size):
        print(f"computing finite difference for idx {idx}")
        param_data_flat = param_data.flatten()

        loss_pos = torch.zeros(1, device=device)
        loss_pos = wp.from_torch(loss_pos, requires_grad=True)

        # Perturb positively
        param_data_flat[idx] += epsilon

        # Reset mpm_state
        mpm_state = MPMStateStruct()
        mpm_state.init(init_position.shape[0], device=device, requires_grad=True)
        mpm_state.from_torch(
            tensor_init_pos,
            volume_tensor,
            tensor_init_cov,
            tensor_init_velocity,
            device=device,
            requires_grad=True,
            n_grid=200,
            grid_lim=1.0,
        )

        # Reset mpm_model
        perturbed_model = MPMModelStruct()
        perturbed_model.init(n_particles, device=device, requires_grad=True)
        perturbed_model.init_other_params(n_grid=200, grid_lim=1.0, device=device)

        E_tensor = (
            (torch.ones(n_particles) * param_data_flat).contiguous().to(device)
        )
        nu_tensor = (
            (torch.ones(n_particles) * material_params["nu"]).contiguous().to(device)
        )
        perturbed_model.from_torch(E_tensor, nu_tensor, device=device, requires_grad=True)

        mpm_solver = Custom_MPM_Simulator_WARPDiff(
            n_particles, n_grid=200, grid_lim=1.0, device=device
        )

        mpm_solver.set_parameters_dict(perturbed_model, mpm_state, material_params)
        # set boundary conditions
        static_center_point = (
            torch.from_numpy(gaussian_dict["satic_center_point"]).float().to(device)
        )
        max_static_offset = (
            torch.from_numpy(gaussian_dict["max_static_offset"]).float().to(device)
        )
        velocity = torch.zeros_like(static_center_point)
        mpm_solver.enforce_particle_velocity_translation(
            mpm_state,
            static_center_point,
            max_static_offset,
            velocity,
            start_time=0,
            end_time=1000,
            device=device,
        )

        mpm_state.set_require_grad(True)

        with tape:
            # Run simulation
            for k in range(10):
                mpm_solver.p2g2p(perturbed_model, mpm_state, k, 0.001, device=device)
              
            wp.launch(
                kernel=compute_loss,
                dim = mpm_state.particle_x.shape[0],
                inputs = [mpm_state.particle_x, loss_pos],
                device = device
            )

        print(f"finish computing loss_pos {loss_pos.numpy()}")

        loss_neg = torch.zeros(1, device=device)
        loss_neg = wp.from_torch(loss_neg, requires_grad=True)

        # Perturb negatively
        param_data_flat[idx] -= 2 * epsilon


        # Reset mpm_state
        mpm_state = MPMStateStruct()
        mpm_state.init(init_position.shape[0], device=device, requires_grad=True)
        mpm_state.from_torch(
            tensor_init_pos,
            volume_tensor,
            tensor_init_cov,
            tensor_init_velocity,
            device=device,
            requires_grad=True,
            n_grid=200,
            grid_lim=1.0,
        )

        # Reset mpm_model
        perturbed_model = MPMModelStruct()
        perturbed_model.init(n_particles, device=device, requires_grad=True)
        perturbed_model.init_other_params(n_grid=200, grid_lim=1.0, device=device)

        E_tensor = (torch.ones(n_particles) * param_data_flat).contiguous().to(device)
        nu_tensor = (
            (torch.ones(n_particles) * material_params["nu"]).contiguous().to(device)
        )
        perturbed_model.from_torch(E_tensor, nu_tensor, device=device, requires_grad=True)

        mpm_solver = Custom_MPM_Simulator_WARPDiff(
            n_particles, n_grid=200, grid_lim=1.0, device=device
        )

        mpm_solver.set_parameters_dict(perturbed_model, mpm_state, material_params)
        # set boundary conditions
        static_center_point = (
            torch.from_numpy(gaussian_dict["satic_center_point"]).float().to(device)
        )
        max_static_offset = (
            torch.from_numpy(gaussian_dict["max_static_offset"]).float().to(device)
        )
        velocity = torch.zeros_like(static_center_point)
        mpm_solver.enforce_particle_velocity_translation(
            mpm_state,
            static_center_point,
            max_static_offset,
            velocity,
            start_time=0,
            end_time=1000,
            device=device,
        )

        
        mpm_state.set_require_grad(True)

        with tape:
            # Run simulation
            for k in range(10):
                mpm_solver.p2g2p(perturbed_model, mpm_state, k, 0.001, device=device)
              
            wp.launch(
                kernel=compute_loss,
                dim = mpm_state.particle_x.shape[0],
                inputs = [mpm_state.particle_x, loss_neg],
                device = device
            )

        print(f"finish computing loss_neg {loss_neg.numpy()}")

        # Restore original value
        param_data_flat[idx] += epsilon
        print(f"idx {idx}, loss_pos {loss_pos}, loss_neg {loss_neg}")

        # Compute gradient
        grad.view(-1)[idx] = (wp.to_torch(loss_pos) - wp.to_torch(loss_neg)) / (2 * epsilon)

    return grad

# Main script
if __name__ == "__main__":
    device = "cuda:0"


    # # Initial position and velocity for a single particle
    # init_position = np.array([[0.5, 0.5, 0.5]])
    # init_velocity = np.array([[0.1, 0.0, 0.0]])
    # init_cov = np.array([[0.1, 0.0, 0.0, 0.1, 0.0, 0.1]])
    # volume_array = get_volume(init_position)

    # Initialize position and velocity from gaussians
    input_dir = "data/octopus_plush"
    gaussian_dict, scale, shift = load_gaussians(input_dir)

    velocity_scaling = 0.5
    init_velocity = velocity_scaling * gaussian_dict["velocity"]
    init_position = gaussian_dict["position"]
    init_cov = gaussian_dict["cov"]

    volume_array_path = os.path.join(input_dir, "volume_array.npy")
    if os.path.exists(volume_array_path):
        volume_array = np.load(volume_array_path)
        volume_tensor = torch.from_numpy(volume_array).float().to(device)
    else:
        volume_array = get_volume(init_position)
        np.save(volume_array_path, volume_array)
        volume_tensor = torch.from_numpy(volume_array).float().to(device)


    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # randomly select a subset of particles
    n_particles = init_position.shape[0]
    n_particles_subset_ratio = 0.01
    n_particles_subset = int(n_particles * n_particles_subset_ratio)
    subset_indices = random.sample(range(n_particles), n_particles_subset)
    init_position = init_position[subset_indices]
    init_cov = init_cov[subset_indices]
    init_velocity = init_velocity[subset_indices]
    volume_array = volume_array[subset_indices]


    print(f"check gaussian_dict: {gaussian_dict.keys()}")
    # position: np.array (N,3)
    # cov: np.array (N,6)
    # velocity: np.array (N,3)
    # static_center_point: np.array (3,) (satic_center_point) [0.2833241 , 0.33753598, 0.15322769] 
    # max_static_offset: np.array (3,)  [0.0922425 , 0.09224635, 0.03966616]
    pdb.set_trace()

    wp.init()

    tensor_init_pos = torch.from_numpy(init_position).float().to(device)
    tensor_init_cov = torch.from_numpy(init_cov).float().to(device)
    tensor_init_velocity = torch.from_numpy(init_velocity).float().to(device)
    volume_tensor = torch.from_numpy(volume_array).float().to(device)
    material_params = {
    "E": 2.0,  # 0.1-200 MPa
    "nu": 0.1,  # > 0.35
    "material": "jelly",
    # "material": "metal",
    # "friction_angle": 25,
    "g": [0.0, 0.0, 0],
    "density": 0.02,  # kg / m^3
    }

    n_particles = tensor_init_pos.shape[0]
    mpm_state = MPMStateStruct()
    mpm_state.init(init_position.shape[0], device=device, requires_grad=True)
    mpm_state.from_torch(
        tensor_init_pos,
        volume_tensor,
        tensor_init_cov,
        tensor_init_velocity,
        device=device,
        requires_grad=True,
        n_grid=200,
        grid_lim=1.0,
    )

    mpm_model = MPMModelStruct()
    mpm_model.init(n_particles, device=device, requires_grad=True)
    mpm_model.init_other_params(n_grid=200, grid_lim=1.0, device=device)

    E_tensor = (torch.ones(n_particles) * material_params["E"]).contiguous().to(device)
    nu_tensor = (
        (torch.ones(n_particles) * material_params["nu"]).contiguous().to(device)
    )
    mpm_model.from_torch(E_tensor, nu_tensor, device=device, requires_grad=True)

    mpm_solver = Custom_MPM_Simulator_WARPDiff(
        n_particles, n_grid=200, grid_lim=1.0, device=device
    )

    mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

    # set boundary conditions
    static_center_point = (
        torch.from_numpy(gaussian_dict["satic_center_point"]).float().to(device)
    )
    max_static_offset = (
        torch.from_numpy(gaussian_dict["max_static_offset"]).float().to(device)
    )
    velocity = torch.zeros_like(static_center_point)
    mpm_solver.enforce_particle_velocity_translation(
        mpm_state,
        static_center_point,
        max_static_offset,
        velocity,
        start_time=0,
        end_time=1000,
        device=device,
    )


    mpm_state.set_require_grad(True)

    loss = torch.zeros(1, device=device)
    loss = wp.from_torch(loss, requires_grad=True)


    # Autodiff Gradient Calculation
    tape = MyTape()
    with tape:
        # Run simulation
        # mpm_solver.simulate(mpm_model, perturbed_state)  
        for k in range(10):
            mpm_solver.p2g2p(mpm_model, mpm_state, k, 0.001, device=device)
            
        wp.launch(
            kernel=compute_loss,
            dim = mpm_state.particle_x.shape[0],
            inputs = [mpm_state.particle_x, loss],
            device = device
        )
        # loss = simple_loss(mpm_state)  # Compute loss
    print("Loss pre backward:", loss)
    tape.backward(loss)  # Backpropagate
    # loss.backward()  # Backpropagate

    # autodiff_grad = mpm_state.particle_v.grad
    autodiff_grad = mpm_model.E.grad
    autodiff_grad = wp.to_torch(autodiff_grad).clone().detach()

    print("Autodiff Gradients:", autodiff_grad)

    # Finite Difference Gradient Calculation
    numerical_grad = finite_difference_loss(
        mpm_solver, mpm_state, mpm_model, simple_loss, mpm_model.E
    )

    # Compare Gradients
    print("Autodiff Gradients:", autodiff_grad)
    print("Numerical Gradients:", numerical_grad)
    print("Max Difference:", torch.abs(autodiff_grad - numerical_grad).max())
