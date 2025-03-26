# %% [markdown]
# # check the operation separately without the forward and backward pass

# %%
import numpy as np
import warp as wp
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp
from datetime import datetime
import pdb

"""
Compare summation operation within a 3D grid VS summaiton operation of a flattend array
"""

today = datetime.today().strftime('%Y%m%d')
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)

data_path = "/data/ruihan/projects/PhysDreamer/physdreamer/warp_mpm/logs/20250325"
device = "cuda:0"

wp.init()

@wp.kernel
def sum_grid_v_in_flattened(
    grid_v_in: wp.array(dtype=wp.vec3d),
    loss: wp.array(dtype=wp.float64) 
):
    """
    Compute how grid velocity `grid_v_in` contributes to the loss gradient.
    """

    i = wp.tid()  # Thread index for grid

    grid_v_tensor = grid_v_in[i]

    # Accumulate the loss gradient contribution
    wp.atomic_add(loss, 0, grid_v_tensor[0])
    wp.atomic_add(loss, 0, grid_v_tensor[1])
    wp.atomic_add(loss, 0, grid_v_tensor[2])


@wp.kernel
def sum_grid_vin(
    grid_v_in: wp.array(dtype=wp.vec3d, ndim=3),
    loss: wp.array(dtype=wp.float64),
):
    """
    Compute how grid velocity `grid_v_in` contributes to the loss gradient.
    """

    i, j, k = wp.tid()  # Thread index for grid

    # Fetch the current grid velocity
    grid_v_tensor = grid_v_in[i, j, k]

    # Accumulate the loss gradient contribution
    wp.atomic_add(loss, 0, grid_v_tensor[0])
    wp.atomic_add(loss, 0, grid_v_tensor[1])
    wp.atomic_add(loss, 0, grid_v_tensor[2])


def compute_relative_error(a, b):
    if a == 0:
        if b == 0:
            rel_error = 0
        else:
            rel_error = (b - a) / b * 100
    else:
        rel_error = (b - a) / a * 100
    return rel_error

rel_error_torch_warp_list = []
rel_error_torch_numpy_list = []
rel_error_warp_numpy_list = []
sum_torch_list = []
sum_wp_list = []
sum_np_list = []
local_sum_list = []

# grid_v_in = np.load(osp.join(data_path, "grid_v_in_torch.npz"))["grid_v_in"].reshape(-1, 3)
grid_v_in = np.load(osp.join(data_path, "grid_v_in_wp.npz"))["grid_v_in"]
print(f"grid_v_in shape {grid_v_in.shape}") # (5, 5, 5, 3)

sum_flattened_list = []
sum_3d_list = []

grid_side_length = 5

# for grid_side_length in range(1, 6):
print(f"grid_side_length {grid_side_length}")
# only use data in a subset of the grid
grid_v_in_wp = grid_v_in[:grid_side_length, :grid_side_length, :grid_side_length]
grid_size = (grid_v_in_wp.shape[0], grid_v_in_wp.shape[1], grid_v_in_wp.shape[2])

grid_v_in_warp = wp.from_numpy(grid_v_in_wp, dtype=wp.vec3d, device=device)
# print(f"grid_v_in_warp type {type(grid_v_in_warp)}, dtype {grid_v_in_warp.dtype}") # type <class 'warp.types.array'>, dtype <class 'warp.types.vec3d'>
grid_v_in_warp_flattened = wp.from_numpy((grid_v_in_wp.reshape(-1, 3)), dtype=wp.vec3d, device=device)
# print(f"grid_v_in_warp_flattened shape {grid_v_in_warp_flattened.numpy().shape}")

sum_flattened = wp.zeros(1, dtype=wp.float64, device=device)
sum_3d = wp.zeros(1, dtype=wp.float64, device=device)

# compute summation of a 3D grid
wp.launch(
    kernel=sum_grid_vin,
    dim=grid_size,
    inputs=[grid_v_in_warp, sum_3d],
    device=device
)
print(f"sum_3d {sum_3d}")
sum_3d_list.append(sum_3d.numpy()[0])

# # compute summation of a flattened array
# wp.launch(
#     kernel=sum_grid_v_in_flattened,
#     dim=len(grid_v_in_warp_flattened),
#     inputs=[grid_v_in_warp_flattened, sum_flattened],
#     device=device
# )
# print(f"sum_flattened {sum_flattened}")
# sum_flattened_list.append(sum_flattened.numpy()[0])


# # plot the sum w.r.t. the grid size
# plt.plot(range(1, len(sum_3d_list)+1), sum_3d_list, label="sum_3d", marker="x")
# plt.plot(range(1, len(sum_flattened_list)+1), sum_flattened_list, label="sum_flattened", marker="o")
# plt.yscale("symlog")
# plt.xlabel("grid side length")
# plt.ylabel("sum")
# plt.legend()
# plt.title("sum_3d and sum_flattened vs grid side length")
# plt.tight_layout()
# plt.savefig(osp.join(log_dir, "sum_3d_vs_flattened.png"))
# plt.close()




