# %% [markdown]
# # check the operation separately without the forward and backward pass

# %%
import numpy as np
import warp as wp
import torch
import matplotlib.pyplot as plt
import os
import os.path as osp

# %%
from datetime import datetime
today = datetime.today().strftime('%Y%m%d')
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)

data_path = "/data/ruihan/projects/PhysDreamer/physdreamer/warp_mpm/logs/20250321"
device = "cuda:0"

wp.init()

@wp.kernel
def sum_grid_v_in(
    grid_v_in: wp.array(dtype=wp.vec3d, ndim=3),
    loss: wp.array(dtype=wp.float64) 
):
    """
    Compute how grid velocity `grid_v_in` contributes to the loss gradient.
    """

    i, j, k = wp.tid()  # Thread index for grid

    grid_v_tensor = grid_v_in[i, j, k]


    # Accumulate the loss gradient contribution
    wp.atomic_add(loss, 0, grid_v_tensor[0])
    wp.atomic_add(loss, 0, grid_v_tensor[1])
    wp.atomic_add(loss, 0, grid_v_tensor[2])


rel_error_list = []
sum_torch_list = []
sum_wp_list = []

for grid_side_length in range(1, 6):
    # %%
    # load the grid_v_in data from torch and warp respectively
    grid_v_in_torch = np.load(osp.join(data_path, "grid_v_in_torch.npz"))["grid_v_in"]
    grid_v_in_torch = grid_v_in_torch[:grid_side_length, :grid_side_length, :grid_side_length]
    grid_v_in_torch = torch.from_numpy(grid_v_in_torch).to(device)
    print(f"check grid_v_in_torch shape {grid_v_in_torch.shape}, dtype {grid_v_in_torch.dtype}")


    # %%
    grid_v_in_warp = np.load(osp.join(data_path, "grid_v_in_wp.npz"))["grid_v_in"]
    # grid_v_in_warp = np.load(osp.join(data_path, "grid_v_in_torch.npz"))["grid_v_in"]
    grid_v_in_warp = grid_v_in_warp[:grid_side_length, :grid_side_length, :grid_side_length]
    print(f"shape {grid_v_in_warp.shape}")
    grid_size = (grid_v_in_warp.shape[0], grid_v_in_warp.shape[1], grid_v_in_warp.shape[2])
    print(f"grid_size {grid_size}")
    grid_v_in_warp = wp.from_numpy(grid_v_in_warp, dtype=wp.vec3d, device=device) # dtype <class 'warp.types.vec3d'>, type <class 'warp.types.array'>

    # %%
    # compute torch sum
    sum_torch = torch.sum(grid_v_in_torch)
    # compute warp sum
    sum_wp = wp.zeros(1, dtype=wp.float64, device=device)



    wp.launch(
        kernel=sum_grid_v_in,
        dim=grid_size,
        inputs=[grid_v_in_warp, sum_wp],
        device=device
    ) 

    sum_wp = sum_wp.numpy()[0]
    sum_torch = sum_torch.item()
    print(f"sum_wp {sum_wp}, sum_torch {sum_torch}")
    # compute a relative error
    rel_error = (sum_wp - sum_torch) / sum_torch * 100
    print(f"rel_error {rel_error}")

    rel_error_list.append(rel_error)
    sum_torch_list.append(sum_torch)
    sum_wp_list.append(sum_wp)

# plot the relative error w.r.t. the grid size
plt.plot(range(1, 6), rel_error_list)
plt.yscale("symlog")
plt.xlabel("grid size")
plt.ylabel("relative error (%)")
plt.title("relative error vs grid size")
plt.savefig(osp.join(log_dir, "check_sum_kernel_dtype_relative.png"))
plt.close()

# plot the value of sum_wp and sum_torch w.r.t. the grid size. use log scale for y axis
plt.scatter(range(1, 6), sum_torch_list, label="sum_torch")
plt.scatter(range(1, 6), sum_wp_list, label="sum_wp")
plt.yscale("symlog")
plt.xlabel("grid size")
plt.ylabel("sum")
plt.legend()
plt.title("sum_wp and sum_torch vs grid size")
plt.savefig(osp.join(log_dir, "check_sum_kernel_dtype_sum.png"))
plt.close()

