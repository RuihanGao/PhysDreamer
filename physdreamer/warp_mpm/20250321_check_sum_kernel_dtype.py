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
grid_side_length = 5

# %%
# load the grid_v_in data from torch and warp respectively
grid_v_in_torch = np.load(osp.join(data_path, "grid_v_in_torch.npz"))["grid_v_in"]
grid_v_in_torch = grid_v_in_torch[:grid_side_length, :grid_side_length, :grid_side_length]
grid_v_in_torch = torch.from_numpy(grid_v_in_torch)
print(f"check grid_v_in_torch shape {grid_v_in_torch.shape}, dtype {grid_v_in_torch.dtype}")


# %%
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

# %%
grid_v_in_warp = np.load(osp.join(data_path, "grid_v_in_wp.npz"))["grid_v_in"]
grid_v_in_warp = grid_v_in_warp[:grid_side_length, :grid_side_length, :grid_side_length]
print(f"shape {grid_v_in_warp.shape}")
grid_size = (grid_v_in_warp.shape[0], grid_v_in_warp.shape[1], grid_v_in_warp.shape[2])
print(f"grid_size {grid_size}")
grid_v_in_warp = wp.from_numpy(grid_v_in_warp, dtype=wp.vec3d) # dtype <class 'warp.types.vec3d'>, type <class 'warp.types.array'>

# %%
# compute torch sum
sum_torch = torch.sum(grid_v_in_torch)
# compute warp sum
sum_wp = wp.zeros(1, dtype=wp.float64)



wp.launch(
    kernel=sum_grid_v_in,
    dim=grid_size,
    inputs=[grid_v_in_warp, sum_wp],
) 

print(f"sum_wp {sum_wp}")
print(f"sum_torch {sum_torch}")

# %%

