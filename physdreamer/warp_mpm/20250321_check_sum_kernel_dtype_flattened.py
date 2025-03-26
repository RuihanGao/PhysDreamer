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

data_path = "/data/ruihan/projects/PhysDreamer/physdreamer/warp_mpm/logs/20250324"
device = "cuda:0"

wp.init()

@wp.kernel
def sum_grid_v_in(
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

grid_v_in = np.load(osp.join(data_path, "grid_v_in_torch.npz"))["grid_v_in"].reshape(-1, 3)
grid_v_in_wp = np.load(osp.join(data_path, "grid_v_in_wp.npz"))["grid_v_in"].reshape(-1, 3)


print(f"grid_v_in shape {grid_v_in.shape}")

for i in range(1, len(grid_v_in)+1):
    grid_v_in_np = grid_v_in[:i]
    
    grid_v_in_torch = torch.from_numpy(grid_v_in_np).to(device)
    # grid_v_in_warp = wp.from_torch(grid_v_in_torch, dtype=wp.vec3d)
    grid_v_in_warp = wp.from_numpy(grid_v_in_wp[:i], dtype=wp.vec3d, device=device)

    # compute torch sum
    sum_torch = torch.sum(grid_v_in_torch)
    # compute warp sum
    sum_wp = wp.zeros(1, dtype=wp.float64, device=device)

    wp.launch(
        kernel=sum_grid_v_in,
        dim=i,
        inputs=[grid_v_in_warp, sum_wp],
        device=device
    ) 

    sum_wp = sum_wp.numpy()[0]
    sum_torch = sum_torch.item()
    sum_np = np.sum(grid_v_in_np)
    print(f"sum_wp {sum_wp}, sum_torch {sum_torch}")
    # compute a relative error
    rel_error_torch_warp = compute_relative_error(sum_wp, sum_torch)
    print(f"rel_error {rel_error_torch_warp}")

    local_sum_list.append(grid_v_in[i-1, 0]+grid_v_in[i-1, 1]+grid_v_in[i-1, 2])

    rel_error_torch_warp_list.append(rel_error_torch_warp)
    rel_error_torch_numpy_list.append(compute_relative_error(sum_np, sum_torch))
    rel_error_warp_numpy_list.append(compute_relative_error(sum_np, sum_wp))

    sum_torch_list.append(sum_torch)
    sum_wp_list.append(sum_wp)
    sum_np_list.append(sum_np)

# plot the relative error w.r.t. the grid size
# cconcatenate the following two plots into one as subplots, sharing the same x axis
fig, axes = plt.subplots(5, 1, figsize=(6, 15))
axes[0].plot(range(1, len(grid_v_in)+1), rel_error_torch_warp_list)
axes[0].set_yscale("symlog")
axes[0].set_xlabel("# elements")
axes[0].set_ylabel("relative error (%)")
axes[0].set_title("relative error - torch vs warp")

# plot relative error between torch and numpy, warp and numpy
axes[1].plot(range(1, len(grid_v_in)+1), rel_error_torch_numpy_list)
axes[1].set_yscale("symlog")
axes[1].set_xlabel("# elements")
axes[1].set_ylabel("relative error (%)")
axes[1].set_title("relative error - torch vs numpy")

axes[2].plot(range(1, len(grid_v_in)+1), rel_error_warp_numpy_list)
axes[2].set_yscale("symlog")
axes[2].set_xlabel("# elements")
axes[2].set_ylabel("relative error (%)")
axes[2].set_title("relative error - warp vs numpy")


axes[3].scatter(range(1, len(grid_v_in)+1), sum_torch_list, label="sum_torch", alpha=0.5, marker="x")
axes[3].scatter(range(1, len(grid_v_in)+1), sum_wp_list, label="sum_wp", alpha=0.5, marker="o")
axes[3].scatter(range(1, len(grid_v_in)+1), sum_np_list, label="sum_np", alpha=0.5, marker="s")
axes[3].set_yscale("symlog")
axes[3].set_xlabel("# elements")
axes[3].set_ylabel("sum")
axes[3].legend()
axes[3].set_title("sum_wp and sum_torch vs #elements")

# plot grid_v_in
axes[4].scatter(range(1, len(grid_v_in)+1), local_sum_list)
axes[4].set_yscale("symlog")
axes[4].set_xlabel("element index")
axes[4].set_ylabel("element sum")
axes[4].set_title("element sum vs element index")

plt.tight_layout()
plt.savefig(osp.join(log_dir, "check_sum_kernel_dtype_flattened_all_fullapic.png"))
plt.close()

