# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

# create a log directory by date
from datetime import datetime
today = datetime.today().strftime('%Y%m%d')
log_dir = f"logs/{today}"
os.makedirs(log_dir, exist_ok=True)



# %%
# Define your 3x3x3 matrices (these are just examples, replace them with your actual matrices)

# ### dt = 1e-1 ###
# matrix_1 = np.array([[[ 0.0304, -0.0012,  0.0049],
#                      [ 0.0227, -0.0009,  0.0037],
#                      [ 0.0230, -0.0009,  0.0037]]])

# matrix_2 = np.array([[[ 0.0304, -0.0012,  0.0049],
#                      [ 0.0227, -0.0009,  0.0037],
#                      [ 0.0230, -0.0009,  0.0037]]])

# matrix_3 = np.array([[[ 0.0304, -0.0012,  0.0049],
#                      [ 0.0227, -0.0009,  0.0037],
#                      [ 0.0230, -0.0009,  0.0037]]])


### dt = 1e-3 ###
matrix_1 = np.array([[[ 3.0404e-04, -1.2194e-05,  4.8954e-05],
         [ 2.2716e-04, -9.0193e-06,  3.6645e-05],
         [ 2.3019e-04, -9.2209e-06,  3.7069e-05]]])


matrix_2 = np.array([[[ 3.0421e-04, -1.0512e-05,  4.8872e-05],
         [ 2.2705e-04, -8.8334e-06,  3.6883e-05],
         [ 2.3040e-04, -7.7336e-06,  3.7038e-05]]])


matrix_3 = np.array([[[ 3.0684e-04, -1.0512e-05,  4.9283e-05],
         [ 2.3126e-04, -9.2983e-06,  3.5694e-05],
         [ 2.2588e-04,  5.9489e-06,  3.8743e-05]]])




# Stack matrices into a single 3D array for easier processing
matrices = np.stack([matrix_1, matrix_2, matrix_3], axis=0).squeeze()
print(f"check matrices shape: {matrices.shape}")

# Set up the figure with 3x3 subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Loop through each subplot and each entry in the 3x3 matrices
for i in range(3):
    for j in range(3):
        # Extract the values of the current entry across the three matrices
        values = matrices[:, i, j]
        
        # Plot the variation of the entry across the three matrices
        axs[i, j].plot([1, 2, 3], values, marker='o')
        
        # Set labels and title for the subplot
        axs[i, j].set_title(f"Entry ({i+1}, {j+1})")
        axs[i, j].set_xticks([1, 2, 3])
        axs[i, j].set_xticklabels(['eps 1e-5', 'eps 1e-6', 'eps 1e-7'])
        axs[i, j].set_ylabel('Value')

# Adjust the layout and show the plot
plt.tight_layout()
plt.savefig(f"{log_dir}/FD_gradient_sensitive2dt_1e-3.png")
plt.show()


# %%
# Draw a plot showing the sensitivity to dt for each perturbation
relative_error = {0.1: [2.3167537975921708e-07, 2.599955257935366e-06, 1.9179839043873368e-05, 0.00022981775703404954],
                  1e-4: [0.00023036421564328263, 0.0015130722245136697, 0.01993903876961619, 0.22159959754365474]}
perturbation_list = [1e-4, 1e-5, 1e-6, 1e-7]

# relative_error = {1e-4: [0.22159959754365474, 0.01993903876961619, 0.0015130722245136697, 0.00023036421564328263, 3.165412893528505e-05, 1.596435933212839e-06, 1.7432400554728576e-07, 2.0315638019750444e-08, 1.3545093514876725e-09, 2.0269608250837206e-10, 3.183474060020554e-11, 1.5567998790129007e-12, 1.4956132827250622e-13, 2.6897811786157334e-14, 2.255338784691201e-15, 1.7786762281922374e-16, 8.116432705736976e-16, 3.4155696594530477e-15, 7.704317013345693e-14, 6.338965563667228e-13, 3.715291165765527e-12, 2.4349919369033386e-11, 2.871809895183311e-10, 4.348467464753782e-09, 1.5526355925273765e-08, 3.5004697599126e-07, 4.668722256042711e-06]}
# perturbation_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000, 10000000000000000, 100000000000000000, 1000000000000000000, 10000000000000000000]


for dt, errors in relative_error.items():
    print(f"check list lenth: {len(errors)}, {len(perturbation_list)}")
    plt.plot(perturbation_list, errors, marker='o', label=f"dt = {dt}")
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Perturbation ')
plt.ylabel('Relative Error')
plt.title('Sensitivity of Relative Error to Perturbation for Different dt')
plt.legend()
plt.tight_layout()
plt.savefig(f"{log_dir}/FD_gradient_sensitive2dt_relative_error.png")
plt.close()


# %%
