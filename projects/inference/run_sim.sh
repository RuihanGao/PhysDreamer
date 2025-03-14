# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.01 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.005 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &
# CUDA_VISIBLE_DEVICES=2 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.001 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.0005 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &

# # CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.005 --cam_id 0 --eval_ys 5000000 --hide_force --static_camera --downsample_scale 0.02

# # adjust material density
# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 5e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --impulse_mode particle --postfix _particle
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 5e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --impulse_mode grid --postfix _grid

# # generate a video hiding the force as the reference video for training
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 5e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --impulse_mode grid --postfix _grid --hide_force



# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --impulse_mode grid --postfix _grid

# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 0.02 --impulse_mode grid --postfix _grid


# # 2025-02-13 
# # When generating the reference video, we use velocity scaling = 1.0
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 5e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0
# # generate reference video with pretrained velocity field with velocity scaling = 1.0
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --force_id 0  --point_id 0 --force_mag 5e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --impulse_mode grid --postfix _grid_lv --hide_force --velo_scaling 1.0
# # checkpoint path: check checkpoint_path: ../../models/physdreamer/carnations/model

# # 2025-02-20 change the downsample scale to 0.04 to match training 
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 5e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.04 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0

# # 2025-02-23. force=1e-5, use filtered downsample point cloud for foreground. get rid of downsampling process
# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0


# # 2025-02-24 generate reference video with other initE to check expected pos difference
# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0

# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0

# # 2025-02-25 Generate referece video with IMPULSE force, where force is only applied before the first frame (fps=30)
# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &


# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &


# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-3 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-3 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-3 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &


# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-2 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-2 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-2 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --hide_force --velo_scaling 1.0 --force_duration 0.03 &




# # 2025-02-26 Generate a reference video with pretrained velocity field + uniform material + clean downsampled point cloud
# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid_lv --hide_force --velo_scaling 1.0



# Generate a reference video with "instantaneous" force
# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 0.03 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 0.03 &



# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-4 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-4 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-4

# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-6 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5

# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-5 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5

# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
# CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 1e-4 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5



# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.1 --cam_id 0 --eval_ys 1000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.1 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 &
CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.1 --cam_id 0 --eval_ys 8000000 --static_camera --downsample_scale 1.0 --impulse_mode grid --postfix _grid --velo_scaling 1.0 --force_duration 4.3e-5 
