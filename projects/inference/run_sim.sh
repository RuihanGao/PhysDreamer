CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.01 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &
CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.005 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &
CUDA_VISIBLE_DEVICES=2 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.001 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &
CUDA_VISIBLE_DEVICES=3 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.0005 --cam_id 0 --eval_ys 100000 --hide_force --static_camera &

# CUDA_VISIBLE_DEVICES=0 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.005 --cam_id 0 --eval_ys 5000000 --hide_force --static_camera --downsample_scale 0.02

# adjust material density
# CUDA_VISIBLE_DEVICES=1 python3 demo.py --scene_name carnation --apply_force --force_id 0  --point_id 0 --force_mag 0.05 --cam_id 0 --eval_ys 5000000 --static_camera --downsample_scale 0.02 --mat_density 500 --postfix _particle_den_500