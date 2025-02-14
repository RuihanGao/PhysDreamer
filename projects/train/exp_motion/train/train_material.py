import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor
from jaxtyping import Float, Int, Shaped
from typing import List

import point_cloud_utils as pcu

from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

import numpy as np
import logging
import argparse
import shutil
import wandb
import torch
import os
import pdb
# add the path to import motionrep
import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(os.getcwd()))
sys.path.append(parent_dir)
physdreamer_dir = osp.dirname(osp.dirname(parent_dir))
sys.path.append(physdreamer_dir)
print(f"add {parent_dir} and {physdreamer_dir} to sys path")

from motionrep.utils.config import create_config
from motionrep.utils.optimizer import get_linear_schedule_with_warmup
from time import time
from omegaconf import OmegaConf
from PIL import Image
import imageio
import numpy as np

# from motionrep.utils.torch_utils import get_sync_time
from einops import rearrange, repeat

from motionrep.gaussian_3d.gaussian_renderer.feat_render import render_feat_gaussian
from motionrep.gaussian_3d.scene import GaussianModel
from motionrep.fields.se3_field import TemporalKplanesSE3fields

from motionrep.data.datasets.multiview_dataset import MultiviewImageDataset
from motionrep.data.datasets.multiview_video_dataset import (
    MultiviewVideoDataset,
    camera_dataset_collate_fn,
)

from motionrep.data.datasets.multiview_dataset import (
    camera_dataset_collate_fn as camera_dataset_collate_fn_img,
)

from typing import NamedTuple
import torch.nn.functional as F

from motionrep.utils.img_utils import compute_psnr, compute_ssim

from physdreamer.warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
)
from physdreamer.warp_mpm.mpm_solver_diff import MPMWARPDiff
from physdreamer.warp_mpm.gaussian_sim_utils import get_volume
import warp as wp
import random

from physdreamer.local_utils import (
    cycle,
    load_motion_model,
    create_motion_model,
    create_spatial_fields,
    find_far_points,
    LinearStepAnneal,
    apply_grid_bc_w_freeze_pts,
    render_gaussian_seq_w_mask_cam_seq,
    downsample_with_kmeans_gpu,
    render_gaussian_seq_w_mask_with_disp,
    render_gaussian_seq_w_mask_cam_seq_with_force_with_disp,
    add_constant_force,
)
from interface import (
    MPMDifferentiableSimulationWCheckpoint,
    MPMDifferentiableSimulationClean,
)
from motionrep.utils.io_utils import save_video_imageio, save_gif_imageio

logger = get_logger(__name__, log_level="INFO")

model_dict = {
    # psnr: 29.9
    # "videos": "../../output/inverse_sim/fast_hat_velopretraindecay_1.0_substep_96_se3_field_lr_0.001_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000299",
    # psnr: 30.25
    "videos": "../../output/inverse_sim/fast_hat_velopretrain_g48-192decay_1.0_substep_192_se3_field_lr_0.003_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000199",
    # psnr: 30.52
    "videos_2": "../../output/inverse_sim/fast_hat_videos2_velopretraindecay_1.0_substep_96_se3_field_lr_0.003_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000199",
}


def create_dataset(args):
    assert args.dataset_res in ["middle", "small", "large"]
    if args.dataset_res == "middle":
        res = [320, 576]
    elif args.dataset_res == "small":
        res = [192, 320]
    elif args.dataset_res == "large":
        res = [576, 1024]
    else:
        raise NotImplementedError

    video_dir_name = "videos"
    video_dir_name = args.video_dir_name

    if args.test_convergence:
        video_dir_name = "simulated_videos"
    dataset = MultiviewVideoDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        scale_x_angle=1.0,
        video_dir_name=video_dir_name,
    )

    test_dataset = MultiviewImageDataset(
        args.dataset_dir,
        use_white_background=False,
        resolution=res,
        # use_index=list(range(0, 30, 4)),
        # use_index=[0],
        scale_x_angle=1.0,
        fitler_with_renderd=False,
        load_imgs=False,
    )
    print("len of test dataset", len(test_dataset))
    return dataset, test_dataset


class Trainer:
    def __init__(self, args):
        self.args = args

        self.ssim = args.ssim
        args.warmup_step = int(args.warmup_step * args.gradient_accumulation_steps)
        args.train_iters = int(args.train_iters * args.gradient_accumulation_steps)
        os.environ["WANDB__SERVICE_WAIT"] = "600"
        # args.wandb_name += (
        #     "decay_{}_substep_{}_{}_lr_{}_tv_{}_iters_{}_sw_{}_cw_{}".format(
        #         args.loss_decay,
        #         args.substep,
        #         args.model,
        #         args.lr,
        #         args.tv_loss_weight,
        #         args.train_iters,
        #         args.start_window_size,
        #         args.compute_window,
        #     )
        # )

        # omit default params to save space
        args.wandb_name += (
            "_lr_{}".format(
                args.lr,
            )
        )

        args.wandb_name = args.wandb_name + args.postfix
        logging_dir = os.path.join(args.output_dir, args.wandb_name)
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,  # args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        set_seed(args.seed + accelerator.process_index)
        print("process index", accelerator.process_index)
        if accelerator.is_main_process:
            output_path = os.path.join(logging_dir, f"seed{args.seed}")
            os.makedirs(output_path, exist_ok=True)
            self.output_path = output_path

        self.rand_bg = args.rand_bg
        # setup the dataset
        dataset, test_dataset = create_dataset(args)
        self.test_dataset = test_dataset

        dataset_dir = test_dataset.data_dir
        self.dataset = dataset

        gaussian_path = os.path.join(dataset_dir, "point_cloud.ply")
        aabb = self.setup_eval(
            args,
            gaussian_path,
            white_background=True,
        )
        self.aabb = aabb
        self.model = create_motion_model(
            args,
            aabb=aabb,
            num_frames=9,
        )
        if args.motion_model_path is not None:
            self.model = load_motion_model(self.model, args.motion_model_path)
        self.model.eval()

        self.num_frames = int(args.num_frames)
        self.window_size_schduler = LinearStepAnneal(
            args.train_iters,
            start_state=[args.start_window_size],
            end_state=[13],
            plateau_iters=-1,
            warmup_step=20,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=camera_dataset_collate_fn_img,
        )
        # why prepare here again?
        test_dataloader = accelerator.prepare(test_dataloader)
        self.test_dataloader = cycle(test_dataloader)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=camera_dataset_collate_fn,
        )
        # why prepare here again?
        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        self.train_iters = args.train_iters
        self.accelerator = accelerator
        # init traiable params
        self.initE = args.initE
        E_nu_list = self.init_trainable_params()
        for p in E_nu_list:
            p.requires_grad = True
        self.E_nu_list = E_nu_list
        print(f"check E_nu_list: {E_nu_list}")

        self.model = accelerator.prepare(self.model)
        self.setup_simulation(dataset_dir, grid_size=args.grid_size)

        if args.checkpoint_path == "None":
            args.checkpoint_path = None
        if args.checkpoint_path is not None:
            if args.video_dir_name in model_dict:
                args.checkpoint_path = model_dict[args.video_dir_name]
            self.load(args.checkpoint_path)
            trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list
            optim_list = [
                {"params": self.E_nu_list, "lr": args.lr * 1e-10},
                {
                    "params": self.sim_fields.parameters(),
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                },
                # {"params": self.velo_fields.parameters(), "lr": args.lr * 1e-3, "weight_decay": 1e-4},
            ]

            if args.update_velo:
                self.freeze_velo = False
                velo_optim = [
                    {
                        "params": self.velo_fields.parameters(),
                        "lr": args.lr * 1e-4,
                        "weight_decay": 1e-4,
                    },
                ]
                self.velo_optimizer = torch.optim.AdamW(
                    velo_optim,
                    lr=args.lr,
                    weight_decay=0.0,
                )
                self.velo_scheduler = get_linear_schedule_with_warmup(
                    optimizer=self.velo_optimizer,
                    num_warmup_steps=args.warmup_step,
                    num_training_steps=args.train_iters,
                )
            else:
                self.freeze_velo = True
                self.velo_optimizer = None
        else:
            trainable_params = list(self.sim_fields.parameters()) + self.E_nu_list
            optim_list = [
                {"params": self.E_nu_list, "lr": args.lr * 1e-10},
                {
                    "params": self.sim_fields.parameters(),
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                },
            ]
            self.freeze_velo = False
            self.window_size_schduler.warmup_step = 800

            velo_optim = [
                {
                    "params": self.velo_fields.parameters(),
                    "lr": args.lr,
                    "weight_decay": 1e-4,
                },
            ]
            self.velo_optimizer = torch.optim.AdamW(
                velo_optim,
                lr=args.lr,
                weight_decay=0.0,
            )
            self.velo_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.velo_optimizer,
                num_warmup_steps=args.warmup_step,
                num_training_steps=args.train_iters // 3,
            )
            self.velo_optimizer, self.velo_scheduler = accelerator.prepare(
                self.velo_optimizer, self.velo_scheduler
            )

        self.optimizer = torch.optim.AdamW(
            optim_list,
            lr=args.lr,
            weight_decay=0.0,
        )
        self.trainable_params = trainable_params
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )
        self.sim_fields, self.optimizer, self.scheduler = accelerator.prepare(
            self.sim_fields, self.optimizer, self.scheduler
        )
        self.velo_fields = accelerator.prepare(self.velo_fields)

        # setup train info
        self.step = 0
        self.batch_size = args.batch_size
        self.tv_loss_weight = args.tv_loss_weight

        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.max_grad_norm = args.max_grad_norm

        self.use_wandb = args.use_wandb
        if self.accelerator.is_main_process:
            if args.use_wandb:
                run = wandb.init(
                    config=dict(args),
                    dir=self.output_path,
                    **{
                        "mode": "online",
                        # "entity": args.wandb_entity,
                        "project": args.wandb_project,
                    },
                )

                # self.wandb_run = wandb.init(project="SKIT", name=opt.name, config=opt) if not wandb.run else wandb.run
                # self.wandb_run._label(repo="SKIT")

                wandb.run.log_code(".")
                wandb.run.name = args.wandb_name
                print(f"run dir: {run.dir}")
                self.wandb_folder = run.dir
                os.makedirs(self.wandb_folder, exist_ok=True)

        
        self.apply_force = args.apply_force
        if self.apply_force:
            # TODO: combine these to a single self.force_config
            self.center_point = args.center_point
            self.force_dir = args.force_dir
            self.force_mag = args.force_mag
            self.force_duration = args.force_duration
            self.force_radius = args.force_radius
            

    def init_trainable_params(
        self,
    ):

        # init young modulus and poisson ratio

        young_numpy = np.exp(np.random.uniform(np.log(1e-3), np.log(1e3))).astype(
            np.float32
        )

        young_numpy = np.array([self.initE]).astype(np.float32)

        young_modulus = torch.tensor(young_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        poisson_numpy = np.random.uniform(0.1, 0.4)
        poisson_ratio = torch.tensor(poisson_numpy, dtype=torch.float32).to(
            self.accelerator.device
        )

        trainable_params = [young_modulus, poisson_ratio]

        print(
            "init young modulus: ",
            young_modulus.item(),
            "poisson ratio: ",
            poisson_ratio.item(),
        )
        return trainable_params

    def setup_simulation(self, dataset_dir, grid_size=100):

        device = "cuda:{}".format(self.accelerator.process_index)

        xyzs = self.render_params.gaussians.get_xyz.detach().clone()
        sim_xyzs = xyzs[self.sim_mask_in_raw_gaussian, :]
        sim_cov = (
            self.render_params.gaussians.get_covariance()[
                self.sim_mask_in_raw_gaussian, :
            ]
            .detach()
            .clone()
        )

        # scale, and shift
        pos_max = sim_xyzs.max()
        pos_min = sim_xyzs.min()
        scale = (pos_max - pos_min) * 1.8
        shift = -pos_min + (pos_max - pos_min) * 0.25
        self.scale, self.shift = scale, shift
        print("scale, shift", scale, shift)

        # filled
        filled_in_points_path = os.path.join(dataset_dir, "internal_filled_points.ply")

        if os.path.exists(filled_in_points_path):
            fill_xyzs = pcu.load_mesh_v(filled_in_points_path)  # [n, 3]
            fill_xyzs = fill_xyzs[
                np.random.choice(
                    fill_xyzs.shape[0], int(fill_xyzs.shape[0] * 0.25), replace=False
                )
            ]
            fill_xyzs = torch.from_numpy(fill_xyzs).float().to("cuda")
            self.fill_xyzs = fill_xyzs
            print(
                "loaded {} internal filled points from: ".format(fill_xyzs.shape[0]),
                filled_in_points_path,
            )
        else:
            self.fill_xyzs = None

        if self.fill_xyzs is not None:
            render_mask_in_sim_pts = torch.cat(
                [
                    torch.ones_like(sim_xyzs[:, 0]).bool(),
                    torch.zeros_like(fill_xyzs[:, 0]).bool(),
                ],
                dim=0,
            ).to(device)
            sim_xyzs = torch.cat([sim_xyzs, fill_xyzs], dim=0)
            sim_cov = torch.cat(
                [sim_cov, sim_cov.new_ones((fill_xyzs.shape[0], sim_cov.shape[-1]))],
                dim=0,
            )
            self.render_mask = render_mask_in_sim_pts
        else:
            self.render_mask = torch.ones_like(sim_xyzs[:, 0]).bool().to(device)

        sim_xyzs = (sim_xyzs + shift) / scale

        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        # point cloud resample with kmeans

        downsample_scale = self.args.downsample_scale
        num_cluster = int(sim_xyzs.shape[0] * downsample_scale)
        print(f"Before downsample: {sim_xyzs.shape[0]}, num_cluster: {num_cluster}")
        sim_xyzs = downsample_with_kmeans_gpu(sim_xyzs, num_cluster) # downsample the simulated particles for faster simulation


        sim_gaussian_pos = self.render_params.gaussians.get_xyz.detach().clone()[
            self.sim_mask_in_raw_gaussian, :
        ]
        sim_gaussian_pos = (sim_gaussian_pos + shift) / scale

        cdist = torch.cdist(sim_gaussian_pos, sim_xyzs) * -1.0
        _, top_k_index = torch.topk(cdist, self.args.top_k, dim=-1)
        self.top_k_index = top_k_index

        print("Downsampled to: ", sim_xyzs.shape[0], "by", downsample_scale)

        points_volume = get_volume(sim_xyzs.detach().cpu().numpy())

        num_particles = sim_xyzs.shape[0]

        sim_aabb = torch.stack(
            [torch.min(sim_xyzs, dim=0)[0], torch.max(sim_xyzs, dim=0)[0]], dim=0
        )
        sim_aabb = (
            sim_aabb - torch.mean(sim_aabb, dim=0, keepdim=True)
        ) * 1.2 + torch.mean(sim_aabb, dim=0, keepdim=True)

        print("simulation aabb: ", sim_aabb)

        wp.init()
        wp.config.mode = "debug"
        wp.config.verify_cuda = True

        mpm_state = MPMStateStruct()
        mpm_state.init(num_particles, device=device, requires_grad=True)

        self.particle_init_position = sim_xyzs.clone()

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            torch.from_numpy(points_volume).float().to(device).clone(),
            sim_cov,
            device=device,
            requires_grad=True,
            n_grid=grid_size,
            grid_lim=1.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(num_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=1.0, device=device)

        material_params = {
            "material": "jelly",  # "jelly", "metal", "sand", "foam", "snow", "plasticine", "neo-hookean"
            "g": [0.0, 0.0, 0.0],
            "density": 200,  # kg / m^3
            "grid_v_damping_scale": 1.1,  # 0.999,
        }

        self.v_damping = material_params["grid_v_damping_scale"]
        self.material_name = material_params["material"]
        mpm_solver = MPMWARPDiff(
            num_particles, n_grid=grid_size, grid_lim=1.0, device=device
        )
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        # setup boundary condition:
        moving_pts_path = os.path.join(dataset_dir, "moving_part_points.ply")
        if os.path.exists(moving_pts_path):
            moving_pts = pcu.load_mesh_v(moving_pts_path)
            moving_pts = torch.from_numpy(moving_pts).float().to(device)
            moving_pts = (moving_pts + shift) / scale
            freeze_mask = find_far_points(
                sim_xyzs, moving_pts, thres=0.5 / grid_size
            ).bool()
            freeze_pts = sim_xyzs[freeze_mask, :]

            grid_freeze_mask = apply_grid_bc_w_freeze_pts(
                grid_size, 1.0, freeze_pts, mpm_solver
            )
            self.freeze_mask = freeze_mask

            # does not prefer boundary condition on particle
            # freeze_mask_select = setup_boundary_condition_with_points(sim_xyzs, moving_pts,
            #                                                         self.mpm_solver, self.mpm_state, thres=0.5 / grid_size)
            # self.freeze_mask = freeze_mask_select.bool()
        else:
            raise NotImplementedError

        num_freeze_pts = self.freeze_mask.sum()
        print(
            "num freeze pts in total",
            num_freeze_pts.item(),
            "num moving pts",
            num_particles - num_freeze_pts.item(),
        )

        # init fields for simulation, e.g. density, external force, etc.

        # padd init density, youngs,
        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * material_params["density"]
        )
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.E_nu_list[0].detach()
        )
        poisson_ratio = torch.ones_like(self.particle_init_position[..., 0]) * 0.3

        # load stem for higher density
        stem_pts_path = os.path.join(dataset_dir, "stem_points.ply")
        if os.path.exists(stem_pts_path):
            stem_pts = pcu.load_mesh_v(stem_pts_path)
            stem_pts = torch.from_numpy(stem_pts).float().to(device)
            stem_pts = (stem_pts + shift) / scale
            no_stem_mask = find_far_points(
                sim_xyzs, stem_pts, thres=2.0 / grid_size
            ).bool()
            stem_mask = torch.logical_not(no_stem_mask)
            density[stem_mask] = 2000
            print("num stem pts", stem_mask.sum().item())

        self.density = density
        self.young_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio

        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus.clone(), poisson_ratio.clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

        self.sim_fields = create_spatial_fields(self.args, 1, sim_aabb)
        self.sim_fields.train()

        self.args.sim_res = 24
        # self.velo_fields = create_velocity_model(self.args, sim_aabb)
        self.velo_fields = create_spatial_fields(
            self.args, 3, sim_aabb, add_entropy=False
        )
        self.velo_fields.train()

    def get_simulation_input(self, device, delta_time=1.0 / 30, impulse_mode="grid"):
        """
        Outs: All padded
            density: [N]
            young_modulus: [N]
            poisson_ratio: [N]
            velocity: [N, 3]
            query_mask: [N]
        """

        # Get material params
        density, youngs_modulus, ret_poisson, entropy = self.get_material_params(device)
        initial_position_time0 = self.particle_init_position.clone()

        query_mask = torch.logical_not(self.freeze_mask)
        query_pts = initial_position_time0[query_mask, :]

        # Get velocity
        if self.apply_force:
            # Instead of loading velocity from velocity field, we set initial velocity to zero and apply force to the particles
            ret_velocity = torch.zeros_like(initial_position_time0)

            center_point = (
                torch.from_numpy(np.array(self.center_point)).to(device).float()
            )
            force = torch.from_numpy(np.array(self.force_dir)*self.force_mag).to(device).float()

            force_duration = self.force_duration  # sec
            force_duration_steps = int(force_duration / delta_time)

            # apply force to points within the radius of the center point
            force_radius = self.force_radius

            # add constant force
            xyzs = self.particle_init_position.clone() * self.scale - self.shift
            print(f"add constant force: center_point {center_point}, force_radius {force_radius}, force {force}, delta_time {delta_time}, force_duration {force_duration}")
            add_constant_force(
                self.mpm_solver,
                self.mpm_state,
                xyzs,
                center_point,
                force_radius,
                force,
                delta_time,
                0.0,
                force_duration,
                device=device,
                impulse_mode=impulse_mode
            )

            # prepare to render force in simulated videos:
            #   find the closest point to the force center, and will use it to render the force
            xyzs = self.render_params.gaussians.get_xyz.detach().clone()
            dist = torch.norm(xyzs - center_point.unsqueeze(dim=0), dim=-1)
            closest_idx = torch.argmin(dist)
            closest_xyz = xyzs[closest_idx, :]
            render_force = force / force.norm() * 0.1
            # print(f"check particle_velo after apply_force") # all zeros here. only append function to pre_p2g_operations here
            # pdb.set_trace()     


        else:
            # velocity = self.velo_fields(torch.cat([query_pts, time_array.unsqueeze(-1)], dim=-1))[..., :3]
            velocity = self.velo_fields(query_pts)[..., :3]

            # scaling
            velocity = velocity * 0.1  # not padded yet # TODO: why scale the velocity by 0.1?

            ret_velocity = torch.zeros_like(initial_position_time0)
            ret_velocity[query_mask, :] = velocity

        # init F as Idensity Matrix, and C and Zero Matrix
        I_mat = torch.eye(3, dtype=torch.float32).to(device)
        particle_F = torch.repeat_interleave(
            I_mat[None, ...], initial_position_time0.shape[0], dim=0
        )
        particle_C = torch.zeros_like(particle_F)
        

        return (
            density,
            youngs_modulus,
            ret_poisson,
            ret_velocity,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        )

    def get_material_params(self, device):

        initial_position_time0 = self.particle_init_position.detach()

        # query_mask = torch.logical_not(self.freeze_mask)
        query_mask = torch.ones_like(self.freeze_mask).bool()
        query_pts = initial_position_time0[query_mask, :]
        if self.args.entropy_cls > 0:
            sim_params, entropy = self.sim_fields(query_pts)
        else:
            sim_params = self.sim_fields(query_pts)
            entropy = torch.zeros(1).to(sim_params.device)

        sim_params = sim_params * 1000
        # sim_params = torch.exp(self.sim_fields(query_pts))

        # density = sim_params[..., 0]

        youngs_modulus = self.young_modulus.detach().clone()
        youngs_modulus[query_mask] += sim_params[..., 0]

        # young_modulus = torch.exp(sim_params[..., 0]) + init_young
        youngs_modulus = torch.clamp(youngs_modulus, 1000.0, 5e8)

        density = self.density.detach().clone()
        # density[self.freeze_mask] = 100000
        ret_poisson = self.poisson_ratio.detach().clone()

        return density, youngs_modulus, ret_poisson, entropy

    def train_one_step(self):

        self.sim_fields.train()
        self.velo_fields.train()
        self.model.eval()
        accelerator = self.accelerator
        device = "cuda:{}".format(accelerator.process_index)
        data = next(self.dataloader)
        cam = data["cam"][0]

        gt_videos = data["video_clip"][0, 1 : self.num_frames, ...]
        print(f"check video data: video_clip {data['video_clip'].shape}, gt_videos {gt_videos.shape}")
        # video_clip torch.Size([1, 25, 3, 576, 1024]), gt_videos torch.Size([13, 3, 576, 1024])

        window_size = int(self.window_size_schduler.compute_state(self.step)[0]) # number of frames to run simulation in this training iteration
        print(f"window_size {window_size}") 
        stop_velo_opt_thres = 15
        do_velo_opt = not self.freeze_velo
        if not do_velo_opt:
            stop_velo_opt_thres = (
                0  # stop velocity optimization if we are loading from checkpoint
            )
            self.velo_fields.eval()

        rendered_video_list = []
        rendered_video_w_force_list = []
        log_loss_dict = {
            "loss": [],
            "l2_loss": [],
            "psnr": [],
            "ssim": [],
            "entropy": [],
        }
        log_psnr_dict = {}

        particle_pos = self.particle_init_position.clone()
        # clean grid, stress, F, C and rest initial position
        self.mpm_state.reset_state(
            particle_pos.clone(),
            None,
            None,  # .clone(),
            device=device,
            requires_grad=True,
        )
        self.mpm_state.set_require_grad(True)

        (
            density,
            youngs_modulus,
            poisson,
            particle_velo,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        ) = self.get_simulation_input(device)

        init_velo_mean = particle_velo[query_mask, :].mean().item()
        init_velo_max = particle_velo[query_mask, :].max().item()

        if not do_velo_opt:
            particle_velo = particle_velo.detach()
        # print("does do velo opt": do_velo_opt)

        # print(f"check particle_velo after getting simulation input: {particle_velo.shape}") # [5342, 3]
        # tensor([[-0.0225,  0.0221, -0.0139],
        # [-0.0215,  0.0211, -0.0134],
        # [-0.0216,  0.0213, -0.0134],
        # ...,
        # [-0.0216,  0.0212, -0.0134],
        # [-0.0222,  0.0218, -0.0138],
        # [-0.0222,  0.0218, -0.0138]], device='cuda:0')
        # pdb.set_trace()

        num_particles = particle_pos.shape[0]

        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / self.args.substep # 1/30/768 = 4.34e-5
        num_substeps = int(delta_time / substep_size)

        checkpoint_steps = self.args.checkpoint_steps  

        temporal_stride = self.args.stride

        if temporal_stride < 0 or temporal_stride > window_size:
            temporal_stride = window_size

        for start_time_idx in range(0, window_size, temporal_stride):

            end_time_idx = min(start_time_idx + temporal_stride, window_size)
            print(f"start_time_idx {start_time_idx}, end_time_idx {end_time_idx}, window_size {window_size}, temporal_stride {temporal_stride}")
            # end_time_idx-start_time_idx => frame per stage
            num_step_with_grad = num_substeps * (end_time_idx - start_time_idx) # ~= num_substeps per stage

            gt_frame = gt_videos[[end_time_idx - 1]]

            if start_time_idx != 0:
                density, youngs_modulus, poisson, entropy = self.get_material_params(
                    device
                )

            if checkpoint_steps > 0 and checkpoint_steps < num_step_with_grad:
                for time_step in range(0, num_step_with_grad, checkpoint_steps):
                    num_step = min(num_step_with_grad - time_step, checkpoint_steps)
                    if num_step == 0:
                        break
                    particle_pos, particle_velo, particle_F, particle_C = (
                        MPMDifferentiableSimulationWCheckpoint.apply(
                            self.mpm_solver,
                            self.mpm_state,
                            self.mpm_model,
                            substep_size,
                            num_step,
                            particle_pos,
                            particle_velo,
                            particle_F,
                            particle_C,
                            youngs_modulus,
                            self.E_nu_list[1],
                            density,
                            query_mask,
                            device,
                            True,
                            0,
                        )
                    )
            else:
                particle_pos, particle_velo, particle_F, particle_C, particle_cov = (
                    MPMDifferentiableSimulationClean.apply(
                        self.mpm_solver,
                        self.mpm_state,
                        self.mpm_model,
                        substep_size,
                        num_step_with_grad,
                        particle_pos,
                        particle_velo,
                        particle_F,
                        particle_C,
                        youngs_modulus,
                        self.E_nu_list[1],
                        density,
                        query_mask,
                        device,
                        True,
                        0,
                        start_time_idx,
                        self.step,
                    )
                )

            # substep-3: render gaussian
            gaussian_pos = particle_pos * self.scale - self.shift
            undeformed_gaussian_pos = (
                self.particle_init_position * self.scale - self.shift
            )
            disp_offset = gaussian_pos - undeformed_gaussian_pos.detach()
            # gaussian_pos.requires_grad = True

            simulated_video = render_gaussian_seq_w_mask_with_disp(
                cam,
                self.render_params,
                undeformed_gaussian_pos.detach(),
                self.top_k_index,
                [disp_offset],
                self.sim_mask_in_raw_gaussian,
            )

            # print(f'step {self.step}, start_time_idx {start_time_idx}, check simulated_video {simulated_video.shape}, gt_frame {gt_frame.shape}')
            # print(f"check particle_v in simualted_video")
            # print(self.mpm_state.particle_v.numpy())
            # pdb.set_trace()

            # if self.apply_force:
            #     simulate_video_w_force = render_gaussian_seq_w_mask_cam_seq_with_force_with_disp(
            #         [cam],
            #         self.render_params,
            #         undeformed_gaussian_pos.detach(),
            #         self.top_k_index,
            #         [disp_offset],
            #         self.sim_mask_in_raw_gaussian,
            #         closest_idx,
            #         render_force,
            #         force_duration_steps,
            #         hide_force=False
            #     )

            # print("debug", simulated_video.shape, gt_frame.shape, gaussian_pos.shape, init_xyzs.shape, density.shape, query_mask.sum().item())
            rendered_video_list.append(simulated_video.detach())
            # if self.apply_force:
            #     rendered_video_w_force_list.append(simulate_video_w_force)

            l2_loss = 0.5 * F.mse_loss(simulated_video, gt_frame, reduction="mean")
            ssim_loss = compute_ssim(simulated_video, gt_frame)
            loss = l2_loss * (1.0 - self.ssim) + (1.0 - ssim_loss) * self.ssim

            loss = loss * (self.args.loss_decay**end_time_idx)
            sm_velo_loss = self.velo_fields.compute_smoothess_loss() * 10.0
            if not (do_velo_opt and start_time_idx == 0):
                sm_velo_loss = sm_velo_loss.detach()

            sm_spatial_loss = self.sim_fields.compute_smoothess_loss()

            sm_loss = (
                sm_velo_loss + sm_spatial_loss
            )  # typically 20 times larger than rendering loss

            loss = loss + sm_loss * self.tv_loss_weight
            loss = loss + entropy * self.args.entropy_reg
            loss = loss / self.args.compute_window
            print(f"check loss details, sm_velo_loss {sm_velo_loss.item()}, sm_spatial_loss {sm_spatial_loss.item()}, entropy {entropy.item()}, l2_loss {l2_loss.item()}, ssim_loss {ssim_loss.item()}, loss {loss.item()}")
            print(f"step: {self.step}, start_time_idx {start_time_idx}, check loss before backward {loss.item()}")
            # pdb.set_trace()
            loss.backward()

            # from IPython import embed; embed()
            # print(self.E_nu_list[1].grad)

            particle_pos, particle_velo, particle_F, particle_C = (
                particle_pos.detach(),
                particle_velo.detach(),
                particle_F.detach(),
                particle_C.detach(),
            )

            with torch.no_grad():
                psnr = compute_psnr(simulated_video, gt_frame).mean()
                log_loss_dict["loss"].append(loss.item())
                log_loss_dict["l2_loss"].append(l2_loss.item())
                log_loss_dict["psnr"].append(psnr.item())
                log_loss_dict["ssim"].append(ssim_loss.item())
                log_loss_dict["entropy"].append(entropy.item())

                print(
                    f"step {self.step}, start_time_idx {start_time_idx}, psnr {psnr.item()}, end_time_idx {end_time_idx}, youngs_modulus max {youngs_modulus.max().item()}, density max {density.max().item()}"
                )
                log_psnr_dict["psnr_frame_{}".format(end_time_idx)] = psnr.item()
                # print(psnr.item(), end_time_idx, youngs_modulus.max().item(), density.max().item())

        nu_grad_norm = self.E_nu_list[1].grad.norm(2).item()
        spatial_grad_norm = 0
        for p in self.sim_fields.parameters():
            if p.grad is not None:
                spatial_grad_norm += p.grad.norm(2).item()
        velo_grad_norm = 0
        for p in self.velo_fields.parameters():
            if p.grad is not None:
                velo_grad_norm += p.grad.norm(2).item()
        print(f"finish all start_time_idx, check grad before clipping")

        renderd_video = torch.cat(rendered_video_list, dim=0)
        renderd_video = torch.clamp(renderd_video, 0.0, 1.0)
        visual_video = (renderd_video.detach().cpu().numpy() * 255.0).astype(np.uint8)
        print(f"check visual_video type {type(visual_video)} shape {visual_video.shape}") # [6, 3, 576, 1024]
        gt_video = (gt_videos.detach().cpu().numpy() * 255.0).astype(np.uint8)
        # if self.apply_force:
        #     print(f"check rendered_video_w_force_list type {type(rendered_video_w_force_list)} {type(rendered_video_w_force_list[0])}, shape {rendered_video_w_force_list[0].shape}")
        #     visual_video_w_force = np.concatenate(rendered_video_w_force_list, axis=0) # [6, 3, 576, 1024]
        #     visual_video_w_force = np.clip(visual_video_w_force, 0.0, 255.0).astype(np.uint8)

        if (
            self.step % self.gradient_accumulation_steps == 0
            or self.step == (self.train_iters - 1)
            or (self.step % self.log_iters == self.log_iters - 1)
        ):

            torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.max_grad_norm,
                error_if_nonfinite=False,
            )  # error if nonfinite is false

            # # save visual_video and gt_video as .mp4
            # print(f"save visual_video and gt_video as .mp4, shape ", visual_video.shape, gt_video.shape) #[6, 3, 576, 1024], [13, 3 ,576, 1024]
            # visual_video_path = os.path.join(self.output_path, f"rendered_video_{self.step:06d}.mp4")
            # save_video_imageio(visual_video_path, visual_video.transpose(0, 2, 3, 1), fps=30)
            # gt_video_path = os.path.join(self.output_path, f"gt_video_{self.step:06d}.mp4")
            # save_video_imageio(gt_video_path, gt_video.transpose(0, 2, 3, 1), fps=30)
            # if self.apply_force:
            #     visual_video_w_force_path = os.path.join(self.output_path, f"rendered_video_w_force_{self.step:06d}.mp4")
            #     save_video_imageio(visual_video_w_force_path, visual_video_w_force.transpose(0, 2, 3, 1), fps=30)

            # # save per frame as .png
            # for i in range(visual_video.shape[0]):
            #     visual_frame_path = os.path.join(self.output_path, f"visual_frame_{self.step:06d}_{i:02d}.png")
            #     imageio.imwrite(visual_frame_path, visual_video[i].transpose(1, 2, 0))
            # for i in range(gt_video.shape[0]):
            #     gt_frame_path = os.path.join(self.output_path, f"gt_frame_{self.step:06d}_{i:02d}.png")
            #     imageio.imwrite(gt_frame_path, gt_video[i].transpose(1, 2, 0))
            # if self.apply_force:
            #     for i in range(visual_video_w_force.shape[0]):
            #         visual_frame_w_force_path = os.path.join(self.output_path, f"visual_frame_w_force_{self.step:06d}_{i:02d}.png")
            #         imageio.imwrite(visual_frame_w_force_path, visual_video_w_force[i].transpose(1, 2, 0))

            # pdb.set_trace()

            self.optimizer.step()
            self.optimizer.zero_grad()
            if do_velo_opt:
                assert self.velo_optimizer is not None
                torch.nn.utils.clip_grad_norm_(
                    self.velo_fields.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=False,
                )  # error if nonfinite is false
                self.velo_optimizer.step()
                self.velo_optimizer.zero_grad()
                self.velo_scheduler.step()
            with torch.no_grad():
                # TODO: check the clipping range. make sure it is consistent with the scale of parameter during material modeling
                self.E_nu_list[0].data.clamp_(1e-1, 1e8)
                self.E_nu_list[1].data.clamp_(1e-2, 0.449)
        self.scheduler.step()

        for k, v in log_loss_dict.items():
            log_loss_dict[k] = np.mean(v)

        print(log_loss_dict)
        print(
            "nu: ",
            self.E_nu_list[1].item(),
            nu_grad_norm,
            spatial_grad_norm,
            velo_grad_norm,
            "young_mean, max:",
            youngs_modulus.mean().item(),
            youngs_modulus.max().item(),
            do_velo_opt,
            "init_velo_mean:",
            init_velo_mean,
        )

        if accelerator.is_main_process and (self.step % self.wandb_iters == 0):
            with torch.no_grad():
                wandb_dict = {
                    "nu_grad_norm": nu_grad_norm,
                    "spatial_grad_norm": spatial_grad_norm,
                    "velo_grad_norm": velo_grad_norm,
                    "nu": self.E_nu_list[1].item(),
                    # "mean_density": density.mean().item(),
                    "mean_E": youngs_modulus.mean().item(),
                    "max_E": youngs_modulus.max().item(),
                    "min_E": youngs_modulus.min().item(),
                    "smoothness_loss": sm_loss.item(),
                    "window_size": window_size,
                    "max_particle_velo": particle_velo.max().item(),
                    "init_velo_mean": init_velo_mean,
                    "init_velo_max": init_velo_max,
                }

                wandb_dict.update(log_psnr_dict)
                simulated_video = self.inference(cam, substep=num_substeps)
                sim_video_torch = (
                    torch.from_numpy(simulated_video).float().to(device) / 255.0
                )
                gt_video_torch = torch.from_numpy(gt_video).float().to(device) / 255.0

                full_psnr = compute_psnr(sim_video_torch[1:], gt_video_torch)

                first_psnr = full_psnr[:6].mean().item()
                last_psnr = full_psnr[-6:].mean().item()
                full_psnr = full_psnr.mean().item()
                wandb_dict["full_psnr"] = full_psnr
                wandb_dict["first_psnr"] = first_psnr
                wandb_dict["last_psnr"] = last_psnr
                wandb_dict.update(log_loss_dict)

                # add young render
                youngs_norm = youngs_modulus - youngs_modulus.min() + 1e-2
                young_color = youngs_norm / torch.quantile(youngs_norm, 0.99)
                young_color = torch.clamp(young_color, 0.0, 1.0)
                young_color[self.freeze_mask] = 0.0
                queryed_young_color = young_color[self.top_k_index]  # [n_raw, topk]
                young_color = queryed_young_color.mean(dim=-1)

                young_color_full = torch.ones_like(
                    self.render_params.gaussians._xyz[:, 0]
                )

                young_color_full[self.sim_mask_in_raw_gaussian] = young_color
                young_color = torch.stack(
                    [young_color_full, young_color_full, young_color_full], dim=-1
                )

                young_img = render_feat_gaussian(
                    cam,
                    self.render_params.gaussians,
                    self.render_params.render_pipe,
                    self.render_params.bg_color,
                    young_color,
                )["render"]
                young_img = (
                    (young_img.detach().cpu().numpy() * 255.0)
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
                wandb_dict["young_img"] = wandb.Image(young_img)

                if self.step % int(10 * self.wandb_iters) == 0:

                    wandb_dict["rendered_video"] = wandb.Video(
                        visual_video, fps=visual_video.shape[0]
                    )

                    wandb_dict["gt_video"] = wandb.Video(
                        gt_video,
                        fps=gt_video.shape[0],
                    )

                    wandb_dict["inference_video"] = wandb.Video(
                        simulated_video,
                        fps=simulated_video.shape[0],
                    )

                    print(f"generate inference_video_v5_t3 for step {self.step}")
                    simulated_video = self.inference(
                        cam, velo_scaling=5.0, num_sec=3, substep=num_substeps
                    )
                    wandb_dict["inference_video_v5_t3"] = wandb.Video(
                        simulated_video,
                        fps=30,
                    )

                if self.use_wandb:
                    wandb.log(wandb_dict, step=self.step)

        self.accelerator.wait_for_everyone()

    def train(self):
        # might remove tqdm when multiple node
        for index in tqdm(range(self.step, self.train_iters), desc="Training progress"):
            self.train_one_step()
            if self.step % self.log_iters == self.log_iters - 1:
                if self.accelerator.is_main_process:
                    self.save()
                    # self.test()
            # self.accelerator.wait_for_everyone()
            self.step += 1
        if self.accelerator.is_main_process:
            self.save()

    @torch.no_grad()
    def inference(
        self,
        cam,
        velo_scaling=1.0,
        num_sec=1,
        nu=None,
        young_scaling=1.0,
        substep=64,
        youngs_modulus=None,
    ):

        self.sim_fields.eval()
        self.velo_fields.eval()

        device = "cuda:{}".format(self.accelerator.process_index)

        # delta_time = 1.0 / (self.num_frames - 1)
        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / substep
        num_substeps = int(delta_time / substep_size)
        print(f"run get_simulation_input in inference ...")
        (
            density,
            youngs_modulus_,
            poisson,
            init_velocity,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        ) = self.get_simulation_input(device, delta_time)

        poisson = self.E_nu_list[1].detach().clone()  # override poisson

        if youngs_modulus is None:
            youngs_modulus = youngs_modulus_ * young_scaling
        init_xyzs = self.particle_init_position.clone()

        init_velocity[query_mask, :] = init_velocity[query_mask, :] * velo_scaling

        num_particles = init_xyzs.shape[0]
        # reset state

        self.mpm_state.reset_density(
            density.clone(), query_mask, device, update_mass=True
        )
        self.mpm_solver.set_E_nu_from_torch(
            self.mpm_model, youngs_modulus.clone(), poisson.clone(), device
        )
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

        self.mpm_state.continue_from_torch(
            init_xyzs,
            init_velocity,
            particle_F,
            particle_C,
            device=device,
            requires_grad=False,
        )

        pos_list = [self.particle_init_position.clone() * self.scale - self.shift]

        prev_state = self.mpm_state

        # print(f"start running simulation, check grid_v_in")
        # print(prev_state.grid_v_in.numpy())

        for i in tqdm(range((self.num_frames - 1) * num_sec)):
            # for substep in range(num_substeps):
            #     self.mpm_solver.p2g2p(self.mpm_model, self.mpm_state, substep, substep_size, device="cuda:0")
            # pos = wp.to_torch(self.mpm_state.particle_x).clone()
            for substep_local in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=False)
                # print(f"run p2g2p_differentiable in inference for i {i}, substep_local {substep_local}")
                self.mpm_solver.p2g2p_differentiable(
                    self.mpm_model, prev_state, next_state, substep_size, device=device
                )
                prev_state = next_state

            pos = wp.to_torch(next_state.particle_x).clone()
            pos = (pos * self.scale) - self.shift
            pos_list.append(pos)

        init_pos = pos_list[0].clone()
        pos_diff_list = [_ - init_pos for _ in pos_list]
        # print(f"pos_diff_list")
        # print(pos_diff_list)
        # pdb.set_trace()

        video_array = render_gaussian_seq_w_mask_with_disp(
            cam,
            self.render_params,
            init_pos,
            self.top_k_index,
            pos_diff_list,
            self.sim_mask_in_raw_gaussian,
        )

        video_numpy = video_array.detach().cpu().numpy() * 255
        video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)

        return video_numpy

    def save(
        self,
    ):
        # training states
        output_path = os.path.join(
            self.output_path, f"checkpoint_model_{self.step:06d}"
        )
        os.makedirs(output_path, exist_ok=True)

        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate(
            [
                self.accelerator.unwrap_model(self.velo_fields, keep_fp32_wrapper=True),
                self.accelerator.unwrap_model(self.sim_fields, keep_fp32_wrapper=True),
            ]
        ):
            model_name = name_list[i]
            model_path = os.path.join(output_path, model_name + ".pt")
            torch.save(model.state_dict(), model_path)

    def load(self, checkpoint_dir):
        name_list = [
            "velo_fields",
            "sim_fields",
        ]
        for i, model in enumerate([self.velo_fields, self.sim_fields]):
            model_name = name_list[i]
            if model_name == "sim_fields" and (not self.args.load_sim):
                continue
            model_path = os.path.join(checkpoint_dir, model_name + ".pt")
            print("=> loading: ", model_path)
            model.load_state_dict(torch.load(model_path))

    def setup_eval(self, args, gaussian_path, white_background=True):
        # setup gaussians
        class RenderPipe(NamedTuple):
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False

        class RenderParams(NamedTuple):
            render_pipe: RenderPipe
            bg_color: bool
            gaussians: GaussianModel
            camera_list: list

        gaussians = GaussianModel(3)
        camera_list = self.dataset.test_camera_list

        gaussians.load_ply(gaussian_path)
        gaussians.detach_grad()
        print(
            "load gaussians from: {}".format(gaussian_path),
            "... num gaussians: ",
            gaussians._xyz.shape[0],
        )
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_pipe = RenderPipe()

        render_params = RenderParams(
            render_pipe=render_pipe,
            bg_color=background,
            gaussians=gaussians,
            camera_list=camera_list,
        )
        self.render_params = render_params

        # get_gaussian scene box
        scaler = 1.1
        points = gaussians._xyz

        min_xyz = torch.min(points, dim=0)[0]
        max_xyz = torch.max(points, dim=0)[0]

        center = (min_xyz + max_xyz) / 2

        scaled_min_xyz = (min_xyz - center) * scaler + center
        scaled_max_xyz = (max_xyz - center) * scaler + center

        aabb = torch.stack([scaled_min_xyz, scaled_max_xyz], dim=0)

        # add filled in points
        gaussian_dir = os.path.dirname(gaussian_path)

        clean_points_path = os.path.join(gaussian_dir, "clean_object_points.ply")
        if os.path.exists(clean_points_path):
            clean_xyzs = pcu.load_mesh_v(clean_points_path)
            clean_xyzs = torch.from_numpy(clean_xyzs).float().to("cuda")
            self.clean_xyzs = clean_xyzs
            print(
                "loaded {} clean points from: ".format(clean_xyzs.shape[0]),
                clean_points_path,
            )
            # we can use tight threshold here
            not_sim_maks = find_far_points(
                gaussians._xyz, clean_xyzs, thres=0.01
            ).bool()
            sim_mask_in_raw_gaussian = torch.logical_not(not_sim_maks)
            # [N]
            self.sim_mask_in_raw_gaussian = sim_mask_in_raw_gaussian
        else:
            self.clean_xyzs = None
            self.sim_mask_in_raw_gaussian = torch.ones_like(gaussians._xyz[:, 0]).bool()

        return aabb

    def demo(
        self,
        velo_scaling=5.0,
        num_sec=8.0,
        eval_ys=1.0,
        static_camera=False,
        save_name="demo_3sec",
    ):

        result_dir = "output/alocasia/results"
        pos_path = os.path.join(result_dir, save_name + "_pos.npy")

        if os.path.exists(pos_path):
            pos_array = np.load(pos_path)
        else:
            pos_array = None
        pos_array = None
        accelerator = self.accelerator
        data = next(self.dataloader)
        cam = data["cam"][0]

        for i in range(10):
            next_data = next(self.test_dataloader)
        next_cam = next_data["cam"][0]

        substep = self.args.substep  # 1e-4

        youngs_modulus = None

        self.sim_fields.eval()
        self.velo_fields.eval()

        device = "cuda:{}".format(self.accelerator.process_index)

        # delta_time = 1.0 / (self.num_frames - 1)
        delta_time = 1.0 / 30  # 30 fps
        substep_size = delta_time / substep
        num_substeps = int(delta_time / substep_size)

        (
            density,
            youngs_modulus_,
            poisson,
            init_velocity,
            query_mask,
            particle_F,
            particle_C,
            entropy,
        ) = self.get_simulation_input(device, delta_time)

        poisson = self.E_nu_list[1].detach().clone()  # override poisson

        if eval_ys < 10:
            youngs_modulus = youngs_modulus_
        else:
            youngs_modulus = torch.ones_like(youngs_modulus_) * eval_ys

        # from IPython import embed; embed()

        if pos_array is None:
            init_xyzs = self.particle_init_position.clone()

            init_velocity[query_mask, :] = init_velocity[query_mask, :] * velo_scaling

            num_particles = init_xyzs.shape[0]
            # reset state

            self.mpm_state.reset_density(
                density.clone(), query_mask, device, update_mass=True
            )
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model, youngs_modulus.clone(), poisson.clone(), device
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

            self.mpm_state.continue_from_torch(
                init_xyzs,
                init_velocity,
                particle_F,
                particle_C,
                device=device,
                requires_grad=False,
            )

            pos_list = [self.particle_init_position.clone() * self.scale - self.shift]

            prev_state = self.mpm_state
            for i in tqdm(range(int((self.num_frames - 1) * num_sec))):
                # for substep in range(num_substeps):
                #     self.mpm_solver.p2g2p(self.mpm_model, self.mpm_state, substep, substep_size, device="cuda:0")
                # pos = wp.to_torch(self.mpm_state.particle_x).clone()

                for substep_local in range(num_substeps):
                    next_state = prev_state.partial_clone(requires_grad=False)
                    self.mpm_solver.p2g2p_differentiable(
                        self.mpm_model,
                        prev_state,
                        next_state,
                        substep_size,
                        device=device,
                    )
                    prev_state = next_state

                pos = wp.to_torch(next_state.particle_x).clone()
                pos = (pos * self.scale) - self.shift
                pos_list.append(pos)

            numpy_pos = torch.stack(pos_list, dim=0).detach().cpu().numpy()

            np.save(pos_path, numpy_pos)
        else:
            pos_list = []
            for i in range(pos_array.shape[0]):
                pos = pos_array[i, ...]
                pos_list.append(torch.from_numpy(pos).to(device))

        init_pos = pos_list[0].clone()
        pos_diff_list = [_ - init_pos for _ in pos_list]

        video_array = render_gaussian_seq_w_mask_with_disp(
            cam,
            self.render_params,
            init_pos,
            self.top_k_index,
            pos_diff_list,
            self.sim_mask_in_raw_gaussian,
        )

        video_numpy = video_array.detach().cpu().numpy() * 255
        video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)
        video_numpy = np.transpose(video_numpy, [0, 2, 3, 1])

        save_path = os.path.join(
            save_name
            + "_jelly_video_substep_{}_grid_{}_evalys_{}".format(
                substep, self.args.grid_size, eval_ys
            )
            + ".gif"
        )
        print("save video to ", save_path)
        save_gif_imageio(save_path, video_numpy, fps=30)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")

    # dataset params
    parser.add_argument(
        "--dataset_dir",
        type=str,
        # default="../../data/physics_dreamer/hat_nerfstudio/",
        default="../../../../data/physics_dreamer/carnations/",
    )
    parser.add_argument("--video_dir_name", type=str, default="videos")
    parser.add_argument(
        "--dataset_res",
        type=str,
        default="large",  # ["middle", "small", "large"]
    )
    parser.add_argument(
        "--motion_model_path",
        type=str,
        default=None,  # not used
        help="path to load the pretrained motion model from",
    )

    parser.add_argument("--model", type=str, default="se3_field")
    parser.add_argument("--feat_dim", type=int, default=64)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--spatial_res", type=int, default=32)
    parser.add_argument("--zero_init", type=bool, default=True)

    parser.add_argument("--entropy_cls", type=int, default=-1)
    parser.add_argument("--entropy_reg", type=float, default=1e-2)

    parser.add_argument("--num_frames", type=str, default=14)

    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--sim_res", type=int, default=8)
    parser.add_argument("--sim_output_dim", type=int, default=1)
    parser.add_argument("--substep", type=int, default=768) # 1s / 30 fps / 768 substeps = 4.34e-5 s/substep
    parser.add_argument("--loss_decay", type=float, default=1.0)
    parser.add_argument("--start_window_size", type=int, default=6)
    parser.add_argument("--compute_window", type=int, default=1)
    parser.add_argument("--grad_window", type=int, default=14)
    # -1 means no gradient checkpointing
    parser.add_argument("--checkpoint_steps", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--downsample_scale", type=float, default=0.04)
    parser.add_argument("--top_k", type=int, default=8)

    # loss parameters
    parser.add_argument("--tv_loss_weight", type=float, default=1e-4)
    parser.add_argument("--ssim", type=float, default=0.9)

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="../../output/inverse_sim")
    parser.add_argument("--log_iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        # psnr 29.0
        default="../../output/inverse_sim/fast_alocasia_velopretrain_cleandecay_1.0_substep_96_se3_field_lr_0.01_tv_0.01_iters_300_sw_2_cw_2/seed0/checkpoint_model_000299",
        help="path to load velocity pretrain checkpoint from",
    )
    # training parameters
    parser.add_argument("--train_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )

    # wandb parameters
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="inverse_sim")
    parser.add_argument("--wandb_iters", type=int, default=10)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--run_eval", action="store_true", default=False)
    parser.add_argument("--load_sim", action="store_true", default=False)
    parser.add_argument("--test_convergence", action="store_true", default=False)
    parser.add_argument("--update_velo", action="store_true", default=False)
    parser.add_argument("--eval_iters", type=int, default=8)
    parser.add_argument("--eval_ys", type=float, default=1e6)
    parser.add_argument("--demo_name", type=str, default="demo_3sec_sv_gres48_lr1e-2")
    parser.add_argument("--velo_scaling", type=float, default=5.0)

    # distributed training args
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    # impulse force condition
    parser.add_argument("--apply_force", action="store_true", default=False)
    parser.add_argument("--initE", type=float, default=1e5)
    parser.add_argument("--postfix", type=str, default="")

    args, extra_args = parser.parse_known_args()
    cfg = create_config(args.config, args, extra_args)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    print(args.local_rank, "local rank")

    return cfg


if __name__ == "__main__":
    args = parse_args()

    # torch.backends.cuda.matmul.allow_tf32 = True

    trainer = Trainer(args)

    if args.run_eval:
        trainer.demo(
            velo_scaling=args.velo_scaling,
            eval_ys=args.eval_ys,
            save_name=args.demo_name,
        )
    else:
        # trainer.debug()
        trainer.train()
