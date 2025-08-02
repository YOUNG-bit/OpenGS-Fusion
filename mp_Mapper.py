import os
import torch
import torch.multiprocessing as mp
import copy
import random
import sys
import cv2
import numpy as np
import time
import rerun as rr

sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim
from scene import GaussianModel
from gaussian_renderer import render, render_3, network_gui
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import matplotlib.pyplot as plt

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

class Mapper(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
        self.rerun_viewer = slam.rerun_viewer
        self.iter_shared = slam.iter_shared
        self.saving_all_keyframe = slam.saving_all_keyframe

        self.out_online_frame_path = os.path.join(self.output_path, "online_frame")
        os.makedirs(self.out_online_frame_path, exist_ok=True)

        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0., 0., 1]])

        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq

        # Camera poses and keyframes
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        self.prune_th = 2.5 if self.trajmanager.which_dataset == "replica" else 5.0

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.simple_saving_cams = []
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        # Shared memory objects
        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.is_simple_saving_keyframe_shared = slam.is_simple_saving_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started

    def run(self):
        """Entry point for the mapping process."""
        self.mapping()

    def mapping(self):
        """Main mapping loop. Receives new frames, updates map, and trains the model."""
        t = torch.zeros((1, 1)).float().cuda()
        if self.verbose:
            network_gui.init("127.0.0.1", 6009)

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()

        # Signal that mapping process is ready
        self.is_mapping_process_started[0] = 1

        # Wait for initial keyframe from tracking
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)

        self.total_start_time_viewer = time.time()

        # Initialize Gaussian map from first keyframe
        points, colors, rots, scales, z_values, trackable_filter, voxel_index = self.shared_new_gaussians.get_values()
        self.gaussians.create_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter, voxel_index)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.is_tracking_keyframe_shared[0] = 0

        # Demo mode: run viewer for 30 seconds
        if self.demo[0]:
            a = time.time()
            while (time.time() - a) < 30.:
                print(30. - (time.time() - a))
                self.run_viewer()
        self.demo[0] = 0

        # Add first camera to mapping
        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()
        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams) - 1)

        new_keyframe = False
        while True:
            if self.end_of_dataset[0]:
                break

            if self.verbose:
                self.run_viewer()

            # Handle new tracking keyframe
            if self.is_tracking_keyframe_shared[0]:
                points, colors, rots, scales, z_values, trackable_filter, voxel_index = self.shared_new_gaussians.get_values()
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter, voxel_index)
                target_points, target_rots, target_scales = self.gaussians.get_trackable_gaussians_tensor(self.trackable_opacity_th)
                self.shared_target_gaussians.input_values(target_points, target_rots, target_scales)
                self.target_gaussians_ready[0] = 1

                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams) - 1)
                self.is_tracking_keyframe_shared[0] = 0

            # Handle new mapping keyframe
            elif self.is_mapping_keyframe_shared[0]:
                points, colors, rots, scales, z_values, _, voxel_index = self.shared_new_gaussians.get_values()
                self.gaussians.add_from_pcd2_tensor(points, colors, rots, scales, z_values, [], voxel_index)

                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams) - 1)
                self.is_mapping_keyframe_shared[0] = 0

                # Save all keyframes if enabled
                if self.saving_all_keyframe:
                    self.simple_saving_cams.append(newcam)
                    with torch.no_grad():
                        render_pkg = render(newcam, self.gaussians, self.pipe, self.background)
                        image = render_pkg["render"]
                        image = image.cpu().numpy().transpose(1, 2, 0)
                        image = np.clip(image, 0., 1.0) * 255
                        image = image.astype(np.uint8)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{self.out_online_frame_path}/{newcam.cam_idx[0]}.png", image)

            # Handle simple saving keyframe
            elif self.is_simple_saving_keyframe_shared[0]:
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                with torch.no_grad():
                    render_pkg = render(newcam, self.gaussians, self.pipe, self.background)
                    image = render_pkg["render"]
                    image = image.cpu().numpy().transpose(1, 2, 0)
                    image = np.clip(image, 0., 1.0) * 255
                    image = image.astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    newcam.on_cpu()
                    self.simple_saving_cams.append(newcam)
                cv2.imwrite(f"{self.out_online_frame_path}/{newcam.cam_idx[0]}.png", image)
                self.is_simple_saving_keyframe_shared[0] = 0

            # Training step
            if len(self.mapping_cams) > 0:
                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True
                else:
                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]

                # Select ground truth image and depth based on training stage
                if self.training_stage == 0:
                    gt_image = viewpoint_cam.original_image.cuda()
                    gt_depth_image = viewpoint_cam.original_depth_image.cuda()
                elif self.training_stage == 1:
                    gt_image = viewpoint_cam.rgb_level_1.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_1.cuda()
                elif self.training_stage == 2:
                    gt_image = viewpoint_cam.rgb_level_2.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_2.cuda()

                self.training = True
                render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe, self.background, training_stage=self.training_stage)
                depth_image = render_pkg["render_depth"]
                image = render_pkg["render"]
                viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                mask = (gt_depth_image > 0.).detach()
                gt_image = gt_image * mask

                # Loss calculation
                Ll1_map, Ll1 = l1_loss(image, gt_image)
                L_ssim_map, L_ssim = ssim(image, gt_image)
                d_max = 10.
                Ll1_d_map, Ll1_d = l1_loss(depth_image / d_max, gt_depth_image / d_max)
                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                loss_d = Ll1_d
                loss = loss_rgb + 0.1 * loss_d

                loss.backward()
                with torch.no_grad():
                    if self.train_iter % 200 == 0:
                        self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                    # Send rendered image to rerun viewer if new keyframe
                    if new_keyframe and self.rerun_viewer:
                        current_i = copy.deepcopy(self.iter_shared[0])
                        rgb_np = image.cpu().numpy().transpose(1, 2, 0)
                        rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                        rr.log("rendered_rgb", rr.Image(rgb_np))
                        new_keyframe = False

                self.training = False
                self.train_iter += 1

        # If verbose, keep viewer running
        if self.verbose:
            while True:
                self.run_viewer(False)

        # Save results at the end of mapping
        self.gaussians.save_ply(os.path.join(self.output_path, "gs_scene.ply"))
        if self.saving_all_keyframe:
            self.gaussians.save_npz(os.path.join(self.output_path, "gs_scene.npz"), self.simple_saving_cams)
        else:
            self.gaussians.save_npz(os.path.join(self.output_path, "gs_scene.npz"), self.mapping_cams)

        self.calc_2d_metric()

    def run_viewer(self, lower_speed=True):
        """Viewer process for network GUI."""
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            if time.time() - self.last_t < 1 / self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path)
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def set_downsample_filter(self, downsample_scale):
        """Calculate downsample indices and precomputed x/y values for point cloud."""
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0, int(self.H / sample_interval) + 1)
        h_val = h_val - 1
        h_val[0] = 0
        h_val = h_val * self.W
        a, b = torch.meshgrid(h_val, torch.arange(0, self.W, sample_interval))
        pick_idxs = ((a + b).flatten(),)
        v, u = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        x_pre = (u - self.cx) / self.fx
        y_pre = (v - self.cy) / self.fy
        return pick_idxs, x_pre, y_pre

    def get_image_dirs(self, images_folder):
        """Get image and depth file lists for the dataset."""
        color_paths = []
        depth_paths = []
        if self.trajmanager.which_dataset == "replica":
            images_folder = os.path.join(images_folder, "images")
            image_files = sorted(os.listdir(images_folder))
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                color_paths.append(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_paths.append(f"{self.dataset_path}/depth_images/{depth_image_name}.png")
            return color_paths, depth_paths
        elif self.trajmanager.which_dataset == "tum":
            return self.trajmanager.color_paths, self.trajmanager.depth_paths
        elif self.trajmanager.which_dataset == "scannet":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = sorted(os.listdir(rgb_folder), key=lambda x: int(x.split(".")[0]))
            depth_files = sorted(os.listdir(depth_folder), key=lambda x: int(x.split(".")[0]))
            color_paths = [os.path.join(self.dataset_path, "rgb", key) for key in image_files]
            depth_paths = [os.path.join(self.dataset_path, "depth", key) for key in depth_files]
            return color_paths, depth_paths

    def calc_2d_metric(self):
        """Calculate 2D metrics (PSNR, SSIM, LPIPS, Depth L1) for rendered images."""
        psnrs = []
        ssims = []
        lpips = []
        depth_l1 = []

        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")
        original_resolution = True
        image_names, depth_image_names = self.get_image_dirs(self.dataset_path)
        final_poses = self.final_pose
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        with torch.no_grad():
            for i in tqdm(range(len(image_names))):
                cam = self.mapping_cams[0]
                c2w = final_poses[i]

                if original_resolution:
                    gt_rgb = cv2.imread(image_names[i])
                    if self.trajmanager.which_dataset == "scannet":
                        gt_rgb = cv2.resize(gt_rgb, (self.W, self.H))
                    gt_depth = cv2.imread(depth_image_names[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / self.depth_scale
                    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                    gt_rgb = gt_rgb / 255
                    gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2, 0, 1)
                    gt_depth_ = torch.from_numpy(gt_depth).float().cuda().unsqueeze(0)
                else:
                    gt_rgb_ = cam.original_image.cuda()
                    gt_rgb = np.asarray(gt_rgb_.detach().cpu()).squeeze().transpose((1, 2, 0))
                    gt_depth_ = cam.original_depth_image.cuda()
                    gt_depth = np.asarray(cam.original_depth_image.detach().cpu()).squeeze() / self.depth_scale

                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3].transpose()
                T = w2c[:3, 3]
                cam.R = torch.tensor(R)
                cam.t = torch.tensor(T)
                if original_resolution:
                    cam.image_width = gt_rgb_.shape[2]
                    cam.image_height = gt_rgb_.shape[1]
                cam.update_matrix()

                render_pkg = render(cam, self.gaussians, self.pipe, self.background)
                ours_rgb_ = render_pkg["render"]
                ours_depth_ = render_pkg["render_depth"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()
                saveed_rgb_ = copy.deepcopy(ours_rgb_)

                valid_depth_mask_ = (gt_depth_ > 0)
                gt_rgb_ = gt_rgb_ * valid_depth_mask_
                ours_rgb_ = ours_rgb_ * valid_depth_mask_

                square_error = (gt_rgb_ - ours_rgb_) ** 2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)
                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]

                valid_depth_mask = (gt_depth_ > 0)
                d_max = 10.
                depth_error = (gt_depth_ / d_max - ours_depth_ / d_max).abs()
                depth_error = depth_error * valid_depth_mask
                depth_error = depth_error[depth_error > 0]
                depth_error = depth_error.mean()
                depth_l1.append(depth_error.detach().cpu() * 100)

                os.makedirs(os.path.join(self.output_path, "rendered_images"), exist_ok=True)
                if self.save_results and ((i + 1) % 100 == 0 or i == len(image_names) - 1):
                    ours_rgb = np.asarray(saveed_rgb_.detach().cpu()).squeeze().transpose((1, 2, 0))
                    ours_rgb = np.clip(ours_rgb, 0., 1.0) * 255
                    ours_rgb = ours_rgb.astype(np.uint8)
                    ours_rgb = cv2.cvtColor(ours_rgb, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"{self.output_path}/rendered_images/{i}.png", ours_rgb)

                torch.cuda.empty_cache()

            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)
            depth_l1 = np.array(depth_l1)

            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}")
            print(f"Depth_L1: {depth_l1.mean():.3f}")

def mse2psnr(x):
    """Convert MSE to PSNR."""
    return -10. * torch.log(x) / torch.log(torch.tensor(10.))