import os
import torch
import torch.multiprocessing as mp
import sys
import cv2
import numpy as np
import open3d as o3d
import time
import rerun as rr

sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.graphics_utils import focal2fov
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints, SharedVDBPoints, ShareOnlineSAM
from gaussian_renderer import render, network_gui
from mp_Tracker import Tracker
from mp_Mapper import Mapper
from mp_VdbFusion import VdbFusion
# from MobileSAM_Clip import MobileSAM2_CLIP    # Deprecated

torch.multiprocessing.set_sharing_strategy('file_system')

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

class OpenGS_Fusion(SLAMParameters):
    def __init__(self, args):
        super().__init__()
        self.dataset_path = args.dataset_path
        self.config = args.config
        self.output_path = args.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = args.verbose
        self.keyframe_th = float(args.keyframe_th)
        self.knn_max_distance = float(args.knn_maxd)
        self.overlapped_th = float(args.overlapped_th)
        self.max_correspondence_distance = float(args.max_correspondence_distance)
        self.trackable_opacity_th = float(args.trackable_opacity_th)
        self.overlapped_th2 = float(args.overlapped_th2)
        self.downsample_rate = int(args.downsample_rate)
        self.test = args.test
        self.save_results = args.save_results
        self.rerun_viewer = args.rerun_viewer
        self.with_sem = args.with_sem
        self.use_tracking = args.use_tracking
        self.rot_thr = float(args.rot_thr)
        self.trans_thr = float(args.trans_thr)
        self.debug_sem = args.debug_sem
        self.weight_cal_way = int(args.weight_cal_way)
        self.sam_model = args.sam_model
        self.online_sam = args.online_sam
        self.sam_save_results = args.sam_save_results
        self.saving_all_keyframe = args.saving_all_keyframe

        # Initialize rerun viewer if enabled
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.spawn(connect=False)

        # Load camera parameters
        with open(self.config) as camera_parameters_file:
            camera_parameters_ = camera_parameters_file.readlines()
        self.camera_parameters = camera_parameters_[2].split()
        self.W = int(self.camera_parameters[0])
        self.H = int(self.camera_parameters[1])
        self.fx = float(self.camera_parameters[2])
        self.fy = float(self.camera_parameters[3])
        self.cx = float(self.camera_parameters[4])
        self.cy = float(self.camera_parameters[5])
        self.depth_scale = float(self.camera_parameters[6])
        self.depth_trunc = float(self.camera_parameters[7])
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # VDB Fusion parameters
        self.max_points = 300000
        self.max_label = 100
        self.voxel_size = args.voxel_size
        self.sdf_trunc = args.sdf_trunc
        self.space_carving = args.space_carving
        self.fill_holes = True
        self.min_weight = args.min_weight

        # Online SAM parameters
        if self.online_sam:
            self.sam_resize_size = (244, 244)
            self.sam_output_path = os.path.join(self.output_path, "sam_output")

        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Trajectory manager
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)

        # Prepare test image and point cloud for shared memory allocation
        test_rgb_img, test_depth_img = self.get_test_image(f"{self.dataset_path}/images")
        test_points, _, _, _ = self.downsample_and_make_pointcloud(test_depth_img, test_rgb_img)
        self.per_max_points = test_points.shape[0]
        print("per_max_points: ", self.per_max_points)

        num_final_poses = len(self.trajmanager.gt_poses)

        # Shared memory objects
        self.shared_cam = SharedCam(
            FoVx=focal2fov(self.fx, self.W),
            FoVy=focal2fov(self.fy, self.H),
            image=test_rgb_img,
            depth_image=test_depth_img,
            cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy
        )
        self.shared_new_points = SharedPoints(test_points.shape[0])
        self.shared_new_gaussians = SharedGaussians(test_points.shape[0])
        self.shared_new_points_for_vdb = SharedVDBPoints(test_points.shape[0])

        # Target gaussians for tracking
        if self.use_tracking:
            self.shared_target_gaussians = SharedTargetPoints(10000000)
        else:
            self.shared_target_gaussians = SharedTargetPoints(1)

        # Shared flags and buffers
        self.end_of_dataset = torch.zeros((1)).int()
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()
        self.is_mapping_keyframe_shared = torch.zeros((1)).int()
        self.is_simple_saving_keyframe_shared = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.new_points_ready_for_vdb = torch.zeros((1)).int()
        self.final_pose = torch.zeros((num_final_poses, 4, 4)).float()
        self.demo = torch.zeros((1)).int()
        self.is_mapping_process_started = torch.zeros((1)).int()
        self.iter_shared = torch.zeros((1)).int()

        # Online SAM shared memory
        if self.online_sam:
            self.shared_online_sam = ShareOnlineSAM(test_rgb_img, self.downsample_idxs)
            self.new_image_ready = torch.zeros((1)).int()
            self.sam_result_ready = torch.zeros((1)).int()
            self.read_current_image = torch.zeros((1)).int()
            self.shared_online_sam.share_memory()
            self.new_image_ready.share_memory_()
            self.sam_result_ready.share_memory_()
            self.read_current_image.share_memory_()

        # Share memory for all shared objects
        self.shared_cam.share_memory()
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_mapping_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.new_points_ready_for_vdb.share_memory_()
        self.final_pose.share_memory_()
        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        self.iter_shared.share_memory_()
        self.is_simple_saving_keyframe_shared.share_memory_()

        self.demo[0] = args.demo
        self.mapper = Mapper(self)
        self.tracker = Tracker(self)
        self.vdbfusion = VdbFusion(self)
        if self.online_sam:
            self.sam_model = MobileSAM2_CLIP(self, self.sam_output_path, self.sam_save_results, resize_size=self.sam_resize_size)

    def tracking(self, rank):
        """Tracking process entry point."""
        self.tracker.run()

    def mapping(self, rank):
        """Mapping process entry point."""
        self.mapper.run()

    def vdb_fusion(self, rank):
        """VDB Fusion process entry point."""
        self.vdbfusion.run()

    def run_online_sam(self, rank):
        """Online SAM process entry point."""
        self.sam_model.run()

    def run(self):
        """Main entry point for launching all processes."""
        processes = []
        num_processes = 3
        if self.online_sam:
            num_processes = 4
        for rank in range(num_processes):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank,))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank,))
            elif rank == 2:
                print("Starting VDB Fusion process...")
                p = mp.Process(target=self.vdb_fusion, args=(rank,))
            elif rank == 3:
                print("Starting Online SAM process...")
                p = mp.Process(target=self.run_online_sam, args=(rank,))
            else:
                continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def get_test_image(self, images_folder):
        """Load the first RGB and depth image for initialization."""
        if self.camera_parameters[8] == "replica":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = sorted(os.listdir(images_folder))
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
            rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
            depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png")).astype(np.float32)
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        elif self.camera_parameters[8] == "scannet":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            rgb_image = cv2.resize(rgb_image, (self.W, self.H))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        return rgb_image, depth_image

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

    def downsample_and_make_pointcloud(self, depth_img, rgb_img):
        """Downsample images and generate point cloud."""
        colors = torch.from_numpy(rgb_img).reshape(-1, 3).float()[self.downsample_idxs] / 255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs] / self.depth_scale
        filter = torch.where((z_values != 0) & (z_values <= self.depth_trunc))
        x = self.x_pre * z_values
        y = self.y_pre * z_values
        points = torch.stack([x, y, z_values], dim=-1)
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()

    def get_image_dirs(self, images_folder):
        """Get image and depth file lists for the dataset."""
        if self.camera_parameters[8] == "replica":
            images_folder = os.path.join(self.dataset_path, "images")
            image_files = sorted(os.listdir(images_folder))
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
        elif self.camera_parameters[8] == "tum":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = os.listdir(rgb_folder)
            depth_files = os.listdir(depth_folder)
        elif self.camera_parameters[8] == "scannet":
            rgb_folder = os.path.join(self.dataset_path, "rgb")
            depth_folder = os.path.join(self.dataset_path, "depth")
            image_files = os.listdir(rgb_folder)
            depth_files = os.listdir(depth_folder)
        return image_files, depth_files

if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    parser.add_argument("--config", help="caminfo", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--debug_sem", action='store_true', default=False)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--rot_thr", default=0)
    parser.add_argument("--trans_thr", default=0.0)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action='store_true', default=None)
    parser.add_argument("--sam_save_results", action='store_true', default=False)
    parser.add_argument("--rerun_viewer", action="store_true", default=False)
    parser.add_argument("--use_tracking", action="store_true", default=False)
    parser.add_argument("--weight_cal_way", default=2)
    parser.add_argument("--online_sam", action='store_true', default=False)
    parser.add_argument("--saving_all_keyframe", action='store_true', default=False)
    parser.add_argument("--voxel_size", default=0.05)
    parser.add_argument("--sdf_trunc", default=0.03)
    parser.add_argument("--space_carving", action='store_true', default=False)
    parser.add_argument("--with_sem", action='store_true', default=False)
    parser.add_argument("--min_weight", default=5.0)
    parser.add_argument("--sam_model", default="mobilesam", help="langsplat / mobilesam")
    args = parser.parse_args()

    opengs_fusion = OpenGS_Fusion(args)
    opengs_fusion.run()

    # Example usage:
    # python opengs_fusion.py --dataset_path /path/to/dataset --config /path/to/config.txt --output_path /path/to/output --rerun_viewer --save_results
    