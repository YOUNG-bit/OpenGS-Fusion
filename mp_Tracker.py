import copy
import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from gaussian_renderer import render, render_2, network_gui
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R

class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
        self.test = slam.test
        self.rerun_viewer = slam.rerun_viewer
        self.iter_shared = slam.iter_shared
        self.with_sem = slam.with_sem
        self.use_tracking = slam.use_tracking
        self.rot_thr = slam.rot_thr
        self.trans_thr = slam.trans_thr
        self.sam_model = slam.sam_model
        self.online_sam = slam.online_sam
        self.saving_all_keyframe = slam.saving_all_keyframe

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
                                       [0.,0.,1]])

        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = pygicp.FastGICP()

        # Camera poses
        self.trajmanager = TrajManager(self.camera_parameters[8], self.dataset_path)
        self.poses = [self.trajmanager.gt_poses[0]]
        self.last_pose = self.poses[-1]

        # Keyframes and state
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False

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
        self.image_files = []
        self.debug_sem = slam.debug_sem

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # Shared memory
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.is_simple_saving_keyframe_shared = slam.is_simple_saving_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started

        # VDB shared memory
        self.new_points_ready_for_vdb = slam.new_points_ready_for_vdb
        self.shared_new_points_for_vdb = slam.shared_new_points_for_vdb
        self.semantic_fused_times = 0

        # Shared Online SAM
        if self.online_sam:
            self.shared_online_sam = slam.shared_online_sam
            self.new_image_ready = slam.new_image_ready
            self.sam_result_ready = slam.sam_result_ready
            self.read_current_image = slam.read_current_image

        # TSDF filtering
        self.tsdf_min_value_abs = slam.voxel_size
        self.gs_density_min_weight = 0

    def run(self):
        self.tracking()

    def tracking(self):
        tt = torch.zeros((1,1)).float().cuda()

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()

        # Dataset indices
        start_idx = 0
        interval = 1
        end_idx = -1

        # Load images
        self.rgb_images, self.depth_images = self.get_images(f"{self.dataset_path}/images", start_idx, end_idx, interval)
        if self.with_sem:
            if self.online_sam:
                pass
            else:
                if self.sam_model == "langsplat":
                    self.sem_label_map, self.sem_label_feature = self.get_label_info_from_langsplat(f"{self.dataset_path}/language_features")
                elif self.sam_model == "mobilesam":
                    self.sem_label_map, self.sem_label_feature = self.get_label_info_from_mobilesam(f"{self.dataset_path}/mobile_sam_feature")
                else:
                    raise ValueError("Unknown semantic model")
                print(f"Sem label map shape: {self.sem_label_map[0].shape}, rgb_images shape: {self.rgb_images[0].shape[:2]}")
                assert self.rgb_images[0].shape[:2] == self.sem_label_map[0].shape, "Shape not match"

        self.num_images = len(self.rgb_images)
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        if_mapping_keyframe = False
        if_simple_saving_keyframe = False

        self.total_start_time = time.time()
        pbar = tqdm(total=self.num_images)

        # Prepare for online SAM
        if self.online_sam:
            self.read_current_image[0] = 1

        for ii in range(self.num_images):
            self.iter_shared[0] = ii
            current_image = self.rgb_images.pop(0)
            current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
            depth_image = self.depth_images.pop(0)
            gt_pose = self.trajmanager.gt_poses[start_idx+ii*interval]

            # Check motion threshold
            if_motion_exceeded = self.is_motion_exceeded(self.last_pose, gt_pose, self.rot_thr, self.trans_thr)

            _sem_label_map = None
            _sem_label_feature = None
            if self.with_sem:
                if self.online_sam:
                    pass
                else:
                    _sem_label_map = self.sem_label_map.pop(0)
                    _sem_label_feature = self.sem_label_feature.pop(0)

            # Generate point cloud
            points, colors, z_values, trackable_filter, _label_map, _zero_filter = self.downsample_and_make_pointcloud2(depth_image, current_image, _sem_label_map)

            # Online SAM semantic processing
            if self.with_sem and self.online_sam:
                if if_motion_exceeded or ii == 0 or ii == self.num_images-1:
                    while not self.read_current_image[0]:
                        time.sleep(1e-15)
                    self.shared_online_sam.input_image_from_tracker(current_image, _zero_filter[0])
                    self.sam_result_ready[0] = 0
                    self.read_current_image[0] = 0
                    self.new_image_ready[0] = 1

            # VDBFusion integration: transform to world coordinates
            world_points = (gt_pose[:3, :3] @ points.T).T + gt_pose[:3, 3]
            while self.new_points_ready_for_vdb[0]:
                time.sleep(1e-15)
            if if_motion_exceeded or ii == 0 or ii == self.num_images-1:
                if self.debug_sem:
                    print(f"~~~~~~~~~~~~~~~~Current iteration for sem: {self.iteration_images}~~~~~~~~~~~~~~~~~")
                self.shared_new_points_for_vdb.input_values_from_tracker(torch.tensor(world_points), torch.tensor(gt_pose),
                                                                         torch.tensor(_label_map) if _label_map is not None else None,
                                                                         torch.tensor(_sem_label_feature) if _sem_label_feature is not None else None)
                self.last_pose = copy.deepcopy(gt_pose)
                self.semantic_fused_times += 1
            else:
                self.shared_new_points_for_vdb.input_values_from_tracker(torch.tensor(world_points), torch.tensor(gt_pose), None, None)
            self.new_points_ready_for_vdb[0] = 1

            # GICP registration
            if self.iteration_images == 0:
                current_pose = copy.deepcopy(gt_pose)

                if self.rerun_viewer:
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Camera pose update
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()

                # Transform current points to camera coordinates
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                self.reg.set_input_target(points)

                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]

                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots = self.reg.get_target_rotationsq()
                scales = self.reg.get_target_scales()
                rots = np.reshape(rots, (-1,4))
                scales = np.reshape(scales, (-1,3))

                # Wait for VDBFusion to finish
                while self.new_points_ready_for_vdb[0]:
                    time.sleep(1e-15)
                voxel_index, voxel_tsdf, voxel_weights = self.shared_new_points_for_vdb.get_value_from_vdb()

                # Assign first gaussian to shared memory
                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors),
                                                       torch.tensor(rots), torch.tensor(scales),
                                                       torch.tensor(z_values), torch.tensor(trackable_filter),
                                                       torch.tensor(voxel_index))

                # Add first keyframe
                depth_image = depth_image.astype(np.float32)/self.depth_scale
                self.shared_cam.setup_cam(R, T, current_image, depth_image)
                self.shared_cam.cam_idx[0] = self.iteration_images

                self.is_tracking_keyframe_shared[0] = 1

                while self.demo[0]:
                    time.sleep(1e-15)
                    self.total_start_time = time.time()
                if self.rerun_viewer:
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))
            else:
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)

                # Use ground truth pose for initialization
                initial_pose = copy.deepcopy(gt_pose)
                current_pose = self.reg.align(initial_pose)
                current_pose = initial_pose
                self.poses.append(current_pose)

                if self.rerun_viewer:
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Camera pose update
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()

                # Transform current points to camera coordinates
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)

                if self.use_tracking:
                    target_corres, distances = self.reg.get_source_correspondence()
                    len_corres = len(np.where(distances<self.overlapped_th)[0])
                    if  (self.iteration_images >= self.num_images-1 \
                        or len_corres/distances.shape[0] < self.keyframe_th):
                        if_tracking_keyframe = True
                        self.from_last_tracking_keyframe += 1
                    else:
                        if_tracking_keyframe = False
                        self.from_last_tracking_keyframe += 1

                self.from_last_tracking_keyframe += 1
                if_tracking_keyframe = False
                distances = False

                # Mapping keyframe selection
                if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                    if_mapping_keyframe = True
                else:
                    if_mapping_keyframe = False

                if self.saving_all_keyframe and if_mapping_keyframe == False:
                    if_simple_saving_keyframe = True

                # Tracking keyframe update
                if if_tracking_keyframe:
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)

                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)
                    R_d_q = R_d.as_quat()
                    rots = self.quaternion_multiply(R_d_q, rots)

                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2)
                    trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]

                    while self.new_points_ready_for_vdb[0]:
                        time.sleep(1e-15)
                    voxel_index, voxel_tsdf, voxel_weights = self.shared_new_points_for_vdb.get_value_from_vdb()

                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors),
                                                           torch.tensor(rots), torch.tensor(scales),
                                                           torch.tensor(z_values), torch.tensor(trackable_filter),
                                                           torch.tensor(voxel_index))

                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images

                    self.is_tracking_keyframe_shared[0] = 1

                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)
                    target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
                    self.reg.set_input_target(target_points)
                    self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                    self.target_gaussians_ready[0] = 0

                    if self.rerun_viewer:
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.01))

                # Mapping keyframe update
                elif if_mapping_keyframe:
                    while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                        time.sleep(1e-15)

                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)
                    R_d_q = R_d.as_quat()
                    rots = self.quaternion_multiply(R_d_q, rots)

                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    while self.new_points_ready_for_vdb[0]:
                        time.sleep(1e-15)
                    voxel_index, voxel_tsdf, voxel_weights = self.shared_new_points_for_vdb.get_value_from_vdb()

                    # Filter by voxel weights
                    if self.gs_density_min_weight > 0:
                        mask_voxel_weights = voxel_weights > self.gs_density_min_weight
                        voxel_index = voxel_index[mask_voxel_weights]
                        points = points[mask_voxel_weights]
                        colors = colors[mask_voxel_weights]
                        rots = rots[mask_voxel_weights]
                        scales = scales[mask_voxel_weights]
                        z_values = z_values[mask_voxel_weights]
                        print(f"Filter out {mask_voxel_weights.shape[0] - mask_voxel_weights.sum()} points")

                    self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors),
                                                           torch.tensor(rots), torch.tensor(scales),
                                                           torch.tensor(z_values), torch.tensor(trackable_filter),
                                                           torch.tensor(voxel_index))

                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images

                    self.is_mapping_keyframe_shared[0] = 1

                    if self.rerun_viewer:
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.01))

                # Simple keyframe saving
                elif if_simple_saving_keyframe:
                    while self.is_simple_saving_keyframe_shared[0]:
                        time.sleep(1e-15)

                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image)
                    self.shared_cam.cam_idx[0] = self.iteration_images

                    self.is_simple_saving_keyframe_shared[0] = 1

            pbar.update(1)
            torch.cuda.empty_cache()

            # Control FPS
            while 1/((time.time() - self.total_start_time)/(self.iteration_images+1)) > 22.:
                time.sleep(1e-15)

            self.iteration_images += 1

        # End of tracking
        pbar.close()
        self.end_of_dataset[0] = 1
        self.final_pose[:,:,:] = torch.tensor(self.poses).float()

        print(f"System FPS: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
        print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}")
        print(f"Semantic fused times: {self.semantic_fused_times}")

    def get_label_info_from_langsplat(self, label_folder, level="l"):
        label_map = []
        lang_embed = []
        for i in tqdm(range(len(self.image_files))):
            image_name = os.path.basename(self.image_files[i])[:-4]
            _label_map = np.load(os.path.join(label_folder, image_name + '_s.npy'))
            _lang_embed = np.load(os.path.join(label_folder, image_name + '_f.npy'))

            level_map = {'default':0, "s":1, "m":2, "l":3}
            level_idx = level_map[level]
            _label_map = _label_map[level_idx]

            # Resize if shape mismatch
            if _label_map.shape[0] != self.H or _label_map.shape[1] != self.W:
                _label_map = cv2.resize(_label_map, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

            _label_map += 1
            _lang_embed = np.vstack((np.zeros((1, _lang_embed.shape[1]), dtype=np.float32), _lang_embed))

            unique_labels = np.unique(_label_map)
            for j in range(len(unique_labels)):
                _label_map[_label_map == unique_labels[j]] = j
            label_map.append(_label_map)
            _lang_embed = _lang_embed[unique_labels.astype(np.int32)]
            lang_embed.append(_lang_embed)

        return label_map, lang_embed

    def get_label_info_from_mobilesam(self, label_folder):
        label_map = []
        lang_embed = []
        for i in tqdm(range(len(self.image_files))):
            image_name = os.path.basename(self.image_files[i])[:-4]
            _label_map = np.load(os.path.join(label_folder, image_name + '_s.npy'))
            _lang_embed = np.load(os.path.join(label_folder, image_name + '_f.npy'))

            _label_map = _label_map[0]

            if _label_map.shape[0] != self.H or _label_map.shape[1] != self.W:
                _label_map = cv2.resize(_label_map, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

            label_map.append(_label_map)
            lang_embed.append(_lang_embed)

        return label_map, lang_embed

    def get_images(self, images_folder, start_idx=0, end_idx=-1, interval = 1):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset == "replica":
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            self.image_files = image_files
            if end_idx != -1:
                self.image_files = self.image_files[start_idx:end_idx]
            else:
                self.image_files = self.image_files[start_idx:]
            if interval != -1:
                self.image_files = self.image_files[::interval]
            for key in tqdm(self.image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                rgb_image = cv2.imread(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.dataset_path}/depth_images/{depth_image_name}.png"))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

        elif self.trajmanager.which_dataset == "tum":
            self.image_files = self.trajmanager.color_paths
            if end_idx != -1:
                self.image_files = self.image_files[start_idx:end_idx]
            else:
                self.image_files = self.image_files[start_idx:]
            if interval != -1:
                self.image_files = self.image_files[::interval]
            for i in tqdm(range(len(self.image_files))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

        elif self.trajmanager.which_dataset == "scannet":
            images_folder = os.path.join(self.dataset_path, "rgb")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy(), key=lambda x: int(x.split(".")[0]))
            self.image_files = image_files
            if end_idx != -1:
                self.image_files = self.image_files[start_idx:end_idx]
            else:
                self.image_files = self.image_files[start_idx:]
            if interval != -1:
                self.image_files = self.image_files[::interval]
            for key in tqdm(self.image_files):
                img_file = os.path.join(self.dataset_path, "rgb", key)
                depth_file = os.path.join(self.dataset_path, "depth", key.replace("jpg", "png"))
                rgb_image = cv2.imread(img_file)
                rgb_image = cv2.resize(rgb_image, (self.W, self.H))
                depth_image = np.array(o3d.io.read_image(depth_file))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path)
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def is_motion_exceeded(self, pose1, pose2, rot_thr, trans_thr):
        """
        Check if the rotation or translation between two poses exceeds the threshold.

        Args:
            pose1, pose2: 4x4 numpy arrays representing poses.
            rot_thr: Rotation threshold in degrees.
            trans_thr: Translation threshold in meters.

        Returns:
            True if exceeded, False otherwise.
        """
        if rot_thr == 0.0 or trans_thr == 0.0:
            return True

        rot1 = pose1[:3, :3]
        rot2 = pose2[:3, :3]
        r1 = R.from_matrix(rot1)
        r2 = R.from_matrix(rot2)
        rot_diff = r1.inv() * r2
        angle_diff = rot_diff.magnitude()
        angle_diff_deg = np.degrees(angle_diff)

        trans1 = pose1[:3, 3]
        trans2 = pose2[:3, 3]
        trans_diff = np.linalg.norm(trans1 - trans2)

        if angle_diff_deg > rot_thr or trans_diff > trans_thr:
            return True
        return False

    def quaternion_multiply(self, q1, Q2):
        # Multiply quaternion q1 with each quaternion in Q2
        x0, y0, z0, w0 = q1
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

    def set_downsample_filter(self, downsample_scale):
        # Calculate downsample indices and precomputed x/y values
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        pick_idxs = ((a+b).flatten(),)
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        x_pre = (u-self.cx)/self.fx
        y_pre = (v-self.cy)/self.fy
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img, sem_label_map=None):
        # Downsample and generate point cloud from depth and RGB images
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]
        if sem_label_map is not None:
            _sem_label_map = torch.from_numpy(sem_label_map).flatten()[self.downsample_idxs][zero_filter]
        else:
            _sem_label_map = None
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), _sem_label_map, zero_filter

    def eliminate_overlapped2(self, distances, threshold):
        # Filter out overlapped points based on distance threshold
        new_p_indices = np.where(distances>threshold)
        return new_p_indices

    def align(self, model, data):
        # Align two point clouds using SVD
        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))
        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))
        model_aligned = rot * model + trans
        alignment_error = model_aligned - data
        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]
        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):
        # Evaluate Absolute Trajectory Error (ATE)
        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T
        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T
        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)
        avg_trans_error = trans_error.mean()
        return avg_trans_error