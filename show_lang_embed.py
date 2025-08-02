import argparse
from copy import deepcopy
import copy
import os
import time
import cv2
import torch
import torch.multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import TextBox, Slider

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.traj_utils import TrajManager
import sys
from utils.openclip_encoder import OpenCLIPNetwork

import open3d as o3d
from scene.shared_objs import SharedCam

def unitquat_to_rotmat(quat):
    """
    Converts unit quaternion into rotation matrix representation.
    Args:
        quat (...x4 tensor, XYZW convention): batch of unit quaternions.
    Returns:
        batch of rotation matrices (...x3x3 tensor).
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)
    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = - x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)
    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = - x2 - y2 + z2 + w2
    return matrix

def unflatten_batch_dims(tensor, batch_shape):
    """Revert flattening of a tensor."""
    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)

def flatten_batch_dims(tensor, end_dim):
    """Flatten multiple batch dimensions into a single one, or add a batch dimension if there is none."""
    batch_shape = tensor.shape[:end_dim+1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape

def rotmat_to_unitquat(R):
    """
    Converts rotation matrix to unit quaternion representation.
    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """
    matrix, batch_shape = flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert((D1, D2) == (3,3)), "Input should be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return unflatten_batch_dims(quat, batch_shape)

def build_rotation(q):
    """Convert quaternion to rotation matrix."""
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r, x, y, z = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def quat_mult(q1, q2):
    """Quaternion multiplication."""
    x1, y1, z1, w1 = q1.T
    x2, y2, z2, w2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def on_click(x, y, button, pressed, mouse_pos):
    if button == button.left and pressed:
        mouse_pos[0] = x
        mouse_pos[1] = y
        print(f"Mouse clicked at: {x}, {y}")

def start_listener(mouse_pos):
    from pynput.mouse import Listener
    with Listener(on_click=lambda x, y, button, pressed: on_click(x, y, button, pressed, mouse_pos)) as listener:
        listener.join()

def traj_load_poses(path):
    """Load camera poses from file."""
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    R_list, T_list = [], []
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w = np.linalg.inv(c2w)
        R_list.append(c2w[:3, :3].transpose())
        T_list.append(c2w[:3, 3])
        poses.append(c2w)
    return R_list, T_list

def create_camera_mesh(pose, scale=0.06, color=[1, 0, 0]):
    """Create a simple camera mesh at a given pose."""
    camera = o3d.geometry.LineSet()
    points = np.array([
        [0, 0, 0],
        [scale, scale, scale],
        [scale, -scale, scale],
        [-scale, -scale, scale],
        [-scale, scale, scale]
    ])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    camera.points = o3d.utility.Vector3dVector((pose[:3, :3] @ points.T).T + pose[:3, 3])
    camera.lines = o3d.utility.Vector2iVector(lines)
    camera.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return camera

def inverse_transform(R, T):
    """Convert R, T back to original pose."""
    recovered_pose = np.eye(4)
    recovered_pose[:3, :3] = R.transpose()
    recovered_pose[:3, 3] = T
    pose = np.linalg.inv(recovered_pose)
    return pose

class MyVisualizer():
    def __init__(self, args, mouse_pos):
        # Load dataset and configuration paths
        self.dataset_path = args.dataset_path
        self.config = args.config
        self.scene_npz = args.scene_npz
        self.dataset_type = args.dataset_type
        self.view_scale = args.view_scale
        self.mouse_pos = mouse_pos
        self.output_path = args.output_path
        self.output_M_path = os.path.join(self.output_path, "mapping_images")
        os.makedirs(self.output_M_path, exist_ok=True)
        self.show_voxel = args.show_voxel
        voxel_size = 0.08
        self.gs_mask_moes = 0

        # Load test images and camera parameters
        self.test_rgb_img, self.test_depth_img = self.get_test_image(self.dataset_type, self.dataset_path, f"{self.dataset_path}/images")
        with open(self.config) as camera_parameters_file:
            camera_parameters_ = camera_parameters_file.readlines()
        camera_parameters = camera_parameters_[2].split()
        self.W = int(camera_parameters[0])
        self.H = int(camera_parameters[1])
        self.fx = float(camera_parameters[2])
        self.fy = float(camera_parameters[3])
        self.cx = float(camera_parameters[4])
        self.cy = float(camera_parameters[5])
        self.depth_scale = float(camera_parameters[6])
        self.depth_trunc = float(camera_parameters[7])

        # Load language model and embeddings
        self.clip_model = OpenCLIPNetwork(device="cuda")
        self.input_text = 'stool'
        self.threshold = 2.7
        lang_path = os.path.join(os.path.dirname(self.scene_npz), "global_feature.npz")
        self.lang_embed = None
        if os.path.exists(lang_path):
            self.lang_embed = np.load(lang_path)["feature"]
            self.lang_embed = torch.from_numpy(self.lang_embed).cuda().contiguous()
            self.lang_embed = torch.nn.functional.normalize(self.lang_embed, p=2, dim=1)
            print(f"Loaded language embedding: {self.lang_embed.shape}")

        # Load and global labels if available
        self.current_show_label = 0
        global_label_path = os.path.join(os.path.dirname(self.scene_npz), "global_label.npy")
        if os.path.exists(global_label_path):
            self.global_voxel_label = np.load(global_label_path)
            self.global_voxel_label = torch.from_numpy(self.global_voxel_label).cuda().contiguous()
            print(f"Loaded global label: {self.global_voxel_label.shape}")

        # Initialize state variables
        self.min_xyz = None
        self.max_xyz = None
        self.capture = False
        self.on_M_capture = False
        self.show_voxel = False
        self.on_J_pressed = False
        self.highlighe_labels = []
        self.start_w2c = None
        self.pre_label_2_color = None
        self.x_pian = 141
        self.y_pian = 127

        # Camera intrinsic matrix
        k = torch.tensor([self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3).cuda()
        self.K = k

        # Initialize camera object
        rendered_cam = SharedCam(FoVx=self.focal2fov(self.fx, self.W), FoVy=self.focal2fov(self.fy, self.H),
                                image=self.test_rgb_img, depth_image=self.test_depth_img,
                                cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)

        # Load scene parameters from npz file
        scene_npz = np.load(self.scene_npz)
        self.xyz = torch.tensor(scene_npz["xyz"]).cuda().float().contiguous()
        self.opacity = torch.tensor(scene_npz["opacity"]).cuda().float().contiguous()
        self.scales = torch.tensor(scene_npz["scales"]).cuda().float().contiguous()
        self.rotation = torch.tensor(scene_npz["rotation"]).cuda().float().contiguous()
        self.shs = torch.tensor(scene_npz["shs"]).cuda().float().contiguous()
        self.gs_2_voxel_index = torch.tensor(scene_npz["voxel_index"]).cuda().int().contiguous()
        print(f"xyz shape: {self.xyz.shape}")
        self.with_sem_labels = False

        # Load camera trajectory
        if "R_list" not in scene_npz or "T_list" not in scene_npz:
            print("No R_list or T_list in scene_npz")
            self.R_list = [np.eye(3)]
            self.t_list = [np.zeros(3)]
        else:
            self.R_list = scene_npz["R_list"]
            self.t_list = scene_npz["T_list"]

        self.RT_list = []
        for i in range(len(self.R_list)):
            RT = np.eye(4)
            RT[:3, :3] = self.R_list[i]
            RT[:3, 3] = self.t_list[i]
            self.RT_list.append(RT)
        self.RT_list = torch.tensor(self.RT_list).cuda().float().contiguous()

        # Try to load trajectory from traj.txt
        traj_path = os.path.join(self.dataset_path, "traj.txt")
        self.R_list, self.t_list = traj_load_poses(traj_path)

        # Set initial camera pose
        rendered_cam.R = torch.tensor(self.R_list[0]).float().cuda()
        rendered_cam.t = torch.tensor(self.t_list[0]).float().cuda()
        rendered_cam.update_matrix()
        rendered_cam.on_cuda()
        w2c = np.eye(4)
        if self.start_w2c is not None:
            w2c = self.start_w2c
        else:
            w2c[:3, :3] = self.R_list[0]
            w2c[:3, 3] = self.t_list[0]

        # Create matplotlib window for UI
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax.axis('off')

        geometries = []

        # Compute inverse camera poses for visualization
        self.RT_list_inv = []
        for i in range(len(self.RT_list)):
            R = self.RT_list[i].cpu().numpy()[:3, :3]
            T = self.RT_list[i].cpu().numpy()[:3, 3]
            self.RT_list_inv.append(inverse_transform(R, T))
        self.RT_list_inv = torch.tensor(self.RT_list_inv).cuda().float().contiguous()

        # Create camera mesh for visualization
        self.camera_mesh = []
        self.special_cam = None
        for i in range(len(self.RT_list)):
            self.camera_mesh.append(create_camera_mesh(self.RT_list_inv[i].cpu().numpy()))

        # Create trajectory line for visualization
        self.trajectory = []
        for i in range(len(self.RT_list_inv)):
            self.trajectory.append(self.RT_list_inv[i].cpu().numpy()[:3, 3])
        self.trajectory_line = o3d.geometry.LineSet()
        if len(self.trajectory) > 1:
            trajectory_points = np.array(self.trajectory)
            trajectory_lines = [[i, i + 1] for i in range(len(self.trajectory) - 1)]
            self.trajectory_line.points = o3d.utility.Vector3dVector(trajectory_points)
            self.trajectory_line.lines = o3d.utility.Vector2iVector(trajectory_lines)
            self.trajectory_line.colors = o3d.utility.Vector3dVector([[1, 1, 0] for _ in trajectory_lines])

        # Create text box for language input
        self.text_box = TextBox(plt.axes([0.25, 0.8, 0.65, 0.1]), 'Input text:')
        self.text_box.label.set_fontsize(17)
        self.text_box.text_disp.set_fontsize(17)
        self.text_box.on_submit(self.on_submit)

        # Create slider for threshold adjustment
        self.slider_ax = plt.axes([0.25, 0.5, 0.65, 0.1])
        self.slider = Slider(self.slider_ax, 'Threshold', 0.0, 4.0, valinit=3.2)
        self.slider.label.set_fontsize(17)
        self.slider.on_changed(self.on_slider_change)

        # Initialize Open3D visualizer and register key callbacks
        self.view_mode = "color"
        self.feature_mode = "voxel"
        self.gs_mask = None
        self.xyz_trans = torch.zeros(3).cuda()
        self.xyz_per_trans = 0.005
        self.object_view_mode = False
        vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis = vis
        vis.create_window(width=int(self.W * self.view_scale), height=int(self.H * self.view_scale), visible=True)
        vis.register_key_callback(ord("T"), self.on_T_key_press)      # Switch color/label mode
        vis.register_key_callback(ord("J"), self.on_J_key_press)      # Highlight object
        vis.register_key_callback(ord("K"), self.on_K_key_press)      # Screenshot
        vis.register_key_callback(ord("O"), self.on_O_key_press)      # Print current view
        vis.register_key_callback(ord("M"), self.on_M_key_press)      # Switch camera view
        vis.register_key_callback(ord("P"), self.on_P_key_press)      # Downsample point cloud
        vis.register_key_callback(ord("="), self.on_equl_key_press)   # Save current mask point cloud
        vis.register_key_callback(ord("L"), self.on_L_new_key_press)  # Show voxel

        self.mouse_x = 0
        self.mouse_y = 0

        # Render initial point cloud
        self.bg = torch.tensor([1.0, 1.0, 1.0]).float().cuda()
        with torch.no_grad():
            time1 = time.time()
            im, depth, visibility_filter = self.render(rendered_cam, self.xyz, self.opacity, self.scales, self.rotation, self.shs, self.bg)
            print(f"Render Time cost: {time.time() - time1}")
            cv2.imwrite(f"{self.output_path}/init.png", cv2.cvtColor(im.cpu().detach().numpy().transpose(1,2,0)*255, cv2.COLOR_BGR2RGB))
        init_pts, init_cols = self.rgbd2pcd(im, depth, w2c, k)
        pcd = o3d.geometry.PointCloud()
        pcd.points = init_pts
        pcd.colors = init_cols
        vis.add_geometry(pcd)

        # Set up Open3D view control and rendering options
        view_control = vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        view_w2c = w2c.astype(np.float64)
        view_k = deepcopy(k) * self.view_scale
        view_k = view_k.cpu().numpy().astype(np.float64)
        view_k[2, 2] = 1
        cparams.extrinsic = view_w2c
        cparams.intrinsic.intrinsic_matrix = view_k
        cparams.intrinsic.height = int(self.H * self.view_scale)
        cparams.intrinsic.width = int(self.W * self.view_scale)
        view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
        render_options = vis.get_render_option()
        render_options.point_size = float(self.view_scale)
        render_options.light_on = False

        # Initialize rendering state variables
        self.change_views = False
        self.changed_w2c = np.eye(4)
        self.mapping_cam_index = 328
        self.start_rgbd_time = 0
        self.start_sem_time = 0
        self.render_counts_rgbd = 0
        self.render_counts2_sem = 0
        self.render_rgbd_total_time = 0
        self.render_sem_total_time = 0

        # Main interactive rendering loop
        while True:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / self.view_scale
            k[2,2] = 1
            w2c = cam_params.extrinsic

            if self.change_views:
                w2c = self.changed_w2c
                self.change_views = False
                cparams.extrinsic = w2c.astype(np.float64)
                view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

            self.w2c = w2c

            R = torch.tensor(w2c[:3,:3]).cuda()
            t = torch.tensor(w2c[:3,3]).cuda()
            rendered_cam.R = R
            rendered_cam.t = t
            rendered_cam.update_matrix()
            rendered_cam.on_cuda()

            # Render color or label mode
            if self.view_mode == "color":
                with torch.no_grad():
                    self.start_rgbd_time = time.time()
                    im, depth, visibility_filter = self.render(rendered_cam, self.xyz, self.opacity, self.scales, self.rotation, self.shs, self.bg)
                    self.render_counts_rgbd += 1
                    self.render_rgbd_total_time += time.time() - self.start_rgbd_time
            elif self.view_mode == "label" and self.gs_mask is not None:
                with torch.no_grad():
                    self.start_sem_time = time.time()
                    im, depth, visibility_filter = self.render(rendered_cam, self.xyz, self.opacity, self.scales, self.rotation, self.shs, self.bg, gs_mask=self.gs_mask)
                    self.render_counts2_sem += 1
                    self.render_sem_total_time += time.time() - self.start_sem_time

            pts, cols = self.rgbd2pcd(im, depth, w2c, k)
            pcd.points = pts
            pcd.colors = cols

            # Show voxel grid if requested
            if self.show_voxel:
                for geo in geometries:
                    vis.remove_geometry(geo)
                pcd2 = copy.deepcopy(pcd)
                num1 = len(pcd2.points)
                cl, ind = pcd2.remove_radius_outlier(nb_points=40, radius=0.05)
                pcd2 = pcd2.select_by_index(ind)
                num2 = len(pcd2.points)
                pcd2 = pcd2.voxel_down_sample(voxel_size)
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2, voxel_size)
                geometries = []
                for voxel in voxel_grid.get_voxels():
                    center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
                    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                        o3d.geometry.AxisAlignedBoundingBox(
                            min_bound=center - voxel_size / 2,
                            max_bound=center + voxel_size / 2
                        )
                    )
                    bbox.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]] * 12))
                    rx, ry, rz = 0, 0, 0
                    R = bbox.get_rotation_matrix_from_xyz((rx, ry, rz))
                    center_of_bbox = (bbox.get_min_bound() + bbox.get_max_bound()) / 2
                    bbox.rotate(R, center=center_of_bbox)
                    geometries.append(bbox)
                for geo in geometries:
                    vis.add_geometry(geo)
                view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
                self.show_voxel = False

            vis.update_geometry(pcd)

            # Save screenshot if requested
            if self.capture:
                timestamps = time.time()
                output_name = f"{self.output_path}/{timestamps}.png"
                cv2.imwrite(output_name, cv2.cvtColor(im.cpu().detach().numpy().transpose(1,2,0)*255, cv2.COLOR_BGR2RGB))
                print(f"Saving image to {output_name}")
                depth_name = output_name.replace(".png", "_depth.png")
                depth = depth.cpu().detach().numpy()
                depth = depth.transpose(1,2,0)
                depth = depth * 6553.5
                depth = depth.astype(np.uint16)
                cv2.imwrite(depth_name, depth)
                print(f"Saving depth to {depth_name}")
                self.capture = False

            # Save mapping image if requested
            if self.on_M_capture:
                output_name = f"{self.output_M_path}/{self.mapping_cam_index}.png"
                cv2.imwrite(output_name, cv2.cvtColor(im.cpu().detach().numpy().transpose(1,2,0)*255, cv2.COLOR_BGR2RGB))
                print(f"Saving image to {output_name}")
                self.on_M_capture = False

            if not vis.poll_events():
                break
            vis.update_renderer()
            plt.pause(0.01)

            mouse_x = self.mouse_pos[0] - self.x_pian
            mouse_y = self.mouse_pos[1] - self.y_pian

        vis.destroy_window()

    def rgbd2pcd(self, color, depth, w2c, intrinsics):
        width, height = color.shape[2], color.shape[1]
        CX = intrinsics[0][2]
        CY = intrinsics[1][2]
        FX = intrinsics[0][0]
        FY = intrinsics[1][1]
        xx = torch.tile(torch.arange(width).cuda(), (height,))
        yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
        xx = (xx - CX) / FX
        yy = (yy - CY) / FY
        z_depth = depth[0].reshape(-1)
        pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(torch.tensor(w2c).cuda().float())
        pts = (c2w @ pts4.T).T[:, :3]
        pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
        return pts, cols

    def fov2focal(self, fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    def focal2fov(self, focal, pixels):
        return 2*math.atan(pixels/(2*focal))

    def on_submit(self, text):
        print('Input text:', text)
        self.input_text = text
        self.on_slash_key_press()

    def on_slider_change(self, value):
        self.threshold = value
        self.on_slash_key_press()

    def on_5_key_press(self, vis):
        self.gs_mask_moes = 1 if self.gs_mask_moes == 0 else 0
        print(f"gs_mask_moes: {self.gs_mask_moes}")
        return True

    def on_slash_key_press(self):
        print("Slash key pressed")
        text_embed = self.clip_model.encode_text([self.input_text], device='cuda').float()
        if self.feature_mode == "voxel":
            if self.lang_embed is None:
                print("No lang_embed")
                return False
            cos_sim = torch.matmul(text_embed, self.lang_embed.T)
            print(f'max cos_sim: {cos_sim.max()}')
            mask = cos_sim > self.threshold
            mask = mask.squeeze(0)
            select_voxel_index = torch.nonzero(mask, as_tuple=True)[0]
            self.gs_mask = torch.isin(self.gs_2_voxel_index, select_voxel_index)
            print(f"Select gs num: {self.gs_mask.sum()}")
        return True

    def SH2RGB(self,sh):
        C0 = 0.28209479177387814
        return sh * C0 + 0.5

    def RGB2SH(self, rgb):
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

    def on_equl_key_press(self, vis):
        if self.gs_mask is not None:
            points = self.xyz[self.gs_mask]
            colors = self.shs[self.gs_mask]
            colors = self.SH2RGB(colors)
            colors = colors.squeeze(1)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy())
            pcd.colors = o3d.utility.Vector3dVector(colors.cpu().detach().numpy())
            save_path = f"{self.output_path}/mask_my.ply"
            print(f"Save mask to {save_path}")
            o3d.io.write_point_cloud(save_path, pcd)

    def on_L_new_key_press(self, vis):
        self.show_voxel = True
        print("Show Current Voxel")
        return True

    def on_T_key_press(self, vis):
        if self.view_mode == "color":
            print("Switch to label mode")
            self.view_mode = "label"
        else:
            print("Switch to color mode")
            self.view_mode = "color"
        return True

    def on_O_key_press(self, vis):
        print(self.w2c)

    def on_J_key_press(self, vis):
        if self.gs_mask is not None:
            gs_mask_label = self.gs_mask
        else:
            return True
        if self.on_J_pressed:
            self.shs[self.last_gs_mask] = self.last_sh
            self.on_J_pressed = False
        self.last_gs_mask = copy.deepcopy(gs_mask_label)
        self.last_sh = copy.deepcopy(self.shs[gs_mask_label])
        rgb = self.SH2RGB(self.shs[gs_mask_label])
        rgb = rgb + torch.tensor([0.0, 0.0, 0.5]).cuda().float()
        rgb[rgb > 1] = 1
        self.shs[gs_mask_label] = self.RGB2SH(rgb)
        self.on_J_pressed = True
        return True


    def on_K_key_press(self, vis):
        print("Enter key pressed")
        self.capture = True
        return True

    def on_P_key_press(self, vis):
        print("P key pressed")
        with torch.no_grad():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.xyz.cpu().detach().numpy())
            voxel_size = 0.03
            downpcd = pcd.voxel_down_sample(voxel_size)
            down_pts = np.asarray(downpcd.points)
            tree = cKDTree(self.xyz.cpu().detach().numpy())
            _, idx = tree.query(down_pts)
            idx = torch.tensor(idx).cuda()
            self.xyz = self.xyz[idx]
            self.opacity = self.opacity[idx]
            self.scales = self.scales[idx]
            self.rotation = self.rotation[idx]
            self.shs = self.shs[idx]
            if self.with_sem_labels:
                self.sem_labels = self.sem_labels[idx]
                self.colors_precomp = self.colors_precomp[idx]
        return True

    def on_M_key_press(self, vis):
        self.mapping_cam_index += 1
        if self.mapping_cam_index >= len(self.R_list):
            self.mapping_cam_index = 0
        R = self.R_list[self.mapping_cam_index]
        t = self.t_list[self.mapping_cam_index]
        self.changed_w2c[:3, :3] = R
        self.changed_w2c[:3, 3] = t
        self.change_views = True
        print(f"Change to {self.mapping_cam_index} camera")
        self.on_M_capture = True
        return True

    def on_L_key_press(self, vis):
        print("L key pressed")
        with torch.no_grad():
            self.scales += 0.001
            self.scales[self.scales > 1] = 1
        return True

    def on_mouse_click(self, vis, button, action, mods):
        print(f"Mouse clicked: button={button}, action={action}, mods={mods}")
        return True

    def on_mouse_move(self, vis, x, y):
        self.mouse_x = x
        self.mouse_y = y
        print(f"Mouse moved to position: x={self.mouse_x}, y={self.mouse_y}")
        return True

    def get_test_image(self, dataset_type, dataset_path, images_folder):
        if dataset_type == "replica":
            images_folder = os.path.join(dataset_path, "images")
            image_files = sorted(os.listdir(images_folder))
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
            rgb_image = cv2.imread(f"{dataset_path}/images/{image_name}.jpg")
            depth_image = np.array(o3d.io.read_image(f"{dataset_path}/depth_images/{depth_image_name}.png")).astype(np.float32)
        elif dataset_type == "tum":
            rgb_folder = os.path.join(dataset_path, "rgb")
            depth_folder = os.path.join(dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        elif dataset_type == "scannet":
            rgb_folder = os.path.join(dataset_path, "rgb")
            depth_folder = os.path.join(dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            rgb_image = cv2.resize(rgb_image, (640, 480))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        elif dataset_type == "others":
            rgb_folder = os.path.join(dataset_path, images_folder)
            rgb_file = os.listdir(rgb_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = rgb_image[:, :, 0]
        elif dataset_type == "self_capture":
            rgb_folder = os.path.join(dataset_path, "rgb")
            depth_folder = os.path.join(dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        return rgb_image, depth_image

    def render(self, viewpoint_camera, xyz: torch.Tensor, opicity: torch.Tensor, scales:torch.Tensor,
               rotations:torch.Tensor, shs:torch.Tensor, bg_color : torch.Tensor, scaling_modifier = 1.0, gs_mask=None):
        """
        Render the scene.
        Background tensor (bg_color) must be on GPU!
        """
        if gs_mask is not None:
            xyz = xyz[gs_mask]
            opicity = opicity[gs_mask]
            scales = scales[gs_mask]
            rotations = rotations[gs_mask]
            shs = shs[gs_mask]

        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=False, device="cuda") + 0
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        active_sh_degree = 0

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = xyz
        means2D = screenspace_points
        opacity = opicity

        colors_precomp = None
        cov3D_precomp = None

        depth_image, rendered_image, radii, is_used = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        visibility_filter = radii > 0
        return rendered_image, depth_image, visibility_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script parameters")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--config", type=str, help="Path to the camera parameters")
    parser.add_argument("--scene_npz", type=str, help="Path to the scene npz file")
    parser.add_argument("--dataset_type", type=str, help="Type of the dataset, e.g., replica, scannet, tum, others")
    parser.add_argument("--view_scale", type=float, default=1.0, help="Scale of the view")
    parser.add_argument("--show_voxel", action="store_true", help="Show voxel or not")
    args = parser.parse_args()
    
    args.output_path = os.path.join(os.path.dirname(args.scene_npz), "vis")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    manager = mp.Manager()
    mouse_pos = manager.list([0, 0])
    vis = MyVisualizer(args, mouse_pos)
    
    