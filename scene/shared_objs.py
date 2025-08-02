import torch
import numpy as np
import cv2
import torch.nn as nn
import copy
import math

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    Compute world-to-view transformation matrix.
    """
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    Compute projection matrix from camera parameters.
    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class SharedPoints(nn.Module):
    """
    Shared buffer for point cloud data between processes.
    """
    def __init__(self, num_points):
        super().__init__()
        self.points = torch.zeros((num_points, 3)).float()
        self.colors = torch.zeros((num_points, 3)).float()
        self.z_values = torch.zeros((num_points)).float()
        self.filter = torch.zeros((num_points)).int()
        self.using_idx = torch.zeros((1)).int()
        self.filter_size = torch.zeros((1)).int()
    
    def input_values(self, new_points, new_colors, new_z_values, new_filter):
        self.using_idx[0] = new_points.shape[0]
        self.points[:self.using_idx[0], :] = new_points
        self.colors[:self.using_idx[0], :] = new_colors
        self.z_values[:self.using_idx[0]] = new_z_values
        self.filter_size[0] = new_filter.shape[0]
        self.filter[:self.filter_size[0]] = new_filter

    def get_values(self):
        return (copy.deepcopy(self.points[:self.using_idx[0], :].numpy()),
                copy.deepcopy(self.colors[:self.using_idx[0], :].numpy()),
                copy.deepcopy(self.z_values[:self.using_idx[0]].numpy()),
                copy.deepcopy(self.filter[:self.filter_size[0]].numpy()))

class ShareOnlineSAM(nn.Module):
    """
    Shared buffer for Online SAM segmentation and CLIP embedding.
    """
    def __init__(self, image, downsample_idxs, max_obj_single_frame=200, clip_embedding_dim=512):
        super().__init__()
        self.H, self.W = image.shape[:2]
        self.downsample_idxs = copy.deepcopy(downsample_idxs[0])
        self.image = torch.zeros((self.H, self.W, 3)).to(torch.uint8)
        self.label_map = torch.zeros(self.downsample_idxs.shape[0]).int().cuda()
        self.using_idx_label_map = torch.zeros((1)).int()
        self.clip_embedding = torch.zeros((max_obj_single_frame, clip_embedding_dim)).float().cuda()
        self.using_idx = torch.zeros((1)).int()
        self.zero_filter = torch.zeros(self.downsample_idxs.shape[0]).int().cuda()
        self.used_zero_filter = torch.zeros((1)).int()
        
    def input_image_from_tracker(self, new_image, _zero_filter):
        self.image[:] = torch.from_numpy(new_image).to(torch.uint8)
        self.used_zero_filter[0] = _zero_filter.shape[0]
        self.zero_filter[:self.used_zero_filter[0]] = _zero_filter
        
    def get_image_from_tracker(self):
        return (copy.deepcopy(self.image.numpy()), copy.deepcopy(self.zero_filter[:self.used_zero_filter[0]]))
    
    def input_value_from_sam_model(self, new_label_map, new_clip_embedding):
        self.using_idx_label_map[0] = new_label_map.shape[0]
        self.label_map[:self.using_idx_label_map[0]] = new_label_map
        self.using_idx[0] = new_clip_embedding.shape[0]
        self.clip_embedding[:self.using_idx[0], :] = new_clip_embedding
        
    def get_value_from_sam_model(self):
        return (copy.deepcopy(self.label_map[:self.using_idx_label_map[0]]),
                copy.deepcopy(self.clip_embedding[:self.using_idx[0], :]))

class SharedVDBPoints(nn.Module):
    """
    Shared buffer for VDB fusion points and semantic features.
    """
    def __init__(self, num_points):
        super().__init__()
        self.points = torch.zeros((num_points, 3)).float()
        self.points_voxel_index = torch.zeros((num_points)).int()
        self.points_voxel_tsdf = torch.zeros((num_points)).float()
        self.points_voxel_weight = torch.zeros((num_points)).float()
        self.points_label = torch.zeros((num_points)).int()
        self.label_feature = torch.zeros(1000, 512).float()
        self.pose = torch.zeros((4, 4)).float()
        self.using_idx = torch.zeros((1)).int()
        self.label_feature_idx = torch.zeros((1)).int()
    
    def input_values_from_tracker(self, new_points, pose, new_point_label=None, new_label_feature=None):
        self.using_idx[0] = new_points.shape[0]
        self.points[:self.using_idx[0], :] = new_points
        self.pose[:, :] = pose
        if new_point_label is not None:
            self.points_label[:self.using_idx[0]] = new_point_label
        else:
            self.points_label[:self.using_idx[0]] = -1
        if new_label_feature is not None:
            self.label_feature_idx[0] = new_label_feature.shape[0]
            self.label_feature[:self.label_feature_idx[0], :] = new_label_feature
        else:
            self.label_feature_idx[0] = 0
        
    def get_values_from_tracker(self):
        return (copy.deepcopy(self.points[:self.using_idx[0], :].numpy()),
                copy.deepcopy(self.pose.numpy()),
                copy.deepcopy(self.points_label[:self.using_idx[0]].numpy()) if self.points_label[0] != -1 else None,
                copy.deepcopy(self.label_feature[:self.label_feature_idx[0], :].numpy()) if self.label_feature_idx[0] != 0 else None)

    def input_value_from_vdb(self, points_voxel_index, points_voxel_tsdf, points_voxel_weight):
        assert points_voxel_index.shape[0] == self.using_idx[0] and \
               points_voxel_tsdf.shape[0] == self.using_idx[0] and \
               points_voxel_weight.shape[0] == self.using_idx[0], \
               "Length of points_voxel_index, points_voxel_tsdf, points_voxel_weight should be equal to using_idx[0]"
        self.points_voxel_index[:self.using_idx[0]] = points_voxel_index
        self.points_voxel_tsdf[:self.using_idx[0]] = points_voxel_tsdf
        self.points_voxel_weight[:self.using_idx[0]] = points_voxel_weight
        
    def get_value_from_vdb(self):
        return (copy.deepcopy(self.points_voxel_index[:self.using_idx[0]].numpy()),
                copy.deepcopy(self.points_voxel_tsdf[:self.using_idx[0]].numpy()),
                copy.deepcopy(self.points_voxel_weight[:self.using_idx[0]].numpy()))

class SharedGaussians(nn.Module):
    """
    Shared buffer for Gaussian parameters between processes.
    """
    def __init__(self, num_points):
        super().__init__()
        self.xyz = torch.zeros((num_points, 3)).float().cuda()
        self.colors = torch.zeros((num_points, 3)).float().cuda()
        self.rots = torch.zeros((num_points, 4)).float().cuda()
        self.scales = torch.zeros((num_points, 3)).float().cuda()
        self.z_values = torch.zeros((num_points)).float().cuda()
        self.trackable_filter = torch.zeros((num_points)).long().cuda()
        self.voxel_index = torch.zeros((num_points), dtype=torch.int).cuda()
        self.using_idx = torch.zeros((1)).int().cuda()
        self.filter_size = torch.zeros((1)).int().cuda()

    def input_values(self, new_xyz, new_colors, new_rots, new_scales, new_z_values, new_trackable_filter, new_voxel_index):
        self.using_idx[0] = new_xyz.shape[0]
        self.xyz[:self.using_idx[0], :] = new_xyz
        self.colors[:self.using_idx[0], :] = new_colors
        self.rots[:self.using_idx[0], :] = new_rots
        self.scales[:self.using_idx[0], :] = new_scales
        self.z_values[:self.using_idx[0]] = new_z_values
        self.voxel_index[:self.using_idx[0]] = new_voxel_index
        self.filter_size[0] = new_trackable_filter.shape[0]
        self.trackable_filter[:self.filter_size[0]] = new_trackable_filter
    
    def get_values(self):
        return (copy.deepcopy(self.xyz[:self.using_idx[0], :]),
                copy.deepcopy(self.colors[:self.using_idx[0], :]),
                copy.deepcopy(self.rots[:self.using_idx[0], :]),
                copy.deepcopy(self.scales[:self.using_idx[0], :]),
                copy.deepcopy(self.z_values[:self.using_idx[0]]),
                copy.deepcopy(self.trackable_filter[:self.filter_size[0]]),
                copy.deepcopy(self.voxel_index[:self.using_idx[0]]))

class SharedTargetPoints(nn.Module):
    """
    Shared buffer for target points used in tracking.
    """
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.xyz = torch.zeros((num_points, 3)).float()
        self.rots = torch.zeros((num_points, 4)).float()
        self.scales = torch.zeros((num_points, 3)).float()
        self.using_idx = torch.zeros((1)).int()

    def input_values(self, new_xyz, new_rots, new_scales):
        self.using_idx[0] = new_xyz.shape[0]
        if self.using_idx[0] > self.num_points:
            print("Too many target points")
        self.xyz[:self.using_idx[0], :] = new_xyz
        self.rots[:self.using_idx[0], :] = new_rots
        self.scales[:self.using_idx[0], :] = new_scales
    
    def get_values_tensor(self):
        return (copy.deepcopy(self.xyz[:self.using_idx[0], :]),
                copy.deepcopy(self.rots[:self.using_idx[0], :]),
                copy.deepcopy(self.scales[:self.using_idx[0], :]))

    def get_values_np(self):
        return (copy.deepcopy(self.xyz[:self.using_idx[0], :].numpy()),
                copy.deepcopy(self.rots[:self.using_idx[0], :].numpy()),
                copy.deepcopy(self.scales[:self.using_idx[0], :].numpy()))

class SharedCam(nn.Module):
    """
    Shared camera buffer for SLAM processes.
    """
    def __init__(self, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super().__init__()
        self.cam_idx = torch.zeros((1)).int()
        self.R = torch.eye(3, 3).float()
        self.t = torch.zeros((3)).float()
        self.FoVx = torch.tensor([FoVx])
        self.FoVy = torch.tensor([FoVy])
        self.image_width = torch.tensor([image.shape[1]])
        self.image_height = torch.tensor([image.shape[0]])
        self.cx = torch.tensor([cx])
        self.cy = torch.tensor([cy])
        self.fx = torch.tensor([fx])
        self.fy = torch.tensor([fy])
        
        self.original_image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0)
        
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = getWorld2View2(self.R, self.t, trans, scale).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def update_matrix(self):
        self.world_view_transform[:, :] = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        self.full_proj_transform[:, :] = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center[:] = self.world_view_transform.inverse()[3, :3]
    
    def setup_cam(self, R, t, rgb_img, depth_img):
        self.R[:, :] = torch.from_numpy(R)
        self.t[:] = torch.from_numpy(t)
        self.update_matrix()
        self.original_image[:, :, :] = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255
        self.original_depth_image[:, :, :] = torch.from_numpy(depth_img).float().unsqueeze(0)
    
    def on_cuda(self):
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()
        self.original_image = self.original_image.cuda()
        self.original_depth_image = self.original_depth_image.cuda()
        
    def on_cpu(self):
        self.world_view_transform = self.world_view_transform.cpu()
        self.projection_matrix = self.projection_matrix.cpu()
        self.full_proj_transform = self.full_proj_transform.cpu()
        self.camera_center = self.camera_center.cpu()
        self.original_image = self.original_image.cpu()
        self.original_depth_image = self.original_depth_image.cpu()

class MappingCam(nn.Module):
    """
    Camera class for mapping, used for evaluation and visualization.
    """
    def __init__(self, cam_idx, R, t, FoVx, FoVy, image, depth_image,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super().__init__()
        self.cam_idx = cam_idx
        self.R = R
        self.t = t
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.last_loss = 0.

        self.original_image = torch.from_numpy(image).float().cuda().permute(2, 0, 1) / 255
        self.original_depth_image = torch.from_numpy(depth_image).float().unsqueeze(0).cuda()
        
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, t, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.t, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]