
import os

import cv2 as cv
import numpy as np
import open3d as o3d
import torch


class Replica:
    def __init__(self, data_source, get_color=False, get_sem=False, split=None, downsample_rate=5):
    
        self.name = data_source.split("/")[-1]

        self.get_color = get_color
        self.data_source = data_source

        self.rgb_list = os.listdir(os.path.join(self.data_source, "images"))
        self.rgb_list.sort()
        self.rgb_path_list = [os.path.join(self.data_source, "images", rgb) for rgb in self.rgb_list]
        
        self.depth_list = os.listdir(os.path.join(self.data_source, "depth_images"))
        self.depth_list.sort()
        self.depth_path_list = [os.path.join(self.data_source, "depth_images", depth) for depth in self.depth_list]

        self.pose_txt = os.path.join(self.data_source, "traj.txt")
        self.pose_list = self.replica_load_poses(self.pose_txt)
        
        # 读取第一张图片，获取图像尺寸
        img = cv.imread(os.path.join(self.data_source, "images", self.rgb_list[0]))
        self.H, self.W, _ = img.shape
        self.K = np.array([[600.0, 0.0, 599.5],
                    [0.0, 600.0, 339.5],
                    [0.0, 0.0, 1.0]])
        
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        
        self.depth_scale = 6553.5
        self.depth_trunc = 12.0 
        
        self.downsample_rate = downsample_rate
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        self.get_sem = get_sem

    def __len__(self):
        return len(self.pose_list)
        
    def replica_load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return np.array(poses)
    
    def set_downsample_filter(self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre
    
    def gen_langual_embed_from_langSplat(self, rgb_img_name, level="l"):
        '''
        生成语义嵌入，这里先直接读取LangSplat输出的语义嵌入
        '''
        language_feature_root = os.path.join(self.data_source, 'language_features')
        lang_embed = np.load(os.path.join(language_feature_root, rgb_img_name[:-4] + '_f.npy'))   # K * 512
        label_map = np.load(os.path.join(language_feature_root, rgb_img_name[:-4] + '_s.npy'))
        
        level_map = {'default':0, "s":1, "m":2, "l":3}
        level_idx = level_map[level]
        label_map = label_map[level_idx]    # (H, W)
        
        # 判断label_map中是否有-1，有的话报错
        if -1 in label_map:
            # 将label_map整体+1，-1变成0，同时要再lang_embed中前面塞一行
            label_map += 1
            lang_embed = np.vstack((np.zeros((1, lang_embed.shape[1]), dtype=np.float32), lang_embed))
            # raise ValueError("label_map has -1")
        
        # final_lang_embed = np.zeros((label_map.shape[0], label_map.shape[1], lang_embed.shape[1]), dtype=np.float32)
        # # label_map中存储的是lang_embed的索引，过滤掉-1的索引
        # final_lang_embed[label_map != -1] = lang_embed[label_map[label_map != -1].astype(np.int32)]
        
        return label_map, lang_embed
    
    def __getitem__(self, idx):
        
        rgb_image = cv.imread(self.rgb_path_list[idx])
        depth_image = np.array(o3d.io.read_image(self.depth_path_list[idx]))
        
        pose = self.pose_list[idx]
        
        label_map, lang_embed = self.gen_langual_embed_from_langSplat(self.rgb_list[idx])
        
        cam_points, colors, z_values, filter, sem_img = self.downsample_and_make_pointcloud2(depth_image, rgb_image, label_map)
        
        workd_points = (pose[:3, :3] @ cam_points.T).T + pose[:3, 3]
        
        if self.get_color:
            return workd_points, colors, np.array(pose)
        
        if self.get_sem:
            return workd_points, np.array(pose), sem_img, lang_embed
        
        return workd_points, np.array(pose)
        
        
        
    def downsample_and_make_pointcloud2(self, depth_img, rgb_img, sem_img=None):
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        self.zero_filter = zero_filter  # 留着后面filter sem时候用，这种情况一般出现在使用自己的run_sam的时候。
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]
        
        # Semantic
        if sem_img is not None:
            sem_img = torch.from_numpy(sem_img).reshape(-1,1).float()[self.downsample_idxs][zero_filter][:,0]
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), None if sem_img is None else sem_img.numpy()
    