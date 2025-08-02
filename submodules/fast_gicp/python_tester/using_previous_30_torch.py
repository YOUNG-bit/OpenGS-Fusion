import os
import sys
import time
import numpy as np
import pygicp
from matplotlib import pyplot
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import torch
import math

def quaternion_rotation_matrix(Q, t):

	r = R.from_quat(Q)
	rotation_mat = r.as_matrix()
	# rotation_mat = np.transpose(rotation_mat)
	# rotation_mat = np.linalg.inv(rotation_mat)
	T = np.empty((4, 4))
	T[:3, :3] = rotation_mat
	T[:3, 3] = [t[0], t[1], t[2]]

	T[3, :] = [0, 0, 0, 1]     
    # return np.linalg.inv(T)
	return T

def replica_load_poses(path):
	poses = []
	with open(path, "r") as f:
		lines = f.readlines()
	for i in range(2000):
		line = lines[i]
		c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
		# c2w[:3, 1] *= -1
		# c2w[:3, 2] *= -1
		# c2w = torch.from_numpy(c2w).float()
		poses.append(c2w)
	return np.array(poses)

def tum_load_poses(path):
	poses = []
	times = []
	association_file = open("associations/fr3_office.txt")
	for association_line in association_file:
		association_line = association_line.split()
		time_dis = 99
		final_idx = 0
		with open(path, "r") as f:
			lines = f.readlines()

		# 가까운 timestamp 찾기
		for i in range(len(lines)):
			if i < 3:
				continue
			line = lines[i].split()
			t = abs(float(line[0]) - float(association_line[2]))
			if time_dis > t:
				time_dis = t
				final_idx = i
    
		# 최종 gt pose 저장
		line = lines[final_idx].split()
		xyz = np.array([  	float(line[1]),
									float(line[2]),
									float(line[3])])
		q = np.array([	float(line[4]),
						float(line[5]),
						float(line[6]),
						float(line[7])])
		c2w = quaternion_rotation_matrix(q, xyz)
		poses.append(c2w)
		times.append(line[0])
	return np.array(poses), times


def make_pointcloud_from_img(depth_img, rgb_img, 
                             H, W, fx, fy, cx, cy, 
                             depth_scale, depth_trunc):
	
	depth_img = torch.from_numpy(depth_img.astype(np.float32)) / depth_scale
	v, u = torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
	x = (u - cx) * depth_img / fx
	y = (v - cy) * depth_img / fy
	points = torch.stack([x,y,depth_img], dim=-1).reshape(-1,3)
	
	depth_img_flatten = depth_img.flatten()
	filter = torch.where((depth_img_flatten!=0)&(depth_img_flatten<=depth_trunc))
	points = points[filter]
	
	if rgb_img != None:
		colors = torch.from_numpy(rgb_img).reshape(-1,3)
		colors = colors[filter]
		return points.numpy(), colors.numpy()
	else:
		return points.numpy()

def main():
	if len(sys.argv) < 5:
		print('usage: gicp_odometry2.py [dataset_path] [tum or replica] [downsample_resolution] [visualize?]')
		print("ex) python gicp_odometry2.py ./dataset/room0 0.05 true")
		return

	# List input files
	# /home/lair99/dataset/Replica/room0
	seq_path = sys.argv[1]

	if sys.argv[2] == "tum":
		mode = "tum"
	elif sys.argv[2] == "replica":
		mode = "replica"
	else:
		print("Unknown dataset")
		print('usage: gicp_odometry.py [dataset_path] ["tum" or "replica"] [downsample_resolution] [visualize?]')
		return
 
	if sys.argv[3] != "false":
		downsample_resolution = float(sys.argv[3])
	else:
		downsample_resolution = None

	if sys.argv[4] == "true":
		visualize = True
	else:
		visualize = False
	if visualize:
		vis = o3d.visualization.Visualizer()
		vis.create_window()

	pointclouds_stack = []
	# intrinsics = o3d.camera.PinholeCameraIntrinsic()
 
	if mode == "replica":
		filenames = sorted([seq_path + '/results/' + x for x in os.listdir(seq_path+'/results/') if x.endswith('.png')])
		gt_poses = replica_load_poses(seq_path + '/traj.txt')
		# intrinsics.set_intrinsics(
		# 	1200, 680, 
		# 	600.0, 600.0, 599.5, 339.5)
		W = 1200
		H = 680
		fx = 600.0
		fy = 600.0
		cx = 599.5
		cy = 339.5
		depth_scale = 6553.5
		depth_trunc = 12.0
	else:
		filenames = sorted([seq_path + '/depth/' + x for x in os.listdir(seq_path+'/depth/') if x.endswith('.png')])
		gt_poses, gt_timestamps = tum_load_poses(seq_path + '/groundtruth.txt')
		# intrinsics.set_intrinsics(
		# 	640, 480, 
		# 	535.4, 539.2, 320.1, 247.6)
		W = 640
		H = 480
		fx = 535.4
		fy = 539.2
		cx = 320.1
		cy = 247.6
		depth_scale = 5000.0
		depth_trunc = 3.0
	
	gt_traj_vis = np.array([x[:3, 3] for x in gt_poses])

	reg = pygicp.FastGICP()
	reg.set_max_correspondence_distance(0.05)

	stamps = []		# for FPS calculation
	poses = [gt_poses[0]]	# camera trajectory
 
	total_start_time = time.time()
	for i, filename in enumerate(filenames):
		start = time.time()
		debug = 0
		a = time.time()
		
		# Read depth image
		depth_image = np.array(o3d.io.read_image(filename))
		# fps
		

		print(f"debug{debug} : {time.time()-a}")
		debug += 1
		a = time.time()
  
		# points_ = o3d.geometry.PointCloud.create_from_depth_image(
		# 	depth=o3d.geometry.Image(depth_image),
		# 	intrinsic = intrinsics,
		# 	depth_scale = depth_scale,
		# 	depth_trunc = depth_trunc
		# )
		# points = np.asarray(points_.points)

		points = make_pointcloud_from_img(	depth_image, None, 
											H, W, fx, fy, cx, cy, 
											depth_scale, depth_trunc)
  
		print(f'before : {points.shape}')

		print(f"debug{debug} : {time.time()-a}")
		debug += 1
		a = time.time()

		if downsample_resolution != None:
			selected_indices = np.random.choice(len(points), size=(int)(len(points)*downsample_resolution), replace=False)
			points = points[selected_indices]
			# points = points[torch.randint(0, len(points), size=(int)(len(points)*downsample_resolution))]
		
		print(f"debug{debug} : {time.time()-a}")
		debug += 1
		a = time.time()
  
		print(f'after : {points.shape}')


		if i == 0:
			current_pose = poses[-1]
			h_points = np.column_stack((points, np.ones(len(points))))
			points_registered = np.dot(current_pose, h_points.T).T
			points_registered = points_registered[:, :3]
			reg.set_input_target(points_registered)
			map_points = points_registered.copy()
		else:
			reg.set_input_source(points)
			current_pose = reg.align(poses[-1])

		# Accumulate the delta to compute the full sensor trajectory
		# current_pose = poses[-1].dot(delta)
		
		if i != 0:
			poses.append(current_pose)
			p_t = time.time()
			points_registered = []
			h_points = np.column_stack((points, np.ones(len(points))))
			points_registered = np.dot(current_pose, h_points.T).T
			points_registered = points_registered[:, :3]

			if i % 30 == 1:
				pointclouds_stack.append(points_registered)
				# if len(pointclouds_stack) > 30:
				# 	pointclouds_stack.pop()
				map_points = pointclouds_stack[0]
				if len(pointclouds_stack) > 1:
					for idx in range(len(pointclouds_stack)-1):
						map_points = np.vstack((map_points, pointclouds_stack[idx+1]))

			# target_points_h = np.column_stack((map_points, np.ones(len(map_points))))
			# target_points = np.dot(np.linalg.inv(current_pose), target_points_h.T).T
			# target_points = target_points[:, :3]
				# map_points = pygicp.downsample(map_points, 0.05)
				reg.set_input_target(map_points)
			
			# reg.set_input_target(map_points_translated)
			print(1/(time.time() - p_t))

		# poses = T

		# fps
		stamps.append(1/(time.time()-start))
		stamps_ = stamps[-9:]
		fps = sum(stamps_) / len(stamps_)
		print('fps:%.3f' % fps)

		# visualize
		# if visualize and i%5==1:
		# 	# points_
		# 	# points_.transform(poses[-1].dot(delta))
		# 	# vis.add_geometry(points_)
		# 	# vis.update_geometry(points_)
		# 	# test
		# 	pcd = o3d.geometry.PointCloud()
		# 	pcd.points = o3d.utility.Vector3dVector(points_registered)
		# 	# pcd.transform(poses[-1].dot(delta))
		# 	vis.add_geometry(pcd)
		# 	vis.update_geometry(pcd)
   
		# 	vis.poll_events()
		# 	vis.update_renderer()
  
		# FPS calculation for the last ten frames
		
		# Plot the estimated trajectory
		traj = np.array([x[:3, 3] for x in poses])
		
		if (i % 30 == 0) or (i == len(filenames)-1):
			pyplot.clf()
			pyplot.title(f'Downsample ratio {downsample_resolution}\nfps : {fps:.2f}')
			pyplot.plot(traj[:, 0], traj[:, 1], label='g-icp trajectory', linewidth=3)
			pyplot.legend()
			pyplot.plot(gt_traj_vis[:, 0], gt_traj_vis[:, 1], label='ground truth trajectory')
			pyplot.legend()
			pyplot.axis('equal')
			pyplot.pause(1e-15)
	print(f"total fps : {1/((time.time() - total_start_time)/len(filenames))}")
	pyplot.show()
	if visualize:
		vis.run()




if __name__ == '__main__':
	print("start")
	main()
