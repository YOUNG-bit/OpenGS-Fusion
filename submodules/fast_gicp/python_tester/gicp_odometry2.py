import os
import sys
import time
import numpy as np
import pygicp
from matplotlib import pyplot
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2

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

	intrinsics = o3d.camera.PinholeCameraIntrinsic()
 
	if mode == "replica":
		filenames = sorted([seq_path + '/results/' + x for x in os.listdir(seq_path+'/results/') if x.endswith('.png')])
		gt_poses = replica_load_poses(seq_path + '/traj.txt')
		intrinsics.set_intrinsics(
			1200, 680, 
			600.0, 600.0, 599.5, 339.5)
		depth_scale = 6553.5
		depth_trunc = 12.0
	else:
		filenames = sorted([seq_path + '/depth/' + x for x in os.listdir(seq_path+'/depth/') if x.endswith('.png')])
		gt_poses, gt_timestamps = tum_load_poses(seq_path + '/groundtruth.txt')
		intrinsics.set_intrinsics(
			640, 480, 
			535.4, 539.2, 320.1, 247.6)
		depth_scale = 5000.0
		depth_trunc = 3.0
	
	gt_traj_vis = np.array([x[:3, 3] for x in gt_poses])

	reg = pygicp.FastGICP()
	reg.set_max_correspondence_distance(0.05)

	stamps = []		# for FPS calculation
	poses = [gt_poses[0]]	# camera trajectory

	for i, filename in enumerate(filenames):

		# Read depth image
		depth_image = np.array(o3d.io.read_image(filename))
		# fps
		start = time.time()
  
		points_ = o3d.geometry.PointCloud.create_from_depth_image(
			depth=o3d.geometry.Image(depth_image),
			intrinsic = intrinsics,
			depth_scale = depth_scale,
			depth_trunc = depth_trunc
		)
		points = np.asarray(points_.points)
		print(f'before : {points.shape}')
  
		if downsample_resolution != None:
			selected_indices = np.random.choice(len(points), size=(int)(len(points)*downsample_resolution), replace=False)
			points = points[selected_indices]
		
		print(f'after : {points.shape}')

		# initial_T = np.identity(4)

		# align
		if i == 0:
			reg.set_input_target(points)
			delta = np.identity(4)
		else:
			reg.set_input_source(points)
			delta = reg.align()
			reg.set_input_target(points)

		# Save trajectory
		poses.append(poses[-1].dot(delta))

		# fps
		stamps.append(1/(time.time()-start))
		stamps_ = stamps[-9:]
		fps = sum(stamps_) / len(stamps_)
		print('fps:%.3f' % fps)

		# visualize pointcloud
		if visualize and i%5==0:
			points_.transform(poses[-1].dot(delta))
			vis.add_geometry(points_)
			vis.update_geometry(points_)
			vis.poll_events()
			vis.update_renderer()
  
		# plot trajectory
		traj = np.array([x[:3, 3] for x in poses])
		if (i % 30 == 0) or (i == len(filenames)-1):
			pyplot.clf()
			pyplot.title(f'Downsample ratio {downsample_resolution}\nfps : {fps:.2f}')
			pyplot.plot(traj[:, 0], traj[:, 1], label='g-icp trajectory', linewidth=3)
			pyplot.legend()
			pyplot.plot(gt_traj_vis[:, 0], gt_traj_vis[:, 1], label='ground truth trajectory')
			pyplot.legend()
			pyplot.axis('equal')
			pyplot.pause(0.01)
			
	pyplot.show()
	if visualize:
		vis.run()

if __name__ == '__main__':
	print("start")
	main()
