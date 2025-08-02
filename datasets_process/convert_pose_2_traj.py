import os
import numpy as np
import sys

def convert_scannet_pose(pose_folder):
    """
    Convert all pose files in a folder to a single trajectory file (traj.txt).
    Each pose file is expected to contain a pose matrix.
    """
    # List all pose files and sort by frame index
    pose_files = os.listdir(pose_folder)
    pose_files.sort(key=lambda x: int(x.split('.')[0]))
    pose_files = [os.path.join(pose_folder, file) for file in pose_files]

    # Read each pose file and collect pose matrices
    poses = []
    for pose_file in pose_files:
        pose = np.loadtxt(pose_file)
        poses.append(pose)

    # Save all poses to traj.txt (one line per pose, flattened)
    traj_file = os.path.join(os.path.dirname(pose_folder), 'traj.txt')
    with open(traj_file, 'w') as f:
        for pose in poses:
            pose = pose.flatten()
            pose_str = ' '.join([str(p) for p in pose])
            f.write(pose_str + '\n')

# Example usage: convert poses for multiple scenes
scene_list = [
    "scene0030_00", "scene0046_00", "scene0086_00",
    "scene0222_00", "scene0378_00", "scene0389_00", "scene0435_00"
]
for scene in scene_list:
    # Compose the pose folder path for the current scene
    pose_folder = f'your_path/Scannet/data/{scene}/pose'
    # Convert pose files to trajectory
    convert_scannet_pose(pose_folder)
    print(f"{scene} done")