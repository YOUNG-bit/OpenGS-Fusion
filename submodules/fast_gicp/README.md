
* Ref: https://github.com/SMRT-AIST/fast_gicp

- Add some useful functions for both cpp & python
- Modify gicp as it utilizes raw covariance by following normalized_ellipse mode (not plane mode), in order to meet the scales for multiple 3D pointclouds
=> scale = scale / scale[1] .max(1e-3)
  
* note that cov = R*S*(R*S)^T = R*SS*R^T,   S = scale.asDiagonal();
* here, R = quaternion.toRotation();
* q = (q_x, q_y, q_z, q_w)
* R and SS can be obtained by SVD; R=U, scale**2 = singular_values.array()

Install for python
```shell
cd catkin_workspace
catkin_make -DCMAKE_BUILD_TYPE=Release
cd src/fast_gicp
python3 setup.py install --user
```

python usage (see src/fast_gicp/python):

```python
import pygicp

gicp = pygicp.FastGICP()

gicp.set_input_target(target)
gicp.set_input_source(source)

# set covariance from quaternion and scale by following normalized_elipse
nparray_of_quaternions = nparray_of_quaternions_Nx4.flatten()
nparray_of_scales = nparray_of_scales_NX3.flatten()
gicp.set_target_covariance_fromqs(nparray_of_quaternions, nparray_of_scales) => 0.002180 sec
gicp.set_source_covariance_fromqs(nparray_of_quaternions, nparray_of_scales)

# compute covariance by following normalized_elipse
calculate_target_covariance() # compute covariance from given input target pointcloud
calculate_source_covariance() # compute covariance from given input source pointcloud

# after gicp.align()
correspondences, sq_distances = gicp.get_source_correspondence()
covariances = get_target_covariances()
covariances = get_source_covariances()
nparray_of_quaternions = get_target_rotationsq() => 0.00002277 sec
nparray_of_quaternions = get_source_rotationsq() 
nparray_of_scales = get_target_scales()          => 0.00002739 sec
nparray_of_scales = get_source_scales()
nparray_of_quaternions_Nx4 = np.reshape(nparray_of_quaternions, (-1,4))
nparray_of_scales_NX3 = np.reshape(nparray_of_scales, (-1,3))

```
## fast_gicp_tester
- We provide simple test code on Replica/TUM dataset.
```bash
cd python_tester
```

### Download dataset
One need to download datasets for testing. To download Replica and TUM dataset, use bash files.
```bash
# Replica
bash download_replica.sh
# TUM
bash download_tum.sh
```

### Usage

```bash
python gicp_odometry2.py [dataset_path] [tum or replica] [downsample_resolution] [visualize?]
```

### Example
Test fast-gicp on replica room0, with random downsampling ratio 0.05. And visualize registered pointclouds.

```bash
python gicp_odometry2.py ./dataset/Replica/room0 replica 0.05 true
```
<p align="center">
  <img width="35%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/replica_0.05_traj.png"/>
  <img width="40%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/replica_0.05_pointcloud.gif"/>
</p>




Test fast-gicp on TUM_fr3_office, without random downsampling and visualizing pointclouds.
```bash
python gicp_odometry2.py ./dataset/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household tum false false
```

Test fast-gicp on TUM_fr3_office, with random downsampling ratio 0.05, and visualizing pointclouds. 
```bash
python using_previous_30.py dataset/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household tum 0.05 true
```

<p align="center">
  <img width="40%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/tum_30_elipse.png"/>
  <img width="40%" src="https://github.com/Lab-of-AI-and-Robotics/fast_gicp/blob/main/data/tum_30_elipse.gif"/>
</p>





