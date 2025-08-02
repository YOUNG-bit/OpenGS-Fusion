# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
from typing import Any, Optional, Tuple, Callable, overload

import numpy as np
import torch
import torch.multiprocessing as mp
import open3d as o3d
import sys

# 获取current_dir，到submodules/vdbfusion/src/vdbfusion/pybind
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_dir,"submodules","vdbfusion","src","vdbfusion","pybind"))
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
import vdbfusion_pybind

class VdbFusion(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.max_points = slam.max_points
        self.per_max_points = slam.per_max_points
        self.max_label = slam.max_label
        self.voxel_size = slam.voxel_size
        self.sdf_trunc = slam.sdf_trunc
        self.space_carving = slam.space_carving
        self.fill_holes = slam.fill_holes
        self.min_weight = slam.min_weight
        self.with_sem = slam.with_sem
        self.debug_sem = slam.debug_sem
        self.weight_cal_way = slam.weight_cal_way
        self.online_sam = slam.online_sam

        self.current_points_size = 0
        self.influence_voxel = np.zeros((self.per_max_points + 10, 30), dtype=np.int32)
        self.influence_voxel_last_weight = np.zeros((self.per_max_points + 10, 30), dtype=np.float32)
        self.influence_voxel_new_weight = np.zeros((self.per_max_points + 10, 30), dtype=np.float32)
        self.points_2_voxel_index = np.zeros((self.per_max_points + 10), dtype=np.int32)
        self.points_2_voxel_tsdf = np.zeros((self.per_max_points + 10), dtype=np.float32)
        self.points_2_voxel_weight = np.zeros((self.per_max_points + 10), dtype=np.float32)

        self.global_voxel_label = np.zeros((self.max_points), dtype=np.int32)
        self.fake_global_voxel_label = np.zeros((self.max_points), dtype=np.int32)
        self.global_voxel_feature_fusion_count = np.zeros((self.max_points), dtype=np.int32)

        self.global_clip_feature = torch.zeros((self.per_max_points, 512), dtype=torch.float32, device='cuda').contiguous()
        self.global_feature_fusion_count1 = torch.zeros((self.per_max_points), dtype=torch.int32, device='cuda').contiguous()
        self.current_points_label = None
        self.current_label_feature = None

        self.end_of_dataset = slam.end_of_dataset
        self.output_path = slam.output_path
        self._volume = None

        self.new_points_ready_for_vdb = slam.new_points_ready_for_vdb
        self.shared_new_points_for_vdb = slam.shared_new_points_for_vdb

        if self.online_sam:
            self.shared_online_sam = slam.shared_online_sam
            self.new_image_ready = slam.new_image_ready
            self.sam_result_ready = slam.sam_result_ready

        print("VDBVolume initialized")

    def run(self):
        self.vdbfusion()

    def vdbfusion(self):
        times = []
        # Create VDB volume instance (cannot be shared between processes)
        self._volume = vdbfusion_pybind._VDBVolume(
            voxel_size=np.float32(self.voxel_size),
            sdf_trunc=np.float32(self.sdf_trunc),
            space_carving=self.space_carving,
            max_label=self.max_label,
            max_points=self.max_points,
            influence_voxel=self.influence_voxel,
            influence_voxel_last_weight=self.influence_voxel_last_weight,
            influence_voxel_new_weight=self.influence_voxel_new_weight,
            points_2_voxel_index=self.points_2_voxel_index,
            points_2_voxel_tsdf=self.points_2_voxel_tsdf,
            points_2_voxel_weight=self.points_2_voxel_weight,
        )
        # Sync parameters from C++ API
        self.voxel_size = self._volume._voxel_size
        self.sdf_trunc = self._volume._sdf_trunc
        self.space_carving = self._volume._space_carving
        self.pyopenvdb_support_enabled = self._volume.PYOPENVDB_SUPPORT_ENABLED
        if self.pyopenvdb_support_enabled:
            self.tsdf = self._volume._tsdf
            self.weights = self._volume._weights

        while True:
            if self.end_of_dataset[0]:
                print(f"Final Voxel Count: {self.get_voxel_count()}")
                break

            if self.new_points_ready_for_vdb[0]:
                scan, pose, scan_label, label_feature = self.shared_new_points_for_vdb.get_values_from_tracker()
                scan = scan.astype(np.float64)
                pose = pose.astype(np.float64)
                self.integrate(scan, pose)
                self.shared_new_points_for_vdb.input_value_from_vdb(
                    torch.tensor(self.points_2_voxel_index[:scan.shape[0]]),
                    torch.tensor(self.points_2_voxel_tsdf[:scan.shape[0]]),
                    torch.tensor(self.points_2_voxel_weight[:scan.shape[0]])
                )
                self.new_points_ready_for_vdb[0] = 0

                if scan_label is not None:
                    scan_label = torch.tensor(scan_label, dtype=torch.int32).cuda().contiguous()
                    label_feature = torch.tensor(label_feature, dtype=torch.float32).cuda().contiguous()
                    self.set_current_frame_feature(scan_label, label_feature)
                elif self.online_sam:
                    while not self.sam_result_ready[0]:
                        time.sleep(1e-15)
                    scan_label, label_feature = self.shared_online_sam.get_value_from_sam_model()
                    self.sam_result_ready[0] = 0

                if scan_label is not None:
                    self.set_current_frame_feature(scan_label, label_feature)
                    self.update_global_feature()

        # Save mesh and features after processing
        self._res = {"mesh": self._get_o3d_mesh(), "times": times}
        self._write_ply()
        if self.with_sem:
            self.save_points_and_feature(self.output_path)

    def _get_o3d_mesh(self):
        vertices, triangles = self.extract_triangle_mesh(self.fill_holes, self.min_weight)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        return mesh

    def _write_ply(self):
        os.makedirs(self.output_path, exist_ok=True)
        filename = os.path.join(self.output_path, "vdb_mesh") + ".ply"
        o3d.io.write_triangle_mesh(filename, self._res["mesh"])

    # Update global feature matrix for all voxels
    def update_global_feature(self) -> None:
        mask0 = self.current_points_label != 0
        mask0 = mask0.cpu().numpy()
        mask1 = self.influence_voxel[:self.current_points_size, :][mask0] != 0
        _cu_incfluence_voxel = torch.from_numpy(self.influence_voxel[:self.current_points_size, :][mask0][mask1]).to(torch.int32).reshape(-1).cuda().contiguous()

        if _cu_incfluence_voxel.numel() > 0:
            max_voxel_idx = _cu_incfluence_voxel.max()
        else:
            max_voxel_idx = 0
        if max_voxel_idx >= self.global_clip_feature.size(0):
            self.global_clip_feature = torch.cat(
                (self.global_clip_feature,
                 torch.zeros((max_voxel_idx - self.global_clip_feature.size(0) + 10, 512), dtype=torch.float32, device='cuda').contiguous()), dim=0)
            self.global_feature_fusion_count1 = torch.cat(
                (self.global_feature_fusion_count1,
                 torch.zeros((max_voxel_idx - self.global_feature_fusion_count1.size(0) + 10), dtype=torch.int32, device='cuda').contiguous()), dim=0)

        if self.weight_cal_way == 1:
            _cu_incfluence_voxel_last_weight = torch.from_numpy(self.influence_voxel_last_weight[:self.current_points_size, :][mask0][mask1]).to(torch.float32).reshape(-1).cuda().contiguous()
            _cu_incfluence_voxel_new_weight = torch.from_numpy(self.influence_voxel_new_weight[:self.current_points_size, :][mask0][mask1]).to(torch.float32).reshape(-1).cuda().contiguous()
        elif self.weight_cal_way == 2:
            _cu_incfluence_voxel_last_weight = self.global_feature_fusion_count1[_cu_incfluence_voxel].cuda().contiguous()
            _cu_incfluence_voxel_new_weight = torch.ones_like(_cu_incfluence_voxel_last_weight).cuda().contiguous()

        _cu_current_points_label = self.current_points_label.unsqueeze(1).repeat(1, 30).to(torch.int32)[mask0][mask1].reshape(-1).cuda().contiguous()

        self.global_clip_feature[_cu_incfluence_voxel, :] = (
            self.global_clip_feature[_cu_incfluence_voxel, :] * _cu_incfluence_voxel_last_weight.unsqueeze(1) +
            self.current_label_feature[_cu_current_points_label, :] * _cu_incfluence_voxel_new_weight.unsqueeze(1)
        ) / (_cu_incfluence_voxel_last_weight.unsqueeze(1) + _cu_incfluence_voxel_new_weight.unsqueeze(1))
        self.global_feature_fusion_count1[_cu_incfluence_voxel] += 1

    def torch_unique(self, data):
        tensor_data = torch.from_numpy(data).cuda()
        unique_data, counts = torch.unique(tensor_data, return_counts=True, dim=0)
        return unique_data.cpu().numpy(), counts.cpu().numpy()

    def set_current_frame_feature(self, points_label, label_feature) -> None:
        self.current_points_label = points_label
        self.current_label_feature = label_feature
        assert self.current_points_label.shape[0] == self.current_points_size, "current_points_label size must be equal to current_points_size"

    def __repr__(self) -> str:
        return (
            f"VDBVolume with:\n"
            f"voxel_size    = {self.voxel_size}\n"
            f"sdf_trunc     = {self.sdf_trunc}\n"
            f"space_carving = {self.space_carving}\n"
        )

    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray, weighting_function: Callable[[float], float]) -> None: ...
    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray, weight: float) -> None: ...
    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray) -> None: ...
    @overload
    def integrate(self, grid, weighting_function: Callable[[float], float]) -> None: ...
    @overload
    def integrate(self, grid, weight: float) -> None: ...
    @overload
    def integrate(self, grid) -> None: ...

    def integrate(
        self,
        points: Optional[np.ndarray] = None,
        extrinsic: Optional[np.ndarray] = None,
        grid: Optional[Any] = None,
        weight: Optional[float] = None,
        weighting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.current_points_size = points.shape[0]
        if self.current_points_size > self.per_max_points or self.current_points_size == 0:
            print(f"Current points size is {self.current_points_size}")
            raise ValueError(f"points size must be in [1, {self.per_max_points}]")

        self.influence_voxel[:points.shape[0], :] = 0
        self.influence_voxel_last_weight[:points.shape[0], :] = 0
        self.influence_voxel_new_weight[:points.shape[0], :] = 0
        self.points_2_voxel_index[:points.shape[0]] = -1
        self.points_2_voxel_tsdf[:points.shape[0]] = 0
        self.points_2_voxel_weight[:points.shape[0]] = 0

        if grid is not None:
            if not self.pyopenvdb_support_enabled:
                raise NotImplementedError("Please compile with PYOPENVDB_SUPPORT_ENABLED")
            if weighting_function is not None:
                return self._volume._integrate(grid, weighting_function)
            if weight is not None:
                return self._volume._integrate(grid, weight)
            return self._volume._integrate(grid)
        else:
            assert isinstance(points, np.ndarray), "points must by np.ndarray(n, 3)"
            assert points.dtype == np.float64, "points dtype must be np.float64"
            assert isinstance(extrinsic, np.ndarray), "origin/extrinsic must by np.ndarray"
            assert extrinsic.dtype == np.float64, "origin/extrinsic dtype must be np.float64"
            assert extrinsic.shape in [(3,), (3, 1), (4, 4)], "origin/extrinsic must be a (3,) array or a (4,4) matrix"

            _points = vdbfusion_pybind._VectorEigen3d(points)
            if weighting_function is not None:
                return self._volume._integrate(_points, extrinsic, weighting_function)
            if weight is not None:
                return self._volume._integrate(_points, extrinsic, weight)
            self._volume._integrate(_points, extrinsic)

    @overload
    def update_tsdf(self, sdf: float, ijk: np.ndarray, weighting_function: Optional[Callable[[float], float]]) -> None: ...
    @overload
    def update_tsdf(self, sdf: float, ijk: np.ndarray) -> None: ...

    def update_tsdf(
        self,
        sdf: float,
        ijk: np.ndarray,
        weighting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        if weighting_function is not None:
            return self._volume._update_tsdf(sdf, ijk, weighting_function)
        return self._volume._update_tsdf(sdf, ijk)

    def extract_triangle_mesh(self, fill_holes: bool = True, min_weight: float = 0.0) -> Tuple:
        vertices, triangles = self._volume._extract_triangle_mesh(fill_holes, min_weight)
        return np.asarray(vertices), np.asarray(triangles)

    def extract_points_and_indices(self, min_weight: float = 0.0) -> Tuple:
        points, index = self._volume._extract_points_and_indices(min_weight)
        return np.asarray(points), np.asarray(index)

    def extract_vdb_grids(self, out_file: str) -> None:
        self._volume._extract_vdb_grids(out_file)

    def prune(self, min_weight: float):
        return self._volume._prune(min_weight)

    def update_label_attribute(self, old_value: int, new_value: int) -> None:
        self._volume._update_label_attribute(old_value, new_value)

    def get_points_label(self):
        return np.array(self._volume.get_points_label())

    def get_voxel_count(self):
        return self._volume.get_voxel_count()

    def get_global_feature_idx(self):
        return self._volume.get_global_feature_idx()

    def test(self, points: np.ndarray) -> None:
        assert points.ndim == 2, "points must be 2D array"
        self._volume._test(points)

    def get_influence_voxel(self) -> torch.Tensor:
        return self._volume.get_influence_voxel()

    # Save points and features to disk
    def save_points_and_feature(self, out_folder: str) -> None:
        points, indices = self.extract_points_and_indices(self.min_weight)
        # np.savez(os.path.join(out_folder, "points.npz"), points=points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.io.write_point_cloud(os.path.join(out_folder, "points.ply"), pcd)

        assert indices.max() < self.global_clip_feature.size(0), "Indices out of bounds in global_clip_feature"
        feature = self.global_clip_feature[indices.astype(np.int32)].cpu().numpy()
        # np.savez(os.path.join(out_folder, "points_feature.npz"), feature=feature)
        # np.save(os.path.join(out_folder, "points_indices.npy"), indices)

        voxel_num = self.get_voxel_count()
        feature = self.global_clip_feature[:voxel_num, :].cpu().numpy()
        np.savez(os.path.join(out_folder, "global_feature.npz"), feature=feature)

        label = self.global_voxel_label[:voxel_num]
        np.save(os.path.join(out_folder, "global_label.npy"), label)

        points_label = self.global_voxel_label[indices.astype(np.int32)]
        colors = np.random.rand(self.max_label + 1, 3)
        colors[0] = np.array([1, 1, 1])
        points_color = colors[points_label]
        pcd.colors = o3d.utility.Vector3dVector(points_color)
        # o3d.io.write_point_cloud(os.path.join(out_folder, "points_with_label.ply"), pcd)

        print(f"!!!!!!!!!Successfully saved points and feature to {out_folder}!!!!!!!!!")