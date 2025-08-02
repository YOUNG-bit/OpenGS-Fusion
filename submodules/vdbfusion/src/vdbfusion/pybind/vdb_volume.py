# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
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
from typing import Any, Optional, Tuple, Callable, overload

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import open3d as o3d
from . import vdbfusion_pybind


class VDBVolume:
    def __init__(
        self,
        voxel_size: float,
        sdf_trunc: float,
        space_carving: bool = False,
    ):
        # Maximum number of voxels and points
        self.max_points = 3000000
        self.per_max_points = 400000
        self.max_label = 10000

        # Internal buffers for voxel influence and weights
        self.current_points_size = 0
        self.influence_voxel = np.zeros((self.per_max_points, 30), dtype=np.int32)
        self.influence_voxel_last_weight = np.zeros((self.per_max_points, 30), dtype=np.float32)
        self.influence_voxel_new_weight = np.zeros((self.per_max_points, 30), dtype=np.float32)
        self.points_2_voxel_index = np.zeros((self.per_max_points), dtype=np.int32)

        # Global feature matrix for all points
        self.global_clip_feature = torch.zeros((self.per_max_points, 512), dtype=torch.float32, device='cuda').contiguous()
        self.current_points_label = None
        self.current_label_feature = None

        # Initialize C++ backend
        self._volume = vdbfusion_pybind._VDBVolume(
            voxel_size=np.float32(voxel_size),
            sdf_trunc=np.float32(sdf_trunc),
            space_carving=space_carving,
            max_label=self.max_label,
            max_points=self.max_points,
            influence_voxel=self.influence_voxel,
            influence_voxel_last_weight=self.influence_voxel_last_weight,
            influence_voxel_new_weight=self.influence_voxel_new_weight,
            points_2_voxel_index=self.points_2_voxel_index,
        )

        # Passthrough all data members from the C++ API
        self.voxel_size = self._volume._voxel_size
        self.sdf_trunc = self._volume._sdf_trunc
        self.space_carving = self._volume._space_carving
        self.pyopenvdb_support_enabled = self._volume.PYOPENVDB_SUPPORT_ENABLED
        if self.pyopenvdb_support_enabled:
            self.tsdf = self._volume._tsdf
            self.weights = self._volume._weights

        print("VDBVolume initialized")

    ########## Custom Functions ##########

    def update_global_feature(self) -> None:
        """
        Update the global feature matrix using the influence voxel and weights.
        This step distills the influence_voxel array and updates the global_clip_feature tensor.
        """
        mask1 = self.influence_voxel[:self.current_points_size, :] != 0
        _cu_incfluence_voxel = torch.from_numpy(self.influence_voxel[:self.current_points_size, :][mask1]).to(torch.int32).reshape(-1).cuda().contiguous()
        _cu_incfluence_voxel_last_weight = torch.from_numpy(self.influence_voxel_last_weight[:self.current_points_size, :][mask1]).to(torch.float32).reshape(-1).cuda().contiguous()
        _cu_incfluence_voxel_new_weight = torch.from_numpy(self.influence_voxel_new_weight[:self.current_points_size, :][mask1]).to(torch.float32).reshape(-1).cuda().contiguous()
        _cu_current_points_label = self.current_points_label.unsqueeze(1).repeat(1, 30).to(torch.int32)[mask1].reshape(-1).cuda().contiguous()

        # Expand global_clip_feature if needed
        max_voxel_idx = _cu_incfluence_voxel.max()
        if max_voxel_idx >= self.global_clip_feature.size(0):
            self.global_clip_feature = torch.cat(
                (
                    self.global_clip_feature,
                    torch.zeros((max_voxel_idx - self.global_clip_feature.size(0) + 10, 512), dtype=torch.float32, device='cuda').contiguous()
                ),
                dim=0
            )

        # Weighted update of global_clip_feature
        self.global_clip_feature[_cu_incfluence_voxel, :] = (
            self.global_clip_feature[_cu_incfluence_voxel, :] * _cu_incfluence_voxel_last_weight.unsqueeze(1)
            + self.current_label_feature[_cu_current_points_label, :] * _cu_incfluence_voxel_new_weight.unsqueeze(1)
        ) / (
            _cu_incfluence_voxel_last_weight.unsqueeze(1) + _cu_incfluence_voxel_new_weight.unsqueeze(1)
        )

    def set_current_frame_feature(self, points_label, label_feature) -> None:
        """
        Set the current frame's point labels and label features.
        points_label: N x 1 tensor, each point's label
        label_feature: M x 512 tensor, each label's feature
        """
        self.current_points_label = points_label
        self.current_label_feature = label_feature

    ########## Predefined Functions ##########

    def __repr__(self) -> str:
        return (
            f"VDBVolume with:\n"
            f"voxel_size    = {self.voxel_size}\n"
            f"sdf_trunc     = {self.sdf_trunc}\n"
            f"space_carving = {self.space_carving}\n"
        )

    @overload
    def integrate(
        self,
        points: np.ndarray,
        extrinsic: np.ndarray,
        weighting_function: Callable[[float], float],
    ) -> None:
        ...

    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray, weight: float) -> None:
        ...

    @overload
    def integrate(self, points: np.ndarray, extrinsic: np.ndarray) -> None:
        ...

    @overload
    def integrate(self, grid, weighting_function: Callable[[float], float]) -> None:
        ...

    @overload
    def integrate(self, grid, weight: float) -> None:
        ...

    @overload
    def integrate(self, grid) -> None:
        ...

    def integrate(
        self,
        points: Optional[np.ndarray] = None,
        extrinsic: Optional[np.ndarray] = None,
        grid: Optional[Any] = None,
        weight: Optional[float] = None,
        weighting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        """
        Integrate new points or grid into the TSDF volume.
        This function supports multiple overloads for different input types.
        """
        self.current_points_size = points.shape[0]
        if self.current_points_size > self.per_max_points or self.current_points_size == 0:
            print(f"Current points size is {self.current_points_size}")
            raise ValueError(f"points size must be in [1, {self.per_max_points}]")

        # Reset influence voxel buffers for current points
        self.influence_voxel[:points.shape[0], :] = 0
        self.influence_voxel_last_weight[:points.shape[0], :] = 0
        self.influence_voxel_new_weight[:points.shape[0], :] = 0

        if grid is not None:
            if not self.pyopenvdb_support_enabled:
                raise NotImplementedError("Please compile with PYOPENVDB_SUPPORT_ENABLED")
            if weighting_function is not None:
                return self._volume._integrate(grid, weighting_function)
            if weight is not None:
                return self._volume._integrate(grid, weight)
            return self._volume._integrate(grid)
        else:
            assert isinstance(points, np.ndarray), "points must be np.ndarray(n, 3)"
            assert points.dtype == np.float64, "points dtype must be np.float64"
            assert isinstance(extrinsic, np.ndarray), "origin/extrinsic must be np.ndarray"
            assert extrinsic.dtype == np.float64, "origin/extrinsic dtype must be np.float64"
            assert extrinsic.shape in [
                (3,),
                (3, 1),
                (4, 4),
            ], "origin/extrinsic must be a (3,) array or a (4,4) matrix"

            _points = vdbfusion_pybind._VectorEigen3d(points)
            if weighting_function is not None:
                return self._volume._integrate(_points, extrinsic, weighting_function)
            if weight is not None:
                return self._volume._integrate(_points, extrinsic, weight)
            self._volume._integrate(_points, extrinsic)

    @overload
    def update_tsdf(
        self, sdf: float, ijk: np.ndarray, weighting_function: Optional[Callable[[float], float]]
    ) -> None:
        ...

    @overload
    def update_tsdf(self, sdf: float, ijk: np.ndarray) -> None:
        ...

    def update_tsdf(
        self,
        sdf: float,
        ijk: np.ndarray,
        weighting_function: Optional[Callable[[float], float]] = None,
    ) -> None:
        """
        Update the TSDF value for a given voxel index.
        """
        if weighting_function is not None:
            return self._volume._update_tsdf(sdf, ijk, weighting_function)
        return self._volume._update_tsdf(sdf, ijk)

    def extract_triangle_mesh(self, fill_holes: bool = True, min_weight: float = 0.0) -> Tuple:
        """
        Extract the triangle mesh from the TSDF volume.
        Returns vertices and triangles arrays.
        Example usage:
            vertices, triangles = integrator.extract_triangle_mesh()
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(triangles),
            )
        """
        vertices, triangles = self._volume._extract_triangle_mesh(fill_holes, min_weight)
        return np.asarray(vertices), np.asarray(triangles)

    def extract_points_and_indices(self, min_weight: float = 0.0) -> Tuple:
        """
        Extract points and their voxel indices from the TSDF volume.
        Returns points and indices arrays.
        """
        points, index = self._volume._extract_points_and_indices(min_weight)
        return np.asarray(points), np.asarray(index)

    def extract_vdb_grids(self, out_file: str) -> None:
        """
        Write the internal map representation to a file.
        Contains both D(x) and W(x) grids.
        """
        self._volume._extract_vdb_grids(out_file)

    def prune(self, min_weight: float):
        """
        Cleanup the TSDF grid using the W(x) weights grid according to a minimum weight threshold.
        This function is ideal to cleanup the TSDF grid before exporting it.
        """
        return self._volume._prune(min_weight)

    def update_label_attribute(self, old_value: int, new_value: int) -> None:
        """
        Update label attributes in the backend.
        """
        self._volume._update_label_attribute(old_value, new_value)

    def get_points_label(self):
        """
        Get the labels of all points in the volume.
        """
        return np.array(self._volume.get_points_label())

    def get_voxel_count(self):
        """
        Get the total number of voxels in the volume.
        """
        return self._volume.get_voxel_count()

    def get_global_feature_idx(self):
        """
        Get the indices of global features.
        """
        return self._volume.get_global_feature_idx()

    def test(self, points: np.ndarray) -> None:
        """
        Test function for backend integration.
        """
        assert points.ndim == 2, "points must be 2D array"
        self._volume._test(points)

    def get_influence_voxel(self) -> torch.Tensor:
        """
        Returns the influence voxel array as a torch tensor.
        """
        return self._volume.get_influence_voxel()

    ########## Save Functions ##########

    def save_points_and_feature(self, out_folder: str) -> None:
        """
        Save the points and feature to files.
        Saves points as .npz and .ply, and features as .npz.
        """
        points, indices = self.extract_points_and_indices(2.0)
        np.savez(os.path.join(out_folder, "points.npz"), points=points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(os.path.join(out_folder, "points.ply"), pcd)

        feature = self.global_clip_feature[indices.astype(np.int32)].cpu().numpy()
        np.savez(os.path.join(out_folder, "feature.npz"), feature=feature)