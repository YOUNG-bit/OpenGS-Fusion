/*
 * MIT License
 *
 * Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

 #pragma once

 #include <openvdb/openvdb.h>
 #include <Eigen/Core>
 #include <functional>
 #include <tuple>
 #include <vector>
 #include <array>
 #include "Vec512.h"
 #include <pybind11/eigen.h>
 #include <pybind11/functional.h>
 #include <pybind11/numpy.h>
 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
 #include <pybind11/stl_bind.h>
 
 namespace vdbfusion {
 
 /**
  * @brief VDBVolume class for TSDF fusion and related operations.
  */
 class VDBVolume {
 public:
     /**
      * @brief Constructor for VDBVolume.
      * @param voxel_size Size of each voxel.
      * @param sdf_trunc Truncation value for SDF.
      * @param space_carving Enable space carving.
      * @param max_label Maximum label count.
      * @param max_points Maximum points count.
      * @param influence_voxel Input influence voxel array.
      * @param influence_voxel_last_weight Last weight for influence voxel.
      * @param influence_voxel_new_weight New weight for influence voxel.
      * @param points_2_voxel_index Mapping from points to voxel index.
      * @param points_2_voxel_tsdf Mapping from points to voxel TSDF.
      * @param points_2_voxel_weight Mapping from points to voxel weight.
      */
     VDBVolume(float voxel_size, float sdf_trunc, bool space_carving = false, int max_label = 10000, int max_points = 5000000,
         pybind11::array_t<int> influence_voxel = pybind11::array_t<int>(), 
         pybind11::array_t<float> influence_voxel_last_weight = pybind11::array_t<float>(), 
         pybind11::array_t<float> influence_voxel_new_weight = pybind11::array_t<float>(),
         pybind11::array_t<int> points_2_voxel_index = pybind11::array_t<int>(),
         pybind11::array_t<float> points_2_voxel_tsdf = pybind11::array_t<float>(),
         pybind11::array_t<float> points_2_voxel_weight = pybind11::array_t<float>()
     );
     ~VDBVolume() = default;
 
 public:
     /**
      * @brief Integrate a new (globally aligned) PointCloud into the current TSDF volume.
      * @param points Input point cloud.
      * @param origin Sensor origin.
      * @param weighting_function Weighting function for integration.
      */
     void Integrate(const std::vector<Eigen::Vector3d>& points,
                    const Eigen::Vector3d& origin,
                    const std::function<float(float)>& weighting_function);
 
     /**
      * @brief Integrate a new (globally aligned) PointCloud using extrinsics.
      * @param points Input point cloud.
      * @param extrinsics Extrinsic matrix.
      * @param weighting_function Weighting function for integration.
      */
     inline void Integrate(const std::vector<Eigen::Vector3d>& points,
                           const Eigen::Matrix4d& extrinsics,
                           const std::function<float(float)>& weighting_function) {
         // Extract origin from extrinsics and call main Integrate
         const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
         Integrate(points, origin, weighting_function);
     }
 
     /**
      * @brief Integrate incoming TSDF grid inside the current volume using TSDF equations.
      * @param grid Input TSDF grid.
      * @param weighting_function Weighting function for integration.
      */
     void Integrate(openvdb::FloatGrid::Ptr grid,
                    const std::function<float(float)>& weighting_function);
 
     /**
      * @brief Fuse a new given SDF value at the given voxel location, thread-safe.
      * @param sdf Signed distance value.
      * @param voxel Voxel coordinate.
      * @param weighting_function Weighting function for update.
      */
     void UpdateTSDF(const float& sdf,
                     const openvdb::Coord& voxel,
                     const std::function<float(float)>& weighting_function);
 
     /**
      * @brief Prune TSDF grids, useful for cleaning up before exporting.
      * @param min_weight Minimum weight threshold.
      * @return Pruned TSDF grid.
      */
     openvdb::FloatGrid::Ptr Prune(float min_weight) const;
 
     /**
      * @brief Extract a TriangleMesh as the iso-surface in the actual volume.
      * @param fill_holes Fill holes in mesh.
      * @param min_weight Minimum weight threshold.
      * @return Tuple of vertices and triangle indices.
      */
     [[nodiscard]] std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>>
     ExtractTriangleMesh(bool fill_holes = true, float min_weight = 0.5) const;
 
     /**
      * @brief Extract points and their indices from the volume.
      * @param min_weight Minimum weight threshold.
      * @return Tuple of points and indices.
      */
     std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>> ExtractPointsAndIndices(float min_weight = 0.0) const;
 
     /**
      * @brief Update label attribute for voxels.
      * @param old_value Old label value.
      * @param new_value New label value.
      */
     void UpdateLabelAttribute(int old_value, int new_value);
 
     /**
      * @brief Get the points_label array.
      * @return Reference to points_label vector.
      */
     const std::vector<int>& GetPointsLabel() const;
 
     /**
      * @brief Get the number of voxels.
      * @return Voxel count.
      */
     size_t GetVoxelCount() const;
 
     /**
      * @brief Get the global feature index.
      * @return Global feature index.
      */
     size_t GetGlobalFeatureIdx() const;
 
     /**
      * @brief Test function for input points.
      * @param points Input points array.
      */
     void test(pybind11::array_t<float> points) const;
 
 public:
     // OpenVDB grids modeling the signed distance field and the weight grid
     openvdb::FloatGrid::Ptr tsdf_;
     openvdb::FloatGrid::Ptr weights_;
     openvdb::Int32Grid::Ptr label_;         // Label grid
     openvdb::Int32Grid::Ptr voxel_index_;   // Feature index for each voxel
 
     // Public properties
     float voxel_size_;
     float sdf_trunc_;
     bool space_carving_;
 
     int max_label;      // Maximum label count
     int max_points;     // Maximum points count
 
     // Input arrays
     pybind11::array_t<int> influence_voxel;
     pybind11::array_t<float> influence_voxel_last_weight;
     pybind11::array_t<float> influence_voxel_new_weight;
     pybind11::array_t<int> points_2_voxel_index;
     pybind11::array_t<float> points_2_voxel_tsdf;
     pybind11::array_t<float> points_2_voxel_weight;
 
     // Internal arrays
     std::vector<float> ray_label;           // Stores max label index for each ray
     std::vector<int> points_label;          // Stores final label for each ray
     std::vector<std::vector<int>> voxel_coord; // Stores voxel coordinates (x, y, z)
     int global_voxel_index;                 // Global feature index, starts from 1
     int current_points_size = 0;            // Current size of points
 };
 
 }  // namespace vdbfusion
 
 // Example: Extract origin from extrinsics matrix
 // const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
 
 // Example: Integrate a point cloud
 // volume.Integrate(points, origin, weighting_function);
 
 // Example: Update TSDF value at a voxel
 // volume.UpdateTSDF(sdf, voxel, weighting_function);