// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "VDBVolume.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace {

// Compute the signed distance function (SDF) for a voxel
float ComputeSDF(const Eigen::Vector3d& origin,
                 const Eigen::Vector3d& point,
                 const Eigen::Vector3d& voxel_center) {
    const Eigen::Vector3d v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3d v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);
    const double sign = proj / std::abs(proj);
    return static_cast<float>(sign * dist);
}

// Get the center of a voxel in world coordinates
Eigen::Vector3d GetVoxelCenter(const openvdb::Coord& voxel, const openvdb::math::Transform& xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return Eigen::Vector3d(v_wf.x(), v_wf.y(), v_wf.z());
}

}  // namespace

namespace vdbfusion {

VDBVolume::VDBVolume(float voxel_size, float sdf_trunc, bool space_carving /* = false*/, int max_label /* = 10000*/, int max_points /* = 10000000*/, 
                    pybind11::array_t<int> influence_voxel, 
                    pybind11::array_t<float> influence_voxel_last_weight, 
                    pybind11::array_t<float> influence_voxel_new_weight,
                    pybind11::array_t<int> points_2_voxel_index,
                    pybind11::array_t<float> points_2_voxel_tsdf,
                    pybind11::array_t<float> points_2_voxel_weight)
    : voxel_size_(voxel_size), 
      sdf_trunc_(sdf_trunc), 
      space_carving_(space_carving), 
      max_label(max_label),
      max_points(max_points),
      influence_voxel(influence_voxel),
      influence_voxel_last_weight(influence_voxel_last_weight),
      influence_voxel_new_weight(influence_voxel_new_weight),
      points_2_voxel_index(points_2_voxel_index),
      points_2_voxel_tsdf(points_2_voxel_tsdf),
      points_2_voxel_weight(points_2_voxel_weight),
      ray_label(max_label, 0.0f),
      points_label(max_points, 0),
      global_voxel_index(1),  // 0 is reserved for None
      voxel_coord(max_points, std::vector<int>(3, 0))
{
    // Initialize TSDF grid
    tsdf_ = openvdb::FloatGrid::create(sdf_trunc_);
    tsdf_->setName("D(x): signed distance grid");
    tsdf_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    tsdf_->setGridClass(openvdb::GRID_LEVEL_SET);

    // Initialize weights grid
    weights_ = openvdb::FloatGrid::create(0.0f);
    weights_->setName("W(x): weights grid");
    weights_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    weights_->setGridClass(openvdb::GRID_UNKNOWN);

    // Initialize label grid
    label_ = openvdb::Int32Grid::create(0);
    label_->setName("L(x): label grid");
    label_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    label_->setGridClass(openvdb::GRID_UNKNOWN);

    // Initialize voxel index grid
    voxel_index_ = openvdb::Int32Grid::create(0);
    voxel_index_->setName("F(x): voxel index grid");
    voxel_index_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    voxel_index_->setGridClass(openvdb::GRID_UNKNOWN);

    // Example: Access and modify pybind11 array
    // auto influence_voxel_mutable = influence_voxel.mutable_unchecked<2>();
    // printf("points_mutable.shape(0): %d\n", influence_voxel_mutable.shape(0));
    // printf("points_mutable.shape(1): %d\n", influence_voxel_mutable.shape(1));
    // for (ssize_t i = 0; i < influence_voxel_mutable.shape(0); i++) {
    //     for (ssize_t j = 0; j < influence_voxel_mutable.shape(1); j++) {
    //         influence_voxel_mutable(i, j) += 1000;
    //     }
    // }
}

// Extract points and their corresponding voxel indices with weight filtering
std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>> VDBVolume::ExtractPointsAndIndices(float min_weight) const {
    std::vector<Eigen::Vector3d> points;
    std::vector<int> indices;

    auto tsdf_acc = tsdf_->getAccessor();
    auto voxel_index_acc = voxel_index_->getAccessor();
    auto weights_acc = weights_->getAccessor();
    const openvdb::math::Transform& xform = tsdf_->transform();

    for (auto iter = tsdf_->beginValueOn(); iter; ++iter) {
        const openvdb::Coord& voxel = iter.getCoord();
        float tsdf_value = tsdf_acc.getValue(voxel);
        float weight = weights_acc.getValue(voxel);

        if (weight > min_weight) {
            Eigen::Vector3d point = GetVoxelCenter(voxel, xform);
            points.push_back(point);

            int voxel_idx = voxel_index_acc.getValue(voxel);
            indices.push_back(voxel_idx);
        }
    }

    return std::make_tuple(points, indices);
}

// Update TSDF and weights for a given voxel
void VDBVolume::UpdateTSDF(const float& sdf,
                           const openvdb::Coord& voxel,
                           const std::function<float(float)>& weighting_function) {
    using AccessorRW = openvdb::tree::ValueAccessorRW<openvdb::FloatTree>;
    if (sdf > -sdf_trunc_) {
        AccessorRW tsdf_acc = AccessorRW(tsdf_->tree());
        AccessorRW weights_acc = AccessorRW(weights_->tree());
        const float tsdf = std::min(sdf_trunc_, sdf);
        const float weight = weighting_function(sdf);
        const float last_weight = weights_acc.getValue(voxel);
        const float last_tsdf = tsdf_acc.getValue(voxel);
        const float new_weight = weight + last_weight;
        const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
        tsdf_acc.setValue(voxel, new_tsdf);
        weights_acc.setValue(voxel, new_weight);
    }
}

// Integrate a grid into the TSDF volume
void VDBVolume::Integrate(openvdb::FloatGrid::Ptr grid,
                          const std::function<float(float)>& weighting_function) {
    for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
        const auto& sdf = iter.getValue();
        const auto& voxel = iter.getCoord();
        this->UpdateTSDF(sdf, voxel, weighting_function);
    }
}

// Integrate a point cloud into the TSDF volume
void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    current_points_size = points.size();
    auto influence_voxel_mutable = influence_voxel.mutable_unchecked<2>();
    auto influence_voxel_last_weight_mutable = influence_voxel_last_weight.mutable_unchecked<2>();
    auto influence_voxel_new_weight_mutable = influence_voxel_new_weight.mutable_unchecked<2>();
    auto points_2_voxel_index_mutable = points_2_voxel_index.mutable_unchecked<1>();
    auto points_2_voxel_tsdf_mutable = points_2_voxel_tsdf.mutable_unchecked<1>();
    auto points_2_voxel_weight_mutable = points_2_voxel_weight.mutable_unchecked<1>();

    const openvdb::math::Transform& xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();
    auto voxel_index_acc = voxel_index_->getAccessor();

    std::fill(points_label.begin(), points_label.end(), 0);
    int idx = 0;

    int max_influence_num = 0;
    int average_influence_num = 0;

    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        std::fill(ray_label.begin(), ray_label.end(), 0.0f);
        int label_with_max_weight = 0;
        float max_weight = 0.0f;

        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Traverse voxels along the ray using DDA
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);

        int influence_num = 0;
        float min_sdf = 10000000;

        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                if (weight < 0.0f) {
                    printf("Negative weight: %f\n", weight);
                }
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);
                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);

                // Assign voxel index if not set
                int voxel_idx = voxel_index_acc.getValue(voxel);
                if(voxel_idx == 0) {
                    voxel_index_acc.setValue(voxel, global_voxel_index);
                    voxel_idx = global_voxel_index;
                    global_voxel_index++;
                    if (global_voxel_index > 50000000) {
                        printf("voxel_idx out of range\n");
                        exit(0);
                    }
                    voxel_coord[voxel_idx] = {voxel.x(), voxel.y(), voxel.z()};
                }

                // Update influence_voxel arrays
                if(influence_num < 30){
                    influence_voxel_mutable(idx, influence_num) = voxel_idx;
                    influence_voxel_last_weight_mutable(idx, influence_num) = last_weight;
                    influence_voxel_new_weight_mutable(idx, influence_num) = weight;
                }

                // Update points_2_voxel_index with the voxel having minimum SDF
                if (std::abs(sdf) < min_sdf) {
                    min_sdf = std::abs(sdf);
                    points_2_voxel_index_mutable(idx) = voxel_idx;
                    points_2_voxel_tsdf_mutable(idx) = tsdf;
                    points_2_voxel_weight_mutable(idx) = new_weight;
                }

                // Voting for label (if used)
                // ray_label[current_label] += new_weight;
                // if (ray_label[current_label] > max_weight) {
                //     max_weight = ray_label[current_label];
                //     label_with_max_weight = current_label;
                // }
            
                influence_num++;
            }
        } while (dda.step());

        // Update points_label if label voting is used
        // points_label[idx++] = label_with_max_weight;

        // Track influence statistics if needed
        // if(influence_num > max_influence_num) {
        //     max_influence_num = influence_num;
        // }
        // average_influence_num += influence_num;

        idx++;
    });

    // Example: Print integration statistics
    // printf("Integrating %d points\n", idx);
    // printf("max_influence_num: %d\n", max_influence_num);
    // printf("average_influence_num: %d\n", average_influence_num / points.size());
}

// Update label grid values from old_value to new_value
void VDBVolume::UpdateLabelAttribute(int old_value, int new_value) {
    auto label_acc = label_->getAccessor();
    for (auto iter = label_->beginValueOn(); iter; ++iter) {
        if (iter.getValue() == old_value) {
            iter.setValue(new_value);
        }
    }
}

// Prune TSDF grid based on minimum weight
openvdb::FloatGrid::Ptr VDBVolume::Prune(float min_weight) const {
    const auto weights = weights_->tree();
    const auto tsdf = tsdf_->tree();
    const auto background = sdf_trunc_;
    openvdb::FloatGrid::Ptr clean_tsdf = openvdb::FloatGrid::create(sdf_trunc_);
    clean_tsdf->setName("D(x): Pruned signed distance grid");
    clean_tsdf->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    clean_tsdf->setGridClass(openvdb::GRID_LEVEL_SET);
    clean_tsdf->tree().combine2Extended(tsdf, weights, [=](openvdb::CombineArgs<float>& args) {
        if (args.aIsActive() && args.b() > min_weight) {
            args.setResult(args.a());
            args.setResultIsActive(true);
        } else {
            args.setResult(background);
            args.setResultIsActive(false);
        }
    });
    return clean_tsdf;
}

// Get the label for each point
const std::vector<int>& VDBVolume::GetPointsLabel() const {
    return points_label;
}

// Get the number of active voxels in the TSDF grid
size_t VDBVolume::GetVoxelCount() const {
    size_t voxel_count = 0;
    for (auto iter = tsdf_->beginValueOn(); iter; ++iter) {
        ++voxel_count;
    }
    return voxel_count;
}

// Get the current global voxel index
size_t VDBVolume::GetGlobalFeatureIdx() const {
    return global_voxel_index;
}

// Example: Access and modify pybind11 array
void VDBVolume::test(pybind11::array_t<float> points) const {
    auto points_mutable = points.mutable_unchecked<2>();
    for (ssize_t i = 0; i < points_mutable.shape(0); i++) {
        for (ssize_t j = 0; j < points_mutable.shape(1); j++) {
            points_mutable(i, j) += 9.0;
        }
    }
}

// Example: Feature vector accessors (commented out)
// std::vector<float> VDBVolume::GetFeature(size_t index) const {
//     if (index >= feature_vector_.size()) {
//         throw std::out_of_range("Index out of range");
//     }
//     return feature_vector_[index];
// }

// void VDBVolume::SetFeature(size_t index, const std::vector<float>& feature) {
//     if (feature.size() != 512) {
//         throw std::invalid_argument("Feature vector must be of size 512");
//     }
//     if (index >= feature_vector_.size()) {
//         throw std::out_of_range("Index out of range");
//     }
//     feature_vector_[index] = feature;
// }

}  // namespace vdbfusion