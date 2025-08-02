#!/bin/bash

# Define global data root path
DATA_ROOT="your_path/Scannet/data"

# Define scenes array
scenes=("scene0062_00" "scene0070_00" "scene0140_00" "scene0200_00" "scene0347_00" "scene0400_00")
scenes=("scene0011_00")

# Traverse scenes and run commands
for scene_name in "${scenes[@]}"; do
    # Output current scene name
    echo "Processing scene: ${scene_name}"
    
    # Define paths using the global data root
    scene_path="${DATA_ROOT}/${scene_name}"
    color_path="${scene_path}/rgb"
    feature_path="${scene_path}/mobile_sam_feature"
    
    # First command: run mobilesamv2_clip.py
    echo "Running mobilesamv2_clip.py for ${scene_name}..."
    python mobilesamv2_clip.py \
        --image_folder "${color_path}" \
        --output_dir "${feature_path}" \
        --save_results
    
    # Second command: run opengs_fusion.py with proxychains
    echo "Running opengs_fusion.py for ${scene_name}..."
    python opengs_fusion.py \
        --dataset_path "${scene_path}" \
        --config ./configs/Scannet/"${scene_name}".txt \
        --output_path ./output/Scannet_sem/"${scene_name}"/default_with_sem \
        --save_results --overlapped_th 1e-3 --max_correspondence_distance 0.03 --trackable_opacity_th 0.09 --overlapped_th2 1e-3 --downsample_rate 8 --keyframe_th 0.81 --knn_maxd 99999.0 --voxel_size 0.05 --with_sem \
        --sam_model mobilesam
    
    # Output completion message
    echo "Finished processing ${scene_name}."
    echo ""
done