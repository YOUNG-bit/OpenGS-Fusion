#!/bin/bash

# Define global paths
DATA_ROOT=/your/path/to/Replica
OUTPUT_ROOT="./output/Replica"
CONFIG_PATH="./configs/Replica/caminfo.txt"
SCRIPT_PATH="opengs_fusion.py"

# Define scenes array
scenes=("room0" "room1" "room2" "office1" "office2" "office3" "office4")

# Traverse scenes and run commands
for scene_name in "${scenes[@]}"; do
    # Output current scene name
    echo "========================================"
    echo "Processing scene: ${scene_name}"
    echo "========================================"
    
    # Define paths for current scene
    scene_path="${DATA_ROOT}/${scene_name}"
    color_path="${scene_path}/rgb"
    feature_path="${scene_path}/mobile_sam_feature"
    output_path="${OUTPUT_ROOT}/${scene_name}/${scene_name}_default_each_sem"
    
    # Create output directories if they don't exist
    mkdir -p "${feature_path}"
    mkdir -p "${output_path}"
    
    # Step 1: Run MobileSAM feature extraction
    echo "[1/2] Running MobileSAM feature extraction for ${scene_name}..."
    python mobilesamv2_clip.py \
        --image_folder "${color_path}" \
        --output_dir "${feature_path}" \
        --save_results
    
    # Step 2: Run GS-ICP-SLAM with semantic features
    echo "[2/2] Running GS-ICP-SLAM with semantic features for ${scene_name}..."
    python "${SCRIPT_PATH}" \
        --dataset_path "${scene_path}" \
        --config "${CONFIG_PATH}" \
        --output_path "${output_path}" \
        --save_results \
        --with_sem \
        --sam_model mobilesam \
    
    # Output completion message
    echo "Successfully processed ${scene_name}"
    echo ""
done

echo "All scenes processed successfully!"