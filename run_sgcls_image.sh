#!/bin/bash

# Constants
CKPT_DIRECTORY_PATH="/data/rohith/ag/checkpoints"
METHODS=("dsgdetr")
TASKS=("sgg")
MODES=("sgcls")
CORRUPTION_TYPES=("gaussian_noise" "shot_noise" "impulse_noise" "speckle_noise" "gaussian_blur" "defocus_blur" "fog" "frost" "spatter" "contrast" "brightness" "pixelate" "jpeg_compression" "sun_glare" "dust" "saturate")
DATASET_CORRUPTION_MODES=("fixed" "mixed")
VIDEO_CORRUPTION_MODES=("fixed" "mixed")
SEVERITY_LEVELS=(3)

# Main function to execute commands iteratively
generate_run_scripts() {
    local current_dir=$(pwd)
    for task in "${TASKS[@]}"; do
        for method_name in "${METHODS[@]}"; do
            for mode in "${MODES[@]}"; do
                for dataset_corruption_mode in "${DATASET_CORRUPTION_MODES[@]}"; do
                    if [[ "$dataset_corruption_mode" == "fixed" ]]; then
                        for severity_level in "${SEVERITY_LEVELS[@]}"; do
                            for corruption_type in "${CORRUPTION_TYPES[@]}"; do
                                # Execute Python script directly for each combination
                                python test_sgg_methods.py --task_name $task --method_name $method_name --ckpt $CKPT_DIRECTORY_PATH/${task}/${method_name}/${method_name}_${mode}_epoch_3.tar --use_input_corruptions --dataset_corruption_mode $dataset_corruption_mode --dataset_corruption_type $corruption_type --corruption_severity_level $severity_level
                                echo "-----------------------------------------------------------------------------"
                            done
                        done
                    elif [[ "$dataset_corruption_mode" == "mixed" ]]; then
                        for severity_level in "${SEVERITY_LEVELS[@]}"; do
                            for video_corruption_mode in "${VIDEO_CORRUPTION_MODES[@]}"; do
                                # Execute Python script directly for each combination
                                python test_sgg_methods.py --task_name $task --method_name $method_name --ckpt $CKPT_DIRECTORY_PATH/${task}/${method_name}/${method_name}_${mode}_epoch_3.tar --use_input_corruptions --dataset_corruption_mode $dataset_corruption_mode --video_corruption_mode $video_corruption_mode --dataset_corruption_type $video_corruption_mode --corruption_severity_level $severity_level
                                echo "-----------------------------------------------------------------------------"
                            done
                        done
                    fi
                done
            done
        done
    done
}

# Run the main function
generate_run_scripts