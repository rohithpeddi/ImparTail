import os
# import sys
# sys.path.insert(1, '/home/maths/btech/mt1200841/scratch/NeSyRobSGG')

from constants import CorruptionConstants as const

def generate_run_scripts():
    # 1. Method parent directory
    # 2. mode file
    # 3. run_script for all the methods with parameters
    current_dir = os.path.dirname(os.path.realpath(__file__))
    for i, method_name in enumerate(methods):
        method_dir = os.path.join(current_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        for mode in modes:
            mode_method_file_path = os.path.join(method_dir, f"{method_name}_{mode}.txt")
            with open(mode_method_file_path, 'w') as f:
                for dataset_corruption_mode in dataset_corruption_modes:
                    if dataset_corruption_mode == const.FIXED:
                        for severity_level in severity_levels:
                            for corruption_type in corruption_types:
                                f.write(
                                    f"python test_sgg_methods.py --task_name {task} --method_name {method_name} --mode {mode} "
                                    f"--ckpt {ckpt_directory_path}/{mode}/{method_name}_{mode}.tar "
                                    f"--dataset_corruption_mode {dataset_corruption_mode} --dataset_corruption_type {corruption_type} "
                                    f"--corruption_severity_level {severity_level}\n")
                                f.write(
                                    "-----------------------------------------------------------------------------\n")
                    elif dataset_corruption_mode == const.MIXED:
                        for severity_level in severity_levels:
                            for video_corruption_mode in video_corruption_modes:
                                f.write(
                                    f"python test_sgg_methods.py --method_name {method_name} --mode {mode} --ckpt {ckpt_directory_path}/{mode}/{method_name}_{mode}.tar "
                                    f"--dataset_corruption_mode {dataset_corruption_mode} "
                                    f"--video_corruption_mode {video_corruption_mode} --corruption_severity_level {severity_level}\n")
                                f.write(
                                    "-----------------------------------------------------------------------------\n")

def main():
    generate_run_scripts()


if __name__ == '__main__':
    ckpt_directory_path = "/data/rohith/ag/checkpoints"
    data_path = "/data/rohith/ag"
    task = const.SGG
    methods = ["sttran", "dsgdetr", "tempura"]
    modes = [const.SGDET, const.SGCLS, const.PREDCLS]
    partial_percentages = [10, 40, 70]

    corruption_types = [
        const.NO_CORRUPTION, const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE, const.GAUSSIAN_BLUR,
        const.DEFOCUS_BLUR, const.MOTION_BLUR, const.ZOOM_BLUR, const.FOG, const.FROST,
        const.SNOW, const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.ELASTIC_TRANSFORM, const.PIXELATE,
        const.JPEG_COMPRESSION, const.SUN_GLARE, const.RAIN, const.DUST, const.WILDFIRE_SMOKE,  const.SATURATE
    ]

    dataset_corruption_modes = [const.FIXED]
    video_corruption_modes = [const.FIXED, const.MIXED]

    severity_levels = [3]
    main()