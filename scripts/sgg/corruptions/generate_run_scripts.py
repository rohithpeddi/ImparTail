import os

from constants import CorruptionConstants as const


class GenerateRunScripts:

    def __init__(self):
        self.ckpt_directory_path = "/data/rohith/ag/checkpoints"
        self.data_path = "/data/rohith/ag"
        self.task = const.SGG
        self.methods = ["sttran", "dsgdetr"]
        self.modes = [const.SGCLS, const.PREDCLS]
        self.corruption_types = [
            const.NO_CORRUPTION, const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
            const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.MOTION_BLUR, const.ZOOM_BLUR, const.FOG, const.FROST,
            const.SNOW, const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.ELASTIC_TRANSFORM, const.PIXELATE,
            const.JPEG_COMPRESSION, const.SUN_GLARE, const.RAIN, const.DUST, const.WILDFIRE_SMOKE, const.SATURATE
        ]
        self.dataset_corruption_modes = [const.FIXED]
        self.video_corruption_modes = [const.FIXED, const.MIXED]
        self.severity_levels = [3]
        self.generate_scripts()

    def generate_scripts(self):
        # 1. Method parent directory
        # 2. mode file
        # 3. run_script for all the methods with parameters
        current_dir = os.path.dirname(os.path.realpath(__file__))
        for i, method_name in enumerate(self.methods):
            method_dir = os.path.join(current_dir, method_name)
            os.makedirs(method_dir, exist_ok=True)
            for mode in self.modes:
                mode_method_file_path = os.path.join(method_dir, f"{method_name}_{mode}.txt")
                with open(mode_method_file_path, 'w') as f:
                    for dataset_corruption_mode in self.dataset_corruption_modes:
                        if dataset_corruption_mode == const.FIXED:
                            for severity_level in self.severity_levels:
                                for corruption_type in self.corruption_types:
                                    f.write(
                                        f"python test_sgg_methods.py --task_name {self.task} --method_name {method_name} "
                                        f"--ckpt {self.ckpt_directory_path}/{self.task}/{method_name}/{method_name}_{mode}_epoch_3.tar "
                                        f"--use_input_corruptions --dataset_corruption_mode {dataset_corruption_mode} "
                                        f"--dataset_corruption_type {corruption_type} "
                                        f"--corruption_severity_level {severity_level}\n")
                                    f.write(
                                        "-----------------------------------------------------------------------------\n")
                        elif dataset_corruption_mode == const.MIXED:
                            for severity_level in self.severity_levels:
                                for video_corruption_mode in self.video_corruption_modes:
                                    f.write(
                                        f"python test_sgg_methods.py --task_name {self.task} --method_name {method_name} "
                                        f"--ckpt {self.ckpt_directory_path}/{self.task}/{method_name}/{method_name}_{mode}_epoch_3.tar "
                                        f"--use_input_corruptions --dataset_corruption_mode {dataset_corruption_mode} "
                                        f"--video_corruption_mode {video_corruption_mode} --corruption_severity_level {severity_level}\n")
                                    f.write(
                                        "-----------------------------------------------------------------------------\n")


def main():
    generate_run_scripts = GenerateRunScripts()
    generate_run_scripts.generate_scripts()


if __name__ == '__main__':
    main()
