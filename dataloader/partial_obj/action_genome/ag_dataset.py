import collections
import os
import random

import cv2
import numpy as np
import torch
import json

from dataloader.base_ag_dataset import BaseAG
from constants import Constants as const
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from utils import NumpyEncoder


class PartialObjAG(BaseAG):

    def __init__(
            self,
            phase,
            mode,
            maintain_distribution,
            datasize,
            partial_percentage=10,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):
        super().__init__(phase, mode, datasize, data_path, filter_nonperson_box_frame, filter_small_box)

        self._maintain_distribution = maintain_distribution
        # Filter out objects in the ground truth based on object observation ratio.
        filtered_gt_annotations = self.filter_gt_annotations(partial_percentage)
        # self._gt_annotations = filtered_gt_annotations
        self._gt_annotations_mask = filtered_gt_annotations


    @staticmethod
    def estimate_distribution(data):
        object_counts = collections.Counter()
        total_annotations = 0

        for video in data:
            for frame in video:
                for obj in frame:
                    object_counts[obj] += 1
                    total_annotations += 1

        distribution = {obj: count / total_annotations for obj, count in object_counts.items()}
        return distribution, total_annotations, object_counts

    def filter_annotations_preserve_distribution(self, data, partial_annotation_ratio):
        # First, estimate the distribution
        distribution, total_annotations, object_counts = self.estimate_distribution(data)

        if self._maintain_distribution:
            target_counts = {obj: int(round(count * partial_annotation_ratio)) for obj, count in object_counts.items()}
        else:
            target_total_annotations = int(round(total_annotations * partial_annotation_ratio))
            objects = list(object_counts.keys())
            total_relations = len(objects)

            # Generate random counts such that their sum equals target_total_annotations
            counts = np.random.multinomial(target_total_annotations, np.ones(total_relations) / total_relations)

            # Assign counts to corresponding objects
            target_counts = {obj: count for obj, count in zip(objects, counts)}

        # Collect positions of each object
        obj_positions = collections.defaultdict(list)
        for v_idx, video in enumerate(data):
            for f_idx, frame in enumerate(video):
                for o_idx, obj in enumerate(frame):
                    obj_positions[obj].append((v_idx, f_idx, o_idx))

        # For each object, randomly select target_counts[obj] positions to keep
        positions_to_keep = set()
        for obj, positions in obj_positions.items():
            k = target_counts[obj]
            if k > len(positions):
                k = len(positions)
            selected_positions = random.sample(positions, k)
            positions_to_keep.update(selected_positions)

        # Reconstruct the data
        filtered_data = []
        for v_idx, video in enumerate(data):
            filtered_video = []
            for f_idx, frame in enumerate(video):
                filtered_frame = []
                for o_idx, obj in enumerate(frame):
                    if (v_idx, f_idx, o_idx) in positions_to_keep:
                        filtered_frame.append(obj)
                filtered_video.append(filtered_frame)
            filtered_data.append(filtered_video)

        return filtered_data

    def filter_gt_annotations(self, partial_percentage):
        # Load from cache if the partial file exists in the cache directory.
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        cache_file = os.path.join(annotations_path, const.PARTIAL_OBJ,  f'{self._mode}_partial_obj_{partial_percentage}.json')

        if os.path.exists(cache_file):
            print(f"Loading filtered ground truth annotations from {cache_file}")
            with open(cache_file, 'r') as file:
                filtered_gt_annotations = json.loads(file.read())

            # def get_video_id_to_annotation(annotations):
            #     video_id_to_annotation = {}
            #     for video_num, video_annotation in enumerate(annotations):
            #         if video_annotation:
            #             video_id = video_annotation[0][0]["frame"].split("/")[0]
            #             video_id_to_annotation[video_id] = video_annotation
            #     return video_id_to_annotation
            #
            # def get_video_id_to_index(annotations):
            #     video_id_to_index = {}
            #     for video_num, video_annotation in enumerate(annotations):
            #         if video_annotation:
            #             video_id = video_annotation[0][0]["frame"].split("/")[0]
            #             video_id_to_index[video_id] = video_num
            #     return video_id_to_index
            #
            # # Inverse map from video_id to annotation
            # self.filtered_video_id_to_annotation = get_video_id_to_annotation(filtered_gt_annotations)
            # self.gt_video_id_to_annotation = get_video_id_to_annotation(self._gt_annotations)
            #
            # self.gt_video_id_to_index = get_video_id_to_index(self._gt_annotations)
            # self.filtered_video_id_to_index = get_video_id_to_index(filtered_gt_annotations)

            return filtered_gt_annotations

        #--------------------------------------------------------------------------------------------
        # Filter out objects in the ground truth based on partial observation ratio.
        #--------------------------------------------------------------------------------------------
        print("--------------------------------------------------------------------------------")
        print("No file found in the cache directory.")
        print(f"Filtering ground truth annotations based on partial observation ratio: {partial_percentage}%")
        print("--------------------------------------------------------------------------------")

        # 1. Estimate statistics of object class occurrences in the ground truth annotations.
        data_obj_class_list = []
        for video_id, video_annotation_dict in enumerate(self._gt_annotations):
            video_obj_list = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                video_frame_gt_obj_id_list = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        # video_frame_gt_obj_id_list.append(0)
                        continue
                    obj_class_id = frame_obj_dict[const.CLASS]
                    video_frame_gt_obj_id_list.append(obj_class_id)
                video_obj_list.append(video_frame_gt_obj_id_list)
            data_obj_class_list.append(video_obj_list)

        # 2. Construct filter based on the probability distribution of the obj class occurrences.
        filtered_data_obj_class_list = self.filter_annotations_preserve_distribution(
            data=data_obj_class_list,
            partial_annotation_ratio=partial_percentage * 0.01
        )

        # 3. Construct filtered ground truth annotations based on filtered obj class occurrences.
        filtered_gt_annotations = []
        for video_id, video_annotation_dict in enumerate(self._gt_annotations):
            filtered_video_annotation_dict = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                filtered_video_frame_annotation_dict = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        continue
                    obj_class_id = frame_obj_dict[const.CLASS]
                    if obj_class_id in filtered_data_obj_class_list[video_id][video_frame_id]:
                        filtered_video_frame_annotation_dict.append(frame_obj_dict)
                    else:
                        filtered_video_frame_annotation_dict.append([])
                filtered_video_frame_annotation_dict.insert(0, video_frame_annotation_dict[0])
                filtered_video_annotation_dict.append(filtered_video_frame_annotation_dict)
            # Don't change this logic as the ground truth annotations are loaded based on the video index
            # Number of gt annotations should remain the same as the original annotations.
            filtered_gt_annotations.append(filtered_video_annotation_dict)

        # 4. Save the filtered ground truth annotations to the cache directory.
        os.makedirs(os.path.join(annotations_path, const.PARTIAL_OBJ), exist_ok=True)
        filtered_gt_annotations_json = json.dumps(filtered_gt_annotations, cls=NumpyEncoder)
        with open(cache_file, 'w') as f:
            f.write(filtered_gt_annotations_json)

        return filtered_gt_annotations


    def __getitem__(self, index):
        frame_names = self._video_list[index]
        processed_ims = []
        im_scales = []
        for idx, name in enumerate(frame_names):
            im = cv2.imread(os.path.join(self._frames_path, name))  # channel h,w,3
            # im = im[:, :, ::-1]  # rgb -> bgr
            # cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
            im_scales.append(im_scale)
            processed_ims.append(im)
        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
