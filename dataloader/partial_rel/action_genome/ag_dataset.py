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


class PartialRelAG(BaseAG):

    def __init__(
            self,
            phase,
            datasize,
            partial_percentage=10,
            data_path=None,
            filter_nonperson_box_frame=True,
            filter_small_box=False
    ):
        super().__init__(phase, datasize, data_path, filter_nonperson_box_frame, filter_small_box)
        # Filter out objects in the ground truth based on object observation ratio.
        filtered_gt_annotations = self.filter_gt_annotations(partial_percentage)
        self._gt_annotations_mask = filtered_gt_annotations

    @staticmethod
    def estimate_rel_distribution(data):
        rel_counts = collections.Counter()
        total_annotations = 0

        for video in data:
            for frame in video:
                for rel in frame:
                    rel_counts[rel] += 1
                    total_annotations += 1

        distribution = {obj: count / total_annotations for obj, count in rel_counts.items()}
        return distribution, total_annotations, rel_counts

    def filter_annotations_preserve_distribution(self, data, partial_annotation_ratio):
        # First, estimate the distribution
        distribution, total_annotations, rel_counts = self.estimate_rel_distribution(data)

        # Compute target counts for each object
        target_total_annotations = int(round(total_annotations * partial_annotation_ratio))
        target_counts = {obj: int(round(count * partial_annotation_ratio)) for obj, count in rel_counts.items()}

        # Collect positions of each object
        rel_positions = collections.defaultdict(list)
        for v_idx, video in enumerate(data):
            for f_idx, frame in enumerate(video):
                for rel_idx, rel in enumerate(frame):
                    rel_positions[rel].append((v_idx, f_idx, rel_idx))

        # For each relation, randomly select target_counts[obj] positions to keep
        positions_to_keep = set()
        for rel, positions in rel_positions.items():
            k = target_counts[rel]
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
                for rel_idx, rel in enumerate(frame):
                    if (v_idx, f_idx, rel_idx) in positions_to_keep:
                        filtered_frame.append(rel)
                filtered_video.append(filtered_frame)
            filtered_data.append(filtered_video)

        return filtered_data

    def filter_gt_annotations(self, partial_percentage):
        # Load from cache if the partial file exists in the cache directory.
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        cache_file = os.path.join(annotations_path, const.PARTIAL_REL,  f'partial_{partial_percentage}.json')

        if os.path.exists(cache_file):
            print(f"Loading filtered ground truth annotations from {cache_file}")
            with open(cache_file, 'r') as file:
                filtered_gt_annotations = json.loads(file.read())
            return filtered_gt_annotations

        #--------------------------------------------------------------------------------------------
        # Filter out objects in the ground truth based on partial observation ratio.
        #--------------------------------------------------------------------------------------------
        print("--------------------------------------------------------------------------------")
        print("No file found in the cache directory.")
        print(f"Filtering ground truth annotations based on partial observation ratio: {partial_percentage}%")
        print("--------------------------------------------------------------------------------")

        # 1. Estimate statistics of object class occurrences in the ground truth annotations.
        data_rel_class_list = []
        for video_id, video_annotation_dict in enumerate(self._gt_annotations):
            video_rel_list = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                video_frame_gt_rel_id_list = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        # video_frame_gt_obj_id_list.append(0)
                        continue
                    attention_rel = frame_obj_dict[const.ATTENTION_REL]
                    spatial_rel = frame_obj_dict[const.SPATIAL_REL]
                    contacting_rel = frame_obj_dict[const.CONTACTING_REL]

                    video_frame_gt_rel_id_list.append(attention_rel)
                    video_frame_gt_rel_id_list.append(spatial_rel)
                    video_frame_gt_rel_id_list.append(contacting_rel)

                video_rel_list.append(video_frame_gt_rel_id_list)
            data_rel_class_list.append(video_rel_list)

        # 2. Construct filter based on the probability distribution of the obj class occurrences.
        filtered_data_rel_class_list = self.filter_annotations_preserve_distribution(
            data=data_rel_class_list,
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
                    attention_rel = frame_obj_dict[const.ATTENTION_REL]
                    spatial_rel = frame_obj_dict[const.SPATIAL_REL]
                    contacting_rel = frame_obj_dict[const.CONTACTING_REL]

                    filtered_frame_obj_dict = frame_obj_dict.copy()
                    filtered_frame_obj_dict[const.ATTENTION_REL] = []
                    filtered_frame_obj_dict[const.SPATIAL_REL] = []
                    filtered_frame_obj_dict[const.CONTACTING_REL] = []

                    rel_enabled = False
                    if attention_rel in filtered_data_rel_class_list[video_id][video_frame_id]:
                        filtered_frame_obj_dict[const.ATTENTION_REL] = attention_rel
                        rel_enabled = True
                    if spatial_rel in filtered_data_rel_class_list[video_id][video_frame_id]:
                        filtered_frame_obj_dict[const.SPATIAL_REL] = spatial_rel
                        rel_enabled = True
                    if contacting_rel in filtered_data_rel_class_list[video_id][video_frame_id]:
                        filtered_frame_obj_dict[const.CONTACTING_REL] = contacting_rel
                        rel_enabled = True

                    if rel_enabled:
                        filtered_video_frame_annotation_dict.append(filtered_frame_obj_dict)
                # Add a person object frame only when an object is observed in the frame.
                if len(filtered_video_frame_annotation_dict) > 0:
                    filtered_video_frame_annotation_dict.insert(0, video_frame_annotation_dict[0])
                    filtered_video_annotation_dict.append(filtered_video_frame_annotation_dict)
            filtered_gt_annotations.append(filtered_video_annotation_dict)

        # 4. Save the filtered ground truth annotations to the cache directory.
        os.makedirs(os.path.join(annotations_path, "partial"), exist_ok=True)
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
