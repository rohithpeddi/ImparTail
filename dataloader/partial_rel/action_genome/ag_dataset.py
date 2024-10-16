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
        self._gt_annotations_mask = filtered_gt_annotations
        # self._gt_annotations = self.filter_gt_annotations(partial_percentage)

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

        if self._maintain_distribution:
            target_counts = {obj: int(round(count * partial_annotation_ratio)) for obj, count in rel_counts.items()}
        else:
            target_total_annotations = int(round(total_annotations * partial_annotation_ratio))
            objects = list(rel_counts.keys())
            total_relations = len(objects)

            # Generate random counts such that their sum equals target_total_annotations
            counts = np.random.multinomial(target_total_annotations, np.ones(total_relations) / total_relations)

            # Assign counts to corresponding objects
            target_counts = {obj: count for obj, count in zip(objects, counts)}

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

    @staticmethod
    def get_rel_class_list(gt_annotations, relationship_name):
        data_rel_class_list = []
        for video_id, video_annotation_dict in enumerate(gt_annotations):
            video_rel_list = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                video_frame_gt_rel_id_list = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        # video_frame_gt_obj_id_list.append(0)
                        continue

                    rel_id_list = frame_obj_dict[relationship_name]

                    if isinstance(rel_id_list, torch.Tensor):
                        rel_id_list = rel_id_list.detach().cpu().numpy().tolist()

                    assert isinstance(rel_id_list, list)
                    video_frame_gt_rel_id_list.extend(rel_id_list)
                video_rel_list.append(video_frame_gt_rel_id_list)
            data_rel_class_list.append(video_rel_list)

        return data_rel_class_list

    def filter_gt_annotations(self, partial_percentage):
        # Load from cache if the partial file exists in the cache directory.
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        cache_file = os.path.join(annotations_path, const.PARTIAL_REL,  f'{self._mode}_partial_rel_{partial_percentage}.json')

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
        attention_rel_class_list = self.get_rel_class_list(self._gt_annotations, const.ATTENTION_RELATIONSHIP)
        spatial_rel_class_list = self.get_rel_class_list(self._gt_annotations, const.SPATIAL_RELATIONSHIP)
        contacting_rel_class_list = self.get_rel_class_list(self._gt_annotations, const.CONTACTING_RELATIONSHIP)


        # 2. Construct filter based on the probability distribution of the obj class occurrences.
        filtered_attention_rel_class_list = self.filter_annotations_preserve_distribution(
            data=attention_rel_class_list,
            partial_annotation_ratio=partial_percentage * 0.01
        )

        filtered_spatial_rel_class_list = self.filter_annotations_preserve_distribution(
            data=spatial_rel_class_list,
            partial_annotation_ratio=partial_percentage * 0.01
        )

        filtered_contacting_rel_class_list = self.filter_annotations_preserve_distribution(
            data=contacting_rel_class_list,
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
                    attention_rel = frame_obj_dict[const.ATTENTION_RELATIONSHIP]
                    spatial_rel = frame_obj_dict[const.SPATIAL_RELATIONSHIP]
                    contacting_rel = frame_obj_dict[const.CONTACTING_RELATIONSHIP]

                    if isinstance(attention_rel, torch.Tensor):
                        attention_rel = attention_rel.detach().cpu().numpy().tolist()

                    if isinstance(spatial_rel, torch.Tensor):
                        spatial_rel = spatial_rel.detach().cpu().numpy().tolist()

                    if isinstance(contacting_rel, torch.Tensor):
                        contacting_rel = contacting_rel.detach().cpu().numpy().tolist()

                    filtered_frame_obj_dict = frame_obj_dict.copy()
                    filtered_frame_obj_dict[const.ATTENTION_RELATIONSHIP] = []
                    filtered_frame_obj_dict[const.SPATIAL_RELATIONSHIP] = []
                    filtered_frame_obj_dict[const.CONTACTING_RELATIONSHIP] = []

                    # Check the intersection of the attention_rel list with the filtered_attention_rel_class_list.
                    # Add all the intersection elements to the filtered_frame_obj_dict.
                    filtered_attention_rel = set(attention_rel).intersection(filtered_attention_rel_class_list[video_id][video_frame_id])
                    filtered_attention_rel = list(filtered_attention_rel)
                    if len(filtered_attention_rel) > 0:
                        filtered_frame_obj_dict[const.ATTENTION_RELATIONSHIP] = filtered_attention_rel

                    # Check the intersection of the spatial_rel list with the filtered_spatial_rel_class_list.
                    # Add all the intersection elements to the filtered_frame_obj_dict.
                    filtered_spatial_rel = set(spatial_rel).intersection(filtered_spatial_rel_class_list[video_id][video_frame_id])
                    filtered_spatial_rel = list(filtered_spatial_rel)
                    if len(filtered_spatial_rel) > 0:
                        filtered_frame_obj_dict[const.SPATIAL_RELATIONSHIP] = filtered_spatial_rel

                    # Check the intersection of the contacting_rel list with the filtered_contacting_rel_class_list.
                    # Add all the intersection elements to the filtered_frame_obj_dict.
                    filtered_contacting_rel = set(contacting_rel).intersection(filtered_contacting_rel_class_list[video_id][video_frame_id])
                    filtered_contacting_rel = list(filtered_contacting_rel)
                    if len(filtered_contacting_rel) > 0:
                        filtered_frame_obj_dict[const.CONTACTING_RELATIONSHIP] = filtered_contacting_rel

                    filtered_video_frame_annotation_dict.append(filtered_frame_obj_dict)

                # Add a person object frame only when an object is observed in the frame.
                if len(filtered_video_frame_annotation_dict) > 0:
                    filtered_video_frame_annotation_dict.insert(0, video_frame_annotation_dict[0])

                filtered_video_annotation_dict.append(filtered_video_frame_annotation_dict)

            # Don't change this logic as the ground truth annotations are loaded based on the video index
            # Number of gt annotations should remain the same as the original annotations.
            filtered_gt_annotations.append(filtered_video_annotation_dict)

        # 4. Save the filtered ground truth annotations to the cache directory.
        os.makedirs(os.path.join(annotations_path, const.PARTIAL_REL), exist_ok=True)
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
