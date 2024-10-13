import collections
import os
import random

import cv2
import numpy as np
import torch

from dataloader.base_ag_dataset import BaseAG
from constants import Constants as const
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


class PartialAG(BaseAG):

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
        self._gt_annotations = self.filter_gt_annotations(partial_percentage * 0.01)

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

        # Compute target counts for each object
        target_total_annotations = int(round(total_annotations * partial_annotation_ratio))
        target_counts = {obj: int(round(count * partial_annotation_ratio)) for obj, count in object_counts.items()}

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

    def filter_gt_annotations(self, partial_ratio):
        # Load from cache if the partial file exists in the cache directory.
        annotations_path = os.path.join(self._data_path, const.ANNOTATIONS)
        cache_file = os.path.join(annotations_path, "partial",  f'partial_{partial_ratio}.npy')

        if os.path.exists(cache_file):
            np_filtered_gt_annotations = np.load(cache_file, allow_pickle=True)
            # convert numpy array to list of lists
            return np_filtered_gt_annotations.tolist()

        #--------------------------------------------------------------------------------------------
        # Filter out objects in the ground truth based on partial observation ratio.
        #--------------------------------------------------------------------------------------------

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
        filtered_data_obj_class_list = self.filter_annotations_preserve_distribution(data_obj_class_list, partial_ratio)

        # 3. Construct filtered ground truth annotations based on filtered obj class occurrences.
        filtered_gt_annotations = []
        for video_id, video_annotation_dict in enumerate(self._gt_annotations):
            filtered_video_annotation_dict = []
            for video_frame_id, video_frame_annotation_dict in enumerate(video_annotation_dict):
                filtered_video_frame_annotation_dict = []
                for frame_obj_id, frame_obj_dict in enumerate(video_frame_annotation_dict):
                    if frame_obj_id == 0:
                        filtered_video_frame_annotation_dict.append(frame_obj_dict)
                        continue
                    obj_class_id = frame_obj_dict[const.CLASS]
                    if obj_class_id in filtered_data_obj_class_list[video_id][video_frame_id]:
                        filtered_video_frame_annotation_dict.append(frame_obj_dict)
                filtered_video_annotation_dict.append(filtered_video_frame_annotation_dict)
            filtered_gt_annotations.append(filtered_video_annotation_dict)

        # 4. Save the filtered ground truth annotations to the cache directory.
        os.makedirs(os.path.join(annotations_path, "partial"), exist_ok=True)
        np_filtered_gt_annotations = np.array(filtered_gt_annotations, dtype=object)
        np.save(cache_file, np_filtered_gt_annotations)

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
