import copy
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import Constants as const

from dataloader.label_noise.action_genome.ag_dataset import LabelNoiseAG
from dataloader.partial.action_genome.ag_dataset import PartialAG
from dataloader.standard.action_genome.ag_dataset import StandardAG
from dataloader.standard.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from lib.object_detector import Detector
from stsg_base import STSGBase


class TrainSGGBase(STSGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._model = None

        # Load while initializing the object detector
        self._object_detector = None

        # Load while initializing the dataset
        self._train_dataset = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._object_classes = None

        # Load checkpoint name
        self._checkpoint_name = None
        self._checkpoint_save_dir_path = None

    def _init_loss_functions(self):
        self._bce_loss = nn.BCELoss()
        self._ce_loss = nn.CrossEntropyLoss()
        self._mlm_loss = nn.MultiLabelMarginLoss()
        self._bbox_loss = nn.SmoothL1Loss()
        self._abs_loss = nn.L1Loss()
        self._mse_loss = nn.MSELoss()

    def _init_object_detector(self):
        self._object_detector = Detector(
            train=True,
            object_classes=self._object_classes,
            use_SUPPLY=True,
            mode=self._conf.mode
        ).to(device=self._device)
        self._object_detector.eval()

    def _prepare_labels_and_distribution(self, pred, distribution_type, label_type, max_len):
        total_labels = len(pred[label_type])
        pred_distribution = pred[distribution_type]

        # Filter out both the distribution and labels if all the labels are masked
        # Note: Loss should not include items if all the labels are masked
        filtered_labels = []
        filtered_distribution = []
        if not self._conf.bce_loss:
            # For Multi Label Margin Loss (MLM)
            for i in range(total_labels):
                gt = torch.tensor(pred[label_type][i], device=self._device)
                mask = torch.tensor(pred[f'{label_type}_mask'][i], device=self._device)
                gt_masked = gt[mask == 1]
                pred_distribution_i = pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = -torch.ones([max_len], dtype=torch.long, device=self._device)
                    label[:gt_masked.size(0)] = gt_masked
                    filtered_labels.append(label)
                    filtered_distribution.append(pred_distribution_i)
        else:
            # For Binary Cross Entropy Loss (BCE)
            for i in range(total_labels):
                gt = torch.tensor(pred[label_type][i], device=self._device)
                mask = torch.tensor(pred[f'{label_type}_mask'][i], device=self._device)
                gt_masked = gt[mask == 1]
                pred_distribution_i = pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = torch.zeros([max_len], dtype=torch.float32, device=self._device)
                    label[gt_masked] = 1
                    filtered_labels.append(label)
                    filtered_distribution.append(pred_distribution_i)

        if len(filtered_labels) == 0 and len(filtered_distribution) == 0:
            return None, None

        filtered_labels = torch.stack(filtered_labels)
        filtered_distribution = torch.stack(filtered_distribution)

        return filtered_distribution, filtered_labels

    def _calculate_losses_for_partial_annotations(self, pred):
        losses = {}

        # 1. Object Loss
        if self._conf.mode in [const.SGCLS, const.SGDET]:
            losses[const.OBJECT_LOSS] = self._ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

        # 2. Attention Loss
        attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
        attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(device=self._device).squeeze()
        attention_label_mask = torch.tensor(pred[const.ATTENTION_GT_MASK], dtype=torch.float32).to(
            device=self._device).squeeze()
        assert attention_label.shape == attention_label_mask.shape
        # Change to shape [1] if the tensor defaults to a single value
        if len(attention_label.shape) == 0:
            attention_label = attention_label.unsqueeze(0)
            attention_label_mask = attention_label_mask.unsqueeze(0)

        # Filter attention distribution and attention label based on the attention label mask
        filtered_attention_distribution = attention_distribution[attention_label_mask == 1]
        filtered_attention_label = attention_label[attention_label_mask == 1]

        assert filtered_attention_distribution.shape[0] == filtered_attention_label.shape[0]
        losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(filtered_attention_distribution, filtered_attention_label)

        # --------------------------------------------------------------------------------------------
        # For both spatial and contacting relations, if all the annotations are masked then the loss is not calculated

        # 3. Spatial Loss
        filtered_spatial_distribution, filtered_spatial_labels = self._prepare_labels_and_distribution(
            pred=pred,
            distribution_type=const.SPATIAL_DISTRIBUTION,
            label_type=const.SPATIAL_GT,
            max_len=6
        )

        if filtered_spatial_distribution is not None and filtered_spatial_labels is not None:
            if not self._conf.bce_loss:
                losses[const.SPATIAL_RELATION_LOSS] = self._mlm_loss(filtered_spatial_distribution, filtered_spatial_labels)
            else:
                losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(filtered_spatial_distribution, filtered_spatial_labels)

        # 4. Contacting Loss
        filtered_contact_distribution, filtered_contact_labels = self._prepare_labels_and_distribution(
            pred=pred,
            distribution_type=const.CONTACTING_DISTRIBUTION,
            label_type=const.CONTACTING_GT,
            max_len=17
        )

        if filtered_contact_distribution is not None and filtered_contact_labels is not None:
            if not self._conf.bce_loss:
                losses[const.CONTACTING_RELATION_LOSS] = self._mlm_loss(filtered_contact_distribution, filtered_contact_labels)
            else:
                losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(filtered_contact_distribution, filtered_contact_labels)

        return losses

    def _calculate_losses_for_full_annotations(self, pred):
        attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
        spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
        contact_distribution = pred[const.CONTACTING_DISTRIBUTION]

        attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(device=self._device).squeeze()
        if not self._conf.bce_loss:
            # Adjust Labels for MLM Loss
            spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(device=self._device)
            contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(device=self._device)
            for i in range(len(pred[const.SPATIAL_GT])):
                spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
                contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
        else:
            # Adjust Labels for BCE Loss
            spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(device=self._device)
            contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(device=self._device)
            for i in range(len(pred[const.SPATIAL_GT])):
                spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
                contact_label[i, pred[const.CONTACTING_GT][i]] = 1

        losses = {}
        # 1. Object Loss
        if self._conf.mode == const.SGCLS or self._conf.mode == const.SGDET:
            losses[const.OBJECT_LOSS] = self._ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

        # 2. Attention Loss
        losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(attention_distribution, attention_label)

        # 3. Spatial Loss and Contacting Loss
        if not self._conf.bce_loss:
            losses[const.SPATIAL_RELATION_LOSS] = self._mlm_loss(spatial_distribution, spatial_label)
            losses[const.CONTACTING_RELATION_LOSS] = self._mlm_loss(contact_distribution, contact_label)
        else:
            losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(spatial_distribution, spatial_label)
            losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(contact_distribution, contact_label)
        return losses

    def _train_model(self):
        tr = []
        for epoch in range(self._conf.nepoch):
            self._model.train()
            train_iter = iter(self._dataloader_train)
            counter = 0
            start_time = time.time()
            self._object_detector.is_train = True
            for train_idx in tqdm(range(len(self._dataloader_train))):
                data = next(train_iter)
                im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                video_index = data[4]

                gt_annotation = self._train_dataset.gt_annotations[video_index]
                gt_annotation_mask = None
                if self._conf.use_partial_annotations:
                    gt_annotation_mask = self._train_dataset.gt_annotations_mask[video_index]

                if len(gt_annotation) == 0:
                    print(f'No annotations found in the video {video_index}. Skipping...')
                    continue

                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
                with torch.no_grad():
                    entry = self._object_detector(
                        im_data,
                        im_info,
                        gt_boxes,
                        num_boxes,
                        gt_annotation,
                        im_all=None,
                        gt_annotation_mask=gt_annotation_mask
                    )

                # ----------------- Process the video (Method Specific)-----------------
                pred = self.process_train_video(entry, frame_size, gt_annotation)
                # ----------------------------------------------------------------------

                if self._conf.use_partial_annotations:
                    losses = self._calculate_losses_for_partial_annotations(pred)
                else:
                    losses = self._calculate_losses_for_full_annotations(pred)


                self._optimizer.zero_grad()
                loss = sum(losses.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5, norm_type=2)
                self._optimizer.step()

                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

                if counter % 1000 == 0 and counter >= 1000:
                    time_per_batch = (time.time() - start_time) / 1000
                    print(
                        "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter,
                                                                                      len(self._dataloader_train),
                                                                                      time_per_batch,
                                                                                      len(self._dataloader_train) * time_per_batch / 60))

                    mn = pd.concat(tr[-1000:], axis=1).mean(1)
                    print(mn)
                    start_time = time.time()
                counter += 1

            self._save_model(
                model=self._model,
                epoch=epoch,
                checkpoint_save_file_path=self._checkpoint_save_dir_path,
                checkpoint_name=self._checkpoint_name,
                method_name=self._conf.method_name
            )

            test_iter = iter(self._dataloader_test)
            self._model.eval()
            self._object_detector.is_train = False
            with torch.no_grad():
                for b in tqdm(range(len(self._dataloader_test))):
                    data = next(test_iter)
                    im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                    gt_annotation = self._test_dataset.gt_annotations[data[4]]
                    frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

                    # ----------------- Process the video (Method Specific)-----------------
                    pred = self.process_test_video(entry, frame_size, gt_annotation)
                    # ----------------------------------------------------------------------

                    self._evaluator.evaluate_scene_graph(gt_annotation, pred)
                print('-----------------------------------------------------------------------------------', flush=True)
            score = np.mean(self._evaluator.result_dict[self._conf.mode + "_recall"][20])
            self._evaluator.print_stats()
            self._evaluator.reset_result()
            self._scheduler.step(score)

    # ----------------- Load the dataset -------------------------
    # Three main settings:
    # (a) Standard Dataset: Where full annotations are used
    # (b) Partial Annotations: Where partial object and relationship annotations are used
    # (c) Label Noise: Where label noise is added to the dataset
    # -------------------------------------------------------------
    def init_dataset(self):

        if self._conf.use_partial_annotations:
            print("-----------------------------------------------------")
            print("Loading the partial annotations dataset with percentage:", self._conf.partial_percentage)
            print("-----------------------------------------------------")
            self._train_dataset = PartialAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                partial_percentage=self._conf.partial_percentage,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
        elif self._conf.use_label_noise:
            print("-----------------------------------------------------")
            print("Loading the dataset with label noise percentage:", self._conf.label_noise_percentage)
            print("-----------------------------------------------------")
            self._train_dataset = LabelNoiseAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                noise_percentage=self._conf.label_noise_percentage,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
        else:
            print("-----------------------------------------------------")
            print("Loading the standard dataset")
            print("-----------------------------------------------------")
            self._train_dataset = StandardAG(
                phase="train",
                mode=self._conf.mode,
                datasize=self._conf.datasize,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True
            )

        self._test_dataset = StandardAG(
            phase="test",
            mode=self._conf.mode,
            datasize=self._conf.datasize,
            data_path=self._conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self._conf.mode == 'predcls' else True
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=False
        )

        self._object_classes = self._train_dataset.object_classes

    @abstractmethod
    def process_train_video(self, video, frame_size, gt_annotation) -> dict:
        pass

    @abstractmethod
    def process_test_video(self, video, frame_size, gt_annotation) -> dict:
        pass

    def init_method_training(self):
        # 0. Initialize the config
        self._init_config()

        # 1. Initialize the dataset
        self.init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pre-trained models
        self.init_model()
        self._init_loss_functions()
        self._load_checkpoint()
        self._init_object_detector()
        self._init_optimizer()
        self._init_scheduler()

        # 4. Initialize model training
        self._train_model()