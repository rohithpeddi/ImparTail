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
from dataloader.partial_obj.action_genome.ag_dataset import PartialObjAG
from dataloader.partial_rel.action_genome.ag_dataset import PartialRelAG
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

    @staticmethod
    def filter_predictions_for_missing_labels(pred):
        # Attention, Spatial, Contacting distributions are of the format [bbox_num, 3], [bbox_num, 6], [bbox_num, 17] respectively
        # Attention, Spatial, Contacting labels are lists each containing [bbox_num] items
        # We need to filter out the predictions for which the labels are missing i,e labels have empty lists []
        attention_dist = pred[const.ATTENTION_DISTRIBUTION]
        spatial_dist = pred[const.SPATIAL_DISTRIBUTION]
        contact_dist = pred[const.CONTACTING_DISTRIBUTION]

        attention_labels = pred[const.ATTENTION_GT]
        spatial_labels = pred[const.SPATIAL_GT]
        contact_labels = pred[const.CONTACTING_GT]

        total_bbox_num = attention_dist.shape[0]

        # Filtering non-empty label distributions using list comprehensions
        filtered_attention = [(dist, lbl) for dist, lbl in zip(attention_dist, attention_labels) if lbl]
        filtered_spatial = [(dist, lbl) for dist, lbl in zip(spatial_dist, spatial_labels) if lbl]
        filtered_contact = [(dist, lbl) for dist, lbl in zip(contact_dist, contact_labels) if lbl]

        # Updating predictions dictionary with filtered results
        if filtered_attention:
            pred[const.ATTENTION_DISTRIBUTION], pred[const.ATTENTION_GT] = zip(*filtered_attention)
            pred[const.ATTENTION_DISTRIBUTION] = torch.stack(pred[const.ATTENTION_DISTRIBUTION])
        else:
            pred[const.ATTENTION_DISTRIBUTION], pred[const.ATTENTION_GT] = torch.tensor([]), []

        if filtered_spatial:
            pred[const.SPATIAL_DISTRIBUTION], pred[const.SPATIAL_GT] = zip(*filtered_spatial)
            pred[const.SPATIAL_DISTRIBUTION] = torch.stack(pred[const.SPATIAL_DISTRIBUTION])
        else:
            pred[const.SPATIAL_DISTRIBUTION], pred[const.SPATIAL_GT] = torch.tensor([]), []

        if filtered_contact:
            pred[const.CONTACTING_DISTRIBUTION], pred[const.CONTACTING_GT] = zip(*filtered_contact)
            pred[const.CONTACTING_DISTRIBUTION] = torch.stack(pred[const.CONTACTING_DISTRIBUTION])
        else:
            pred[const.CONTACTING_DISTRIBUTION], pred[const.CONTACTING_GT] = torch.tensor([]), []

        # print(f"Filtered dataset: "
        #       f"Attention [{pred[const.ATTENTION_DISTRIBUTION].shape[0]}/{total_bbox_num}], "
        #       f"Spatial [{pred[const.SPATIAL_DISTRIBUTION].shape[0]}/{total_bbox_num}], "
        #       f"Contacting [{pred[const.CONTACTING_DISTRIBUTION].shape[0]}/{total_bbox_num}]")

        return pred

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
                        im_all=None
                    )

                # ----------------- Process the video (Method Specific)-----------------
                pred = self.process_train_video(entry, frame_size, gt_annotation)
                # ----------------------------------------------------------------------

                if self._conf.use_partial_rel_annotations:
                    # Filter out distributions for which labels are missing
                    pred = self.filter_predictions_for_missing_labels(pred)

                losses = {}

                # 1. Object Loss
                if self._conf.mode == const.SGCLS or self._conf.mode == const.SGDET:
                    losses[const.OBJECT_LOSS] = self._ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

                # 2. Attention Loss
                attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
                if attention_distribution.shape[0] > 0:
                    attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(
                        device=self._device).squeeze()
                    # Change to shape [1] if the tensor defaults to a single value
                    if len(attention_label.shape) == 0:
                        attention_label = attention_label.unsqueeze(0)

                    assert attention_distribution.shape[0] == attention_label.shape[0]
                    losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(attention_distribution, attention_label)

                # 2. Spatial Loss
                spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
                if spatial_distribution.shape[0] > 0:
                    tot_obj_spatial_labels = len(pred[const.SPATIAL_GT])
                    if not self._conf.bce_loss:
                        # Adjust Labels for MLM Loss
                        spatial_label = -torch.ones([tot_obj_spatial_labels, 6], dtype=torch.long).to(
                            device=self._device)
                        for i in range(tot_obj_spatial_labels):
                            spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
                    else:
                        # Adjust Labels for BCE Loss
                        spatial_label = torch.zeros([tot_obj_spatial_labels, 6], dtype=torch.float32).to(
                            device=self._device)
                        for i in range(tot_obj_spatial_labels):
                            spatial_label[i, pred[const.SPATIAL_GT][i]] = 1

                    assert spatial_distribution.shape == spatial_label.shape
                    if not self._conf.bce_loss:
                        losses[const.SPATIAL_RELATION_LOSS] = self._mlm_loss(spatial_distribution, spatial_label)
                    else:
                        losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(spatial_distribution, spatial_label)

                # 3. Contacting Loss
                contact_distribution = pred[const.CONTACTING_DISTRIBUTION]
                if contact_distribution.shape[0] > 0:
                    tot_obj_contact_labels = len(pred[const.CONTACTING_GT])
                    if not self._conf.bce_loss:
                        # Adjust Labels for MLM Loss
                        contact_label = -torch.ones([tot_obj_contact_labels, 17], dtype=torch.long).to(
                            device=self._device)
                        for i in range(tot_obj_contact_labels):
                            contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(
                                pred[const.CONTACTING_GT][i])
                    else:
                        # Adjust Labels for BCE Loss
                        contact_label = torch.zeros([tot_obj_contact_labels, 17], dtype=torch.float32).to(
                            device=self._device)
                        for i in range(tot_obj_contact_labels):
                            contact_label[i, pred[const.CONTACTING_GT][i]] = 1

                    assert contact_distribution.shape == contact_label.shape

                    if not self._conf.bce_loss:
                        losses[const.CONTACTING_RELATION_LOSS] = self._mlm_loss(contact_distribution, contact_label)
                    else:
                        losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(contact_distribution, contact_label)

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

    def init_dataset(self):

        if self._conf.use_partial_obj_annotations:
            print("-----------------------------------------------------")
            print("Loading the partial object dataset")
            print("-----------------------------------------------------")
            self._train_dataset = PartialObjAG(
                phase="train",
                mode=self._conf.mode,
                datasize=self._conf.datasize,
                partial_percentage=self._conf.partial_percentage,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
        elif self._conf.use_partial_rel_annotations:
            print("-----------------------------------------------------")
            print("Loading the partial relation dataset")
            print("-----------------------------------------------------")
            self._train_dataset = PartialRelAG(
                phase="train",
                mode=self._conf.mode,
                datasize=self._conf.datasize,
                partial_percentage=self._conf.partial_percentage,
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