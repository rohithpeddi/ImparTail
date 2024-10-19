import os
from abc import abstractmethod

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import Constants as const
from lib.AdamW import AdamW
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher


class STSGBase:

    def __init__(self, conf):
        self._train_dataset = None
        self._evaluator = None
        self._model = None
        self._conf = None
        self._device = None

        self._conf = conf

        # Load checkpoint name
        self._checkpoint_name = None
        self._checkpoint_save_dir_path = None

        # Init Wandb
        self._enable_wandb = self._conf.use_wandb

    def _init_config(self, is_train=True):
        print('The CKPT saved here:', self._conf.save_path)
        os.makedirs(self._conf.save_path, exist_ok=True)

        # Set the preferred device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._conf.ckpt is not None:
            if self._conf.task_name == const.SGG:
                # Checkpoint name is of the format for model trained with full annotations: sttran_sgdet_epoch_1.tar, dsgdetr_sgdet_epoch_1.tar
                # Checkpoint name is of the format for model trained with partial annotations: sttran_partial_10_sgdet_epoch_1.tar
                # Checkpoint name is of the format for model trained with label noise: sttran_label_noise_10_sgdet_epoch_1.tar
                self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
                self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])
                self._conf.mode = self._checkpoint_name.split('_')[-1]
                print("--------------------------------------------------------")
                print(f"Loading checkpoint with name: {self._checkpoint_name}")
                print(f"Mode: {self._conf.mode}")
                print("--------------------------------------------------------")
            elif self._conf.task_name == const.SGA:
                # Checkpoint name format for full annotations: sttran_ant_sgdet_future_3_epoch_1.tar
                # Checkpoint name format for partial annotations: sttran_ant_partial_10_sgdet_future_3_epoch_1.tar
                # Checkpoint name format for label noise: sttran_ant_label_noise_10_sgdet_future_3_epoch_1.tar
                self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
                self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])
                self._conf.max_window = int(self._checkpoint_name.split('_')[-1])
                self._conf.mode = self._checkpoint_name.split('_')[-3]
                print("--------------------------------------------------------")
                print(f"Loading checkpoint with name: {self._checkpoint_name}")
                print(f"Mode: {self._conf.mode}")
                print(f"Max Window: {self._conf.max_window}")
                print("--------------------------------------------------------")
        else:
            # Set the checkpoint name and save path details
            if self._conf.task_name == const.SGG:
                if self._conf.use_partial_annotations:
                    self._checkpoint_name = f"{self._conf.method_name}_partial_{self._conf.partial_percentage}_{self._conf.mode}"
                elif self._conf.use_label_noise:
                    self._checkpoint_name = f"{self._conf.method_name}_label_noise_{self._conf.label_noise_percentage}_{self._conf.mode}"
                else:
                    self._checkpoint_name = f"{self._conf.method_name}_{self._conf.mode}"
                print("--------------------------------------------------------")
                print(f"Training model with name: {self._checkpoint_name}")
                print("--------------------------------------------------------")
            elif self._conf.task_name == const.SGA:
                if self._conf.use_partial_annotations:
                    self._checkpoint_name = f"{self._conf.method_name}_partial_{self._conf.partial_percentage}_{self._conf.mode}_future_{self._conf.max_window}"
                elif self._conf.use_label_noise:
                    self._checkpoint_name = f"{self._conf.method_name}_label_noise_{self._conf.label_noise_percentage}_{self._conf.mode}_future_{self._conf.max_window}"
                else:
                    self._checkpoint_name = f"{self._conf.method_name}_{self._conf.mode}_future_{self._conf.max_window}"
                print("--------------------------------------------------------")
                print(f"Training model with name: {self._checkpoint_name}")
                print("--------------------------------------------------------")

        self._checkpoint_save_dir_path = os.path.join(self._conf.save_path, self._conf.task_name, self._conf.method_name)
        os.makedirs(self._checkpoint_save_dir_path, exist_ok=True)

        # --------------------------- W&B CONFIGURATION ---------------------------
        if self._enable_wandb:
            wandb.init(project=self._checkpoint_name, config=self._conf)

        print("-------------------- CONFIGURATION DETAILS ------------------------")
        for i in self._conf.args:
            print(i, ':', self._conf.args[i])
        print("-------------------------------------------------------------------")

    def _init_optimizer(self):
        if self._conf.optimizer == const.ADAMW:
            self._optimizer = AdamW(self._model.parameters(), lr=self._conf.lr)
        elif self._conf.optimizer == const.ADAM:
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._conf.lr)
        elif self._conf.optimizer == const.SGD:
            self._optimizer = optim.SGD(self._model.parameters(), lr=self._conf.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise NotImplementedError

    def _init_scheduler(self):
        self._scheduler = ReduceLROnPlateau(self._optimizer, "max", patience=1, factor=0.5, verbose=True,
                                            threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def _init_matcher(self):
        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def _load_checkpoint(self):
        if self._model is None:
            raise ValueError("Model is not initialized")

        if self._conf.ckpt:
            if os.path.exists(self._conf.ckpt) is False:
                raise ValueError(f"Checkpoint file {self._conf.ckpt} does not exist")

            try:
                # Load checkpoint to the specified device
                ckpt = torch.load(self._conf.ckpt, map_location=self._device)

                # Determine the key for the state_dict based on availability
                state_dict_key = 'state_dict' if 'state_dict' in ckpt else f'{self._conf.method_name}_state_dict'

                # Load the state dictionary
                self._model.load_state_dict(ckpt[state_dict_key], strict=False)
                print(f"Loaded model from checkpoint {self._conf.ckpt}")

            except FileNotFoundError:
                print(f"Error: Checkpoint file {self._conf.ckpt} not found.")
            except KeyError:
                print(f"Error: Appropriate state_dict not found in the checkpoint.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    @staticmethod
    def _save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, method_name):
        print("*" * 40)
        os.makedirs(checkpoint_save_file_path, exist_ok=True)
        torch.save({f"{method_name}_state_dict": model.state_dict()},
                   os.path.join(checkpoint_save_file_path, f"{checkpoint_name}_epoch_{epoch}.tar"))
        print(f"Saved {method_name} checkpoint after {epoch} epochs")
        print("*" * 40)

    def _init_evaluators(self):
        # For VidSGG set iou_threshold=0.5
        # For SGA set iou_threshold=0
        iou_threshold = 0.5 if self._conf.task_name == 'sgg' else 0.0

        self._evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._train_dataset.object_classes,
            AG_all_predicates=self._train_dataset.relationship_classes,
            AG_attention_predicates=self._train_dataset.attention_relationships,
            AG_spatial_predicates=self._train_dataset.spatial_relationships,
            AG_contacting_predicates=self._train_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            save_file=os.path.join(self._conf.save_path, const.PROGRESS_TEXT_FILE),
            constraint='with'
        )

    @abstractmethod
    def _init_object_detector(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @staticmethod
    def get_sequence_no_tracking(entry, task="sgcls"):
        if task == "predcls":
            indices = []
            for i in entry["labels"].unique():
                indices.append(torch.where(entry["labels"] == i)[0])
            entry["indices"] = indices
            return

        if task == "sgdet" or task == "sgcls":
            # for sgdet, use the predicted object classes, as a special case of
            # the proposed method, comment this out for general coase tracking.
            indices = [[]]
            # indices[0] store single-element sequence, to save memory
            pred_labels = torch.argmax(entry["distribution"], 1)
            for i in pred_labels.unique():
                index = torch.where(pred_labels == i)[0]
                if len(index) == 1:
                    indices[0].append(index)
                else:
                    indices.append(index)
            if len(indices[0]) > 0:
                indices[0] = torch.cat(indices[0])
            else:
                indices[0] = torch.tensor([])
            entry["indices"] = indices
            return