import os
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader

from dataloader.base_easg_dataset import EASGData
from dataloader.label_noise.easg.easg_dataset import LabelNoiseEASG
from dataloader.partial.action_genome.ag_dataset import PartialAG
from dataloader.partial.easg.easg_dataset import PartialEASG
from dataloader.standard.easg.easg_dataset import StandardEASG
from lib.supervised.sgg.easg.easg import EASG
from test_eg_sgg_base import evaluation

from constants import EgoConstants as const

class TrainEgoSGGBase:

    def __init__(self, conf):
        self._train_dataset = None
        self._val_dataset = None
        self._dataloader_train = None
        self._dataloader_val = None
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
            if self._conf.task_name == const.EASG:
                # Checkpoint name format for model trained with full annotations: easg_sgcls_epoch_1.tar
                # Checkpoint name format for model trained with partial annotations: easg_partial_10_sgdet_epoch_1.tar
                # Checkpoint name format for model trained with label noise: easg_label_noise_10_sgdet_epoch_1.tar
                self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
                self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])
                self._conf.mode = self._checkpoint_name.split('_')[-1]
                print("--------------------------------------------------------")
                print(f"Loading checkpoint with name: {self._checkpoint_name}")
                print(f"Mode: {self._conf.mode}")
                print("--------------------------------------------------------")
        else:
            # Set the checkpoint name and save path details
            if self._conf.task_name == const.EASG:
                if self._conf.use_partial_annotations:
                    self._checkpoint_name = f"{self._conf.method_name}_partial_{self._conf.partial_percentage}_{self._conf.mode}"
                elif self._conf.use_label_noise:
                    self._checkpoint_name = f"{self._conf.method_name}_label_noise_{self._conf.label_noise_percentage}_{self._conf.mode}"
                else:
                    self._checkpoint_name = f"{self._conf.method_name}_{self._conf.mode}"
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
            self._train_dataset = PartialEASG(conf=self._conf, split = const.TRAIN)
        elif self._conf.use_label_noise:
            print("-----------------------------------------------------")
            print("Loading the dataset with label noise percentage:", self._conf.label_noise_percentage)
            print("-----------------------------------------------------")
            self._train_dataset = LabelNoiseEASG(conf=self._conf, split = const.TRAIN)
        else:
            print("-----------------------------------------------------")
            print("Loading the standard dataset")
            print("-----------------------------------------------------")
            self._train_dataset = StandardEASG(conf=self._conf, split=const.TRAIN)

        self._val_dataset = StandardEASG(conf=self._conf, split=const.VAL)
        self._dataloader_train = DataLoader(self._train_dataset, shuffle=True)
        self._dataloader_val = DataLoader(self._val_dataset, shuffle=False)


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




def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_annotations', type=Path)
    parser.add_argument('path_to_data', type=Path)
    parser.add_argument('path_to_output', type=Path)
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--sch_param', type=int, default=10, help='parameter for lr scheduler')
    parser.add_argument('--num_epoch', type=int, default=100, help='total number of epochs')
    parser.add_argument('--random_guess', action='store_true', help='for random guessing')
    args = parser.parse_args()
    return args


args = parse_args()

with open(args.path_to_annotations / 'verbs.txt') as f:
    verbs = [l.strip() for l in f.readlines()]

with open(args.path_to_annotations / 'objects.txt') as f:
    objs = [l.strip() for l in f.readlines()]

with open(args.path_to_annotations / 'relationships.txt') as f:
    rels = [l.strip() for l in f.readlines()]

dataset_train = EASGData(args.path_to_annotations, args.path_to_data, 'train', verbs, objs, rels)
dataset_val = EASGData(args.path_to_annotations, args.path_to_data, 'val', verbs, objs, rels)

device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
model = EASG(verbs, objs, rels)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sch_param)
criterion = CrossEntropyLoss()
criterion_rel = BCEWithLogitsLoss()

for epoch in range(1, args.num_epoch + 1):
    model.train()

    list_index = list(range(len(dataset_train)))
    random.shuffle(list_index)
    loss_train = 0.
    for idx in list_index:
        if args.random_guess:
            break

        optimizer.zero_grad()

        graph = dataset_train[idx]
        clip_feat = graph['clip_feat'].unsqueeze(0).to(device)
        obj_feats = graph['obj_feats'].to(device)
        out_verb, out_objs, out_rels = model(clip_feat, obj_feats)

        verb_idx = graph['verb_idx'].to(device)
        obj_indices = graph['obj_indices'].to(device)
        rels_vecs = graph['rels_vecs'].to(device)
        loss = criterion(out_verb, verb_idx) + criterion(out_objs, obj_indices) + criterion_rel(out_rels, rels_vecs)
        loss.backward()
        loss_train += loss.item()
        optimizer.step()

    scheduler.step()

    loss_train /= len(dataset_train)
    recall_predcls_with, recall_predcls_no, recall_sgcls_with, recall_sgcls_no, recall_easgcls_with, recall_easgcls_no = evaluation(
        dataset_val, model, device, args)