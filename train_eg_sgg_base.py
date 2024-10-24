import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from dataloader.base_easg_dataset import EASGData
from lib.supervised.sgg.easg.easg import EASG
from test_eg_sgg_base import evaluation


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