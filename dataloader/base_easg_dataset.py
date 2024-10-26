import os
import pickle

import torch
from torch.utils.data import Dataset


class BaseEASGData(Dataset):

    def __init__(self, conf, split):
        self._conf = conf
        self._path_to_annotations = self._conf.path_to_annotations
        self._path_to_data = self._conf.path_to_data
        self._split = split

        print(f"[{self._conf.method_name}_{self._split}] LOADING ANNOTATION DATA")

        annotations_file_path = os.path.join(self._path_to_annotations, f'easg_{self._split}.pkl')
        with open(annotations_file_path, 'rb') as f:
            annotations = pickle.load(f)

        verbs_file_path = os.path.join(self._path_to_annotations, 'verbs.txt')
        with open(verbs_file_path) as f:
            verbs = [l.strip() for l in f.readlines()]

        objs_file_path = os.path.join(self._path_to_annotations, 'objects.txt')
        with open(objs_file_path) as f:
            objs = [l.strip() for l in f.readlines()]

        rels_file_path = os.path.join(self._path_to_annotations, 'relationships.txt')
        with open(rels_file_path) as f:
            rels = [l.strip() for l in f.readlines()]

        print(f"[{self._conf.method_name}_{self._split}] LOADING FEATURES DATA ")

        roi_feats_file_path = os.path.join(self._path_to_data, f'roi_feats_{self._split}.pkl')
        with open(roi_feats_file_path, 'rb') as f:
            roi_feats = pickle.load(f)

        clip_feats_file_path = os.path.join(self._path_to_data, f'verb_features.pt')
        clip_feats = torch.load(clip_feats_file_path)

        # Making these things accessible for functions in other methods.
        self.roi_feats = roi_feats
        self.clip_feats = clip_feats
        self.verbs = verbs
        self.objs = objs
        self.rels = rels

        """
        graph:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['objs']: dict of obj_idx
                dict[obj_idx]: dict
                    dict['obj_feat']: 1024-D ROI feature vector
                    dict['rels_vec']: multi-hot vector of relationships

        graph_batch:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['obj_indices']: batched version of obj_idx
            dict['obj_feats']: batched version of obj_feat
            dict['rels_vecs']: batched version of rels_vec
            dict['triplets']: all the triplets consisting of (verb, obj, rel)
        """

        print(f"[{self._conf.method_name}_{self._split}] PREPARING GT GRAPH DATA AND FEATURES ")
        graphs = []
        for graph_uid in annotations:
            graph = {}
            for aid in annotations[graph_uid]['annotations']:
                for i, annt in enumerate(annotations[graph_uid]['annotations'][aid]):
                    verb_idx = verbs.index(annt['verb'])
                    if verb_idx not in graph:
                        graph[verb_idx] = {}
                        graph[verb_idx]['verb_idx'] = verb_idx
                        graph[verb_idx]['objs'] = {}

                    graph[verb_idx]['clip_feat'] = clip_feats[aid]

                    obj_idx = objs.index(annt['obj'])
                    if obj_idx not in graph[verb_idx]['objs']:
                        graph[verb_idx]['objs'][obj_idx] = {}
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.zeros((0, 1024), dtype=torch.float32)
                        graph[verb_idx]['objs'][obj_idx]['rels_vec'] = torch.zeros(len(rels), dtype=torch.float32)

                    rel_idx = rels.index(annt['rel'])
                    graph[verb_idx]['objs'][obj_idx]['rels_vec'][rel_idx] = 1

                    for frameType in roi_feats[graph_uid][aid][i]:
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.cat((graph[verb_idx]['objs'][obj_idx]['obj_feat'], roi_feats[graph_uid][aid][i][frameType]), dim=0)

            for verb_idx in graph:
                for obj_idx in graph[verb_idx]['objs']:
                    graph[verb_idx]['objs'][obj_idx]['obj_feat'] = graph[verb_idx]['objs'][obj_idx]['obj_feat'].mean(dim=0)

                graphs.append(graph[verb_idx])

        print(f"[{self._conf.method_name}_{self._split}] PREPARING GT GRAPH BATCH DATA ")

        self.graphs = []
        for graph in graphs:
            graph_batch = {}
            verb_idx = graph['verb_idx']
            graph_batch['verb_idx'] = torch.tensor([verb_idx], dtype=torch.long)
            graph_batch['clip_feat'] = graph['clip_feat']
            graph_batch['obj_indices'] = torch.zeros(0, dtype=torch.long)
            graph_batch['obj_feats'] = torch.zeros((0, 1024), dtype=torch.float32)
            graph_batch['rels_vecs'] = torch.zeros((0, len(rels)), dtype=torch.float32)
            graph_batch['triplets'] = torch.zeros((0, 3), dtype=torch.long)

            for obj_idx in graph['objs']:
                graph_batch['obj_indices'] = torch.cat((graph_batch['obj_indices'], torch.tensor([obj_idx], dtype=torch.long)), dim=0)
                graph_batch['obj_feats'] = torch.cat((graph_batch['obj_feats'], graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0)

                rels_vec = graph['objs'][obj_idx]['rels_vec']
                graph_batch['rels_vecs'] = torch.cat((graph_batch['rels_vecs'], rels_vec.unsqueeze(0)), dim=0)

                triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    triplets.append((verb_idx, obj_idx, rel_idx.item()))
                graph_batch['triplets'] = torch.cat((graph_batch['triplets'], torch.tensor(triplets, dtype=torch.long)), dim=0)

            self.graphs.append(graph_batch)

        print(f"[{self._conf.method_name}_{self._split}] Finished processing graph data ")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]