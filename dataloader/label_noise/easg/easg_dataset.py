import random

import torch

from dataloader.base_easg_dataset import BaseEASGData


class LabelNoiseEASG(BaseEASGData):

    def __init__(self, conf, split):
        super().__init__(conf, split)

    def _init_graphs(self):
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
        for graph_uid in self.annotations:
            graph = {}
            for aid in self.annotations[graph_uid]['annotations']:
                for i, annt in enumerate(self.annotations[graph_uid]['annotations'][aid]):
                    verb_idx = self.verbs.index(annt['verb'])
                    if verb_idx not in graph:
                        graph[verb_idx] = {}
                        graph[verb_idx]['verb_idx'] = verb_idx
                        graph[verb_idx]['objs'] = {}

                    graph[verb_idx]['clip_feat'] = self.clip_feats[aid]

                    obj_idx = self.objs.index(annt['obj'])
                    if obj_idx not in graph[verb_idx]['objs']:
                        graph[verb_idx]['objs'][obj_idx] = {}
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.zeros((0, 1024), dtype=torch.float32)
                        graph[verb_idx]['objs'][obj_idx]['rels_vec'] = torch.zeros(len(self.rels), dtype=torch.float32)

                    rel_idx = self.rels.index(annt['rel'])
                    graph[verb_idx]['objs'][obj_idx]['rels_vec'][rel_idx] = 1

                    for frameType in self.roi_feats[graph_uid][aid][i]:
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.cat((graph[verb_idx]['objs'][obj_idx][
                                                                                      'obj_feat'],
                                                                                  self.roi_feats[graph_uid][aid][i][
                                                                                      frameType]), dim=0)

            for verb_idx in graph:
                for obj_idx in graph[verb_idx]['objs']:
                    graph[verb_idx]['objs'][obj_idx]['obj_feat'] = graph[verb_idx]['objs'][obj_idx]['obj_feat'].mean(
                        dim=0)

                graphs.append(graph[verb_idx])

        print(f"[{self._conf.method_name}_{self._split}] PREPARING GT GRAPH BATCH DATA ")

        if self._conf.use_label_noise:
            total_num_objs = 0
            for graph in graphs:
                total_num_objs += len(graph['objs'])
            total_num_obj_idx_changes = int(self._conf.label_noise_percentage * 0.01 * total_num_objs)
            num_obj_idx_changes_remaining = 0

        self.graphs = []
        for graph in graphs:
            graph_batch = {}
            verb_idx = graph['verb_idx']

            if self._conf.use_label_noise:
                label_noise_percentage = self._conf.label_noise_percentage * 0.01
                random_num = random.random()
                if random_num <= label_noise_percentage:
                    verb_idx = random.randint(0, len(self.verbs) - 1)

            graph_batch['verb_idx'] = torch.tensor([verb_idx], dtype=torch.long)
            graph_batch['clip_feat'] = graph['clip_feat']
            graph_batch['obj_indices'] = torch.zeros(0, dtype=torch.long)
            graph_batch['obj_feats'] = torch.zeros((0, 1024), dtype=torch.float32)
            graph_batch['rels_vecs'] = torch.zeros((0, len(self.rels)), dtype=torch.float32)
            graph_batch['triplets'] = torch.zeros((0, 3), dtype=torch.long)

            for obj_idx in graph['objs']:
                graph_batch['obj_indices'] = torch.cat(
                    (graph_batch['obj_indices'], torch.tensor([obj_idx], dtype=torch.long)), dim=0
                )
                graph_batch['obj_feats'] = torch.cat(
                    (graph_batch['obj_feats'], graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0
                )

                rels_vec = graph['objs'][obj_idx]['rels_vec']
                graph_batch['rels_vecs'] = torch.cat(
                    (graph_batch['rels_vecs'], rels_vec.unsqueeze(0)), dim=0
                )

                triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    triplets.append((verb_idx, obj_idx, rel_idx.item()))
                graph_batch['triplets'] = torch.cat(
                    (graph_batch['triplets'], torch.tensor(triplets, dtype=torch.long)), dim=0
                )

            self.graphs.append(graph_batch)

        print(f"[{self._conf.method_name}_{self._split}] Finished processing graph data ")

    def __getitem__(self, idx):
        return self.graphs[idx]
