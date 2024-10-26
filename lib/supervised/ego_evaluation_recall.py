from math import ceil

import torch


class BasicEgoActionSceneGraphEvaluator:

    def __init__(self, conf):
        self._conf = conf

        self.num_top_verb = 5
        self.num_top_rel_with = 1
        self.num_top_rel_no = 5
        self.list_k = [10, 20, 50]

        self._init_recall_dicts()

    def _init_recall_dicts(self):
        self.recall_predcls_with = {k: [] for k in self.list_k}
        self.recall_predcls_no = {k: [] for k in self.list_k}
        self.recall_sgcls_with = {k: [] for k in self.list_k}
        self.recall_sgcls_no = {k: [] for k in self.list_k}
        self.recall_easgcls_with = {k: [] for k in self.list_k}
        self.recall_easgcls_no = {k: [] for k in self.list_k}

    def reset_result(self):
        for k in self.list_k:
            self.recall_predcls_with[k] = []
            self.recall_predcls_no[k] = []
            self.recall_sgcls_with[k] = []
            self.recall_sgcls_no[k] = []
            self.recall_easgcls_with[k] = []
            self.recall_easgcls_no[k] = []

    def print_stats(self):
        for k in self.list_k:
            self.recall_predcls_with[k] = sum(self.recall_predcls_with[k]) / len(self.recall_predcls_with[k]) * 100
            self.recall_predcls_no[k] = sum(self.recall_predcls_no[k]) / len(self.recall_predcls_no[k]) * 100
            self.recall_sgcls_with[k] = sum(self.recall_sgcls_with[k]) / len(self.recall_sgcls_with[k]) * 100
            self.recall_sgcls_no[k] = sum(self.recall_sgcls_no[k]) / len(self.recall_sgcls_no[k]) * 100
            self.recall_easgcls_with[k] = sum(self.recall_easgcls_with[k]) / len(self.recall_easgcls_with[k]) * 100
            self.recall_easgcls_no[k] = sum(self.recall_easgcls_no[k]) / len(self.recall_easgcls_no[k]) * 100

        for k in self.list_k:
            print("Recall@{}:".format(k))
            print("  PredCls with: {:.2f}".format(self.recall_predcls_with[k]))
            print("  PredCls no: {:.2f}".format(self.recall_predcls_no[k]))
            print("  SGCls with: {:.2f}".format(self.recall_sgcls_with[k]))
            print("  SGCls no: {:.2f}".format(self.recall_sgcls_no[k]))
            print("  EASGCls with: {:.2f}".format(self.recall_easgcls_with[k]))
            print("  EASGCls no: {:.2f}".format(self.recall_easgcls_no[k]))


    @staticmethod
    def intersect_2d(out, gt):
        return (out[..., None] == gt.T[None, ...]).all(1)

    def evaluate_scene_graph(self, out_verb, out_objs, out_rels, gt_graph):
        scores_verb = out_verb[0].detach().cpu().softmax(dim=0)
        scores_objs = out_objs.detach().cpu().softmax(dim=1)
        scores_rels = out_rels.detach().cpu().sigmoid()

        if self._conf.random_guess:
            scores_verb = torch.rand(scores_verb.shape)
            scores_verb /= scores_verb.sum()
            scores_objs = torch.rand(scores_objs.shape)
            scores_objs /= scores_objs.sum()
            scores_rels = torch.rand(scores_rels.shape)

        verb_idx = gt_graph['verb_idx']
        obj_indices = gt_graph['obj_indices']
        rels_vecs = gt_graph['rels_vecs']
        triplets_gt = gt_graph['triplets']
        num_obj = obj_indices.shape[0]

        # make triplets for precls
        triplets_pred_with = []
        scores_pred_with = []
        triplets_pred_no = []
        scores_pred_no = []
        for obj_idx, scores_rel in zip(obj_indices, scores_rels):
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for ri in sorted_scores_rel[:self.num_top_rel_with]:
                triplets_pred_with.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_with.append(scores_rel[ri].item())

            for ri in sorted_scores_rel[:ceil(max(self.list_k) / num_obj)]:
                triplets_pred_no.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_no.append(scores_rel[ri].item())

        # make triplets for sgcls
        triplets_sg_with = []
        scores_sg_with = []
        triplets_sg_no = []
        scores_sg_no = []
        num_top_obj_with = ceil(max(self.list_k) / (self.num_top_rel_with * num_obj))
        num_top_obj_no = ceil(max(self.list_k) / (self.num_top_rel_no * num_obj))
        for scores_obj, scores_rel in zip(scores_objs, scores_rels):
            sorted_scores_obj = scores_obj.argsort(descending=True)
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for oi in sorted_scores_obj[:num_top_obj_with]:
                for ri in sorted_scores_rel[:self.num_top_rel_with]:
                    triplets_sg_with.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_with.append((scores_obj[oi] + scores_rel[ri]).item())
            for oi in sorted_scores_obj[:num_top_obj_no]:
                for ri in sorted_scores_rel[:self.num_top_rel_no]:
                    triplets_sg_no.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_no.append((scores_obj[oi] + scores_rel[ri]).item())

        # make triplets for easgcls
        triplets_easg_with = []
        scores_easg_with = []
        triplets_easg_no = []
        scores_easg_no = []
        num_top_obj_with = ceil(max(self.list_k) / (self.num_top_verb * self.num_top_rel_with * num_obj))
        num_top_obj_no = ceil(max(self.list_k) / (self.num_top_verb * self.num_top_rel_no * num_obj))
        for vi in scores_verb.argsort(descending=True)[:self.num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:self.num_top_rel_with]:
                        triplets_easg_with.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_with.append((scores_verb[vi] + scores_obj[oi] + scores_rel[ri]).item())
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:self.num_top_rel_no]:
                        triplets_easg_no.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_no.append((scores_verb[vi] + scores_obj[oi] + scores_rel[ri]).item())

        triplets_pred_with = torch.tensor(triplets_pred_with, dtype=torch.long)
        triplets_pred_no = torch.tensor(triplets_pred_no, dtype=torch.long)
        triplets_sg_with = torch.tensor(triplets_sg_with, dtype=torch.long)
        triplets_sg_no = torch.tensor(triplets_sg_no, dtype=torch.long)
        triplets_easg_with = torch.tensor(triplets_easg_with, dtype=torch.long)
        triplets_easg_no = torch.tensor(triplets_easg_no, dtype=torch.long)

        # sort the triplets using the averaged scores
        triplets_pred_with = triplets_pred_with[torch.argsort(torch.tensor(scores_pred_with), descending=True)]
        triplets_pred_no = triplets_pred_no[torch.argsort(torch.tensor(scores_pred_no), descending=True)]
        triplets_sg_with = triplets_sg_with[torch.argsort(torch.tensor(scores_sg_with), descending=True)]
        triplets_sg_no = triplets_sg_no[torch.argsort(torch.tensor(scores_sg_no), descending=True)]
        triplets_easg_with = triplets_easg_with[torch.argsort(torch.tensor(scores_easg_with), descending=True)]
        triplets_easg_no = triplets_easg_no[torch.argsort(torch.tensor(scores_easg_no), descending=True)]

        out_to_gt_pred_with = self.intersect_2d(triplets_gt, triplets_pred_with)
        out_to_gt_pred_no = self.intersect_2d(triplets_gt, triplets_pred_no)
        out_to_gt_sg_with = self.intersect_2d(triplets_gt, triplets_sg_with)
        out_to_gt_sg_no = self.intersect_2d(triplets_gt, triplets_sg_no)
        out_to_gt_easg_with = self.intersect_2d(triplets_gt, triplets_easg_with)
        out_to_gt_easg_no = self.intersect_2d(triplets_gt, triplets_easg_no)

        num_gt = triplets_gt.shape[0]
        for k in self.list_k:
            self.recall_predcls_with[k].append(out_to_gt_pred_with[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_predcls_no[k].append(out_to_gt_pred_no[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_sgcls_with[k].append(out_to_gt_sg_with[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_sgcls_no[k].append(out_to_gt_sg_no[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_easgcls_with[k].append(out_to_gt_easg_with[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_easgcls_no[k].append(out_to_gt_easg_no[:, :k].any(dim=1).sum().item() / num_gt)
