from abc import abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.standard.easg.easg_dataset import StandardEASG
from easg_base import EASGBase
from Constants import EgoConstants as const

class TestEASGBase(EASGBase):

    def __init__(self, conf):
        super().__init__(conf)

    # ----------------- Load the dataset -------------------------
    # Three main settings:
    # (a) Standard Dataset: Where full annotations are used
    # (b) Partial Annotations: Where partial object and relationship annotations are used
    # (c) Label Noise: Where label noise is added to the dataset
    # -------------------------------------------------------------
    def _init_dataset(self):
        self._val_dataset = StandardEASG(conf=self._conf, split=const.VAL)
        self._dataloader_val = DataLoader(self._val_dataset, shuffle=False)

    def _test_model(self):
        val_iter = iter(self._dataloader_val)
        self._model.eval()
        # self._object_detector.is_train = False
        with torch.no_grad():
            for b in tqdm(range(len(self._dataloader_val))):
                graph = next(val_iter)

                clip_feat = graph['clip_feat'].unsqueeze(0).to(self._device)
                obj_feats = graph['obj_feats'].to(self._device)
                out_verb, out_objs, out_rels = self._model(clip_feat, obj_feats)

                self._evaluator.evaluate_scene_graph(out_verb, out_objs, out_rels, graph)

    def _collate_evaluation_stats(self):
        with_constraint_evaluator_stats = self._evaluator.fetch_stats_json()

        collated_stats = [
            self._conf.method_name,
            with_constraint_evaluator_stats["recall"][10],
            with_constraint_evaluator_stats["recall"][20],
            with_constraint_evaluator_stats["recall"][50],
            with_constraint_evaluator_stats["recall"][100],
            with_constraint_evaluator_stats["mean_recall"][10],
            with_constraint_evaluator_stats["mean_recall"][20],
            with_constraint_evaluator_stats["mean_recall"][50],
            with_constraint_evaluator_stats["mean_recall"][100],
            with_constraint_evaluator_stats["harmonic_mean_recall"][10],
            with_constraint_evaluator_stats["harmonic_mean_recall"][20],
            with_constraint_evaluator_stats["harmonic_mean_recall"][50],
            with_constraint_evaluator_stats["harmonic_mean_recall"][100],
        ]
        return collated_stats


    @abstractmethod
    def init_model(self):
        pass

    def init_method_evaluation(self):
        # 0. Init config
        self._init_config()

        # 1. Initialize the dataset
        self._init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pretrained model
        self.init_model()
        self._load_checkpoint()
        # self._init_object_detector()

        # 4. Test the model
        self._test_model()

        # 5. Publish the evaluation results
        self._publish_evaluation_results()