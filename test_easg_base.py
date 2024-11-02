import csv
import os
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.standard.easg.easg_dataset import StandardEASG
from easg_base import EASGBase
from constants import EgoConstants as const


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
        with torch.no_grad():
            for b in tqdm(range(len(self._dataloader_val))):
                graph = next(val_iter)

                clip_feat = graph['clip_feat'].unsqueeze(0).to(self._device)
                obj_feats = graph['obj_feats'].to(self._device)
                out_verb, out_objs, out_rels = self._model(clip_feat, obj_feats)

                self._evaluator.evaluate_scene_graph(out_verb, out_objs, out_rels, graph)

    def _collate_evaluation_stats(self):
        stats_json = self._evaluator.fetch_stats_json()

        def evaluator_stats_to_list(with_constraint_evaluator_stats, no_constraint_evaluator_stats):
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
                no_constraint_evaluator_stats["recall"][10],
                no_constraint_evaluator_stats["recall"][20],
                no_constraint_evaluator_stats["recall"][50],
                no_constraint_evaluator_stats["recall"][100],
                no_constraint_evaluator_stats["mean_recall"][10],
                no_constraint_evaluator_stats["mean_recall"][20],
                no_constraint_evaluator_stats["mean_recall"][50],
                no_constraint_evaluator_stats["mean_recall"][100],
                no_constraint_evaluator_stats["harmonic_mean_recall"][10],
                no_constraint_evaluator_stats["harmonic_mean_recall"][20],
                no_constraint_evaluator_stats["harmonic_mean_recall"][50],
                no_constraint_evaluator_stats["harmonic_mean_recall"][100],
            ]

            return collated_stats

        mode_evaluator_stats_dict = {
            "predcls": evaluator_stats_to_list(stats_json["predcls_with"], stats_json["predcls_no"]),
            "sgcls": evaluator_stats_to_list(stats_json["sgcls_with"], stats_json["sgcls_no"]),
            "easg": evaluator_stats_to_list(stats_json["easg_with"], stats_json["easg_no"])
        }

        return mode_evaluator_stats_dict

    def _publish_evaluation_results(self):
        mode_evaluator_stats_dict = self._collate_evaluation_stats()
        self._write_evaluation_statistics(mode_evaluator_stats_dict)

    def _write_evaluation_statistics(self, mode_evaluator_stats_dict):
        # Create the results directory
        results_dir = os.path.join(os.getcwd(), 'results')
        task_dir = os.path.join(results_dir, "easg")

        for mode in mode_evaluator_stats_dict.keys():
            if self._conf.use_input_corruptions:
                scenario_dir = os.path.join(task_dir, "corruptions")
                file_name = f'{self._conf.method_name}_{mode}_{self._corruption_name}.csv'
            elif self._conf.use_partial_annotations:
                scenario_dir = os.path.join(task_dir, "partial")
                file_name = f'{self._conf.method_name}_partial_{self._conf.partial_percentage}_{mode}.csv'
            elif self._conf.use_label_noise:
                scenario_dir = os.path.join(task_dir, "labelnoise")
                file_name = f'{self._conf.method_name}_labelnoise_{self._conf.label_noise_percentage}_{mode}.csv'
            else:
                scenario_dir = os.path.join(task_dir, "full")
                file_name = f'{self._conf.method_name}_{mode}.csv'

            assert scenario_dir is not None, "Scenario directory is not set"
            mode_results_dir = os.path.join(scenario_dir, mode)
            os.makedirs(mode_results_dir, exist_ok=True)
            results_file_path = os.path.join(mode_results_dir, file_name)

            with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
                writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
                # Use this as the reference knowing what we write in the csv file

                # if not os.path.isfile(results_file_path):
                #     writer.writerow([
                #         "Method Name",
                #         "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                #         "hR@100",
                #         "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                #         "hR@100"
                #     ])
                writer.writerow(mode_evaluator_stats_dict[mode])

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
