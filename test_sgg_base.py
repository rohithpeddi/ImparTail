import copy
import csv
import os
import torch
from abc import abstractmethod

from torch.utils.data import DataLoader
from tqdm import tqdm
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result, ResultDetails, Metrics
from dataloader.corrupted.image_based.ag_dataset import ImageCorruptedAG
from dataloader.corrupted.image_based.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from lib.object_detector import Detector
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from stsg_base import STSGBase


class TestSGGBase(STSGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._model = None

        # Load while initializing the object detector
        self._object_detector = None

        # Load while initializing the dataset
        self._dataloader_test = None
        self._test_dataset = None
        self._object_classes = None

        # Load checkpoint name
        self._checkpoint_name = None

        # Load the evaluators
        self._evaluators = []

        # Define the corruption function name
        # (Dataset - Video - Corruption Type) Possible Values
        # (Fixed - Fixed - {corruption_name})
        # (Mixed - Fixed), (Mixed - Mixed)
        # [{fixed_fixed_rain}, {mixed_fixed}, {mixed_mixed}]
        self._corruption_name = None

    def _init_evaluators(self):
        print("Replaced default single evaluator with multiple evaluators")
        # Evaluators order - [With Constraint, No Constraint, Semi Constraint]
        iou_threshold = 0.5 if self._conf.task_name == 'sgg' else 0.0
        self._with_constraint_evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes,
            AG_all_predicates=self._test_dataset.relationship_classes,
            AG_attention_predicates=self._test_dataset.attention_relationships,
            AG_spatial_predicates=self._test_dataset.spatial_relationships,
            AG_contacting_predicates=self._test_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            constraint='with')

        self._no_constraint_evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes,
            AG_all_predicates=self._test_dataset.relationship_classes,
            AG_attention_predicates=self._test_dataset.attention_relationships,
            AG_spatial_predicates=self._test_dataset.spatial_relationships,
            AG_contacting_predicates=self._test_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            constraint='no')

        self._semi_constraint_evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes,
            AG_all_predicates=self._test_dataset.relationship_classes,
            AG_attention_predicates=self._test_dataset.attention_relationships,
            AG_spatial_predicates=self._test_dataset.spatial_relationships,
            AG_contacting_predicates=self._test_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            constraint='semi', semi_threshold=0.9)

        self._evaluators.append(self._with_constraint_evaluator)
        self._evaluators.append(self._no_constraint_evaluator)
        self._evaluators.append(self._semi_constraint_evaluator)

    def _init_object_detector(self):
        self._object_detector = Detector(
            train=True,
            object_classes=self._object_classes,
            use_SUPPLY=True,
            mode=self._conf.mode
        ).to(device=self._device)
        self._object_detector.eval()

    def _collate_evaluation_stats(self):
        with_constraint_evaluator_stats = self._with_constraint_evaluator.fetch_stats_json()
        no_constraint_evaluator_stats = self._no_constraint_evaluator.fetch_stats_json()
        semi_constraint_evaluator_stats = self._semi_constraint_evaluator.fetch_stats_json()

        collated_stats = [
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
            semi_constraint_evaluator_stats["recall"][10],
            semi_constraint_evaluator_stats["recall"][20],
            semi_constraint_evaluator_stats["recall"][50],
            semi_constraint_evaluator_stats["recall"][100],
            semi_constraint_evaluator_stats["mean_recall"][10],
            semi_constraint_evaluator_stats["mean_recall"][20],
            semi_constraint_evaluator_stats["mean_recall"][50],
            semi_constraint_evaluator_stats["mean_recall"][100],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][10],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][20],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][50],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][100]
        ]
        return collated_stats

    def _evaluate_predictions(self, gt_annotation, prediction):
        for evaluator in self._evaluators:
            evaluator.evaluate_scene_graph(gt_annotation, prediction)

    def _publish_evaluation_results(self):
        # 1. Collate the evaluation statistics
        self._collated_stats = self._collate_evaluation_stats()
        # 2. Write to the CSV File
        self._write_evaluation_statistics()
        # 3. Publish the results to Firebase
        # self._publish_results_to_firebase()

    def _write_evaluation_statistics(self):
        # Create the results directory
        results_dir = os.path.join(os.getcwd(), 'results')
        mode_results_dir = os.path.join(results_dir, self._conf.mode)
        os.makedirs(mode_results_dir, exist_ok=True)

        # Create the results file
        results_file_path = os.path.join(mode_results_dir,
                                         f'{self._conf.method_name}_{self._conf.mode}_{self._corruption_name}.csv')

        with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
            writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
            # Write the header if the file is empty
            if not os.path.isfile(results_file_path):
                writer.writerow([
                    "Method Name",
                    "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                    "hR@100"
                    "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                    "hR@100",
                    "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                    "hR@100"
                ])
                # Write the results row
            writer.writerow(self._collated_stats)

    @staticmethod
    def _prepare_metrics_from_stats(evaluator_stats):
        metrics = Metrics(
            evaluator_stats["recall"][10],
            evaluator_stats["recall"][20],
            evaluator_stats["recall"][50],
            evaluator_stats["recall"][100],
            evaluator_stats["mean_recall"][10],
            evaluator_stats["mean_recall"][20],
            evaluator_stats["mean_recall"][50],
            evaluator_stats["mean_recall"][100],
            evaluator_stats["harmonic_mean_recall"][10],
            evaluator_stats["harmonic_mean_recall"][20],
            evaluator_stats["harmonic_mean_recall"][50],
            evaluator_stats["harmonic_mean_recall"][100]
        )

        return metrics

    def _publish_results_to_firebase(self):
        db_service = FirebaseService()

        result = Result(
            task_name=self._conf.task_name,
            method_name=self._conf.method_name,
            mode=self._conf.mode,
            corruption_type=self._corruption_name,
        )

        result_details = ResultDetails()
        with_constraint_metrics = self._prepare_metrics_from_stats(self._evaluators[0].fetch_stats_json())
        no_constraint_metrics = self._prepare_metrics_from_stats(self._evaluators[1].fetch_stats_json())
        semi_constraint_metrics = self._prepare_metrics_from_stats(self._evaluators[2].fetch_stats_json())

        result_details.add_with_constraint_metrics(with_constraint_metrics)
        result_details.add_no_constraint_metrics(no_constraint_metrics)
        result_details.add_semi_constraint_metrics(semi_constraint_metrics)

        result.add_result_details(result_details)

        print("Saving result: ", result.result_id)
        db_service.update_result(result.result_id, result.to_dict())
        print("Saved result: ", result.result_id)

        return result

    def _test_model(self):
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        self._object_detector.is_train = False
        with torch.no_grad():
            for num_video_id in tqdm(range(len(self._dataloader_test)), desc="Testing Progress", ascii=True):
                data = next(test_iter)
                im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                gt_annotation = self._test_dataset.gt_annotations[data[4]]
                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

                # ----------------- Process the video (Method Specific) -----------------
                prediction = self.process_test_video(entry, frame_size, gt_annotation)
                # ----------------------------------------------------------------------

                self._evaluate_predictions(gt_annotation, prediction)
            print('-----------------------------------------------------------------------------------', flush=True)

    def _init_dataset(self):
        # Using the parameters set in the configuration file, initialize the corrupted dataset
        self._test_dataset = ImageCorruptedAG(
            phase='test',
            datasize=self._conf.datasize,
            data_path=self._conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self._conf.mode == 'predcls' else True,
            dataset_corruption_mode=self._conf.dataset_corruption_mode,
            video_corruption_mode=self._conf.video_corruption_mode,
            dataset_corruption_type=self._conf.dataset_corruption_type,
            corruption_severity_level=self._conf.corruption_severity_level
        )

        self._object_classes = self._test_dataset.object_classes

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=False
        )

        self._corruption_name = (f"{self._conf.dataset_corruption_mode}_{self._conf.video_corruption_mode}_"
                                 f"{self._conf.dataset_corruption_type}_{self._conf.corruption_severity_level}")

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
        self._init_object_detector()

        # 4. Test the model
        self._test_model()

        # 5. Publish the evaluation results
        self._publish_evaluation_results()

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def process_test_video(self, video_entry, frame_size, gt_annotation) -> dict:
        pass
