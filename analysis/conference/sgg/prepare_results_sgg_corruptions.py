import csv
import os

from analysis.conference.prepare_results_base import PrepareResultsBase
from constants import CorruptionConstants as const


class PrepareResultsSGGCorruptions(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGGCorruptions, self).__init__()
		self.mode_list = ["sgcls", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		self.scenario_list = ["full", "partial"]
		self.partial_percentages = ["10"]
		
		self.corruption_types = [
			const.NO_CORRUPTION, const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
			const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.MOTION_BLUR, const.ZOOM_BLUR, const.FOG, const.FROST,
			const.SNOW, const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.ELASTIC_TRANSFORM, const.PIXELATE,
			const.JPEG_COMPRESSION, const.SUN_GLARE, const.RAIN, const.DUST, const.WILDFIRE_SMOKE, const.SATURATE,
			const.GLASS_BLUR
		]
		self.dataset_corruption_modes = [const.FIXED, const.MIXED]
		self.video_corruption_modes = [const.FIXED, const.MIXED]
		self.severity_levels = ["3"]
		
		self.task_name = "sgg"
	
	def fetch_sgg_results_json(self):
		db_results = self.fetch_db_sgg_corruptions_results()
		sgg_results_json = {}
		for mode in self.mode_list:
			sgg_results_json[mode] = {}
			for method_name in self.method_list:
				sgg_results_json[mode][method_name] = {}
				for scenario in self.scenario_list:
					sgg_results_json[mode][method_name][scenario] = {}
					if scenario == "full":
						for dataset_corruption_mode in self.dataset_corruption_modes:
							sgg_results_json[mode][method_name][scenario][dataset_corruption_mode] = {}
							if dataset_corruption_mode == const.FIXED:
								for dataset_corruption_type in self.corruption_types:
									sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
										dataset_corruption_type] = {}
									for severity_level in self.severity_levels:
										sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
											dataset_corruption_type][
											severity_level] = self.fetch_empty_metrics_json()
							elif dataset_corruption_mode == const.MIXED:
								for video_corruption_mode in self.video_corruption_modes:
									sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
										video_corruption_mode] = {}
									for severity_level in self.severity_levels:
										sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
											video_corruption_mode][
											severity_level] = self.fetch_empty_metrics_json()
					elif scenario == "partial":
						for partial_num in self.partial_percentages:
							sgg_results_json[mode][method_name][scenario][partial_num] = {}
							for dataset_corruption_mode in self.dataset_corruption_modes:
								sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode] = {}
								if dataset_corruption_mode == const.FIXED:
									for dataset_corruption_type in self.corruption_types:
										sgg_results_json[mode][method_name][scenario][partial_num][
											dataset_corruption_mode][dataset_corruption_type] = {}
										for severity_level in self.severity_levels:
											sgg_results_json[mode][method_name][scenario][partial_num][
												dataset_corruption_mode][dataset_corruption_type][
												severity_level] = self.fetch_empty_metrics_json()
								elif dataset_corruption_mode == const.MIXED:
									for video_corruption_mode in self.video_corruption_modes:
										sgg_results_json[mode][method_name][scenario][partial_num][
											dataset_corruption_mode][video_corruption_mode] = {}
										for severity_level in self.severity_levels:
											sgg_results_json[mode][method_name][scenario][partial_num][
												dataset_corruption_mode][video_corruption_mode][
												severity_level] = self.fetch_empty_metrics_json()
		
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			dataset_corruption_type = sgg_result.dataset_corruption_type
			dataset_corruption_mode = sgg_result.dataset_corruption_mode
			video_corruption_mode = sgg_result.video_corruption_mode
			severity_level = sgg_result.corruption_severity_level
			scenario = sgg_result.scenario_name
			
			if str(severity_level) not in self.severity_levels:
				print(f"Skipping severity level: {severity_level}")
				continue
			
			if scenario == "full":
				if dataset_corruption_mode == const.FIXED:
					sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][dataset_corruption_type][
						severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				elif dataset_corruption_mode == const.MIXED:
					sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][video_corruption_mode][
						severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				else:
					print(f"Skipping dataset corruption mode: {dataset_corruption_mode} under full scenario")
			elif scenario == "partial":
				partial_num = str(sgg_result.partial_percentage)
				if dataset_corruption_mode == const.FIXED:
					sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][
						dataset_corruption_type][severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				elif dataset_corruption_mode == const.MIXED:
					sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][
						video_corruption_mode][severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				else:
					print(f"Skipping dataset corruption mode: {dataset_corruption_mode} under partial scenario")
			else:
				print(f"Skipping scenario: {scenario}")
		
		return sgg_results_json
	
	def generate_sgg_combined_results_csvs_method_wise(self, sgg_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sgg_corruptions_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs",
			                             "mode_results_sgg_corruptions", csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario", "Partial Percentage", "Dataset Corruption Mode", "Video Corruption Mode",
					"Corruption Type", "Severity Level",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100"
				])
				for method_name in self.method_list:
					for scenario in self.scenario_list:
						
						if scenario == "full":
							for dataset_corruption_mode in self.dataset_corruption_modes:
								if dataset_corruption_mode == const.FIXED:
									for corruption_type in self.corruption_types:
										for severity_level in self.severity_levels:
											writer.writerow([
												method_name,
												scenario,
												"-",
												dataset_corruption_mode,
												"Fixed",
												corruption_type,
												severity_level,
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@100"]
											])
								
								elif dataset_corruption_mode == const.MIXED:
									for video_corruption_mode in self.video_corruption_modes:
										for severity_level in self.severity_levels:
											writer.writerow([
												method_name,
												scenario,
												"-",
												dataset_corruption_mode,
												video_corruption_mode,
												"Mixed",
												severity_level,
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@100"]
											])
						
						elif scenario == "partial":
							for partial_num in self.partial_percentages:
								for dataset_corruption_mode in self.dataset_corruption_modes:
									if dataset_corruption_mode == const.FIXED:
										for corruption_type in self.corruption_types:
											for severity_level in self.severity_levels:
												writer.writerow([
													method_name,
													scenario,
													partial_num,
													dataset_corruption_mode,
													"Fixed",
													corruption_type,
													severity_level,
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@100"]
												])
									
									elif dataset_corruption_mode == const.MIXED:
										for video_corruption_mode in self.video_corruption_modes:
											for severity_level in self.severity_levels:
												writer.writerow([
													method_name,
													scenario,
													partial_num,
													dataset_corruption_mode,
													video_corruption_mode,
													"Mixed",
													severity_level,
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@100"]
												])
	
	def compile_sgg_method_wise_results(self):
		sgg_results_json = self.fetch_sgg_results_json()
		self.generate_sgg_combined_results_csvs_method_wise(sgg_results_json)


def main():
	prepare_results_sgg = PrepareResultsSGGCorruptions()
	prepare_results_sgg.compile_sgg_method_wise_results()


def combine_results():
	prepare_results_sgg = PrepareResultsSGGCorruptions()
	prepare_results_sgg.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_sgg_corruptions",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_sgg_corruptions\sgg_corruptions_modes_combined_results.xlsx"
	)


if __name__ == '__main__':
	main()
	combine_results()