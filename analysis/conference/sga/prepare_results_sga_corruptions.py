from analysis.conference.prepare_results_base import PrepareResultsBase
from constants import CorruptionConstants as const


class PrepareResultsSGA(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGA, self).__init__()
		self.scenario_list = ["full", "partial"]
		self.mode_list = ["sgcls", "predcls"]
		self.method_list = ["sttran_ant", "dsgdetr_ant", "sttran_gen_ant", "dsgdetr_gen_ant", "ode", "sde"]
		self.partial_percentages = [10]
		self.task_name = "sga"
		
		self.corruption_types = [
			const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
			const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.MOTION_BLUR, const.ZOOM_BLUR, const.FOG, const.FROST,
			const.SNOW, const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.ELASTIC_TRANSFORM, const.PIXELATE,
			const.JPEG_COMPRESSION, const.SUN_GLARE, const.RAIN, const.DUST, const.WILDFIRE_SMOKE, const.SATURATE,
			const.GLASS_BLUR
		]
		self.dataset_corruption_modes = [const.FIXED, const.MIXED]
		self.video_corruption_modes = [const.FIXED, const.MIXED]
		self.severity_levels = ["3"]
		
		self.context_fraction_list = ['0.3', '0.5', '0.7', '0.9']
	
	def fetch_sga_results_json(self):
		db_results = self.fetch_db_sgg_corruptions_results()
		sga_results_json = {}
		for mode in self.mode_list:
			sga_results_json[mode] = {}
			for method_name in self.method_list:
				sga_results_json[mode][method_name] = {}
				for scenario in self.scenario_list:
					sga_results_json[mode][method_name][scenario] = {}
					if scenario == "full":
						for dataset_corruption_mode in self.dataset_corruption_modes:
							sga_results_json[mode][method_name][scenario][dataset_corruption_mode] = {}
							if dataset_corruption_mode == const.FIXED:
								for dataset_corruption_type in self.corruption_types:
									sga_results_json[mode][method_name][scenario][dataset_corruption_mode][
										dataset_corruption_type] = {}
									for severity_level in self.severity_levels:
										sga_results_json[mode][method_name][scenario][dataset_corruption_mode][
											dataset_corruption_type][
											severity_level] = self.fetch_empty_metrics_json()
							elif dataset_corruption_mode == const.MIXED:
								for video_corruption_mode in self.video_corruption_modes:
									sga_results_json[mode][method_name][scenario][dataset_corruption_mode][
										video_corruption_mode] = {}
									for severity_level in self.severity_levels:
										sga_results_json[mode][method_name][scenario][dataset_corruption_mode][
											video_corruption_mode][
											severity_level] = self.fetch_empty_metrics_json()
					elif scenario == "partial":
						for partial_num in self.partial_percentages:
							sga_results_json[mode][method_name][scenario][partial_num] = {}
							for dataset_corruption_mode in self.dataset_corruption_modes:
								sga_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode] = {}
								if dataset_corruption_mode == const.FIXED:
									for dataset_corruption_type in self.corruption_types:
										sga_results_json[mode][method_name][scenario][partial_num][
											dataset_corruption_mode][dataset_corruption_type] = {}
										for severity_level in self.severity_levels:
											sga_results_json[mode][method_name][scenario][partial_num][
												dataset_corruption_mode][dataset_corruption_type][
												severity_level] = self.fetch_empty_metrics_json()
								elif dataset_corruption_mode == const.MIXED:
									for video_corruption_mode in self.video_corruption_modes:
										sga_results_json[mode][method_name][scenario][partial_num][
											dataset_corruption_mode][video_corruption_mode] = {}
										for severity_level in self.severity_levels:
											sga_results_json[mode][method_name][scenario][partial_num][
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
					sga_results_json[mode][method_name][scenario][dataset_corruption_mode][dataset_corruption_type][
						severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				elif dataset_corruption_mode == const.MIXED:
					sga_results_json[mode][method_name][scenario][dataset_corruption_mode][video_corruption_mode][
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
					sga_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][
						dataset_corruption_type][severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				elif dataset_corruption_mode == const.MIXED:
					sga_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][
						video_corruption_mode][severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				else:
					print(f"Skipping dataset corruption mode: {dataset_corruption_mode} under partial scenario")
			else:
				print(f"Skipping scenario: {scenario}")
		
		return sga_results_json
