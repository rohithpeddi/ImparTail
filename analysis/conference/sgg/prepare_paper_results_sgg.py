import csv
import os

from analysis.conference.prepare_results_base import PrepareResultsBase


class PreparePaperResultSGG(PrepareResultsBase):
	
	def __init__(self):
		super(PreparePaperResultSGG, self).__init__()
		self.scenario_list = ["full", "partial"]
		self.mode_list = ["sgcls", "sgdet", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		self.partial_percentages = [10]
		self.task_name = "sgg"
	
	def fetch_sgg_recall_results_json(self):
		db_results = self.fetch_db_sgg_results()
		
		sgg_results_json = {}
		# Has the following structure:
		# {
		# 	"sttran": {
		# 		"partial": {
		#			"10": {
		# 			    "with_constraint": {
		# 				    "R@10": 0.0,
		# 				    "R@20": 0.0,
		# 				    "R@50": 0.0,
		# 				    "R@100": 0.0
		# 			    },
		# 			    "no_constraint": {
		# 				    "R@10": 0.0,
		# 				    "R@20": 0.0,
		# 				    "R@50": 0.0,
		# 				    "R@100": 0.0
		# 			    },
		# 			    "semi_constraint": {
		# 				    "R@10": 0.0,
		# 				    "R@20": 0.0,
		# 				    "R@50": 0.0,
		# 				    "R@100": 0.0
		# 			    }
		# 			}
		# 		},
		# 		"full": {....}
		# 	},
		# 	"dsgdetr": {....}
		# }
		
		for method in self.method_list:
			sgg_results_json[method] = {}
			for scenario_name in self.scenario_list:
				sgg_results_json[method][scenario_name] = {}
				if scenario_name == "full":
					for mode in self.mode_list:
						sgg_results_json[method][scenario_name][
							mode] = self.fetch_paper_recall_empty_metrics_json()
				else:
					percentage_list = self.partial_percentages
					for percentage_num in percentage_list:
						sgg_results_json[method][scenario_name][percentage_num] = {}
						for mode in self.mode_list:
							sgg_results_json[method][scenario_name][percentage_num][
								mode] = self.fetch_paper_recall_empty_metrics_json()
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][mode] = completed_recall_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][percentage_num][mode] = completed_recall_metrics_json
		return sgg_results_json
	
	def fetch_sgg_mean_recall_results_json(self):
		db_results = self.fetch_db_sgg_results()
		
		sgg_results_json = {}
		# Has the following structure:
		# {
		# 	"sttran": {
		# 		"partial": {
		#			"10": {
		# 			    "with_constraint": {
		# 				    "mR@10": 0.0,
		# 				    "mR@20": 0.0,
		# 				    "mR@50": 0.0,
		# 				    "mR@100": 0.0
		# 			    },
		# 			    "no_constraint": {
		# 				    "mR@10": 0.0,
		# 				    "mR@20": 0.0,
		# 				    "mR@50": 0.0,
		# 				    "mR@100": 0.0
		# 			    },
		# 			    "semi_constraint": {
		# 				    "mR@10": 0.0,
		# 				    "mR@20": 0.0,
		# 				    "mR@50": 0.0,
		# 				    "mR@100": 0.0
		# 			    }
		# 			}
		# 		},
		# 		"full": {....}
		# 	},
		# 	"dsgdetr": {....}
		# }
		
		for method in self.method_list:
			sgg_results_json[method] = {}
			for scenario_name in self.scenario_list:
				sgg_results_json[method][scenario_name] = {}
				if scenario_name == "full":
					for mode in self.mode_list:
						sgg_results_json[method][scenario_name][
							mode] = self.fetch_paper_mean_recall_empty_metrics_json()
				else:
					percentage_list = self.partial_percentages
					for percentage_num in percentage_list:
						sgg_results_json[method][scenario_name][percentage_num] = {}
						for mode in self.mode_list:
							sgg_results_json[method][scenario_name][percentage_num][
								mode] = self.fetch_paper_mean_recall_empty_metrics_json()
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][mode] = completed_mean_recall_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][percentage_num][mode] = completed_mean_recall_metrics_json
		return sgg_results_json
	
	def generate_sgg_mean_recall_results_csvs_method_wise(self, sgg_mean_recall_results_json):
		csv_file_name = "sgg_mean_recall.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_sgg_results_csvs",
		                             csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Method Name", "Scenario Name", "Partial Percentage",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
			])
			
			for method_name in self.method_list:
				for scenario_name in self.scenario_list:
					if scenario_name == "full":
						writer.writerow([
							method_name,
							scenario_name,
							"-",
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@100"]
						])
						continue
					else:
						percentage_list = self.partial_percentages
						for percentage_num in percentage_list:
							writer.writerow([
								method_name,
								scenario_name,
								percentage_num,
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@100"]
							])
	
	def generate_sgg_recall_results_csvs_method_wise(self, sgg_recall_results_json):
		csv_file_name = "sgg_recall.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_sgg_results_csvs",
		                             csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Method Name", "Scenario Name", "Partial Percentage",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100",
			])
			
			for method_name in self.method_list:
				for scenario_name in self.scenario_list:
					if scenario_name == "full":
						writer.writerow([
							method_name,
							scenario_name,
							"-",
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@100"]
						])
						continue
					else:
						percentage_list = self.partial_percentages
						for percentage_num in percentage_list:
							writer.writerow([
								method_name,
								scenario_name,
								percentage_num,
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@100"]
							])
	
	def compile_sgg_method_wise_results(self):
		sgg_mean_recall_results_json = self.fetch_sgg_mean_recall_results_json()
		self.generate_sgg_mean_recall_results_csvs_method_wise(sgg_mean_recall_results_json)
		
		sgg_recall_results_json = self.fetch_sgg_recall_results_json()
		self.generate_sgg_recall_results_csvs_method_wise(sgg_recall_results_json)


def main():
	prepare_paper_results_sgg = PreparePaperResultSGG()
	prepare_paper_results_sgg.compile_sgg_method_wise_results()


def combine_results():
	prepare_paper_results_sgg = PreparePaperResultSGG()
	prepare_paper_results_sgg.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\paper_sgg_results_csvs",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\paper_sgg_results_csvs\sgg_combined_results.xlsx"
	)


if __name__ == '__main__':
	# main()
	combine_results()