import argparse

from prepare_results_base import *
from constants import CorruptionConstants as const


def compile_image_based_corruption_results(self):
	rob_sgg_results = self.fetch_rob_sgg_results()
	image_based_corruption_results_json = {}
	for corruption_name in image_based_corruptions:
		image_based_corruption_results_json[corruption_name] = {}
		for mode in modes:
			image_based_corruption_results_json[corruption_name][mode] = {}
			for method_name in methods:
				method_name = fetch_method_name_json(method_name)
				image_based_corruption_results_json[corruption_name][mode][method_name] = {}
				for severity_level in severity_levels:
					image_based_corruption_results_json[corruption_name][mode][method_name][
						severity_level] = fetch_empty_metrics_json()
	
	image_based_corruption_rob_sgg_list = []
	for result in rob_sgg_results:
		corruption_name, severity_level = fetch_corruption_name(result.dataset_corruption_type)
		print(f"Corruption Name: {corruption_name}, Severity Level: {severity_level}")
		if corruption_name in image_based_corruptions:
			image_based_corruption_rob_sgg_list.append(result)
	
	for result in image_based_corruption_rob_sgg_list:
		mode = result.mode
		method_name = fetch_method_name_json(result.method_name)
		corruption_name, severity_level = fetch_corruption_name(result.dataset_corruption_type)
		severity_level = str(severity_level)
		
		with_constraint_metrics = result.result_details.with_constraint_metrics
		no_constraint_metrics = result.result_details.no_constraint_metrics
		semi_constraint_metrics = result.result_details.semi_constraint_metrics
		
		completed_metrics_json = fetch_completed_metrics_json(
			with_constraint_metrics,
			no_constraint_metrics,
			semi_constraint_metrics
		)
		
		if method_name in ["DSGDetr", "STTran"]:
			image_based_corruption_results_json[corruption_name][mode][method_name][
				severity_level] = completed_metrics_json
	
	return image_based_corruption_results_json


# --------------------------------------------------------------------------------------------
# RESULTS IN PAPER
# --------------------------------------------------------------------------------------------


def generate_recall_results_csvs(image_based_corruption_results_json):
	for mode in modes:
		csv_file_name = f"recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../results_docs", "mode_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Corruption Name", "Method Name", "Severity Level", "R@10", "R@20", "R@50", "R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100"
			])
			for corruption_name in image_based_corruptions:
				for method_name in methods:
					method_name = fetch_method_name_json(method_name)
					for severity_level in severity_levels:
						writer.writerow([
							corruption_name,
							method_name,
							severity_level,
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"R@10"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"R@20"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"R@50"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"R@100"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"R@10"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"R@20"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"R@50"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"R@100"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"R@10"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"R@20"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"R@50"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"R@100"]
						])


def generate_mean_recall_results_csvs(image_based_corruption_results_json):
	for mode in modes:
		csv_file_name = f"mean_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../results_docs", "mode_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Corruption Name", "Method Name", "Severity Level", "mR@10", "mR@20", "mR@50", "mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100"
			])
			for corruption_name in image_based_corruptions:
				for method_name in methods:
					method_name = fetch_method_name_json(method_name)
					for severity_level in severity_levels:
						writer.writerow([
							corruption_name,
							method_name,
							severity_level,
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"mR@10"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"mR@20"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"mR@50"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][0][
								"mR@100"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"mR@10"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"mR@20"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"mR@50"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][1][
								"mR@100"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"mR@10"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"mR@20"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"mR@50"],
							image_based_corruption_results_json[corruption_name][mode][method_name][severity_level][2][
								"mR@100"]
						])


def compile_image_corruption_results():
	image_based_corruption_results_json = compile_image_based_corruption_results()
	
	generate_recall_results_csvs(image_based_corruption_results_json)
	generate_mean_recall_results_csvs(image_based_corruption_results_json)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder_path', type=str)
	parser.add_argument('-result_file_path', type=str)
	
	# modes = ["sgdet", "sgcls", "predcls"]
	# methods = ["sttran", "dsgdetr", "tempura"]
	# severity_levels = ["1", "2", "3", "4", "5"]
	
	modes = ["sgdet", "sgcls", "predcls"]
	methods = ["sttran", "dsgdetr"]
	severity_levels = ["1", "3", "5"]
	
	image_based_corruptions = [
		const.NO_CORRUPTION, const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
		const.GAUSSIAN_BLUR, const.GLASS_BLUR, const.DEFOCUS_BLUR, const.MOTION_BLUR, const.ZOOM_BLUR, const.FOG,
		const.FROST, const.SNOW, const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.ELASTIC_TRANSFORM,
		const.PIXELATE, const.JPEG_COMPRESSION, const.SUN_GLARE, const.RAIN, const.DUST, const.WILDFIRE_SMOKE,
		const.SATURATE, "mixed_fixed", "mixed_mixed"
	]
	
	args = parser.parse_args()
	db_service = FirebaseService()
	compile_image_corruption_results()
