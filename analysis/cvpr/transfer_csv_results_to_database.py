import csv
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Metrics, ResultDetails, Result
from constants import ResultConstants as const


def process_result_details_from_csv_row(row):
	method_name = row[0]
	print("Processing method: ", method_name)
	with_constraint_metrics = Metrics(
		row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]
	)
	no_constraint_metrics = Metrics(
		row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23],
		row[24]
	)
	semi_constraint_metrics = Metrics(
		row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35],
		row[36]
	)
	result_details = ResultDetails()
	result_details.add_with_constraint_metrics(with_constraint_metrics)
	result_details.add_no_constraint_metrics(no_constraint_metrics)
	result_details.add_semi_constraint_metrics(semi_constraint_metrics)
	return result_details, method_name

# Methods: sttran, dsgdetr
# Modes: sgcls, sgdet, predcls
# Partial Percentages: 10, 40, 70
# Label Noise Percentages: 10, 20, 30
# Scenario: partial, labelnoise, full
def transfer_sgg(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGG
):
	base_file_name = os.path.basename(result_file_path)
	details = base_file_name.split('_')
	if mode is None:
		mode = details[1]
	
	with open(result_file_path, 'r') as read_obj:
		# pass the file object to reader() to get the reader object
		csv_reader = csv.reader(read_obj)
		for row in csv_reader:
			result_details, method_name = process_result_details_from_csv_row(row)
			result = Result(
				task_name=task_name,
				scenario_name=scenario_name,
				method_name=method_name,
				mode=mode,
			)
			
			if scenario_name == const.PARTIAL:
				partial_percentage = 10
				result.partial_percentage = partial_percentage
			elif scenario_name == const.LABEL_NOISE:
				label_noise_percentage = 20
				result.label_noise_percentage = label_noise_percentage
			
			result.add_result_details(result_details)
			print("-----------------------------------------------------------------------------------")
			print("Saving result: ", result.result_id)
			db_service.update_result_to_db("results_29_10", result.result_id, result.to_dict())
			print("Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_results_from_directories():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	results_parent_directory = r"E:\results\legal"
	for task_name in os.listdir(results_parent_directory):
		print("##################################################################################")
		print(f"[{task_name}] Processing files for task: ", task_name)
		task_directory_path = os.path.join(results_parent_directory, task_name)
		for scenario_name in os.listdir(task_directory_path):
			print(f"[{task_name}][{scenario_name}] Processing files for scenario: ", scenario_name)
			scenario_name_path = os.path.join(task_directory_path, scenario_name)
			for mode_name in os.listdir(scenario_name_path):
				print("********************************************************************************")
				print(f"[{task_name}][{scenario_name}][{mode_name}] Processing files for mode: ", mode_name)
				mode_name_path = os.path.join(scenario_name_path, mode_name)
				for method_name_csv_file in os.listdir(mode_name_path):
					method_name_csv_path = os.path.join(mode_name_path, method_name_csv_file)
					if task_name == const.SGG and scenario_name in [const.PARTIAL, const.LABEL_NOISE, const.FULL]:
						print(
							f"[{task_name}][{scenario_name}][{mode_name}][{method_name_csv_file[:-4]}] Processing file: ",
							method_name_csv_path)
						transfer_sgg(
							mode=mode_name,
							result_file_path=method_name_csv_path,
							scenario_name=scenario_name
						)


if __name__ == '__main__':
	db_service = FirebaseService()
	transfer_results_from_directories()
