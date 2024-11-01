import csv
import os

from analysis.conference.transfer_base import process_result_details_from_csv_row
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


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
	details = (base_file_name.split('.')[0]).split('_')
	
	assert mode == details[1]
	if mode is None:
		mode = details[1]
	
	assert mode in [const.SGCLS, const.SGDET, const.PREDCLS]
	
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
			elif scenario_name == const.LABELNOISE:
				label_noise_percentage = 20
				result.label_noise_percentage = label_noise_percentage
			
			result.add_result_details(result_details)
			print("-----------------------------------------------------------------------------------")
			print("Saving result: ", result.result_id)
			db_service.update_result_to_db("results_31_10_sgg", result.result_id, result.to_dict())
			print("Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_results_from_directories_sgg():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	task_directory_path = r"E:\results\legal\sgg"
	task_name = const.SGG
	for scenario_name in os.listdir(task_directory_path):
		# Convert the scenario name to lowercase
		scenario_name = scenario_name.lower()
		print(f"[{task_name}][{scenario_name}] Processing files for scenario: ", scenario_name)
		scenario_name_path = os.path.join(task_directory_path, scenario_name)
		for method_name_csv_file in os.listdir(scenario_name_path):
			# Convert the method name to lowercase
			method_name_csv_file = method_name_csv_file.lower()
			method_name_csv_path = os.path.join(scenario_name_path, method_name_csv_file)
			
			mode_name = (method_name_csv_file.split('.')[0]).split('_')[1]
			assert mode_name in [const.SGCLS, const.SGDET, const.PREDCLS]
			if task_name == const.SGG and scenario_name in [const.PARTIAL, const.LABELNOISE, const.FULL]:
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
	transfer_results_from_directories_sgg()
