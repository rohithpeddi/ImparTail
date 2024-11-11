import csv
import os

from analysis.conference.transfer_base import process_result_details_from_csv_row
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


# Methods: sttran_ant, sttran_gen_ant, dsgdetr_ant, dsgdetr_gen_ant, ode, sde
# Modes: sgcls, sgdet, predcls
# Partial Percentages: 10, 40, 70
# Label Noise Percentages: 10, 20, 30
# Scenario: partial, labelnoise, full
def transfer_sga(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGA
):
	base_file_name = os.path.basename(result_file_path)
	# As context fraction has "." in it so we cannot split by "."
	eval_csv_file_name = base_file_name[:-4]
	csv_attributes = eval_csv_file_name.split("_")
	
	# Test future frames
	index_of_test = csv_attributes.index("test")
	train_future_frames = csv_attributes[index_of_test - 1]
	test_num = csv_attributes[index_of_test + 1]
	
	# If "." is present in the test future frames then it is context fraction else it is test future frames
	if "." in test_num:
		test_context_fraction = test_num
		test_future_frames = None
	else:
		test_context_fraction = None
		test_future_frames = test_num
	
	if not os.path.exists(result_file_path):
		print("File does not exist: ", result_file_path)
		return
	
	with open(result_file_path, 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		num_rows = len(list(csv_reader))
	
	with open(result_file_path, 'r') as read_obj:
		# pass the file object to reader() to get the reader object
		csv_reader = csv.reader(read_obj)
		for row_id, row in enumerate(csv_reader):
			if len(row) == 0:
				print(f"[{task_name}][{scenario_name}][{mode}] Skipping empty row")
				continue
			
			# As the first row here corresponds to the full annotations and the second corresponds to the partial annotations
			if scenario_name == const.PARTIAL and row_id == 0 and num_rows > 1:
				print(f"[{task_name}][{scenario_name}][{mode}] Skipping header row")
				continue
			
			result_details, method_name = process_result_details_from_csv_row(row)
			result = Result(
				task_name=task_name,
				scenario_name=scenario_name,
				method_name=method_name,
				mode=mode,
			)
			if scenario_name == const.PARTIAL:
				partial_percentage = 40
				result.partial_percentage = partial_percentage
			elif scenario_name == const.LABELNOISE:
				label_noise_percentage = 20
				result.label_noise_percentage = label_noise_percentage
			
			result.train_num_future_frames = train_future_frames
			result.test_num_future_frames = test_future_frames
			result.context_fraction = test_context_fraction
			
			result.add_result_details(result_details)
			print(f"[{task_name}][{scenario_name}][{mode}] Saving result: ", result.result_id)
			db_service.update_result_to_db("results_11_11_sga", result.result_id, result.to_dict())
			print(f"[{task_name}][{scenario_name}][{mode}] Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_results_from_directories_sga():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	task_directory_path = r"E:\results\legal\sga"
	task_name = const.SGA
	for scenario_name in os.listdir(task_directory_path):
		scenario_name_path = os.path.join(task_directory_path, scenario_name)
		# Convert the scenario name to lowercase
		scenario_name = scenario_name.lower()
		print(f"[{task_name}][{scenario_name}] Processing files for scenario: ", scenario_name)
		for mode_name in os.listdir(scenario_name_path):
			# Convert the mode name to lowercase
			mode_name_path = os.path.join(scenario_name_path, mode_name)
			mode_name = mode_name.lower()
			for method_name_csv_file in os.listdir(mode_name_path):
				# Convert the method name to lowercase
				method_name_csv_path = os.path.join(mode_name_path, method_name_csv_file)
				if task_name == const.SGA and scenario_name in [const.PARTIAL, const.FULL]:
					print(f"[{task_name}][{scenario_name}][{mode_name}][{method_name_csv_file[:-4]}] Processing file: ",
					      method_name_csv_path)
					transfer_sga(
						mode=mode_name,
						result_file_path=method_name_csv_path,
						scenario_name=scenario_name
					)


if __name__ == '__main__':
	db_service = FirebaseService()
	transfer_results_from_directories_sga()
