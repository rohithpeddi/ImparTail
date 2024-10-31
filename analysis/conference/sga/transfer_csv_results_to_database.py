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
def transfer_sga(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGA
):
	base_file_name = os.path.basename(result_file_path)
	details = base_file_name.split('_')
	train_num_future_frames = base_file_name.split('_')[2]
	
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
			
			
if __name__ == '__main__':
	db_service = FirebaseService()
	# transfer_results_from_directories_sgg()