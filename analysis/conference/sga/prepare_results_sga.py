import csv
import os

from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value


class PrepareResultsSGA(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGA, self).__init__()
	
	def generate_context_results_csvs(self, context_results_json, context_fraction_list, train_ff_loss_list, modes,
	                                  methods):
		for cf in context_fraction_list:
			for mode in modes:
				csv_file_name = f"{mode}_{cf}.csv"
				csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs", "context_results_csvs",
				                             csv_file_name)
				os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
				with open(csv_file_path, "a", newline='') as csv_file:
					writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
					writer.writerow([
						"Anticipation Loss", "Method Name",
						"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
						"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
						"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50"
					])
					for train_ff in train_ff_loss_list:
						for method_name in methods:
							method_name = self.fetch_method_name_json(method_name)
							method_name_csv = self.fetch_method_name_csv(method_name)
							writer.writerow([
								train_ff,
								method_name_csv,
								context_results_json[cf][mode][train_ff][method_name][0]["R@10"],
								context_results_json[cf][mode][train_ff][method_name][0]["R@20"],
								context_results_json[cf][mode][train_ff][method_name][0]["R@50"],
								context_results_json[cf][mode][train_ff][method_name][0]["mR@10"],
								context_results_json[cf][mode][train_ff][method_name][0]["mR@20"],
								context_results_json[cf][mode][train_ff][method_name][0]["mR@50"],
								context_results_json[cf][mode][train_ff][method_name][0]["hR@10"],
								context_results_json[cf][mode][train_ff][method_name][0]["hR@20"],
								context_results_json[cf][mode][train_ff][method_name][0]["hR@50"],
								context_results_json[cf][mode][train_ff][method_name][1]["R@10"],
								context_results_json[cf][mode][train_ff][method_name][1]["R@20"],
								context_results_json[cf][mode][train_ff][method_name][1]["R@50"],
								context_results_json[cf][mode][train_ff][method_name][1]["mR@10"],
								context_results_json[cf][mode][train_ff][method_name][1]["mR@20"],
								context_results_json[cf][mode][train_ff][method_name][1]["mR@50"],
								context_results_json[cf][mode][train_ff][method_name][1]["hR@10"],
								context_results_json[cf][mode][train_ff][method_name][1]["hR@20"],
								context_results_json[cf][mode][train_ff][method_name][1]["hR@50"],
								context_results_json[cf][mode][train_ff][method_name][2]["R@10"],
								context_results_json[cf][mode][train_ff][method_name][2]["R@20"],
								context_results_json[cf][mode][train_ff][method_name][2]["R@50"],
								context_results_json[cf][mode][train_ff][method_name][2]["mR@10"],
								context_results_json[cf][mode][train_ff][method_name][2]["mR@20"],
								context_results_json[cf][mode][train_ff][method_name][2]["mR@50"],
								context_results_json[cf][mode][train_ff][method_name][2]["hR@10"],
								context_results_json[cf][mode][train_ff][method_name][2]["hR@20"],
								context_results_json[cf][mode][train_ff][method_name][2]["hR@50"]
							])
	
	def generate_complete_future_frame_results_csvs(self, ff_results_json, test_ff_list, train_ff_loss_list, modes, methods):
		for test_num_ff in test_ff_list:
			for mode in modes:
				csv_file_name = f"{mode}_{test_num_ff}.csv"
				csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
				                             "complete_test_future_results_csvs", csv_file_name)
				os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
				with open(csv_file_path, "a", newline='') as csv_file:
					writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
					writer.writerow([
						"Anticipation Loss", "Method Name",
						"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
						"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
						"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50"
					])
					for train_ff in train_ff_loss_list:
						for method_name in methods:
							method_name = self.fetch_method_name_json(method_name)
							method_name_csv = self.fetch_method_name_csv(method_name)
							writer.writerow([
								train_ff,
								method_name_csv,
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["R@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["R@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["R@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["mR@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["mR@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["mR@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["hR@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["hR@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][0]["hR@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["R@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["R@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["R@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["mR@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["mR@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["mR@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["hR@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["hR@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][1]["hR@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["R@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["R@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["R@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["mR@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["mR@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["mR@50"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["hR@10"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["hR@20"],
								ff_results_json[test_num_ff][mode][train_ff][method_name][2]["hR@50"]
							])
	
	def fill_combined_context_fraction_values_matrix(self, values_matrix, idx, method_name, context_results_json,
	                                                 context_fraction, mode, train_num_future_frame):
		method_name = self.fetch_method_name_json(method_name)
		values_matrix[idx, 0] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@10"])
		values_matrix[idx, 1] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@20"])
		values_matrix[idx, 2] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@50"])
		values_matrix[idx, 3] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@10"])
		values_matrix[idx, 4] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@20"])
		values_matrix[idx, 5] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@50"])
		values_matrix[idx, 6] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@10"])
		values_matrix[idx, 7] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@20"])
		values_matrix[idx, 8] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@50"])
		values_matrix[idx, 9] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@10"])
		values_matrix[idx, 10] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@20"])
		values_matrix[idx, 11] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@50"])
		return values_matrix
	
	def fill_combined_wn_context_fraction_values_matrix(self, values_matrix, idx, method_name, context_results_json,
	                                                    context_fraction, mode, train_num_future_frame):
		method_name = self.fetch_method_name_json(method_name)
		values_matrix[idx, 0] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@10"])
		values_matrix[idx, 1] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@20"])
		values_matrix[idx, 2] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@50"])
		values_matrix[idx, 3] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@10"])
		values_matrix[idx, 4] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@20"])
		values_matrix[idx, 5] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@50"])
		values_matrix[idx, 6] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@10"])
		values_matrix[idx, 7] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@20"])
		values_matrix[idx, 8] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@50"])
		values_matrix[idx, 9] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@10"])
		values_matrix[idx, 10] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@20"])
		values_matrix[idx, 11] = fetch_value(
			context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@50"])
		return values_matrix
	
	def fill_combined_future_frame_values_matrix(self, values_matrix, idx, method_name, context_results_json,
	                                             test_future_frame, mode, train_num_future_frame):
		method_name = self.fetch_method_name_json(method_name)
		values_matrix[idx, 0] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@10"])
		values_matrix[idx, 1] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@20"])
		values_matrix[idx, 2] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@50"])
		values_matrix[idx, 3] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@10"])
		values_matrix[idx, 4] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@20"])
		values_matrix[idx, 5] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@50"])
		values_matrix[idx, 6] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@10"])
		values_matrix[idx, 7] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@20"])
		values_matrix[idx, 8] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@50"])
		values_matrix[idx, 9] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@10"])
		values_matrix[idx, 10] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@20"])
		values_matrix[idx, 11] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@50"])
		return values_matrix
	
	def fill_combined_wn_future_frame_values_matrix(self, values_matrix, idx, method_name, context_results_json,
	                                                test_future_frame, mode, train_num_future_frame):
		method_name = self.fetch_method_name_json(method_name)
		values_matrix[idx, 0] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@10"])
		values_matrix[idx, 1] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@20"])
		values_matrix[idx, 2] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@50"])
		values_matrix[idx, 3] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@10"])
		values_matrix[idx, 4] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@20"])
		values_matrix[idx, 5] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@50"])
		values_matrix[idx, 6] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@10"])
		values_matrix[idx, 7] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@20"])
		values_matrix[idx, 8] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@50"])
		values_matrix[idx, 9] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@10"])
		values_matrix[idx, 10] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@20"])
		values_matrix[idx, 11] = fetch_value(
			context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@50"])
		return values_matrix