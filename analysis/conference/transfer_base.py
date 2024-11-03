from analysis.results.Result import Metrics, ResultDetails


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
