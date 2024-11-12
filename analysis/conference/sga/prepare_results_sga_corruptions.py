from analysis.conference.prepare_results_base import PrepareResultsBase


class PrepareResultsSGA(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGA, self).__init__()
		self.scenario_list = ["full", "partial"]
		self.mode_list = ["sgcls", "predcls"]
		self.method_list = ["sttran_ant", "dsgdetr_ant", "sttran_gen_ant", "dsgdetr_gen_ant", "ode", "sde"]
		self.partial_percentages = [10]
		self.task_name = "sga"
		
		self.context_fraction_list = ['0.3', '0.5', '0.7', '0.9']