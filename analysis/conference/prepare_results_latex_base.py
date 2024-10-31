from analysis.conference.prepare_results_base import PrepareResultsBase


class PrepareResultsLatexBase(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsLatexBase, self).__init__()
	
	@staticmethod
	def fetch_setting_name(mode):
		if mode == "sgdet":
			setting_name = "\\textbf{sgdet}"
		elif mode == "sgcls":
			setting_name = "\\textbf{sgcls}"
		elif mode == "predcls":
			setting_name = "\\textbf{predcls}"
		return setting_name
	
	@staticmethod
	def fetch_method_name_latex(method_name):
		if method_name == "NeuralODE" or method_name == "ode":
			method_name = "\\textbf{SceneSayerODE (Ours)}"
		elif method_name == "NeuralSDE" or method_name == "sde":
			method_name = "\\textbf{SceneSayerSDE (Ours)}"
		elif method_name == "sttran_ant":
			method_name = "STTran+ \cite{cong_et_al_sttran_2021}"
		elif method_name == "sttran_gen_ant":
			method_name = "STTran++ \cite{cong_et_al_sttran_2021}"
		elif method_name == "dsgdetr_ant":
			method_name = "DSGDetr+ \cite{Feng_2021}"
		elif method_name == "dsgdetr_gen_ant":
			method_name = "DSGDetr++ \cite{Feng_2021}"
		elif method_name == "sde_wo_bb":
			method_name = "\\textbf{SceneSayerSDE (w/o BB)}"
		elif method_name == "sde_wo_recon":
			method_name = "\\textbf{SceneSayerSDE (w/o Recon)}"
		elif method_name == "sde_wo_gen":
			method_name = "\\textbf{SceneSayerSDE (w/o GenLoss)}"
		elif method_name == "sttran":
			method_name = "STTran \cite{cong_et_al_sttran_2021}"
		elif method_name == "dsgdetr":
			method_name = "DSGDetr \cite{Feng_2021}"
		elif method_name == "tempura":
			method_name = "Tempura \cite{tempura_2021}"
		return method_name
	
	def generate_combined_recalls_latex_header(self, setting_name, metric, train_horizon, mode, eval_horizon):
		tab_name = self.fetch_ref_tab_name(mode, eval_horizon, train_horizon)
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Results for " + setting_name + ", when trained using anticipatory horizon of " + train_horizon + " future frames.}\n"
		latex_header += "    \\label{tab:anticipation_results_" + tab_name + "_" + metric + "}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{ll|cccccc|cccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "        \\multicolumn{2}{c}{\\textbf{" + setting_name + "}} & \\multicolumn{6}{c}{\\textbf{With Constraint}} & \\multicolumn{6}{c}{\\textbf{No Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n "
		
		latex_header += (
			"        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & "
			"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
			"\\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & "
			"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}   \\\\ \\hline\n")
		return latex_header
	
	def generate_combined_wn_recalls_latex_header(self, setting_name, metric, train_horizon, mode, eval_horizon):
		tab_name = self.fetch_ref_tab_name(mode, eval_horizon, train_horizon)
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Results for " + setting_name + ", when trained using anticipatory horizon of " + train_horizon + " future frames.}\n"
		latex_header += "    \\label{tab:anticipation_results_" + tab_name + "_" + metric + "}\n"
		latex_header += "    \\setlength{\\tabcolsep}{5pt} \n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{ll|cccccc|cccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         & & \\multicolumn{6}{c|}{\\textbf{Recall (R)}} & \\multicolumn{6}{c}{\\textbf{Mean Recall (mR)}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n "
		latex_header += "        \\multicolumn{2}{c|}{\\textbf{" + setting_name + "}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c|}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}}\\\\ \n"
		latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \n "
		latex_header += ("        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{10} & \\textbf{20} & \\textbf{50} & "
		                 "\\textbf{10} & \\textbf{20} & \\textbf{50} & "
		                 "\\textbf{10} & \\textbf{20} & \\textbf{50}  & "
		                 "\\textbf{10} & \\textbf{20} & \\textbf{50}   \\\\ \\hline\n")
		return latex_header
	
	@staticmethod
	def generate_latex_footer():
		latex_footer = "    \\end{tabular}\n"
		latex_footer += "    }\n"
		latex_footer += "\\end{table}\n"
		return latex_footer
