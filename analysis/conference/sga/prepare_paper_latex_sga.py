from analysis.conference.prepare_results_latex_base import PrepareResultsLatexBase


class PrepareLatexSGA(PrepareResultsLatexBase):
	
	def __init__(self):
		super(PrepareLatexSGA, self).__init__()
	
	# TODO: Correct this.
	@staticmethod
	def generate_sga_paper_latex_header():
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Results for SGG.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{ll|ccccccccc|ccccccccc|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "       &  & \\multicolumn{9}{c}{\\textbf{SGDET}} & \\multicolumn{9}{c}{\\textbf{SGCLS}} & \\multicolumn{9}{c}{\\textbf{PREDCLS}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){2-10} \\cmidrule(lr){11-19} \\cmidrule(lr){20-28} \n "
		latex_header += "       &  & \\multicolumn{3}{c}{\\textbf{With}} & \\multicolumn{3}{c}{\\textbf{No}} & \\multicolumn{3}{c}{\\textbf{Semi}} & \\multicolumn{3}{c}{\\textbf{With}} & \\multicolumn{3}{c}{\\textbf{No}} & \\multicolumn{3}{c}{\\textbf{Semi}} & \\multicolumn{3}{c}{\\textbf{With}} & \\multicolumn{3}{c}{\\textbf{No}} & \\multicolumn{3}{c}{\\textbf{Semi}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1} \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \\cmidrule(lr){17-19} \\cmidrule(lr){20-22} \\cmidrule(lr){23-25} \\cmidrule(lr){26-28} \n "
		
		latex_header += (
				"        \\textbf{Method} & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50} & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & " + " \\\\ \\hline\n")
		return latex_header