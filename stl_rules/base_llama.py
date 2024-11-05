import torch
from transformers import pipeline


class BaseInstructLLama:
	
	def __init__(self):
		self.pipe = None
		self.model_id = None
		self.captions = None
	
	def init_llama_3_1(self):
		self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

		self.pipe = pipeline(
			"text-generation",
			model=self.model_id,
			model_kwargs={"torch_dtype": torch.bfloat16},
			device_map="auto"
		)

	def init_llama_3_2(self):
		self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
		self.pipe = pipeline(
			"text-generation",
			model=self.model_id,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)