import torch
from transformers import pipeline
from transformers import MllamaForConditionalGeneration, AutoProcessor

class BaseInstructLLama:
	
	def __init__(self):
		self.processor = None
		self.model = None
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
		
	def init_llama_3_2_vision_11B(self):
		self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
		
		self.model = MllamaForConditionalGeneration.from_pretrained(
			self.model_id,
			torch_dtype=torch.bfloat16,
			device_map="auto",
		)
		self.processor = AutoProcessor.from_pretrained(self.model_id)