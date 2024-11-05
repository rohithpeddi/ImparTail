import json
import os
import random

import torch
from tqdm import tqdm
from transformers import pipeline


class MultiAgentVideoCapSummarizer:
	
	def __init__(self):
		self.pipe = None
		self.model_id = None
		self.captions = None

		self.captions_dir = "/data/rohith/ag/captions/summarized/"
		os.makedirs(self.captions_dir, exist_ok=True)

		with open("multi_agent_cap.json", "r") as f:
			self.captions = json.load(f)

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

	def construct_prompts(self, video_id):
		# Construct the prompt
		system_prompt = f'''
				In this task, you are provided with video captions from three different sources, each describing the same video. 
				Your objective is to synthesize a concise video summary. 
				Focus on sequencing the events chronologically by paying attention to temporal indicators like "then," "while," "before," and "after." 
				Incorporate spatial and interaction relationships such as [looking at, not looking at, unsure, above, beneath, in front of, behind, on the side of, in, carrying, covered by, drinking from, eating, have it on the back, holding, leaning on, lying on, not contacting, other relationship, sitting on, standing on, touching, twisting, wearing, wiping, writing on] where relevant. 
				Avoid details about colors and attire; focus solely on actions, object categories, and their interrelations.
				This approach should yield a concise action- and relation-focused summary that aids in understanding the dynamics depicted in the video.
				Only include the summary in your response.
				'''
		# input_content = f'''
		# 		Input:
		# 		Source Caption 1: The video shows a man cleaning the floor with a mop. He starts by putting the mop in a bucket, then proceeds to clean the floor. After finishing, he puts the mop back in the bucket. In the first scene, the man is seen putting the mop in the bucket, indicating that he is preparing to clean the floor. The second scene shows him cleaning the floor, using the mop to remove dirt and debris. The final scene shows him putting the mop back in the bucket, indicating that he has finished cleaning. The video provides a simple yet satisfying visual of a man completing a task, and the use of the mop as a tool to clean the floor. The repetitive actions of the man, such as putting the mop in the bucket and cleaning the floor, add to the sense of satisfaction and completion. Overall, the video is a simple yet effective representation of a mundane task being completed with care and attention to detail.
		# 		Source Caption 2: The video captures a sequence of events in a domestic setting, focusing on a person engaged in cleaning activities. Initially, the individual is seen bending over a dining table, seemingly picking up or organizing items on the table. The room is well-lit, with a domestic environment featuring a dining table, chairs, a cabinet with various items, and a wall adorned with decorative elements including a tree decal and a framed picture. A black bucket and a broom are also visible, indicating cleaning activities. As the video progresses, the person's actions evolve from bending over the table to standing and engaging in different cleaning tasks. They are observed picking up an orange object from the table, then proceeding to clean the floor with a broom. The individual's movements are methodical, shifting from one cleaning task to another, including sweeping the floor and later, cleaning the baseboard and floor near a doorway. The person's attire, a blue and white striped shirt and dark pants, remains consistent throughout the video, as does the domestic setting.\n\nThe camera's perspective shifts slightly throughout the video, at times focusing more closely on the person's actions, particularly when they are bending over or cleaning the floor. Despite these changes in perspective, the overall setting remains unchanged, with the room's layout and the positioning of objects like the cabinet, table, and the black bucket staying constant. Towards the end of the video, the individual's activities continue to revolve around cleaning, with the final scenes showing them engaging in tasks such as sweeping the floor and interacting with the black bucket, possibly disposing of debris or cleaning materials. The video concludes with the person standing upright, facing the camera, suggesting a pause or completion of their cleaning task. Throughout the video, the camera maintains a steady focus on the individual's actions, with no significant movement or change in the environment, allowing for a clear observation of the cleaning process within a domestic setting.
		# 		Source Caption 3: A person dusts off some shelves furniture, then sweeps under a doorway.;A person is tidying up the living room;A person was dusting furniture with a cloth. The person then picked up a broom and began to sweep the floor. The person then put the broom on the floor and picked up a trash bin.
		#
		# 		Output: The video shows a person tidying up a domestic setting. Initially, the person bends over a dining table, picking up or organizing items on the table, including an orange object. The person dusts off shelves and furniture with a cloth. Next, the person picks up a broom and begins to sweep the floor, including sweeping under a doorway and cleaning the baseboard near a doorway. After sweeping, the person puts the broom on the floor and picks up a black bucket, possibly to dispose of debris or cleaning materials. The person then prepares to mop the floor by putting the mop in the bucket. Proceeding to clean the floor with the mop, the person uses it to remove dirt and debris. After finishing mopping, the person puts the mop back in the bucket, indicating that the cleaning task is complete. The video concludes with the person standing upright, facing the camera.
		#
		# 		Input:
		# 		Source Caption 1: The video starts with a man walking down a hallway, and he stops to adjust his shoes. The camera then follows him as he enters a bedroom and sits on a bed. He puts on his shoes and leaves the room.\n\nAs the man exits the bedroom, the camera pans to the right, revealing a woman sitting on the bed. She looks up as the man walks by. The camera then follows the man down the hallway again, and he enters a room with a couch and a TV. He sits down on the couch and puts his feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the man on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.
		# 		Source Caption 2: The video depicts a sequence of events in a bedroom, starting with a person lying on a bed with a checkered black and white duvet cover, wearing a black top and blue jeans. Initially, the person is seen lying down, then sitting up, and eventually standing up, indicating a progression from rest to activity. The person's actions include reaching for and interacting with a laptop on the bed, suggesting they are preparing to use it. Throughout these actions, the person's attire changes from a black top and blue jeans to a black jacket and blue jeans, and then to a black jacket with a neon green zipper and blue jeans, indicating a progression in their attire.\n\nThe person's interactions with the laptop evolve from reaching for it to using it, and then to a moment of rest or contemplation, as they lean forward with their head resting on their hands. The person then stands up, indicating a shift from a seated to a standing position. The video captures the person's movements from the bed to standing, with the camera following their actions, maintaining a consistent focus on the individual and their immediate surroundings.\n\nThe person's journey from the bedroom to another room is depicted through a series of movements, including standing up, walking away from the bed, and exiting the bedroom. The camera follows the person's movement, capturing their transition from one room to another. The person's attire changes again, this time to a black jacket with a neon green zipper and blue jeans, suggesting a progression in their attire or a change in the context of their activity.\n\nThroughout the video, the camera's perspective shifts to follow the person's movements, capturing their transition from the bedroom to another room. The person's actions, from interacting with the laptop to standing up and walking away, are captured in a continuous sequence, highlighting the person's progression from rest to activity and movement through the space.
		# 		Source Caption 3: A person lounges on a bed with a laptop. this person gets up, puts on their shoes, and then walks out of the room.;A person lying down on a bed while looking at a laptop then stands up and puts on shoes and then exits the bedroom.
		#
		# 		Output: The video begins with a person walking down a hallway; the person stops to adjust their shoes. Then, the person enters a bedroom and sits on a bed with a checkered black and white duvet cover. Wearing a black top and blue jeans, the person begins using a laptop on the bed. After spending some time on the laptop, the person sits up, stands up, and puts on their shoes. The person changes into a black jacket and blue jeans, and then into a black jacket with a neon green zipper and blue jeans. Afterward, the person exits the bedroom—leaving the laptop on the bed—and walks down the hallway. The person then enters a room with a couch and a TV, sits down on the couch, and puts their feet up on the coffee table.
		#
		# 		Input:
		# 		Source Caption 1: {self.captions[video_id]["chatuniv"]}.
		# 		Source Caption 2: {self.captions[video_id]["longvu"]}.
		# 		Source Caption 3: {self.captions[video_id]["charades"]}.
		# 		'''

		# input_content = f'''
		# 				Input:
		# 				Source Caption 1: The video starts with a man walking down a hallway, and he stops to adjust his shoes. The camera then follows him as he enters a bedroom and sits on a bed. He puts on his shoes and leaves the room.\n\nAs the man exits the bedroom, the camera pans to the right, revealing a woman sitting on the bed. She looks up as the man walks by. The camera then follows the man down the hallway again, and he enters a room with a couch and a TV. He sits down on the couch and puts his feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the man on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.\n\nThe camera then pans to the left, revealing a man standing in the doorway. He looks at the woman on the couch and then walks away. The camera follows the man down the hallway again, and he enters a room with a bed. He sits down on the bed and puts his feet up on the bed.\n\nThe camera then pans to the right, revealing a woman standing in the doorway. She looks at the man on the bed and then walks away. The camera follows the woman down the hallway, and she enters a room with a couch and a TV. She sits down on the couch and puts her feet up on the coffee table.
		# 				Source Caption 2: The video depicts a sequence of events in a bedroom, starting with a person lying on a bed with a checkered black and white duvet cover, wearing a black top and blue jeans. Initially, the person is seen lying down, then sitting up, and eventually standing up, indicating a progression from rest to activity. The person's actions include reaching for and interacting with a laptop on the bed, suggesting they are preparing to use it. Throughout these actions, the person's attire changes from a black top and blue jeans to a black jacket and blue jeans, and then to a black jacket with a neon green zipper and blue jeans, indicating a progression in their attire.\n\nThe person's interactions with the laptop evolve from reaching for it to using it, and then to a moment of rest or contemplation, as they lean forward with their head resting on their hands. The person then stands up, indicating a shift from a seated to a standing position. The video captures the person's movements from the bed to standing, with the camera following their actions, maintaining a consistent focus on the individual and their immediate surroundings.\n\nThe person's journey from the bedroom to another room is depicted through a series of movements, including standing up, walking away from the bed, and exiting the bedroom. The camera follows the person's movement, capturing their transition from one room to another. The person's attire changes again, this time to a black jacket with a neon green zipper and blue jeans, suggesting a progression in their attire or a change in the context of their activity.\n\nThroughout the video, the camera's perspective shifts to follow the person's movements, capturing their transition from the bedroom to another room. The person's actions, from interacting with the laptop to standing up and walking away, are captured in a continuous sequence, highlighting the person's progression from rest to activity and movement through the space.
		# 				Source Caption 3: A person lounges on a bed with a laptop. this person gets up, puts on their shoes, and then walks out of the room.;A person lying down on a bed while looking at a laptop then stands up and puts on shoes and then exits the bedroom.
		#
		# 				Output: The video begins with a person walking down a hallway; the person stops to adjust their shoes. Then, the person enters a bedroom and sits on a bed with a checkered black and white duvet cover. Wearing a black top and blue jeans, the person begins using a laptop on the bed. After spending some time on the laptop, the person sits up, stands up, and puts on their shoes. The person changes into a black jacket and blue jeans, and then into a black jacket with a neon green zipper and blue jeans. Afterward, the person exits the bedroom—leaving the laptop on the bed—and walks down the hallway. The person then enters a room with a couch and a TV, sits down on the couch, and puts their feet up on the coffee table.
		#
		#
		# 				Source Caption 1: {self.captions[video_id]["chatuniv"]}.
		# 				Source Caption 2: {self.captions[video_id]["longvu"]}.
		# 				Source Caption 3: {self.captions[video_id]["charades"]}.
		# 				'''

		input_content = f'''
				Source Caption 1: {self.captions[video_id]["chatuniv"]}.
				Source Caption 2: {self.captions[video_id]["longvu"]}.
				Source Caption 3: {self.captions[video_id]["charades"]}.
				'''

		return system_prompt, input_content

	def construct_video_summary(self, video_id):
		system_prompt, content_prompt = self.construct_prompts(video_id)
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": content_prompt},
		]
		outputs = self.pipe(
			messages,
			max_new_tokens=512,
		)

		summary = outputs[0]["generated_text"][-1]["content"]
		summarized_video_file_path = os.path.join(self.captions_dir, f"{video_id}.txt")
		with open(summarized_video_file_path, "w") as f:
			f.write(summary)

	def generate_video_captions(self):
		video_id_list = list(self.captions.keys())
		random.shuffle(video_id_list)
		self.init_llama_3_2()
		for video_id in tqdm(video_id_list):

			if os.path.exists(os.path.join(self.captions_dir, f"{video_id}.txt")):
				print(f"Skipping {video_id}")
				continue

			self.construct_video_summary(video_id)
			
			
def main():
	multi_agent_video_cap_summarizer = MultiAgentVideoCapSummarizer()
	multi_agent_video_cap_summarizer.generate_video_captions()
	

if __name__ == "__main__":
	main()
