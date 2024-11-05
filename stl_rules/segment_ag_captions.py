import json
import os
import random

from tqdm import tqdm

from stl_rules.base_llama import BaseInstructLLama


class CaptionSegmenter(BaseInstructLLama):
	
	def __init__(self):
		super(BaseInstructLLama).__init__()
		
		self.captions_dir = "/data/rohith/ag/captions/segmented/"
		os.makedirs(self.captions_dir, exist_ok=True)

		summary_caption_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_captions.json")

		with open(summary_caption_path, "r") as f:
			self.captions = json.load(f)
	
	def construct_prompts(self, video_id):
		# Remove any new lines in the text file and combine the text into a single paragraph
		caption = self.captions[video_id].replace("\n", " ")
		stripped_caption = caption.strip()
		
		system_prompt = f'''
		In this task, you are given a video caption describing a video.
		Considering the words that indicate the order of events (e.g., then, while, before, and after),
		your job is to split multiple compositional sentences from the given video caption and list them in chronological order.
		Note that you should specify the objects for the pronouns used in each of these sentences.
		'''
		
		input_content = f'''
		Following are a few examples of video captions and their segmented sentences:
		Input: The person is turning on the stove. They then begin to stir some food and after that they pick up a camera and look at it.
        Output: The person is turning on the stove. >> The person stirs some food. >> The person picks up a camera. >> The person looks at a camera.
        Input: A person is sitting in bed texting on a phone while holding a blanket. The person puts the phone down and pulls the blanket up.
        Output: A person is sitting in a bed and texting on a phone while holding a blanket. >> The person puts the phone down. >> The person pulls the blanket up.
        Input: A person picks up a phone and enters the bathroom through a doorway while talking on the phone. The person puts on shoes and picks up clothes while laughing and dresses before walking out of the room.
        Output: A person picks up a phone. >> A person enters the bathroom through a doorway while talking on the phone. >> The person puts on shoes >> The person picks up clothes while laughing >> The person dresses clothes >> The person walks out of the room.
        Input: A person is sitting on a toilet, picks up a phone and battery that are on the ground, puts the battery into the phone, takes off a jacket, then stands and takes selfies against the bathroom door.
        Output: A person is sitting on a toilet. >> A person picks up a phone and battery that are on the ground. >> A person takes off a jacket. >> A person stands and takes selfies against the bathroom door.
        Input: A person is undressing, picks up a towel and cleans some glasses before taking a drink.
        Output: A person is undressing. >> A person picks up a towel. >> A person cleans some glasses. >> A person takes a drink some glasses.
        Input: Person pulls out phone and begins playing with it then sets it down and pulls the blanket further up.
        Output: Person pulls out phone. >> Person plays with the phone. >> Person sets the phone down. >> Person pulls the blanket further up.
        Input: A person watching television and eating a sandwich while laying on the floor and reading book,after a while the person gets up to grab a box.
        Output: A person watches television and eats a sandwich while laying on the floor. >> A person reads a book. >> A person gets up to grab a box.
        Input: A person walks to a pantry, takes out some clothes from it, tosses one on the floor, and puts on another after taking it off again.
        Output: A person walks to a pantry. >> A person takes out some clothes from a pantry. >> A person tosses a cloth on the floor. >> A person puts on a cloth. >> A person takes a cloth off.
		Input: A man enters a house with a boy, where the man vacuums the floor while the boy jumps on it, displaying a playful interaction. The man then stops and puts the vacuum away, allowing the boy to continue jumping. The boy enjoys himself, while the man focuses on cleaning the house. In contrast, a person in a dimly lit corridor is shown picking up and folding various garments, maintaining a consistent perspective on the scene. The individual's movements are focused on handling and organizing the garments, with a sense of purpose and deliberateness. Later, a person is seen picking up clothes from the floor and throwing them on the floor of another room, displaying a different interaction.
		Output: A man enters a house with a boy. >> The man vacuums the floor while the boy jumps on it. >> The man stops vacuuming and puts the vacuum away. >> The boy continues jumping. >> The man focuses on cleaning the house. >> A person is shown picking up and folding various garments in a dimly lit corridor. >> The person picks up clothes from the floor. >> The person throws the clothes on the floor of another room.
		Input: The video begins with a person walking into the frame and sitting down on a bed. They then put on a pair of shoes, indicating a change in activity. The person appears to be in a bedroom setting, surrounded by the basic necessities of a room. As they stand up, the person walks towards a door and enters the room, maintaining a stationary camera perspective. Upon entering, the person moves towards a closed door, then turns around and exits the room, returning to the same location from where they started. The person then re-enters the room and sits in a chair, putting keys on a table before exiting the room again.
		Output: The video begins with a person walking into the frame. >> A person sits down on a bed. >> A person puts on a pair of shoes. >> A person stands up. >> A person walks towards a door and enters the room. >> A person moves towards a closed door. >> A person turns around and exits the room. >> A person re-enters the room. >> A person sits in a chair. >> A person puts keys on a table. >> A person exits the room again.
		Input: The video shows a person cleaning the kitchen floor. The person first picks up a mop and puts it in a bucket, then begins to mop the floor in a back-and-forth motion, thoroughly cleaning the entire area. Before starting, the person is not seen in the kitchen, but in the living room, where they are holding a broom and dustpan. The person then moves towards the kitchen, bends over a cabinet to reach into it, and stands up to engage with the kitchen counter. They are then seen interacting with the kitchen counter, possibly cooking or preparing food. The person's actions suggest a sequence of domestic tasks, including cleaning and cooking. The video captures a simple yet satisfying task of cleaning the kitchen floor, highlighting the importance of maintaining cleanliness in our daily lives.
		Output: The person is holding a broom and dustpan in the living room. >> The person walks towards the kitchen. >> The person bends over a cabinet to reach into it. >> The person stands up to engage with the kitchen counter. >> The person interacts with the kitchen counter, possibly cooking or preparing food. >> The person picks up a mop. >> The person puts the mop in a bucket. >> The person begins to mop the floor in a back-and-forth motion, thoroughly cleaning the kitchen floor.
		
		Follow the same format to segment the video caption for the given video input and make sure to include only the output response. 
		Input: {stripped_caption}.
		'''
		
		return system_prompt, input_content
	
	def segment_video_captions(self, video_id):
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
	
	def generate_video_segments(self):
		video_id_list = list(self.captions.keys())
		random.shuffle(video_id_list)
		self.init_llama_3_2()
		for video_id in tqdm(video_id_list):
			
			if os.path.exists(os.path.join(self.captions_dir, f"{video_id}.txt")):
				print(f"Skipping {video_id}")
				continue
			
			self.segment_video_captions(video_id)


def main():
	caption_segmenter = CaptionSegmenter()
	caption_segmenter.generate_video_segments()


def compile_captions():
	captions = {}
	captions_dir = "/data/rohith/ag/captions/segmented/"
	for file in os.listdir(captions_dir):
		video_id = file.split(".")[0]
		with open(os.path.join(captions_dir, file), "r") as f:
			caption = f.read()
			captions[video_id] = caption
	with open("segmented_captions.json", "w") as f:
		json.dump(captions, f)


if __name__ == "__main__":
	main()
# compile_captions()
