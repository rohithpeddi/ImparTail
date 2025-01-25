import json
import os
import random

from tqdm import tqdm

from lib.stl.vlm.base_llama import BaseInstructLLama


class GenerateClauses(BaseInstructLLama):

    def __init__(self):
        super(BaseInstructLLama).__init__()

        self.captions_dir = "/data/rohith/ag/rules/backward_dependency/"
        os.makedirs(self.captions_dir, exist_ok=True)

        segmented_caption_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/captions/segmented_captions.json")

        with open(segmented_caption_path, "r") as f:
            self.captions = json.load(f)

    def construct_backward_dependency_prompts(self, video_id):
        # Remove any new lines in the text file and combine the text into a single paragraph
        caption = self.captions[video_id].replace("\n", " ")
        stripped_caption = caption.strip()

        system_prompt = f'''
		In this task, you are given a video caption describing a video of a person performing a series of actions. These actions are in the form of sentences separated by >> symbol.
        Considering backward dependency relationships between the various actions of the person, which includes but is not limited to actions which require specific actions to happen before them, your job is to create a set of Signal Temporal Logic (STL) formulas that describe the backward dependency relationships between the Person's actions in the video.        
        The set of objects which you can use in the STL formulas is strictly limited to the following: person, bag, bed, blanket, book, box, broom, chair, closet/cabinet, clothes, cup/glass/bottle, dish, door, doorknob, doorway, floor, food, groceries, laptop, light, medicine, mirror, paper/notebook, phone/camera, picture, pillow, refrigerator, sandwich, shelf, shoe, sofa/couch, table, television, towel, vacuum, window.        
        The set of predicates which you can use in the STL formulas is strictly limited to the following:  looking_at, not_looking_at, unsure, above, beneath, in_front_of, behind, on_the_side_of, in, carrying, covered_by, drinking_from, eating, have_it_on_the_back, holding, leaning_on, lying_on, not_contacting, other_relationship, sitting_on, standing_on, touching, twisting, wearing, wiping, writing_on.       
        Note that you should specify the objects for the pronouns used in each sentence.
		'''

        input_content = f'''
		Following are a few examples:
		
		Example 1: 
        Input: "A young girl stands in a doorway, holding a donut and smiling at the camera. >> A girl opens the door and walks into a well-lit kitchen. >> A person is standing in a dimly lit hallway, holding a sandwich and remaining in the same position, with no changes in their posture or the environment. >> A person is standing by a door, holding some food, and sways side to side before turning off the light."
                
        Output:        
        G(in(person, doorway)∧holding(person,food)⇒F(twisting(person, doorknob)∧carrying(person, food)∧other_relationship(person,light)))
        G(holding(person,sandwich)∧on_the_side_of(person, door)∧standing_on(person,floor)⇒F(holding(person,food)∧in(person,doorway)))
        G(on_the_side_of(person, door)∧holding(person, sandwich)∧standing_on(person, floor)⇒F(twisting(person, light)))
        G(on_the_side_of(person, door)∧holding(person, sandwich)∧standing_on(person, floor) ⇒ F (holding(person,food)∧in(person,doorway)))        
    
        
        Example 2:  
        Input:       
        The video begins with a man standing in a room, holding a piece of paper and seemingly focused on it. >> The man walks over to a table, places the paper down, and then walks over to a chair and sits down. >> The man looks up and appears to be reading or writing on the paper. >> The man stands up and walks towards a desk, holding a book and pen. >> The man starts writing, indicating a shift from reading to writing. >> The man continues writing, possibly on a piece of paper not shown in the scene.
        
        Output: 
        G(holding(person, paper)∧in_front_of(person, table)⇒F(not_contacting(person, paper)))
        G(sitting_on(person, chair)∧looking_at(person, paper)⇒F(holding(person, book)∧holding(person, pen)))
        G(holding(person, book)∧holding(person, pen)∧in_front_of(person, desk)⇒F(writing_on(person, paper)))
        G(holding(person, paper)∧standing_on(person, floor)⇒F(in_front_of(person, table)))
        G(standing_on(person, floor)∧in_front_of(person, desk)⇒F(holding(person, pen)∧writing_on(person, paper)))
        G(looking_at(person, paper)⇒F(holding(person, book)∧holding(person, pen)))
        
        Example 3:         
        Input:
        A woman sits on a bed and uses a laptop. >> A woman takes a sip of her drink. >> A woman picks up her drink. >> A woman continues to work on the laptop, focused on her activity. >> A woman appears to be engaged and dedicated to her work. Later, a young individual sits on a bed, interacting with a laptop and a cup. >> A young individual drinks from the cup. >> A young individual places the cup on the laptop. >> A young individual continues to type on the laptop, their excitement growing.

        Output: 
        G(sitting_on(person,bed)∧touching(person,laptop)⟹F(looking_at(person,laptop)))
        G(holding(person,cup/glass/bottle)⟹F(drinking_from(person,cup/glass/bottle)))
        G(drinking_from(person,cup/glass/bottle)⟹F(looking_at(person,laptop)∧touching(person,laptop)))
        G(sitting_on(person,bed)∧touching(person,laptop)⟹F(holding(person,cup)∧drinking_from(person,cup)))
        G(touching(cup/glass/bottle,laptop)⟹F(typing(person,laptop)))
        G(drinking_from(person,cup/glass/bottle)∧holding(person,cup/glass/bottle)⟹F(touching(cup/glass/bottle,laptop)∧¬holding(person,cup/glass/bottle)))
        G(holding(person,cup/glass/bottle)∧drinking_from(person,cup/glass/bottle) ⟹ F(looking_at(person,laptop)∧touching(person,laptop)))
        G(sitting_on(person,bed)∧holding(person,cup/glass/bottle)⟹F(drinking_from(person,cup/glass/bottle)∧touching(person,laptop)))

		Follow the same format to generate plausible STL Formulas for a given video input and make sure to include only the output response. 
		Input: {stripped_caption}.
		'''

        return system_prompt, input_content


    def construct_implication_prompts(self, video_id):
        pass

    def construct_forward_cancellation_prompts(self, video_id):
        pass

    def generate_backward_dependency_clauses(self, video_id):
        system_prompt, content_prompt = self.construct_backward_dependency_prompts(video_id)
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

    def generate_clauses(self):
        video_id_list = list(self.captions.keys())
        random.shuffle(video_id_list)
        self.init_llama_3_2()
        for video_id in tqdm(video_id_list):

            if os.path.exists(os.path.join(self.captions_dir, f"{video_id}.txt")):
                print(f"Skipping {video_id}")
                continue

            self.generate_backward_dependency_clauses(video_id)


def main():
    clause_generator = GenerateClauses()
    clause_generator.generate_clauses()

def compile_rules():
    captions = {}
    captions_dir = "/data/rohith/ag/captions/segmented/"
    for file in os.listdir(captions_dir):
        video_id = file.split(".")[0]
        with open(os.path.join(captions_dir, file), "r") as f:
            caption = f.read()
            captions[video_id] = caption

    store_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captions")
    os.makedirs(store_file_dir, exist_ok=True)

    store_file_path = os.path.join(store_file_dir, "segmented_captions.json")
    with open(store_file_path, "w") as f:
        json.dump(captions, f)


if __name__ == "__main__":
    main()
    # compile_rules()
