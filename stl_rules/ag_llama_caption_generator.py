import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor


class AgLLamaCaptionGenerator:

    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.video_frames_dir = "/data/rohith/ag/frames_annotated/"
        self.captions_dir = "/data/rohith/ag/captions/llame_3_2_11B/"
        os.makedirs(self.captions_dir, exist_ok=True)

    def generate_video_frame_captions(self, video_id):

        video_frames = os.listdir(os.path.join(self.video_frames_dir, video_id))
        video_image_caption_json = {}
        for video_frame in tqdm(video_frames):
            frame_path = os.path.join(self.video_frames_dir, video_id, video_frame)
            image = Image.open(frame_path)

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": "You are an expert in describing objects and relationships between the person and the observed objects."
                             "Can you describe the given image by focussing only on the objects from the following categories: [person, bag, bed, blanket, book, box, broom, chair, closet/cabinet, clothes, "
                             "cup/glass/bottle, dish, door, doorknob, doorway, floor, food, groceries, laptop, light, medicine, mirror, "
                             "paper/notebook, phone/camera, picture, pillow, refrigerator, sandwich, shelf, shoe, sofa/couch, "
                             "table, television, towel, vacuum, window] and relationships between the person and the object categories"
                             " from the following categories : [looking at, not looking at, unsure, above, beneath, in front of, behind, on the side of, "
                             "in, carrying, covered by, drinking from, eating, have it on the back, "
                             "holding, leaning on, lyingon, not contacting, other relationship, sitting on, standing on, touching, twisting, wearing, wiping, writingon]"
                             "Along with description, provide the relationship triplets <subject, relationship, object> observed in the frames."}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(**inputs, max_new_tokens=512)
            video_image_caption_json[video_frame[:-4]] = self.processor.decode(output[0])

        video_caption_path = os.path.join(self.captions_dir, f"{video_id[:-4]}.json")
        with open(video_caption_path, "w") as f:
            json.dump(video_image_caption_json, f)

    def generate_video_captions(self):
        video_ids = os.listdir(self.video_frames_dir)
        for video_id in tqdm(video_ids):
            if not os.path.exists(os.path.join(self.captions_dir, f"{video_id[:-4]}.json")):
                self.generate_video_frame_captions(video_id)
            else:
                print(f"Skipping {video_id}")


def main():
    ag_llama_caption_generator = AgLLamaCaptionGenerator()
    ag_llama_caption_generator.generate_video_captions()

if __name__ == "__main__":
    main()



