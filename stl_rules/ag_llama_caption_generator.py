import json
import os
from typing import List

import requests
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

    def generate_video_frame_captions(self, video_id):

        video_frames = os.listdir(os.path.join(self.video_frames_dir, video_id))
        video_image_caption_json = {}
        for video_frame in tqdm(video_frames):
            frame_path = os.path.join(self.video_frames_dir, video_id, video_frame)
            image = Image.open(frame_path)

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": " "}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(**inputs, max_new_tokens=128)
            video_image_caption_json[video_frame[:-4]] = self.processor.decode(output[0])

        with open(f"/data/rohith/ag/captions/llame_3_2_11B/{video_id[:-4]}.json", "w") as f:
            json.dump(video_image_caption_json, f)

    def generate_video_captions(self):
        video_ids = os.listdir(self.video_frames_dir)
        for video_id in tqdm(video_ids):
            self.generate_video_frame_captions(video_id)


def main():
    ag_llama_caption_generator = AgLLamaCaptionGenerator()
    ag_llama_caption_generator.generate_video_captions()

if __name__ == "__main__":
    main()



