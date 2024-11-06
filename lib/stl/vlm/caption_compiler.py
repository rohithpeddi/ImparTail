import json
import os

import pandas as pd
from tqdm import tqdm


def compile_longvu_captions_to_json():
    longvu_json = {}
    captions_dir_path = "/data/rohith/ag/captions/longvu/"
    for caption_file in tqdm(os.listdir(captions_dir_path)):
        if caption_file.endswith(".txt"):
            caption_file_path = os.path.join(captions_dir_path, caption_file)
            video_id = caption_file.split("_")[0]

            # Read all text from the file and encode it to json as a string
            with open(caption_file_path, "r") as f:
                caption_text = f.read()
                longvu_json[video_id] = caption_text

    # Write the json to a file
    json_file_path = "/data/rohith/ag/captions/longvu.json"
    with open(json_file_path, "w") as f:
        json.dump(longvu_json, f)


def compile_chatuniv_captions_to_json():
    chatuniv_json = {}
    captions_dir_path = "/data/rohith/ag/captions/chatuniv/"
    for split_json_file in tqdm(os.listdir(captions_dir_path)):
        split_json_file_path = os.path.join(captions_dir_path, split_json_file)
        with open(split_json_file_path, "r") as f:
            split_json = json.load(f)

        for video_id_w_ext, caption in split_json.items():
            video_id = video_id_w_ext[:-4]
            chatuniv_json[video_id] = caption

    # Write the json to a file
    json_file_path = "/data/rohith/ag/captions/chatuniv.json"
    with open(json_file_path, "w") as f:
        json.dump(chatuniv_json, f)


def compile_charades_video_desc_json():
    charades_json = {}
    captions_dir_path = "/data/rohith/ag/captions/charades/"

    for split_csv_file in tqdm(os.listdir(captions_dir_path)):
        split_csv_file_path = os.path.join(captions_dir_path, split_csv_file)
        csv_df = pd.read_csv(split_csv_file_path)

        # Loop through each row in the csv file
        for _, row in csv_df.iterrows():
            video_id = row["id"]
            description = row["descriptions"]
            charades_json[video_id] = description

    # Write the json to a file
    json_file_path = "/data/rohith/ag/captions/charades.json"
    with open(json_file_path, "w") as f:
        json.dump(charades_json, f)


def compile_multi_agent_captions_to_json():
    chatuniv_agent_cap_json_path = "/data/rohith/ag/captions/chatuniv.json"
    longvu_agent_cap_json_path = "/data/rohith/ag/captions/longvu.json"
    charades_agent_cap_json_path = "/data/rohith/ag/captions/charades.json"

    multi_agent_json = {}
    with open(chatuniv_agent_cap_json_path, "r") as f:
        chatuniv_json = json.load(f)

    with open(longvu_agent_cap_json_path, "r") as f:
        longvu_json = json.load(f)

    with open(charades_agent_cap_json_path, "r") as f:
        charades_json = json.load(f)

    for key, value in tqdm(chatuniv_json.items()):
        multi_agent_json[key] = {}
        multi_agent_json[key]["chatuniv"] = value
        multi_agent_json[key]["longvu"] = longvu_json[key]
        multi_agent_json[key]["charades"] = charades_json[key]

    # Write the json to a file
    json_file_path = "captions/multi_agent_cap.json"
    with open(json_file_path, "w") as f:
        json.dump(multi_agent_json, f)



if __name__ == '__main__':
    compile_multi_agent_captions_to_json()