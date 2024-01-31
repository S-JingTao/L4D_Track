#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import os
import torch
import clip
import json


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    caption_all = []
    scene_list = []
    for key, value in content.items():
        scene_list.append(key)
        multi_caption = []
        for _, v in value.items():
            multi_caption.append(v)
        caption_all.append(multi_caption)

    return scene_list, caption_all


def save_caption(save_tensor, scene, save_path):
    save_name = "{0}_caption.pt".format(scene)
    save_dir = os.path.join(save_path, save_name)
    torch.save(save_tensor, save_dir, _use_new_zipfile_serialization=False)


def text_to_tensor(caption):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)

    text = clip.tokenize(caption).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).float()
    return text_features


def main():
    """
    cd ../video_language_model/
    conda create -n clip python=3.6
    conda activate clip
    pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    
    conda install --yes -c cudatoolkit=10.2
    conda install cudnn==7.6.5

    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    """

    dir_path = "/home/sjt/LV-Track/models/"
    file_path = os.path.join(dir_path, "video_language_model")
    caption_file = ["train_data_caption.json", "test_data_caption.json"]

    # print("GPU or CPU?:", torch.cuda.is_available)
    print("available language encoder:", clip.available_models())

    for temp_file in caption_file:

        dataset_name = temp_file.split("_")[0]
        print(dataset_name)
        save_path = os.path.join(file_path, dataset_name + "_caption")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        each_scene_list, each_caption_list = load_json_file(os.path.join(file_path, temp_file))
        for i in range(len(each_scene_list)):
            print(each_scene_list[i])
            each_tensor = text_to_tensor(each_caption_list[i])
            save_caption(each_tensor, each_scene_list[i], save_path)


if __name__ == '__main__':
    main()
