import numpy as np
import os
import io
import cv2
import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can access the GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch cannot access the GPU.")

from config import Config, eval_dict_leaf
from utils import retrieve_text, _frame_from_video, setup_internvideo2

video = cv2.VideoCapture('example1.mp4')
frames = [x for x in _frame_from_video(video)]
print(f'{len(frames)} frames found in sample video')

text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                   "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
                   "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                   "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
                   "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                   "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                   "A playful dog slides down a snowy hill, wagging its tail with delight.",
                   "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
                   "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                   "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]

print('Loading configs!')
config = Config.from_file('internvideo2_stage2_config.py')
config = eval_dict_leaf(config)

print('Loading model!')
model_pth = '/workspace/InternVideo2-stage2_1b-224p-f4.pt'
config['pretrained_path'] = model_pth
intern_model, tokenizer = setup_internvideo2(config)

print('Model type: ', type(intern_model))
print('Tokenizer type: ', type(tokenizer))

texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)
for t, p in zip(texts, probs):
    print(f'text: {t} ~ prob: {p:.4f}')
