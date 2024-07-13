import torch
import cv2
import numpy as np
import json

from single_modality.models.internvideo2 import internvideo2_1B_patch14_224

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can access the GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch cannot access the GPU.")

model_pth = "/workspace/1B_ft_ssv2_f8.pth"
model = internvideo2_1B_patch14_224(use_flash_attn = False,
                                    use_fused_rmsnorm = False,
                                    use_fused_mlp = False,
                                    num_classes = 174)
model = model.to(torch.device("cuda"))
checkpoint = torch.load(model_pth, map_location="cpu")
try:
    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint["module"] # This is a deepspeed stage 1 model
except:  
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

video = cv2.VideoCapture('./video1.mp4')
frames = [x for x in _frame_from_video(video)]
print(f'{len(frames)} frames found in sample video')
context_size = 60
prediction = {}
with open('./labels.json', 'r') as file:
    data = json.load(file)
index_label_map = {int(value): key for key, value in data.items()}

print('Making predictions')
print('-----------------')
for i in range(context_size, len(frames) - context_size):
    context_frames = frames[i - context_size: i + context_size]
    context_tensor = frames2tensor(context_frames)
    context_tensor = context_tensor.permute(0, 2, 1, 3, 4).to(model.dtype)
    with torch.no_grad():
        output = model(context_tensor)
        output = torch.softmax(output, dim=1)
        output = output.cpu().numpy()
        output = output.flatten()
        top2 = np.argsort(output)[-2:]
        top2 = top2[::-1]
        prediction[i] = index_label_map[top2[0]]
        print(f"frame: {i}, top: {index_label_map[top2[0]]}, prob: {output[top2[0]]}")
        print(f"frame: {i}, second_best: {index_label_map[top2[1]]}, prob: {output[top2[1]]}")
        print('--------------------')
    if i > 900:
        break

print('saving predictions!')
with open('video1_predictions.json', 'w') as file:
    json.dump(prediction, file, indent=2)
