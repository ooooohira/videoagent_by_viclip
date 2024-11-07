from ViCLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ViCLIP.viclip import ViCLIP
import torch
import numpy as np
import cv2
import json
from video_frame import split_video_to_segments

clip_candidates = {'viclip':None, 'clip':None}

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            # m = vclip
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)

def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std


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


name='viclip'
clip, tokenizer = get_clip(name)
clip = clip.to(torch.device('cuda'))
# start_num 〜 end_num のなかで、textにヒットするもの1件持ってくる。
def make_emb(frames, start_num, end_num, device=torch.device('cuda')):
    emb_list = []
    for i in range(start_num-3, end_num+1-3):
        target = frames[max(i, 0):min(i+8, len(frames))]
        if len(target) < 8:
            if i < 0:
                pad_front = [frames[0]] * abs(i)
                target = pad_front + target
            else:
                pad_front = [target[-1]] * abs(8-len(target))
                target =  target + pad_front

        frames_tensor = frames2tensor(target, device=device)
        vid_feat = get_vid_feat(frames_tensor, clip).cpu()
        emb_list.append(vid_feat)
        
    return torch.vstack(emb_list)


input_ann_file = "/experiment/data/subset_anno.json"
anns = json.load(open(input_ann_file, "r"))
import pickle
from tqdm import tqdm
print("o.1111111!!!!!!!!!!aaaaaa")
for video_id in tqdm(list(anns.keys())):
    video_path = "/experiment/data/videos/videos/"+video_id+".mp4"
    save_path = "/experiment/data/emb_videos/"+video_id+".pkl"
    _, frames = split_video_to_segments(video_path, 0.1)
    emb_list = make_emb(frames, 0, len(frames)-1)
    print(emb_list.shape)
    with open(save_path, "wb") as f:
        pickle.dump(emb_list, f)
     
