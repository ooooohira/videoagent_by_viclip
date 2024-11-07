from ViCLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ViCLIP.viclip import ViCLIP
import torch
import numpy as np
import cv2

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

# 自分で作った関数
def get_text_feat(texts, clip, tokenizer):
    ret_list = []
    for t in texts:
        text_feat_d = {}
        feat = clip.get_text_features(t, tokenizer, text_feat_d).cpu()
        ret_list.append(feat)
    return ret_list

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
def search_best_segment(frames, text, start_num, end_num, device=torch.device('cuda')):
    text_feat_d = {}
    text_feat_d = get_text_feat_dict([text], clip, tokenizer, text_feat_d)
    text_feat = text_feat_d[text]
    sim_list = []
    index_list = []
    for i in range(start_num-3, end_num+1-3):
        target = frames[max(i, 0):min(i+8, len(frames))]
        if len(target) < 8:
            if i < 0:
                pad_front = [frames[0]] * abs(i)
                target = pad_front + target
            else:
                pad_front = [target[-1]] * abs(8-len(target))
                target = pad_front + target

        frames_tensor = frames2tensor(target, device=device)
        vid_feat = get_vid_feat(frames_tensor, clip)
        sim = torch.matmul(vid_feat , text_feat.T).cpu().item()

        sim_list.append(sim)
        index_list.append(i)
    best_index = index_list[sim_list.index(max(sim_list))]
    return_frames = frames[max(best_index, 0):min(best_index+8, len(frames))]
    if len(return_frames) < 8:
        if best_index < 0:
            pad_front = [return_frames[0]] * abs(best_index)
            return_frames = pad_front + return_frames
        else:
            pad_back = [return_frames[-1]] * abs(8-len(return_frames))
            return_frames = return_frames + pad_back
    return return_frames, best_index + start_num-3 + 3
    # 該当した8枚のフレーム、|||(|)|||| (|)のインデックス


# video = cv2.VideoCapture('example1.mp4')
# frames = [x for x in _frame_from_video(video)]

# a = search_best_segment(frames, "man", 0, len(frames))


# import os
# def save_frames_as_png(frames, output_folder):
#     # # 例
#     # output_folder = "output_frames"
#     # save_frames_as_png(frames, output_folder)

#     # 出力フォルダが存在しない場合、作成する
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # 各フレームをPNG形式で保存
#     for i, frame in enumerate(frames):
#         filename = os.path.join(output_folder, f"frame_{i:04d}.png")  # 4桁のゼロ埋め
#         cv2.imwrite(filename, frame)
#         print(f"Saved: {filename}")

# save_frames_as_png(a, "fuckyou")





# 事前に計算しておいた埋め込みから検索
import pickle
from typing import List
import numpy as np
def get_embeddings(inputs: List[str]) -> np.ndarray:
    return get_text_feat(inputs, clip, tokenizer)
def frame_retrieval_seg_ego(descriptions, video_id, sample_idx):
    with open(f"/experiment/data/emb_videos/{video_id}.pkl", "rb") as f:
        frame_embeddings = pickle.load(f)[::5, :]
    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    # qa_text_embedding = get_embeddings(
    #     [description["pred_answer"] for description in descriptions]
    # )
    q_list = [description["query"] for description in descriptions]
    # 各要素を足し合わせて新しいリストに
    text_embedding = [t for t, q in zip(text_embedding, qa_text_embedding)]
    print(frame_embeddings.shape)

    frame_idx = []
    for idx, description in enumerate(descriptions):
        # seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[max(description["start frame number"]-1,0) : description["end frame number"]]
        seg_similarity = torch.matmul(seg_frame_embeddings, text_embedding[idx].T).squeeze()
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax()
        frame_idx.append(seg_frame_idx.item())

    #     print(seg_frame_embeddings.shape)
    #     print(seg_similarity.shape)
    #     print(seg_similarity)
    #     print(seg_similarity.argmax())
    #     print(type(seg_frame_idx.item()))
    # print(frame_idx)
    return frame_idx, q_list


# frame_retrieval_seg_ego([{"segment_id": "1", "duration": "xxx - xxx", "description": "fruit can"}], "0a8b2c9d-b54c-4811-acf3-5977895d2445", [1, 180])
