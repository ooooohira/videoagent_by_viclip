import av
import numpy as np
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
print(" eval mode!!!!")
# model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf").to('cuda')
# processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", device_map="auto")
model.eval()
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
# 8枚のフレームと、質問を元に答えを生成
def qa_generation(frames, question):
    prompt = f"USER: <video>{question} ASSISTANT:"
    frames = np.array(frames)
    inputs = processor(text=prompt, videos=frames, return_tensors="pt").to(model.device)
    generate_ids = model.generate(**inputs, max_length=80)
    ret_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    start_index = ret_text.find("ASSISTANT: ")
    ret_text = ret_text[start_index + len("ASSISTANT: "):]
    # print(ret_text)
    return ret_text

# # 使用例
# import cv2
# video_path = "/experiment/code/ViCLIP/example1.mp4"
# def _frame_from_video(video):
#     while video.isOpened():
#         success, frame = video.read()
#         if success:
#             yield frame
#         else:
#             break
# video = cv2.VideoCapture(video_path)
# frames = [x for x in _frame_from_video(video)][:8]
# print(qa_generation(frames, "What is there??"))