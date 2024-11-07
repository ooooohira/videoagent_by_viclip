import cv2

# videoをfps指定して、フレームに落とし込む
def split_video_to_segments(video_path, fps):
    # # 使用例
    # video_path = "/experiment/data/videos/videos/0a01d7d0-11d6-4af6-abd9-2025656d3c63.mp4"  # 動画のパスを指定
    # fps = 1  # 1FPSごとにセグメント化
    # segments = split_video_to_segments(video_path, fps)
    # print(segments)

    # 動画を読み込み
    cap = cv2.VideoCapture(video_path)
    
    # 動画の秒数とFPSの取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"fps is {video_fps}")
    duration = total_frames / video_fps
        
    # 指定FPSごとにセグメントを作成
    segment_duration = 1 / fps  # 各セグメントの秒数
    segments = {}
    segment_id = 0
    current_time = 0.0
    frames = []
    
    while current_time < duration:
        # セグメントの開始時間を保存
        segments[segment_id] = round(current_time, 2)
        
        # current_time に対応するフレームを取得
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  # 時間をミリ秒単位で設定
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)  # フレームをframesリストに追加
        else:
            print(f"Failed to capture frame at {current_time} seconds.")
        
        # 次のセグメントの準備
        current_time += segment_duration
        segment_id += 1
    
    cap.release()
    return segments, frames



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


# video_path = "/experiment/data/videos/videos/0a01d7d0-11d6-4af6-abd9-2025656d3c63.mp4"  # 動画のパスを指定
# start_time = 1  # 取得開始秒数
# fps = 1  # 取得したいフレームのFPS
# num_frames = 8  # 取得するフレーム数
# frames = split_video_to_segments(video_path, fps)
# # 例
# output_folder = "output_frames"
# save_frames_as_png(frames, output_folder)