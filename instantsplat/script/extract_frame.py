import cv2
import os
import numpy as np
import argparse

def extract_frames(video_path, output_folder, frame_interval=1):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"{saved_count+1}.png")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        frame_count += 1

    # 确保最后一帧被保存
    last_frame_name = os.path.join(output_folder, f"{saved_count+1}.png")
    cv2.imwrite(last_frame_name, frames[-1])

    cap.release()
    print(f"Extracted {saved_count+1} frames.")  # 加1是因为单独保存了最后一帧

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_type', type=str, default='luma')
    parser.add_argument('--scene_type', type=str, default='real10k', help='Scene type.')
    parser.add_argument('--scene_name', type=str, default='horse', help='Scene name.')
    parser.add_argument('--frame_interval', type=int, default=8, help='Save every nth frame.')
    args = parser.parse_args()

    # 使用示例
    root_path = f'data/images/{args.video_type}/{args.scene_type}/{args.scene_name}'
    os.makedirs(root_path, exist_ok=True)
    video_path = f'{root_path}/{args.scene_name}.mp4'  # 替换为你的视频文件路径
    output_folder = f'{root_path}/{args.scene_name}_frame_{args.frame_interval}'  # 替换为你想保存图片的文件夹路径

    extract_frames(video_path, output_folder, args.frame_interval)
