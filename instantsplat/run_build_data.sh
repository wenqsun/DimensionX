#!/bin/bash

video_type="luma"
scene_type="real10k"
scene_name="0bb99505a71035cc"
frame_interval=8

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video_type) video_type="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --frame_interval) frame_interval="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 从数据集中copy训练和测试图片
source copy.sh --scene_type $scene_type --scene_name $scene_name


echo "----------------extract frames from video----------------"
# 从生成的视频中抽帧
python script/extract_frame.py --video_type $video_type --scene_type $scene_type --scene_name $scene_name --frame_interval $frame_interval

echo "----------------copy data----------------"
# 将抽帧的数据和测试数据并到一起
python script/copy_data.py --video_type $video_type --scene_type $scene_type --scene_name $scene_name --frame_interval $frame_interval
