#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

video_path='/pfs/mt-1oY5F7/chenshuo/good_cases/sparse_view/5_"A_meticulously_crafted_yellow_LEGO_bulldozer_sits_proudly_on_a_wooden_table,_its_intricate_details_capturing_the_essenc/0/000000.mp4'
# 定义参数
method=360
scene_type="test_case"
scene_name="kitchen"
val_length=12
lambda_lpips=0.3
use_confidence=true  # 默认为 false

# dataset_suffix="_all"
train_dataset_suffix="_train"
model_path_suffix="_train/output_30000_lpips_"
source_path_suffix="_test"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --val_length) val_length="$2"; shift ;;
        --lambda_lpips) lambda_lpips="$2"; shift ;;
        --use_confidence) use_confidence=true ;;  # 解析 --use_confidence 参数
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

meta_dataset=${method}/${scene_type}/${scene_name}
dataset="${meta_dataset}"
train_dataset="${meta_dataset}${train_dataset_suffix}"
model_path="${meta_dataset}${model_path_suffix}${lambda_lpips}_use_conf"
source_path="${meta_dataset}${source_path_suffix}"

dataset_path="data/images/${meta_dataset}"

# get frame
python get_frame.py $video_path $dataset_path 38

# python get_test_imgs.py $scene_type $scene_name $method $dataset_path

# python downsample.py $dataset_path

# 执行第一个命令
python dust3r_inference.py --dataset "$dataset"

# copy the 'data/scene/{$dataset}' to 'data/scene/{$dataset}_train' and 'data/scene/{$dataset}_test'
mkdir -p "data/scenes/${dataset}_train"
mkdir -p "data/scenes/${dataset}_test"
cp -r "data/scenes/$dataset/." "data/scenes/${dataset}_train"
cp -r "data/scenes/$dataset/." "data/scenes/${dataset}_test"

# modify the train and test
python script/modify_data.py --dataset $dataset


# 检查上一个命令是否成功执行
if [ $? -eq 0 ]; then
    echo "dust3r_inference.py completed successfully"
    # 执行第二个命令
    python 3dgs.py --dataset "$train_dataset" --lambda_lpips $lambda_lpips --iter 30000

    if [ $? -eq 0 ]; then
        echo "3dgs.py completed successfully"
        # 执行第三个命令
        python render.py --model_path data/scenes/"$model_path" --source_path data/scenes/"$source_path"

        # if [ $? -eq 0 ]; then
        #     echo "render.py completed successfully"
        #     # 执行第四个命令
        #     python metrics.py -m data/scenes/"$model_path"
        # else
        #     echo "render.py failed"
        #     exit 1
        fi
    else
        echo "3dgs.py failed"
        exit 1
    fi
else
    echo "dust3r_inference.py failed"
    exit 1
fi