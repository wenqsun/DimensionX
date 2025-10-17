#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

video_path='/pfs/mt-1oY5F7/chenshuo/good_cases/web_case/360/capybara.mp4'
method=ours_360
scene_type="360_new"
scene_name="capybara"
val_length=12
lambda_lpips=0.4
use_confidence=true  

# dataset_suffix="_all"
train_dataset_suffix="_train"
model_path_suffix="_train/output_30000_lpips_"
source_path_suffix="_test"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --method) method="$2"; shift ;;
        --scene_type) scene_type="$2"; shift ;;
        --scene_name) scene_name="$2"; shift ;;
        --val_length) val_length="$2"; shift ;;
        --lambda_lpips) lambda_lpips="$2"; shift ;;
        --use_confidence) use_confidence=true ;; 
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
python get_frame.py $video_path $dataset_path 40

# python get_test_imgs.py $scene_type $scene_name $method $dataset_path

# python downsample.py $dataset_path

python dust3r_inference.py --dataset "$dataset"

# copy the 'data/scene/{$dataset}' to 'data/scene/{$dataset}_train' and 'data/scene/{$dataset}_test'
mkdir -p "data/scenes/${dataset}_train"
mkdir -p "data/scenes/${dataset}_test"
cp -r "data/scenes/$dataset/." "data/scenes/${dataset}_train"
cp -r "data/scenes/$dataset/." "data/scenes/${dataset}_test"

# modify the train and test
python script/modify_data.py --dataset $dataset


if [ $? -eq 0 ]; then
    echo "dust3r_inference.py completed successfully"
    python 3dgs.py --dataset "$train_dataset" --lambda_lpips $lambda_lpips --iter 30000 --use_confidence

    if [ $? -eq 0 ]; then
        echo "3dgs.py completed successfully"
        python render.py --model_path data/scenes/"$model_path" --source_path data/scenes/"$source_path"

        if [ $? -eq 0 ]; then
            echo "render.py completed successfully"
            # python metrics.py -m data/scenes/"$model_path"
        else
            echo "render.py failed"
            exit 1
        fi
    else
        echo "3dgs.py failed"
        exit 1
    fi
else
    echo "dust3r_inference.py failed"
    exit 1
fi