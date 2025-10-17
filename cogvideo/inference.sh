#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video.py --base configs/cogvideox_5b_i2v_lora.yaml configs/inference.yaml --seed 42 --image2video"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"