#! /bin/bash

export CUDA_VISIBLE_DEVICES=7

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video_lowR.py --base configs/cogvideox_5b_i2v_lora_145.yaml configs/inference_145.yaml --seed 42 --image2video"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"

# Also you can use Real-ESRGAN for super resolution
# python ../Real-ESRGAN/inference_realesrgan_video_batch.py -n RealESRGAN_x4plus -f configs/inference_145.yaml --num_process_per_gpu 1