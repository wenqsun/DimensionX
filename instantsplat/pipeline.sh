CASE_NAME=horse
NUM=50
VIDEO_PATH='/pfs/mt-1oY5F7/chenshuo/CogVideo/sat/outputs/scene_180_109_6000/0_"A_majestic_bronze_horse_statue_stands_proudly_on_a_beige_pedestal_in_a_serene_park_setting._The_statue_captures_the_hor/0/000000.mp4'

export CUDA_VISIBLE_DEVICES=3

python get_frame.py ${VIDEO_PATH} /pfs/mt-1oY5F7/chenshuo/instantsplat/data/images/${CASE_NAME}_${NUM} ${NUM}

# dust3r to estimate the camera poses and initial point clouds
python dust3r_inference.py --dataset ${CASE_NAME}_${NUM}

# dust3r on dynamicrafter generated videos
# CUDA_VISIBLE_DEVICES=2 python dust3r_inference.py --dataset dynamicrafter_data/10_july/bedroom_3

python 3dgs.py --dataset ${CASE_NAME}_${NUM}  --iter 10000 --use_confidence --lambda_lpips 0.3

# use the estimated camera poses and initial point clouds to optimize the 3dgs
# python 3dgs.py --dataset horse --iter 2000

# render 
# python render.py --model_path data/scenes/instantsplat/horse_two_view_train/output_1000_lpips_0.0 --source_path data/scenes/instantsplat/horse_two_view_all

# python render.py -m instantsplat/horse_two_view_train/output_1000_lpips_0.0 -s instantsplat/horse_two_view_all

# metric
# python metrics.py -m data/scenes/instantsplat/horse_two_view_train/output_1000_lpips_0.0

# a unified script for dust3r and 3dgs

# first copy data
# bash run_build_data.sh 
# bash run_recon.sh --method luma --lambda_lpips 0.4