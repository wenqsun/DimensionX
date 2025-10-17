python get_frame.py '/pfs/mt-1oY5F7/chenshuo/CogVideo/sat/outputs/scene_360_145_real10k_8000_rand_1145/12_"Nestled_within_a_serene_park,_a_majestic_tree_stands_tall,_its_gnarled_trunk_encircled_by_a_rustic_wooden_bench._The_be/0/000000.mp4' /pfs/mt-1oY5F7/chenshuo/instantsplat/data/images/tree_50 50
'/pfs/mt-1oY5F7/chenshuo/CogVideo/sat/outputs/scene_180_109_6000/2_"A_vibrant_bouquet_of_flowers_in_a_clear_glass_vase_sits_on_a_weathered_wooden_deck._The_arrangement_includes_large_whit/0/000000.mp4'
# dust3r to estimate the camera poses and initial point clouds
python dust3r_inference.py --dataset horse

# dust3r on dynamicrafter generated videos
CUDA_VISIBLE_DEVICES=2 python dust3r_inference.py --dataset dynamicrafter_data/10_july/bedroom_3

CUDA_VISIBLE_DEVICES=2 python 3dgs.py --dataset dynamicrafter_data/10_july/bedroom_3_gt

CUDA_VISIBLE_DEVICES=2 python 3dgs.py --dataset tree_50 --lambda_lpips 0.3 --iter 10000 --use_confidence
# use the estimated camera poses and initial point clouds to optimize the 3dgs
python 3dgs.py --dataset horse --iter 2000

# render 
python render.py --model_path data/scenes/instantsplat/horse_two_view_train/output_1000_lpips_0.0 --source_path data/scenes/instantsplat/horse_two_view_all

# python render.py -m instantsplat/horse_two_view_train/output_1000_lpips_0.0 -s instantsplat/horse_two_view_all

# metric
python metrics.py -m data/scenes/instantsplat/horse_two_view_train/output_1000_lpips_0.0

# a unified script for dust3r and 3dgs

# first copy data
bash run_build_data.sh 
bash run_recon.sh --method luma --lambda_lpips 0.4