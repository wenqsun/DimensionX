import os
import subprocess
import glob
from multiprocessing import Pool, Manager
import time

# Define the root path of your datasets
root_path = '/pfs/mt-1oY5F7/chenshuo/sparse_test/test_case'
# datasets = ["dl3dv", "llff", "mipnerf360", "TanksandTemples"]
# datasets = ["llff"]
# datasets = ["TanksandTemples"]
datasets = ["dl3dv"]

gpu_count = 2

# Define the common parameters
method = "instantsplat"
val_length = 12
lambda_lpips = 0.5
use_confidence = "true"
train_dataset_suffix = "_train"
model_path_suffix = "_train/output_10000_lpips_"
source_path_suffix = "_test"

# Get all cases with paths for processing
all_cases = []
for dataset in datasets:
    dataset_path = os.path.join(root_path, dataset)
    for case in os.listdir(dataset_path):
        case_path = os.path.join(dataset_path, case)
        image0_path = os.path.join(case_path, 'train/01.png')
        image1_path = os.path.join(case_path, 'train/02.png')
        # if dataset=='dl3dv':
        #     image0_path = os.path.join(case_path, 'train/01.png')
        #     image1_path = os.path.join(case_path, 'train/02.png')
        # if dataset=='mipnerf360':
        #     image0_path = os.path.join(case_path, 'train/01.JPG')
        #     image1_path = os.path.join(case_path, 'train/02.JPG')
        # if dataset=='TanksandTemples':
        #     image0_path = os.path.join(case_path, 'train/01.jpg')
        #     image1_path = os.path.join(case_path, 'train/02.jpg')
        # if dataset=='llff':
        #     image0_path = os.path.join(case_path, 'train/01.JPG')
        #     image1_path = os.path.join(case_path, 'train/02.JPG')

        all_cases.append((dataset, case, image0_path, image1_path))

def process_case(args):
    gpu_id, dataset, case_name, image0_path, image1_path = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    scene_type = dataset
    scene_name = case_name
    meta_dataset = f"{method}/{scene_type}/{scene_name}"
    dataset_dir = f"data/images/{meta_dataset}"
    train_dataset = f"{meta_dataset}{train_dataset_suffix}"
    model_path = f"{meta_dataset}{model_path_suffix}{lambda_lpips}_use_conf"
    source_path = f"{meta_dataset}{source_path_suffix}"

    # Commands
    commands = [
        f"python get_instant_image.py {image0_path} {image1_path} {dataset_dir}",
        f"python get_test_imgs.py {scene_type} {scene_name} {method} {dataset_dir}",
        # f"python downsample.py {dataset_dir}",
        f"python dust3r_inference.py --dataset {meta_dataset}",
        f"mkdir -p data/scenes/{meta_dataset}_train && mkdir -p data/scenes/{meta_dataset}_test",
        f"cp -r data/scenes/{meta_dataset}/. data/scenes/{meta_dataset}_train",
        f"cp -r data/scenes/{meta_dataset}/. data/scenes/{meta_dataset}_test",
        f"python script/modify_data.py --dataset {meta_dataset}",
        f"python 3dgs.py --dataset {train_dataset} --lambda_lpips {lambda_lpips} --iter 10000 --use_confidence",
        f"python render.py --model_path data/scenes/{model_path} --source_path data/scenes/{source_path}",
        f"python metrics.py -m data/scenes/{model_path}"
    ]

    # Execute commands sequentially
    for cmd in commands:
        print(f"[GPU {gpu_id}] Executing: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"[GPU {gpu_id}] Error: Command '{cmd}' failed.")
            break
    else:
        print(f"[GPU {gpu_id}] All commands for case {case_name} completed successfully.")

if __name__ == "__main__":
    # Set up GPU manager to keep track of which GPU is free
    with Manager() as manager:
        gpu_pool = manager.list(range(gpu_count))  # Available GPUs

        def task_manager(case_data):
            while True:
                if gpu_pool:
                    gpu_id = gpu_pool.pop(0)  # Take an available GPU
                    try:
                        process_case((gpu_id, *case_data))
                    finally:
                        gpu_pool.append(gpu_id)  # Return GPU to pool after completion
                    break
                else:
                    time.sleep(1)  # Wait if all GPUs are busy

        # Create a process pool to handle multiple cases
        with Pool(processes=gpu_count) as pool:
            pool.map(task_manager, all_cases)
