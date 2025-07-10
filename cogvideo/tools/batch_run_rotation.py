import subprocess

# List of configurations
configs = [
    "inference_pan_left.yaml",
    "inference_pan_right.yaml",
    "inference_pan_down.yaml",
    "inference_pan_up.yaml",
    "inference_pan_rotation_left.yaml",
    "inference_pan_rotation_right.yaml",
    
]

# Base bash command template
bash_command_template = """\
#! /bin/bash

export CUDA_VISIBLE_DEVICES={gpu_id}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video.py --base configs/cogvideox_5b_i2v_lora.yaml configs/{config_file} --seed 42 --image2video"

echo $run_cmd
eval $run_cmd

echo "DONE on `hostname`"
"""

# Function to run a command in a new subprocess
def run_command(config_file, gpu_id):
    # Format the script with the GPU ID and config file
    script = bash_command_template.format(gpu_id=gpu_id, config_file=config_file)
    # Run the script using /bin/bash
    process = subprocess.Popen(script, shell=True, executable='/bin/bash')
    return process

# Run all configurations in parallel on 6 GPUs
processes = []
for gpu_id, config in enumerate(configs):
    processes.append(run_command(config, gpu_id))

# Wait for all processes to complete
for process in processes:
    process.wait()

print("All commands have been executed.")
