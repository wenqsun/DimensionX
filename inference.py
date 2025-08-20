import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import argparse
import os

# --- Start of Changes ---

# 1. Set up a parser for command-line arguments
parser = argparse.ArgumentParser(description="Run CogVideoX inference with a LoRA.")
parser.add_argument(
    "--lora_path",
    type=str,
    required=True,
    help="Full path to the LoRA .safetensors file (e.g., './motion_lora_weights.safetensors')."
)
parser.add_argument(
    "--prompt",
    type=str,
    default="An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
    help="The text prompt to guide the video generation."
)
parser.add_argument(
    "--image_url",
    type=str,
    default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
    help="URL of the input image."
)
args = parser.parse_args()

# 2. Automatically separate the directory and filename from the provided path
# This allows the load_lora_weights function to work correctly.
lora_directory = os.path.dirname(args.lora_path)
lora_filename = os.path.basename(args.lora_path)

# --- End of Changes ---

# Load the main pipeline
print("Loading pipeline...")
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)

# Load the LoRA weights using the parsed path
print(f"Loading LoRA from: {args.lora_path}")
lora_rank = 256
pipe.load_lora_weights(lora_directory, weight_name=lora_filename, adapter_name="test")
pipe.fuse_lora(lora_scale=1 / lora_rank)
pipe.to("cuda")


# Use the prompt and image from the command-line arguments
prompt = args.prompt
image = load_image(args.image_url)

# Run the inference process
print("Generating video...")
video = pipe(image, prompt, use_dynamic_cfg=True)

# Export the final video
print("Exporting video to output.mp4...")
export_to_video(video.frames[0], "output.mp4", fps=8)

print("Done!")
