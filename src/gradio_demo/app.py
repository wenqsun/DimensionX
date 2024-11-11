"""
THis is the main file for the gradio web demo. It uses the CogVideoX-5B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

Usage:
    OPENAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=your_base_url python app.py
"""

import math
import os
import random
import threading
import time

import cv2
import tempfile
import imageio_ffmpeg
import gradio as gr
import torch
from PIL import Image
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXVideoToVideoPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.utils import load_video, load_image
from datetime import datetime, timedelta

from diffusers.image_processor import VaeImageProcessor
from openai import OpenAI
import moviepy.editor as mp
import utils
from rife_model import load_rife_model, rife_inference_with_latents
from huggingface_hub import hf_hub_download, snapshot_download

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran")
snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to(device)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

pipe_image = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    transformer=CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", subfolder="transformer", torch_dtype=torch.bfloat16
    ),
    vae=pipe.vae,
    scheduler=pipe.scheduler,
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    torch_dtype=torch.bfloat16,
)
lora_path = "your_lora_path"
lora_rank = 256
pipe_image.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
pipe_image.fuse_lora(lora_scale=1 / lora_rank)
pipe_image = pipe_image.to(device)


# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# pipe_image.transformer.to(memory_format=torch.channels_last)
# pipe_image.transformer = torch.compile(pipe_image.transformer, mode="max-autotune", fullgraph=True)

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

upscale_model = utils.load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model("model_rife")

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""


def resize_if_unfit(input_video, progress=gr.Progress(track_tqdm=True)):
    width, height = get_video_dimensions(input_video)

    if width == 720 and height == 480:
        processed_video = input_video
    else:
        processed_video = center_crop_resize(input_video)
    return processed_video


def get_video_dimensions(input_video_path):
    reader = imageio_ffmpeg.read_frames(input_video_path)
    metadata = next(reader)
    return metadata["size"]


def center_crop_resize(input_video_path, target_width=720, target_height=480):
    cap = cv2.VideoCapture(input_video_path)

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width_factor = target_width / orig_width
    height_factor = target_height / orig_height
    resize_factor = max(width_factor, height_factor)

    inter_width = int(orig_width * resize_factor)
    inter_height = int(orig_height * resize_factor)

    target_fps = 8
    ideal_skip = max(0, math.ceil(orig_fps / target_fps) - 1)
    skip = min(5, ideal_skip)  # Cap at 5

    while (total_frames / (skip + 1)) < 49 and skip > 0:
        skip -= 1

    processed_frames = []
    frame_count = 0
    total_read = 0

    while frame_count < 49 and total_read < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if total_read % (skip + 1) == 0:
            resized = cv2.resize(frame, (inter_width, inter_height), interpolation=cv2.INTER_AREA)

            start_x = (inter_width - target_width) // 2
            start_y = (inter_height - target_height) // 2
            cropped = resized[start_y : start_y + target_height, start_x : start_x + target_width]

            processed_frames.append(cropped)
            frame_count += 1

        total_read += 1

    cap.release()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_video_path = temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (target_width, target_height))

        for frame in processed_frames:
            out.write(frame)

        out.release()

    return temp_video_path


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="glm-4-plus",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


def infer(
    prompt: str,
    image_input: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = -1,
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)

    # if video_input is not None:
    #     video = load_video(video_input)[:49]  # Limit to 49 frames
    #     video_pt = pipe_video(
    #         video=video,
    #         prompt=prompt,
    #         num_inference_steps=num_inference_steps,
    #         num_videos_per_prompt=1,
    #         strength=video_strenght,
    #         use_dynamic_cfg=True,
    #         output_type="pt",
    #         guidance_scale=guidance_scale,
    #         generator=torch.Generator(device="cpu").manual_seed(seed),
    #     ).frames
    if image_input is not None:
        image_input = Image.fromarray(image_input).resize(size=(720, 480))  # Convert to PIL
        image = load_image(image_input)
        video_pt = pipe_image(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames
    else:
        video_pt = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames

    return (video_pt, seed)


def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()
examples_images = [["example_images/beef.png"], ["example_images/candle.png"], ["example_images/person.png"]]

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               DimensionX Demo
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            </div>
           """)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("I2V: Image Input (cannot be used simultaneously with video input)", open=False):
                image_input = gr.Image(label="Input Image (will be cropped to 720 * 480)")
                examples_component_images = gr.Examples(examples_images, inputs=[image_input], cache_examples=False)
            # with gr.Accordion("V2V: Video Input (cannot be used simultaneously with image input)", open=False):
            #     video_input = gr.Video(label="Input Video (will be cropped to 49 frames, 6 seconds at 8fps)")
            #     strength = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Strength")
            #     examples_component_videos = gr.Examples(examples_videos, inputs=[video_input], cache_examples=False)
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")
            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=-1
                        )
                    with gr.Row():
                        enable_scale = gr.Checkbox(label="Super-Resolution (720 × 480 -> 2880 × 1920)", value=False)
                        enable_rife = gr.Checkbox(label="Frame Interpolation (8fps -> 16fps)", value=False)
                    gr.Markdown(
                        "✨In this demo, we use [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for frame interpolation and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling(Super-Resolution).<br>&nbsp;&nbsp;&nbsp;&nbsp;The entire process is based on open-source solutions."
                    )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    def generate(
        prompt,
        image_input,
        # video_input,
        # video_strength,
        seed_value,
        scale_status,
        rife_status,
        progress=gr.Progress(track_tqdm=True)
    ):
        latents, seed = infer(
            prompt,
            image_input,
            # video_input,
            # video_strength,
            num_inference_steps=50,  # NOT Changed
            guidance_scale=7.0,  # NOT Changed
            seed=seed_value,
            progress=progress,
        )
        if scale_status:
            latents = utils.upscale_batch_and_concatenate(upscale_model, latents, device)
        if rife_status:
            latents = rife_inference_with_latents(frame_interpolation_model, latents)

        batch_size = latents.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = latents[batch_idx]
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            batch_video_frames.append(image_pil)

        video_path = utils.save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6))
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)
        seed_update = gr.update(visible=True, value=seed)

        return video_path, video_update, gif_update, seed_update

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate,
        inputs=[prompt, image_input, seed_param, enable_scale, enable_rife],
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])
    # video_input.upload(resize_if_unfit, inputs=[video_input], outputs=[video_input])

if __name__ == "__main__":
    demo.queue(max_size=15)
    demo.launch()
