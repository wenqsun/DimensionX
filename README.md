# DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion

[**Paper**](https://arxiv.org/abs/2411.04928) | [**Project Page**](https://chenshuo20.github.io/DimensionX/) | [**Video**](https://youtu.be/ViDQI1HMY2U?si=f1RGd82n6yj6TOFB) | [**🤗 HF Demo**](https://huggingface.co/spaces/fffiloni/DimensionX)

Official implementation of DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion

[Wenqiang Sun*](https://github.com/wenqsun), [Shuo Chen*](https://chenshuo20.github.io/), [Fangfu Liu*](https://liuff19.github.io/), [Zilong Chen](https://scholar.google.com/citations?user=2pbka1gAAAAJ), [Yueqi Duan](https://duanyueqi.github.io/), [Jun Zhang](https://eejzhang.people.ust.hk/), [Yikai Wang](https://yikaiw.github.io/)

Abstract: *In this paper, we introduce DimensionX, a framework designed to generate photorealistic 3D and 4D scenes from just a single image with video diffusion. Our approach begins with the insight that both the spatial structure of a 3D scene and the temporal evolution of a 4D scene can be effectively represented through sequences of video frames. While recent video diffusion models have shown remarkable success in producing vivid visuals, they face limitations in directly recovering 3D/4D scenes due to limited spatial and temporal controllability during generation. To overcome this, we propose ST-Director, which decouples spatial and temporal factors in video diffusion by learning dimension-aware LoRAs from dimension-variant data. This controllable video diffusion approach enables precise manipulation of spatial structure and temporal dynamics, allowing us to reconstruct both 3D and 4D representations from sequential frames with the combination of spatial and temporal dimensions. Additionally, to bridge the gap between generated videos and real-world scenes, we introduce a trajectory-aware mechanism for 3D generation and an identity-preserving denoising strategy for 4D generation. Extensive experiments on various real-world and synthetic datasets demonstrate that DimensionX achieves superior results in controllable video generation, as well as in 3D and 4D scene generation, compared with previous methods.*
<p align="center">
    <img src="assets/file/teaser.png">
</p>

## Project Updates

- 🔥🔥 News: ```2024/11/15```: The Hugging Face online demo is now available! You can try it [here](https://huggingface.co/spaces/fffiloni/DimensionX). Thanks to [fffiloni](https://huggingface.co/fffiloni) for building it!

- 🔥🔥 News: ```2024/11/12```: We have released the Orbit Left and Orbit Up S-Director models. You can download them [here](https://huggingface.co/wenqsun/DimensionX).

## Todo List
- [x] Release part of model checkpoints (S-Director): orbit left & orbit up.
- [ ] Release all model checkpoints.
    - [ ] The rest S-Directors
    - [ ] T-Director
    - [ ] Long video generation model (145 frames)
    - [ ] Video interpolation model (training code + checkpoint)
- [ ] 3dgs optimization code
- [ ] Identity-preserving denoising code for 4D generation
- [ ] Training dataset

## Model checkpoint

We have released part of our model checkpoint in Google drive and Huggingface (orbit left & orbit up): [ckpt_drive](https://drive.google.com/drive/folders/1X0tH3JQke1ZIa62jVoZlQWZR38PCi0Eg?usp=sharing) (Google Drive), [ckpt_huggingface](https://huggingface.co/wenqsun/DimensionX) (Huggingface), [ckpt_modelscope](https://modelscope.cn/models/ShuoChen/DimensionX/) (Modelscope)

We are still refining our model, more camera control checkpoints are coming!

## Inference code

We provide a gradio demo web UI for our model. Thanks to the gradio demo in [CogvideoX](https://github.com/THUDM/CogVideo), we implement our model in `src/gradio_demo/app.py`

### Installation

```
cd src/gradio_demo
pip install -r requirements.txt 
```
run the gradio demo
```
OPENAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=your_base_url python app.py
```

For better result, you'd better use VLM to caption the input image.

We also provide a script below:
```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
lora_path = "your lora path"
lora_rank = 256
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
pipe.fuse_lora(lora_scale=1 / lora_rank)
pipe.to("cuda")


prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
video = pipe(image, prompt, use_dynamic_cfg=True)
export_to_video(video.frames[0], "output.mp4", fps=8)
```

Using the above inference code and our provided pre-trained checkpoint, you can achieve the controllable video generation!



## Method
Our framework is mainly divided into three parts. (a) Controllable Video Generation with ST-Director. We introduce ST-Director to decompose the spatial and temporal parameters in video diffusion models by learning dimension-aware LoRA on our collected dimension-variant datasets.  (b) 3D Scene Generation with S-Director. Given one view, a high-quality 3D scene is recovered from the video frames generated by S-Director.  (c) 4D Scene Generation with ST-Director. Given a single image, a temporal-variant video is produced by T-Director, from which a key frame is selected to generate a spatial-variant reference video. Guided by the reference video, per-frame spatial-variant videos are generated by S-Director, which are then combined into multi-view videos. Through the multi-loop refinement of T-Director, consistent multi-view videos are then passed to optimize the 4D scene.

<p align="center">
    <img src="assets/file/pipeline.png">
</p>


## Notice
Due to the conflict of the LoRA conversion and fuse_lora function in diffusers, you may meet the issue below:

```python
File "/app/src/video_generator/__init__.py", line 7, in <module>
    model_genvid = CogVideo(configs)
                   ^^^^^^^^^^^^^^^^^
  File "/app/src/video_generator/cog/__init__.py", line 82, in __init__
    self.pipe.fuse_lora(adapter_names=["orbit_left"], lora_scale=1 / lora_rank)
  File "/usr/local/lib/python3.11/dist-packages/diffusers/loaders/lora_pipeline.py", line 2888, in fuse_lora
    super().fuse_lora(
  File "/usr/local/lib/python3.11/dist-packages/diffusers/loaders/lora_base.py", line 445, in fuse_lora
    raise ValueError(f"{fuse_component} is not found in {self._lora_loadable_modules=}.")
ValueError: text_encoder is not found in self._lora_loadable_modules=['transformer'].
```
you can solve this error by skipping that part:
```python
for fuse_component in components:
    if fuse_component == 'text_encoder':
    continue
```



## Acknowledgement
- [CogVideoX](https://github.com/THUDM/CogVideo)
- [ReconX](https://github.com/liuff19/ReconX)

From ReconX to DimensionX, we are conducting research about X! Our X Family is coming soon ...

## BibTeX

```bibtex
@misc{sun2024dimensionxcreate3d4d,
      title={DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion}, 
      author={Wenqiang Sun and Shuo Chen and Fangfu Liu and Zilong Chen and Yueqi Duan and Jun Zhang and Yikai Wang},
      year={2024},
      eprint={2411.04928},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.04928}, 
}
```
