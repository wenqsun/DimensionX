# Wild Gaussian Splatting

This project merges [DUSt3R](https://github.com/nerlfield/dust3r)'s capabilities in camera parameter estimation and point cloud creation with the 3D scene representation efficiency of [Gaussian Splatting](https://github.com/nerlfield/gaussian-splatting). The goal is to simplify the process of 3D scene reconstruction and visualization from images without requiring pre-set camera information or specific viewpoint data.

<video loop="loop" autoplay="autoplay" muted>
  <source src="data/assets/results.mp4.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHVxYjRsZXd3dHlrZnljNnVvaWx5cDdyNjJmMjc0YmhpdmppcGp1cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jBxJfbzw9NqUeASrOo/giphy-downsized-large.gif"/>


## Cloning the Repository

To begin, clone the repository and initialize its submodules:

```sh
git clone git@github.com:nerlfield/wild-gaussian-splatting.git --recursive
cd wild-gaussian-splatting
git pull --recurse-submodules
git submodule update --init --recursive
```

## Create conda environment:

```sh
conda create -n wildgaussians python=3.11 cmake=3.14.0 -y
conda activate wildgaussians
```

## Setting Up the Environment:

With the environment activated, run the provided setup script and pass your CUDA version as an argument. This script installs necessary dependencies tailored to your CUDA version:

```sh
./setup_environment.sh <cuda_version>
```

Replace <cuda_version> with the version of CUDA you intend to use (e.g., 10.2, 11.1, 12.1). I used `12.1`.

## Starting Jupyter Lab:

This repository contains two notebooks that showcase fitting gaussians over scene from the wild:
1. `./notebooks/00_dust3r_inference.ipynb` - Runs DUSt3R on a folder of images, saving camera parameters and point clouds in COLMAP format. Note: DUSt3R's memory usage increases quadratically with the number of images. On an L4 instance, it can process up to 32 images of 512x384 size.
1. `./notebooks/01_gaussian_splatting_fitting.ipynb` - Reads DUSt3R results and applies Gaussian splatting. Modifications include the addition of rendering camera trajectory generation and bug fixes.

To launch Jupyter Lab, use:

```sh
jupyter lab --no-browser --ip 0.0.0.0 --port 4546 --allow-root --notebook-dir=.
```
