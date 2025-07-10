#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussian-splatting'))

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

CROP_SIZE = (256, 256)

def readImages(renders_dir, gt_dir):
    """
    读取并处理图像，先按最大正方形中心裁剪，然后调整为 256x256 的分辨率。
    最后将图像转换为 Tensor。
    """
    renders = []
    gts = []
    image_names = []
    
    for fname in os.listdir(renders_dir):
        render_path = Path(renders_dir) / fname
        gt_path = Path(gt_dir) / fname

        # 打开图像
        render = Image.open(render_path)
        gt = Image.open(gt_path)

        # 获取最大正方形边长（取较小的边）
        min_side = min(render.size)
        
        # 先进行最大正方形中心裁剪
        render_cropped = tf.center_crop(render, min_side)
        gt_cropped = tf.center_crop(gt, min_side)
        
        # 调整为目标分辨率
        render_resized = render_cropped.resize(CROP_SIZE, Image.BILINEAR)
        gt_resized = gt_cropped.resize(CROP_SIZE, Image.BILINEAR)
        
        # 转换为 Tensor 并添加到列表
        renders.append(tf.to_tensor(render_resized).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt_resized).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    
    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "train"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                metrics_with_names = []

                # 计算每个图像的 SSIM、PSNR 和 LPIPS
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssim_value = ssim(renders[idx], gts[idx])
                    psnr_value = psnr(renders[idx], gts[idx])
                    lpips_value = lpips(renders[idx], gts[idx], net_type='vgg')

                    ssims.append(ssim_value)
                    psnrs.append(psnr_value)
                    lpipss.append(lpips_value)

                    # 将每个图像的三项指标和文件名存储
                    metrics_with_names.append({
                        "name": image_names[idx],
                        "SSIM": ssim_value.item(),
                        "PSNR": psnr_value.item(),
                        "LPIPS": lpips_value.item()
                    })

                # 按 PSNR 排序，选出前 6 个
                top_6_metrics = sorted(metrics_with_names, key=lambda x: x["PSNR"], reverse=True)[:6]

                # 计算前 6 个图像的平均值
                avg_ssim_top6 = sum([item["SSIM"] for item in top_6_metrics]) / 6
                avg_psnr_top6 = sum([item["PSNR"] for item in top_6_metrics]) / 6
                avg_lpips_top6 = sum([item["LPIPS"] for item in top_6_metrics]) / 6

                # 打印结果
                print(f"Top 6 average metrics for {method}:")
                print(f"  SSIM : {avg_ssim_top6:.7f}")
                print(f"  PSNR : {avg_psnr_top6:.7f}")
                print(f"  LPIPS: {avg_lpips_top6:.7f}")
                print("")

                # 更新字典保存结果
                full_dict[scene_dir][method].update({
                    "Top6_SSIM": avg_ssim_top6,
                    "Top6_PSNR": avg_psnr_top6,
                    "Top6_LPIPS": avg_lpips_top6
                })
                per_view_dict[scene_dir][method].update({
                    "Top6_Metrics": {
                        name["name"]: {
                            "SSIM": name["SSIM"],
                            "PSNR": name["PSNR"],
                            "LPIPS": name["LPIPS"]
                        }
                        for name in top_6_metrics
                    }
                })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
