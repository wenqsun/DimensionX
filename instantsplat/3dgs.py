import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussian-splatting'))
import torch
from random import randint
from dataclasses import dataclass, field
import os
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from PIL import Image
from torchvision import transforms
# save video use mediapy
import mediapy as media

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm.auto import tqdm
from utils.image_utils import psnr
from train import prepare_output_and_logger, training_report

from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix
from utils.camera_utils import visualize_camera_trajectory
import torchvision
import subprocess
import cv2
import json

# add lpips loss function
from lpips import LPIPS


# add a new l1 loss function with confidence maps
def conf_l1_loss(network_output, gt, confidence_map):
    # confidence_map
    return torch.abs((network_output - gt) * confidence_map).mean()

def _conf_ssim(img1, img2, confidence_map, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1 * confidence_map, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2 * confidence_map, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1 * confidence_map, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2 * confidence_map, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2 * confidence_map, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# add a new ssim_loss function with confidence maps
def conf_ssim(img1, img2, confidence_map, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _conf_ssim(img1, img2, confidence_map, window, window_size, channel, size_average)




@dataclass
class ModelParams:
    sh_degree: int = 3
    source_path: str = "../data/scenes/turtle"
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = True
    data_device: str = "cuda"
    eval: bool = False

    def post_init(self):
        self.source_path = os.path.abspath(self.source_path)

@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

@dataclass
class OptimizationParams:
    iterations: int = 30000 #30_000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 500
    densify_until_iter = 15_000
    densify_grad_threshold = 0.0002
    random_background = False

@dataclass
class TrainingArgs:
    ip: str = "0.0.0.0"
    port: int = 6007
    debug_from: int = -1
    detect_anomaly: bool = False
    test_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_iterations: list[int] = field(default_factory=lambda: [1_000, 2_000, 3_000, 5_000, 7_000, 1_0000, 30_000])
    quiet: bool = False
    checkpoint_iterations: list[int] = field(default_factory=lambda: [1_000, 7_000, 15_000, 30_000])
    start_checkpoint: str = None

def images_to_video_cv2(image_folder, output_video, frame_rate=30):
    """
    使用cv2将文件夹中的图片无损转换为视频

    Args:
    - image_folder: 图片文件夹路径
    - output_video: 输出视频文件路径
    - frame_rate: 视频帧率
    """
    # 获取图片文件列表，并确保按文件名排序
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    
    if not images:
        raise ValueError("图片文件夹中没有找到任何图片文件。")
    
    # 获取图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    img = cv2.imread(first_image_path)
    height, width, layers = img.shape

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)
    
    video.release()
    cv2.destroyAllWindows()

# image to video using media
def images_to_video_media(image_folder, output_video, frame_rate=30):
    # List all image files in the folder
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Read images
    images = [media.read_image(image_file) for image_file in image_files]

    # Write video
    media.write_video(output_video, images, fps=frame_rate)




# 焦距转换为视场角
def focal2fov(focal_length, dimension_size):
    return 2 * np.arctan(dimension_size / (2 * focal_length))

def json_to_camera_info(camera_json):
    uid = camera_json['id']
    image_name = camera_json['img_name']
    width = camera_json['width']
    height = camera_json['height']
    pos = np.array(camera_json['position'])
    rot = np.array(camera_json['rotation'])
    fy = camera_json['fy']
    fx = camera_json['fx']
    
    # 将相机位置和旋转矩阵转换回R和T
    W2C = np.eye(4)
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    C2W = np.linalg.inv(W2C)
    R = C2W[:3, :3].transpose()
    T = C2W[:3, 3]

    # 计算视场角
    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)

    return uid, R, T, FovX, FovY, image_name, width, height


@torch.no_grad()
def render_path(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_resize_method='crop', use_render_config=None):
    """
    render_resize_method: crop, pad
    """
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    iteration = scene.loaded_iter

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    model_path = dataset.model_path
    name = "render"

    views = scene.getRenderCameras()

    # get the original train view
    train_views = scene.getTrainCameras()

    # let views = train_views
    # views = train_views

    if use_render_config is not None:
        print('-------Load the render config-------')
        # load json file
        with open(use_render_config, 'r') as f:
            camera_data_list = json.load(f)
            gt_train_views = camera_data_list[:3]
            gt_views = camera_data_list[3:]
        # take the train_views with the length of gt_train_views
        train_views = train_views[:len(gt_train_views)]
        for idx, camera_data in enumerate(gt_train_views):
            uid, R, T, FoVx, FoVy, image_name, width, height = json_to_camera_info(camera_data)
            train_views[idx].uid = uid
            train_views[idx].R = np.array(R)
            train_views[idx].T = np.array(T)
            train_views[idx].FoVx = np.array(FoVx)
            train_views[idx].FoVy = np.array(FoVy)
            # train_views[idx].image_name = image_name
            train_views[idx].image_width = width
            train_views[idx].image_height = height
        for idx, camera_data in enumerate(gt_views):
            uid, R, T, FoVx, FoVy, image_name, width, height = json_to_camera_info(camera_data)
            # views[idx].uid = uid
            views[idx].R = np.array(R)
            views[idx].T = np.array(T)
            views[idx].FoVx = np.array(FoVx)
            views[idx].FoVy = np.array(FoVy)
            # views[idx].image_name = image_name
            views[idx].image_width = width
            views[idx].image_height = height

    # print(len(views))
    train_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "train_renders")
    os.makedirs(train_render_path, exist_ok=True)

    # visualize the camera trajectory
    visualize_camera_trajectory(train_views, views, os.path.join(model_path, name, "ours_{}".format(iteration), "camera_trajectory.png"))

    # render the scene with the training views
    for idx, view in enumerate(tqdm(train_views, desc="Rendering progress for training views")):
        if render_resize_method == 'crop':
            # image_size = 512
            image_width = 512
            image_height = 512
        elif render_resize_method == 'pad':
            # image_size = max(view.image_width, view.image_height)
            image_width = max(view.image_width, view.image_height)
            image_height = max(view.image_width, view.image_height)
        elif render_resize_method == 'original':
            image_width = view.image_width
            image_height = view.image_height
        else:
            raise NotImplementedError
        view.original_image = torch.zeros((3, image_height, image_width), device=view.original_image.device)
        # view.original_image = torch.zeros((3, image_size, image_size), device=view.original_image.device)
        focal_length_x = fov2focal(view.FoVx, view.image_width)
        focal_length_y = fov2focal(view.FoVy, view.image_height)
        view.image_width = image_width
        view.image_height = image_height
        view.FoVx = focal2fov(focal_length_x, image_width)
        view.FoVy = focal2fov(focal_length_y, image_height)
        view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)

        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        img_path = os.path.join(train_render_path, '{0:05d}'.format(idx) + ".png") if ".png" in view.image_name else os.path.join(train_render_path, '{0:05d}'.format(int(view.image_name)) + ".png") 
        torchvision.utils.save_image(rendering, img_path)
        
    images_to_video_cv2(train_render_path, os.path.join(train_render_path, "train_renders.mp4"), frame_rate=30)

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    os.makedirs(render_path, exist_ok=True)
    # render the scene with the interpolation views
    for idx, view in enumerate(tqdm(views, desc="Rendering progress for interpolation views")):
        if render_resize_method == 'crop':
            # image_size = 512
            image_width = 512
            image_height = 512
        elif render_resize_method == 'pad':
            # image_size = max(view.image_width, view.image_height)
            image_width = max(view.image_width, view.image_height)
            image_height = max(view.image_width, view.image_height)
        elif render_resize_method == 'original':
            image_width = view.image_width
            image_height = view.image_height
        else:
            raise NotImplementedError
        view.original_image = torch.zeros((3, image_height, image_width), device=view.original_image.device)
        # view.original_image = torch.zeros((3, image_size, image_size), device=view.original_image.device)
        focal_length_x = fov2focal(view.FoVx, view.image_width)
        focal_length_y = fov2focal(view.FoVy, view.image_height)
        view.image_width = image_width
        view.image_height = image_height
        view.FoVx = focal2fov(focal_length_x, image_width)
        view.FoVy = focal2fov(focal_length_y, image_height)
        view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)

        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        img_path = os.path.join(render_path, '{0:05d}'.format(idx) + ".png") if ".png" in view.image_name else os.path.join(render_path, '{0:05d}'.format(int(view.image_name)) + ".png") 
        torchvision.utils.save_image(rendering, img_path)

    # Use ffmpeg to output video
    renders_path = os.path.join(render_path, "interpolation_renders.mp4")

    print(render_path)
    print(renders_path)
    images_to_video_cv2(render_path, renders_path, frame_rate=30)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--dataset', type=str, default='bedroom_wo_video', help='dataset to use')
    parser.add_argument('--iter', type=int, default=1000, help='iteration to render')
    parser.add_argument('--lambda_lpips', type=float, default=0.0, help='lambda for lpips loss')
    parser.add_argument('--use_render_config', type=str, default=None, help='render config to use')
    parser.add_argument('--use_confidence', action='store_true', help='use confidence map to when training')
    args = parser.parse_args()

    # define model and pipeline
    dataset = ModelParams(source_path=f"data/scenes/{args.dataset}", 
                          model_path=f"data/scenes/{args.dataset}/output_{args.iter}_lpips_{args.lambda_lpips}_use_conf/" if args.use_confidence else f"data/scenes/{args.dataset}/output_{args.iter}_lpips_{args.lambda_lpips}/")
    opt = OptimizationParams(iterations=args.iter)
    pipe = PipelineParams()
    train_args = TrainingArgs()

    # load the confidence map from dust3r to indicate the efficiency of the splatting
    confidence_path = os.path.join(dataset.source_path, "confidence_map")

    testing_iterations = train_args.test_iterations
    saving_iterations = train_args.save_iterations 
    checkpoint_iterations = train_args.checkpoint_iterations 
    checkpoint = train_args.start_checkpoint
    debug_from = train_args.debug_from

    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # introduce lpips loss function to alleivate the inconsistency
    lpips_loss = LPIPS(net='vgg').to("cuda")
    lpips_loss.requires_grad_(False)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss_lpips = lpips_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + args.lambda_lpips * loss_lpips

        # if we use the extra confidence map to mitigate the negative impacts caused by inconsistent images
        if args.use_confidence:
            # load the confidence map (.png file)
            # print(viewpoint_cam.image_name)
            confidence_map = Image.open(os.path.join(confidence_path, f'{viewpoint_cam.image_name}_conf.png'))   # shape
            to_tensor = transforms.ToTensor()
            confidence_map = to_tensor(confidence_map)       # shape: (4, h, w)
            confidence_map = confidence_map[0].unsqueeze(0)  # shape: (1, h, w)
            # print(confidence_map.shape)
            confidence_map = confidence_map.repeat(3, 1, 1).cuda()  # shape: (3, H, W)
            # print(confidence_map)
            Ll1 = conf_l1_loss(image, gt_image, confidence_map)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - conf_ssim(image, gt_image, confidence_map)) + args.lambda_lpips * loss_lpips

        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "L1": f"{Ll1.item():.{7}f}", "SSIM": f"{1.0 - ssim(image, gt_image).item():.{7}f}", "LPIPS": f"{loss_lpips.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # render the scene
    render_path(dataset, 1000, pipe, render_resize_method='original', use_render_config=args.use_render_config)
    render_path(dataset, args.iter, pipe, render_resize_method='original', use_render_config=args.use_render_config)
    if args.iter == 30000:
        render_path(dataset, 30000, pipe, render_resize_method='original', use_render_config=args.use_render_config)



