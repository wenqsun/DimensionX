import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# Add the path to the directory containing the dust3r module to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dust3r'))
# sys.path.append('../dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussian-splatting'))
import os
from typing import NamedTuple, Optional
import cv2  # Assuming OpenCV is used for image saving
from scene.gaussian_model import BasicPointCloud
from PIL import Image
from scene.colmap_loader import rotmat2qvec
from utils.graphics_utils import focal2fov, fov2focal
from scene.dataset_readers import storePly
import trimesh
from scipy.spatial.transform import Rotation


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int
    mask: Optional[np.ndarray] = None
    mono_depth: Optional[np.ndarray] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None
    
def init_filestructure(save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'
    
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)    
    sparse_path.mkdir(exist_ok=True, parents=True)
    
    return save_path, images_path, masks_path, sparse_path

def save_images_masks(imgs, masks, images_path, masks_path):
    # Saving images and optionally masks/depth maps
    for i, (image, mask) in enumerate(zip(imgs, masks)):
        image_save_path = images_path / f"{i}.png"
        
        mask_save_path = masks_path / f"{i}.png"
        # image[~mask] = 1.
        rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2)*255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)
        
        
def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    # Save cameras.txt
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            cameras_file.write(f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")
            
def save_imagestxt(world2cam, sparse_path):
     # Save images.txt
    images_file = sparse_path / 'images.txt'
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            # Convert rotation matrix to quaternion
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {i}.png\n")
            images_file.write("\n") # Placeholder for points, assuming no points are associated with images here

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path):
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / 'points3D.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))
            

def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)          # [N_imgs, h, w, 3]
    pts3d = to_numpy(pts3d)        # length: 3, each element: numpy.array [272, 512, 3]
    mask = to_numpy(mask)
    
    # only take the first and last element
    imgs = imgs[[0, -13]]
    pts3d = [pts3d[i] for i in [0, -13]]
    mask = [mask[i] for i in [0, -13]]

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])        # concate each pointmap together, [N-img * h * w, 3]
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])      # concate each image together, [N-img * h * w, 3]
    
    # original writing
    # pts = pts.reshape(-1, 3)[::3]           # 每隔3个取一个作为初始点云
    # col = col.reshape(-1, 3)[::3]

    # original writing
    pts = pts.reshape(-1, 3)           # 取所有点云
    col = col.reshape(-1, 3)

    print("-----------------------The shape of pts and col-----------------------")
    print(pts.shape)
    print(col.shape)

    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))    
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts

def save_pointcloud(imgs, pts3d, msk, sparse_path):
    save_path = sparse_path / 'points3D.ply'
    pc = get_pc(imgs, pts3d, msk)
    
    pc.export(save_path)


# convert the pointmap to .glb file
def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        print("The type and shape of imgs[i] is: ", type(imgs[i]), imgs[i].shape)
        print("The type and shape of focals[i] is: ", type(focals[i]), focals[i].shape)
        print("The type and shape of camera pose is: ", type(pose_c2w), pose_c2w.shape)
        print("The max and min value of imgs[i] is: ", imgs[i].max(), imgs[i].min())
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--dataset', type=str, default='bedroom_wo_video', help='dataset to use')
    parser.add_argument('--load_test', action='store_true', help='load test data')
    args = parser.parse_args()
    
    device = 'cuda:0'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # Load model
    model_path = "../dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = load_model(model_path, device)

    # Load images
    Path.ls = lambda x: list(x.iterdir())
    image_dir = Path(f'data/images/{args.dataset}/')
    image_files = [str(x) for x in image_dir.ls() if x.suffix in ['.png', '.jpg', '.JPG']]
    image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    print(image_files)
    
    images = load_images(image_files, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)  # get the pointmap
    # import ipdb;ipdb.set_trace()

    min_conf_thr = 5
    # get the camera pose, pointmap, and the loss
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.min_conf_thr = scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # construct the colmap dataset
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()
    world2cam = inv(scene.get_im_poses().detach()).cpu().numpy()
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    imgs = np.array(scene.imgs)
    pts3d = [i.detach() for i in scene.get_pts3d()]      # [272, 512, 3]
    depth_maps = [i.detach() for i in scene.get_depthmaps()]   # [272, 512] * 3

    masks = to_numpy(scene.get_masks())

    save_dir = Path(f'data/scenes/{args.dataset}')
    save_dir.mkdir(exist_ok=True, parents=True)

    save_path, images_path, masks_path, sparse_path = init_filestructure(save_dir)
    save_images_masks(imgs, masks, images_path, masks_path)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=imgs.shape)
    save_imagestxt(world2cam, sparse_path)
    # save_pointcloud(imgs, pts3d, masks, sparse_path)
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)

    # get the confidence map for the images, and save it corresponding to the images
    os.makedirs(f'data/scenes/{args.dataset}/confidence_map', exist_ok=True)
    conf_maps = scene.get_conf()       # shape: [N_img, h, w]
    for i, (conf_map, img) in enumerate(zip(conf_maps, imgs)):
        conf_map = to_numpy(conf_map)
        print(conf_map.shape)
        conf_map = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min())
        plt.imsave(f'data/scenes/{args.dataset}/confidence_map/{i}_conf.png', conf_map)

    # save the depth maps
    os.makedirs(f'data/scenes/{args.dataset}/depth_maps', exist_ok=True)
    for i, depth_map in enumerate(depth_maps):
        depth_map = to_numpy(depth_map)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        plt.imsave(f'data/scenes/{args.dataset}/depth_maps/{i}.png', depth_map)

    # save the pointmap, ie, pts3d, for each pts3d, the shape is [272, 512, 3]
    os.makedirs(f'data/scenes/{args.dataset}/pointmaps', exist_ok=True)
    for i, pts in enumerate(pts3d):
        pts = to_numpy(pts)
        pts = (pts - pts.min()) / (pts.max() - pts.min())
        plt.imsave(f'data/scenes/{args.dataset}/pointmaps/{i}.png', pts)

    # save the .glb file
    outfile = get_3D_model_from_scene(save_dir, silent=False, scene=scene, min_conf_thr=min_conf_thr, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05)


