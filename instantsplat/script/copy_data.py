import os
import shutil
from PIL import Image

def sorted_files_by_int(directory):
    files = os.listdir(directory)
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return files

def resize_scale_and_crop(image, target_size):
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    if image.width < target_size[0] or image.height < target_size[1]:
        return image.resize(target_size, Image.LANCZOS)

    if target_ratio > img_ratio:
        image = image.resize((target_size[0], int(target_size[0] / img_ratio)), Image.LANCZOS)
    else:
        image = image.resize((int(target_size[1] * img_ratio), target_size[1]), Image.LANCZOS)

    width, height = image.size
    left = (width - target_size[0]) / 2
    top = (height - target_size[1]) / 2
    right = (width + target_size[0]) / 2
    bottom = (height + target_size[1]) / 2

    return image.crop((left, top, right, bottom))

def copy_and_resize_files(source_folder, frame_folder, target_folder, resize_size):
    os.makedirs(target_folder, exist_ok=True)

    frame_files = sorted_files_by_int(frame_folder)
    source_files = sorted_files_by_int(source_folder)

    if len(frame_files) < 2 or len(source_files) < 2:
        raise ValueError("File not enough")

    target_index = 1
    for i in range(len(frame_files) - 1):
        src_file = os.path.join(frame_folder, frame_files[i])
        dest_file = os.path.join(target_folder, f"{target_index}.png")
        shutil.copy2(src_file, dest_file)
        target_index += 1

    for i in range(1, len(source_files) - 1):
        src_file = os.path.join(source_folder, source_files[i])
        dest_file = os.path.join(target_folder, f"{target_index}.png")
        img = Image.open(src_file)
        img_resized = resize_scale_and_crop(img, resize_size)
        img_resized.save(dest_file)
        target_index += 1

    last_frame_file = os.path.join(frame_folder, frame_files[-1])
    dest_file = os.path.join(target_folder, f"{target_index}.png")
    shutil.copy2(last_frame_file, dest_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy and resize files.")
    parser.add_argument("--scene_type", type=str, help="Scene type")
    parser.add_argument("--scene_name", type=str, help="Scene name")
    parser.add_argument("--video_type", type=str, help="Video type")
    parser.add_argument("--frame_interval", type=int, help="Frame interval")
    args = parser.parse_args()

    source_folder = f'data/images/test_case/{args.scene_type}/{args.scene_name}'
    frame_folder = f'data/images/{args.video_type}/{args.scene_type}/{args.scene_name}/{args.scene_name}_frame_{args.frame_interval}'
    target_folder = f'data/images/{args.video_type}/{args.scene_type}/{args.scene_name}/{args.scene_name}'
    if args.video_type == 'luma':
        resize_size = (1360, 752)
    elif args.video_type == 'dynamicrafter':
        resize_size = (1280, 720)

    copy_and_resize_files(source_folder, frame_folder, target_folder, resize_size)