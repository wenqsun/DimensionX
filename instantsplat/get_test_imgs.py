import os
import argparse
from PIL import Image

def central_crop_and_resize(img, target_size=(720, 480)):
    """
    对图像进行居中裁剪并调整大小。
    
    参数:
    - img (PIL.Image): 输入的PIL图像。
    - target_size (tuple): 最终输出的尺寸 (宽, 高)，默认值为 (720, 480)。
    
    返回:
    - PIL.Image: 裁剪并调整大小后的图像。
    """
    # 计算目标宽高比
    target_ratio = target_size[0] / target_size[1]
    width, height = img.size
    original_ratio = width / height

    # 根据长宽比进行裁剪
    if original_ratio > target_ratio:
        # 图像太宽，裁剪宽度
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        img = img.crop((left, 0, left + new_width, height))
    else:
        # 图像太高，裁剪高度
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        img = img.crop((0, top, width, top + new_height))

    # 调整大小到目标尺寸
    img_resized = img.resize(target_size, Image.LANCZOS)  # 使用LANCZOS代替ANTIALIAS
    return img_resized

def resize_images_in_folder(folder_path, target_dir, target_size=(1024, 576)):
    """
    将指定文件夹中的所有图片进行居中裁剪并调整大小。

    参数:
    - folder_path (str): 图片所在的文件夹路径。
    - target_size (tuple): 调整后的图片大小，格式为(宽, 高)，默认值为 (720, 480)。
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"路径 {folder_path} 不存在。")
        return

    # 遍历文件夹中的所有文件
    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        # 检查是否为图片文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.JPG')):
            # 打开图像，进行居中裁剪并调整大小
            with Image.open(file_path) as img:
                img_resized = central_crop_and_resize(img, target_size)
                
                # 保存调整后的图片，覆盖原文件
                num = i + 39
                new_path = os.path.join(target_dir, f'{num}.png')
                img_resized.save(new_path)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Central crop and resize images in a folder to 720x480.")
    parser.add_argument("dataset", type=str)
    parser.add_argument("case_name", type=str)
    parser.add_argument("method", type=str)
    parser.add_argument("target_dir", type=str)

    test_base_dir = '/pfs/mt-1oY5F7/chenshuo/sparse_test/test_case'
    # 解析参数
    args = parser.parse_args()
    test_case_path = os.path.join(test_base_dir, args.dataset, args.case_name, 'test')
    resolution = {'ours': (720, 480), 'viewcrafter':(1024, 576), 'instantsplat':(960, 540), 'zeronvs':(256, 256)}
    target_size = resolution[args.method]

    # 调用居中裁剪和调整大小的函数
    resize_images_in_folder(test_case_path, args.target_dir, target_size)


