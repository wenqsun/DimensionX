import os
from PIL import Image
import argparse

def reduce_resolution(image_path):
    """
    将指定图像的分辨率降低到原始分辨率的一半，并保存覆盖原图。
    
    参数:
    - image_path (str): 图像文件路径。
    """
    with Image.open(image_path) as img:
        # 获取图像的当前分辨率
        original_width, original_height = img.size
        # 计算新的分辨率
        new_width = original_width // 2
        new_height = original_height // 2
        # 调整图像分辨率
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        # 保存图像覆盖原文件
        img_resized.save(image_path)
        print(f"Reduced resolution of {image_path} to {new_width}x{new_height}")

def process_images_in_folder(folder_path):
    """
    遍历文件夹中的所有图像，将其分辨率降低到原始分辨率的一半。
    
    参数:
    - folder_path (str): 要处理的文件夹路径。
    """
    # 遍历文件夹中的所有文件和子文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为图像格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')):
                image_path = os.path.join(root, file)
                reduce_resolution(image_path)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Reduce resolution of all images in a folder by half.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images")
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用处理函数
    process_images_in_folder(args.folder_path)
