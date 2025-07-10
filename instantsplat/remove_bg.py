from transformers import AutoModelForImageSegmentation
from PIL import Image
import os
import torch
from torchvision import transforms

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    # Apply mask to original image (making it RGBA)
    image = image.convert("RGBA")  # Ensure the image is in RGBA format
    image.putalpha(mask)  # Add transparency mask
    return image

def process_image(name):
    # Load the image segmentation model
    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()

    # Define directory paths
    dir_path = f'/pfs/mt-1oY5F7/chenshuo/instantsplat/data/images/{name}'
    save_dir = f'/pfs/mt-1oY5F7/chenshuo/instantsplat/data/images/'
    img_list = os.listdir(dir_path)

    for img_name in img_list:
        image_path = os.path.join(dir_path, img_name)

        # Apply segmentation to the input image, this returns a pillow image with transparency (RGBA)
        segmented_image = extract_object(birefnet, imagepath=image_path)

        # Define the output directory and ensure it exists
        output_dir = os.path.join(save_dir, f'{name}_wo_bg')
        os.makedirs(output_dir, exist_ok=True)

        # Save the resulting image to a file
        output_path = os.path.join(output_dir, img_name)
        segmented_image.save(output_path)

        print(f"Segmented image with transparency saved to {output_path}")

# Example usage
process_image('obj_dragon')
