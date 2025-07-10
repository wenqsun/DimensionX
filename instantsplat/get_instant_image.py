import cv2
import os
import argparse
import numpy as np

import os
import shutil

def copy_images_to_output_dir(image_path0, image_path1, output_dir):
    """
    Copies two images to a specified output directory.
    
    Args:
        image_path0 (str): Path to the first image.
        image_path1 (str): Path to the second image.
        output_dir (str): Directory to save the copied images.
        image_prefix (str): Prefix for the copied image filenames (default: 'image_').
        
    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy first image
    shutil.copy(image_path0, output_dir)
    print(f"Copied {image_path0} to {output_dir}")
    
    # Copy second image
    shutil.copy(image_path1, output_dir)
    print(f"Copied {image_path1} to {output_dir}")

    print("Copying complete.")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("image_path0", type=str, help="Path to the input video file.")
    parser.add_argument("image_path1", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory where extracted frames will be saved.")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the frame extraction function
    copy_images_to_output_dir(args.image_path0, args.image_path1, args.output_dir)
