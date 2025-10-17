import cv2
import os
import argparse
import numpy as np

def extract_frames_from_video(video_path, output_dir, num_frames, frame_prefix='frame_'):
    """
    Extracts a specified number of evenly sampled frames from a video file and saves them as images.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted frames.
        num_frames (int): Number of frames to extract evenly.
        frame_prefix (str): Prefix for the frame filenames (default: 'frame_').
        
    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate indices of frames to sample
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Check if the current frame is one of the sampled frames
        if frame_count in frame_indices:
            frame_filename = os.path.join(output_dir, f"{extracted_count}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Extracted frame {frame_count} -> {frame_filename}")
            extracted_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Extraction complete. Total frames extracted: {extracted_count}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory where extracted frames will be saved.")
    parser.add_argument("num_frames", type=int, help="Number of frames to sample from the video.")

    # Parse arguments
    args = parser.parse_args()

    # Call the frame extraction function
    extract_frames_from_video(args.video_path, args.output_dir, args.num_frames)
