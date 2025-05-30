import cv2
import os
from typing import Optional

def extract_frames_from_video(video_path: str, output_dir: str, prefix: str = "frame") -> None:
    """
    Extract all frames from a video and save them as jpg images in the output directory.
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted jpg images.
        prefix (str): Prefix for saved image filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{frame_idx}.jpg")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")

def resize_and_crop_images_to_square(input_dir: str, output_dir: str, x: int = 512, offset_x: int = 0, offset_y: int = 0) -> None:
    """
    Resize and center-crop all images in input_dir to x by x square, saving to output_dir.
    The shorter side is resized to x, and the longer side is center-cropped to x.
    Optionally, the crop can be offset from the center by offset_x and offset_y.
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save processed images.
        x (int): Target size for both width and height.
        offset_x (int): Horizontal offset from center for cropping.
        offset_y (int): Vertical offset from center for cropping.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        # Determine scale and resize
        if w < h:
            scale = x / w
            new_w = x
            new_h = int(h * scale)
        else:
            scale = x / h
            new_h = x
            new_w = int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        # Center crop with offset
        start_x = (new_w - x) // 2 + offset_x
        start_y = (new_h - x) // 2 + offset_y
        # Ensure crop is within bounds
        start_x = max(0, min(start_x, new_w - x))
        start_y = max(0, min(start_y, new_h - x))
        img_cropped = img_resized[start_y:start_y + x, start_x:start_x + x]
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img_cropped)
    print(f"Processed images to {output_dir} with size {x}x{x} and offset ({offset_x}, {offset_y})")

if __name__ == "__main__":
    # Example usage for id6/evie.mp4
    video_path = "data/id6/video/evie.mp4"
    output_dir = "data/id6/ori_imgs"
    extract_frames_from_video(video_path, output_dir)
    resize_and_crop_images_to_square(output_dir, output_dir, x=512, offset_y=-50) 