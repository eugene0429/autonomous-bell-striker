"""
Extract frames from recorded video
Generate images for YOLO training

Usage:
  python extract_frames.py --video dataset/videos/video.mp4
  python extract_frames.py --video dataset/videos/video.mp4 --interval 0.5
  python extract_frames.py --video dataset/videos/video.mp4 --interval 0 --every 10
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from config import PATHS


def extract_frames(video_path, output_dir=None, interval=1.0, every_n=None, prefix="frame"):
    """
    Extract frames from a video

    Args:
        video_path: Input video path
        output_dir: Output directory (default: dataset/images)
        interval: Extraction interval (seconds). Ignored if every_n is specified
        every_n: Extract every N frames. Uses interval if None
        prefix: Filename prefix
    """
    if output_dir is None:
        output_dir = PATHS["images"]
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"[INFO] Video info:")
    print(f"  → File: {video_path}")
    print(f"  → FPS: {fps:.1f}")
    print(f"  → Total frames: {total_frames}")
    print(f"  → Duration: {duration:.1f}s")
    print()

    if every_n:
        frame_interval = every_n
        print(f"[INFO] Extracting every {every_n} frames")
    else:
        frame_interval = max(1, int(fps * interval))
        print(f"[INFO] Extracting at {interval}s intervals (every {frame_interval} frames)")

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{prefix}_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            # Progress display
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"\r[EXTRACT] {progress:5.1f}% | {saved_count} frames extracted", end="", flush=True)

        frame_idx += 1

    cap.release()
    print(f"\n\n[DONE] Extracted {saved_count} frame(s).")
    print(f"  → Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract YOLO training frames from recorded video")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--output", default=None, help="Output directory (default: dataset/images)")
    parser.add_argument("--interval", type=float, default=1.0, help="Extraction interval (seconds). Default: 1.0")
    parser.add_argument("--every", type=int, default=None, help="Extract every N frames. Overrides interval if specified")
    parser.add_argument("--prefix", default="frame", help="Filename prefix. Default: frame")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval, args.every, args.prefix)
