"""
RealSense D435i - YOLO Training Data Capture Tool
==================================================

Controls:
  [S] Manual capture  - Save current frame as image
  [A] Auto capture    - Toggle automatic capture at configured interval
  [R] Video recording - Toggle MP4 recording start/stop
  [D] Depth view      - Toggle depth image display
  [+/-] Resolution    - Switch capture resolution
  [Q] Quit

Usage:
  python capture.py
  python capture.py --no-depth         # Color only, no depth save
  python capture.py --auto 1.0         # Auto capture every 1 second
  python capture.py --prefix obj       # Specify filename prefix
  python capture.py --resolution 1280  # Specify resolution
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CAMERA, PATHS, CAPTURE, DISPLAY
from data_collection.utils import (
    create_directories,
    init_realsense_pipeline,
    get_frames,
    apply_depth_colormap,
    save_image,
    draw_info_overlay,
    get_depth_distance,
    drop_root_ownership,
)


class RealsenseCapture:
    """RealSense D435i capture controller"""

    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.profile = None
        self.align = None

        # State variables
        self.capture_count = 0
        self.is_recording = False
        self.is_auto_capture = False
        self.show_depth = DISPLAY["show_depth"]
        self.video_writer = None
        self.last_auto_time = 0
        self.auto_interval = args.auto if args.auto else CAPTURE["auto_interval"]
        self.prefix = args.prefix
        self.start_time = time.time()

    def start(self):
        """Start capture session"""
        print("=" * 60)
        print("  RealSense D435i - YOLO Training Data Capture Tool")
        print("=" * 60)

        # Create directories
        create_directories()

        # Initialize pipeline
        try:
            self.pipeline, self.profile, self.align = init_realsense_pipeline()
        except Exception as e:
            print(f"\n[ERROR] RealSense camera connection failed: {e}")
            print("  → Check that the camera is connected to a USB 3.0 port.")
            print("  → Check that no other program is using the camera.")
            sys.exit(1)

        # Wait for camera to stabilize
        print("\n[INFO] Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("[INFO] Ready! You can start capturing.\n")

        self._print_controls()
        self._main_loop()

    def _print_controls(self):
        """Print key controls"""
        print("┌─────────────────────────────────────┐")
        print("│           Key Controls               │")
        print("├─────────────────────────────────────┤")
        print("│  [S]     Manual capture (save image) │")
        print("│  [A]     Auto capture ON/OFF         │")
        print("│  [R]     Video recording start/stop  │")
        print("│  [D]     Depth view ON/OFF           │")
        print("│  [Q]     Quit                        │")
        print("└─────────────────────────────────────┘")
        print()

    def _main_loop(self):
        """Main capture loop"""
        try:
            while True:
                # Acquire frames
                color_image, depth_image, depth_frame = get_frames(
                    self.pipeline, self.align
                )

                if color_image is None:
                    continue

                # Auto capture
                if self.is_auto_capture:
                    now = time.time()
                    if now - self.last_auto_time >= self.auto_interval:
                        self._save_current(color_image, depth_image)
                        self.last_auto_time = now

                # Video recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(color_image)

                # Prepare display frame
                display_frame = self._build_display(
                    color_image, depth_image, depth_frame
                )

                # Show frame
                cv2.imshow(DISPLAY["window_name"], display_frame)

                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, color_image, depth_image):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected - shutting down.")
        finally:
            self._cleanup()

    def _build_display(self, color_image, depth_image, depth_frame):
        """Build display frame"""
        display = color_image.copy()

        # Center crosshair + depth display
        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(
            display, (cx, cy), (0, 255, 0),
            cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA
        )
        center_dist = get_depth_distance(depth_frame, cx, cy)

        # Info overlay
        elapsed = time.time() - self.start_time
        info = {
            "Captured": f"{self.capture_count}",
            "Center Depth": f"{center_dist:.2f}m",
            "Auto": f"ON ({self.auto_interval}s)" if self.is_auto_capture else "OFF",
            "Time": f"{int(elapsed)}s",
        }
        display = draw_info_overlay(display, info, self.is_recording)

        # Depth view overlay
        if self.show_depth:
            depth_colormap = apply_depth_colormap(depth_image, depth_frame)
            # Resize depth view to 1/3 of color view and place at bottom-right
            small_h, small_w = h // 3, w // 3
            depth_small = cv2.resize(depth_colormap, (small_w, small_h))

            # Add border
            cv2.rectangle(depth_small, (0, 0), (small_w - 1, small_h - 1), (255, 255, 255), 1)

            # Overlay at bottom-right
            y1 = h - small_h - 10
            x1 = w - small_w - 10
            display[y1:y1 + small_h, x1:x1 + small_w] = depth_small

        return display

    def _handle_key(self, key, color_image, depth_image):
        """
        Handle key input
        Returns: False to exit loop
        """
        if key == ord('q') or key == ord('Q'):
            return False

        elif key == ord('s') or key == ord('S'):
            self._save_current(color_image, depth_image)

        elif key == ord('a') or key == ord('A'):
            self.is_auto_capture = not self.is_auto_capture
            self.last_auto_time = time.time()
            state = "ON" if self.is_auto_capture else "OFF"
            print(f"[AUTO] Auto capture {state} (interval: {self.auto_interval}s)")

        elif key == ord('r') or key == ord('R'):
            self._toggle_recording(color_image)

        elif key == ord('d') or key == ord('D'):
            self.show_depth = not self.show_depth
            state = "ON" if self.show_depth else "OFF"
            print(f"[DEPTH] Depth view {state}")

        return True

    def _save_current(self, color_image, depth_image):
        """Save current frame"""
        depth_to_save = None if self.args.no_depth else depth_image
        filename = save_image(color_image, depth_to_save, self.prefix)
        self.capture_count += 1
        print(f"[SAVE] #{self.capture_count:04d} → {filename}")

    def _toggle_recording(self, color_image):
        """Toggle video recording"""
        if not self.is_recording:
            # Start recording
            timestamp = int(time.time())
            video_path = os.path.join(
                PATHS["videos"],
                f"{self.prefix}_video_{timestamp}{CAPTURE['video_format']}"
            )
            h, w = color_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*CAPTURE["video_codec"])
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, CAPTURE["video_fps"], (w, h)
            )
            drop_root_ownership(video_path)
            self.is_recording = True
            print(f"[REC] Recording started → {video_path}")
        else:
            # Stop recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("[REC] Recording stopped")

    def _cleanup(self):
        """Release resources"""
        print("\n[INFO] Cleaning up...")

        if self.is_recording and self.video_writer:
            self.video_writer.release()
            print("[INFO] Video saved")

        if self.pipeline:
            self.pipeline.stop()
            print("[INFO] RealSense pipeline stopped")

        cv2.destroyAllWindows()

        print(f"\n[RESULT] Captured {self.capture_count} image(s) total.")
        print(f"  → Images: {PATHS['images']}")
        print(f"  → Depth:  {PATHS['depth']}")
        print(f"  → Videos: {PATHS['videos']}")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="RealSense D435i YOLO training data capture tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Disable depth image saving",
    )
    parser.add_argument(
        "--auto", type=float, default=None,
        help=f"Auto capture interval (seconds). Default: {CAPTURE['auto_interval']}",
    )
    parser.add_argument(
        "--prefix", type=str, default="img",
        help="Filename prefix. Default: img",
    )
    parser.add_argument(
        "--resolution", type=int, choices=[640, 1280, 1920], default=None,
        help="Capture resolution (width). Default: uses config.py setting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolution override
    if args.resolution:
        if args.resolution == 640:
            CAMERA["color_width"], CAMERA["color_height"] = 640, 480
            CAMERA["depth_width"], CAMERA["depth_height"] = 640, 480
        elif args.resolution == 1280:
            CAMERA["color_width"], CAMERA["color_height"] = 1280, 720
            CAMERA["depth_width"], CAMERA["depth_height"] = 1280, 720
        elif args.resolution == 1920:
            CAMERA["color_width"], CAMERA["color_height"] = 1920, 1080
            # D435i depth sensor maximum supported resolution is 1280x720
            CAMERA["depth_width"], CAMERA["depth_height"] = 1280, 720

    capture = RealsenseCapture(args)
    capture.start()
