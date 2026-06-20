"""
Utility functions
RealSense pipeline initialization, frame processing, file saving, etc.
"""

import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs

from config import CAMERA, PATHS, CAPTURE, DISPLAY

# ---------------------------------------------------------
# RealSense post-processing filters — global (lazy initialization)
# ---------------------------------------------------------
spatial_filter = None
temporal_filter = None
colorizer = None

def init_filters():
    global spatial_filter, temporal_filter, colorizer
    if spatial_filter is None:
        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        # Do not fill holes arbitrarily (set to 0) to preserve data integrity
        spatial_filter.set_option(rs.option.holes_fill, 0)

        temporal_filter = rs.temporal_filter()
        # Remove hole_filling_filter that creates fake pixels, prioritizing data integrity
        colorizer = rs.colorizer()


def drop_root_ownership(path):
    """If running as root via sudo, chown path back to the invoking user.

    Why: running capture under sudo leaves dataset files owned by root,
    making them unreadable/undeletable by the desktop user afterwards.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if not sudo_uid:
        return
    try:
        os.chown(path, int(sudo_uid), int(sudo_gid) if sudo_gid else -1)
    except OSError:
        pass


def create_directories():
    """Create dataset storage directories"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
        drop_root_ownership(path)
    print("[INFO] Directories created:")
    for name, path in PATHS.items():
        print(f"  → {name}: {path}")


def init_realsense_pipeline():
    """
    Initialize RealSense pipeline
    Returns:
        pipeline: rs.pipeline object
        profile: streaming profile
        align: rs.align object (depth→color alignment)
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Color stream configuration
    config.enable_stream(
        rs.stream.color,
        CAMERA["color_width"],
        CAMERA["color_height"],
        rs.format.bgr8,
        CAMERA["color_fps"],
    )

    # Depth stream configuration
    config.enable_stream(
        rs.stream.depth,
        CAMERA["depth_width"],
        CAMERA["depth_height"],
        rs.format.z16,
        CAMERA["depth_fps"],
    )

    # Start pipeline
    profile = pipeline.start(config)

    # Initialize post-processing filters after sensor and pipeline are ready
    init_filters()

    # IR emitter setup
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if CAMERA["enable_ir_emitter"]:
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    else:
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

    # Depth → Color alignment object
    align = rs.align(rs.stream.color) if CAMERA["align_depth_to_color"] else None

    print(f"[INFO] RealSense pipeline started")
    print(f"  → Color: {CAMERA['color_width']}x{CAMERA['color_height']} @ {CAMERA['color_fps']}fps")
    print(f"  → Depth: {CAMERA['depth_width']}x{CAMERA['depth_height']} @ {CAMERA['depth_fps']}fps")
    print(f"  → Depth alignment: {'ON' if align else 'OFF'}")

    return pipeline, profile, align


def get_frames(pipeline, align=None):
    """
    Acquire color/depth frames from pipeline
    Returns:
        color_image: numpy array (BGR)
        depth_image: numpy array (16bit)
        depth_frame: rs.depth_frame object
    """
    frames = pipeline.wait_for_frames()

    if align:
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None, None

    # Apply post-processing filters to the aligned depth frame for quality improvement
    depth_frame = spatial_filter.process(depth_frame)
    depth_frame = temporal_filter.process(depth_frame)
    # Skip hole filling to honestly preserve zero values (occluded / unmeasurable regions)
    depth_frame = depth_frame.as_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    return color_image, depth_image, depth_frame


def apply_depth_colormap(depth_image, depth_frame=None):
    """Apply colormap to depth image (for visualization)"""
    if depth_frame is not None:
        # Apply Intel-recommended colorizer (minimizes noise, auto-scales)
        colorized_frame = colorizer.colorize(depth_frame)
        return np.asanyarray(colorized_frame.get_data())
    else:
        # Fallback (legacy method)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            DISPLAY.get("depth_colormap", cv2.COLORMAP_JET),
        )
        return depth_colormap


def save_image(color_image, depth_image, prefix="img"):
    """
    Save color + depth images
    Returns:
        filename: saved filename (without extension)
    """
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}"

    # Save color image
    color_path = os.path.join(PATHS["images"], filename + CAPTURE["image_format"])
    params = [cv2.IMWRITE_JPEG_QUALITY, CAPTURE["image_quality"]]
    cv2.imwrite(color_path, color_image, params)
    drop_root_ownership(color_path)

    # Save depth image (16bit PNG) — skipped when caller passes depth_image=None
    if depth_image is not None:
        depth_path = os.path.join(PATHS["depth"], filename + CAPTURE["depth_format"])
        cv2.imwrite(depth_path, depth_image)
        drop_root_ownership(depth_path)

    return filename


def draw_info_overlay(frame, info_dict, recording=False):
    """Draw information overlay on frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_offset = 20
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            DISPLAY["font_scale"],
            DISPLAY["font_color"],
            DISPLAY["font_thickness"],
            cv2.LINE_AA,
        )
        y_offset += 20

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 25, 15), 8, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (w - 60, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # Bottom key hint
    help_text = "[S] Save | [R] Record | [A] Auto | [D] Depth | [Q] Quit"
    cv2.putText(
        frame, help_text, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return frame


def get_depth_distance(depth_frame, x, y):
    """Return depth value (m) at specified coordinates"""
    if depth_frame:
        return depth_frame.get_distance(x, y)
    return 0.0
