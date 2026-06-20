"""bbox + depth → median ROI depth.

Used by VisualServoController to convert raw depth_frame + YOLO bbox into a
single robust depth value at the target's center. Single-pixel depth at bbox
center is noisy and frequently zero (RealSense holes); the central
`roi_frac`-fraction of the bbox provides a more stable estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_target_depth(
    depth: np.ndarray,
    bbox: Tuple[int, int, int, int],
    roi_frac: float = 0.4,
    min_valid_pixels: int = 10,
    depth_scale_m: float = 0.001,    # RealSense default: depth_unit = 1 mm
) -> Optional[float]:
    """
    Parameters
    ----------
    depth : (H, W) uint16 or float numpy array, depth values in raw units
    bbox  : (x1, y1, x2, y2) pixel coords
    roi_frac : fraction of bbox edge used for the central ROI (0 < f ≤ 1)
    min_valid_pixels : need at least this many >0 samples in ROI
    depth_scale_m : multiplier from raw depth units to meters

    Returns
    -------
    Median depth in meters, or None if not enough valid pixels.
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    # clip bbox to image bounds before computing central ROI
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    # central ROI inside bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    rw = max(1.0, bw * roi_frac)
    rh = max(1.0, bh * roi_frac)
    rx1 = int(round(cx - rw / 2))
    ry1 = int(round(cy - rh / 2))
    rx2 = int(round(cx + rw / 2))
    ry2 = int(round(cy + rh / 2))

    # clip to image
    rx1 = max(0, rx1); ry1 = max(0, ry1)
    rx2 = min(w, rx2); ry2 = min(h, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    patch = depth[ry1:ry2, rx1:rx2]
    valid = patch[patch > 0]
    if valid.size < min_valid_pixels:
        return None

    return float(np.median(valid)) * depth_scale_m
