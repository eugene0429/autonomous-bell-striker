"""
Target 3D Position Estimation
==============================

Combines detection results (2D bbox) + depth data + VIO pose
to estimate the target's 3D position in world coordinates.

Pipeline:
  1. Detect target bbox with YOLO
  2. Acquire depth value at bbox center
  3. Convert 2D pixel + depth → 3D camera coordinates (deprojection)
  4. Transform camera coordinates → world coordinates using VIO pose

TODO:
  - Depth filtering (use median depth within bbox region)
  - Camera → world coordinate transform
  - Multi-frame position averaging / filtering
"""

import numpy as np


class PositionEstimator:
    """Target 3D position estimator"""

    def __init__(self, camera):
        """
        Args:
            camera: RealSenseCamera instance (common.realsense_wrapper)
        """
        self.camera = camera

    def estimate(self, detection, depth_frame, camera_pose=None):
        """
        Estimate 3D world coordinates for a single detection

        Args:
            detection: dict with 'bbox' key (x1, y1, x2, y2)
            depth_frame: rs.depth_frame object
            camera_pose: 4x4 transformation matrix (from VIO; returns camera coords only if None)

        Returns:
            position_3d: (x, y, z) world coordinates (meters), or None
        """
        # Bbox center coordinates
        bbox = detection["bbox"]
        cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

        # 3D position in camera frame
        point_camera = self.camera.pixel_to_3d(depth_frame, cx, cy)
        if point_camera is None:
            return None

        # Transform to world coordinates if VIO pose is available
        if camera_pose is not None:
            point_camera_h = np.array([*point_camera, 1.0])
            point_world = camera_pose @ point_camera_h
            return tuple(point_world[:3])

        return tuple(point_camera)

    def estimate_batch(self, detections, depth_frame, camera_pose=None):
        """
        Estimate 3D positions for multiple detections at once

        Args:
            detections: list of detection dicts
            depth_frame: rs.depth_frame
            camera_pose: 4x4 transformation matrix

        Returns:
            results: list of dict (detection + position_3d)
        """
        results = []
        for det in detections:
            pos = self.estimate(det, depth_frame, camera_pose)
            result = {**det, "position_3d": pos}
            results.append(result)
        return results
