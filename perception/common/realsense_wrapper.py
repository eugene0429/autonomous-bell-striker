"""
RealSense D435i Common Wrapper
Pipeline initialization, frame acquisition, and shared functionality for all modules
"""

import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """RealSense D435i camera management class"""

    def __init__(self, config, hardware_reset_on_start=False):
        """
        Args:
            config: Camera configuration dict (CAMERA config)
            hardware_reset_on_start: If True, power-cycle the D435i via
                hardware_reset() before opening the pipeline. Recovers from
                'failed to set power state' / USB suspend left over from a
                dirty prior session. Adds ~5s startup delay.
        """
        self.config = config
        self.hardware_reset_on_start = hardware_reset_on_start
        self.pipeline = None
        self.profile = None
        self.align = None
        self.depth_sensor = None
        self.intrinsics = None
        self._rs_config = None  # last rs.config actually used (for restart)

    @staticmethod
    def _hardware_reset_all():
        """Power-cycle every connected RealSense device.

        Recovers from 'failed to set power state' / USB suspend left over from
        a dirty prior session. Blocks ~5s for the device to re-enumerate.
        """
        ctx = rs.context()
        for d in ctx.query_devices():
            try:
                name = d.get_info(rs.camera_info.name)
            except Exception:
                name = "?"
            print(f"[RealSense] hardware_reset {name}")
            try:
                d.hardware_reset()
            except Exception as e:
                print(f"[RealSense] hardware_reset failed: {e}")
        time.sleep(5.0)

    def _build_config(self):
        """Build the rs.config for the current self.config (color + depth [+ IMU])."""
        rs_config = rs.config()

        # Color stream
        rs_config.enable_stream(
            rs.stream.color,
            self.config["color_width"],
            self.config["color_height"],
            rs.format.bgr8,
            self.config["color_fps"],
        )

        # Depth stream
        rs_config.enable_stream(
            rs.stream.depth,
            self.config["depth_width"],
            self.config["depth_height"],
            rs.format.z16,
            self.config["depth_fps"],
        )

        # IMU stream (used by VIO)
        if self.config.get("enable_imu", False):
            # Specify FPS explicitly to avoid 'Couldn't resolve requests' on Ubuntu/Linux
            # D435i standard: Accel=100Hz, Gyro=200Hz
            rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
            rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        return rs_config

    def start(self):
        """Start pipeline"""
        if self.hardware_reset_on_start:
            self._hardware_reset_all()

        self.pipeline = rs.pipeline()
        rs_config = self._build_config()

        try:
            self.profile = self.pipeline.start(rs_config)
            self._rs_config = rs_config
        except RuntimeError as e:
            if self.config.get("enable_imu", False):
                print(f"[ERROR] Pipeline start failed (IMU conflict / known macOS bug): {e}")
                print("[INFO] Fallback triggered: retrying without IMU in Visual-Only mode...")

                # Rebuild config without IMU and retry
                self.config["enable_imu"] = False  # Update state
                rs_config = self._build_config()
                self.profile = self.pipeline.start(rs_config)
                self._rs_config = rs_config
            else:
                raise e

        self._configure_device()

        print(f"[INFO] RealSense pipeline started")
        print(f"  → Color: {self.config['color_width']}x{self.config['color_height']} @ {self.config['color_fps']}fps")
        print(f"  → Depth: {self.config['depth_width']}x{self.config['depth_height']} @ {self.config['depth_fps']}fps")
        print(f"  → Depth alignment: {'ON' if self.align else 'OFF'}")
        print(f"  → IMU: {'ON' if self.config.get('enable_imu', False) else 'OFF'}")

        return self

    def _configure_device(self):
        """Configure device options + post-processing filters after pipeline start."""
        # IR emitter setup
        device = self.profile.get_device()
        # Enable Global Time on all sensors (improves timestamp synchronization accuracy).
        # Skipped on macOS: known to stall first-frame delivery on librealsense for D435i.
        if sys.platform != "darwin":
            for s in device.query_sensors():
                if s.supports(rs.option.global_time_enabled):
                    s.set_option(rs.option.global_time_enabled, 1)

        self.depth_sensor = device.first_depth_sensor()
        emitter = 1 if self.config.get("enable_ir_emitter", False) else 0
        self.depth_sensor.set_option(rs.option.emitter_enabled, emitter)

        # Depth → Color alignment
        if self.config.get("align_depth_to_color", True):
            self.align = rs.align(rs.stream.color)

        # Depth post-processing filter initialization
        self.depth_filters = []
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 2)

        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        # Disable hole filling to preserve data integrity (same as capture mode)
        self.spatial_filter.set_option(rs.option.holes_fill, 0)

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

        # Use same colorizer as capture mode
        self.colorizer = rs.colorizer()

        # Remove hole_filling_filter
        self.depth_filters = [
            self.spatial_filter,
            self.temporal_filter,
        ]

        # Store camera intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    def _restart_with_reset(self):
        """Power-cycle the device and re-open the pipeline with the same config.

        Used to self-heal a wedged USB/power state where the pipeline starts
        but never delivers frames (common on Raspberry Pi after a dirty exit).
        """
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self._hardware_reset_all()
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(self._rs_config)
        self._configure_device()

    def get_frames(self):
        """
        Acquire color/depth frames

        Returns:
            color_image: numpy array (BGR)
            depth_image: numpy array (16bit)
            depth_frame: rs.depth_frame object
        """
        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # Apply depth post-processing filters
        for f in self.depth_filters:
            depth_frame = f.process(depth_frame).as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, depth_frame

    def get_frames_vio(self):
        """
        Acquire VIO frames (color + depth + IMU + timestamp)

        Returns:
            dict: {
                'color': numpy BGR image,
                'depth': numpy 16bit depth image,
                'depth_frame': rs.depth_frame,
                'accel': (x,y,z) or None,
                'gyro': (x,y,z) or None,
                'timestamp': frame timestamp (ms),
            } or None
        """
        frames = self.pipeline.wait_for_frames()
        timestamp = frames.get_timestamp()

        # Collect IMU data
        accel_data = None
        gyro_data = None
        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame().get_motion_data()
                if frame.get_profile().stream_type() == rs.stream.accel:
                    accel_data = (motion.x, motion.y, motion.z)
                elif frame.get_profile().stream_type() == rs.stream.gyro:
                    gyro_data = (motion.x, motion.y, motion.z)

        # Depth-color alignment
        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        # Apply depth post-processing filters
        for f in self.depth_filters:
            depth_frame = f.process(depth_frame).as_depth_frame()

        return {
            'color': np.asanyarray(color_frame.get_data()),
            'depth': np.asanyarray(depth_frame.get_data()),
            'depth_frame': depth_frame,
            'accel': accel_data,
            'gyro': gyro_data,
            'timestamp': timestamp,
        }

    def get_imu_data(self, frames):
        """
        Acquire IMU data (for VIO)

        Args:
            frames: RealSense frameset

        Returns:
            accel: (x, y, z) accelerometer data
            gyro: (x, y, z) gyroscope data
        """
        accel_data = None
        gyro_data = None

        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame().get_motion_data()
                if frame.get_profile().stream_type() == rs.stream.accel:
                    accel_data = (motion.x, motion.y, motion.z)
                elif frame.get_profile().stream_type() == rs.stream.gyro:
                    gyro_data = (motion.x, motion.y, motion.z)

        return accel_data, gyro_data

    def get_intrinsics(self):
        """Return camera intrinsic parameters"""
        return self.intrinsics

    def pixel_to_3d(self, depth_frame, pixel_x, pixel_y):
        """
        Convert 2D pixel coordinates + depth → 3D camera coordinates

        Args:
            depth_frame: rs.depth_frame
            pixel_x, pixel_y: pixel coordinates

        Returns:
            (x, y, z): 3D position in camera frame (meters)
        """
        depth = depth_frame.get_distance(pixel_x, pixel_y)
        if depth == 0:
            return None
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [pixel_x, pixel_y], depth
        )
        return point_3d

    def pixel_to_3d_with_depth(self, pixel_x, pixel_y, depth_m):
        """
        Deproject (pixel_x, pixel_y) using externally-provided depth in meters.

        Used by Phase 2 target estimator: it computes ROI-median depth via
        compute_target_depth, then feeds that depth back through deprojection
        at the bbox center pixel — separating lateral pixel position (bbox
        center) from depth measurement (ROI median).

        Args:
            pixel_x, pixel_y: pixel coordinates
            depth_m: depth in meters (caller-provided, e.g. from ROI median)

        Returns:
            (x, y, z): 3D position in camera frame (meters), or None if
                      depth_m <= 0.
        """
        if depth_m <= 0:
            return None
        return rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [pixel_x, pixel_y], float(depth_m)
        )

    def stop(self):
        """Stop pipeline"""
        if self.pipeline:
            self.pipeline.stop()
            print("[INFO] RealSense pipeline stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def warmup(self, num_frames=30, timeout_ms=5000, max_resets=2):
        """Wait for camera to stabilize, self-healing a wedged USB/power state.

        Streaming a few frames lets auto-exposure converge before real use. If a
        frame fails to arrive within ``timeout_ms`` (typically a stuck USB/power
        state on Raspberry Pi after a dirty exit), the device is hardware-reset
        and the pipeline restarted, then warmup is retried from scratch — up to
        ``max_resets`` times before giving up.

        Args:
            num_frames: frames to stream before considering the camera ready.
            timeout_ms: per-frame wait timeout. A healthy camera delivers a
                frame in well under this; a large value just delays recovery.
            max_resets: how many hardware-reset + restart attempts to make.

        Raises:
            RuntimeError: if frames still don't arrive after ``max_resets``.
        """
        print("[INFO] Warming up camera...")
        resets = 0
        i = 0
        while i < num_frames:
            try:
                self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                i += 1
            except RuntimeError as e:
                if resets >= max_resets:
                    print(f"[ERROR] Camera not delivering frames after "
                          f"{max_resets} hardware resets, giving up.")
                    raise
                resets += 1
                print(f"[WARN] Frame timeout during warmup ({e}); "
                      f"hardware-resetting camera (attempt {resets}/{max_resets})...")
                self._restart_with_reset()
                i = 0  # restart warmup count on the fresh pipeline
        print("[INFO] Camera ready!")


def apply_depth_colormap(depth_image, depth_frame=None, colorizer=None):
    """Apply colormap to depth image (for visualization)"""
    if depth_frame is not None and colorizer is not None:
        colorized_frame = colorizer.colorize(depth_frame)
        return np.asanyarray(colorized_frame.get_data())
    else:
        return cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )


def get_depth_distance(depth_frame, x, y):
    """Return depth value (m) at specified coordinates"""
    if depth_frame:
        return depth_frame.get_distance(x, y)
    return 0.0
