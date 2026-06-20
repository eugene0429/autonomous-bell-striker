"""
VIO Real-time Execution Loop
Runs VIO localization with RealSense D435i and visualizes results
"""

import cv2
import numpy as np
import time

from config import CAMERA, VIO as VIO_CONFIG
from common.realsense_wrapper import RealSenseCamera, apply_depth_colormap


def draw_overlay(image, tracker, fps):
    """Display VIO status overlay"""
    stats = tracker.get_stats()
    pos = tracker.get_position()
    euler = tracker.get_euler_degrees()

    imu_state = "IMU" if stats.get("imu_ready") else "VO"
    zupt_flag = " ZUPT" if stats.get("stationary") else ""
    lines = [
        f"FPS: {fps:.1f}  [{imu_state}{zupt_flag}]",
        f"Features: {stats['tracked_features']} | Inliers: {stats['inliers']}",
        f"Keyframe age: {stats['frames_since_keyframe']}  |Vel|={stats.get('ekf_vel_norm', 0):.3f}m/s",
        f"Pos: X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f} m",
        f"Rot: R={euler[0]:.1f} P={euler[1]:.1f} Y={euler[2]:.1f} deg",
    ]

    for i, line in enumerate(lines):
        y = 25 + i * 25
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return image


def draw_trajectory(traj_image, positions, scale=100, size=400):
    """Draw 2D trajectory (XZ plane, top-down view)"""
    traj_image[:] = 40  # Dark background

    center = np.array([size // 2, size // 2])

    if len(positions) < 2:
        return traj_image

    def _to_pt(p, c, s, sz):
        x = int(np.clip(c[0] + p[0] * s, 0, sz - 1))
        y = int(np.clip(c[1] - p[2] * s, 0, sz - 1))
        return (x, y)

    for i in range(1, len(positions)):
        p0 = positions[i - 1]
        p1 = positions[i]
        # Skip NaN/inf values
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        pt0 = _to_pt(p0, center, scale, size)
        pt1 = _to_pt(p1, center, scale, size)
        cv2.line(traj_image, pt0, pt1, (0, 255, 0), 1, cv2.LINE_AA)

    # Current position
    cur = positions[-1]
    cur_pt = (int(center[0] + cur[0] * scale), int(center[1] - cur[2] * scale))
    cv2.circle(traj_image, cur_pt, 4, (0, 0, 255), -1)

    # Axis labels
    cv2.putText(traj_image, "X", (size - 20, size // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    cv2.putText(traj_image, "Z", (size // 2 + 5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    cv2.putText(traj_image, "Trajectory (top-down)", (5, size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    return traj_image


def _init_vio(use_imu):
    """Common VIO initialization: camera reset, start, warmup, tracker creation"""
    cam_config = {**CAMERA, "enable_imu": use_imu}

    if use_imu:
        print("[VIO] Starting in Visual-Inertial Odometry (VIO) mode (IMU enabled)")
    else:
        print("[VIO] Starting in Visual-Only Odometry mode (IMU disabled)")

    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > 0:
        print("[VIO] Performing camera hardware reset...")
        devices[0].hardware_reset()
        time.sleep(5)

    camera = RealSenseCamera(cam_config)
    camera.start()
    camera.warmup(30)

    from vio.vio_tracker import VIOTracker
    tracker = VIOTracker(camera.get_intrinsics(), VIO_CONFIG)
    return camera, tracker


def run_vio_headless(use_imu=True):
    """Headless VIO loop — prints world-frame (x, y, theta) to terminal.

    World coordinate convention (camera starts at origin):
      - world X = camera Z (forward)
      - world Y = camera -X (left)
      - theta   = yaw angle (rad), CCW positive viewed from above
    """
    import math

    camera, tracker = _init_vio(use_imu)

    print("[VIO] Headless mode — world-frame (x, y, theta) output")
    print("[VIO] Press Ctrl+C to stop")
    print(f"{'time_s':>8s}  {'x_m':>8s}  {'y_m':>8s}  {'theta_deg':>10s}  {'fps':>5s}")

    frame_count = 0
    fps = 0.0
    t_start = time.time()

    try:
        while True:
            data = camera.get_frames_vio()
            if data is None:
                continue

            tracker.update(
                color_image=data['color'],
                depth_image=data['depth'],
                accel=data['accel'],
                gyro=data['gyro'],
                timestamp=data['timestamp'],
            )

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Camera frame → world frame
            pos = tracker.get_position()      # (x, y, z) in camera frame
            R = tracker.get_rotation()         # 3x3 rotation matrix

            world_x = pos[2]                   # camera Z = forward
            world_y = -pos[0]                  # camera -X = left

            # Yaw: angle of forward direction on ground plane
            forward = R[:, 2]                  # camera Z column
            theta_rad = math.atan2(-forward[0], forward[2])
            theta_deg = math.degrees(theta_rad)

            ts = data['timestamp']
            print(f"\r{ts:8.2f}  {world_x:8.3f}  {world_y:8.3f}  {theta_deg:10.2f}  {fps:5.1f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[VIO] Stopped (Ctrl+C)")
    finally:
        camera.stop()
        print("[VIO] Shutdown complete")


def run_vio(use_imu=True):
    """VIO main loop with GUI visualization"""
    camera, tracker = _init_vio(use_imu)

    # Trajectory history
    positions = []
    traj_size = 400
    traj_image = np.zeros((traj_size, traj_size, 3), dtype=np.uint8)

    print("[VIO] Running... press 'q' to quit, 'r' to reset")

    frame_count = 0
    fps = 0.0
    t_start = time.time()

    try:
        while True:
            data = camera.get_frames_vio()
            if data is None:
                continue

            pose = tracker.update(
                color_image=data['color'],
                depth_image=data['depth'],
                accel=data['accel'],
                gyro=data['gyro'],
                timestamp=data['timestamp'],
            )

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Record trajectory
            pos = tracker.get_position()
            positions.append(pos.copy())
            if len(positions) > 2000:
                positions[:] = positions[1000:]

            # Visualization
            display = data['color'].copy()
            draw_overlay(display, tracker, fps)

            # Draw tracked feature points
            if tracker.prev_points is not None and len(tracker.prev_points) > 0:
                for pt in tracker.prev_points:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            # Draw trajectory
            draw_trajectory(traj_image, positions, scale=100, size=traj_size)

            # Depth colormap (apply rs.colorizer like in capture mode)
            depth_color = apply_depth_colormap(
                data['depth'],
                depth_frame=data.get('depth_frame'),
                colorizer=camera.colorizer
            )

            # Layout: [camera view | depth | trajectory]
            depth_resized = cv2.resize(depth_color, (traj_size, traj_size))
            display_resized = cv2.resize(display, (int(traj_size * display.shape[1] / display.shape[0]), traj_size))

            combined = np.hstack([display_resized, depth_resized, traj_image])
            cv2.imshow("VIO Tracker", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.reset()
                positions.clear()
                print("[VIO] Reset complete")

    except KeyboardInterrupt:
        print("\n[VIO] Stopped (Ctrl+C)")
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        print("[VIO] Shutdown complete")
