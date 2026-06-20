"""
RealSense D435i Perception Configuration
Data collection / VIO localization / Target detection unified settings
"""

import os

# ============================================================
# Camera settings
# ============================================================
CAMERA = {
    # Color stream resolution and FPS
    "color_width": 640,
    "color_height": 480,
    "color_fps": 30,

    # Depth stream resolution and FPS
    "depth_width": 640,
    "depth_height": 480,
    "depth_fps": 30,

    # Enable Depth → Color alignment
    "align_depth_to_color": True,

    # Enable IR emitter (required for stereo matching quality)
    "enable_ir_emitter": True,

    # Enable IMU stream (used by VIO)
    "enable_imu": False,
}

# ============================================================
# Storage path settings
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

PATHS = {
    "dataset": DATASET_DIR,
    "images": os.path.join(DATASET_DIR, "images"),
    "depth": os.path.join(DATASET_DIR, "depth"),
    "videos": os.path.join(DATASET_DIR, "videos"),
    "labels": os.path.join(DATASET_DIR, "labels"),
}

# ============================================================
# Capture settings
# ============================================================
CAPTURE = {
    # Auto capture interval (seconds)
    "auto_interval": 0.5,

    # Image save format
    "image_format": ".jpg",
    "image_quality": 95,       # JPEG quality (0-100)

    # Depth image save format (16bit PNG to preserve raw depth values)
    "depth_format": ".png",

    # Video codec
    "video_codec": "mp4v",
    "video_format": ".mp4",
    "video_fps": 30,
}

# ============================================================
# Display settings
# ============================================================
DISPLAY = {
    "window_name": "RealSense D435i - YOLO Data Capture",
    "show_depth": True,
    "show_info_overlay": True,
    "depth_colormap": 2,       # cv2.COLORMAP_JET = 2
    "font_scale": 0.6,
    "font_color": (0, 255, 0),
    "font_thickness": 1,
}

# ============================================================
# YOLO dataset structure settings
# ============================================================
YOLO = {
    # Class list (modify as needed)
    "classes": [
        # "class_0",
        # "class_1",
    ],
    # Train/Val split ratio
    "train_ratio": 0.8,
    "val_ratio": 0.2,
}

# ============================================================
# VIO settings
# ============================================================
VIO = {
    # Feature detector
    "feature_type": "FAST",
    "fast_threshold": 20,
    "max_features": 300,

    # Lucas-Kanade Optical Flow parameters
    "lk_win_size": (21, 21),
    "lk_max_level": 3,

    # Keyframe selection criteria
    "keyframe_min_features": 80,       # Trigger keyframe when tracked features fall below this
    "keyframe_max_interval": 10,       # Maximum keyframe interval (frames)

    # PnP RANSAC parameters
    "pnp_reproj_threshold": 3.0,       # Reprojection error threshold (px)
    "pnp_confidence": 0.99,
    "pnp_min_inliers": 10,             # Minimum number of inliers

    # Depth filter
    "depth_min": 0.3,                  # Minimum depth (m)
    "depth_max": 5.0,                  # Maximum depth (m)

    # ── EKF IMU noise parameters ──
    # Approximate values based on D435i BMI055
    "accel_noise_std": 0.1,            # Accelerometer measurement noise std (m/s²/sample)
    "gyro_noise_std": 0.01,            # Gyroscope measurement noise std (rad/s/sample)
    "accel_bias_std": 0.005,           # Accelerometer bias random walk coefficient (m/s²/√s)
    "gyro_bias_std": 0.0005,           # Gyroscope bias random walk coefficient (rad/s²/√s)
    "max_accel_bias": 0.5,             # Accelerometer bias clipping limit (m/s²)
    "max_gyro_bias": 0.15,             # Gyroscope bias clipping limit (rad/s)

    # ── IMU initialization / stationary detection ──
    "imu_init_samples": 200,           # IMU samples to collect while stationary (2s at 100Hz)
    "gravity_magnitude": 9.81,         # Gravitational acceleration magnitude (m/s²)
    "stationary_accel_tol": 0.35,      # | |a_unbiased| - g | threshold for stationary detection (m/s²)
    "stationary_gyro_tol": 0.08,       # |ω_unbiased| threshold for stationary detection (rad/s)

    # ── ZUPT (Zero-velocity UPdate) ──
    # Uses EKF measurement update instead of hard reset → preserves covariance consistency
    "zupt_vel_noise": 0.02,            # ZUPT measurement noise std (m/s) — smaller = stronger zero-velocity constraint
    "zupt_position_damping": 1.0,      # Acceleration integration suppression during stationary (0~1, 1=full suppression)
    "zupt_min_frames": 2,              # Consecutive stationary frames required before ZUPT triggers (hysteresis)

    # ── Vision velocity measurement noise ──
    # Noise std when using vision finite-difference velocity as EKF measurement (m/s)
    # Smaller = stronger constraint (too small propagates PnP noise into velocity)
    "visual_vel_noise": 0.3,
    "max_visual_vel": 3.0,             # Maximum allowed visual velocity (m/s); rejected if exceeded

    # ── PnP result validity check ──
    "pnp_max_position_jump": 0.5,      # Maximum allowed PnP position deviation from EKF prediction (m)

    # ── Non-Holonomic Constraint (NHC) — ground robots only ──
    # Enable for tank/wheel rovers that have no lateral or vertical motion
    # Applied every frame even while moving → constrains drift direction to forward only
    "enable_nhc": True,                # True for ground robots, False for drones/handheld
    "nhc_lateral_noise": 0.05,         # Lateral velocity constraint noise (m/s); smaller = stronger
    "nhc_vertical_noise": 0.02,        # Vertical velocity constraint noise (m/s)

    # ── Velocity damping when vision correction is unavailable ──
    # 0.85^30fps ≈ 0.004/s → velocity decays to ~0 within 0.5s
    "no_vision_velocity_damping": 0.85,

    # ── PnP observation noise (baseline for adaptive scaling) ──
    "pnp_pos_noise": 0.03,             # Baseline position observation noise (m)
    "pnp_rot_noise": 0.015,            # Baseline orientation observation noise (rad)

    # ── Depth sampling ──
    "depth_sample_window": 3,          # Neighborhood pixel median depth sampling window size

    # ── Keyframe trigger thresholds ──
    "keyframe_max_rotation_deg": 15.0, # Rotation trigger relative to keyframe (degrees)
    "keyframe_max_translation": 0.3,   # Translation trigger relative to keyframe (m)

    # ── Numerical stability ──
    "normalize_rotation_interval": 30, # SO(3) re-normalization period (frames)

    # ── Initial uncertainty ──
    "init_pos_std": 0.01,
    "init_vel_std": 0.5,               # Larger initial velocity uncertainty for faster cross-covariance buildup
    "init_att_std": 0.05,
    "init_bias_std": 0.1,
}

# ============================================================
# Detection settings
# ============================================================
DETECTION = {
    # YOLO model path
    "model_path": os.path.join(BASE_DIR, "models", "best.pt"),
    # Minimum confidence threshold
    "confidence_threshold": 0.5,
}
