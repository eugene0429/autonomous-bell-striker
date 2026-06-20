"""
VIO (Visual-Inertial Odometry) Camera Localization
===================================================

Estimates the camera's 6DoF pose (position + orientation) in real-time
using RealSense D435i color camera + depth + IMU data.

Pipeline:
  1. FAST feature detection (on keyframes)
  2. Lucas-Kanade Optical Flow tracking (every frame)
  3. PnP + RANSAC pose estimation (using depth)
  4. IMU pre-integration + Error-state EKF fusion

Improvements (referencing MSCKF / VINS-Mono / OpenVINS):
  [EKF formulation fixes]
  - Q matrix: discretized from continuous-time noise model
      * Removed direct process noise on position (physically incorrect)
      * Bias random walk noise: dt² → dt correction
      * Velocity noise: R @ diag(σ_a²) @ R.T * dt²
  - F Jacobian: linearized at pre-update orientation (R_prev) for consistency
  - Added covariance symmetrization (numerical stability)
  - Common Kalman update logic unified in _kalman_update()

  [Drift fixes]
  - Completely removed external velocity estimation (_update_velocity_from_vision)
      * This function was overwriting EKF's correct velocity correction with
        finite-difference values, which was the direct cause of post-motion drift
      * EKF cross-covariance (position-velocity) automatically updates velocity
        during visual correction
  - ZUPT processed as EKF measurement update (not velocity hard reset)
      * Maintains filter consistency, correctly updates covariance matrix
      * Ref: Foxlin (2005), "Pedestrian Tracking with Shoe-Mounted Inertial Sensors"
  - Bias clipping to prevent filter divergence

  [Observation model improvements]
  - Adaptive observation noise based on PnP inlier count
      * Higher inlier ratio → smaller noise → stronger correction
  - IMU-predicted orientation used as PnP initial guess (faster convergence)

  [Numerical stability]
  - Periodic SO(3) re-normalization of rotation matrix (SVD)
  - Added EKFState.normalize() method

  [Performance / reliability]
  - Vectorized _filter_with_depth (Python loop → NumPy)
  - Improved depth reliability: neighbor pixel median depth sampling
  - Improved keyframe criteria: rotation/translation-based triggers
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class IMUPreintegrator:
    """IMU pre-integrator — accumulates IMU measurements between two keyframes

    Kept for reference. To be used for future tight-coupling extensions.
    Ref: Forster et al., "On-Manifold Preintegration for Real-Time
         Visual-Inertial Odometry" (TRO 2017)
    """

    def __init__(self, accel_noise_std, gyro_noise_std):
        self.accel_noise = accel_noise_std
        self.gyro_noise = gyro_noise_std
        self.reset()

    def reset(self):
        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_R = np.eye(3)
        self.dt_sum = 0.0

    def integrate(self, accel, gyro, dt):
        """Integrate a single IMU measurement"""
        if dt <= 0 or dt > 0.5:
            return

        accel = np.array(accel)
        gyro = np.array(gyro)

        angle = gyro * dt
        angle_norm = np.linalg.norm(angle)
        dR = Rotation.from_rotvec(angle).as_matrix() if angle_norm > 1e-8 else np.eye(3)

        accel_world = self.delta_R @ accel
        self.delta_p += self.delta_v * dt + 0.5 * accel_world * dt * dt
        self.delta_v += accel_world * dt
        self.delta_R = self.delta_R @ dR
        self.dt_sum += dt


class EKFState:
    """
    Error-state EKF state (15-dimensional)
    [position(3), velocity(3), attitude_error(3), accel_bias(3), gyro_bias(3)]

    Refs:
      - Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation" (2005)
      - Geneva et al., "OpenVINS: A Research Platform for VIO" (ICRA 2020)
      - Sola et al., "A micro Lie theory for state estimation in robotics" (2018)
    """

    def __init__(self, config):
        self.config = config
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # Covariance matrix (15x15, error-state)
        self.P = np.diag([
            config["init_pos_std"]**2,  config["init_pos_std"]**2,  config["init_pos_std"]**2,
            config["init_vel_std"]**2,  config["init_vel_std"]**2,  config["init_vel_std"]**2,
            config["init_att_std"]**2,  config["init_att_std"]**2,  config["init_att_std"]**2,
            config["init_bias_std"]**2, config["init_bias_std"]**2, config["init_bias_std"]**2,
            config["init_bias_std"]**2, config["init_bias_std"]**2, config["init_bias_std"]**2,
        ])

        self.accel_noise = config["accel_noise_std"]
        self.gyro_noise = config["gyro_noise_std"]
        self.accel_bias_noise = config["accel_bias_std"]
        self.gyro_bias_noise = config["gyro_bias_std"]
        self.gravity = np.array([0.0, config.get("gravity_magnitude", 9.81), 0.0], dtype=np.float64)

        # Bias clipping limits (prevent filter divergence)
        self._max_accel_bias = config.get("max_accel_bias", 0.5)   # m/s²
        self._max_gyro_bias = config.get("max_gyro_bias", 0.15)    # rad/s

    def predict(self, accel, gyro, dt, stationary=False):
        """IMU-based state prediction (predict step)

        Q matrix notes:
          - No direct process noise on position (naturally propagated via velocity integration)
          - Velocity noise: R_prev @ diag(σ_a²) @ R_prev.T * dt²
          - Attitude noise: σ_g² * I * dt²
          - Bias random walk: σ² * dt  [previously σ² * dt² was incorrect]

        F Jacobian: linearized at pre-update orientation (R_prev)
        """
        if dt <= 0 or dt > 0.5:
            return

        # Save pre-update orientation (used for Jacobian linearization)
        R_prev = self.orientation.copy()

        accel_corr = np.array(accel) - self.accel_bias
        gyro_corr = np.array(gyro) - self.gyro_bias

        # World-frame acceleration (gravity compensated)
        accel_world = self.orientation @ accel_corr + self.gravity
        if stationary:
            # Suppress acceleration integration when stationary (used with ZUPT measurement update)
            damping = float(np.clip(self.config.get("zupt_position_damping", 1.0), 0.0, 1.0))
            accel_world = accel_world * (1.0 - damping)

        self.position += self.velocity * dt + 0.5 * accel_world * (dt * dt)
        self.velocity += accel_world * dt

        # Orientation update (Rodrigues)
        angle = gyro_corr * dt
        angle_norm = np.linalg.norm(angle)
        dR = Rotation.from_rotvec(angle).as_matrix() if angle_norm > 1e-8 else np.eye(3)
        self.orientation = self.orientation @ dR

        # ── Jacobian F (15×15) ──
        # Note: uses R_prev (linearize at pre-update orientation for consistency)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt                                     # p ← v
        F[0:3, 6:9] = -0.5 * R_prev @ _skew(accel_corr) * (dt * dt)     # p ← δθ
        F[3:6, 6:9] = -R_prev @ _skew(accel_corr) * dt                   # v ← δθ
        F[0:3, 9:12] = -R_prev * (dt * dt) * 0.5                         # p ← b_a
        F[3:6, 9:12] = -R_prev * dt                                       # v ← b_a
        F[6:9, 12:15] = -np.eye(3) * dt                                   # θ ← b_g

        # ── Process noise Q (discretized from continuous-time model) ──
        # Position: no direct noise (propagated via cross-covariance)
        # Velocity: accelerometer noise → R @ diag(σ_a²) @ R.T * dt²
        # Attitude: gyroscope noise → σ_g² * I * dt²
        # Bias: random walk → σ_rw² * dt  (not dt²!)
        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = R_prev @ (np.eye(3) * self.accel_noise**2) @ R_prev.T * (dt * dt)
        Q[6:9, 6:9] = np.eye(3) * (self.gyro_noise * dt) ** 2
        Q[9:12, 9:12] = np.eye(3) * (self.accel_bias_noise ** 2 * dt)
        Q[12:15, 12:15] = np.eye(3) * (self.gyro_bias_noise ** 2 * dt)

        self.P = F @ self.P @ F.T + Q
        # Numerical stability: maintain covariance symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def correct_pose(self, measured_position, measured_rotation, pos_noise=0.05, rot_noise=0.02):
        """Vision-based 6-DoF pose correction (position + orientation)

        Velocity is also indirectly corrected via EKF cross-covariance (P[3:6, 0:3]).
        Never modify velocity externally as it would invalidate this correction.
        """
        z_pos = measured_position - self.position
        R_err = measured_rotation @ self.orientation.T
        rot_vec_err = Rotation.from_matrix(R_err).as_rotvec()
        z = np.concatenate([z_pos, rot_vec_err])

        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)   # position observation
        H[3:6, 6:9] = np.eye(3)   # orientation error observation

        R_obs = np.diag([pos_noise**2] * 3 + [rot_noise**2] * 3)
        self._kalman_update(H, z, R_obs)

    def correct_zupt(self, vel_noise_std=0.05):
        """ZUPT (Zero-velocity UPdate) — as EKF measurement update

        Processed as EKF measurement update instead of velocity hard reset:
          - Covariance matrix is correctly updated, maintaining filter consistency
          - Velocity smoothly converges to zero (no jumps)
          - Position / attitude / bias are also indirectly corrected via cross-covariance

        Ref: Foxlin (2005), "Pedestrian Tracking with Shoe-Mounted Inertial Sensors"
        """
        z = -self.velocity  # Residual: measurement(0) - current velocity

        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)

        R_obs = np.eye(3) * vel_noise_std ** 2
        self._kalman_update(H, z, R_obs)

    def correct_nhc(self, lateral_noise_std=0.05, vertical_noise_std=0.02):
        """Non-Holonomic Constraint (NHC) — ground robots only

        Tank/wheel rovers do not slip laterally or vertically.
        Processed as EKF measurements:
          v_lateral  ≈ 0  (lateral velocity = 0)
          v_vertical ≈ 0  (vertical velocity = 0)

        → Strongly suppresses velocity drift even while moving
        → Can be applied every frame (no stationary detection required, unlike ZUPT)

        Ref: Shin et al., "Estimation Techniques for Low-Cost Inertial Navigation" (2005)
        """
        # Forward vector in world frame: transform camera z-axis to world
        forward_world = self.orientation[:, 2]   # Camera optical frame z = forward
        up_world = -self.orientation[:, 1]        # Camera optical frame -y = up

        # Constrain everything except forward velocity to zero (lateral + vertical)
        # H_lat: lateral velocity observation
        # H_vert: vertical velocity observation
        lateral_world = np.cross(up_world, forward_world)
        lateral_world /= (np.linalg.norm(lateral_world) + 1e-9)

        H = np.zeros((2, 15))
        H[0, 3:6] = lateral_world     # lateral velocity
        H[1, 3:6] = up_world          # vertical velocity

        z = np.array([
            -np.dot(self.velocity, lateral_world),   # lateral velocity residual
            -np.dot(self.velocity, up_world),         # vertical velocity residual
        ])

        R_obs = np.diag([lateral_noise_std**2, vertical_noise_std**2])
        self._kalman_update(H, z, R_obs)

    def correct_velocity(self, measured_velocity, vel_noise=0.3):
        """Process vision finite-difference velocity as EKF measurement

        Uses Kalman update instead of directly overwriting velocity → maintains filter consistency.
        Larger vel_noise = weaker constraint (0.3 m/s recommended: accounts for PnP noise propagation).
        """
        z = np.array(measured_velocity) - self.velocity

        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)

        R_obs = np.eye(3) * vel_noise ** 2
        self._kalman_update(H, z, R_obs)

    def _kalman_update(self, H, z, R_obs):
        """Common Kalman measurement update logic (Joseph form)

        Joseph form: P = (I-KH) P (I-KH)^T + K R K^T
          → Guarantees numerical positive definiteness
        """
        S = H @ self.P @ H.T + R_obs
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T

        dx = K @ z
        self.position += dx[0:3]
        self.velocity += dx[3:6]

        dtheta = dx[6:9]
        if np.linalg.norm(dtheta) > 1e-10:
            dR = Rotation.from_rotvec(dtheta).as_matrix()
            self.orientation = dR @ self.orientation

        self.accel_bias += dx[9:12]
        self.gyro_bias += dx[12:15]

        # Bias clipping (prevent divergence)
        self.accel_bias = np.clip(self.accel_bias, -self._max_accel_bias, self._max_accel_bias)
        self.gyro_bias = np.clip(self.gyro_bias, -self._max_gyro_bias, self._max_gyro_bias)

        # Covariance update (Joseph form)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_obs @ K.T
        # Numerical stability: maintain symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def normalize(self):
        """SO(3) re-normalization of rotation matrix (SVD)

        Prevents rotation matrix from drifting off the orthogonality manifold
        due to accumulated IMU integration error.
        Called periodically (every config: normalize_rotation_interval frames).
        """
        U, _, Vt = np.linalg.svd(self.orientation)
        self.orientation = U @ Vt
        # If det = -1, it's a reflection matrix → flip sign of last column
        if np.linalg.det(self.orientation) < 0:
            U[:, -1] *= -1
            self.orientation = U @ Vt


def _skew(v):
    """Skew-symmetric matrix of a 3D vector"""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0],
    ])


def _rotation_between_vectors(src, dst):
    """3×3 rotation matrix that rotates vector src to vector dst"""
    src = np.array(src, dtype=np.float64)
    dst = np.array(dst, dtype=np.float64)

    src_norm = np.linalg.norm(src)
    dst_norm = np.linalg.norm(dst)
    if src_norm < 1e-8 or dst_norm < 1e-8:
        return np.eye(3)

    src = src / src_norm
    dst = dst / dst_norm
    cross = np.cross(src, dst)
    dot = np.clip(np.dot(src, dst), -1.0, 1.0)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - src * np.dot(axis, src)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()

    vx = _skew(cross)
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (cross_norm ** 2))


class VIOTracker:
    """Visual-Inertial Odometry tracker"""

    def __init__(self, camera_intrinsics, config=None):
        """
        Args:
            camera_intrinsics: RealSense camera intrinsic parameters (rs.intrinsics)
            config: VIO configuration dict (VIO from config.py)
        """
        if config is None:
            from config import VIO as _vio_cfg
            config = _vio_cfg

        self.config = config

        # Camera matrix
        self.fx = camera_intrinsics.fx
        self.fy = camera_intrinsics.fy
        self.cx = camera_intrinsics.ppx
        self.cy = camera_intrinsics.ppy
        self.camera_matrix = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(camera_intrinsics.coeffs, dtype=np.float64)

        # FAST detector
        self.detector = cv2.FastFeatureDetector_create(
            threshold=config.get("fast_threshold", 20),
            nonmaxSuppression=True,
        )

        # LK Optical Flow parameters
        win = config.get("lk_win_size", (21, 21))
        self.lk_params = dict(
            winSize=win,
            maxLevel=config.get("lk_max_level", 3),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Pose state
        self.pose = np.eye(4)
        self.prev_gray = None
        self.prev_points = None
        self.prev_points_3d = None
        self.frames_since_keyframe = 0
        self.is_initialized = False
        self.prev_timestamp = None
        self.keyframe_pose = np.eye(4)

        # EKF
        self.ekf = EKFState(config)

        # IMU pre-integrator (reference / future extension)
        self.imu_preint = IMUPreintegrator(
            config["accel_noise_std"],
            config["gyro_noise_std"],
        )

        # IMU initialization
        self.gravity_magnitude = config.get("gravity_magnitude", 9.81)
        self.imu_init_samples_required = config.get("imu_init_samples", 200)
        self.imu_init_accel_samples = []
        self.imu_init_gyro_samples = []
        self.imu_ready = False

        # Stationary detection counter
        self._stationary_count = 0

        # SO(3) normalization counter
        self._frames_total = 0

        # Previous position + timestamp for vision-based velocity estimation
        # (timestamp must be stored together for correct dt calculation)
        self.last_vis_pos = None
        self.last_vis_timestamp = None

        # Statistics
        self.tracked_count = 0
        self.inlier_count = 0

    # ──────────────────────────────────────────────────────────────────────
    # Main update loop
    # ──────────────────────────────────────────────────────────────────────

    def update(self, color_image, depth_image, accel=None, gyro=None, timestamp=None):
        """
        Update pose with a new frame

        Args:
            color_image: BGR color image
            depth_image: depth image (16bit, unit: mm)
            accel: (x, y, z) accelerometer data (optional)
            gyro: (x, y, z) gyroscope data (optional)
            timestamp: frame timestamp (ms)

        Returns:
            pose: 4×4 transformation matrix (camera → world)
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        depth_m = depth_image.astype(np.float32) * 0.001  # mm → m

        # dt calculation
        dt = 0.0
        if timestamp is not None and self.prev_timestamp is not None:
            dt = (timestamp - self.prev_timestamp) * 0.001  # ms → s
            if dt < 0 or dt > 1.0:
                dt = 0.0

        # IMU coordinate transform:
        # RealSense IMU (x-right, y-up, z-back) → Camera Optical (x-right, y-down, z-forward)
        accel_cam = None
        gyro_cam = None
        if accel is not None:
            accel_cam = np.array([accel[0], -accel[1], -accel[2]], dtype=np.float64)
        if gyro is not None:
            gyro_cam = np.array([gyro[0], -gyro[1], -gyro[2]], dtype=np.float64)

        # ── IMU initialization ──
        if accel_cam is not None and gyro_cam is not None:
            self._accumulate_imu_init(accel_cam, gyro_cam)
            if not self.imu_ready and self._try_initialize_imu():
                self.pose[:3, :3] = self.ekf.orientation
                # Coordinate frame changed — force new keyframe on next frame
                self.is_initialized = False
                self.prev_points = None

        # ── IMU Predict ──
        has_imu = accel_cam is not None and gyro_cam is not None
        if has_imu and dt > 0 and self.imu_ready:
            if self._is_stationary(accel_cam, gyro_cam):
                self._stationary_count += 1
            else:
                self._stationary_count = 0

            zupt_min = self.config.get("zupt_min_frames", 3)
            stationary = self._stationary_count >= zupt_min

            # Predict (suppress acceleration integration when stationary)
            self.ekf.predict(accel_cam, gyro_cam, dt, stationary=stationary)

            # ZUPT: EKF measurement update instead of hard reset (maintains covariance consistency)
            if stationary:
                zupt_noise = self.config.get("zupt_vel_noise", 0.05)
                self.ekf.correct_zupt(zupt_noise)

            # NHC: ground robot non-holonomic constraint (applied every frame even while moving)
            # Tank rover has lateral/vertical velocity ≈ 0 → constrains drift to forward direction only
            if self.config.get("enable_nhc", False):
                self.ekf.correct_nhc(
                    lateral_noise_std=self.config.get("nhc_lateral_noise", 0.05),
                    vertical_noise_std=self.config.get("nhc_vertical_noise", 0.02),
                )

        # ── First frame initialization ──
        if not self.is_initialized:
            if self.imu_ready:
                self.pose[:3, :3] = self.ekf.orientation
                self.pose[:3, 3] = self.ekf.position
            self._init_frame(gray, depth_m)
            self.prev_timestamp = timestamp
            return self.pose.copy()

        # ── Optical Flow tracking ──
        tracked_2d, tracked_3d, status = self._track_features(gray)
        self.tracked_count = len(tracked_2d) if tracked_2d is not None else 0

        # ── Keyframe decision ──
        need_keyframe = self._need_keyframe(tracked_2d)

        # ── Pose estimation ──
        if tracked_2d is not None and len(tracked_2d) >= self.config.get("pnp_min_inliers", 10):
            success, R_est, t_est, inliers = self._estimate_pose(tracked_2d, tracked_3d)

            if success:
                self.inlier_count = len(inliers)

                # Adaptive observation noise based on inlier ratio
                # More inliers → higher confidence → smaller noise → stronger correction
                inlier_ratio = len(inliers) / max(len(tracked_2d), 1)
                noise_scale = float(np.clip(1.0 / (inlier_ratio + 0.1), 0.5, 3.0))
                base_pos_noise = self.config.get("pnp_pos_noise", 0.03)
                base_rot_noise = self.config.get("pnp_rot_noise", 0.015)
                eff_pos_noise = base_pos_noise * noise_scale
                eff_rot_noise = base_rot_noise * noise_scale

                if has_imu and dt > 0 and self.imu_ready:
                    # ── PnP result validity check ──
                    # If too far from EKF prediction, PnP converged to wrong solution → reject
                    max_pos_jump = self.config.get("pnp_max_position_jump", 0.5)  # m
                    pos_deviation = float(np.linalg.norm(t_est.flatten() - self.ekf.position))
                    if pos_deviation > max_pos_jump:
                        # PnP outlier → keep IMU prediction, skip correction this frame
                        self._damp_velocity_without_vision()
                        self.pose[:3, :3] = self.ekf.orientation
                        self.pose[:3, 3] = self.ekf.position
                        need_keyframe = True
                    else:
                        # 1) 6-DoF pose correction
                        self.ekf.correct_pose(
                            t_est.flatten(), R_est,
                            pos_noise=eff_pos_noise,
                            rot_noise=eff_rot_noise,
                        )

                        # 2) Additional velocity correction using visual velocity as EKF measurement
                        # Key: use last_vis_timestamp for accurate dt calculation
                        # (use actual time difference between two successful PnP calls, not frame dt)
                        if (self.last_vis_pos is not None
                                and self.last_vis_timestamp is not None
                                and timestamp is not None):
                            vel_dt = (timestamp - self.last_vis_timestamp) * 0.001
                            if 0.005 < vel_dt < 0.5:  # Trust only 5ms~500ms range
                                vis_vel = (t_est.flatten() - self.last_vis_pos) / vel_dt
                                # Reject unrealistic velocities (max allowed: 3 m/s)
                                if np.linalg.norm(vis_vel) < self.config.get("max_visual_vel", 3.0):
                                    vis_vel_noise = self.config.get("visual_vel_noise", 0.3)
                                    self.ekf.correct_velocity(vis_vel, vis_vel_noise)

                        self.last_vis_pos = t_est.flatten().copy()
                        self.last_vis_timestamp = timestamp
                        self.pose[:3, :3] = self.ekf.orientation
                        self.pose[:3, 3] = self.ekf.position
                else:
                    # No IMU: use vision pose directly
                    self.pose[:3, :3] = R_est
                    self.pose[:3, 3] = t_est.flatten()
                    self.ekf.position = t_est.flatten().copy()
                    self.ekf.orientation = R_est.copy()
            else:
                # PnP failed → keep IMU prediction, damp velocity, force keyframe
                if self.imu_ready:
                    self._damp_velocity_without_vision()
                    self.pose[:3, :3] = self.ekf.orientation
                    self.pose[:3, 3] = self.ekf.position
                need_keyframe = True
        else:
            # Tracking failed
            if self.imu_ready:
                self._damp_velocity_without_vision()
                self.pose[:3, :3] = self.ekf.orientation
                self.pose[:3, 3] = self.ekf.position
            need_keyframe = True

        # ── Keyframe update ──
        if need_keyframe:
            self._init_frame(gray, depth_m)
        else:
            self.prev_gray = gray
            self.prev_points = tracked_2d
            self.prev_points_3d = tracked_3d
            self.frames_since_keyframe += 1

        # ── Periodic SO(3) normalization ──
        self._frames_total += 1
        if self._frames_total % self.config.get("normalize_rotation_interval", 30) == 0:
            self.ekf.normalize()

        self.prev_timestamp = timestamp
        return self.pose.copy()

    # ──────────────────────────────────────────────────────────────────────
    # IMU initialization
    # ──────────────────────────────────────────────────────────────────────

    def _accumulate_imu_init(self, accel_cam, gyro_cam):
        """Collect IMU mean values from initial stationary period"""
        if self.imu_ready:
            return

        accel_norm = np.linalg.norm(accel_cam)
        gyro_norm = np.linalg.norm(gyro_cam)
        if abs(accel_norm - self.gravity_magnitude) > 1.5 or gyro_norm > 0.3:
            # Motion detected → reset collection (only collect while stationary)
            self.imu_init_accel_samples.clear()
            self.imu_init_gyro_samples.clear()
            return

        self.imu_init_accel_samples.append(accel_cam.copy())
        self.imu_init_gyro_samples.append(gyro_cam.copy())

        max_keep = max(self.imu_init_samples_required, 1)
        if len(self.imu_init_accel_samples) > max_keep:
            self.imu_init_accel_samples.pop(0)
        if len(self.imu_init_gyro_samples) > max_keep:
            self.imu_init_gyro_samples.pop(0)

    def _try_initialize_imu(self):
        """Initialize orientation / bias from initial IMU mean"""
        if self.imu_ready or len(self.imu_init_accel_samples) < self.imu_init_samples_required:
            return False

        accel_samples = np.array(self.imu_init_accel_samples)
        gyro_samples = np.array(self.imu_init_gyro_samples)

        # Check variance: reset collection if too noisy
        accel_std = np.std(accel_samples, axis=0)
        if np.max(accel_std) > 0.4:
            self.imu_init_accel_samples.clear()
            self.imu_init_gyro_samples.clear()
            return False

        accel_mean = np.mean(accel_samples, axis=0)
        gyro_mean = np.mean(gyro_samples, axis=0)
        accel_norm = np.linalg.norm(accel_mean)
        if accel_norm < 1e-6:
            return False

        # Align initial orientation with gravity direction
        # accel_mean when stationary = specific force = opposite of gravity (points up in camera optical frame)
        # target_up = [0, -1, 0]: up in optical frame = -Y
        target_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        self.ekf.orientation = _rotation_between_vectors(accel_mean, target_up)

        # Initialize gyroscope bias (measurement when stationary = bias)
        self.ekf.gyro_bias = gyro_mean.copy()

        # Initialize accelerometer bias
        # expected specific force = gravity magnitude in the measured direction
        expected_sf = accel_mean / accel_norm * self.gravity_magnitude
        self.ekf.accel_bias = accel_mean - expected_sf

        self.imu_ready = True
        return True

    def _is_stationary(self, accel_cam, gyro_cam):
        """Detect stationary state after bias correction"""
        accel_unbiased = accel_cam - self.ekf.accel_bias
        gyro_unbiased = gyro_cam - self.ekf.gyro_bias
        accel_norm = np.linalg.norm(accel_unbiased)
        gyro_norm = np.linalg.norm(gyro_unbiased)

        accel_tol = self.config.get("stationary_accel_tol", 0.35)
        gyro_tol = self.config.get("stationary_gyro_tol", 0.08)
        return (
            abs(accel_norm - self.gravity_magnitude) < accel_tol
            and gyro_norm < gyro_tol
        )

    # ──────────────────────────────────────────────────────────────────────
    # Feature management
    # ──────────────────────────────────────────────────────────────────────

    def _init_frame(self, gray, depth_m):
        """Keyframe initialization: FAST feature detection + 3D coordinate computation"""
        keypoints = self.detector.detect(gray)
        if len(keypoints) == 0:
            return

        max_feat = self.config.get("max_features", 300)
        if len(keypoints) > max_feat:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_feat]

        points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        valid_2d, valid_3d = self._filter_with_depth(points_2d, depth_m)

        if len(valid_2d) >= self.config.get("pnp_min_inliers", 10):
            self.prev_gray = gray
            self.prev_points = valid_2d
            self.prev_points_3d = valid_3d
            self.keyframe_pose = self.pose.copy()
            self.frames_since_keyframe = 0
            self.is_initialized = True

    def _filter_with_depth(self, points_2d, depth_m):
        """Select only features with valid depth, compute 3D coordinates

        Improvements: vectorized processing + neighbor pixel median depth (reduces single-pixel noise)
        """
        d_min = self.config.get("depth_min", 0.3)
        d_max = self.config.get("depth_max", 5.0)
        win = self.config.get("depth_sample_window", 3)
        h, w = depth_m.shape

        # Median depth sampling (using neighbor pixels)
        half = win // 2
        us = np.round(points_2d[:, 0]).astype(int)
        vs = np.round(points_2d[:, 1]).astype(int)

        depths = np.zeros(len(points_2d), dtype=np.float32)
        for i in range(len(points_2d)):
            u, v = us[i], vs[i]
            if not (0 <= u < w and 0 <= v < h):
                continue
            u0, u1 = max(0, u - half), min(w, u + half + 1)
            v0, v1 = max(0, v - half), min(h, v + half + 1)
            patch = depth_m[v0:v1, u0:u1].ravel()
            valid_patch = patch[(patch > d_min) & (patch < d_max)]
            if len(valid_patch) >= 3:
                depths[i] = float(np.median(valid_patch))

        # Valid depth mask
        valid_mask = depths > 0.0
        if not np.any(valid_mask):
            return np.array([], dtype=np.float32), np.array([], dtype=np.float64)

        pts = points_2d[valid_mask]
        z = depths[valid_mask].astype(np.float64)

        x = (pts[:, 0] - self.cx) * z / self.fx
        y = (pts[:, 1] - self.cy) * z / self.fy
        points_3d = np.column_stack([x, y, z])

        return pts.astype(np.float32), points_3d.astype(np.float64)

    def _track_features(self, gray):
        """Track previous features with LK Optical Flow (with forward-backward verification)"""
        if self.prev_points is None or len(self.prev_points) == 0:
            return None, None, None

        pts = self.prev_points.reshape(-1, 1, 2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts, None, **self.lk_params
        )

        if next_pts is None:
            return None, None, status

        status = status.flatten().astype(bool)

        # Backward verification (forward-backward consistency check)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, next_pts, None, **self.lk_params
        )
        if back_pts is not None:
            back_status = back_status.flatten().astype(bool)
            fb_dist = np.linalg.norm(
                pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            fb_good = fb_dist < 1.0
            status = status & back_status & fb_good

        tracked_2d = next_pts.reshape(-1, 2)[status]
        tracked_3d = self.prev_points_3d[status]

        return tracked_2d, tracked_3d, status

    def _need_keyframe(self, tracked_2d):
        """Determine whether a new keyframe is needed

        Translation-based trigger removed:
          self.pose[:3,3] contains EKF drift, so drift itself would trigger
          keyframes → drift coordinate system becomes fixed → further drift amplified.
          Translation trigger is not used.

        Rotation trigger retained:
          Rotation is directly estimated by PnP and is less affected by drift.
        """
        if tracked_2d is None:
            return True

        min_feat = self.config.get("keyframe_min_features", 80)
        max_interval = self.config.get("keyframe_max_interval", 10)
        max_rot_deg = self.config.get("keyframe_max_rotation_deg", 15.0)

        if len(tracked_2d) < min_feat:
            return True
        if self.frames_since_keyframe >= max_interval:
            return True

        # Rotation-based trigger (rotation is less affected by drift)
        R_delta = self.pose[:3, :3] @ self.keyframe_pose[:3, :3].T
        trace_val = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(trace_val)))

        if rotation_deg > max_rot_deg:
            return True

        return False

    def _estimate_pose(self, points_2d, points_3d):
        """Camera pose estimation with PnP + RANSAC

        Improvement: uses IMU EKF predicted orientation as initial guess
                     → faster convergence during fast motion, avoids local minima
        """
        if len(points_2d) < 4:
            return False, None, None, None

        # Transform 3D points to world frame using keyframe pose
        R_kf = self.keyframe_pose[:3, :3]
        t_kf = self.keyframe_pose[:3, 3]
        world_3d = (R_kf @ points_3d.T).T + t_kf

        # Use IMU EKF predicted orientation as PnP initial guess
        use_imu_prior = self.imu_ready
        rvec_init = tvec_init = None
        if use_imu_prior:
            # EKF holds camera→world; solvePnP requires world→camera
            R_w2c = self.ekf.orientation.T
            t_w2c = -(self.ekf.orientation.T @ self.ekf.position)
            rvec_init, _ = cv2.Rodrigues(R_w2c)
            tvec_init = t_w2c.reshape(3, 1)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_3d.astype(np.float64),
            points_2d.astype(np.float64),
            self.camera_matrix,
            self.dist_coeffs,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=use_imu_prior,
            reprojectionError=self.config.get("pnp_reproj_threshold", 3.0),
            confidence=self.config.get("pnp_confidence", 0.99),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None:
            return False, None, None, None
        if len(inliers) < self.config.get("pnp_min_inliers", 10):
            return False, None, None, None

        R_cam, _ = cv2.Rodrigues(rvec)
        t_cam = tvec.flatten()

        # solvePnP returns world→camera transform → invert to camera→world
        R_world = R_cam.T
        t_world = -R_cam.T @ t_cam

        return True, R_world, t_world.reshape(3, 1), inliers.flatten()

    def _damp_velocity_without_vision(self):
        """Reduce inertial drift when vision correction is unavailable

        0.85^30fps ≈ 0.004/s → velocity decays to ~0 within 0.5s (strengthened from 0.98 → 0.85)
        """
        damping = float(np.clip(self.config.get("no_vision_velocity_damping", 0.85), 0.0, 1.0))
        self.ekf.velocity *= damping

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def get_position(self):
        """Return current position (x, y, z) in meters"""
        return self.pose[:3, 3].copy()

    def get_rotation(self):
        """Return current rotation matrix (3×3)"""
        return self.pose[:3, :3].copy()

    def get_pose(self):
        """Return current 6DoF pose (4×4 transformation matrix)"""
        return self.pose.copy()

    def get_euler_degrees(self):
        """Return current orientation as Euler angles (roll, pitch, yaw) in degrees"""
        r = Rotation.from_matrix(self.pose[:3, :3])
        return r.as_euler('xyz', degrees=True)

    def get_stats(self):
        """Return tracking status statistics"""
        return {
            'tracked_features': self.tracked_count,
            'inliers': self.inlier_count,
            'frames_since_keyframe': self.frames_since_keyframe,
            'initialized': self.is_initialized,
            'imu_ready': self.imu_ready,
            'stationary': self._stationary_count >= self.config.get("zupt_min_frames", 3),
            'ekf_vel_norm': float(np.linalg.norm(self.ekf.velocity)),
        }

    def reset(self):
        """Reset pose"""
        self.pose = np.eye(4)
        self.prev_gray = None
        self.prev_points = None
        self.prev_points_3d = None
        self.frames_since_keyframe = 0
        self.is_initialized = False
        self.prev_timestamp = None
        self.ekf = EKFState(self.config)
        self.imu_preint.reset()
        self.imu_init_accel_samples.clear()
        self.imu_init_gyro_samples.clear()
        self.imu_ready = False
        self.keyframe_pose = np.eye(4)
        self._stationary_count = 0
        self._frames_total = 0
        self.last_vis_pos = None
        self.last_vis_timestamp = None
        self.tracked_count = 0
        self.inlier_count = 0
