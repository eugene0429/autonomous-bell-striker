"""
Dummy Target Provider — for pipeline integration testing before YOLO training.

A stub that bypasses the real detection pipeline (YOLO + depth + frame averaging
+ camera→world transform) and returns the Phase 1 / Phase 2 target coordinates
*directly*.

Once YOLO training is done, replace it with the real chain of
[perception/detection/detector.py](detector.py) +
[perception/detection/position_estimator.py](position_estimator.py). At that
point this class is no longer needed.

Coordinate frames
-----------------
- get_phase1_target()  : world frame (x, y) [m]
                         robot starting pose = origin, world_x = camera forward.
- get_phase2_target()  : plate-base frame (x, y, z) [m]
                         bell position seen from the plate center (0, 0, H0=Lc).
                         (in the real system, a value with the Camera→Plate extrinsic transform applied)

Oscillating bell simulation
--------------------------
On each call, get_phase2_target() adds z noise of up to phase2_jitter (m) before
returning (simulating a bell oscillating vertically at 3 m height). This lets you
verify that the "re-estimate the 3D vector right before each strike" logic works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DummyTargetConfig:
    # ── Phase 1: ground-projection coordinates of the bell base (world frame) ──
    phase1_target: Tuple[float, float] = (3.0, 2.0)        # (x, y) [m]

    # ── Phase 2: 3D vector from the plate center → the bell ──
    phase2_target: Tuple[float, float, float] = (0.10, 0.00, 3.00)  # (x, y, z) [m]
    phase2_jitter: float = 0.05                            # ±jitter z noise [m]
    phase2_jitter_seed: int = 42

    # ── Phase 1 multi-frame averaging simulation (optional) ──
    phase1_noise_std: float = 0.0                          # single-detection noise std [m]
    phase1_avg_frames: int = 1                             # number of frames used for averaging

    # ── visual-servo sim (Phase 1 bypass) ──
    bell_height_m: float = 3.0                    # mean ground-frame z of the bell
    camera_height_m: float = 0.30                 # ground-frame z of camera
    fx: float = 615.0                             # focal length [px], D435i color stream
    fy: float = 615.0
    img_w: int = 640
    img_h: int = 480
    bbox_pixels: int = 80                         # synthetic bbox side length
    vs_bbox_noise_px: float = 0.0
    vs_depth_noise_m: float = 0.0
    vs_dropout_prob: float = 0.0                  # probability a frame returns None

    # ── bell vertical oscillation (spec §9) ──
    # amp=0 → bell stationary (default). amp>0 → on reaching each endpoint, resample
    # the traverse time from a uniform (lo, hi) distribution → speed = amp / traverse_time,
    # oscillating back and forth in +/- direction.
    bell_height_amp_m: float = 0.0                          # peak-to-peak [m]
    bell_endpoint_period_s: Tuple[float, float] = (0.5, 2.5)
    bell_dt_s: float = 0.067                                # advance dt per call


class DummyTargetProvider:
    """Target coordinate provider for pipeline integration testing."""

    def __init__(self, cfg: DummyTargetConfig | None = None):
        self.cfg = cfg if cfg is not None else DummyTargetConfig()
        self._rng = np.random.default_rng(self.cfg.phase2_jitter_seed)
        # bell vertical motion state (spec §9 — amp=0 → no motion)
        self._bell_offset_m: float = 0.0       # offset from bell_height_m
        self._bell_dir: float = 1.0            # +1 ascending, -1 descending
        self._bell_speed_m_per_s: float = 0.0  # |dz/dt| in current traverse
        if self.cfg.bell_height_amp_m > 0:
            self._start_new_traverse()

    # ── bell oscillation helpers ──
    def _start_new_traverse(self) -> None:
        """At each endpoint, sample a new traverse time → set speed."""
        lo, hi = self.cfg.bell_endpoint_period_s
        traverse_s = float(self._rng.uniform(lo, hi))
        # Speed so that we cover full amp in `traverse_s` seconds
        self._bell_speed_m_per_s = self.cfg.bell_height_amp_m / max(traverse_s, 1e-6)

    def _advance_bell(self) -> None:
        """Step bell offset by one `bell_dt_s`; clamp + reverse at endpoints."""
        c = self.cfg
        if c.bell_height_amp_m <= 0:
            return
        self._bell_offset_m += self._bell_dir * self._bell_speed_m_per_s * c.bell_dt_s
        half = c.bell_height_amp_m / 2.0
        if self._bell_offset_m > half:
            self._bell_offset_m = half
            self._bell_dir = -1.0
            self._start_new_traverse()
        elif self._bell_offset_m < -half:
            self._bell_offset_m = -half
            self._bell_dir = 1.0
            self._start_new_traverse()

    # ── Phase 1 ──
    def get_phase1_target(self) -> Tuple[float, float]:
        """
        Bell's world (x, y), estimated by YOLO multi-frame averaging right before departure.

        In the real implementation:
          for _ in range(N): bbox = yolo.detect(frame); xyz = depth_deproj(bbox);
          xy_world = T_world_cam @ xyz   →   average
        """
        c = self.cfg
        if c.phase1_noise_std <= 0 or c.phase1_avg_frames <= 1:
            return c.phase1_target

        # multi-frame averaging simulation
        samples = np.array([
            (c.phase1_target[0] + self._rng.normal(0, c.phase1_noise_std),
             c.phase1_target[1] + self._rng.normal(0, c.phase1_noise_std))
            for _ in range(c.phase1_avg_frames)
        ])
        avg = samples.mean(axis=0)
        return (float(avg[0]), float(avg[1]))

    # ── Phase 2 ──
    def get_phase2_target(self) -> Tuple[float, float, float]:
        """
        3D vector to the bell relative to the plate center (z oscillation jitter applied on each call).

        In the real implementation:
          bbox  = yolo.detect(frame_after_tilt)
          xyz_c = depth_deproject(bbox)
          xyz_p = T_plate_cam @ xyz_c        # camera → plate extrinsic transform
        """
        c = self.cfg
        x, y, z = c.phase2_target
        if c.phase2_jitter > 0:
            z = z + float(self._rng.uniform(-c.phase2_jitter, c.phase2_jitter))
        return (float(x), float(y), float(z))

    # ── Phase 1 visual-servo synthesis ──
    def get_visual_servo_detection(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        tilt_deg: float,
    ):
        """Synthesize a YOLO-like detection dict from current pose + tilt.

        Each call advances bell vertical motion by `cfg.bell_dt_s` (no-op if
        `bell_height_amp_m == 0`). Returns {bbox, conf, depth_m} or None if
        the target is out of FOV / randomly dropped per cfg.vs_dropout_prob.
        """
        c = self.cfg
        # Advance bell oscillation FIRST (drives current tz)
        self._advance_bell()
        # Dropout simulation
        if c.vs_dropout_prob > 0 and self._rng.random() < c.vs_dropout_prob:
            return None

        # Target ground-frame position from phase1_target + bell motion
        tx, ty = c.phase1_target
        tz = c.bell_height_m + self._bell_offset_m

        # Robot → bell vector in world frame
        dx = tx - robot_x
        dy = ty - robot_y
        dz = tz - c.camera_height_m

        # Express in robot body frame (forward = +x_body)
        cth = np.cos(robot_theta); sth = np.sin(robot_theta)
        x_body =  cth * dx + sth * dy
        y_body = -sth * dx + cth * dy
        z_body = dz

        # Apply tilt (pitch up by tilt_deg): rotate around y_body
        t = np.deg2rad(tilt_deg)
        x_cam =  np.cos(t) * x_body + np.sin(t) * z_body
        z_cam = -np.sin(t) * x_body + np.cos(t) * z_body
        y_cam =  y_body

        # Behind camera or non-positive depth → not visible
        # Camera convention: +Z forward, +X right, +Y down (OpenCV pinhole)
        Z = x_cam   # axis pointing forward through camera
        X = -y_cam  # body-y is robot-LEFT (ROS); camera-X is image-RIGHT — flip sign
        Y = -z_cam  # camera-down corresponds to -z_body after tilt
        if Z <= 0.1:
            return None

        u = c.fx * (X / Z) + c.img_w / 2.0
        v = c.fy * (Y / Z) + c.img_h / 2.0

        if c.vs_bbox_noise_px > 0:
            u += self._rng.normal(0.0, c.vs_bbox_noise_px)
            v += self._rng.normal(0.0, c.vs_bbox_noise_px)

        # FOV check
        bw = c.bbox_pixels
        if not (bw / 2 <= u <= c.img_w - bw / 2 and
                bw / 2 <= v <= c.img_h - bw / 2):
            return None

        depth_m = float(Z)
        if c.vs_depth_noise_m > 0:
            depth_m += self._rng.normal(0.0, c.vs_depth_noise_m)
            depth_m = max(0.05, depth_m)

        return {
            "bbox": (int(u - bw/2), int(v - bw/2), int(u + bw/2), int(v + bw/2)),
            "conf": 0.95,
            "depth_m": depth_m,
        }
