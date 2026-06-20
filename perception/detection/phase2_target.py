"""Phase 2 target estimator — bell 3D measurement in plate frame.

Camera→Plate coordinate transform + single-frame estimator + 1-second
multi-frame aggregation. Replaces `DummyTargetProvider.get_phase2_target()`
stub in real mode.

See: docs/SW_ARCHITECTURE.md §5-6 (Phase 2 aiming, Camera->Plate transform)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class Phase2MeasurementError(RuntimeError):
    """Raised when a measurement window yields zero valid detections."""

from .visual_servo_target import compute_target_depth


@dataclass
class CameraToPlateExtrinsic:
    """Fixed extrinsic from camera optical frame → plate frame.

    Plate frame: +X forward, +Y left, +Z up. Origin = plate center.
    Camera frame (RealSense): +Z optical axis (out of lens),
                              +X image right, +Y image down.

    Default: natural mounting (camera roll 0°, tilted 90° about plate +Y).
    Lens at (0.20, 0, -0.10) m in plate frame. If post-calibration shows
    image-right ≠ plate -Y or image-down ≠ plate +X, flip the corresponding
    sign field. `t_y_m` accounts for RealSense RGB-vs-depth lateral offset
    (RGB sensor sits a few mm off the depth module centerline).
    """

    t_x_m: float = 0.20
    t_y_m: float = 0.0
    t_z_m: float = -0.10
    image_right_sign: int = -1   # +1 if image-right → plate +Y; -1 default
    image_down_sign:  int = +1   # +1 if image-down → plate +X (default)

    def transform(self, p_cam: np.ndarray) -> np.ndarray:
        Xc, Yc, Zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        return np.array([
            self.image_down_sign  * Yc + self.t_x_m,
            self.image_right_sign * Xc + self.t_y_m,
            Zc + self.t_z_m,
        ])


class Phase2TargetEstimator:
    """Single frame → plate-frame 3D point.

    Pipeline: detector.detect(color) → pick top-1 conf above threshold →
    ROI-median depth via compute_target_depth → deproject bbox center with
    that depth → extrinsic transform to plate frame.

    Returns None if any step fails (no detection, low conf, no valid depth).
    """

    def __init__(
        self,
        camera,
        detector,
        extrinsic: CameraToPlateExtrinsic,
        roi_frac: float = 0.4,
        min_conf: float = 0.5,
    ):
        self.camera = camera
        self.detector = detector
        self.extrinsic = extrinsic
        self.roi_frac = roi_frac
        self.min_conf = min_conf

    def estimate(self, color: np.ndarray, depth_image: np.ndarray) -> Optional[np.ndarray]:
        detections = self.detector.detect(color)
        if not detections:
            return None

        best = max(detections, key=lambda d: d.get("conf", 0.0))
        if best.get("conf", 0.0) < self.min_conf:
            return None

        bbox = best["bbox"]
        depth_m = compute_target_depth(depth_image, bbox, roi_frac=self.roi_frac)
        if depth_m is None:
            return None

        x1, y1, x2, y2 = bbox
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        p_cam = self.camera.pixel_to_3d_with_depth(cx, cy, depth_m)
        if p_cam is None:
            return None

        return self.extrinsic.transform(np.asarray(p_cam, dtype=float))


class RealPhase2TargetProvider:
    """1-second measurement window → per-axis median plate-frame target.

    pipeline.CapstonePipeline calls get_phase2_target() once per shot.
    Same signature as DummyTargetProvider.get_phase2_target().

    Args:
        camera: RealSenseCamera (exposes get_frames() -> (color, depth, frame))
        estimator: Phase2TargetEstimator
        measurement_duration_s: window length (default 1.0 s)
        min_valid_frames: warn-threshold; below this, warning logged but
                         still proceeds with whatever samples were collected.
                         Mission aborts only when zero valid frames.
        time_source: callable() -> float (defaults to time.monotonic).
                    Tests inject a FakeClock for determinism.
    """

    def __init__(
        self,
        camera,
        estimator: Phase2TargetEstimator,
        measurement_duration_s: float = 1.0,
        min_valid_frames: int = 15,
        time_source: Callable[[], float] = None,
    ):
        self.camera = camera
        self.estimator = estimator
        self.measurement_duration_s = measurement_duration_s
        self.min_valid_frames = min_valid_frames
        self._time = time_source if time_source is not None else time.monotonic

    def get_phase2_target(self) -> Tuple[float, float, float]:
        samples = []
        t_start = self._time()

        while self._time() - t_start < self.measurement_duration_s:
            color, depth_image, _depth_frame = self.camera.get_frames()
            if color is None:
                continue
            p_plate = self.estimator.estimate(color, depth_image)
            if p_plate is not None:
                samples.append(p_plate)

        n_valid = len(samples)
        if n_valid == 0:
            raise Phase2MeasurementError(
                f"no valid detections in {self.measurement_duration_s:.2f}s window"
            )
        if n_valid < self.min_valid_frames:
            log.warning(
                "only %d valid frames (< %d) — proceeding with current samples",
                n_valid, self.min_valid_frames,
            )

        target_xyz = np.median(np.stack(samples), axis=0)
        return tuple(float(v) for v in target_xyz)
