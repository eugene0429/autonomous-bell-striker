"""Tests for perception.detection.phase2_target."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from perception.detection.phase2_target import (
    CameraToPlateExtrinsic,
    Phase2MeasurementError,
    Phase2TargetEstimator,
    RealPhase2TargetProvider,
)


class TestCameraToPlateExtrinsic:
    """Spec §3 sanity check + sign-flip cases."""

    def test_camera_origin_maps_to_lens_position(self):
        e = CameraToPlateExtrinsic()
        out = e.transform(np.array([0.0, 0.0, 0.0]))
        assert out == pytest.approx([0.20, 0.0, -0.10])

    def test_directly_above_lens(self):
        e = CameraToPlateExtrinsic()
        out = e.transform(np.array([0.0, 0.0, 3.0]))
        assert out == pytest.approx([0.20, 0.0, 2.90])

    def test_directly_above_plate_center(self):
        # Plate center is 20cm behind lens along +X. So plate (0,0,3.0)
        # appears ~20cm "above" image center after the 90° tilt.
        e = CameraToPlateExtrinsic()
        out = e.transform(np.array([0.0, -0.20, 3.10]))
        assert out == pytest.approx([0.0, 0.0, 3.0])

    def test_image_right_maps_to_robot_right(self):
        e = CameraToPlateExtrinsic()
        out = e.transform(np.array([0.10, 0.0, 1.0]))
        assert out == pytest.approx([0.20, -0.10, 0.90])

    def test_image_down_maps_to_robot_forward(self):
        e = CameraToPlateExtrinsic()
        out = e.transform(np.array([0.0, 0.10, 1.0]))
        assert out == pytest.approx([0.30, 0.0, 0.90])

    def test_image_right_sign_flip_inverts_plate_y(self):
        e = CameraToPlateExtrinsic(image_right_sign=+1)
        out = e.transform(np.array([0.10, 0.0, 1.0]))
        assert out == pytest.approx([0.20, +0.10, 0.90])

    def test_image_down_sign_flip_inverts_plate_x_contribution(self):
        e = CameraToPlateExtrinsic(image_down_sign=-1)
        out = e.transform(np.array([0.0, 0.10, 1.0]))
        assert out == pytest.approx([0.10, 0.0, 0.90])   # 0.20 - 0.10

    def test_custom_translation(self):
        e = CameraToPlateExtrinsic(t_x_m=0.15, t_z_m=-0.05)
        out = e.transform(np.array([0.0, 0.0, 1.0]))
        assert out == pytest.approx([0.15, 0.0, 0.95])


def _make_estimator(detector, depth_m_value, roi_frac=0.4, min_conf=0.5):
    """Build estimator with mocked camera + detector.

    The camera mock returns a fixed deprojected point so the test can
    verify the extrinsic transform end-to-end without real intrinsics.
    """
    camera = MagicMock()
    # We assert on the args passed to this — return value reused per test
    camera.pixel_to_3d_with_depth.return_value = (0.0, 0.0, depth_m_value)
    extrinsic = CameraToPlateExtrinsic()
    return Phase2TargetEstimator(
        camera=camera, detector=detector,
        extrinsic=extrinsic, roi_frac=roi_frac, min_conf=min_conf,
    ), camera


class TestPhase2TargetEstimator:
    """Single-frame: detect + ROI median depth + extrinsic transform."""

    def _depth_image(self, value_mm=2000):
        """100x100 uniform-depth synthetic image."""
        return np.full((100, 100), value_mm, dtype=np.uint16)

    def _detector_with_bbox(self, bbox=(30, 30, 70, 70), conf=0.9):
        det = MagicMock()
        det.detect.return_value = [{"bbox": bbox, "conf": conf}]
        return det

    def test_happy_path_returns_plate_frame_point(self):
        detector = self._detector_with_bbox()
        estimator, camera = _make_estimator(detector, depth_m_value=2.0)
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = self._depth_image(value_mm=2000)

        out = estimator.estimate(color, depth)

        # Camera mock returned (0, 0, 2.0). Extrinsic default:
        # plate = (Y_cam + 0.20, -X_cam, Z_cam - 0.10) = (0.20, 0, 1.90)
        assert out is not None
        assert out == pytest.approx([0.20, 0.0, 1.90])
        # Verify bbox center (50, 50) was used as deprojection pixel,
        # ROI median depth (2.0 m from uniform 2000mm) was used as depth.
        camera.pixel_to_3d_with_depth.assert_called_once_with(50, 50, 2.0)

    def test_no_detections_returns_none(self):
        detector = MagicMock()
        detector.detect.return_value = []
        estimator, _ = _make_estimator(detector, depth_m_value=2.0)

        assert estimator.estimate(
            np.zeros((100, 100, 3), dtype=np.uint8),
            self._depth_image(),
        ) is None

    def test_low_confidence_returns_none(self):
        detector = self._detector_with_bbox(conf=0.3)  # below 0.5 default
        estimator, _ = _make_estimator(detector, depth_m_value=2.0)

        assert estimator.estimate(
            np.zeros((100, 100, 3), dtype=np.uint8),
            self._depth_image(),
        ) is None

    def test_depth_all_holes_returns_none(self):
        # depth image of all zeros → compute_target_depth returns None
        detector = self._detector_with_bbox()
        estimator, _ = _make_estimator(detector, depth_m_value=2.0)
        depth = np.zeros((100, 100), dtype=np.uint16)

        assert estimator.estimate(
            np.zeros((100, 100, 3), dtype=np.uint8),
            depth,
        ) is None

    def test_picks_highest_confidence_detection(self):
        detector = MagicMock()
        detector.detect.return_value = [
            {"bbox": (10, 10, 30, 30), "conf": 0.6},
            {"bbox": (30, 30, 70, 70), "conf": 0.9},   # winner
            {"bbox": (60, 60, 90, 90), "conf": 0.7},
        ]
        estimator, camera = _make_estimator(detector, depth_m_value=2.0)
        depth = self._depth_image(value_mm=2000)

        estimator.estimate(np.zeros((100, 100, 3), dtype=np.uint8), depth)
        # winner bbox center is (50, 50)
        camera.pixel_to_3d_with_depth.assert_called_once_with(50, 50, 2.0)

    def test_deproject_returns_none_propagates(self):
        detector = self._detector_with_bbox()
        camera = MagicMock()
        camera.pixel_to_3d_with_depth.return_value = None
        estimator = Phase2TargetEstimator(
            camera=camera, detector=detector,
            extrinsic=CameraToPlateExtrinsic(),
        )
        assert estimator.estimate(
            np.zeros((100, 100, 3), dtype=np.uint8),
            self._depth_image(),
        ) is None


class FakeClock:
    """Monotonic-time substitute for deterministic tests."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


def _estimator_returning(values):
    """Mock estimator: estimate() returns each item in `values` per call.

    Items may be np.ndarray or None.
    """
    est = MagicMock()
    est.estimate.side_effect = values
    return est


def _camera_clock_pair(n_valid, clock, dt=0.033, post_none=1000):
    """Build a mock camera whose get_frames() advances `clock` by `dt`
    and yields `n_valid` valid (color, depth, frame) tuples then None tails.
    """
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    depth = np.full((10, 10), 3000, dtype=np.uint16)
    valid = (color, depth, MagicMock())
    invalid = (None, None, None)
    frames = iter([valid] * n_valid + [invalid] * post_none)
    camera = MagicMock()
    camera.get_frames = MagicMock(
        side_effect=lambda: (clock.tick(dt) or next(frames))
    )
    return camera


class TestRealPhase2TargetProvider:

    def test_all_frames_identical_median_equals_value(self):
        clock = FakeClock()
        camera = _camera_clock_pair(n_valid=30, clock=clock)
        target_xyz = np.array([0.1, 0.05, 2.0])
        estimator = _estimator_returning([target_xyz.copy() for _ in range(30)])

        provider = RealPhase2TargetProvider(
            camera=camera, estimator=estimator,
            measurement_duration_s=1.0, min_valid_frames=15,
            time_source=clock,
        )

        out = provider.get_phase2_target()
        assert out == pytest.approx((0.1, 0.05, 2.0))

    def test_jittered_samples_median_close_to_center(self):
        clock = FakeClock()
        camera = _camera_clock_pair(n_valid=30, clock=clock)
        center = np.array([0.0, 0.0, 3.0])
        rng = np.random.default_rng(seed=42)
        values = [center + rng.normal(0, 0.02, 3) for _ in range(30)]
        estimator = _estimator_returning(values)

        provider = RealPhase2TargetProvider(
            camera=camera, estimator=estimator,
            measurement_duration_s=1.0, min_valid_frames=15,
            time_source=clock,
        )

        out = np.array(provider.get_phase2_target())
        # 30 samples, σ=0.02 → expected error ~ 0.02 * 1.25 / √30 ≈ 4.5 mm.
        # Allow 2 cm margin.
        assert np.allclose(out, center, atol=0.02)

    def test_partial_dropouts_warning_but_progresses(self, caplog):
        import logging
        clock = FakeClock()
        camera = _camera_clock_pair(n_valid=30, clock=clock)
        target_xyz = np.array([0.0, 0.0, 3.0])
        # 15 valid + 15 None (estimator-side dropouts)
        values = [target_xyz.copy() for _ in range(15)] + [None] * 15
        estimator = _estimator_returning(values)

        provider = RealPhase2TargetProvider(
            camera=camera, estimator=estimator,
            measurement_duration_s=1.0,
            min_valid_frames=20,         # require more than we'll get
            time_source=clock,
        )

        with caplog.at_level(logging.WARNING):
            out = provider.get_phase2_target()
        assert out == pytest.approx((0.0, 0.0, 3.0))
        assert any("valid frames" in rec.message for rec in caplog.records)

    def test_zero_valid_raises(self):
        clock = FakeClock()
        camera = _camera_clock_pair(n_valid=30, clock=clock)
        estimator = _estimator_returning([None] * 30)

        provider = RealPhase2TargetProvider(
            camera=camera, estimator=estimator,
            measurement_duration_s=1.0, min_valid_frames=15,
            time_source=clock,
        )

        with pytest.raises(Phase2MeasurementError):
            provider.get_phase2_target()

    def test_skip_frames_when_color_none(self):
        """If camera.get_frames returns (None, None, None), provider skips
        without calling estimator."""
        clock = FakeClock()
        # Custom mix: 5 invalid frames at start, then 15 valid frames
        color = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.full((10, 10), 3000, dtype=np.uint16)
        _frames = iter([(None, None, None)] * 5
                       + [(color, depth, MagicMock())] * 15
                       + [(None, None, None)] * 1000)
        camera = MagicMock()
        camera.get_frames = MagicMock(
            side_effect=lambda: (clock.tick(0.033) or next(_frames))
        )

        target_xyz = np.array([0.0, 0.0, 3.0])
        estimator = _estimator_returning([target_xyz.copy() for _ in range(15)])

        provider = RealPhase2TargetProvider(
            camera=camera, estimator=estimator,
            measurement_duration_s=1.0, min_valid_frames=10,
            time_source=clock,
        )

        out = provider.get_phase2_target()
        assert out == pytest.approx((0.0, 0.0, 3.0))
        # estimator called only on valid frames (15), not the 5 None frames.
        assert estimator.estimate.call_count == 15

    def test_vertical_oscillation_median_converges_to_center(self):
        """Sin wave z-motion: median z ≈ center."""
        clock = FakeClock()
        N = 30
        camera = _camera_clock_pair(n_valid=N, clock=clock)
        # 1 cycle of z oscillation centered at 3.0, amp 0.1 m
        zs = 3.0 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, N))
        values = [np.array([0.0, 0.0, z]) for z in zs]
        estimator = _estimator_returning(values)

        provider = RealPhase2TargetProvider(
            camera=camera, estimator=estimator,
            measurement_duration_s=1.0, min_valid_frames=15,
            time_source=clock,
        )

        out = provider.get_phase2_target()
        # median of sin wave is exactly center
        assert out[2] == pytest.approx(3.0, abs=0.02)
