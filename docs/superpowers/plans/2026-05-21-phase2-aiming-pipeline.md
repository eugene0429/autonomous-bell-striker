# Phase 2 Aiming Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `pipeline.py` 의 Phase 2 stub 을 실측정 파이프라인 (Hailo YOLO + RealSense depth → plate-frame median target → IK → fire) 으로 교체.

**Architecture:** 신규 `perception/detection/phase2_target.py` 에 좌표 변환 (`CameraToPlateExtrinsic`) + 단일 프레임 추정 (`Phase2TargetEstimator`) + 1초 측정창 집계 (`RealPhase2TargetProvider`) 3-단 구조. `pipeline.py` 는 `phase2_target_provider` 별도 인자로 dummy / real 분리, 나머지는 최소 수정.

**Tech Stack:** Python 3, numpy, pyrealsense2, pytest, RealSense D435i, Hailo HEF YOLO.

**Spec:** [docs/superpowers/specs/2026-05-21-phase2-aiming-pipeline-design.md](../specs/2026-05-21-phase2-aiming-pipeline-design.md)

---

## File Structure

| 파일 | 액션 | 책임 |
|---|---|---|
| `perception/detection/phase2_target.py` | NEW (~180 LOC) | `Phase2MeasurementError`, `CameraToPlateExtrinsic`, `Phase2TargetEstimator`, `RealPhase2TargetProvider` |
| `perception/detection/tests/test_phase2_target.py` | NEW (~280 LOC) | 위 4 클래스의 unit tests |
| `perception/common/realsense_wrapper.py` | MODIFY (+10 LOC) | `RealSenseCamera.pixel_to_3d_with_depth(x, y, depth_m)` helper |
| `pipeline.py` | MODIFY (~60 LOC delta) | `CapstonePipeline.__init__` 신규 인자, `phase2_aiming` 에러 처리 + settle vars, `RealRobot` camera/detector 보유, `build_pipeline` 와이어링, 신규 CLI args 4개 |
| `SW_ARCHITECTURE.md` | MODIFY | §5 Phase 2 실 구현 반영, §6 extrinsic 수치, §9 TODO 정리 |

### 의존성 그래프

```
Task 1 (Extrinsic) ──┐
Task 2 (pixel_to_3d_with_depth) ─┐
                                 ├─→ Task 3 (Estimator) ──┐
                                                          ├─→ Task 4 (Provider) ──┐
                                                                                  ├─→ Task 5 (Pipeline orchestrator) ──→ Task 6 (RealRobot + build_pipeline + CLI) ──→ Task 7 (docs + sim regression)
```

---

## Task 1: `CameraToPlateExtrinsic` (좌표 변환)

**Files:**
- Create: `perception/detection/phase2_target.py`
- Test: `perception/detection/tests/test_phase2_target.py`

- [ ] **Step 1: Create the test file with failing tests**

Create `perception/detection/tests/test_phase2_target.py`:

```python
"""Tests for perception.detection.phase2_target."""

from __future__ import annotations

import numpy as np
import pytest

from perception.detection.phase2_target import CameraToPlateExtrinsic


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest perception/detection/tests/test_phase2_target.py -v`
Expected: `ModuleNotFoundError: No module named 'perception.detection.phase2_target'`

- [ ] **Step 3: Create `phase2_target.py` with just the extrinsic class**

Create `perception/detection/phase2_target.py`:

```python
"""Phase 2 target estimator — bell 3D measurement in plate frame.

Camera→Plate coordinate transform + single-frame estimator + 1-second
multi-frame aggregation. Replaces `DummyTargetProvider.get_phase2_target()`
stub in real mode.

See: docs/superpowers/specs/2026-05-21-phase2-aiming-pipeline-design.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraToPlateExtrinsic:
    """Fixed extrinsic from camera optical frame → plate frame.

    Plate frame: +X forward, +Y left, +Z up. Origin = plate center.
    Camera frame (RealSense): +Z optical axis (out of lens),
                              +X image right, +Y image down.

    Default: natural mounting (camera roll 0°, tilted 90° about plate +Y).
    Lens at (0.20, 0, -0.10) m in plate frame. If post-calibration shows
    image-right ≠ plate -Y or image-down ≠ plate +X, flip the corresponding
    sign field.
    """

    t_x_m: float = 0.20
    t_z_m: float = -0.10
    image_right_sign: int = -1   # +1 if image-right → plate +Y; -1 default
    image_down_sign:  int = +1   # +1 if image-down → plate +X (default)

    def transform(self, p_cam: np.ndarray) -> np.ndarray:
        Xc, Yc, Zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        return np.array([
            self.image_down_sign  * Yc + self.t_x_m,
            self.image_right_sign * Xc,
            Zc + self.t_z_m,
        ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest perception/detection/tests/test_phase2_target.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add perception/detection/phase2_target.py perception/detection/tests/test_phase2_target.py
git commit -m "feat(phase2): CameraToPlateExtrinsic with sanity tests"
```

---

## Task 2: `RealSenseCamera.pixel_to_3d_with_depth` helper

**Files:**
- Modify: `perception/common/realsense_wrapper.py:259-276` (add new method below `pixel_to_3d`)

- [ ] **Step 1: Add the helper method**

In `perception/common/realsense_wrapper.py`, locate `pixel_to_3d` (around line 259) and add this method directly below it:

```python
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
```

Note: pyrealsense2's `rs2_deproject_pixel_to_point` is a C-extension call; we test this method indirectly through Phase 2 estimator tests (Task 3) which mock the camera. A direct unit test for this thin wrapper provides no value over the C-extension's own correctness.

- [ ] **Step 2: Verify the file still imports cleanly**

Run: `python -c "from perception.common.realsense_wrapper import RealSenseCamera; print('OK')"`
Expected: `OK` (must work on systems with pyrealsense2 installed).

If pyrealsense2 is not installed locally (e.g. on a dev Mac without RealSense), the import will already have been failing before this change — that's pre-existing, not introduced by this task.

- [ ] **Step 3: Commit**

```bash
git add perception/common/realsense_wrapper.py
git commit -m "feat(realsense): pixel_to_3d_with_depth helper for ROI-median deprojection"
```

---

## Task 3: `Phase2TargetEstimator` (single-frame)

**Files:**
- Modify: `perception/detection/phase2_target.py` (add class)
- Modify: `perception/detection/tests/test_phase2_target.py` (add tests)

- [ ] **Step 1: Add failing tests to the test file**

Append to `perception/detection/tests/test_phase2_target.py`:

```python
from unittest.mock import MagicMock

from perception.detection.phase2_target import Phase2TargetEstimator


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest perception/detection/tests/test_phase2_target.py::TestPhase2TargetEstimator -v`
Expected: `ImportError: cannot import name 'Phase2TargetEstimator'`

- [ ] **Step 3: Add `Phase2TargetEstimator` to `phase2_target.py`**

Add to `perception/detection/phase2_target.py` (after `CameraToPlateExtrinsic`):

```python
from typing import Optional

from .visual_servo_target import compute_target_depth


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest perception/detection/tests/test_phase2_target.py -v`
Expected: 14 passed (8 from Task 1 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add perception/detection/phase2_target.py perception/detection/tests/test_phase2_target.py
git commit -m "feat(phase2): Phase2TargetEstimator single-frame measurement"
```

---

## Task 4: `RealPhase2TargetProvider` + `Phase2MeasurementError`

**Files:**
- Modify: `perception/detection/phase2_target.py` (add error + provider class)
- Modify: `perception/detection/tests/test_phase2_target.py` (add tests)

- [ ] **Step 1: Add failing tests to the test file**

Append to `perception/detection/tests/test_phase2_target.py`:

```python
from perception.detection.phase2_target import (
    Phase2MeasurementError,
    RealPhase2TargetProvider,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest perception/detection/tests/test_phase2_target.py::TestRealPhase2TargetProvider -v`
Expected: `ImportError: cannot import name 'RealPhase2TargetProvider'`

- [ ] **Step 3: Add `Phase2MeasurementError` + `RealPhase2TargetProvider`**

Add to top of `perception/detection/phase2_target.py` (after the module docstring imports):

```python
import logging
import time
from typing import Callable, Tuple

log = logging.getLogger(__name__)


class Phase2MeasurementError(RuntimeError):
    """Raised when a measurement window yields zero valid detections."""
```

Append at the bottom of `perception/detection/phase2_target.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest perception/detection/tests/test_phase2_target.py -v`
Expected: 20 passed (14 from prior tasks + 6 new).

- [ ] **Step 5: Commit**

```bash
git add perception/detection/phase2_target.py perception/detection/tests/test_phase2_target.py
git commit -m "feat(phase2): RealPhase2TargetProvider 1s median measurement"
```

---

## Task 5: `pipeline.py` — `CapstonePipeline.__init__` 신규 인자 + `phase2_aiming` 변경

**Files:**
- Modify: `pipeline.py:231-343`

- [ ] **Step 1: Add new params to `CapstonePipeline.__init__`**

In `pipeline.py`, locate `CapstonePipeline.__init__` (around line 234). Modify the signature and body:

```python
class CapstonePipeline:
    """Phase 1 (Driving) → Phase 2 (Aiming & Strike ×N) 통합 실행기."""

    def __init__(
        self,
        robot,
        target_provider: DummyTargetProvider,
        ctrl: DrivingController,
        ik: LevelingIK,
        dt: float = 0.067,
        phase1_timeout_sec: float = 60.0,
        num_strikes: int = 2,
        strike_interval_sec: float = 1.0,
        drive_mode: str = "slam",
        # ── Phase 2 신규 ──
        phase2_target_provider=None,
        tilt_settle_sec: float = 0.5,
        plate_settle_sec: float = 0.3,
    ):
        self.robot = robot
        self.target_provider = target_provider
        self.ctrl = ctrl
        self.ik = ik
        self.dt = dt
        self.phase1_timeout_sec = phase1_timeout_sec
        self.num_strikes = num_strikes
        self.strike_interval_sec = strike_interval_sec
        self.drive_mode = drive_mode
        # ── Phase 2 신규 ──
        self.phase2_target_provider = phase2_target_provider or target_provider
        self.tilt_settle_sec = tilt_settle_sec
        self.plate_settle_sec = plate_settle_sec
```

Backward compat: existing code that calls `CapstonePipeline(robot, target_provider, ctrl, ik, ...)` still works — new args have defaults.

- [ ] **Step 2: Update `phase2_aiming` to use new attributes + catch `Phase2MeasurementError`**

In `pipeline.py`, locate `phase2_aiming` (around line 302). Replace its body with:

```python
    # ── Phase 2 ──
    def phase2_aiming(self) -> bool:
        from detection.phase2_target import Phase2MeasurementError   # lazy

        print(f"── PHASE 2: AIMING & STRIKE x{self.num_strikes} ──")

        # 카메라 90° 틸트 (위로)
        self.robot.tilt_camera(90.0)
        time.sleep(self.tilt_settle_sec)

        successful = 0
        for shot in range(1, self.num_strikes + 1):
            print(f"\n  ── shot {shot}/{self.num_strikes} ──")

            # 매 타격 직전 종 위치 재추정 (1초 측정창 + per-axis median)
            try:
                target_xyz = self.phase2_target_provider.get_phase2_target()
            except Phase2MeasurementError as e:
                print(f"  ✗ measurement failed: {e} — skip shot")
                continue

            print(f"  target (plate frame): ({target_xyz[0]:+.3f}, "
                  f"{target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}) m")

            out = self.ik.aim_at(target_xyz)

            if out["angles_deg"] is None:
                print("  ✗ leg length infeasible — skip")
                continue

            ball = ", ".join(f"{b:.2f}" for b in out["ball_deg"])
            print(f"  motor angles : {[f'{a:+.3f}' for a in out['angles_deg']]} deg")
            print(f"  encoder steps: {out['angles_steps']}")
            print(f"  ball P deg   : [{ball}] (lim={self.ik.cfg.ball_max_deg})")
            print(f"  feasible     : {out['ok']}")

            if not out["ok"]:
                print("  ⚠ ball joint limit exceeded — proceeding anyway "
                      "(assumption violated; runtime trigger for follow-up)")

            self.robot.send_leveling_angles(out["angles_deg"], out["angles_steps"])
            time.sleep(self.plate_settle_sec)
            self.robot.fire()
            successful += 1

            if shot < self.num_strikes:
                time.sleep(self.strike_interval_sec)

        print(f"\n  → {successful}/{self.num_strikes} strikes executed")
        return successful == self.num_strikes
```

Three changes:
1. `time.sleep(0.3)` after tilt → `time.sleep(self.tilt_settle_sec)`
2. `target_provider.get_phase2_target()` → `phase2_target_provider.get_phase2_target()` wrapped in try/except
3. `time.sleep(0.3)` before fire → `time.sleep(self.plate_settle_sec)`

- [ ] **Step 3: Run existing sim regression**

Run: `python3 pipeline.py --mode sim --phase2-jitter 0.05 --num-strikes 2 --phase1-x 1 --phase1-y 0`
Expected:
- Phase 1 completes (driving sim)
- Phase 2 prints `target (plate frame): (+0.10, +0.00, +X.XX) m` for each of 2 shots (X varies with jitter)
- Final line: `2/2 strikes executed`

If output shows the original line `--phase2-jitter` is still honored — backward compat preserved.

- [ ] **Step 4: Add pipeline unit test for provider routing**

Create or extend a test file (suggest `tests/test_pipeline_phase2.py` at repo root, or `Driving/tests/test_pipeline_phase2.py`. Given existing test layout, create at repo root for orchestrator-level tests):

Create `tests/test_pipeline_phase2.py`:

```python
"""Test that CapstonePipeline routes Phase 2 measurement through the
correct provider when phase2_target_provider is set."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for sub in ("Driving", "LevelingPlatform", "perception"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import CapstonePipeline, SimulatedRobot
from controller import ControllerConfig, DrivingController
from leveling_ik import LevelingConfig, LevelingIK
from detection.dummy_detector import DummyTargetConfig, DummyTargetProvider


def _build_pipeline(phase2_target_provider=None):
    robot = SimulatedRobot()
    target_provider = DummyTargetProvider(DummyTargetConfig())
    ctrl = DrivingController(ControllerConfig())
    ik = LevelingIK(LevelingConfig())
    return CapstonePipeline(
        robot, target_provider, ctrl, ik,
        num_strikes=1,
        phase2_target_provider=phase2_target_provider,
    )


def test_phase2_uses_phase2_target_provider_when_set():
    """phase2_aiming routes through phase2_target_provider, not target_provider."""
    phase2_provider = MagicMock()
    phase2_provider.get_phase2_target.return_value = (0.05, 0.0, 3.0)

    pipeline = _build_pipeline(phase2_target_provider=phase2_provider)
    pipeline.phase2_aiming()

    phase2_provider.get_phase2_target.assert_called_once()


def test_phase2_falls_back_to_target_provider_when_unset():
    """phase2_target_provider=None → uses target_provider (backward compat)."""
    pipeline = _build_pipeline(phase2_target_provider=None)
    # Direct identity check
    assert pipeline.phase2_target_provider is pipeline.target_provider


def test_phase2_skips_shot_on_measurement_error():
    """Phase2MeasurementError → skip the shot, return False (0/1 success)."""
    from detection.phase2_target import Phase2MeasurementError
    phase2_provider = MagicMock()
    phase2_provider.get_phase2_target.side_effect = Phase2MeasurementError("test")

    pipeline = _build_pipeline(phase2_target_provider=phase2_provider)
    ok = pipeline.phase2_aiming()
    assert ok is False
```

- [ ] **Step 5: Run pipeline tests**

Run: `pytest tests/test_pipeline_phase2.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add pipeline.py tests/test_pipeline_phase2.py
git commit -m "feat(pipeline): Phase 2 provider routing + measurement error handling"
```

---

## Task 6: `pipeline.py` — `RealRobot` 카메라/detector + `build_pipeline` 와이어링 + CLI args

**Files:**
- Modify: `pipeline.py:163-225` (RealRobot)
- Modify: `pipeline.py:349-457` (build_pipeline + main)

- [ ] **Step 1: Wire camera + detector into `RealRobot`**

In `pipeline.py`, locate `RealRobot.__init__` (around line 166). Modify to lazy-import + instantiate camera and detector:

```python
class RealRobot:
    """실제 RealSense + ORB-SLAM3 측위. 모터는 stub (TODO: 시리얼 드라이버 연결)."""

    def __init__(self, wheel_diameter: float = 0.10, wheel_base: float = 0.30):
        from vio.orbslam_localizer import (  # lazy import
            LocalizerConfig, OrbSlamLocalizer,
        )
        from common.realsense_wrapper import RealSenseCamera   # lazy import
        from detection.detector import TargetDetector          # lazy import
        from config import CAMERA, DETECTION                   # lazy import

        self.wheel_diameter = wheel_diameter
        self.wheel_base = wheel_base
        self.localizer = OrbSlamLocalizer(LocalizerConfig())
        self.camera = RealSenseCamera(CAMERA)                  # NEW
        self.detector = TargetDetector(DETECTION)              # NEW
        self._fired = 0
```

Also update `start()` and `stop()`:

```python
    def start(self) -> None:
        self.localizer.start()
        print("[REAL] waiting for SLAM tracking OK ...")
        ok = self.localizer.wait_for_tracking(timeout=30.0)
        print(f"[REAL] tracking_ok = {ok}")
        if not ok:
            raise RuntimeError("ORB-SLAM3 did not reach tracking OK within 30s")
        # Phase 2 카메라 stream 은 tilt_camera() 진입 시 lazily start
        # (Phase 1 SLAM 종료 후) 해서 concurrent stream 충돌 방지.
        # See spec §5 'Phase 1 ↔ Phase 2 직렬 카메라 사용'.
        self._phase2_camera_started = False

    def stop(self) -> None:
        self.localizer.stop()
        if self._phase2_camera_started:
            try:
                self.camera.stop()
            except Exception as e:
                print(f"[REAL] camera.stop() ignored: {e}")
        print(f"[REAL] shutdown (fired {self._fired} times)")
```

Note: per spec §5 "직렬 카메라 사용" — Phase 1 (ORB-SLAM3) holds the RealSense pipeline; Phase 2 measurement reopens it. To support this cleanly without rewriting the camera lifecycle now, we add a `tilt_camera()` hook that also (re)starts the camera if needed:

```python
    def tilt_camera(self, deg: float) -> None:
        # First Phase-2 entry: stop SLAM and start the Phase-2 camera pipeline.
        if not self._phase2_camera_started:
            if self.localizer.is_alive():
                self.localizer.stop()
            self.camera.start()
            self._phase2_camera_started = True
        # 틸트 서보 명령 (별도 PR / TILT_ASYNC v1.1 활용은 follow-up)
        print(f"\n[REAL TODO] camera tilt → {deg:+.1f}°")
```

- [ ] **Step 2: Wire `build_pipeline` real mode + add CLI args**

In `pipeline.py`, modify `build_pipeline` (around line 349):

```python
def build_pipeline(args) -> CapstonePipeline:
    target_cfg = DummyTargetConfig(
        phase1_target=(args.phase1_x, args.phase1_y),
        phase2_target=(args.phase2_x, args.phase2_y, args.phase2_z),
        phase2_jitter=args.phase2_jitter,
        vs_bbox_noise_px=args.vs_bbox_noise,
        vs_depth_noise_m=args.vs_depth_noise,
        vs_dropout_prob=args.vs_dropout,
    )
    target_provider = DummyTargetProvider(target_cfg)

    ctrl = DrivingController(ControllerConfig(
        wheel_diameter=args.wheel_diameter,
        wheel_base=args.wheel_base,
    ))
    ik = LevelingIK(LevelingConfig())

    if args.mode == "sim":
        robot = SimulatedRobot(
            start_xy=(args.start_x, args.start_y),
            start_theta=np.deg2rad(args.start_theta_deg),
            wheel_diameter=args.wheel_diameter,
            wheel_base=args.wheel_base,
        )
        phase2_target_provider = target_provider           # dummy 재사용
    elif args.mode == "real":
        robot = RealRobot(
            wheel_diameter=args.wheel_diameter,
            wheel_base=args.wheel_base,
        )
        from detection.phase2_target import (              # lazy
            CameraToPlateExtrinsic,
            Phase2TargetEstimator,
            RealPhase2TargetProvider,
        )
        phase2_target_provider = RealPhase2TargetProvider(
            camera=robot.camera,
            estimator=Phase2TargetEstimator(
                camera=robot.camera,
                detector=robot.detector,
                extrinsic=CameraToPlateExtrinsic(),
            ),
            measurement_duration_s=args.phase2_meas_sec,
            min_valid_frames=args.phase2_min_frames,
        )
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    return CapstonePipeline(
        robot, target_provider, ctrl, ik,
        dt=args.dt,
        phase1_timeout_sec=args.phase1_timeout,
        num_strikes=args.num_strikes,
        strike_interval_sec=args.strike_interval,
        drive_mode=args.drive_mode,
        phase2_target_provider=phase2_target_provider,
        tilt_settle_sec=args.tilt_settle_sec,
        plate_settle_sec=args.plate_settle_sec,
    )
```

- [ ] **Step 3: Add new CLI args in `main()`**

In `pipeline.py`, locate `main()` (around line 391). After the existing `--strike-interval` arg, add:

```python
    # ── Phase 2 measurement (real mode) ──
    ap.add_argument("--phase2-meas-sec", type=float, default=1.0,
                    help="Phase 2 측정창 길이 [s] (real mode only)")
    ap.add_argument("--phase2-min-frames", type=int, default=15,
                    help="Phase 2 1초 측정창 최소 valid frame 수 (real mode only). "
                         "미달 시 경고만 출력 후 진행")
    ap.add_argument("--tilt-settle-sec", type=float, default=0.5,
                    help="90° 틸트 후 대기 [s]")
    ap.add_argument("--plate-settle-sec", type=float, default=0.3,
                    help="레벨링 모터 명령 후 발사 전 대기 [s]")
```

- [ ] **Step 4: Verify CLI args appear in --help**

Run: `python3 pipeline.py --help 2>&1 | grep -E "phase2-meas|phase2-min|tilt-settle|plate-settle"`
Expected: 4 lines of output, one per new flag.

- [ ] **Step 5: Verify sim mode still works end-to-end**

Run: `python3 pipeline.py --mode sim --phase2-jitter 0.05 --num-strikes 2 --phase1-x 1 --phase1-y 0 --tilt-settle-sec 0.1 --plate-settle-sec 0.1`
Expected:
- Phase 1 completes
- Phase 2 prints 2 shots, each with `target (plate frame): ...`
- Final: `2/2 strikes executed`
- Total wall time noticeably less than default (settle reduced from 0.3 to 0.1)

- [ ] **Step 6: Commit**

```bash
git add pipeline.py
git commit -m "feat(pipeline): wire RealRobot camera/detector + Phase 2 real-mode provider"
```

---

## Task 7: `SW_ARCHITECTURE.md` 업데이트 + sim regression smoke

**Files:**
- Modify: `SW_ARCHITECTURE.md`

- [ ] **Step 1: Update §5 (Phase 2) to reflect real implementation**

In `SW_ARCHITECTURE.md`, locate §5.2 "파이프라인" (around line 156). Add a new sub-bullet under step 2 ("Bell 3D vector estimation") referencing the new modules:

```markdown
2. **Bell 3D vector estimation**
   - YOLO 검출 → bbox → depth 디프로젝션으로 카메라 좌표계의 (X, Y, Z)
   - 카메라 ↔ 레벨링 플랫폼 중심 사이의 알려진 외부 변환을 적용해 **플랫폼 중심 → 종까지의 3D 벡터**를 얻음
   - **실제 구현**: [perception/detection/phase2_target.py](perception/detection/phase2_target.py) `Phase2TargetEstimator` (single frame) + `RealPhase2TargetProvider` (1초 측정창, per-axis median). 매 shot 직전 호출.
   - **Camera→Plate extrinsic**: lens 가 plate center 기준 `(+0.20, 0, -0.10) m` 에 위치, 90° pitch-up. 회전행렬 + 부호 옵션은 `CameraToPlateExtrinsic` 의 dataclass 필드로 노출.
```

- [ ] **Step 2: Update §6 (좌표계) with extrinsic numbers**

In §6 (`주요 좌표계와 정합`, around line 201), append after the existing paragraph:

```markdown
Camera→Plate 외부 변환은 [perception/detection/phase2_target.py](perception/detection/phase2_target.py) `CameraToPlateExtrinsic` 에 다음 기본값으로 캡슐화되어 있다:

- `t_x_m = +0.20`, `t_z_m = -0.10` (lens가 plate center 기준 (+0.20, 0, -0.10) m)
- `image_right_sign = -1`, `image_down_sign = +1` (자연 마운트, camera roll 0°)

캘리브레이션 절차는 [Phase 2 design spec](docs/superpowers/specs/2026-05-21-phase2-aiming-pipeline-design.md) §3 참조.
```

- [ ] **Step 3: Update §9 TODO list**

In §9 (`미구현 / TODO`, around line 311), mark these as completed and add Phase 2 follow-ups:

Replace:
```markdown
- [ ] Phase 1 다중 프레임 평균 + Phase 2 카메라→플레이트 변환 (현재는 [dummy_detector.py](perception/detection/dummy_detector.py) 가 stub)
- [ ] Camera ↔ Plate 외부 변환 캘리브레이션 절차 문서화
```

With:
```markdown
- [ ] Phase 1 다중 프레임 평균 + Phase 2 카메라→플레이트 변환 (Phase 2 부분은 완료: [phase2_target.py](perception/detection/phase2_target.py))
- [x] Camera ↔ Plate 외부 변환 캘리브레이션 절차 문서화 ([2026-05-21 design spec](docs/superpowers/specs/2026-05-21-phase2-aiming-pipeline-design.md) §3)
- [ ] Phase 2 bench test: extrinsic 캘리브레이션 부호 확정 + 정적 종 1초 측정 std_z < 5 cm 검증
- [ ] Phase 2 ↔ Phase 1 카메라 stream 동시 사용 최적화 (현재는 직렬: SLAM stop → camera reopen)
```

- [ ] **Step 4: Run full test suite for regression check**

Run: `pytest perception/detection/tests/ tests/ -v`
Expected: 23+ passed (20 from Tasks 1, 3, 4 + 3 from Task 5).

Run: `python3 pipeline.py --mode sim --phase2-jitter 0.05 --num-strikes 2 --phase1-x 1 --phase1-y 0`
Expected: `2/2 strikes executed`.

- [ ] **Step 5: Commit**

```bash
git add SW_ARCHITECTURE.md
git commit -m "docs(arch): reflect Phase 2 real implementation + extrinsic spec link"
```

---

## Final Verification

After all 7 tasks:

- [ ] `pytest perception/detection/tests/ tests/ -v` → all green
- [ ] `python3 pipeline.py --help` → shows 4 new args
- [ ] `python3 pipeline.py --mode sim --num-strikes 2 --phase1-x 1 --phase1-y 0` → 2/2 strikes
- [ ] Git log shows 7 commits, one per task
- [ ] `phase2_target.py` ≤ ~200 LOC (focus discipline check)
- [ ] No `# TODO`, `# FIXME`, or `NotImplementedError` introduced in this PR (other than what existed before)

Bench test (manual, on Pi5 + RealSense + Hailo + 종 mockup) is **not** part of this plan — it's a follow-up to be executed once hardware is available. Procedure documented in design spec §6.
