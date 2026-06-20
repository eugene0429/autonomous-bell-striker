"""Tests for DummyTargetProvider.get_visual_servo_detection."""

import math

import pytest

from perception.detection.dummy_detector import (
    DummyTargetConfig,
    DummyTargetProvider,
)


def test_target_directly_ahead_projects_to_center():
    cfg = DummyTargetConfig(
        phase1_target=(3.0, 0.0), bell_height_m=3.0, camera_height_m=0.30,
    )
    p = DummyTargetProvider(cfg)
    det = p.get_visual_servo_detection(
        robot_x=0.0, robot_y=0.0, robot_theta=0.0,
        tilt_deg=math.degrees(math.atan2(3.0 - 0.30, 3.0)),
    )
    assert det is not None
    cx = (det["bbox"][0] + det["bbox"][2]) / 2
    cy = (det["bbox"][1] + det["bbox"][3]) / 2
    assert abs(cx - cfg.img_w / 2) < 5
    assert abs(cy - cfg.img_h / 2) < 5


def test_target_right_of_robot_projects_right_of_center():
    cfg = DummyTargetConfig(
        phase1_target=(3.0, -1.0), bell_height_m=3.0, camera_height_m=0.30,
    )
    p = DummyTargetProvider(cfg)
    det = p.get_visual_servo_detection(0.0, 0.0, 0.0, tilt_deg=40.0)
    assert det is not None
    cx = (det["bbox"][0] + det["bbox"][2]) / 2
    assert cx > cfg.img_w / 2


def test_target_behind_robot_returns_none():
    cfg = DummyTargetConfig(phase1_target=(-3.0, 0.0))
    p = DummyTargetProvider(cfg)
    det = p.get_visual_servo_detection(0.0, 0.0, 0.0, tilt_deg=0.0)
    assert det is None


# ── 종 vertical oscillation (spec §9) ──
def test_bell_static_when_amp_zero():
    """Default amp=0 → bell offset stays 0 across many calls."""
    cfg = DummyTargetConfig(phase1_target=(3.0, 0.0), bell_height_amp_m=0.0)
    p = DummyTargetProvider(cfg)
    for _ in range(50):
        p.get_visual_servo_detection(0.0, 0.0, 0.0, tilt_deg=40.0)
    assert p._bell_offset_m == 0.0


def test_bell_oscillates_within_amplitude_bounds():
    """amp=0.5 → offset stays in [-0.25, +0.25] m for many cycles."""
    cfg = DummyTargetConfig(
        phase1_target=(3.0, 0.0),
        bell_height_amp_m=0.5,
        bell_endpoint_period_s=(0.5, 1.0),
        bell_dt_s=0.067,
    )
    p = DummyTargetProvider(cfg)
    seen_max, seen_min = 0.0, 0.0
    # 100 calls × 0.067 s ≈ 6.7 s → multiple traverses
    for _ in range(100):
        p.get_visual_servo_detection(0.0, 0.0, 0.0, tilt_deg=40.0)
        seen_max = max(seen_max, p._bell_offset_m)
        seen_min = min(seen_min, p._bell_offset_m)
    # Bounds — must respect ±amp/2
    assert seen_max <= 0.25 + 1e-9
    assert seen_min >= -0.25 - 1e-9
    # Must actually have moved through a meaningful fraction of the range
    assert seen_max > 0.15
    assert seen_min < -0.15


def test_bell_depth_changes_with_oscillation():
    """Oscillation should make depth_m vary between calls (no movement otherwise)."""
    cfg = DummyTargetConfig(
        phase1_target=(2.0, 0.0),
        bell_height_amp_m=0.5,
        bell_endpoint_period_s=(0.5, 0.5),  # deterministic timing
        bell_dt_s=0.067,
    )
    p = DummyTargetProvider(cfg)
    depths = []
    # Camera at (0, 0, 0.30), tilt up to roughly see bell at mean height
    tilt = math.degrees(math.atan2(3.0 - 0.30, 2.0))
    for _ in range(30):
        det = p.get_visual_servo_detection(0.0, 0.0, 0.0, tilt_deg=tilt)
        if det is not None:
            depths.append(det["depth_m"])
    assert len(depths) > 5
    assert (max(depths) - min(depths)) > 0.05  # bell vertical motion shows up
