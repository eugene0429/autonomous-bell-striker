"""Tests for visual_servo_target.compute_target_depth."""

from __future__ import annotations

import numpy as np
import pytest

from perception.detection.visual_servo_target import compute_target_depth


def test_uniform_depth_returns_median():
    # 100x100 depth, all 2000mm. bbox at center 40x40.
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    bbox = (30, 30, 70, 70)
    out = compute_target_depth(depth, bbox, roi_frac=0.4, min_valid_pixels=10,
                               depth_scale_m=0.001)
    assert out == pytest.approx(2.0, abs=1e-6)


def test_zero_holes_excluded():
    # half holes, half valid 2500mm → median = 2.5 m
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[40:60, 40:60] = 2500   # 20x20 valid block at center
    bbox = (30, 30, 70, 70)
    out = compute_target_depth(depth, bbox, roi_frac=0.4, min_valid_pixels=10,
                               depth_scale_m=0.001)
    assert out == pytest.approx(2.5, abs=1e-6)


def test_too_few_valid_returns_none():
    depth = np.zeros((100, 100), dtype=np.uint16)
    depth[50, 50] = 1500          # only 1 valid pixel
    bbox = (30, 30, 70, 70)
    out = compute_target_depth(depth, bbox, roi_frac=0.4, min_valid_pixels=10,
                               depth_scale_m=0.001)
    assert out is None


def test_bbox_clipped_to_image():
    # bbox extends past image bounds → clip to image
    depth = np.full((50, 50), 1000, dtype=np.uint16)
    bbox = (40, 40, 80, 80)       # right/bottom past image
    out = compute_target_depth(depth, bbox, roi_frac=0.4, min_valid_pixels=1,
                               depth_scale_m=0.001)
    assert out == pytest.approx(1.0, abs=1e-6)


def test_degenerate_bbox_returns_none():
    depth = np.full((50, 50), 1000, dtype=np.uint16)
    bbox = (20, 20, 20, 20)       # zero-area
    out = compute_target_depth(depth, bbox, roi_frac=0.4, min_valid_pixels=1,
                               depth_scale_m=0.001)
    assert out is None
