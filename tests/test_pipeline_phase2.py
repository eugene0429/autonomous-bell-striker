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
