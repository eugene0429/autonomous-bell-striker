"""Tests for TiltClient (sync) and TiltAsyncClient (fire-and-forget)."""

from __future__ import annotations

import pytest

from LevelingPlatform.tilt_motor import (
    TiltAsyncClient,
    TiltClient,
    TiltMotorConfig,
)


def _dry_cfg() -> TiltMotorConfig:
    return TiltMotorConfig(dry_run=True)


# ── sync TiltClient ──
def test_tilt_sync_clamps_to_step_range():
    cli = TiltClient(_dry_cfg())
    cli.tilt(9999)
    assert cli.sent_lines == ["TILT 2047"]

    cli2 = TiltClient(_dry_cfg())
    cli2.tilt(-9999)
    assert cli2.sent_lines == ["TILT -2047"]


def test_tilt_sync_rounds_to_int():
    cli = TiltClient(_dry_cfg())
    cli.tilt(100.7)
    assert cli.sent_lines == ["TILT 101"]


# ── async TiltAsyncClient ──
def test_tilt_async_uses_async_command():
    cli = TiltAsyncClient(_dry_cfg())
    cli.send(-800)
    assert cli.sent_lines == ["TILT_ASYNC -800"]


def test_tilt_async_clamps_to_step_range():
    cli = TiltAsyncClient(_dry_cfg())
    cli.send(9999)
    cli.send(-9999)
    assert cli.sent_lines == ["TILT_ASYNC 2047", "TILT_ASYNC -2047"]


def test_step_from_deg_roundtrip():
    cli = TiltAsyncClient(_dry_cfg())
    # 0° → 0 step, 90° → +1024 step at default 11.378 steps/deg
    assert cli.step_from_deg(0.0) == 0
    assert cli.step_from_deg(90.0) == 1024
    # round to nearest int
    assert cli.step_from_deg(45.0) == 512
