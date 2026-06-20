"""Tests for VisualServoController state machine + control output.

Coordinate convention (matches spec §4.2):
  - image: x right +, y down +
  - ω > 0 = CCW (turn left)
  - err_x_px > 0 (target on right) → ω < 0 (turn right)
"""

from __future__ import annotations

import math

import pytest

from Driving.visual_servo_controller import (
    VisualServoConfig,
    VisualServoController,
)


def _det(cx_px, cy_px, depth_m, w=640, h=480, bw=80, bh=80, conf=0.9):
    """Synthesize a detection dict centered at (cx_px, cy_px)."""
    x1 = cx_px - bw // 2
    y1 = cy_px - bh // 2
    return {
        "bbox": (x1, y1, x1 + bw, y1 + bh),
        "conf": conf,
        "depth_m": depth_m,
    }


def _ctrl():
    return VisualServoController(VisualServoConfig())


# ── TRACK: bbox centered, mid-range depth → forward motion ──
def test_track_centered_target_drives_forward():
    c = _ctrl()
    out = c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["state"] == "TRACK"
    assert out["reached"] is False
    assert out["failed"] is False
    assert out["v"] > 0
    assert abs(out["omega"]) < 0.05   # roughly straight


# ── horiz_dist = depth · cos(tilt) ──
def test_horiz_dist_computed_from_depth_and_tilt():
    c = _ctrl()
    out = c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=60.0)
    expected = 2.0 * math.cos(math.radians(60.0))
    assert out["horiz_dist"] == pytest.approx(expected, abs=1e-6)


# ── sign convention ──
def test_target_on_right_turns_right():
    """err_x_px > 0 (target right) → ω < 0 (turn right, CW)."""
    c = _ctrl()
    out = c.step(_det(420, 240, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["omega"] < 0
    # wheel ω: right wheel slower than left for CW turn
    assert out["wheel_omega_right"] < out["wheel_omega_left"]


def test_target_on_left_turns_left():
    c = _ctrl()
    out = c.step(_det(220, 240, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["omega"] > 0
    assert out["wheel_omega_left"] < out["wheel_omega_right"]


def test_target_above_lifts_tilt():
    """err_y_px < 0 (target above center) → tilt_cmd > tilt_cur."""
    c = _ctrl()
    out = c.step(_det(320, 100, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["tilt_cmd_deg"] > 30.0


def test_target_below_lowers_tilt():
    c = _ctrl()
    out = c.step(_det(320, 380, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["tilt_cmd_deg"] < 30.0


# ── DONE / stop ──
def test_stop_when_tilt_in_range_and_close():
    """tilt 88° + horiz_dist 0.05m → DONE after stop_debounce_frames consecutive frames."""
    c = _ctrl()
    # depth=1.5m, tilt=88° → horiz = 1.5·cos(88°) ≈ 0.052m
    # First (debounce - 1) frames satisfy condition but state stays TRACK.
    for i in range(c.cfg.stop_debounce_frames - 1):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
        assert out["state"] == "TRACK", f"premature DONE at frame {i}"
    # Final frame trips the debounce → DONE.
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["state"] == "DONE"
    assert out["reached"] is True
    assert out["v"] == 0.0
    assert out["omega"] == 0.0


def test_no_stop_when_tilt_below_range():
    """tilt 80° even with tiny horiz_dist → not DONE."""
    c = _ctrl()
    # depth=0.5m, tilt=80° → horiz ≈ 0.087m < d_stop, but tilt < 85°
    out = c.step(_det(320, 240, depth_m=0.5), tilt_deg_cur=80.0)
    assert out["state"] != "DONE"


def test_no_stop_when_horiz_dist_too_large():
    c = _ctrl()
    # depth=2m, tilt=88° → horiz ≈ 0.07m → would stop
    # but at depth=3m, tilt=88° → horiz ≈ 0.105m → no stop
    out = c.step(_det(320, 240, depth_m=3.0), tilt_deg_cur=88.0)
    assert out["state"] != "DONE"


# ── robustness vs. 종 vertical oscillation (spec §4.1 step 2.5 / 6) ──
def test_horiz_dist_lpf_smooths_step_input():
    """First frame: filt == raw. Step jump on next: filt lags raw."""
    c = _ctrl()
    # Seed with depth=2.0, tilt=60° → horiz_raw = 1.0
    out0 = c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=60.0)
    assert out0["horiz_dist"] == pytest.approx(1.0, abs=1e-6)
    assert out0["horiz_dist_raw"] == pytest.approx(1.0, abs=1e-6)
    # Step jump: depth=3.0 → horiz_raw = 1.5
    out1 = c.step(_det(320, 240, depth_m=3.0), tilt_deg_cur=60.0)
    assert out1["horiz_dist_raw"] == pytest.approx(1.5, abs=1e-6)
    # filt must not jump to 1.5 in one frame (α=0.2 → filt = 0.2*1.5 + 0.8*1.0 = 1.1)
    assert 1.0 < out1["horiz_dist"] < 1.5
    expected = c.cfg.horiz_dist_lp_alpha * 1.5 + (1 - c.cfg.horiz_dist_lp_alpha) * 1.0
    assert out1["horiz_dist"] == pytest.approx(expected, abs=1e-6)


def test_tilt_deadband_freezes_command_for_small_err():
    """|err_y_px| < tilt_err_deadband_px → tilt_cmd_deg == tilt_deg_cur."""
    c = _ctrl()
    deadband = c.cfg.tilt_err_deadband_px
    # cy at center + (deadband - 1) → err_y = deadband - 1 < deadband → frozen
    small_off = deadband - 1
    out = c.step(_det(320, 240 + small_off, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["tilt_cmd_deg"] == pytest.approx(30.0, abs=1e-9)


def test_stop_requires_consecutive_frames():
    """Streak counter requires consecutive satisfying frames; single miss resets it."""
    # Disable LPF for a clean streak test (filt == raw every frame).
    from Driving.visual_servo_controller import VisualServoConfig, VisualServoController
    cfg = VisualServoConfig(horiz_dist_lp_alpha=1.0)
    c = VisualServoController(cfg)
    # (debounce - 1) satisfying frames — still TRACK
    for _ in range(cfg.stop_debounce_frames - 1):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
        assert out["state"] == "TRACK"
    # Mid-streak miss (tilt out of stop range)
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=50.0)
    assert out["state"] == "TRACK"
    assert c._stop_streak == 0
    # Resume satisfying — needs full debounce again
    for _ in range(cfg.stop_debounce_frames - 1):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
        assert out["state"] == "TRACK"
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["state"] == "DONE"


def _seed_track(c, frames=1):
    """Drive controller through `frames` TRACK frames so it has a last_wheel."""
    for _ in range(frames):
        c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=30.0)


def test_single_lost_frame_enters_coast_with_scaled_wheels():
    c = _ctrl()
    _seed_track(c, frames=2)
    last_wL_before = c._last_wheel[0]
    last_wR_before = c._last_wheel[1]
    out = c.step(None, tilt_deg_cur=30.0)
    assert out["state"] == "COAST"
    assert out["wheel_omega_left"] == pytest.approx(last_wL_before * 0.7, rel=1e-6)
    assert out["wheel_omega_right"] == pytest.approx(last_wR_before * 0.7, rel=1e-6)


def test_three_lost_frames_enters_hold_with_zero_wheels():
    c = _ctrl()
    _seed_track(c)
    for _ in range(3):
        out = c.step(None, tilt_deg_cur=30.0)
    assert out["state"] == "HOLD"
    assert out["wheel_omega_left"] == 0.0
    assert out["wheel_omega_right"] == 0.0


def test_fifteen_lost_frames_enters_search_with_forward_creep():
    c = _ctrl()
    c.step(_det(420, 240, depth_m=2.0), tilt_deg_cur=30.0)
    for _ in range(15):
        out = c.step(None, tilt_deg_cur=30.0)
    assert out["state"] == "SEARCH"
    # Creep forward, no rotation — heading lock preserved.
    assert out["v"] == pytest.approx(c.cfg.search_creep_v, abs=1e-9)
    assert out["omega"] == pytest.approx(0.0, abs=1e-9)
    assert out["wheel_omega_left"] == pytest.approx(out["wheel_omega_right"], abs=1e-9)
    assert out["wheel_omega_left"] > 0.0


def test_search_timeout_triggers_fail():
    c = _ctrl()
    c.step(_det(420, 240, depth_m=2.0), tilt_deg_cur=30.0)
    # Drive through search; cfg.dt=0.067, search_timeout_s=15.0 → ~225 frames
    frames_to_fail = int(c.cfg.search_timeout_s / c.cfg.dt) + 20
    for _ in range(frames_to_fail):
        out = c.step(None, tilt_deg_cur=30.0)
    assert out["state"] == "FAIL"
    assert out["failed"] is True
    assert out["wheel_omega_left"] == 0.0
    assert out["wheel_omega_right"] == 0.0


def test_found_resets_to_track_from_search():
    c = _ctrl()
    c.step(_det(420, 240, depth_m=2.0), tilt_deg_cur=30.0)
    for _ in range(20):
        c.step(None, tilt_deg_cur=30.0)
    assert c._state == "SEARCH"
    out = c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["state"] == "TRACK"


def test_found_resets_lost_counter_from_coast():
    c = _ctrl()
    _seed_track(c)
    c.step(None, tilt_deg_cur=30.0)  # 1 lost
    c.step(None, tilt_deg_cur=30.0)  # 2 lost
    out = c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=30.0)
    assert out["state"] == "TRACK"
    assert c._lost_frames == 0


# ── ALIGN (post-stop fine alignment) ──
def _align_cfg(**overrides):
    """Config with ALIGN enabled and the LPF disabled (filt == raw).

    tilt_brake_start_deg is set so the stop trigger fires on tilt band alone,
    independent of whatever d_stop_m the per-test override uses. That way
    _enter_align() reliably trips the debounce in two-line tests.
    """
    kwargs = dict(
        horiz_dist_lp_alpha=1.0,
        tilt_brake_start_deg=50.0,
        align_enabled=True,
        align_v=0.05,
        align_tol_m=0.02,
        align_debounce_frames=3,
        align_timeout_s=10.0,
    )
    kwargs.update(overrides)
    return VisualServoConfig(**kwargs)


def _enter_align(c):
    """Trip the stop-debounce so the controller enters ALIGN state."""
    for _ in range(c.cfg.stop_debounce_frames):
        c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert c._state == "ALIGN"


def test_stop_trigger_enters_align_when_enabled():
    """align_enabled=True → stop-debounce transitions to ALIGN, not DONE."""
    c = VisualServoController(_align_cfg())
    # First (debounce - 1) frames satisfy condition but state stays TRACK.
    for _ in range(c.cfg.stop_debounce_frames - 1):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
        assert out["state"] == "TRACK"
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["state"] == "ALIGN"
    assert out["reached"] is False     # not done yet — fine alignment runs
    assert out["v"] == 0.0             # stop trigger frame holds
    assert out["omega"] == 0.0


def test_align_too_far_drives_forward():
    """horiz_dist > d_stop_m + tol → v = +align_v (forward, omega=0)."""
    cfg = _align_cfg(d_stop_m=0.20)
    c = VisualServoController(cfg)
    _enter_align(c)
    # depth=1.0, tilt=88° → horiz ≈ 0.0349 — too close (will reverse). Force "too
    # far" by feeding a horiz well above d_stop_m: depth=2.0, tilt=80° → horiz≈0.347
    out = c.step(_det(320, 240, depth_m=2.0), tilt_deg_cur=80.0)
    assert out["state"] == "ALIGN"
    assert out["v"] == pytest.approx(cfg.align_v, abs=1e-9)
    assert out["omega"] == 0.0


def test_align_too_close_drives_backward():
    """horiz_dist < d_stop_m - tol → v = -align_v (reverse)."""
    cfg = _align_cfg(d_stop_m=0.40)
    c = VisualServoController(cfg)
    _enter_align(c)
    # depth=1.5, tilt=88° → horiz ≈ 0.0523 → way below 0.40 → reverse
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["state"] == "ALIGN"
    assert out["v"] == pytest.approx(-cfg.align_v, abs=1e-9)
    assert out["omega"] == 0.0


def test_align_within_tolerance_holds_and_streaks_to_done():
    """|err| < tol for align_debounce_frames consecutive frames → DONE."""
    # d_stop=0.05 so depth=1.5, tilt=88° gives horiz ≈ 0.052, within tol=0.02
    cfg = _align_cfg(d_stop_m=0.05, align_debounce_frames=3)
    c = VisualServoController(cfg)
    _enter_align(c)
    # First (debounce - 1) frames within tol — still ALIGN.
    for _ in range(cfg.align_debounce_frames - 1):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
        assert out["state"] == "ALIGN"
        assert out["v"] == 0.0
    # Final frame commits.
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["state"] == "DONE"
    assert out["reached"] is True
    assert out["v"] == 0.0


def test_align_streak_breaks_on_out_of_tolerance_frame():
    """A mid-streak miss resets the streak; requires full debounce again."""
    cfg = _align_cfg(d_stop_m=0.05, align_debounce_frames=3)
    c = VisualServoController(cfg)
    _enter_align(c)
    # Two in-tolerance frames.
    for _ in range(2):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
        assert out["state"] == "ALIGN"
    # Out-of-tolerance frame (horiz way off) — streak resets.
    out = c.step(_det(320, 240, depth_m=3.0), tilt_deg_cur=60.0)
    assert out["state"] == "ALIGN"
    assert c._align_streak == 0


def test_align_holds_and_resets_streak_when_depth_missing():
    """Lost detection during ALIGN: hold (v=0), reset streak; stay ALIGN."""
    cfg = _align_cfg(d_stop_m=0.05)
    c = VisualServoController(cfg)
    _enter_align(c)
    # Build partial streak.
    c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert c._align_streak == 2
    # Detection-less frame.
    out = c.step(None, tilt_deg_cur=88.0)
    assert out["state"] == "ALIGN"
    assert out["v"] == 0.0
    assert out["omega"] == 0.0
    assert c._align_streak == 0


def test_align_timeout_commits_done_even_when_unaligned():
    """align_timeout_s expiry → DONE regardless of alignment state."""
    cfg = _align_cfg(d_stop_m=0.40, align_timeout_s=0.2)  # short timeout
    c = VisualServoController(cfg)
    _enter_align(c)
    # Feed too-close depth so it would keep reversing forever.
    n = int(cfg.align_timeout_s / cfg.dt) + 5
    out = None
    for _ in range(n):
        out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["state"] == "DONE"
    assert out["reached"] is True


def test_align_tilt_cmd_held_at_last_value():
    """During ALIGN, tilt_cmd is frozen at the last TRACK value."""
    cfg = _align_cfg(d_stop_m=0.05)
    c = VisualServoController(cfg)
    _enter_align(c)
    held_tilt = c._last_tilt_cmd_deg
    out = c.step(_det(320, 240, depth_m=1.5), tilt_deg_cur=88.0)
    assert out["tilt_cmd_deg"] == pytest.approx(held_tilt, abs=1e-9)


def test_lost_state_emits_live_tilt_not_stale_init():
    """Bug fix: before TRACK ever runs, _handle_lost must emit the *live* mast tilt
    (via _last_tilt_cmd_deg) rather than the reset() default of 0.0.

    Without this, the driver pins the servo to 0° during initial SEARCH, making
    elevated targets unreachable (the failure mode that surfaced in Task 10
    integration tests for off-axis targets).
    """
    c = _ctrl()
    # No prior detection: controller has never been in TRACK.
    out = c.step(None, tilt_deg_cur=45.0)
    assert out["state"] == "COAST"
    assert out["tilt_cmd_deg"] == pytest.approx(45.0, abs=1e-9)
    # After many lost frames, still tracking live mast.
    for _ in range(20):
        out = c.step(None, tilt_deg_cur=42.0)
    assert out["state"] == "SEARCH"
    assert out["tilt_cmd_deg"] == pytest.approx(42.0, abs=1e-9)
