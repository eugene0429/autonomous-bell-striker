"""Tests for perception.detection.phase2_lead_aim.

Validates BellMotionTracker against a deterministic triangular-wave bell
simulator. Synthetic motion (not camera-based) so this runs in CI without
hardware.

Motion model (matches spec): bell moves at constant velocity within each
half-cycle; half-period H_i is fixed per cycle but may differ across
cycles to mimic the spec's random sampling.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from perception.detection.phase2_lead_aim import (
    BellMotionTracker,
    LeadAimParams,
    _linfit_slope,
)


# ────────────────────────── triangle-wave simulator ──────────────────────
class _TriangleWaveBell:
    """Deterministic triangular-wave bell motion generator.

    At t=0 bell sits at the start endpoint (bottom if start_at_bottom else
    top) and moves toward the opposite endpoint at constant v = 2A/H_0.
    At t = H_0 it reverses, v ← -2A/H_1, and so on.
    """

    def __init__(
        self,
        amplitude: float,
        z_center: float,
        half_periods: List[float],
        start_at_bottom: bool = True,
    ):
        self.A = float(amplitude)
        self.zc = float(z_center)
        self.Hs = [float(h) for h in half_periods]
        self.bdy = np.cumsum([0.0] + self.Hs)
        # start_dir = +1 ⇒ first endpoint is at zc - A, v_0 = +2A/H_0.
        self.start_dir = 1.0 if start_at_bottom else -1.0

    def _half_cycle_idx(self, t: float) -> int:
        if t < 0.0:
            return 0
        if t >= self.bdy[-1]:
            return len(self.Hs) - 1
        return int(np.searchsorted(self.bdy, t, side="right") - 1)

    def z(self, t: float) -> float:
        if t <= 0.0:
            return self.zc - self.start_dir * self.A
        t_clamped = min(t, float(self.bdy[-1]))
        i = self._half_cycle_idx(t_clamped)
        direction = self.start_dir * ((-1.0) ** i)
        z_start = self.zc - direction * self.A
        v = direction * 2.0 * self.A / self.Hs[i]
        return float(z_start + v * (t_clamped - self.bdy[i]))

    def v(self, t: float) -> float:
        if t < 0.0 or t >= self.bdy[-1]:
            return 0.0
        i = self._half_cycle_idx(t)
        direction = self.start_dir * ((-1.0) ** i)
        return float(direction * 2.0 * self.A / self.Hs[i])

    def endpoints(self) -> List[Tuple[float, float]]:
        """Return (t, z) of every reversal endpoint, including t=0."""
        out: List[Tuple[float, float]] = []
        for i in range(len(self.bdy)):
            direction = self.start_dir * ((-1.0) ** i)
            out.append((float(self.bdy[i]), self.zc - direction * self.A))
        return out


def _feed(
    tracker: BellMotionTracker,
    bell: _TriangleWaveBell,
    t_start: float,
    t_end: float,
    dt: float,
    noise_std: float = 0.0,
    seed: int = 0,
) -> None:
    """Sample bell at dt (optional gaussian noise) and feed to tracker."""
    rng = np.random.default_rng(seed)
    ts = np.arange(t_start, t_end + 1e-9, dt)
    for t in ts:
        z = bell.z(float(t))
        if noise_std > 0.0:
            z += float(rng.normal(0.0, noise_std))
        tracker.update(float(t), float(z))


# ──────────────────────────── _linfit_slope ─────────────────────────────
class TestLinFitSlope:
    def test_single_point_returns_zero(self):
        assert _linfit_slope(np.array([1.0]), np.array([2.0])) == 0.0

    def test_zero_variance_t_returns_zero(self):
        ts = np.array([1.0, 1.0, 1.0])
        zs = np.array([0.0, 1.0, 2.0])
        assert _linfit_slope(ts, zs) == 0.0

    def test_known_positive_slope(self):
        ts = np.linspace(0.0, 1.0, 11)
        zs = 0.5 + 3.0 * ts
        assert _linfit_slope(ts, zs) == pytest.approx(3.0)

    def test_known_negative_slope(self):
        ts = np.linspace(0.0, 2.0, 21)
        zs = 1.0 - 0.4 * ts
        assert _linfit_slope(ts, zs) == pytest.approx(-0.4)


# ──────────────────────────── warmup behavior ────────────────────────────
class TestWarmup:
    def test_empty_tracker_not_ready(self):
        t = BellMotionTracker(LeadAimParams())
        assert t.ready is False
        assert t.predict_z(0.5) is None
        assert t.is_safe_to_fire(0.5) is False
        assert t.velocity is None
        assert t.z_center is None
        assert t.endpoints_seen == 0
        assert t.latest_sample() is None

    def test_single_half_cycle_no_endpoint_not_ready(self):
        # Constant-velocity line through one half-cycle. No reversal ⇒ no
        # endpoint detected ⇒ z_center None ⇒ not ready.
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [p.half_period_min_s])
        _feed(t, bell, 0.0, p.half_period_min_s - 0.1, dt=1 / 30)
        assert t.endpoints_seen == 0
        assert t.ready is False


# ───────────────────────── endpoint detection ───────────────────────────
class TestEndpointDetection:
    def test_top_endpoint_then_descent(self):
        # Start at bottom, cross top at t=4, continue down. z_center should
        # be recovered from the single endpoint within a few cm.
        p = LeadAimParams()
        t = BellMotionTracker(p)
        zc_true = 1.5
        bell = _TriangleWaveBell(p.amplitude_m, zc_true, [4.0, 4.0],
                                 start_at_bottom=True)
        _feed(t, bell, 0.0, 6.0, dt=1 / 30)
        assert t.endpoints_seen >= 1
        assert abs(t.z_center - zc_true) < 0.05

    def test_bottom_endpoint_then_ascent(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        zc_true = 0.8
        bell = _TriangleWaveBell(p.amplitude_m, zc_true, [4.0, 4.0],
                                 start_at_bottom=False)
        _feed(t, bell, 0.0, 6.0, dt=1 / 30)
        assert t.endpoints_seen >= 1
        assert abs(t.z_center - zc_true) < 0.05

    def test_last_endpoint_accessor(self):
        # Public accessor for logger / viewer integration.
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [4.0, 4.0])
        _feed(t, bell, 0.0, 6.0, dt=1 / 30)
        ep = t.last_endpoint()
        assert ep is not None
        t_ep, z_ep = ep
        # First reversal is the top endpoint at t=4.0, z=1.25.
        assert t_ep == pytest.approx(4.0, abs=0.05)
        assert z_ep == pytest.approx(1.25, abs=0.01)


# ─────────────────────────── velocity fit ───────────────────────────────
class TestVelocityEstimation:
    def test_velocity_matches_ground_truth_noiseless(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [4.0, 4.0])
        _feed(t, bell, 0.0, 2.0, dt=1 / 30)
        # v during first half-cycle: 2A/H = 0.5/4 = 0.125
        assert t.velocity == pytest.approx(bell.v(2.0), abs=0.005)

    def test_velocity_sign_flips_after_endpoint(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [5.0, 5.0])
        _feed(t, bell, 0.0, 7.0, dt=1 / 30)
        # After endpoint at t=5, bell descends ⇒ v should be negative.
        assert t.velocity < -0.05


# ──────────────────────────── prediction ────────────────────────────────
class TestPrediction:
    def test_predict_z_noiseless_matches_ground_truth(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [5.0, 5.0])
        _feed(t, bell, 0.0, 5.5, dt=1 / 30)
        z_pred = t.predict_z(0.3)
        assert z_pred is not None
        assert z_pred == pytest.approx(bell.z(5.8), abs=0.01)

    def test_predict_z_with_noise_rmse_bounded(self):
        # 2 mm gaussian noise (above typical RealSense depth jitter at 3m).
        # RMSE over multiple seeds should stay under 1.5 cm with Δt=0.5s.
        p = LeadAimParams()
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [5.0, 5.0, 5.0, 5.0])
        errs = []
        for seed in range(8):
            t = BellMotionTracker(p)
            _feed(t, bell, 0.0, 11.0, dt=1 / 30,
                  noise_std=0.002, seed=seed)
            if not t.ready:
                continue
            z_pred = t.predict_z(0.5)
            z_gt = bell.z(11.5)
            errs.append(z_pred - z_gt)
        assert len(errs) >= 6, "most trials should reach ready state"
        rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        assert rmse < 0.015

    def test_time_to_next_endpoint(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [5.0, 5.0, 5.0])
        _feed(t, bell, 0.0, 6.0, dt=1 / 30)
        # t=6.0 is 1.0s into 2nd half-cycle (5.0 → 10.0). τ = 4.0s.
        tau = t.time_to_next_endpoint()
        assert tau == pytest.approx(4.0, abs=0.3)


# ────────────────────────── safety gate ─────────────────────────────────
class TestSafetyGate:
    def test_safe_when_deep_in_band(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [5.0, 5.0])
        # Mid 2nd half-cycle (t=7.5): bell is right at z_center. Safe.
        _feed(t, bell, 0.0, 7.5, dt=1 / 30)
        assert t.ready
        assert t.is_safe_to_fire(0.1) is True

    def test_unsafe_when_crossing_endpoint_within_dt(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [5.0, 5.0, 5.0])
        # Feed past 1st endpoint (at t=5) and approach 2nd (at t=10).
        # At t=9.5, bell at z ≈ 0.80. Predicted z(t=10.0) ≈ 0.75 (the
        # endpoint itself) ⇒ outside safety band.
        _feed(t, bell, 0.0, 9.5, dt=1 / 30)
        assert t.ready
        assert t.is_safe_to_fire(0.5) is False

    def test_unsafe_when_not_ready(self):
        t = BellMotionTracker(LeadAimParams())
        assert t.is_safe_to_fire(0.5) is False


# ──────────────────────────── reset ─────────────────────────────────────
class TestReset:
    def test_reset_clears_all_state(self):
        p = LeadAimParams()
        t = BellMotionTracker(p)
        bell = _TriangleWaveBell(p.amplitude_m, 1.0, [4.0, 4.0])
        _feed(t, bell, 0.0, 6.0, dt=1 / 30)
        assert t.endpoints_seen >= 1
        assert t.ready

        t.reset()
        assert t.endpoints_seen == 0
        assert t.ready is False
        assert t.velocity is None
        assert t.z_center is None
        assert t.predict_z(0.5) is None
        assert t.is_safe_to_fire(0.5) is False
        assert t.latest_sample() is None
        assert t.last_endpoint() is None


# ───────────────────── z_center blending across multiple endpoints ──────
class TestMultipleEndpoints:
    def test_z_center_stable_across_variable_half_periods(self):
        # Variable H_i (matches spec's random-half-period assumption). With
        # noiseless data, blended z_center should equal ground truth.
        p = LeadAimParams()
        t = BellMotionTracker(p)
        zc_true = 0.7
        bell = _TriangleWaveBell(p.amplitude_m, zc_true,
                                 [4.0, 3.0, 5.0, 4.0, 3.5])
        _feed(t, bell, 0.0, 18.0, dt=1 / 30)
        assert t.endpoints_seen >= 3
        assert abs(t.z_center - zc_true) < 0.03
