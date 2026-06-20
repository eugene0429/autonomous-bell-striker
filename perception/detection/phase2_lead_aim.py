"""Lead-aim tracker for vertically oscillating bell target.

Motion model (triangular wave with random half-period):
  - Bell moves at constant velocity between endpoints (NOT simple harmonic).
  - At each endpoint v reverses sign and a new half-period H is randomly
    sampled from [H_min, H_max].
  - Travel distance per half-cycle is fixed (peak-to-peak = 2·amplitude).
  - Vertical motion only; lateral (x, y) is treated as quasi-static and
    re-measured per-frame by the caller.

Prediction strategy:
  - Per-frame z observation feeds the tracker.
  - Endpoint detected as direction reversal: in the current half-cycle
    buffer, find an interior extremum whose pre/post slopes have opposite
    signs and both |v| > eps.
  - Within a half-cycle, fit constant velocity v from recent samples
    (linear regression slope, fit_window_samples points).
  - Extrapolate linearly: z(t_arrival) = z(now) + v·Δt.
  - Safety: reject opportunity if the predicted z would lie within
    safety_margin of an endpoint (= half-cycle ends mid-flight). On
    reject, caller waits and re-evaluates the next frame.

Used by: run_phase2_aiming.py:run_phase2_lead_aim.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class LeadAimParams:
    """Lead-aim tracker configuration.

    amplitude_m         : half peak-to-peak (= |z_endpoint − z_center|).
                          User spec: 50 cm per half-cycle → A = 0.25 m.
    half_period_min_s   : fastest half-cycle. User spec: 3.0 s.
    half_period_max_s   : slowest half-cycle. User spec: 6.0 s.
    min_fit_samples     : min samples per side for slope fit / endpoint
                          detection.
    fit_window_samples  : trailing samples used for velocity fit (smaller
                          = lower latency, larger = less noise).
    endpoint_v_eps_mps  : |v| below this is treated as "stationary" — used
                          to gate endpoint detection and `ready` flag so
                          near-extremum jitter doesn't masquerade as a
                          half-cycle reversal.
    safety_margin_m     : require predicted z to stay strictly inside
                          (center ± A − margin). Larger margin = safer but
                          fewer firing opportunities per half-cycle.
    """
    amplitude_m: float = 0.25
    half_period_min_s: float = 3.0
    half_period_max_s: float = 6.0
    min_fit_samples: int = 5
    fit_window_samples: int = 8
    endpoint_v_eps_mps: float = 0.03
    safety_margin_m: float = 0.03

    @property
    def v_min_mps(self) -> float:
        """Slowest expected |v| = 2A / H_max."""
        return 2.0 * self.amplitude_m / self.half_period_max_s

    @property
    def v_max_mps(self) -> float:
        """Fastest expected |v| = 2A / H_min."""
        return 2.0 * self.amplitude_m / self.half_period_min_s


class BellMotionTracker:
    """Online z(t) tracker → per-half-cycle linear v → future-z prediction.

    Usage:
        tracker = BellMotionTracker(LeadAimParams())
        # per frame
        tracker.update(time.monotonic(), z_plate_frame)
        if tracker.ready and tracker.is_safe_to_fire(dt_ahead):
            z_pred = tracker.predict_z(dt_ahead)
            ...
    """

    def __init__(self, params: LeadAimParams):
        self.p = params
        # Global buffer (kept short for memory, used for debug/inspection).
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=240)
        # Samples since last detected endpoint (current half-cycle).
        self._cycle: List[Tuple[float, float]] = []
        self._endpoints_seen: int = 0
        self._last_endpoint_t: Optional[float] = None
        self._last_endpoint_z: Optional[float] = None
        self._z_center: Optional[float] = None
        self._v: Optional[float] = None  # signed m/s, current half-cycle

    # ── ingestion ──
    def update(self, t: float, z: float) -> None:
        self._samples.append((t, z))
        self._cycle.append((t, z))
        self._try_detect_endpoint()
        self._update_velocity()

    def reset(self) -> None:
        """Forget all history. Call after a strike if the impact may have
        knocked the bell into a new motion regime."""
        self._samples.clear()
        self._cycle.clear()
        self._endpoints_seen = 0
        self._last_endpoint_t = None
        self._last_endpoint_z = None
        self._z_center = None
        self._v = None

    # ── endpoint detection ──
    def _try_detect_endpoint(self) -> None:
        n = len(self._cycle)
        if n < 2 * self.p.min_fit_samples:
            return
        ts = np.fromiter((s[0] for s in self._cycle), dtype=np.float64, count=n)
        zs = np.fromiter((s[1] for s in self._cycle), dtype=np.float64, count=n)
        # Test both extrema as endpoint candidates; the first that passes wins.
        for cand_idx in (int(np.argmax(zs)), int(np.argmin(zs))):
            if cand_idx < self.p.min_fit_samples:
                continue
            if n - cand_idx - 1 < self.p.min_fit_samples:
                continue
            v_before = _linfit_slope(ts[: cand_idx + 1], zs[: cand_idx + 1])
            v_after = _linfit_slope(ts[cand_idx:], zs[cand_idx:])
            if (abs(v_before) > self.p.endpoint_v_eps_mps
                    and abs(v_after) > self.p.endpoint_v_eps_mps
                    and np.sign(v_before) != np.sign(v_after)):
                self._register_endpoint(
                    float(ts[cand_idx]), float(zs[cand_idx]),
                    float(np.sign(v_after)),
                )
                # Trim cycle buffer so only post-endpoint samples remain.
                self._cycle = self._cycle[cand_idx:]
                return

    def _register_endpoint(self, t_e: float, z_e: float, v_after_sign: float) -> None:
        # v > 0 after endpoint ⇒ endpoint was a minimum ⇒ center is +A above.
        if v_after_sign > 0:
            z_center_est = z_e + self.p.amplitude_m
        else:
            z_center_est = z_e - self.p.amplitude_m
        if self._z_center is None:
            self._z_center = z_center_est
        else:
            # Blend to dampen jitter once multiple endpoints have been seen.
            self._z_center = 0.5 * self._z_center + 0.5 * z_center_est
        self._last_endpoint_t = t_e
        self._last_endpoint_z = z_e
        self._endpoints_seen += 1

    # ── velocity fit ──
    def _update_velocity(self) -> None:
        if len(self._cycle) < self.p.min_fit_samples:
            return
        recent = self._cycle[-self.p.fit_window_samples:]
        ts = np.fromiter((s[0] for s in recent), dtype=np.float64, count=len(recent))
        zs = np.fromiter((s[1] for s in recent), dtype=np.float64, count=len(recent))
        self._v = _linfit_slope(ts, zs)

    # ── queries ──
    @property
    def ready(self) -> bool:
        """True once z_center is bootstrapped (≥1 endpoint) AND v is well
        above noise floor."""
        return (self._endpoints_seen >= 1
                and self._z_center is not None
                and self._v is not None
                and abs(self._v) > self.p.endpoint_v_eps_mps)

    @property
    def velocity(self) -> Optional[float]:
        return self._v

    @property
    def z_center(self) -> Optional[float]:
        return self._z_center

    @property
    def endpoints_seen(self) -> int:
        return self._endpoints_seen

    def latest_sample(self) -> Optional[Tuple[float, float]]:
        return self._samples[-1] if self._samples else None

    def last_endpoint(self) -> Optional[Tuple[float, float]]:
        """(t, z) of most recently detected reversal, or None if 0 seen."""
        if self._last_endpoint_t is None or self._last_endpoint_z is None:
            return None
        return (self._last_endpoint_t, self._last_endpoint_z)

    def predict_z(self, dt_ahead: float) -> Optional[float]:
        """Linear extrapolation z(now + dt_ahead). Returns unclamped value —
        caller must use `is_safe_to_fire` to gate endpoint-crossing cases."""
        if not self.ready:
            return None
        _, z_now = self._samples[-1]
        return z_now + self._v * dt_ahead

    def time_to_next_endpoint(self) -> Optional[float]:
        """Seconds until current half-cycle ends (= predicted z hits ±A).
        Returns None if not ready or v ≈ 0."""
        if not self.ready:
            return None
        _, z_now = self._samples[-1]
        z_next = self._z_center + (self.p.amplitude_m if self._v > 0
                                   else -self.p.amplitude_m)
        return (z_next - z_now) / self._v

    def is_safe_to_fire(self, dt_ahead: float) -> bool:
        """Predicted z stays strictly inside (z_center ± (A − margin)).

        Equivalent to "no endpoint crossing during dt_ahead, with margin"
        — protects against the case where the half-cycle ends and v
        reverses while the projectile is in flight."""
        if not self.ready:
            return False
        z_pred = self.predict_z(dt_ahead)
        if z_pred is None:
            return False
        return abs(z_pred - self._z_center) <= self.p.amplitude_m - self.p.safety_margin_m


def _linfit_slope(ts: np.ndarray, zs: np.ndarray) -> float:
    """Slope of OLS linear regression zs vs ts (m/s). 0 if degenerate."""
    if ts.size < 2:
        return 0.0
    t_mean = float(ts.mean())
    z_mean = float(zs.mean())
    num = float(((ts - t_mean) * (zs - z_mean)).sum())
    den = float(((ts - t_mean) ** 2).sum())
    if den < 1e-12:
        return 0.0
    return num / den
