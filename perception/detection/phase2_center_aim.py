"""Static center-aim tracker for vertically oscillating bell target.

Aiming strategy (alternative to lead-aim — see phase2_lead_aim.py):
  - Bell oscillates vertically between two **fixed** endpoints
    (z_top, z_bot).  In production these are auto-calibrated at the start
    of Phase 2 by observing the bell for a configurable window
    (see run_phase2_aiming._center_aim_calibrate) so this module does NOT
    estimate them online; CLI fallbacks exist for debug / re-runs.
  - z_center = (z_top + z_bot) / 2 is the *aim point*.  The leveling plate
    is positioned at (x_bell, y_bell, z_center) **once** at the start of
    Phase 2 and held there for every shot.
  - Hit condition: bell sphere (diameter `bell_diameter_m`) contains the
    projectile path at arrival time.  Vertically this means the bell's
    z at projectile arrival must satisfy |z − z_center| ≤ bell_radius.
  - We fire LOAD when the linear extrapolation z_now + v̂·Δt lies within
    (bell_radius − safety_margin) of z_center, where Δt = launcher delay
    (LOAD ack + ball flight; no plate-motion / settle term because the
    plate stays put).

Why this exists (in addition to lead-aim):
  - lead-aim requires z_center / amplitude bootstrap from observed endpoints
    (≥1 endpoint → up to H_max ≈ 6 s warmup) and skips fast half-cycles
    where `2·(A − margin) − 2·|v|·Δt ≤ 0` (no safe window). Aimlog analysis
    (2026-05-25) shows H as short as 1.7 s in practice, where lead-aim
    cannot fire safely with Δt=0.7 s.
  - center-aim has no warmup (z_center is known a priori) and fires twice
    per full cycle regardless of half-period, because the bell always
    passes through the center even in fast half-cycles. The narrower
    aim window (hit ⇔ within bell radius) is the price for a smaller Δt
    and no plate motion.

Motion model assumption is the same as lead-aim: piecewise-linear z(t)
within each half-cycle (constant velocity between endpoints).  Velocity
v̂ is fit from a trailing window of samples (same approach as
BellMotionTracker but without the endpoint-detection / center-bootstrap
state machine — those are unnecessary here).

Used by: run_phase2_aiming.py:run_phase2_center_aim.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class CenterAimParams:
    """Static center-aim tracker configuration.

    z_top_m, z_bot_m     : known bell-trajectory endpoints in plate frame
                           (meters, plate +Z = up). Pre-calibrated.
    bell_diameter_m      : bell sphere diameter [m]. Default 12 cm per the
                           hardware spec; hit window half-width = diameter/2.
    safety_margin_m      : require predicted z to lie within
                           (bell_radius − margin) of z_center. Trades
                           false-positive shots for fewer firing
                           opportunities. Typical 1–2 cm.
    min_fit_samples      : min samples before velocity fit is published.
                           Also used as the pre/post window size for
                           endpoint (direction-reversal) detection.
    fit_window_samples   : trailing samples used for velocity fit, OR
                           **0 to fit over the entire current half-cycle**
                           (the constant-velocity prior — bell moves at
                           constant v within a half-cycle by spec). Full-
                           cycle fit drops σ_v from ~2 cm/s (W=8) to
                           <0.2 cm/s once the cycle has >50 samples,
                           massively cleaning up the v̂ trace. The price
                           is a brief bias around undetected endpoints
                           (mixed pre/post-reversal samples), but |v̂|
                           shrinks toward 0 then so `ready` blocks
                           firing during the transition. Default 0.
    endpoint_v_eps_mps   : minimum |v| for (a) endpoint-detection sanity
                           check (both pre & post slopes must exceed
                           this and have opposite signs) and (b) the
                           `ready` flag — when |v̂| ≤ eps the direction
                           is ambiguous (we're near an extremum) and
                           firing is suppressed. Default 3 cm/s ≈ the
                           v_min predicted by 2A / H_max for spec
                           H_max = 6 s.
    direction_streak_required:
                           manual-mode (no calibration seeding)
                           shortcut: if we've seen N consecutive
                           same-sign frame-to-frame z-increments
                           (above `direction_streak_eps_m`), accept
                           the direction as established even before
                           the first endpoint detection. Lets the
                           manual flow start firing within ~5 frames
                           of stable motion instead of waiting up to
                           a full half-cycle for the first endpoint.
                           Ignored in auto-cal mode (seeding triggers
                           endpoints_seen ≥ 1 directly).
    direction_streak_eps_m:
                           noise floor for the streak check — diffs
                           below this magnitude are skipped (neither
                           extend nor reset the streak). Per-frame
                           travel at v_min ≈ 3 cm/s · 1/17.5 s ≈
                           1.7 mm; with σ_diff ≈ 12 mm a 5 mm floor
                           rejects ~30 % of slow-direction diffs as
                           noise-dominated but never confuses sign.
    """
    z_top_m: float
    z_bot_m: float
    bell_diameter_m: float = 0.12
    safety_margin_m: float = 0.02
    min_fit_samples: int = 5
    fit_window_samples: int = 0   # 0 → fit over entire current half-cycle
    endpoint_v_eps_mps: float = 0.03
    direction_streak_required: int = 5
    direction_streak_eps_m: float = 0.005

    def __post_init__(self) -> None:
        if self.z_top_m <= self.z_bot_m:
            raise ValueError(
                f"z_top_m ({self.z_top_m}) must be > z_bot_m ({self.z_bot_m}) "
                "(plate +Z = up, top endpoint sits at larger z)"
            )
        if self.bell_diameter_m <= 0:
            raise ValueError(f"bell_diameter_m must be > 0, got {self.bell_diameter_m}")
        if self.safety_margin_m < 0:
            raise ValueError(
                f"safety_margin_m must be ≥ 0, got {self.safety_margin_m}"
            )
        if self.safety_margin_m >= self.bell_radius_m:
            raise ValueError(
                f"safety_margin_m ({self.safety_margin_m}) ≥ bell radius "
                f"({self.bell_radius_m}) — no firing opportunity would ever pass"
            )

    @property
    def z_center(self) -> float:
        return 0.5 * (self.z_top_m + self.z_bot_m)

    @property
    def amplitude_m(self) -> float:
        """Half peak-to-peak (= |z_endpoint − z_center|)."""
        return 0.5 * (self.z_top_m - self.z_bot_m)

    @property
    def bell_radius_m(self) -> float:
        return 0.5 * self.bell_diameter_m

    @property
    def fire_window_half_m(self) -> float:
        """|z_pred − z_center| must be ≤ this for fire trigger."""
        return self.bell_radius_m - self.safety_margin_m


class CenterAimTracker:
    """Endpoint-aware velocity tracker for center-aim mode.

    Maintains two buffers:
      - `_samples`: bounded deque of all (t, z) — used only for
        `latest_sample()` / debug.
      - `_cycle`  : samples since the last detected direction reversal.
        Velocity fit is computed from this buffer ONLY, so a fit window
        straddling an endpoint never contaminates v̂.

    Endpoint detection (mirrors lead-aim's BellMotionTracker):
      - Once `_cycle` has ≥ 2·min_fit_samples points, look for an
        interior extremum (argmax / argmin) whose pre/post linear-fit
        slopes have opposite signs AND both exceed `endpoint_v_eps_mps`
        in magnitude. If found → trim `_cycle` to keep only post-extremum
        samples; this advances the "current half-cycle" pointer.

    `ready` requires |v̂| > endpoint_v_eps_mps so the direction is
    unambiguous before any firing decision. Combined with the cycle-buffer
    trim, this prevents false-positive shots while the bell is near an
    endpoint (where v ≈ 0 and v̂ is dominated by noise) or where the fit
    window straddles a reversal.

    Usage:
        tracker = CenterAimTracker(CenterAimParams(z_top_m=2.09, z_bot_m=1.77))
        # optionally seed with prior observations (e.g. calibration samples)
        for (t, z) in calibration_samples:
            tracker.update(t, z)
        # then per frame in the firing loop
        tracker.update(time.monotonic(), z_plate_frame)
        if tracker.ready and tracker.should_fire(dt_ahead):
            ...
    """

    def __init__(self, params: CenterAimParams):
        self.p = params
        # Global ring (for `latest_sample()` and debugging)
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=240)
        # Current half-cycle buffer — trimmed on each endpoint detection.
        self._cycle: List[Tuple[float, float]] = []
        self._endpoints_seen: int = 0
        self._v: Optional[float] = None  # signed m/s, current half-cycle
        # Sign-streak heuristic (manual-mode fast-ready path).
        self._diff_sign_streak: int = 0
        self._last_diff_sign: int = 0

    # ── ingestion ──
    def update(self, t: float, z: float) -> None:
        # Sign-streak: compare to previous sample BEFORE appending.
        # Skip noise-floor diffs (don't extend nor reset).
        if self._samples:
            diff = z - self._samples[-1][1]
            if abs(diff) >= self.p.direction_streak_eps_m:
                new_sign = 1 if diff > 0 else -1
                if new_sign == self._last_diff_sign:
                    self._diff_sign_streak += 1
                else:
                    self._diff_sign_streak = 1
                self._last_diff_sign = new_sign

        self._samples.append((t, z))
        self._cycle.append((t, z))
        self._try_detect_endpoint()
        self._update_velocity()

    def reset(self) -> None:
        """Forget all history. Optional — call after a strike if the
        impact may have knocked the bell into a new motion regime."""
        self._samples.clear()
        self._cycle = []
        self._endpoints_seen = 0
        self._v = None
        self._diff_sign_streak = 0
        self._last_diff_sign = 0

    # ── endpoint detection ──
    def _try_detect_endpoint(self) -> None:
        n = len(self._cycle)
        if n < 2 * self.p.min_fit_samples:
            return
        ts = np.fromiter((s[0] for s in self._cycle), dtype=np.float64, count=n)
        zs = np.fromiter((s[1] for s in self._cycle), dtype=np.float64, count=n)
        # Test both extrema as endpoint candidates; first match wins.
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
                # Trim — keep only post-endpoint samples in current cycle.
                self._cycle = self._cycle[cand_idx:]
                self._endpoints_seen += 1
                # Direction reversed: streak from the OLD half-cycle is
                # stale. Reset; new streak builds from post-endpoint diffs.
                self._diff_sign_streak = 0
                self._last_diff_sign = 0
                return

    # ── velocity fit (over current half-cycle only) ──
    def _update_velocity(self) -> None:
        if len(self._cycle) < self.p.min_fit_samples:
            self._v = None
            return
        # fit_window_samples = 0 → leverage the constant-velocity-per-half-
        # cycle prior: fit over the entire current half-cycle buffer. As
        # samples accumulate σ_v ∝ 1/(T·√N) drops below the trailing-W
        # noise floor by an order of magnitude.
        if self.p.fit_window_samples > 0:
            recent = self._cycle[-self.p.fit_window_samples:]
        else:
            recent = self._cycle
        n = len(recent)
        ts = np.fromiter((s[0] for s in recent), dtype=np.float64, count=n)
        zs = np.fromiter((s[1] for s in recent), dtype=np.float64, count=n)
        self._v = _linfit_slope(ts, zs)

    # ── queries ──
    @property
    def ready(self) -> bool:
        """v̂ well-defined AND magnitude above the endpoint-noise floor
        AND direction has been observed long enough to trust.

        The third condition is satisfied either by:
          (a) at least one endpoint reversal observed → we know which
              half-cycle we're in (auto-cal mode hits this immediately
              via seeding), OR
          (b) N consecutive same-sign frame-to-frame z-diffs above the
              `direction_streak_eps_m` noise floor — a cheap heuristic
              for manual mode where seeding is absent and waiting for
              the first endpoint detection would delay the first shot
              by up to a full half-cycle.
        """
        if self._v is None or abs(self._v) <= self.p.endpoint_v_eps_mps:
            return False
        if self._endpoints_seen >= 1:
            return True
        return self._diff_sign_streak >= self.p.direction_streak_required

    @property
    def velocity(self) -> Optional[float]:
        return self._v

    @property
    def z_center(self) -> float:
        return self.p.z_center

    @property
    def endpoints_seen(self) -> int:
        return self._endpoints_seen

    @property
    def direction_streak(self) -> int:
        """Current consecutive same-sign-diff streak. See `ready` notes."""
        return self._diff_sign_streak

    def latest_sample(self) -> Optional[Tuple[float, float]]:
        return self._samples[-1] if self._samples else None

    def predict_z(self, dt_ahead: float) -> Optional[float]:
        """Linear extrapolation z(now + dt_ahead). None if not ready."""
        if not self.ready:
            return None
        _, z_now = self._samples[-1]
        return z_now + self._v * dt_ahead

    def should_fire(self, dt_ahead: float) -> bool:
        """True ⇔ predicted z at projectile arrival is within
        (bell_radius − safety_margin) of z_center.

        A tighter margin than just `bell_radius` accounts for the
        residual velocity-fit error (~1–2 cm at 0.4 s prediction horizon
        per aimlog analysis) and small launcher pointing error."""
        z_pred = self.predict_z(dt_ahead)
        if z_pred is None:
            return False
        return abs(z_pred - self.p.z_center) <= self.p.fire_window_half_m


def _linfit_slope(ts: np.ndarray, zs: np.ndarray) -> float:
    """Slope of OLS linear regression zs vs ts (m/s). 0 if degenerate.

    Identical to phase2_lead_aim._linfit_slope but kept private to this
    module so the two trackers stay independent (different evolution).
    """
    if ts.size < 2:
        return 0.0
    t_mean = float(ts.mean())
    z_mean = float(zs.mean())
    num = float(((ts - t_mean) * (zs - z_mean)).sum())
    den = float(((ts - t_mean) ** 2).sum())
    if den < 1e-12:
        return 0.0
    return num / den
