"""Visual-Servo Controller — bbox + depth + tilt → (v, ω, tilt_cmd, state).

State machine (spec §5):
    TRACK → COAST → HOLD → SEARCH → FAIL
        ↓
       DONE

This file implements the controller as a pure function `step()` over
(detection, current tilt). The driver loop ([visual_servo_driver.py]) wraps it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class VisualServoConfig:
    # ── camera frame ──
    img_w: int = 640
    img_h: int = 480

    # ── gains ──
    kp_tilt: float = 0.05         # deg / px
    ki_tilt: float = 0.0
    kp_h: float = 0.005           # (rad/s) / px
    ki_h: float = 0.0003
    kd_h: float = 0.000
    kp_v: float = 1.0             # (m/s) / m

    # ── limits ──
    v_max: float = 0.3
    omega_max: float = 1.0
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 95.0
    # Per-frame tilt slew cap [deg/s]. TILT_ASYNC is open-loop (no encoder
    # feedback), so the controller assumes each command is reached instantly.
    # Capping the command rate to the mast's physical slew keeps believed tilt
    # ≈ actual, preventing the runaway where err_y stays large and tilt winds to
    # max while the mast lags. None = unlimited (preserves legacy / test behavior).
    tilt_max_rate_dps: Optional[float] = None

    # ── stop ──
    d_stop_m: float = 0.20
    tilt_stop_range_deg: Tuple[float, float] = (85.0, 95.0)
    # Tilt-based approach brake. Once tilt_cmd_deg exceeds
    # `tilt_brake_start_deg`, v is multiplied by a linear scale that reaches
    # 0 at `tilt_stop_range_deg[0]`. Decouples deceleration from horiz_dist
    # (depth is unreliable when the bell is overhead), using tilt as a clean
    # geometric proxy for "we're nearly under it". None = disabled (legacy).
    tilt_brake_start_deg: Optional[float] = None

    # ── differential-drive geometry (for ω_L/ω_R output) ──
    wheel_diameter: float = 0.10
    wheel_base: float = 0.30
    max_wheel_omega: float = 30.0

    # ── FSM ──
    coast_lost_frames: int = 3
    hold_lost_frames: int = 15
    coast_speed_scale: float = 0.7
    # SEARCH 동작: 회전 대신 천천히 전진(creep). 헤딩 락을 유지하면서
    # 종을 향해 다가가다 보면 (대개 종이 정면이라) 재포착 가능성이 높고,
    # 회전 search 시 발생하던 limit-cycle 위험도 없다.
    search_creep_v: float = 0.05         # [m/s] forward creep during SEARCH
    search_timeout_s: float = 30.0

    # ── robustness vs. 종 vertical oscillation (spec §9) ──
    horiz_dist_lp_alpha: float = 0.2     # LPF α on horiz_dist (τ ≈ 0.27s)
    tilt_err_deadband_px: int = 8        # |err_y_px| < N → tilt 갱신 skip
    stop_debounce_frames: int = 3        # stop 조건 연속 N 프레임 필요

    # ── post-stop fine alignment (horiz_dist → d_stop_m) ──
    # 정지 조건 충족 후 별도 ALIGN 단계에서 아주 느린 속도로 전후진하여
    # horiz_dist 를 d_stop_m 에 맞춘다. tilt-brake 가 켜진 기본 설정에서는
    # 정지 시점의 horiz_dist 가 target 보다 가깝거나 멀 수 있어 마무리 정렬이
    # 필요. enabled=False (기본) → 기존 TRACK→DONE 직행 거동 그대로.
    align_enabled: bool = False
    align_v: float = 0.05                # [m/s] 매우 느린 정렬 속도
    align_tol_m: float = 0.02            # |horiz_dist - d_stop_m| < tol → 만족
    align_debounce_frames: int = 5       # 만족 프레임 연속 N → DONE
    align_timeout_s: float = 10.0        # 안전 timeout (도달 못 해도 commit)

    # ── loop ──
    dt: float = 0.067            # 15 Hz


class VisualServoController:
    def __init__(self, cfg: Optional[VisualServoConfig] = None):
        self.cfg = cfg if cfg is not None else VisualServoConfig()
        self.reset()

    def reset(self) -> None:
        self._state: str = "TRACK"
        self._lost_frames: int = 0
        self._last_err_x_px: float = 0.0
        self._integ_x: float = 0.0
        self._integ_y: float = 0.0
        # None = no valid previous TRACK frame yet → suppress derivative kick on
        # the first TRACK after reset / re-acquire. With sparse detection the
        # stale _prev_err_x produces a huge false d_err that saturates ω, whipping
        # the robot through the target on every reacquisition.
        self._prev_err_x: Optional[float] = None
        self._last_wheel: Tuple[float, float] = (0.0, 0.0)
        self._last_tilt_cmd_deg: float = 0.0
        self._search_elapsed_s: float = 0.0
        # bell-oscillation robustness state (spec §9)
        self._horiz_dist_filt: Optional[float] = None  # LPF state, None = uninit
        self._stop_streak: int = 0                     # consecutive stop frames
        # ALIGN-phase state
        self._align_streak: int = 0
        self._align_elapsed_s: float = 0.0

    # ── public API ──
    def step(
        self,
        detection: Optional[Dict],
        tilt_deg_cur: float,
    ) -> Dict:
        # ALIGN is sticky: once entered, stay until DONE (or timeout). Detection
        # may be None during ALIGN — handled inside _align (hold + streak break).
        if self._state == "ALIGN":
            return self._align(detection, tilt_deg_cur)
        if detection is not None:
            return self._track(detection, tilt_deg_cur)
        return self._handle_lost(tilt_deg_cur)

    # ── TRACK (Task 4 minimal) ──
    def _track(self, detection: Dict, tilt_deg_cur: float) -> Dict:
        c = self.cfg
        # capture lost state BEFORE resetting — used below to suppress derivative
        # kick on re-acquisition (stale _prev_err_x → false ω spike).
        just_reacquired = self._lost_frames > 0
        # reset lost counters on found
        self._lost_frames = 0
        self._search_elapsed_s = 0.0
        self._state = "TRACK"

        bbox = detection["bbox"]
        # depth_m may be None: bbox는 유효하지만 깊이 ROI가 비어 거리 미상.
        # 이 경우 조향/틸트는 계속하고 전진(v)·정지판정만 건너뛴다 (heading lock 유지).
        depth_raw = detection.get("depth_m")
        have_depth = depth_raw is not None and math.isfinite(float(depth_raw))
        cx_px = 0.5 * (bbox[0] + bbox[2])
        cy_px = 0.5 * (bbox[1] + bbox[3])
        err_x_px = cx_px - (c.img_w / 2.0)
        err_y_px = cy_px - (c.img_h / 2.0)

        if have_depth:
            horiz_dist_raw = float(depth_raw) * math.cos(math.radians(tilt_deg_cur))
            # LPF on horiz_dist (spec §4.1 step 2.5)
            if self._horiz_dist_filt is None:
                self._horiz_dist_filt = horiz_dist_raw
            else:
                a = c.horiz_dist_lp_alpha
                self._horiz_dist_filt = a * horiz_dist_raw + (1.0 - a) * self._horiz_dist_filt
            horiz_dist_filt = self._horiz_dist_filt
        else:
            horiz_dist_raw = float("nan")
            # keep last LPF state (do not poison it); report nan if never seen
            horiz_dist_filt = (self._horiz_dist_filt
                               if self._horiz_dist_filt is not None else float("nan"))

        # tilt dead-band: small err_y_px → skip tilt update (spec §4.1 step 2.5)
        if abs(err_y_px) < c.tilt_err_deadband_px:
            err_y_px_eff = 0.0
        else:
            err_y_px_eff = err_y_px

        # tilt PI
        self._integ_y += err_y_px_eff * c.dt
        d_tilt = -c.kp_tilt * err_y_px_eff - c.ki_tilt * self._integ_y
        # slew-rate limit (open-loop mast, see tilt_max_rate_dps doc)
        if c.tilt_max_rate_dps is not None:
            d_max = c.tilt_max_rate_dps * c.dt
            d_tilt = self._clip(d_tilt, -d_max, d_max)
        tilt_cmd_deg = self._clip(tilt_deg_cur + d_tilt,
                                  c.tilt_min_deg, c.tilt_max_deg)

        # heading PID — note negative gains (err_x>0 → ω<0).
        # Derivative anti-kick: on the first TRACK frame (after reset) or on
        # re-acquisition after a lost gap, _prev_err_x is stale (or 0). Computing
        # d_err against it produces a huge false spike that saturates ω and
        # whips the robot through the target. Treat such frames as fresh.
        self._integ_x += err_x_px * c.dt
        if self._prev_err_x is None or just_reacquired:
            d_err_x = 0.0
        else:
            d_err_x = (err_x_px - self._prev_err_x) / c.dt
        self._prev_err_x = err_x_px
        omega = (- c.kp_h * err_x_px
                 - c.ki_h * self._integ_x
                 - c.kd_h * d_err_x)
        omega = self._clip(omega, -c.omega_max, c.omega_max)

        # forward velocity (use filtered horiz_dist) — only when depth is valid.
        # No depth → can't advance safely; hold position (v=0) but keep steering.
        if have_depth:
            align = max(0.2, 1.0 - abs(err_x_px) / (c.img_w / 2.0))
            # Tilt-based approach brake: as tilt_cmd rises through the brake
            # zone, scale v linearly to 0. Tilt is a robust proxy for "near
            # the bell" — works regardless of depth noise at close range.
            # `brake_hi` is the stop-band UPPER edge (e.g. 95°), not the
            # lower edge. That way the brake still leaves some forward speed
            # at the moment tilt enters the stop band (e.g. 85°), so the
            # robot can creep into a DONE-trigger position instead of
            # stalling just outside the band with v=0.
            tilt_brake = 1.0
            if c.tilt_brake_start_deg is not None:
                brake_lo = c.tilt_brake_start_deg
                brake_hi = c.tilt_stop_range_deg[1]
                if tilt_cmd_deg >= brake_hi:
                    tilt_brake = 0.0
                elif tilt_cmd_deg > brake_lo:
                    tilt_brake = (brake_hi - tilt_cmd_deg) / max(
                        brake_hi - brake_lo, 1e-6
                    )
            v = self._clip(c.kp_v * horiz_dist_filt * align * tilt_brake,
                           0.0, c.v_max)
        else:
            v = 0.0

        # stop? — debounce: require stop_debounce_frames consecutive frames.
        # depth-less frame can't satisfy the distance gate → breaks the streak.
        #
        # When the tilt-brake is enabled, the brake itself prevents tilt cmd
        # from overshooting reality (it only reaches the stop band when v has
        # been ramped down by the approach gain). Tilt alone is then a clean
        # geometric "we're under it" signal, and the horiz_dist gate is
        # dropped — otherwise the robot brakes to v=0 just outside d_stop_m
        # and gets stuck there forever (no motion → horiz_dist never drops
        # below the gate).
        tilt_lo, tilt_hi = c.tilt_stop_range_deg
        in_tilt_band = tilt_lo <= tilt_cmd_deg <= tilt_hi
        if c.tilt_brake_start_deg is not None:
            cond = in_tilt_band  # depth not required; brake guarantees approach
        else:
            cond = (have_depth and in_tilt_band
                    and (horiz_dist_filt < c.d_stop_m))
        if cond:
            self._stop_streak += 1
        else:
            self._stop_streak = 0
        if self._stop_streak >= c.stop_debounce_frames:
            # Branch: ALIGN (post-stop fine alignment) vs. legacy direct DONE.
            # ALIGN keeps the driver loop running so a slow forward/back creep
            # can pull horiz_dist onto d_stop_m before reaching="True" exits.
            v = 0.0
            omega = 0.0
            if c.align_enabled:
                self._state = "ALIGN"
                self._align_streak = 0
                self._align_elapsed_s = 0.0
            else:
                self._state = "DONE"

        wL, wR = self._wheel_omegas(v, omega)
        self._last_wheel = (wL, wR)
        self._last_tilt_cmd_deg = tilt_cmd_deg
        self._last_err_x_px = err_x_px

        return {
            "state": self._state,
            "v": v,
            "omega": omega,
            "wheel_omega_left": wL,
            "wheel_omega_right": wR,
            "tilt_cmd_deg": tilt_cmd_deg,
            "err_x_px": err_x_px,
            "err_y_px": err_y_px,
            "horiz_dist": horiz_dist_filt,
            "horiz_dist_raw": horiz_dist_raw,
            "reached": self._state == "DONE",
            "failed": False,
        }

    # ── ALIGN (post-stop fine alignment) ──
    def _align(self, detection: Optional[Dict], tilt_deg_cur: float) -> Dict:
        """Slow forward/backward creep to drive horiz_dist → d_stop_m.

        Heading frozen (ω=0) and tilt_cmd held at last value — the stop trigger
        already confirmed both were within band. Without a valid depth we can't
        verify alignment, so we just hold (and reset the streak); the timeout
        safety still commits to DONE eventually.
        """
        c = self.cfg
        self._align_elapsed_s += c.dt
        tilt_cmd_deg = self._last_tilt_cmd_deg

        have_depth = False
        horiz_dist_raw = float("nan")
        horiz_dist_filt = (self._horiz_dist_filt
                           if self._horiz_dist_filt is not None else float("nan"))
        v = 0.0
        if detection is not None:
            depth_raw = detection.get("depth_m")
            have_depth = (depth_raw is not None
                          and math.isfinite(float(depth_raw)))
            if have_depth:
                horiz_dist_raw = float(depth_raw) * math.cos(
                    math.radians(tilt_deg_cur))
                if self._horiz_dist_filt is None:
                    self._horiz_dist_filt = horiz_dist_raw
                else:
                    a = c.horiz_dist_lp_alpha
                    self._horiz_dist_filt = (
                        a * horiz_dist_raw
                        + (1.0 - a) * self._horiz_dist_filt
                    )
                horiz_dist_filt = self._horiz_dist_filt

                err = horiz_dist_filt - c.d_stop_m   # >0 too far, <0 too close
                if abs(err) < c.align_tol_m:
                    self._align_streak += 1
                    v = 0.0
                else:
                    v = math.copysign(c.align_v, err)
                    self._align_streak = 0

        if not have_depth:
            self._align_streak = 0

        # Exit: tolerance streak satisfied, or timeout commits regardless.
        if (self._align_streak >= c.align_debounce_frames
                or self._align_elapsed_s > c.align_timeout_s):
            self._state = "DONE"
            v = 0.0

        omega = 0.0
        wL, wR = self._wheel_omegas(v, omega)
        self._last_wheel = (wL, wR)

        return {
            "state": self._state,
            "v": v,
            "omega": omega,
            "wheel_omega_left": wL,
            "wheel_omega_right": wR,
            "tilt_cmd_deg": tilt_cmd_deg,
            "err_x_px": self._last_err_x_px,
            "err_y_px": 0.0,
            "horiz_dist": horiz_dist_filt,
            "horiz_dist_raw": horiz_dist_raw,
            "reached": self._state == "DONE",
            "failed": False,
        }

    # ── lost detection: dispatcher ──
    def _handle_lost(self, tilt_deg_cur: float) -> Dict:
        c = self.cfg
        self._lost_frames += 1
        # detection-less frame breaks stop streak (spec §4.1 step 6)
        self._stop_streak = 0
        # Keep cmd in sync with live mast so lost-state TILT_ASYNC doesn't slam
        # the servo to the stale init value (was 0.0 before TRACK first ran).
        self._last_tilt_cmd_deg = tilt_deg_cur
        if self._lost_frames < c.coast_lost_frames:
            return self._coast()
        if self._lost_frames < c.hold_lost_frames:
            return self._hold()
        # SEARCH — accumulate elapsed
        self._search_elapsed_s += c.dt
        if self._search_elapsed_s > c.search_timeout_s:
            return self._fail()
        return self._search()

    def _coast(self) -> Dict:
        c = self.cfg
        self._state = "COAST"
        wL = self._last_wheel[0] * c.coast_speed_scale
        wR = self._last_wheel[1] * c.coast_speed_scale
        return self._pack(wL, wR, v=0.0, omega=0.0)

    def _hold(self) -> Dict:
        self._state = "HOLD"
        return self._pack(0.0, 0.0, v=0.0, omega=0.0)

    def _search(self) -> Dict:
        c = self.cfg
        self._state = "SEARCH"
        # 회전 대신 전진 creep — 헤딩 락을 유지해서 재포착 시 컨트롤러가
        # 그대로 이어받게 한다 (회전 search 는 limit-cycle 위험).
        v = c.search_creep_v
        omega = 0.0
        wL, wR = self._wheel_omegas(v, omega)
        return self._pack(wL, wR, v=v, omega=omega)

    def _fail(self) -> Dict:
        self._state = "FAIL"
        return self._pack(0.0, 0.0, v=0.0, omega=0.0, failed=True)

    def _pack(self, wL: float, wR: float, v: float, omega: float,
              failed: bool = False) -> Dict:
        return {
            "state": self._state,
            "v": v,
            "omega": omega,
            "wheel_omega_left": wL,
            "wheel_omega_right": wR,
            "tilt_cmd_deg": self._last_tilt_cmd_deg,
            "err_x_px": self._last_err_x_px,
            "err_y_px": 0.0,
            "horiz_dist": float("nan"),
            "horiz_dist_raw": float("nan"),
            "reached": False,
            "failed": failed,
        }

    # ── kinematics + utilities ──
    def _wheel_omegas(self, v: float, omega: float) -> Tuple[float, float]:
        c = self.cfg
        r = c.wheel_diameter / 2.0
        v_L = v - omega * c.wheel_base / 2.0
        v_R = v + omega * c.wheel_base / 2.0
        wL = self._clip(v_L / r, -c.max_wheel_omega, c.max_wheel_omega)
        wR = self._clip(v_R / r, -c.max_wheel_omega, c.max_wheel_omega)
        return wL, wR

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
