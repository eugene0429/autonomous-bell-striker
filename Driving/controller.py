"""
Driving Controller — production module.

현재 pose (x, y, theta) + 목표 (target_x, target_y) 를 입력 받아
좌·우 바퀴 각속도를 출력하는 차동 구동 (differential drive) 제어기.

차량 모델
---------
- 2-wheel differential drive
- 모바일 중심과 바퀴축 일치 (회전 중심 = 베이스 중심)
- 파라미터: 바퀴 직경 (wheel_diameter), 바퀴 간격 (wheel_base)

제어 로직
---------
- 거리 비례 선속도 + 감속 반경 + 헤딩 오차 클 때 감속
- PID 각속도 (heading error 기반)
- (옵션) SLAM confidence 낮을 때 속도/각속도 스케일 다운

사용 예
-------
    from controller import DrivingController, ControllerConfig

    cfg = ControllerConfig(wheel_diameter=0.10, wheel_base=0.30)
    ctrl = DrivingController(cfg)

    while True:
        x, y, theta = get_pose_from_slam()
        out = ctrl.compute(x, y, theta, target_x, target_y)
        if out['reached']:
            break
        send_to_motors(out['wheel_omega_left'], out['wheel_omega_right'])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class ControllerConfig:
    # ── 차량 기하 (모바일 중심 = 바퀴축 중심) ──
    wheel_diameter: float = 0.10   # 바퀴 직경 [m]
    wheel_base: float = 0.30       # 좌우 바퀴 간격 [m]

    # ── 속도/각속도 한계 ──
    max_speed: float = 0.3         # 최대 선속도 [m/s]
    max_omega: float = 1.0         # 최대 각속도 [rad/s]
    max_wheel_omega: float = 30.0  # 바퀴 각속도 클립 [rad/s]

    # ── 제어기 게인 ──
    kp_linear: float = 0.8
    kp_angular: float = 1.5 
    ki_angular: float = 0.03
    kd_angular: float = 0

    # ── 거동 형상 ──
    slowdown_radius: float = 1.0   # 이 거리 이내부터 선속도 감속 [m]
    goal_tolerance: float = 0.3    # 도달 판정 반경 [m]

    # ── 적분 항 anti-windup ──
    integral_clip: float = 2.0

    # ── 제어 주기 (PID 미분/적분용) ──
    dt: float = 0.067              # 15 Hz 기준 [s]

    # ── SLAM 신뢰도 기반 감속 (옵션) ──
    enable_confidence_scaling: bool = True
    lowconf_threshold: float = 0.8
    lowconf_speed_scale: float = 0.3   # confidence 0 일 때 곱해질 최소 배율


# ──────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────
class DrivingController:
    """현재 pose + 목표 (x, y) → 좌·우 바퀴 각속도."""

    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self._prev_angle_error = 0.0
        self._integral_angle_error = 0.0

    # ── 상태 초기화 (페이즈 전환 시 호출) ──
    def reset(self) -> None:
        self._prev_angle_error = 0.0
        self._integral_angle_error = 0.0

    # ── 메인 API ──
    def compute(
        self,
        x: float,
        y: float,
        theta: float,
        target_x: float,
        target_y: float,
        slam_confidence: float = 1.0,
    ) -> Dict[str, float]:
        """
        한 스텝의 제어 명령 계산.

        Parameters
        ----------
        x, y, theta       현재 pose (world frame, theta in rad)
        target_x, target_y  목표 좌표 (world frame, m)
        slam_confidence   0.0 ~ 1.0, 낮으면 속도 감속 (옵션)

        Returns
        -------
        dict
            v                  : 선속도 [m/s]
            omega              : 각속도 [rad/s]
            wheel_omega_left   : 좌 바퀴 각속도 [rad/s]
            wheel_omega_right  : 우 바퀴 각속도 [rad/s]
            wheel_v_left       : 좌 바퀴 접선 속도 [m/s]
            wheel_v_right      : 우 바퀴 접선 속도 [m/s]
            distance           : 목표까지 거리 [m]
            angle_error        : 헤딩 오차 [rad, in (-pi, pi]]
            reached            : 도달 여부 (distance < goal_tolerance)
        """
        c = self.cfg

        # ── 1) 거리 / 헤딩 오차 ──
        dx = target_x - x
        dy = target_y - y
        distance = float(np.hypot(dx, dy))
        desired_heading = float(np.arctan2(dy, dx))
        angle_error = self._wrap_angle(desired_heading - theta)

        # ── 2) PID 각속도 ──
        self._integral_angle_error += angle_error * c.dt
        self._integral_angle_error = float(np.clip(
            self._integral_angle_error, -c.integral_clip, c.integral_clip))
        derivative = (angle_error - self._prev_angle_error) / c.dt
        self._prev_angle_error = angle_error

        omega = (c.kp_angular * angle_error
                 + c.ki_angular * self._integral_angle_error
                 + c.kd_angular * derivative)
        omega = float(np.clip(omega, -c.max_omega, c.max_omega))

        # ── 3) 선속도 (거리 비례 + 감속 반경 + 헤딩 오차 시 감속) ──
        speed_factor = min(1.0, distance / max(c.slowdown_radius, 1e-9))
        heading_factor = max(0.2, 1.0 - abs(angle_error) / np.pi)
        v = c.kp_linear * distance * speed_factor * heading_factor
        v = float(np.clip(v, 0.0, c.max_speed))

        # ── 4) SLAM confidence 기반 감속 ──
        if c.enable_confidence_scaling and slam_confidence < c.lowconf_threshold:
            ratio = max(0.0, slam_confidence) / max(c.lowconf_threshold, 1e-9)
            conf_scale = c.lowconf_speed_scale + (1.0 - c.lowconf_speed_scale) * ratio
            v *= conf_scale
            omega *= conf_scale

        # ── 5) 차동 구동 역기구학: (v, ω) → (v_L, v_R) → (ω_L, ω_R) ──
        # 모바일 중심 = 바퀴축 중심 가정:
        #   v   = (v_L + v_R) / 2
        #   ω   = (v_R - v_L) / wheel_base
        #   ⇒ v_L = v - ω·L/2,   v_R = v + ω·L/2
        #   ω_wheel = v_wheel / r,  r = wheel_diameter / 2
        wheel_radius = c.wheel_diameter / 2.0
        if wheel_radius <= 0.0:
            raise ValueError("wheel_diameter must be > 0")
        v_left = v - omega * c.wheel_base / 2.0
        v_right = v + omega * c.wheel_base / 2.0
        wheel_omega_left = float(np.clip(
            v_left / wheel_radius, -c.max_wheel_omega, c.max_wheel_omega))
        wheel_omega_right = float(np.clip(
            v_right / wheel_radius, -c.max_wheel_omega, c.max_wheel_omega))

        return {
            "v": v,
            "omega": omega,
            "wheel_omega_left": wheel_omega_left,
            "wheel_omega_right": wheel_omega_right,
            "wheel_v_left": v_left,
            "wheel_v_right": v_right,
            "distance": distance,
            "angle_error": angle_error,
            "reached": distance < c.goal_tolerance,
        }

    # ── 정역기구학 헬퍼 (캘리브레이션·테스트용) ──
    def wheel_omegas_from_twist(
        self, v: float, omega: float
    ) -> Dict[str, float]:
        """(v, ω) → 좌·우 바퀴 각속도 (한계 클립 없이)."""
        c = self.cfg
        r = c.wheel_diameter / 2.0
        v_left = v - omega * c.wheel_base / 2.0
        v_right = v + omega * c.wheel_base / 2.0
        return {
            "wheel_omega_left":  v_left / r,
            "wheel_omega_right": v_right / r,
        }

    def twist_from_wheel_omegas(
        self, wheel_omega_left: float, wheel_omega_right: float
    ) -> Dict[str, float]:
        """좌·우 바퀴 각속도 → (v, ω). 휠 odometry 등에 사용."""
        c = self.cfg
        r = c.wheel_diameter / 2.0
        v_left = wheel_omega_left * r
        v_right = wheel_omega_right * r
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / c.wheel_base
        return {"v": v, "omega": omega}

    # ── 내부 헬퍼 ──
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


# ──────────────────────────────────────────────
# CLI / Demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Differential-drive controller: pose + target → wheel ω")
    ap.add_argument("--x",  type=float, default=0.0, help="current x [m]")
    ap.add_argument("--y",  type=float, default=0.0, help="current y [m]")
    ap.add_argument("--th", type=float, default=0.0, help="current theta [rad]")
    ap.add_argument("--tx", type=float, default=3.0, help="target x [m]")
    ap.add_argument("--ty", type=float, default=2.0, help="target y [m]")
    ap.add_argument("--wheel_d",    type=float, default=0.10,
                    help="wheel diameter [m]")
    ap.add_argument("--wheel_base", type=float, default=0.30,
                    help="distance between wheels [m]")
    ap.add_argument("--conf", type=float, default=1.0,
                    help="SLAM confidence (0~1)")
    args = ap.parse_args()

    cfg = ControllerConfig(
        wheel_diameter=args.wheel_d,
        wheel_base=args.wheel_base,
    )
    ctrl = DrivingController(cfg)
    out = ctrl.compute(args.x, args.y, args.th, args.tx, args.ty,
                       slam_confidence=args.conf)

    print(f"distance      : {out['distance']:+.4f} m")
    print(f"angle_error   : {np.rad2deg(out['angle_error']):+8.3f} deg")
    print(f"v             : {out['v']:+.4f} m/s")
    print(f"omega         : {out['omega']:+.4f} rad/s")
    print(f"wheel v   L/R : {out['wheel_v_left']:+.4f} / "
          f"{out['wheel_v_right']:+.4f} m/s")
    print(f"wheel ω   L/R : {out['wheel_omega_left']:+.4f} / "
          f"{out['wheel_omega_right']:+.4f} rad/s")
    print(f"reached       : {out['reached']}")
