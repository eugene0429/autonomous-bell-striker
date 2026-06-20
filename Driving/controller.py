"""
Driving Controller — production module.

A differential-drive controller that takes the current pose (x, y, theta) +
target (target_x, target_y) and outputs left/right wheel angular velocities.

Vehicle model
-------------
- 2-wheel differential drive
- Mobile center coincides with the wheel axle (rotation center = base center)
- Parameters: wheel diameter (wheel_diameter), wheel spacing (wheel_base)

Control logic
-------------
- Distance-proportional linear velocity + slowdown radius + decelerate on large heading error
- PID angular velocity (based on heading error)
- (Optional) scale down speed/angular velocity when SLAM confidence is low

Usage example
-------------
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
    # ── Vehicle geometry (mobile center = wheel-axle center) ──
    wheel_diameter: float = 0.10   # Wheel diameter [m]
    wheel_base: float = 0.30       # Distance between left/right wheels [m]

    # ── Velocity/angular-velocity limits ──
    max_speed: float = 0.3         # Maximum linear velocity [m/s]
    max_omega: float = 1.0         # Maximum angular velocity [rad/s]
    max_wheel_omega: float = 30.0  # Wheel angular-velocity clip [rad/s]

    # ── Controller gains ──
    kp_linear: float = 0.8
    kp_angular: float = 1.5
    ki_angular: float = 0.03
    kd_angular: float = 0

    # ── Motion profile ──
    slowdown_radius: float = 1.0   # Decelerate linear velocity within this distance [m]
    goal_tolerance: float = 0.3    # Goal-reached decision radius [m]

    # ── Integral-term anti-windup ──
    integral_clip: float = 2.0

    # ── Control period (for PID derivative/integral) ──
    dt: float = 0.067              # Based on 15 Hz [s]

    # ── SLAM-confidence-based deceleration (optional) ──
    enable_confidence_scaling: bool = True
    lowconf_threshold: float = 0.8
    lowconf_speed_scale: float = 0.3   # Minimum scale applied when confidence is 0


# ──────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────
class DrivingController:
    """Current pose + target (x, y) → left/right wheel angular velocities."""

    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self._prev_angle_error = 0.0
        self._integral_angle_error = 0.0

    # ── State reset (called on phase transition) ──
    def reset(self) -> None:
        self._prev_angle_error = 0.0
        self._integral_angle_error = 0.0

    # ── Main API ──
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
        Compute the control command for one step.

        Parameters
        ----------
        x, y, theta       Current pose (world frame, theta in rad)
        target_x, target_y  Target coordinates (world frame, m)
        slam_confidence   0.0 ~ 1.0, decelerate when low (optional)

        Returns
        -------
        dict
            v                  : Linear velocity [m/s]
            omega              : Angular velocity [rad/s]
            wheel_omega_left   : Left wheel angular velocity [rad/s]
            wheel_omega_right  : Right wheel angular velocity [rad/s]
            wheel_v_left       : Left wheel tangential velocity [m/s]
            wheel_v_right      : Right wheel tangential velocity [m/s]
            distance           : Distance to target [m]
            angle_error        : Heading error [rad, in (-pi, pi]]
            reached            : Whether reached (distance < goal_tolerance)
        """
        c = self.cfg

        # ── 1) Distance / heading error ──
        dx = target_x - x
        dy = target_y - y
        distance = float(np.hypot(dx, dy))
        desired_heading = float(np.arctan2(dy, dx))
        angle_error = self._wrap_angle(desired_heading - theta)

        # ── 2) PID angular velocity ──
        self._integral_angle_error += angle_error * c.dt
        self._integral_angle_error = float(np.clip(
            self._integral_angle_error, -c.integral_clip, c.integral_clip))
        derivative = (angle_error - self._prev_angle_error) / c.dt
        self._prev_angle_error = angle_error

        omega = (c.kp_angular * angle_error
                 + c.ki_angular * self._integral_angle_error
                 + c.kd_angular * derivative)
        omega = float(np.clip(omega, -c.max_omega, c.max_omega))

        # ── 3) Linear velocity (distance-proportional + slowdown radius + decelerate on heading error) ──
        speed_factor = min(1.0, distance / max(c.slowdown_radius, 1e-9))
        heading_factor = max(0.2, 1.0 - abs(angle_error) / np.pi)
        v = c.kp_linear * distance * speed_factor * heading_factor
        v = float(np.clip(v, 0.0, c.max_speed))

        # ── 4) SLAM-confidence-based deceleration ──
        if c.enable_confidence_scaling and slam_confidence < c.lowconf_threshold:
            ratio = max(0.0, slam_confidence) / max(c.lowconf_threshold, 1e-9)
            conf_scale = c.lowconf_speed_scale + (1.0 - c.lowconf_speed_scale) * ratio
            v *= conf_scale
            omega *= conf_scale

        # ── 5) Differential-drive inverse kinematics: (v, ω) → (v_L, v_R) → (ω_L, ω_R) ──
        # Assuming mobile center = wheel-axle center:
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

    # ── Forward/inverse kinematics helpers (for calibration/testing) ──
    def wheel_omegas_from_twist(
        self, v: float, omega: float
    ) -> Dict[str, float]:
        """(v, ω) → left/right wheel angular velocities (without limit clipping)."""
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
        """Left/right wheel angular velocities → (v, ω). Used for wheel odometry, etc."""
        c = self.cfg
        r = c.wheel_diameter / 2.0
        v_left = wheel_omega_left * r
        v_right = wheel_omega_right * r
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / c.wheel_base
        return {"v": v, "omega": omega}

    # ── Internal helper ──
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
