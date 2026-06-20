"""Phase 1 driver Protocol + SLAM-based implementation.

Each driver owns the inner loop that converts target → wheel commands until
the rover reaches `goal_tolerance` of the target. Two implementations:
  - SlamPhase1Driver (this file)        — uses pose + DrivingController
  - VisualServoPhase1Driver (other file) — uses bbox/depth + VisualServoController

Both expose `.run() -> bool` (True = reached, False = failed/timeout) so
[pipeline.py] can pick one at startup via --drive-mode.
"""

from __future__ import annotations

import time
from typing import Protocol


class Phase1Driver(Protocol):
    def run(self) -> bool: ...


class SlamPhase1Driver:
    """Existing SLAM-based Phase 1 driver, extracted from pipeline.py."""

    def __init__(
        self,
        robot,
        target_provider,
        ctrl,
        dt: float = 0.067,
        timeout_s: float = 60.0,
    ):
        self.robot = robot
        self.target_provider = target_provider
        self.ctrl = ctrl
        self.dt = dt
        self.timeout_s = timeout_s

    def run(self) -> bool:
        target_xy = self.target_provider.get_phase1_target()
        print(f"\n── PHASE 1: DRIVING (slam) ──")
        print(f"  target (world) : ({target_xy[0]:+.2f}, {target_xy[1]:+.2f}) m")

        self.ctrl.reset()
        max_steps = int(self.timeout_s / self.dt)
        log_every = max(1, int(0.5 / self.dt))

        for step in range(max_steps):
            pose = self.robot.get_pose()
            if pose is None:
                time.sleep(self.dt)
                continue

            if not pose["tracking_ok"]:
                self.robot.send_wheel_omegas(0.0, 0.0, self.dt)
                if step % log_every == 0:
                    print(f"  [{step*self.dt:5.2f}s]  tracking={pose['tracking']} → STOP")
                time.sleep(self.dt)
                continue

            out = self.ctrl.compute(
                pose["x"], pose["y"], pose["theta"], target_xy[0], target_xy[1])

            self.robot.send_wheel_omegas(
                out["wheel_omega_left"], out["wheel_omega_right"], self.dt)

            if step % log_every == 0:
                print(f"  [{step*self.dt:5.2f}s]  pose=({pose['x']:+.2f}, "
                      f"{pose['y']:+.2f}, {pose['theta_deg']:+6.1f}°)  "
                      f"dist={out['distance']:.2f}  v={out['v']:.2f}  "
                      f"ω_L/R=({out['wheel_omega_left']:+.2f}, "
                      f"{out['wheel_omega_right']:+.2f})")

            if out["reached"]:
                self.robot.send_wheel_omegas(0.0, 0.0, self.dt)
                print(f"  ✓ reached @ t={step*self.dt:.2f}s  "
                      f"(final dist={out['distance']:.3f}m)")
                return True

            # Real robot loop pacing — sim runs as fast as possible
            from pipeline import RealRobot   # avoid circular at module load
            if isinstance(self.robot, RealRobot):
                time.sleep(self.dt)

        print(f"  ✗ timeout after {self.timeout_s:.0f}s")
        return False
