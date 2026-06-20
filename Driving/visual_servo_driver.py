"""Visual-servo Phase 1 driver — loops the VisualServoController against a Robot.

Each tick:
  - acquire detection (sim bypass or real YOLO)
  - read current tilt from Robot
  - controller.step(detection, tilt) → wheel ω + tilt_cmd + state
  - dispatch wheel ω and tilt setpoint
  - terminate on DONE (True) / FAIL or timeout (False)
"""

from __future__ import annotations

import time

from Driving.visual_servo_controller import VisualServoController


def _step_from_deg(deg: float, steps_per_deg: float = 1024.0 / 90.0) -> int:
    return int(round(deg * steps_per_deg))


class VisualServoPhase1Driver:
    def __init__(
        self,
        robot,
        target_provider,
        ctrl: VisualServoController,
        dt: float = 0.067,
        timeout_s: float = 60.0,
        steps_per_deg: float = 1024.0 / 90.0,
        log_every_s: float = 0.5,
        bootstrap_creep_v: float = 0.15,
        bootstrap_creep_s: float = 2.0,
        bootstrap_creep_retries: int = 2,
    ):
        self.robot = robot
        self.target_provider = target_provider
        self.ctrl = ctrl
        self.dt = dt
        self.timeout_s = timeout_s
        self.steps_per_deg = steps_per_deg
        # control-loop log cadence [s]. 0 → log every tick (see the real per-tick
        # ω, which the default 0.5s cadence hides — masking the overshoot oscillation).
        self.log_every_s = log_every_s
        # Bootstrap retry policy: if the tilt sweep finds nothing, drive straight
        # forward (no rotation) for bootstrap_creep_s seconds at bootstrap_creep_v
        # m/s, then re-sweep. Repeated up to bootstrap_creep_retries times.
        # Premise: the robot is always placed facing the bell, so creeping closes
        # distance (target grows in frame / depth improves) rather than rotating
        # away from it.
        self.bootstrap_creep_v = bootstrap_creep_v
        self.bootstrap_creep_s = bootstrap_creep_s
        self.bootstrap_creep_retries = bootstrap_creep_retries
        # Exposed so run() can detect a fallback (no target) sweep result without
        # changing acquire_initial_tilt's float return type (tests depend on it).
        self._last_acquire_found: bool = False

    def zero_tilt(self, settle_s: float = 0.5) -> None:
        """Home the mast to 0° via sync TILT (motion-complete) at Phase 1 start.

        TILT is absolute, but RealRobot starts with believed tilt = 0.0 while the
        physical mast may be left anywhere by a prior run / Phase 2. Servoing from
        a mismatched belief makes TILT_ASYNC slam the mast and the err_y loop wind
        up (see VisualServoConfig.tilt_max_rate_dps). Driving to a known 0° here
        syncs belief ≈ actual before the loop begins.
        """
        print("\n── PHASE 1.0a: TILT ZERO (→ 0°) ──")
        self.robot.tilt_camera(0.0)
        if type(self.robot).__name__ == "RealRobot":
            time.sleep(settle_s)
        print(f"  ✓ tilt zeroed (believed={self.robot.get_tilt_deg():.1f}°)")

    def acquire_initial_tilt(
        self,
        start_deg: float = 0.0,
        end_deg: float = 90.0,
        step_deg: float = 5.0,
        settle_s: float = 0.1,
        fallback_deg: float = 45.0,
    ) -> float:
        """Sweep camera tilt from start_deg→end_deg looking for the target.

        Each sweep step: send tilt_camera(deg), (real only) sleep settle_s,
        call get_visual_servo_detection(). Stop on first non-None detection.

        Returns the tilt at which detection succeeded, OR fallback_deg if
        the full sweep yielded no detection (in which case the robot is
        re-tilted to fallback_deg before return). The driver's main loop
        then runs from there; FSM SEARCH handles horizontal recovery if
        the target was horizontally out of FOV during the sweep.
        """
        print(f"\n── PHASE 1.0: TILT SWEEP "
              f"({start_deg:.0f}°→{end_deg:.0f}°, step {step_deg:.0f}°) ──")
        is_real = type(self.robot).__name__ == "RealRobot"
        self._last_acquire_found = False

        deg = start_deg
        while deg <= end_deg + 1e-9:
            self.robot.tilt_camera(deg)
            if is_real:
                time.sleep(settle_s)
            detection = self.robot.get_visual_servo_detection()
            if detection is not None:
                print(f"  ✓ target acquired @ tilt={deg:.1f}°")
                self._last_acquire_found = True
                return deg
            deg += step_deg

        # No detection across the sweep — fall back so FSM SEARCH can recover.
        print(f"  ✗ no detection in sweep — fallback tilt={fallback_deg:.1f}°")
        self.robot.tilt_camera(fallback_deg)
        return fallback_deg

    def creep_forward(self, v: float, duration_s: float) -> None:
        """Drive straight forward at v [m/s] for duration_s seconds, no rotation.

        Used as a bootstrap fallback when the tilt sweep finds nothing. The
        deployment premise is that the robot starts facing the bell, so the right
        recovery is to close distance (target grows in frame, depth improves)
        rather than spin (which would lose the bell).
        """
        cfg = self.ctrl.cfg
        r = cfg.wheel_diameter / 2.0
        w = v / r  # equal wheel ω → straight line
        n_ticks = max(1, int(round(duration_s / self.dt)))
        print(f"  ⟶ creep forward {duration_s:.1f}s @ {v:.2f} m/s ({n_ticks} ticks)")
        is_real = type(self.robot).__name__ == "RealRobot"
        loop_start = time.monotonic()
        for i in range(n_ticks):
            self.robot.send_wheel_omegas(w, w, self.dt)
            if is_real:
                next_tick = loop_start + (i + 1) * self.dt
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        self.robot.send_wheel_omegas(0.0, 0.0, self.dt)  # stop after creep

    def run(self) -> bool:
        print(f"\n── PHASE 1: DRIVING (visual_servo) ──")
        # Ensure sim robot knows where the dummy target is
        if hasattr(self.robot, "set_visual_servo_target_provider"):
            self.robot.set_visual_servo_target_provider(self.target_provider)

        # Always home the mast to 0° first so believed tilt ≈ physical tilt.
        self.zero_tilt()

        # Bootstrap: sweep tilt, and if nothing found, creep forward and re-sweep.
        # Premise: robot starts facing the bell — closing distance is the right
        # recovery, not rotation.
        for attempt in range(self.bootstrap_creep_retries + 1):
            self.acquire_initial_tilt()
            if self._last_acquire_found:
                break
            if attempt < self.bootstrap_creep_retries:
                print(f"  ↻ retry {attempt + 1}/{self.bootstrap_creep_retries}: "
                      f"creep forward then re-sweep")
                self.creep_forward(self.bootstrap_creep_v, self.bootstrap_creep_s)

        self.ctrl.reset()
        max_steps = int(self.timeout_s / self.dt)
        log_every = max(1, int(self.log_every_s / self.dt)) if self.log_every_s > 0 else 1
        is_real = type(self.robot).__name__ == "RealRobot"
        loop_start = time.monotonic()

        for step in range(max_steps):
            detection = self.robot.get_visual_servo_detection()
            tilt_cur = self.robot.get_tilt_deg()
            out = self.ctrl.step(detection, tilt_cur)

            # send tilt setpoint
            self.robot.send_tilt_async(
                _step_from_deg(out["tilt_cmd_deg"], self.steps_per_deg))

            # send wheel command
            self.robot.send_wheel_omegas(
                out["wheel_omega_left"], out["wheel_omega_right"], self.dt)

            if step % log_every == 0:
                print(
                    f"  [{step*self.dt:5.2f}s] state={out['state']:<6} "
                    f"err_x={out['err_x_px']:+6.1f}px  "
                    f"tilt_cmd={out['tilt_cmd_deg']:5.1f}°  "
                    f"horiz={out['horiz_dist']:.2f}m  "
                    f"v={out['v']:.2f}  ω={out['omega']:+.2f}"
                )

            if out["reached"]:
                self.robot.send_wheel_omegas(0.0, 0.0, self.dt)
                print(f"  ✓ reached @ t={step*self.dt:.2f}s  "
                      f"(tilt={out['tilt_cmd_deg']:.1f}°, "
                      f"horiz={out['horiz_dist']:.2f}m)")
                return True

            if out["failed"]:
                self.robot.send_wheel_omegas(0.0, 0.0, self.dt)
                print(f"  ✗ FAIL @ t={step*self.dt:.2f}s  (search timeout)")
                return False

            if is_real:
                # Anchored rate-limit: the next tick is loop_start + (step+1)*dt.
                # If YOLO inference runs long and overruns the deadline, skip the
                # sleep and proceed best-effort.
                next_tick = loop_start + (step + 1) * self.dt
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        print(f"  ✗ timeout after {self.timeout_s:.0f}s")
        return False
