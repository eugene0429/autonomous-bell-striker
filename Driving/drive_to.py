"""
Phase-1 Driving Pipeline — standalone runner.

Reads pose from ORB-SLAM3 in real time, runs the existing DrivingController
to compute (ω_L, ω_R), and streams them to OpenRB-150 over the wheel-motor
serial protocol. Stops cleanly on goal-reach, timeout, SLAM failure, or Ctrl-C.

Spec: docs/superpowers/specs/2026-05-04-driving-pipeline-design.md

Usage
-----
    python Driving/drive_to.py --x 3 --y 2
    python Driving/drive_to.py --x 3 --y 2 --dry-run
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


# ──────────────────────────── safety ────────────────────────────
@dataclass
class SafetyConfig:
    lost_quiet_sec:    float = 0.5
    lost_warn_sec:     float = 45.0    # ORB-SLAM3 watchdog 가 한 번 respawn 할 시간
    # ABORT after lost_quiet_sec + lost_warn_sec total
    # (localizer 가 watchdog 으로 죽었다고 신고하면 즉시 ABORT — 아래 check() 참고)

    jump_factor:       float = 3.0
    jump_outlier_max:  int   = 3
    max_linear_vel:    float = 0.3     # m/s — matches ControllerConfig.max_speed

    warn_log_period:   float = 0.5     # min interval between warn logs


class SafetySupervisor:
    """Per-frame safety check. Returns "OK" | "HOLD" | "ABORT".

    Test seam: `now` defaults to time.monotonic; `log` defaults to print.
    Tests inject deterministic versions.
    """

    def __init__(
        self,
        cfg: Optional[SafetyConfig] = None,
        now: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ):
        self.cfg = cfg if cfg is not None else SafetyConfig()
        self._now = now
        self._log = log
        self.reason: str = ""

        # state
        self._last_ok: Optional[Tuple[float, float, float]] = None  # (x, y, t)
        self._lost_since: Optional[float] = None
        self._consec_outliers: int = 0
        self._last_warn_at: float = -1e9

    def check(self, pose: Optional[Dict], localizer_alive: bool = True) -> str:
        c = self.cfg
        t = self._now()

        # localizer 가 watchdog 한도까지 가서 죽었으면 즉시 ABORT.
        # 회복 가능성 없는데 lost_warn_sec 동안 모터 멈춘 채 기다릴 이유 없음.
        if not localizer_alive:
            self.reason = "localizer watchdog exhausted"
            return "ABORT"

        # Branch A: tracking lost or pose unavailable
        if pose is None or not pose.get("tracking_ok", False):
            if self._lost_since is None:
                # back-date to last known-good timestamp so the quiet/warn
                # window counts from the moment tracking was last confirmed
                last_t = self._last_ok[2] if self._last_ok is not None else t
                self._lost_since = last_t
            dur = t - self._lost_since
            if dur < c.lost_quiet_sec:
                return "HOLD"
            if dur < c.lost_quiet_sec + c.lost_warn_sec:
                if t - self._last_warn_at >= c.warn_log_period:
                    self._log(f"[WARN] tracking lost {dur:.1f}s")
                    self._last_warn_at = t
                return "HOLD"
            self.reason = (
                f"tracking lost {dur:.1f}s "
                f"(>= {c.lost_quiet_sec + c.lost_warn_sec:.1f}s)"
            )
            return "ABORT"

        # pose-jump rejection (only when we have a prior accepted pose)
        if self._last_ok is not None:
            ox, oy, ot = self._last_ok
            dt = t - ot
            if 0.0 < dt < 1.0:
                jump = math.hypot(float(pose["x"]) - ox, float(pose["y"]) - oy)
                threshold = c.max_linear_vel * dt * c.jump_factor
                if jump > threshold:
                    self._consec_outliers += 1
                    if self._consec_outliers >= c.jump_outlier_max:
                        self.reason = (
                            f"pose jump x{c.jump_outlier_max} "
                            f"(last={jump:.2f}m in {dt*1000:.0f}ms)"
                        )
                        return "ABORT"
                    return "HOLD"

        # accepted
        self._last_ok = (float(pose["x"]), float(pose["y"]), t)
        self._lost_since = None
        self._consec_outliers = 0
        return "OK"


# ──────────────────────────── runner ────────────────────────────
@dataclass
class RunArgs:
    x: float
    y: float
    rate: float = 15.0
    timeout: float = 60.0
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    dry_run: bool = False
    verbose: bool = False
    swap_lr: bool = False
    # vehicle geometry + success criterion (forwarded to ControllerConfig)
    wheel_diameter: float = 0.10   # [m]
    wheel_base: float = 0.30       # [m]
    goal_tolerance: float = 0.3    # [m] — distance < this counts as reached
    max_speed: float = 0.3         # [m/s] — controller v cap + safety jump threshold
    max_wheel_omega: float = 30.0  # [rad/s] — per-wheel angular velocity clip
    archive_slam: Optional[str] = None  # preserve SLAM tmp dirs at this path on death
    orb_nfeatures: Optional[int] = 1800  # ORB-SLAM3 nFeatures. base yaml=1500.
                                         # 2000 은 Pi5 CPU 한계로 frame queue overflow → 즉사.
                                         # 1800 이 init 안정성과 runtime margin 의 절충점.


def _log_status(t_elapsed, pose, out, log: Callable[[str], None]) -> None:
    log(
        f"  [{t_elapsed:5.2f}s]  pose=("
        f"{pose['x']:+.2f}, {pose['y']:+.2f}, "
        f"{math.degrees(pose['theta']):+6.1f}°)  "
        f"dist={out['distance']:.2f}  v={out['v']:.2f}  "
        f"ω_L/R=({out['wheel_omega_left']:+.2f}, "
        f"{out['wheel_omega_right']:+.2f})"
    )


def _run_loop(
    args: RunArgs,
    localizer,
    controller,
    motor,
    supervisor: SafetySupervisor,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """Main 15 Hz control loop. Returns a process exit code:
    0 = reached, 1 = timeout, 2 = supervisor ABORT.
    """
    period = 1.0 / args.rate
    t_start = now()
    deadline = t_start + args.timeout
    last_log_at = -1e9

    while now() < deadline:
        t0 = now()
        pose = localizer.get_pose()
        action = supervisor.check(pose, localizer_alive=localizer.is_alive())

        if action == "ABORT":
            log(f"[ABORT] {supervisor.reason}")
            motor.drive(0.0, 0.0)
            return 2

        if action == "HOLD":
            motor.drive(0.0, 0.0)
        else:  # OK
            out = controller.compute(
                pose["x"], pose["y"], pose["theta"], args.x, args.y)
            if out["reached"]:
                motor.drive(0.0, 0.0)
                log(f"✓ reached @ dist={out['distance']:.3f}m")
                return 0
            motor.drive(out["wheel_omega_left"], out["wheel_omega_right"])
            if t0 - last_log_at >= 0.5:
                _log_status(t0 - t_start, pose, out, log)
                last_log_at = t0

        elapsed = now() - t0
        sleep(max(0.0, period - elapsed))

    log(f"[TIMEOUT] {args.timeout:.1f}s elapsed, goal not reached")
    motor.drive(0.0, 0.0)
    return 1


# ───────────────────────────── main ─────────────────────────────
def main(argv: Optional[list] = None) -> int:
    import argparse
    from pathlib import Path as _P

    ap = argparse.ArgumentParser(
        description="Phase-1 driving runner: target (x,y) + ORB-SLAM3 → wheel ω → OpenRB")
    ap.add_argument("--x", type=float, required=True, help="target x [m, world frame]")
    ap.add_argument("--y", type=float, required=True, help="target y [m, world frame]")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=float, default=15.0,
                    help="control loop rate [Hz]")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="phase-1 timeout [s]")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip serial open; print DRIVE/STOP/PING lines")
    ap.add_argument("--verbose", action="store_true",
                    help="emit serial send/recv lines to stderr")
    ap.add_argument("--wheel-diameter", type=float, default=0.17,
                    help="wheel diameter [m] (controller v→ω conversion)")
    ap.add_argument("--wheel-base", type=float, default=0.27,
                    help="distance between wheels [m] (controller ω→ωL/ωR split)")
    ap.add_argument("--goal-tolerance", type=float, default=0.3,
                    help="distance to target [m] below which counts as reached")
    ap.add_argument("--max-speed", type=float, default=0.3,
                    help="max linear velocity [m/s] (also scales safety jump threshold)")
    ap.add_argument("--max-wheel-omega", type=float, default=30.0,
                    help="per-wheel angular velocity clip [rad/s]")
    ap.add_argument("--archive-slam", nargs="?", const="/tmp/orbslam_archive",
                    default=None, metavar="DIR",
                    help="preserve SLAM tmp dirs (stdout/stderr/yaml) on subprocess "
                         "death for forensic analysis. Default path: /tmp/orbslam_archive")
    ap.add_argument("--orb-nfeatures", type=int, default=1800,
                    help="ORB-SLAM3 ORBextractor.nFeatures (base yaml=1500). "
                         "Default 1800: init 안정성 ↑ 하면서 Pi5 CPU 마진 유지. "
                         "2000+ 은 frame queue overflow 위험.")
    ap.add_argument("--swap-lr", action="store_true",
                    help="swap L/R wheel commands before sending "
                         "(workaround for inverted firmware wiring)")
    a = ap.parse_args(argv)

    args = RunArgs(
        x=a.x, y=a.y, rate=a.rate, timeout=a.timeout,
        port=a.port, baud=a.baud, dry_run=a.dry_run, verbose=a.verbose,
        swap_lr=a.swap_lr,
        wheel_diameter=a.wheel_diameter, wheel_base=a.wheel_base,
        goal_tolerance=a.goal_tolerance, max_speed=a.max_speed,
        max_wheel_omega=a.max_wheel_omega,
        archive_slam=a.archive_slam,
        orb_nfeatures=a.orb_nfeatures,
    )

    # Repo layout: pipeline.py at the top level adds Driving/, perception/,
    # LevelingPlatform/ to sys.path. We replicate the same idea so
    # `from vio.orbslam_localizer ...`, `from controller ...`, and
    # `from wheel_motor ...` all resolve regardless of cwd.
    repo_root = _P(__file__).resolve().parents[1]
    for sub in ("Driving", "perception"):
        p = str(repo_root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    from controller import ControllerConfig, DrivingController  # noqa: E402
    from wheel_motor import WheelMotorClient, WheelMotorConfig  # noqa: E402
    from vio.orbslam_localizer import LocalizerConfig, OrbSlamLocalizer  # noqa: E402

    ctrl = DrivingController(ControllerConfig(
        dt=1.0 / args.rate,
        wheel_diameter=args.wheel_diameter,
        wheel_base=args.wheel_base,
        goal_tolerance=args.goal_tolerance,
        max_speed=args.max_speed,
        max_wheel_omega=args.max_wheel_omega,
    ))
    motor = WheelMotorClient(WheelMotorConfig(
        port=args.port, baud=args.baud,
        dry_run=args.dry_run, verbose=args.verbose,
        swap_lr=args.swap_lr,
    ))
    loc = OrbSlamLocalizer(LocalizerConfig(
        archive_dir=args.archive_slam,
        orb_nfeatures=args.orb_nfeatures,
    ))
    supervisor = SafetySupervisor(SafetyConfig(max_linear_vel=args.max_speed))

    print("=" * 70)
    print(f"  drive_to → target=({args.x:+.2f}, {args.y:+.2f}) "
          f"rate={args.rate}Hz timeout={args.timeout}s "
          f"{'(DRY-RUN)' if args.dry_run else ''}")
    print("=" * 70)

    with loc, motor:
        if not args.dry_run:
            if not motor.ping():
                print("[FAIL] OpenRB PING failed", file=sys.stderr)
                return 2
        if not loc.wait_for_tracking(timeout=60.0):
            print("[FAIL] SLAM did not reach STABLE tracking OK within 60s",
                  file=sys.stderr)
            return 2

        try:
            return _run_loop(args, loc, ctrl, motor, supervisor)
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl-C — stopping", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"[CRASH] {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        finally:
            try:
                motor.drive(0.0, 0.0)
            except Exception as e:
                print(f"[finally] zero-stop failed: {e}", file=sys.stderr)
            # synchronous STOP is sent by motor.disconnect() via __exit__


if __name__ == "__main__":
    sys.exit(main())
