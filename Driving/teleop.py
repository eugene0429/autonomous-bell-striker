"""
Teleop — keyboard arrow-key driver for the differential-drive rover.

Streams DRIVE packets to the OpenRB-150 at ~15 Hz (watchdog requires <200 ms
cadence). Twist state (v, ω) is held between ticks; keys nudge it. The control
loop runs continuously so the watchdog stays alive even when no key is pressed.

Controls
--------
    ↑ / ↓        v_target  ± v_step      (forward / backward)
    ← / →        ω_target  ± ω_step      (turn left / right)
    space        full stop (v = ω = 0)
    r            reset state to zero (same as space)
    +/-          scale v_step (×1.5 / ÷1.5)
    [ / ]        scale ω_step (÷1.5 / ×1.5)
    q  /  ESC    quit (sends STOP)

Usage
-----
    python Driving/teleop.py                      # real serial, default port
    python Driving/teleop.py --dry-run            # no serial, prints DRIVE lines
    python Driving/teleop.py --port /dev/ttyACM1
"""

from __future__ import annotations

import argparse
import curses
import sys
import time
from pathlib import Path

# Allow running both as `python Driving/teleop.py` and as a module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from controller import ControllerConfig, DrivingController
from wheel_motor import WheelMotorClient, WheelMotorConfig


CONTROL_HZ = 15.0
DT = 1.0 / CONTROL_HZ

V_STEP_INIT = 0.05      # m/s per key press
OMEGA_STEP_INIT = 0.2   # rad/s per key press
STEP_SCALE = 1.5
STEP_MIN = 1e-3
STEP_MAX = 5.0

ESC = 27


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _teleop_loop(stdscr, mc: WheelMotorClient, ctrl: DrivingController) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    cfg = ctrl.cfg
    v_target = 0.0
    omega_target = 0.0
    v_step = V_STEP_INIT
    omega_step = OMEGA_STEP_INIT
    last_key_repr = "—"

    next_tick = time.monotonic()

    while True:
        # ── drain all pending keys this tick (terminal auto-repeat sends bursts) ──
        quit_requested = False
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key in (ord('q'), ord('Q'), ESC):
                quit_requested = True
                last_key_repr = "QUIT"
                break
            elif key == curses.KEY_UP:
                v_target = _clamp(v_target + v_step, -cfg.max_speed, cfg.max_speed)
                last_key_repr = "↑"
            elif key == curses.KEY_DOWN:
                v_target = _clamp(v_target - v_step, -cfg.max_speed, cfg.max_speed)
                last_key_repr = "↓"
            elif key == curses.KEY_LEFT:
                omega_target = _clamp(
                    omega_target + omega_step, -cfg.max_omega, cfg.max_omega)
                last_key_repr = "←"
            elif key == curses.KEY_RIGHT:
                omega_target = _clamp(
                    omega_target - omega_step, -cfg.max_omega, cfg.max_omega)
                last_key_repr = "→"
            elif key in (ord(' '), ord('r'), ord('R')):
                v_target = 0.0
                omega_target = 0.0
                last_key_repr = "STOP"
            elif key in (ord('+'), ord('=')):
                v_step = _clamp(v_step * STEP_SCALE, STEP_MIN, STEP_MAX)
                last_key_repr = f"v_step×{STEP_SCALE}"
            elif key in (ord('-'), ord('_')):
                v_step = _clamp(v_step / STEP_SCALE, STEP_MIN, STEP_MAX)
                last_key_repr = f"v_step÷{STEP_SCALE}"
            elif key == ord(']'):
                omega_step = _clamp(omega_step * STEP_SCALE, STEP_MIN, STEP_MAX)
                last_key_repr = f"ω_step×{STEP_SCALE}"
            elif key == ord('['):
                omega_step = _clamp(omega_step / STEP_SCALE, STEP_MIN, STEP_MAX)
                last_key_repr = f"ω_step÷{STEP_SCALE}"

        if quit_requested:
            return

        # ── twist → wheel ω → DRIVE ──
        wheels = ctrl.wheel_omegas_from_twist(v_target, omega_target)
        wL = _clamp(wheels["wheel_omega_left"],
                    -cfg.max_wheel_omega, cfg.max_wheel_omega)
        wR = _clamp(wheels["wheel_omega_right"],
                    -cfg.max_wheel_omega, cfg.max_wheel_omega)
        mc.drive(wL, wR)

        # ── HUD ──
        stdscr.erase()
        stdscr.addstr(0, 0, "── Teleop — arrow keys to drive, space=stop, q=quit ──")
        stdscr.addstr(2, 2, f"v_target     : {v_target:+.3f} m/s   "
                            f"(step {v_step:.3f}, max {cfg.max_speed:.2f})")
        stdscr.addstr(3, 2, f"ω_target     : {omega_target:+.3f} rad/s "
                            f"(step {omega_step:.3f}, max {cfg.max_omega:.2f})")
        stdscr.addstr(5, 2, f"wheel ω L/R  : {wL:+.3f} / {wR:+.3f} rad/s")
        stdscr.addstr(7, 2, f"last key     : {last_key_repr}")
        stdscr.addstr(9, 2, "↑/↓ v   ←/→ ω   space=stop   +/- v_step   [ /] ω_step   q=quit")
        stdscr.refresh()

        # ── pace to 15 Hz ──
        next_tick += DT
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # We fell behind; resync rather than burning CPU catching up.
            next_tick = time.monotonic()


def main() -> int:
    ap = argparse.ArgumentParser(description="Keyboard teleop for the rover")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--wheel-d", type=float, default=0.10,
                    help="wheel diameter [m]")
    ap.add_argument("--wheel-base", type=float, default=0.30,
                    help="distance between wheels [m]")
    ap.add_argument("--max-speed", type=float, default=0.3,
                    help="max |v| [m/s]")
    ap.add_argument("--max-omega", type=float, default=1.0,
                    help="max |ω| [rad/s]")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't open serial; print would-be DRIVE lines instead")
    ap.add_argument("--verbose", action="store_true",
                    help="log serial traffic to stderr (noisy in curses; use --dry-run)")
    args = ap.parse_args()

    ctrl = DrivingController(ControllerConfig(
        wheel_diameter=args.wheel_d,
        wheel_base=args.wheel_base,
        max_speed=args.max_speed,
        max_omega=args.max_omega,
    ))
    mc_cfg = WheelMotorConfig(
        port=args.port,
        baud=args.baud,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    with WheelMotorClient(mc_cfg) as mc:
        try:
            curses.wrapper(_teleop_loop, mc, ctrl)
        except KeyboardInterrupt:
            pass
        # WheelMotorClient.__exit__ sends a final STOP.
    return 0


if __name__ == "__main__":
    sys.exit(main())
