"""
Leveling motor diagnostic — compare STATUS before and after an AIM command.

Close the GUI and run standalone:
    python LevelingPlatform/diag_motor.py --port /dev/cu.usbmodem11301

At each step it prints the actual motor position to verify "even when AIM
responds OK, did the motor actually move?".

Interpretation
----
- Target step equals the STATUS step → motor reached normally
- STATUS step unchanged from the start position → no motor torque / HW error / stalled
- STATUS step at an intermediate position → partial motion, insufficient time
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from leveling_motor import (LevelingMotorClient, MotorClientConfig,        # noqa: E402
                            MotorProtocolError, MotorTimeoutError)


def parse_status(line: str):
    """`S wL wR s1 s2 s3 s4 s5 rpmT rpmB flags` → dict"""
    parts = line.split()
    if len(parts) != 11 or parts[0] != "S":
        return None
    try:
        return {
            "wL": int(parts[1]), "wR": int(parts[2]),
            "lvl": (int(parts[3]), int(parts[4]), int(parts[5])),
            "tilt": int(parts[6]), "load": int(parts[7]),
            "rpm_top": int(parts[8]), "rpm_bot": int(parts[9]),
            "flags": int(parts[10]),
        }
    except ValueError:
        return None


def fmt_flags(f: int) -> str:
    bits = []
    if f & (1 << 0): bits.append("watchdog")
    if f & (1 << 1): bits.append("lvl_moving")
    if f & (1 << 2): bits.append("tilt_moving")
    if f & (1 << 3): bits.append("load_moving")
    if f & (1 << 4): bits.append("tmotor")
    if f & (1 << 5): bits.append("ERR_LATCHED")
    if f & (1 << 6): bits.append("homed")
    if f & (1 << 7): bits.append("ESTOP")
    return ", ".join(bits) if bits else "(none)"


def query_status(mc: LevelingMotorClient):
    resp = mc._command("STATUS")
    st = parse_status(resp)
    if st is None:
        print(f"  ! STATUS parse fail: {resp!r}")
        return None
    print(f"  STATUS lvl={st['lvl']} tilt={st['tilt']} load={st['load']} "
          f"flags=0b{st['flags']:08b} [{fmt_flags(st['flags'])}]")
    return st


def _report_motion(st0, st1, target_steps):
    if st0 is None or st1 is None:
        return
    d = tuple(b - a for a, b in zip(st0["lvl"], st1["lvl"]))
    target_delta = tuple(t - s for t, s in zip(target_steps, st0["lvl"]))
    print(f"  Δlvl   = {d}    (target Δ = {target_delta})")
    ratios = []
    for actual, intended in zip(d, target_delta):
        if abs(intended) < 5:
            ratios.append("—")
        else:
            ratios.append(f"{actual / intended * 100:+.0f}%")
    print(f"  achieved: {ratios[0]}, {ratios[1]}, {ratios[2]}")


def try_aim(mc, label, s1, s2, s3):
    print(f"\n── {label}: AIM {s1} {s2} {s3} ──")
    print("  [before]"); st0 = query_status(mc)
    t0 = time.monotonic()
    try:
        resp = mc._command(f"AIM {s1} {s2} {s3}")
        dt = time.monotonic() - t0
        print(f"  response: {resp!r}  ({dt*1000:.0f} ms)")
    except MotorProtocolError as e:
        dt = time.monotonic() - t0
        print(f"  ERROR: {e}  ({dt*1000:.0f} ms)")
    time.sleep(0.5)
    print("  [after] "); st1 = query_status(mc)
    _report_motion(st0, st1, (s1, s2, s3))


def try_aimf(mc, label, s1, s2, s3, settle_ms=600):
    """Single AIMF — send once, wait long enough, then STATUS."""
    print(f"\n── {label}: AIMF {s1} {s2} {s3} (single, wait {settle_ms}ms) ──")
    print("  [before]"); st0 = query_status(mc)
    t0 = time.monotonic()
    try:
        resp = mc._command(f"AIMF {s1} {s2} {s3}")
        dt = time.monotonic() - t0
        print(f"  AIMF response: {resp!r}  ({dt*1000:.0f} ms)")
    except MotorProtocolError as e:
        dt = time.monotonic() - t0
        print(f"  AIMF ERROR: {e}  ({dt*1000:.0f} ms)")
    time.sleep(settle_ms / 1000.0)
    print("  [after] "); st1 = query_status(mc)
    _report_motion(st0, st1, (s1, s2, s3))


def try_aimf_stream(mc, label, waypoints, period_ms=80, settle_ms=800):
    """Continuous AIMF stream — same pattern as GUI dragging."""
    print(f"\n── {label}: AIMF stream "
          f"({len(waypoints)} pts @ {period_ms}ms) ──")
    print("  [before]"); st0 = query_status(mc)
    for i, (s1, s2, s3) in enumerate(waypoints):
        try:
            resp = mc._command(f"AIMF {s1} {s2} {s3}")
            if resp != "OK":
                print(f"  AIMF#{i} {(s1,s2,s3)} -> {resp!r}")
        except MotorProtocolError as e:
            print(f"  AIMF#{i} ERROR: {e}")
        time.sleep(period_ms / 1000.0)
    print(f"  stream done, settling {settle_ms}ms ...")
    time.sleep(settle_ms / 1000.0)
    print("  [after] "); st1 = query_status(mc)
    final_target = waypoints[-1]
    _report_motion(st0, st1, final_target)


def main():
    ap = argparse.ArgumentParser(description="Leveling motor physical-motion diagnostic")
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-home", action="store_true",
                    help="skip HOME (diagnose from current position)")
    args = ap.parse_args()

    cfg = MotorClientConfig(port=args.port, baud=args.baud, verbose=False)
    mc = LevelingMotorClient(cfg)
    mc.connect()
    try:
        print("=" * 60)
        print(f"Connected to {args.port}")
        print("=" * 60)

        if not mc.ping():
            print("PING failed")
            return 2
        print("PING ok")

        print("\n[setup state]"); query_status(mc)

        if not args.no_home:
            print("\n── HOME ──")
            try:
                t0 = time.monotonic()
                mc.home()
                print(f"  HOME OK ({(time.monotonic()-t0)*1000:.0f} ms)")
            except (MotorProtocolError, MotorTimeoutError) as e:
                print(f"  HOME failed: {e}")
                return 2
            print("  [after HOME]"); query_status(mc)

        # ── Phase A: AIM (blocking) ───────────────────────────────
        print("\n" + "▼" * 30 + "  Phase A: AIM (blocking)  " + "▼" * 30)
        try_aim(mc, "AIM small",     50,   50,  50)
        try_aim(mc, "AIM medium",   200,  200, 200)
        try_aim(mc, "AIM asymmetric", 200, -300, 150)
        try_aim(mc, "AIM back to 0",   0,    0,   0)

        # ── Phase B: single AIMF (should give the same result as blocking) ──────
        print("\n" + "▼" * 30 + "  Phase B: AIMF (single)  " + "▼" * 30)
        try_aimf(mc, "AIMF single small",  100, 100, 100)
        try_aimf(mc, "AIMF single medium", 250, -200, 150)
        try_aimf(mc, "AIMF back to 0",       0,    0,   0)

        # ── Phase C: AIMF stream (mimics GUI dragging) ──────────────
        # Small step changes (pattern similar to GUI: 70→13→-50→-30→78→148)
        stream_small = [
            ( 70, -84,  14),
            ( 67, -89,  22),
            ( 37, -94,  57),
            ( 13, -96,  82),
            (-10, -91, 101),
            (-38, -83, 122),
            (-53, -82, 135),
            (-32, -88, 120),
            ( 23, -93,  70),
            ( 78, -86,   8),
            (118, -65, -53),
            (148, -30,-118),
        ]
        print("\n" + "▼" * 30 + "  Phase C: AIMF stream  " + "▼" * 30)
        try_aimf_stream(mc, "AIMF small drag (80ms)",  stream_small, period_ms=80)
        try_aimf_stream(mc, "AIMF fast drag (30ms)",   stream_small, period_ms=30)

        # ── Phase D: AIMF stream with large steps (so motion is clearly visible) ────────
        stream_big = [
            (  0,    0,    0),
            (200,  200,  200),
            (200, -300,  150),
            (-200, 200, -200),
            (  0,    0,    0),
        ]
        try_aimf_stream(mc, "AIMF big drag (80ms)", stream_big, period_ms=80)

        print("\n" + "=" * 60)
        print("Diagnostic complete.")
        print("Interpretation:")
        print("  Phase A (AIM)  : 100% achieved means the motor is normal")
        print("  Phase B (AIMF single) : should match AIM. If different, the AIMF handler itself is the problem")
        print("  Phase C (AIMF stream) : should have reached the final target.")
        print("       If Δ is small or 0, the motor stalls in stream mode")
        print("=" * 60)
        return 0
    finally:
        try:
            mc.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
