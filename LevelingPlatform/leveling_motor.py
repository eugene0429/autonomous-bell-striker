"""
Leveling Platform — Pi5 ↔ OpenRB-150 serial client.

Client that sends the output (`angles_steps`) of [LevelingIK.aim_at()](leveling_ik.py)
to the OpenRB-150 to move 3 Dynamixel (or equivalent) motors to absolute positions.

═══════════════════════════════════════════════════════════════════════
  Wire Protocol — ASCII / line-oriented / '\n'-terminated
═══════════════════════════════════════════════════════════════════════

Transport
---------
- USB CDC virtual COM (Linux: `/dev/ttyACM0`, etc.)
- Baud   : 115200  (stable and well-supported; can be raised up to 1 Mbps if needed)
- 8N1, no flow control
- Encoding: 7-bit ASCII
- Line terminator: LF only ('\n' / 0x0A) — CR ignored

Command (Pi → OpenRB)
---------------------
    PING\n
        keep-alive / health check.

    AIM <s1> <s2> <s3>\n
        Move motors 1/2/3 simultaneously to absolute encoder step positions.
        s_i : signed decimal integer.
              For MOTOR_STEPS=4096, [-2048..+2047] is typical.
              The firmware returns ERR OUT_OF_RANGE on a limit violation.
        OpenRB returns 'OK'/'ERR' after all motors complete motion (or timeout/abort).

    HOME\n
        Move all motors to step=0.

    STATUS\n
        Query current position/flags (non-blocking, immediate response).

    STOP\n
        Stop all motors immediately (cancel the current command).

Response (OpenRB → Pi)
----------------------
    OK\n
        Command processed. For AIM/HOME, returns after blocking until motion completes.
    ERR <reason>\n
        Failure. reason e.g.: PARSE / OUT_OF_RANGE / TIMEOUT / NOT_HOMED / BUSY / HW
    PONG\n
        Response to PING.
    S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>\n
        Response to STATUS. (11-field format of firmware v1.2+)
            wL/wR        : wheel ID 6/7 measured speed [mrad/s]
            s1/s2/s3     : leveling LVL_1/2/3 step
            s4           : camera tilt ID 4 step
            s5           : loader ID 5 step
            rpmT/rpmB    : T-motor TOP/BOTTOM current RPM command value
            flags        : bit0=wheel watchdog tripped
                           bit1=leveling moving
                           bit2=tilt moving
                           bit3=loader moving
                           bit4=tmotor active (>100 RPM)
                           bit5=error latched
                           bit6=homed
                           bit7=estop

Synchronization rules
---------------------
- Synchronous request-response. The Pi does not send the next command before
  receiving an OK/ERR/PONG/S line.
- OpenRB processes only one command at a time. Additional commands are not queued
  but rejected with BUSY.
- AIM/HOME withhold their response until motion completes (the simplest sync model).
- On motor stall / timeout detection, automatically STOP and return ERR.

Frame size & timing
-------------------
- Longest typical command: "AIM -2048 -2048 -2048\n" = 23 bytes ≈ 2 ms @ 115200
- Longest response:        "S -30000 30000 -2048 -2048 -2048 -2048 -2048 10000 10000 255\n"
                     ≈ 60 bytes ≈ 5 ms (STATUS, v1.2+ 11-field format)
- One cycle including motion latency ≤ 100 ms is typical (called once just before
  each strike in Phase 2)

═══════════════════════════════════════════════════════════════════════
  Python API
═══════════════════════════════════════════════════════════════════════

    from leveling_ik    import LevelingIK, LevelingConfig
    from leveling_motor import LevelingMotorClient, MotorClientConfig

    ik = LevelingIK(LevelingConfig())

    with LevelingMotorClient(MotorClientConfig(port="/dev/ttyACM0")) as mc:
        mc.ping()                                    # health check
        mc.home()                                    # home at startup
        out = ik.aim_at((0.10, 0.00, 3.0))
        if not out["ok"]:
            raise RuntimeError("IK unreachable")
        mc.aim(out)                                  # move motors + wait for OK
        print(mc.status())  # {wheel_mrad, leveling_steps, tilt_step,
                            #  loader_step, tmotor_rpm, flags,
                            #  watchdog, leveling_moving, tilt_moving,
                            #  loader_moving, tmotor_active, error,
                            #  homed, estop, moving}
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class MotorClientConfig:
    # ── Transport ──
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    read_timeout_sec: float = 5.0           # includes waiting for AIM/HOME motion to complete
    write_timeout_sec: float = 1.0
    open_settle_sec: float = 2.0            # wait for OpenRB to stabilize after port open

    # ── Rotation direction / zero-point calibration ──
    # Applied to the angles_steps from IK: cmd_steps[i] = sign[i] * angles_steps[i] + offset[i]
    direction_signs: Tuple[int, int, int] = (+1, +1, +1)
    home_offsets_steps: Tuple[int, int, int] = (0, 0, 0)

    # ── Safety limits (the firmware cross-checks, but the client also pre-checks) ──
    motor_min_step: int = -2048
    motor_max_step: int = +2047

    # ── Debug ──
    verbose: bool = False                   # print TX/RX lines to stderr
    dry_run: bool = False                   # if True, do not send over serial, just print


# ──────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────
class MotorProtocolError(RuntimeError):
    """The OpenRB firmware returned an ERR response or the response format is malformed."""


class MotorTimeoutError(RuntimeError):
    """No response arrived within the timeout."""


# ──────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────
class LevelingMotorClient:
    """Send 3-RRS leveling motor commands to the OpenRB-150."""

    def __init__(self, cfg: Optional[MotorClientConfig] = None):
        self.cfg = cfg if cfg is not None else MotorClientConfig()
        self._ser = None  # pyserial.Serial
        self._last_cmd: str = ""  # for synthesizing dry-run responses

    # ── lifecycle ──
    def connect(self) -> None:
        if self.cfg.dry_run:
            self._log("[dry-run] skip serial open")
            return
        import serial  # lazy import — even environments without pyserial can import under dry_run
        self._ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baud,
            timeout=self.cfg.read_timeout_sec,
            write_timeout=self.cfg.write_timeout_sec,
        )
        # Wait for the OpenRB to stabilize after USB-CDC reset + flush buffers
        time.sleep(self.cfg.open_settle_sec)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._log(f"[open] {self.cfg.port} @ {self.cfg.baud}")

    def disconnect(self) -> None:
        if self._ser is None:
            return
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass
        self._ser = None
        self._log("[close]")

    def __enter__(self) -> "LevelingMotorClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ── public API ──
    def ping(self) -> bool:
        return self._command("PING") == "PONG"

    def aim(self, ik_result: Dict) -> bool:
        """Take the LevelingIK.aim_at() output dict and send it as an AIM command.

        Raises
        ------
        ValueError              When IK is unreachable (angles_steps is None) or
                                a step value is out of limits.
        MotorProtocolError      OpenRB returned an ERR response.
        MotorTimeoutError       No response arrived within the timeout.
        """
        steps = ik_result.get("angles_steps")
        if steps is None or len(steps) != 3:
            raise ValueError("ik_result['angles_steps'] is missing/invalid "
                             "(IK probably unreachable)")
        cmd_steps = self._apply_calibration(steps)
        for i, s in enumerate(cmd_steps):
            if not (self.cfg.motor_min_step <= s <= self.cfg.motor_max_step):
                raise ValueError(
                    f"motor {i+1} step {s} out of range "
                    f"[{self.cfg.motor_min_step}, {self.cfg.motor_max_step}]")
        return self._command(
            f"AIM {cmd_steps[0]} {cmd_steps[1]} {cmd_steps[2]}") == "OK"

    def aim_fast(self, ik_result: Dict) -> bool:
        """Streaming version of AIM — the firmware only does syncWrite and returns OK immediately.

        Use case
        --------
        When the next command must be sent right away without waiting for motion
        to complete, as in GUI dragging / continuous tracking. The meaning of the
        response differs from AIM:
            AIM       OK = motor reached the target (within tolerance)
            aim_fast  OK = command was received and dispatched (may still be moving)

        For sequences that need a "reach guarantee" (e.g. aim then fire), use AIM.
        """
        steps = ik_result.get("angles_steps")
        if steps is None or len(steps) != 3:
            raise ValueError("ik_result['angles_steps'] is missing/invalid "
                             "(IK probably unreachable)")
        cmd_steps = self._apply_calibration(steps)
        for i, s in enumerate(cmd_steps):
            if not (self.cfg.motor_min_step <= s <= self.cfg.motor_max_step):
                raise ValueError(
                    f"motor {i+1} step {s} out of range "
                    f"[{self.cfg.motor_min_step}, {self.cfg.motor_max_step}]")
        return self._command(
            f"AIMF {cmd_steps[0]} {cmd_steps[1]} {cmd_steps[2]}") == "OK"

    def home(self) -> bool:
        return self._command("HOME") == "OK"

    def stop(self) -> bool:
        return self._command("STOP") == "OK"

    def status(self) -> Dict:
        """Parse the STATUS command response into a dict.

        Firmware response format (v1.2+):
            S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>
        """
        resp = self._command("STATUS")
        parts = resp.split()
        if len(parts) != 11 or parts[0] != "S":
            raise MotorProtocolError(f"bad STATUS reply: {resp!r}")
        try:
            wL, wR = int(parts[1]), int(parts[2])
            leveling = (int(parts[3]), int(parts[4]), int(parts[5]))
            tilt_step = int(parts[6])
            loader_step = int(parts[7])
            rpm_top, rpm_bot = int(parts[8]), int(parts[9])
            flags = int(parts[10])
        except ValueError as e:
            raise MotorProtocolError(f"bad STATUS numbers: {resp!r}") from e
        return {
            "wheel_mrad":      (wL, wR),
            "leveling_steps":  leveling,
            "tilt_step":       tilt_step,
            "loader_step":     loader_step,
            "tmotor_rpm":      (rpm_top, rpm_bot),
            "flags":           flags,
            "watchdog":        bool(flags & (1 << 0)),
            "leveling_moving": bool(flags & (1 << 1)),
            "tilt_moving":     bool(flags & (1 << 2)),
            "loader_moving":   bool(flags & (1 << 3)),
            "tmotor_active":   bool(flags & (1 << 4)),
            "error":           bool(flags & (1 << 5)),
            "homed":           bool(flags & (1 << 6)),
            "estop":           bool(flags & (1 << 7)),
            "moving":          bool(flags & 0b00001110),  # leveling|tilt|loader
        }

    # ── Internal ──
    def _apply_calibration(self, ik_steps) -> List[int]:
        c = self.cfg
        return [int(c.direction_signs[i] * ik_steps[i] + c.home_offsets_steps[i])
                for i in range(3)]

    def _command(self, cmd: str) -> str:
        """Send a one-line command and return a one-line response. ERR responses are converted to exceptions."""
        self._send_line(cmd)
        resp = self._recv_line()
        if resp.startswith("ERR"):
            raise MotorProtocolError(f"controller refused {cmd!r}: {resp}")
        return resp

    def _send_line(self, line: str) -> None:
        self._last_cmd = line
        payload = (line + "\n").encode("ascii")
        self._log(f"→ {line}")
        if self.cfg.dry_run:
            return
        if self._ser is None:
            raise RuntimeError("not connected (call connect() or use context manager)")
        self._ser.write(payload)
        self._ser.flush()

    def _recv_line(self) -> str:
        if self.cfg.dry_run:
            fake = self._dry_run_response(self._last_cmd)
            self._log(f"← (dry-run) {fake}")
            return fake
        if self._ser is None:
            raise RuntimeError("not connected")
        raw = self._ser.readline()       # blocks up to read_timeout_sec
        if not raw:
            raise MotorTimeoutError(
                f"no response within {self.cfg.read_timeout_sec}s")
        line = raw.decode("ascii", errors="replace").rstrip("\r\n")
        self._log(f"← {line}")
        return line

    @staticmethod
    def _dry_run_response(cmd: str) -> str:
        """A fake response conforming to the protocol spec — for dry-run/unit tests."""
        if cmd == "PING":
            return "PONG"
        if cmd == "STATUS":
            # Assume motion finished + homed state (flags = 0b10000)
            return "S 0 0 0 16"
        return "OK"  # AIM / HOME / STOP

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[motor] {msg}", file=sys.stderr)


# ──────────────────────────────────────────────
# CLI — integrated IK + serial send single-shot (for bench verification)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from leveling_ik import LevelingConfig, LevelingIK

    ap = argparse.ArgumentParser(
        description="Leveling Platform: IK → OpenRB serial single-shot")
    ap.add_argument("--target", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "Z"),
                    help="target 3D point in plate-base frame [m]")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--home", action="store_true",
                    help="send HOME before AIM")
    ap.add_argument("--dry-run", action="store_true",
                    help="no serial connection, print command lines only")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ik = LevelingIK(LevelingConfig())
    out = ik.aim_at(tuple(args.target))
    if out["angles_deg"] is None:
        print("IK UNREACHABLE — aborting", file=sys.stderr)
        sys.exit(2)
    print(f"target  : ({args.target[0]:+.4f}, {args.target[1]:+.4f}, "
          f"{args.target[2]:+.4f}) m")
    print(f"angles  : {out['angles_deg']} deg")
    print(f"steps   : {out['angles_steps']}")
    print(f"feasible: {out['ok']}")
    if not out["ok"]:
        print("⚠ ball joint limit exceeded — proceeding anyway", file=sys.stderr)

    cfg = MotorClientConfig(
        port=args.port,
        baud=args.baud,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    with LevelingMotorClient(cfg) as mc:
        if not args.dry_run:
            assert mc.ping(), "PING failed"
        if args.home:
            mc.home()
        mc.aim(out)
        st = mc.status()
        print(f"final status: {st}")
