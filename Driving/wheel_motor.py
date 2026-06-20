"""
Wheel Motor Client — Pi5 ↔ OpenRB-150 serial driver for differential-drive wheel
angular velocities.

Wire protocol (full spec in docs/superpowers/specs/2026-05-04-driving-pipeline-design.md §3):

    Pi → OpenRB                   OpenRB → Pi
    ─────────────                 ─────────────
    PING\\n                        PONG\\n           (sync, health check)
    DRIVE <wL> <wR>\\n              (no reply)       (fire-and-forget @ 15 Hz)
    STOP\\n                        OK\\n             (sync, terminal stop)

`<wL>`, `<wR>` are signed integer mrad/s (rad/s × 1000), clamped to ±30000.
The OpenRB firmware is expected to autonomously zero both motors if no DRIVE
packet arrives within 200 ms (watchdog) — this script is correct only against
that firmware contract.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─────────────────────────── protocol constants ───────────────────────────
_PING = "PING"
_PONG = "PONG"
_STOP = "STOP"
_OK = "OK"
_DRIVE_FMT = "DRIVE {wL} {wR}"


# ──────────────────────────────── config ──────────────────────────────────
@dataclass
class WheelMotorConfig:
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    open_settle_sec: float = 2.0
    sync_read_timeout_sec: float = 1.0
    write_timeout_sec: float = 0.5

    max_wheel_mrad_s: int = 30000
    deadzone_mrad_s: int = 5

    direction_signs: Tuple[int, int] = (+1, +1)
    swap_lr: bool = False

    verbose: bool = False
    dry_run: bool = False


# ──────────────────────────────── client ──────────────────────────────────
class WheelMotorClient:
    """Streaming wheel-velocity client. Use as a context manager."""

    def __init__(self, cfg: Optional[WheelMotorConfig] = None):
        self.cfg = cfg if cfg is not None else WheelMotorConfig()
        self._ser = None
        self.sent_lines: List[str] = []   # populated only when dry_run=True

    # ─────────────────────────── public API ───────────────────────────
    def drive(self, wL_rad_s: float, wR_rad_s: float) -> None:
        """Fire-and-forget: write one DRIVE line. Quantize, sign, clamp, deadzone."""
        wL_mrad, wR_mrad = self._prepare_drive_pair(wL_rad_s, wR_rad_s)
        line = _DRIVE_FMT.format(wL=wL_mrad, wR=wR_mrad)
        self._send_line(line, expect_reply=False)

    def ping(self) -> bool:
        return self._send_line(_PING, expect_reply=True) == _PONG

    def stop(self) -> bool:
        return self._send_line(_STOP, expect_reply=True) == _OK

    # ────────────────────────── helpers ──────────────────────────
    def _prepare_drive_pair(self, wL_rad_s: float, wR_rad_s: float) -> tuple[int, int]:
        c = self.cfg
        wL = int(round(wL_rad_s * 1000.0)) * c.direction_signs[0]
        wR = int(round(wR_rad_s * 1000.0)) * c.direction_signs[1]
        # clamp
        wL = max(-c.max_wheel_mrad_s, min(c.max_wheel_mrad_s, wL))
        wR = max(-c.max_wheel_mrad_s, min(c.max_wheel_mrad_s, wR))
        # deadzone (both sides below → both zero; preserves explicit 0,0 stops)
        if abs(wL) < c.deadzone_mrad_s and abs(wR) < c.deadzone_mrad_s:
            wL = 0
            wR = 0
        if c.swap_lr:
            wL, wR = wR, wL
        return wL, wR

    def _send_line(self, line: str, expect_reply: bool) -> Optional[str]:
        self._log(f"→ {line}")
        if self.cfg.dry_run:
            self.sent_lines.append(line)
            if expect_reply:
                return self._dry_run_response(line)
            return None
        if self._ser is None:
            raise RuntimeError("not connected (call connect() or use context manager)")
        self._ser.write((line + "\n").encode("ascii"))
        self._ser.flush()
        if not expect_reply:
            return None
        raw = self._ser.readline()
        if not raw:
            raise TimeoutError(f"no response to {line!r} within "
                               f"{self.cfg.sync_read_timeout_sec}s")
        resp = raw.decode("ascii", errors="replace").rstrip("\r\n")
        self._log(f"← {resp}")
        return resp

    @staticmethod
    def _dry_run_response(line: str) -> str:
        if line == _PING:
            return _PONG
        if line == _STOP:
            return _OK
        return ""   # DRIVE has no reply

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[wheel] {msg}", file=sys.stderr)

    # ─────────────────────────── lifecycle ───────────────────────────
    def connect(self) -> None:
        if self.cfg.dry_run:
            self._log("[dry-run] skip serial open")
            return
        import serial   # lazy import; pyserial unneeded for dry-run / tests
        self._ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baud,
            timeout=self.cfg.sync_read_timeout_sec,
            write_timeout=self.cfg.write_timeout_sec,
        )
        time.sleep(self.cfg.open_settle_sec)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._log(f"[open] {self.cfg.port} @ {self.cfg.baud}")

    def disconnect(self) -> None:
        # Always attempt a final STOP; swallow errors so finally-paths are robust.
        try:
            self.stop()
        except Exception:
            pass
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._log("[close]")

    def __enter__(self) -> "WheelMotorClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


# ───────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Wheel motor client — single-shot bench test")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--wL", type=float, default=0.0, help="left wheel ω [rad/s]")
    ap.add_argument("--wR", type=float, default=0.0, help="right wheel ω [rad/s]")
    ap.add_argument("--ping", action="store_true",
                    help="send PING and print result, then exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--swap-lr", action="store_true",
                    help="swap L/R before sending (workaround for inverted firmware wiring)")
    args = ap.parse_args()

    cfg = WheelMotorConfig(
        port=args.port, baud=args.baud,
        dry_run=args.dry_run, verbose=args.verbose,
        swap_lr=args.swap_lr,
    )
    with WheelMotorClient(cfg) as mc:
        if args.ping:
            ok = mc.ping()
            print(f"PING → {'PONG' if ok else 'FAIL'}")
        else:
            mc.drive(args.wL, args.wR)
            print(f"sent DRIVE {args.wL:+.3f} {args.wR:+.3f} rad/s")
