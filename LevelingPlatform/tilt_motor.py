"""Tilt motor clients — sync `TILT` (motion-complete) + async `TILT_ASYNC` (f&f).

Wire protocol (v1.1):

    Pi → OpenRB                  OpenRB → Pi
    ─────────────                ─────────────
    TILT <s4>\\n                  OK | ERR <reason>\\n   (sync, motion-complete)
    TILT_ASYNC <s4>\\n            (no reply)             (f&f @ 15 Hz, 200ms watchdog)

`<s4>` is signed integer DXL step in [-2047, +2047]. The OpenRB firmware is
expected to hold the current position when no `TILT_ASYNC` arrives within
200 ms (not snap to 0); this client is correct only against that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Default DXL calibration — 90° = +1024 step (XL-330 / XM-430 family).
# Override in TiltMotorConfig if the mounted servo has a different gearing.
_DEFAULT_STEPS_PER_DEG = 1024.0 / 90.0   # ≈ 11.378
_STEP_MIN = -2047
_STEP_MAX = +2047

_TILT_FMT = "TILT {s}"
_TILT_ASYNC_FMT = "TILT_ASYNC {s}"
_OK = "OK"


@dataclass
class TiltMotorConfig:
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    open_settle_sec: float = 2.0
    sync_read_timeout_sec: float = 5.0   # sync TILT motion-complete up to 4s + margin
    write_timeout_sec: float = 0.5

    steps_per_deg: float = _DEFAULT_STEPS_PER_DEG
    step_min: int = _STEP_MIN
    step_max: int = _STEP_MAX

    verbose: bool = False
    dry_run: bool = False


class _BaseClient:
    """Shared serial plumbing for both Tilt clients."""

    def __init__(self, cfg: Optional[TiltMotorConfig] = None):
        self.cfg = cfg if cfg is not None else TiltMotorConfig()
        self._ser = None
        self.sent_lines: List[str] = []

    def step_from_deg(self, deg: float) -> int:
        return int(round(deg * self.cfg.steps_per_deg))

    def _clamp(self, step: int) -> int:
        return max(self.cfg.step_min, min(self.cfg.step_max, step))

    def _send_line(self, line: str, expect_reply: bool) -> Optional[str]:
        if self.cfg.dry_run:
            self.sent_lines.append(line)
            return _OK if expect_reply else None
        # Real serial path — connection lifecycle managed by the caller
        # (OpenRBClient or context manager). Kept minimal here since unit tests
        # exercise dry_run only.
        import serial  # type: ignore
        if self._ser is None:
            self._ser = serial.Serial(
                self.cfg.port, self.cfg.baud,
                timeout=self.cfg.sync_read_timeout_sec,
                write_timeout=self.cfg.write_timeout_sec,
            )
        self._ser.write((line + "\n").encode("ascii"))
        if not expect_reply:
            return None
        return self._ser.readline().decode("ascii", errors="ignore").strip()


class TiltClient(_BaseClient):
    """Sync `TILT` — motion-complete reply. Use for Phase 2 home/aim."""

    def tilt(self, step) -> bool:
        s = self._clamp(int(round(float(step))))
        line = _TILT_FMT.format(s=s)
        reply = self._send_line(line, expect_reply=True)
        return reply == _OK


class TiltAsyncClient(_BaseClient):
    """Async `TILT_ASYNC` — fire-and-forget, 15Hz streaming. Use for visual servo."""

    def send(self, step) -> None:
        s = self._clamp(int(round(float(step))))
        line = _TILT_ASYNC_FMT.format(s=s)
        self._send_line(line, expect_reply=False)
