# Phase-1 Driving Pipeline (drive_to + wheel_motor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Phase-1 driving runner that reads pose from ORB-SLAM3, computes wheel angular velocities via the existing `DrivingController`, and streams them to OpenRB-150 over an ASCII line protocol — with a Tier-1 safety supervisor for SLAM tracking loss and pose-jump rejection.

**Architecture:** Two new files in `Driving/` that mirror the existing `LevelingPlatform/` decomposition. `wheel_motor.py` is a serial client (sibling of `leveling_motor.py`); `drive_to.py` is the runner that wires localizer + controller + motor + safety together. No changes to existing production modules in this PR.

**Tech Stack:** Python 3 stdlib + numpy (existing) + pyserial 3.5 (already installed). Tests use stdlib `unittest`. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-05-04-driving-pipeline-design.md](../specs/2026-05-04-driving-pipeline-design.md)

---

## File Structure

```
Driving/
├── wheel_motor.py              # NEW — Pi5 ↔ OpenRB-150 wheel-velocity serial client
├── drive_to.py                 # NEW — Phase-1 runner (CLI, _run_loop, SafetySupervisor)
└── tests/
    ├── __init__.py             # NEW — empty marker (matches perception/training/tests/)
    ├── test_wheel_motor.py     # NEW — dry-run unit tests
    └── test_safety_supervisor.py  # NEW — pure unit tests with injected clock
```

`pipeline.py`, `controller.py`, `simulation.py`, and the leveling/perception modules are untouched in this PR. The follow-up PRs (firmware, pipeline.py stub replacement) are listed in the spec §8.

---

## Architectural decisions worth flagging

These are concrete choices that the spec described abstractly:

1. **Test seam — both `WheelMotorClient` and `SafetySupervisor` accept injected callables for testability:**
   - `WheelMotorClient`: when `dry_run=True`, every line that would be written is also appended to `self.sent_lines: list[str]`. Tests inspect this list directly. (No need to capture stderr.)
   - `SafetySupervisor`: constructor takes `now: Callable[[], float] = time.monotonic` and `log: Callable[[str], None] = print`. Tests pass a clock object whose value they advance, and a list-appender for log capture.
2. **Module-level constants for the wire format** in `wheel_motor.py`:
   - `_DRIVE_FMT = "DRIVE {wL} {wR}"`, `_PING = "PING"`, `_PONG = "PONG"`, `_STOP = "STOP"`, `_OK = "OK"`. Used by both production code and tests so a typo in the protocol is caught immediately.
3. **`_run_loop` is split out from `main`** so the loop body can be tested with a fake `LocalizerLike` and `MotorLike` without spawning ORB-SLAM3 or opening serial.

---

## Task 1: Test directory scaffolding

**Files:**
- Create: `Driving/tests/__init__.py` (empty file, just marks the package)

- [ ] **Step 1: Create the empty package marker**

Create [Driving/tests/__init__.py](Driving/tests/__init__.py) as a 0-byte file. This matches [perception/training/tests/__init__.py](perception/training/tests/__init__.py).

- [ ] **Step 2: Verify discovery works**

Run from repo root: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest discover -s Driving/tests -v`
Expected: `Ran 0 tests in 0.000s — OK` (no tests yet, but discovery succeeds with no errors).

- [ ] **Step 3: Commit**

```bash
git add Driving/tests/__init__.py
git commit -m "test(driving): add tests package marker"
```

---

## Task 2: `WheelMotorClient` — config dataclass + skeleton

**Files:**
- Create: `Driving/wheel_motor.py`
- Test: `Driving/tests/test_wheel_motor.py`

- [ ] **Step 1: Write failing test for default config values**

Create [Driving/tests/test_wheel_motor.py](Driving/tests/test_wheel_motor.py):

```python
from __future__ import annotations

import unittest

from Driving.wheel_motor import WheelMotorClient, WheelMotorConfig


class TestWheelMotorConfig(unittest.TestCase):
    def test_defaults(self):
        c = WheelMotorConfig()
        self.assertEqual(c.port, "/dev/ttyACM0")
        self.assertEqual(c.baud, 115200)
        self.assertEqual(c.max_wheel_mrad_s, 30000)
        self.assertEqual(c.deadzone_mrad_s, 5)
        self.assertEqual(c.direction_signs, (+1, +1))
        self.assertFalse(c.verbose)
        self.assertFalse(c.dry_run)


class TestWheelMotorClientConstruction(unittest.TestCase):
    def test_can_instantiate_with_dry_run(self):
        client = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(client.cfg.dry_run)
        self.assertEqual(client.sent_lines, [])
```

- [ ] **Step 2: Run test, confirm it fails with import error**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: `ModuleNotFoundError: No module named 'Driving.wheel_motor'`.

- [ ] **Step 3: Create `wheel_motor.py` with config + skeleton**

Create [Driving/wheel_motor.py](Driving/wheel_motor.py):

```python
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

    verbose: bool = False
    dry_run: bool = False


# ──────────────────────────────── client ──────────────────────────────────
class WheelMotorClient:
    """Streaming wheel-velocity client. Use as a context manager."""

    def __init__(self, cfg: Optional[WheelMotorConfig] = None):
        self.cfg = cfg if cfg is not None else WheelMotorConfig()
        self._ser = None
        self.sent_lines: List[str] = []   # populated only when dry_run=True
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/wheel_motor.py Driving/tests/test_wheel_motor.py
git commit -m "feat(driving): wheel motor client config + skeleton"
```

---

## Task 3: `WheelMotorClient.drive()` — quantization, deadzone, clamp, direction_signs

**Files:**
- Modify: `Driving/wheel_motor.py`
- Modify: `Driving/tests/test_wheel_motor.py`

- [ ] **Step 1: Write failing tests for drive() quantization/clamp/deadzone**

Append to [Driving/tests/test_wheel_motor.py](Driving/tests/test_wheel_motor.py):

```python
class TestDriveQuantization(unittest.TestCase):
    def _client(self, **cfg_overrides):
        cfg = WheelMotorConfig(dry_run=True, **cfg_overrides)
        return WheelMotorClient(cfg)

    def test_zero_zero_emits_zero_zero(self):
        c = self._client()
        c.drive(0.0, 0.0)
        self.assertEqual(c.sent_lines, ["DRIVE 0 0"])

    def test_quantizes_to_mrad_per_sec(self):
        c = self._client()
        c.drive(1.234, -2.345)
        self.assertEqual(c.sent_lines, ["DRIVE 1234 -2345"])

    def test_rounds_to_nearest_mrad(self):
        c = self._client()
        c.drive(0.0014, -0.0016)   # 1.4 → 1, -1.6 → -2
        # both inside deadzone (|w| < 5 mrad) → forced to 0
        self.assertEqual(c.sent_lines, ["DRIVE 0 0"])

    def test_deadzone_zeros_both_when_both_below(self):
        c = self._client()
        c.drive(0.003, -0.004)   # 3 and -4 mrad, both < 5 → 0 0
        self.assertEqual(c.sent_lines, ["DRIVE 0 0"])

    def test_deadzone_does_not_zero_when_one_side_above(self):
        c = self._client()
        c.drive(0.003, 1.0)   # 3 mrad (inside) and 1000 mrad (outside) → keep 3
        self.assertEqual(c.sent_lines, ["DRIVE 3 1000"])

    def test_clamps_to_max(self):
        c = self._client()
        c.drive(50.0, -50.0)
        self.assertEqual(c.sent_lines, ["DRIVE 30000 -30000"])

    def test_direction_signs_flip_right_wheel(self):
        c = self._client(direction_signs=(+1, -1))
        c.drive(1.0, 1.0)
        self.assertEqual(c.sent_lines, ["DRIVE 1000 -1000"])

    def test_multiple_calls_accumulate_in_sent_lines(self):
        c = self._client()
        c.drive(0.0, 0.0)
        c.drive(1.0, -1.0)
        c.drive(0.0, 0.0)
        self.assertEqual(c.sent_lines, ["DRIVE 0 0", "DRIVE 1000 -1000", "DRIVE 0 0"])
```

- [ ] **Step 2: Run tests, confirm they fail with AttributeError or similar**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 8 new tests fail (`AttributeError: 'WheelMotorClient' object has no attribute 'drive'`).

- [ ] **Step 3: Implement `drive()` and helpers in `wheel_motor.py`**

Append to [Driving/wheel_motor.py](Driving/wheel_motor.py) (inside the `WheelMotorClient` class):

```python
    # ─────────────────────────── public API ───────────────────────────
    def drive(self, wL_rad_s: float, wR_rad_s: float) -> None:
        """Fire-and-forget: write one DRIVE line. Quantize, sign, clamp, deadzone."""
        wL_mrad, wR_mrad = self._prepare_drive_pair(wL_rad_s, wR_rad_s)
        line = _DRIVE_FMT.format(wL=wL_mrad, wR=wR_mrad)
        self._send_line(line, expect_reply=False)

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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 10 tests pass (2 from Task 2 + 8 from Task 3).

- [ ] **Step 5: Commit**

```bash
git add Driving/wheel_motor.py Driving/tests/test_wheel_motor.py
git commit -m "feat(driving): wheel motor drive() with quantization/deadzone/clamp"
```

---

## Task 4: `WheelMotorClient.ping()` and `.stop()` — synchronous request-response

**Files:**
- Modify: `Driving/wheel_motor.py`
- Modify: `Driving/tests/test_wheel_motor.py`

- [ ] **Step 1: Write failing tests for ping() and stop()**

Append to [Driving/tests/test_wheel_motor.py](Driving/tests/test_wheel_motor.py):

```python
class TestPingStop(unittest.TestCase):
    def test_ping_returns_true_in_dry_run(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(c.ping())
        self.assertEqual(c.sent_lines, ["PING"])

    def test_stop_returns_true_in_dry_run(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(c.stop())
        self.assertEqual(c.sent_lines, ["STOP"])
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 2 new tests fail with `AttributeError`.

- [ ] **Step 3: Implement ping() and stop()**

Append inside the `WheelMotorClient` class in [Driving/wheel_motor.py](Driving/wheel_motor.py):

```python
    def ping(self) -> bool:
        return self._send_line(_PING, expect_reply=True) == _PONG

    def stop(self) -> bool:
        return self._send_line(_STOP, expect_reply=True) == _OK
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/wheel_motor.py Driving/tests/test_wheel_motor.py
git commit -m "feat(driving): wheel motor PING/STOP synchronous commands"
```

---

## Task 5: Connect/disconnect lifecycle + context manager

**Files:**
- Modify: `Driving/wheel_motor.py`
- Modify: `Driving/tests/test_wheel_motor.py`

- [ ] **Step 1: Write failing tests for context manager + auto-stop on disconnect**

Append to [Driving/tests/test_wheel_motor.py](Driving/tests/test_wheel_motor.py):

```python
class TestLifecycle(unittest.TestCase):
    def test_context_manager_does_not_raise_in_dry_run(self):
        with WheelMotorClient(WheelMotorConfig(dry_run=True)) as c:
            c.drive(1.0, 1.0)
        # On exit, disconnect() should send STOP. In dry-run, that
        # appears in sent_lines.
        self.assertEqual(c.sent_lines[-1], "STOP")

    def test_disconnect_when_never_connected_is_safe(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        c.disconnect()   # should not raise even though connect() never called
        # in dry-run, disconnect sends STOP unconditionally
        self.assertEqual(c.sent_lines, ["STOP"])

    def test_connect_dry_run_is_noop(self):
        c = WheelMotorClient(WheelMotorConfig(dry_run=True))
        c.connect()      # must not try to import or open pyserial
        self.assertIsNone(c._ser)
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 3 new failures (`AttributeError: connect`, etc.).

- [ ] **Step 3: Implement connect/disconnect/__enter__/__exit__**

Append inside the `WheelMotorClient` class in [Driving/wheel_motor.py](Driving/wheel_motor.py):

```python
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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/wheel_motor.py Driving/tests/test_wheel_motor.py
git commit -m "feat(driving): wheel motor connect/disconnect + context manager"
```

---

## Task 6: `WheelMotorClient` — small CLI for manual bench check

**Files:**
- Modify: `Driving/wheel_motor.py`

This task adds a `__main__` block so the file can be invoked directly to test serial round-trip without writing the full `drive_to.py`. Mirrors `LevelingPlatform/leveling_motor.py`'s CLI.

- [ ] **Step 1: Append CLI to wheel_motor.py**

Append at the bottom of [Driving/wheel_motor.py](Driving/wheel_motor.py):

```python
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
    args = ap.parse_args()

    cfg = WheelMotorConfig(
        port=args.port, baud=args.baud,
        dry_run=args.dry_run, verbose=args.verbose,
    )
    with WheelMotorClient(cfg) as mc:
        if args.ping:
            ok = mc.ping()
            print(f"PING → {'PONG' if ok else 'FAIL'}")
        else:
            mc.drive(args.wL, args.wR)
            print(f"sent DRIVE {args.wL:+.3f} {args.wR:+.3f} rad/s")
```

- [ ] **Step 2: Smoke test the CLI in dry-run**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python Driving/wheel_motor.py --dry-run --verbose --wL 1.5 --wR -1.5`
Expected stderr lines: `[wheel] [dry-run] skip serial open`, `[wheel] → DRIVE 1500 -1500`, `[wheel] → STOP` (from auto-disconnect). Expected stdout: `sent DRIVE +1.500 -1.500 rad/s`.

- [ ] **Step 3: Smoke test PING in dry-run**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python Driving/wheel_motor.py --dry-run --ping`
Expected stdout: `PING → PONG`.

- [ ] **Step 4: Confirm all unit tests still pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_wheel_motor -v`
Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/wheel_motor.py
git commit -m "feat(driving): wheel motor CLI for bench testing"
```

---

## Task 7: `SafetySupervisor` — config dataclass + skeleton + OK path

**Files:**
- Create: `Driving/tests/test_safety_supervisor.py`
- Modify: `Driving/drive_to.py` (creates the file with just SafetyConfig + SafetySupervisor + OK path; runner code added later)

- [ ] **Step 1: Write failing tests for OK path**

Create [Driving/tests/test_safety_supervisor.py](Driving/tests/test_safety_supervisor.py):

```python
from __future__ import annotations

import unittest
from typing import List, Tuple

from Driving.drive_to import SafetyConfig, SafetySupervisor


class _Clock:
    def __init__(self, t0: float = 0.0):
        self.t = t0
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


def _ok(x: float, y: float):
    return {"x": x, "y": y, "theta": 0.0, "tracking_ok": True, "tracking": "OK"}

def _lost():
    return {"x": 0.0, "y": 0.0, "theta": 0.0, "tracking_ok": False, "tracking": "LOST"}


class TestSupervisorOKPath(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(),
            now=self.clock,
            log=self.logs.append,
        )

    def test_first_ok_returns_ok(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.assertEqual(self.logs, [])

    def test_consecutive_ok_within_velocity_returns_ok(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)             # 15 Hz period
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.02, 0.0)), "OK")
        self.assertEqual(self.logs, [])
```

- [ ] **Step 2: Run tests, confirm they fail with import error**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: `ModuleNotFoundError: No module named 'Driving.drive_to'`.

- [ ] **Step 3: Create `drive_to.py` with SafetyConfig + SafetySupervisor (OK path only)**

Create [Driving/drive_to.py](Driving/drive_to.py):

```python
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
    lost_warn_sec:     float = 3.0     # logging window above quiet
    # ABORT after lost_quiet_sec + lost_warn_sec total

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

    def check(self, pose: Optional[Dict]) -> str:
        # Lost-tracking and pose-jump branches are added in later tasks.
        # For now: accept every frame.
        t = self._now()
        self._last_ok = (float(pose["x"]), float(pose["y"]), t)
        self._lost_since = None
        self._consec_outliers = 0
        return "OK"
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/drive_to.py Driving/tests/test_safety_supervisor.py
git commit -m "feat(driving): SafetySupervisor skeleton + OK path"
```

---

## Task 8: `SafetySupervisor` — tracking-lost escalation (HOLD → ABORT)

**Files:**
- Modify: `Driving/drive_to.py`
- Modify: `Driving/tests/test_safety_supervisor.py`

- [ ] **Step 1: Write failing tests for lost escalation**

Append to [Driving/tests/test_safety_supervisor.py](Driving/tests/test_safety_supervisor.py):

```python
class TestSupervisorLostEscalation(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(lost_quiet_sec=0.5, lost_warn_sec=3.0,
                             warn_log_period=0.5),
            now=self.clock,
            log=self.logs.append,
        )

    def test_short_lost_under_quiet_threshold_holds_silently(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.clock.advance(0.3)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(self.logs, [])  # silent

    def test_lost_in_warn_window_logs_and_holds(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(0.6)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 1)
        self.assertIn("tracking lost", self.logs[0])
        # next check 0.1s later: still under warn_log_period → no new log
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 1)
        # now 0.5s after first warn → new log line
        self.clock.advance(0.5)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 2)

    def test_lost_beyond_total_threshold_aborts(self):
        self.sup.check(_ok(0.0, 0.0))
        # quiet (0.5) + warn (3.0) = 3.5s total before abort
        self.clock.advance(3.6)
        self.assertEqual(self.sup.check(_lost()), "ABORT")
        self.assertIn("tracking lost", self.sup.reason)

    def test_recovery_clears_lost_state(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(1.0)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.logs.clear()
        # should not log "lost" anymore
        self.clock.advance(0.6)
        # treat next pose as fresh; no new lost
        self.assertEqual(self.sup.check(_ok(0.001, 0.0)), "OK")
        self.assertEqual(self.logs, [])
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 4 new tests fail.

- [ ] **Step 3: Implement lost-tracking branch in `check()`**

Replace the body of `SafetySupervisor.check()` in [Driving/drive_to.py](Driving/drive_to.py) with:

```python
    def check(self, pose: Optional[Dict]) -> str:
        c = self.cfg
        t = self._now()

        # Branch A: tracking lost or pose unavailable
        if pose is None or not pose.get("tracking_ok", False):
            if self._lost_since is None:
                self._lost_since = t
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

        # POSE_JUMP_REJECTION_HOOK  (next task inserts here)

        # accepted
        self._last_ok = (float(pose["x"]), float(pose["y"]), t)
        self._lost_since = None
        self._consec_outliers = 0
        return "OK"
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/drive_to.py Driving/tests/test_safety_supervisor.py
git commit -m "feat(driving): SafetySupervisor tracking-lost escalation"
```

---

## Task 9: `SafetySupervisor` — pose-jump rejection

**Files:**
- Modify: `Driving/drive_to.py`
- Modify: `Driving/tests/test_safety_supervisor.py`

- [ ] **Step 1: Write failing tests for pose-jump rejection**

Append to [Driving/tests/test_safety_supervisor.py](Driving/tests/test_safety_supervisor.py):

```python
class TestSupervisorPoseJump(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        # max_linear_vel=0.3, jump_factor=3 → at dt=0.067s, threshold = 0.06m
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(max_linear_vel=0.3, jump_factor=3.0,
                             jump_outlier_max=3),
            now=self.clock,
            log=self.logs.append,
        )

    def test_single_jump_holds_then_recovers(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)                    # threshold ≈ 0.06 m
        self.assertEqual(self.sup.check(_ok(1.0, 0.0)), "HOLD")  # 1m jump
        self.clock.advance(0.067)
        # next plausible pose (close to last_ok=(0,0)) → OK, counter resets
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")

    def test_three_consecutive_jumps_abort(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        for i in range(3):
            self.clock.advance(0.067)
            res = self.sup.check(_ok(10.0 + i, 0.0))
            if i < 2:
                self.assertEqual(res, "HOLD")
            else:
                self.assertEqual(res, "ABORT")
        self.assertIn("pose jump", self.sup.reason)

    def test_non_consecutive_jump_does_not_accumulate(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(5.0, 0.0)), "HOLD")     # jump 1
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")       # reset
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(5.0, 0.0)), "HOLD")     # jump 1 again, not 2
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.02, 0.0)), "OK")

    def test_long_dt_disables_jump_check(self):
        # If dt >= 1s (e.g. after a long pause), don't classify as a jump.
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(1.5)
        self.assertEqual(self.sup.check(_ok(2.0, 0.0)), "OK")
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 4 new tests fail.

- [ ] **Step 3: Insert pose-jump rejection into `check()`**

In [Driving/drive_to.py](Driving/drive_to.py), replace the `# POSE_JUMP_REJECTION_HOOK  (next task inserts here)` line with:

```python
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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Driving/drive_to.py Driving/tests/test_safety_supervisor.py
git commit -m "feat(driving): SafetySupervisor pose-jump rejection"
```

---

## Task 10: `_run_loop` — control loop with injectable dependencies

**Files:**
- Modify: `Driving/drive_to.py`
- Modify: `Driving/tests/test_safety_supervisor.py` → rename intent: this file gets a new `TestRunLoop` class. (No new test file — keeps related fakes in one place.)

The loop is testable because its dependencies are passed as arguments. We don't need a Localizer abstract class — duck typing is enough since we only use `get_pose()`.

- [ ] **Step 1: Write failing tests for `_run_loop` happy path and ABORT path**

Append to [Driving/tests/test_safety_supervisor.py](Driving/tests/test_safety_supervisor.py):

```python
from Driving.drive_to import RunArgs, _run_loop


class _FakeLocalizer:
    def __init__(self, poses):
        self._poses = list(poses)
    def get_pose(self):
        if not self._poses:
            return None
        return self._poses.pop(0)


class _FakeMotor:
    def __init__(self):
        self.calls: List[Tuple[float, float]] = []
    def drive(self, wL, wR):
        self.calls.append((wL, wR))


class _FakeController:
    """Returns a fixed (small) command and reports reached after N calls."""
    def __init__(self, reach_after: int):
        self.calls = 0
        self._reach_after = reach_after
    def compute(self, x, y, theta, tx, ty):
        self.calls += 1
        return {
            "wheel_omega_left": 1.0,
            "wheel_omega_right": 1.0,
            "v": 0.1, "omega": 0.0,
            "distance": 0.05 if self.calls >= self._reach_after else 1.0,
            "angle_error": 0.0,
            "reached": self.calls >= self._reach_after,
        }


class _SleepNoop:
    def __init__(self): self.calls = 0
    def __call__(self, _seconds): self.calls += 1


class TestRunLoop(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(x=1.0, y=0.0, rate=15.0, timeout=10.0,
                        port="/dev/null", baud=115200,
                        dry_run=True, verbose=False)
        defaults.update(overrides)
        return RunArgs(**defaults)

    def test_reaches_target_returns_zero(self):
        clock = _Clock()
        sup = SafetySupervisor(SafetyConfig(), now=clock, log=lambda _m: None)
        loc = _FakeLocalizer([_ok(0.0, 0.0), _ok(0.01, 0.0), _ok(0.02, 0.0)])
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=3)
        sleep = _SleepNoop()

        rc = _run_loop(self._args(), loc, ctrl, motor, sup,
                       now=clock, sleep=sleep)
        self.assertEqual(rc, 0)
        # last call must be (0, 0) per "send zero on reach" requirement
        self.assertEqual(motor.calls[-1], (0.0, 0.0))

    def test_abort_from_supervisor_returns_two(self):
        clock = _Clock()
        sup = SafetySupervisor(
            SafetyConfig(lost_quiet_sec=0.0, lost_warn_sec=0.0),
            now=clock, log=lambda _m: None,
        )
        # First pose triggers ABORT (lost from frame 1, total threshold = 0)
        loc = _FakeLocalizer([_lost()])
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=999)
        sleep = _SleepNoop()

        rc = _run_loop(self._args(), loc, ctrl, motor, sup,
                       now=clock, sleep=sleep)
        self.assertEqual(rc, 2)

    def test_timeout_returns_one(self):
        clock = _Clock()
        sup = SafetySupervisor(SafetyConfig(), now=clock, log=lambda _m: None)
        # Endless OK frames; controller never reaches.
        loc = _FakeLocalizer([_ok(0.0, 0.0)] * 10000)
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=10**9)
        # advance the clock inside sleep so timeout actually fires
        period = 1.0 / 15.0

        def sleep_fn(_secs):
            clock.advance(period)

        args = self._args(timeout=0.5)   # ≤ 8 iterations at 15 Hz
        rc = _run_loop(args, loc, ctrl, motor, sup, now=clock, sleep=sleep_fn)
        self.assertEqual(rc, 1)

    def test_hold_sends_zero_velocity(self):
        clock = _Clock()
        sup = SafetySupervisor(SafetyConfig(lost_quiet_sec=0.5),
                               now=clock, log=lambda _m: None)
        loc = _FakeLocalizer([_ok(0.0, 0.0), _lost(), _lost()])
        motor = _FakeMotor()
        ctrl = _FakeController(reach_after=999)

        period = 1.0 / 15.0
        def sleep_fn(_secs): clock.advance(period)

        # Three frames; we expect the second and third to issue HOLD → drive(0,0)
        args = self._args(timeout=0.25)   # 0.25 / period ≈ 3 iterations
        _ = _run_loop(args, loc, ctrl, motor, sup, now=clock, sleep=sleep_fn)
        # At least the HOLD frames should have produced (0.0, 0.0).
        self.assertIn((0.0, 0.0), motor.calls)
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 4 new failures (`ImportError: cannot import name 'RunArgs'` or `_run_loop`).

- [ ] **Step 3: Add `RunArgs` and `_run_loop` to `drive_to.py`**

Append to [Driving/drive_to.py](Driving/drive_to.py):

```python
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
        action = supervisor.check(pose)

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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests.test_safety_supervisor -v`
Expected: 14 tests pass (10 supervisor + 4 run-loop).

- [ ] **Step 5: Commit**

```bash
git add Driving/drive_to.py Driving/tests/test_safety_supervisor.py
git commit -m "feat(driving): drive_to _run_loop with injectable dependencies"
```

---

## Task 11: `main()` — CLI parser + integration of localizer/controller/motor

**Files:**
- Modify: `Driving/drive_to.py`

This task wires the real `OrbSlamLocalizer`, `DrivingController`, and `WheelMotorClient` together. It is harder to unit-test because of the import-time RealSense/ORB-SLAM dependencies; we verify it via a manual dry-run smoke test in Task 12.

- [ ] **Step 1: Add `main()` and the CLI parser to `drive_to.py`**

Append to [Driving/drive_to.py](Driving/drive_to.py):

```python
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
    a = ap.parse_args(argv)

    args = RunArgs(
        x=a.x, y=a.y, rate=a.rate, timeout=a.timeout,
        port=a.port, baud=a.baud, dry_run=a.dry_run, verbose=a.verbose,
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

    ctrl = DrivingController(ControllerConfig(dt=1.0 / args.rate))
    motor = WheelMotorClient(WheelMotorConfig(
        port=args.port, baud=args.baud,
        dry_run=args.dry_run, verbose=args.verbose,
    ))
    loc = OrbSlamLocalizer(LocalizerConfig())
    supervisor = SafetySupervisor(SafetyConfig())

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
        if not loc.wait_for_tracking(timeout=30.0):
            print("[FAIL] SLAM did not reach tracking OK within 30s",
                  file=sys.stderr)
            return 2

        try:
            return _run_loop(args, loc, ctrl, motor, supervisor)
        finally:
            try:
                motor.drive(0.0, 0.0)
            except Exception:
                pass
            # synchronous STOP is sent by motor.disconnect() via __exit__


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify the CLI parser shows --help cleanly**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python Driving/drive_to.py --help`
Expected: argparse usage block listing `--x`, `--y`, `--port`, `--baud`, `--rate`, `--timeout`, `--dry-run`, `--verbose`. Exit 0.

- [ ] **Step 3: Verify required-arg validation**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python Driving/drive_to.py 2>&1; echo "exit=$?"`
Expected: argparse error message about missing `--x`/`--y` arguments. `exit=2`.

- [ ] **Step 4: Confirm all unit tests still pass**

Run: `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests -v 2>&1 | tail -5`
Expected: ~29 tests pass (15 wheel_motor + 14 supervisor/run_loop).

- [ ] **Step 5: Commit**

```bash
git add Driving/drive_to.py
git commit -m "feat(driving): drive_to main() CLI + module integration"
```

---

## Task 12: Manual dry-run smoke test on Pi5 (operator step)

**Files:** none modified — verification only.

This is a manual step that requires a Pi5 with RealSense + ORB-SLAM3 actually running. It is the only end-to-end check before we hand the script to firmware integration.

- [ ] **Step 1: Verify localizer + controller wiring with `--dry-run`**

On the Pi5, run from repo root: `python Driving/drive_to.py --x 1 --y 0 --dry-run --verbose`

Confirm in the output:
- A `========` banner with the target.
- `[wheel] [dry-run] skip serial open` (verbose stderr line).
- ORB-SLAM3 startup logs from the localizer.
- After tracking acquires (within 30 s), the loop emits one `pose=(... ) dist=... v=... ω_L/R=(...)` line every ~0.5 s.
- Each frame produces a `[wheel] → DRIVE wL_mrad wR_mrad` line on stderr.

- [ ] **Step 2: Verify lost-tracking escalation by covering the camera**

While the script is running, cover the RealSense lens with a hand. Within 0.5 s a `[WARN] tracking lost X.Xs` log should appear; covering for ~3.5 s should trigger `[ABORT] tracking lost ...` and the process should exit with code 2.

Verify the exit code: append `; echo "exit=$?"` to the command line.

- [ ] **Step 3: Verify Ctrl-C cleanup**

Re-run the script normally; press Ctrl-C mid-run. The last lines on stderr should include `[wheel] → DRIVE 0 0` (from the `finally` block) and `[wheel] → STOP` (from the context-manager exit). Exit code is 130.

- [ ] **Step 4: No code change — no commit. Note the result in PR description.**

If any of Steps 1–3 do not behave as described, file a bug task before merging.

---

## Task 13: Update SW_ARCHITECTURE.md

**Files:**
- Modify: `SW_ARCHITECTURE.md`

- [ ] **Step 1: Add a row to the §4.3 modules table**

Open [SW_ARCHITECTURE.md](SW_ARCHITECTURE.md). In the table at §4.3 (lines 106–122), insert two rows immediately after the row for `Driving/controller.py` (the row containing `상위 제어 모듈 (production)`):

```markdown
| **휠 모터 시리얼 클라이언트 (production)** | **[Driving/wheel_motor.py](Driving/wheel_motor.py)** — Pi → OpenRB-150 (ASCII protocol, fire-and-forget DRIVE) |
| **Phase-1 only 주행 러너** | **[Driving/drive_to.py](Driving/drive_to.py)** — 목표 (x, y) → ORB-SLAM3 → controller → 휠 모터 |
```

- [ ] **Step 2: Add a "주행 단독 실행" CLI block to §7**

In §7 ("실행 진입점"), insert the following block immediately before the `# 통합 파이프라인` block:

```markdown
# Phase-1 only 주행 러너 (단독 실행, 개발/실험용)
python Driving/drive_to.py --x 3 --y 2                          # 실시리얼 + ORB-SLAM3
python Driving/drive_to.py --x 3 --y 2 --dry-run --verbose      # 시리얼 미접속, 송신 라인만 출력
```

- [ ] **Step 3: Update §9 TODO list**

In §9, replace the line:

```markdown
- [ ] [pipeline.py](pipeline.py) 의 `RealRobot` 모터 stub → 실제 시리얼 드라이버 연결
```

with:

```markdown
- [x] 휠 시리얼 드라이버 — [Driving/wheel_motor.py](Driving/wheel_motor.py) (ASCII line protocol)
- [ ] [pipeline.py](pipeline.py) 의 `RealRobot.send_wheel_omegas` stub → `WheelMotorClient` 로 교체 (별도 PR)
- [ ] OpenRB 펌웨어에 `DRIVE`/`STOP`/`PING` 핸들러 + 200 ms watchdog 추가 (별도 PR)
```

Also update the line `- [ ] 시리얼 프로토콜 실측 (현재 ... 시뮬레이션 스펙)` to:

```markdown
- [ ] 시리얼 프로토콜 실측 — `wheel_motor.py` 의 ASCII 프로토콜과 OpenRB 펌웨어의 라운드트립 검증
```

- [ ] **Step 4: Commit**

```bash
git add SW_ARCHITECTURE.md
git commit -m "docs(arch): document drive_to + wheel_motor in SW_ARCHITECTURE.md"
```

---

## Final verification checklist

Before opening a PR:

- [ ] `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python -m unittest Driving.tests -v` → all tests pass.
- [ ] `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python Driving/wheel_motor.py --dry-run --verbose --wL 1 --wR -1` → emits expected DRIVE / STOP lines, exit 0.
- [ ] `cd /home/sim2real/CapstoneDesign2026 && .venv/bin/python Driving/drive_to.py --help` → argparse usage prints, exit 0.
- [ ] Task 12 manual smoke test on Pi5 passed (note in PR description).
- [ ] All commits in this branch follow the project's style (`feat(driving)…`, `docs(arch)…`, etc.).

---

## What this PR explicitly does NOT do (deferred)

- **Firmware**: OpenRB-150 sketch update for `DRIVE`/`STOP`/`PING` handlers and the 200 ms watchdog. The script is correct only against that contract; without it, only `--dry-run` mode is usable. Tracked as a separate firmware PR.
- **`pipeline.py` stub replacement**: [pipeline.py:171](pipeline.py#L171) `RealRobot.send_wheel_omegas` still prints. One-line follow-up PR after this lands.
- **Tier 2/3 safety** (geofence, no-progress watchdog, freshness watchdog, encoder dead-reckoning, continuous SLAM confidence): covered in spec §5; not implemented here.
