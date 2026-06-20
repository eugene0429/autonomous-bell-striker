# Phase-1 Driving Pipeline (drive_to + wheel_motor) — Design

**Date:** 2026-05-04
**Owner:** eugene (sim2real)
**Status:** Approved (brainstorming) → ready for plan
**Related:** [SW_ARCHITECTURE.md §4](../../../SW_ARCHITECTURE.md), [pipeline.py](../../../pipeline.py),
[Driving/controller.py](../../../Driving/controller.py),
[LevelingPlatform/leveling_motor.py](../../../LevelingPlatform/leveling_motor.py),
[perception/vio/orbslam_localizer.py](../../../perception/vio/orbslam_localizer.py)

---

## 1. Goal

Provide a standalone Phase-1-only driving pipeline so the driving algorithm
can be exercised end-to-end on the real robot without dragging in Phase-2
(aiming/firing) infrastructure.

Given a target `(x, y)` in world frame (camera-start = origin), the script:

1. Reads pose from ORB-SLAM3 in real time.
2. Runs the existing `DrivingController` to compute left/right wheel
   angular velocities `(ω_L, ω_R)`.
3. Streams those velocities to OpenRB-150 over USB-CDC serial at 15 Hz.
4. Stops cleanly on goal-reach, timeout, SLAM failure, or Ctrl-C — always
   sending a final zero-velocity packet.

This delivers two reusable artifacts:

- **`Driving/wheel_motor.py`** — Pi5 ↔ OpenRB wheel-velocity serial client,
  mirroring the structure of `LevelingPlatform/leveling_motor.py`.
- **`Driving/drive_to.py`** — single-purpose driving runner that wires
  `OrbSlamLocalizer` + `DrivingController` + `WheelMotorClient` together
  with a Tier-1 safety supervisor.

Non-goals:

- Phase 2 (aiming, IK, leveling motors, firing) — already handled by
  `pipeline.py`; this script intentionally stops at goal-reach.
- Replacing or modifying [pipeline.py](../../../pipeline.py). The
  `RealRobot.send_wheel_omegas()` stub at
  [pipeline.py:171](../../../pipeline.py#L171) will be migrated to
  `WheelMotorClient` in a **separate follow-up PR**, keeping this PR's
  scope tight.
- OpenRB firmware changes. The `DRIVE`/`STOP`/`PING` handlers and the
  200 ms motion watchdog described in §3 are part of the protocol contract
  but their implementation is a **separate firmware PR**. Without that
  firmware update, the runner runs end-to-end only in `--dry-run`.
- Wheel odometry, encoder feedback, confidence-based velocity scaling,
  geofencing, no-progress watchdog — explicitly deferred (Tier 2/3 in §5).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Driving/drive_to.py            (Phase 1 runner — new)                   │
│                                                                         │
│   ┌──────────────────────┐                                              │
│   │ OrbSlamLocalizer     │  pose (world x, y, θ, tracking_ok)           │
│   │ (existing)           │ ────────────────────────────┐                │
│   └──────────────────────┘                             │                │
│                                                        ▼                │
│                                            ┌──────────────────────┐    │
│                                            │ SafetySupervisor     │    │
│                                            │ (new, in this file)  │    │
│                                            │  • lost-tracking     │    │
│                                            │    escalation        │    │
│                                            │  • pose-jump reject  │    │
│                                            └─────┬────────┬───────┘    │
│                                       OK         │  HOLD  │ ABORT      │
│                                                  ▼        ▼            │
│                                  ┌──────────────────────┐  exit 2      │
│                                  │ DrivingController    │              │
│                                  │ .compute(pose, x,y)  │              │
│                                  │ (existing)           │              │
│                                  └──────────┬───────────┘              │
│                                             │ ω_L, ω_R [rad/s]         │
│                                             ▼                          │
│                                  ┌──────────────────────┐              │
│                                  │ WheelMotorClient     │              │
│                                  │ .drive(ωL, ωR)       │              │
│                                  │ (NEW, Driving/       │              │
│                                  │  wheel_motor.py)     │              │
│                                  └──────────┬───────────┘              │
└─────────────────────────────────────────────┼──────────────────────────┘
                                              │ DRIVE <wL_mrad> <wR_mrad>\n
                                              ▼
                                ┌─────────────────────────────────┐
                                │ OpenRB-150 firmware (separate)  │
                                │  • streams wL,wR → 2 motors     │
                                │  • 200 ms watchdog → auto stop  │
                                └─────────────────────────────────┘
```

The split mirrors the existing leveling-platform decomposition
([LevelingPlatform/leveling_ik.py](../../../LevelingPlatform/leveling_ik.py)
+ [leveling_motor.py](../../../LevelingPlatform/leveling_motor.py)):
algorithm module + serial client module + thin CLI runner.

---

## 3. Wire protocol — Pi5 ↔ OpenRB-150 (wheel motors)

ASCII line-oriented, `\n`-terminated, 7-bit, 115200 8N1, USB-CDC.
Same shape as the leveling protocol so a developer who knows one can read
the other.

### 3.1 Pi → OpenRB

| Command | Payload | Sync? | Notes |
|---|---|---|---|
| `PING\n` | — | yes (waits `PONG`) | health check at startup |
| `DRIVE <wL> <wR>\n` | signed int, **mrad/s** each | **fire-and-forget** | streamed at 15 Hz |
| `STOP\n` | — | yes (waits `OK`) | issued from `finally` block |

`<wL>`, `<wR>` are signed integers in mrad/s, clamped to ±30000
(matches `ControllerConfig.max_wheel_omega = 30 rad/s`). Floats from the
controller are quantized as `int(round(ω * 1000.0))`.

Example wire bytes:

```
DRIVE 4500 5500\n         ← +4.5 / +5.5 rad/s
DRIVE 0 0\n               ← stop (still streamed each frame so watchdog stays armed)
STOP\n                    ← terminal stop, expects OK
```

### 3.2 OpenRB → Pi

| Response | When | Notes |
|---|---|---|
| `PONG\n` | reply to `PING` | |
| `OK\n` | reply to `STOP` | |
| `ERR <reason>\n` | only on `PING`/`STOP` parse/HW failure | `DRIVE` errors are **silent** (see §3.3) |

`DRIVE` is fire-and-forget: no `OK` is returned. This is intentional —
at 15 Hz, round-trip waits would consume ~30 ms of every 67 ms cycle and
hold the controller hostage to the slowest packet. The cost of that
choice is paid back by the watchdog (§3.3).

### 3.3 Firmware-side rules (separate firmware PR — **NOT** implemented here)

The script is correct only against the following OpenRB-side contract:

- **200 ms motion watchdog.** If no `DRIVE` packet (valid or invalid)
  arrives within 200 ms of the last one, the firmware autonomously
  commands both motors to 0. The Python streaming at 15 Hz (= 67 ms
  period) tolerates 1–2 dropped packets before the watchdog trips.
- **`DRIVE` parse failures are silent.** A malformed `DRIVE` line is
  dropped; the previous valid command continues to apply, and the
  watchdog keeps counting. Sending an `ERR` line back at 15 Hz would
  only saturate the link.
- **`STOP` always replies `OK`.** It also resets the watchdog and the
  current command to (0, 0).
- **`PING` always replies `PONG`** within 50 ms. Used only at startup.

These rules MUST be reflected in the OpenRB sketch before this script
can drive the real robot. They are out of scope for this PR but their
absence is the primary reason `--dry-run` exists.

---

## 4. Module specs

### 4.1 `Driving/wheel_motor.py`

Mirrors the public surface of `LevelingPlatform/leveling_motor.py`.

```python
@dataclass
class WheelMotorConfig:
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    open_settle_sec: float = 2.0       # USB-CDC reset settle
    sync_read_timeout_sec: float = 1.0 # for PING / STOP only
    write_timeout_sec: float = 0.5

    # Quantization & safety clip
    max_wheel_mrad_s: int = 30000      # ±30 rad/s
    deadzone_mrad_s: int = 5           # |w| < 5 mrad/s → 0

    # Per-side calibration (mounting orientation, etc.)
    direction_signs: Tuple[int, int] = (+1, +1)   # (left, right)

    verbose: bool = False
    dry_run: bool = False              # skip serial; print lines instead


class WheelMotorClient:
    def __init__(self, cfg: Optional[WheelMotorConfig] = None): ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def __enter__(self) -> "WheelMotorClient": ...
    def __exit__(self, *exc) -> None: ...

    def ping(self) -> bool:                                # sync, expects PONG
    def drive(self, wL_rad_s: float, wR_rad_s: float) -> None:
        """Fire-and-forget. Quantizes, applies deadzone & direction, clamps,
        writes one DRIVE line. Does NOT read a response."""
    def stop(self) -> bool:                                # sync, expects OK
```

Quantization & safety pipeline inside `drive(...)`:

```
wL_rad_s, wR_rad_s
   → wL_mrad = int(round(wL * 1000)) * direction_signs[0]
   → wR_mrad = int(round(wR * 1000)) * direction_signs[1]
   → clamp to ±max_wheel_mrad_s
   → if |wL| < deadzone and |wR| < deadzone: both ← 0
   → write "DRIVE %d %d\n" (or print if dry_run)
```

`disconnect()` MUST attempt one final `STOP` before closing the port,
swallowing any exception (mirrors `LevelingMotorClient.disconnect()`).

### 4.2 `Driving/drive_to.py`

Top-level structure:

```python
@dataclass
class SafetyConfig:
    lost_quiet_sec:  float = 0.5    # < this: silent HOLD
    lost_warn_sec:   float = 3.0    # logging only
    lost_abort_sec:  float = 3.0    # ABORT after this
    jump_factor:     float = 3.0    # outlier if jump > max_speed*dt*factor
    jump_outlier_max: int  = 3      # consecutive outliers before ABORT
    max_linear_vel:  float = 0.3    # m/s, matches ControllerConfig.max_speed


class SafetySupervisor:
    def __init__(self, cfg: SafetyConfig): ...
    def check(self, pose: Optional[Dict]) -> str:
        """Returns "OK" | "HOLD" | "ABORT". Updates internal state.
           Sets self.reason (str) when returning ABORT."""


def main() -> int:
    args   = parse_cli()
    ctrl   = DrivingController(ControllerConfig(dt=1.0/args.rate, ...))
    motor  = WheelMotorClient(WheelMotorConfig(port=args.port, dry_run=args.dry_run, verbose=args.verbose))
    loc    = OrbSlamLocalizer(LocalizerConfig())
    safety = SafetySupervisor(SafetyConfig())

    with loc, motor:                                    # __exit__ → disconnect() → STOP+close
        if not args.dry_run:
            assert motor.ping(), "OpenRB PING failed"
        if not loc.wait_for_tracking(timeout=30.0):
            print("[FAIL] SLAM did not reach tracking OK within 30s")
            return 2

        try:
            return _run_loop(args, loc, ctrl, motor, safety)
        finally:
            try: motor.drive(0.0, 0.0)                  # fire-and-forget zero stream
            except Exception: pass
            # NOTE: synchronous STOP is sent by motor.disconnect() via __exit__.
            # We don't call it here to avoid two consecutive blocking STOPs on a
            # potentially dead serial link.
```

`_run_loop(args, loc, ctrl, motor, safety) -> int` is the 15 Hz control
loop:

```python
def _run_loop(args, loc, ctrl, motor, safety) -> int:
    period = 1.0 / args.rate
    deadline = time.monotonic() + args.timeout
    last_log = 0.0
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        pose = loc.get_pose()
        action = safety.check(pose)
        if action == "ABORT":
            print(f"[ABORT] {safety.reason}")
            motor.drive(0.0, 0.0)               # immediate stop on supervisor abort
            return 2
        if action == "HOLD":
            motor.drive(0.0, 0.0)
        else:  # OK
            out = ctrl.compute(pose["x"], pose["y"], pose["theta"], args.x, args.y)
            if out["reached"]:
                motor.drive(0.0, 0.0)
                print(f"✓ reached @ dist={out['distance']:.3f}m")
                return 0
            motor.drive(out["wheel_omega_left"], out["wheel_omega_right"])
            if t0 - last_log >= 0.5:
                _log_status(pose, out); last_log = t0
        # fixed-period sleep: keeps 15 Hz even when SLAM I/O jitters
        sleep_for = max(0.0, period - (time.monotonic() - t0))
        time.sleep(sleep_for)
    motor.drive(0.0, 0.0)                       # immediate stop on timeout
    return 1   # timeout
```

Both `_run_loop`'s ABORT and timeout exits issue an explicit `motor.drive(0, 0)`
in addition to the `try/finally` zero-stop in `main()`. Defense in depth: the
firmware watchdog (200ms) is the floor; the in-loop zero-stops give immediate
0-velocity on every exit path; the `finally` block backs them up if any path
short-circuits.

`_log_status(pose, out)` emits a single line matching `pipeline.phase1_driving`'s
format: `[t]  pose=(x,y,θ°)  dist=...  v=...  ω_L/R=(...)`. Printed via
`print(...)` (stdout); `verbose=True` also routes serial send/recv lines
to stderr from `WheelMotorClient`.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | reached (`distance < goal_tolerance`) |
| 1 | timeout (`--timeout` exceeded with no reach) |
| 2 | SLAM failure (lost > `lost_abort_sec`, persistent pose jumps, or no initial tracking) |
| 130 | Ctrl-C (Python default for SIGINT) |

### 4.3 SafetySupervisor — state machine (Tier 1)

`check(pose)` is called once per frame. Internal state:
`last_ok_pose: Optional[(x, y, t)]`, `lost_since: Optional[float]`,
`consec_outliers: int`, `last_warn_at: float`.

```
INPUT: pose (None | dict with tracking_ok)
NOW = time.monotonic()

A) pose is None or not pose["tracking_ok"]:
     if lost_since is None:
         # Back-date to the last accepted pose's timestamp so the
         # lost-duration counts from when tracking was last known good,
         # not from this consecutive-lost streak's first frame. (Single
         # dropped frames after a long OK gap should still trigger the
         # warn window — the system has been "blind" for that whole gap.)
         lost_since = last_ok.t if last_ok is not None else NOW
     dur = NOW - lost_since
     if dur < lost_quiet_sec:                   return "HOLD"   # silent
     if dur < lost_quiet_sec + lost_warn_sec:
         if NOW - last_warn_at >= 0.5:
             log(f"[WARN] tracking lost {dur:.1f}s")
             last_warn_at = NOW
         return "HOLD"
     reason = f"tracking lost {dur:.1f}s (>= {lost_quiet_sec+lost_warn_sec}s)"
     return "ABORT"

B) tracking_ok=True:
     if last_ok_pose is not None:
         dt   = NOW - last_ok_pose.t
         jump = hypot(pose.x - last_ok_pose.x, pose.y - last_ok_pose.y)
         if dt < 1.0 and jump > max_linear_vel * dt * jump_factor:
             consec_outliers += 1
             if consec_outliers >= jump_outlier_max:
                 reason = f"pose jump x{jump_outlier_max} (last={jump:.2f}m in {dt*1000:.0f}ms)"
                 return "ABORT"
             return "HOLD"
     # accepted
     last_ok_pose    = (pose.x, pose.y, NOW)
     lost_since      = None
     consec_outliers = 0
     return "OK"
```

Notes:

- The `dt < 1.0` guard prevents the first frame after a long lost period
  from being misclassified as a jump (post-relocalization re-engagement
  ramp is not needed for v0 — the runner just rejects 3 consecutive
  jumps and moves on).
- `last_ok_pose` is reset only on accepted poses, so the jump comparison
  is always against a known-good reference, not against another outlier.

---

## 5. Out of scope (deferred Tiers)

For traceability — these were considered in brainstorming and intentionally
left out of v0:

- **Geofence** (Tier 2): abort if `‖pose - origin‖ > 10 m`. Cheap to add;
  defer until we see evidence of catastrophic SLAM scale faults in the
  field.
- **No-progress watchdog** (Tier 2): commanded `v > 0.05 m/s` but
  distance-to-goal stalls for 5 s. Useful for physical stalls (slip,
  obstacle); needs distinct tuning vs SLAM jitter.
- **Pose-freshness watchdog** (Tier 2): same `(x, y, θ)` returned for
  > 200 ms. `OrbSlamLocalizer` does not currently expose timestamps,
  would require a small upstream change.
- **Wheel-encoder dead reckoning** (Tier 3): no encoder feedback
  exists in current hardware; would require an OpenRB → Pi telemetry
  channel.
- **Confidence-based velocity scaling** (Tier 3):
  `ControllerConfig.enable_confidence_scaling` is already wired, but
  `OrbSlamLocalizer` exposes only a boolean `tracking_ok`. Plumbing a
  continuous confidence estimate is non-trivial.

---

## 6. CLI

```bash
# Required
python Driving/drive_to.py --x 3 --y 2

# Dry-run (no serial; verifies SLAM + controller + safety end-to-end)
python Driving/drive_to.py --x 3 --y 2 --dry-run

# Override defaults
python Driving/drive_to.py --x 3 --y 2 --port /dev/ttyACM1 --timeout 90 --rate 10 --verbose
```

| Flag | Default | Description |
|---|---|---|
| `--x` | (required) | target x in world frame [m] |
| `--y` | (required) | target y in world frame [m] |
| `--port` | `/dev/ttyACM0` | OpenRB serial port |
| `--baud` | `115200` | |
| `--rate` | `15` (Hz) | control loop rate; loop dt = 1/rate |
| `--timeout` | `60.0` (s) | overall Phase-1 timeout |
| `--dry-run` | False | skip serial open; print DRIVE/STOP/PING lines |
| `--verbose` | False | emit serial send/recv lines to stderr |
| `--wheel-diameter` | `0.10` (m) | wheel diameter — used by controller for v→ω conversion |
| `--wheel-base` | `0.30` (m) | distance between wheels — used by controller for ω→(ωL,ωR) split |
| `--goal-tolerance` | `0.3` (m) | distance to target below which counts as reached (exit 0) |

`--rate` is also passed to `ControllerConfig.dt = 1.0 / rate` so the PID
derivative/integral arithmetic uses the actual loop period. **Caveat**: the
PID gains in [Driving/controller.py](../../../Driving/controller.py)
(`kp_angular=2.5`, `ki_angular=0.1`, `kd_angular=0.3`) are tuned at the
default 15 Hz; large deviations from `--rate 15` may need gain retuning.
Default usage at 15 Hz is the tested path.

---

## 7. Test strategy

### 7.1 Unit — `WheelMotorClient` (dry-run, no hardware)

In `Driving/tests/test_wheel_motor.py`:

- `drive(0.0, 0.0)` → emits `DRIVE 0 0\n`.
- `drive(1.234, -2.345)` → quantizes to `DRIVE 1234 -2345\n`.
- `drive(0.001, -0.003)` → both inside deadzone → `DRIVE 0 0\n`.
- `drive(50.0, -50.0)` → clamps to `DRIVE 30000 -30000\n`.
- `direction_signs=(+1, -1)` flips right wheel sign.
- `stop()` returns True against fake `OK` response.
- `ping()` returns True against fake `PONG` response.

Verification uses `dry_run=True`, capturing stdout (not pyserial mocks),
matching the `LevelingMotorClient` test pattern.

### 7.2 Unit — `SafetySupervisor`

In `Driving/tests/test_safety_supervisor.py`:

- All `tracking_ok=True`, no jumps → all `OK`.
- `tracking_ok=False` for 0.3 s → `HOLD` (silent).
- `tracking_ok=False` for 1.0 s → `HOLD` (warn line printed; captured via pytest's `capsys`, since the supervisor uses `print` not `logging`).
- `tracking_ok=False` for 4.0 s → eventually `ABORT` with reason set.
- Recovery: 0.3 s lost → 1 s tracking → resumes `OK`, counters reset.
- Single jump (one frame `pose+1m`) → `HOLD`, next normal frame → `OK`,
  counter reset.
- 3 consecutive jumps → `ABORT`.

Time is injected via a `now()` callable on the supervisor so tests can
advance the clock deterministically.

### 7.3 Integration — dry-run with real SLAM

Manual: run `python Driving/drive_to.py --x 1 --y 0 --dry-run` on Pi5
with RealSense + ORB-SLAM3 actually streaming. Verify:

1. Tracking acquires within 30 s, loop starts.
2. Reasonable `DRIVE wL wR` lines appear in stdout at ~15 Hz.
3. Covering the camera triggers `[WARN] tracking lost ...` after 0.5 s,
   and `[ABORT]` after ~3.5 s with exit code 2.
4. Ctrl-C produces a final `DRIVE 0 0\n` and `STOP\n` line.

### 7.4 Bench — real serial, motors disconnected

Connect OpenRB but leave motor power off (or torque off). With matching
firmware:

1. `PING` succeeds.
2. `DRIVE` lines stream.
3. Pulling the USB cable mid-run triggers the firmware watchdog within
   200 ms (visible if a log line is wired up firmware-side).

### 7.5 Field — full real run

Out of scope of this design's verification, but the runner is the artifact
that enables it.

---

## 8. Migration / follow-up

After this PR:

- **Firmware PR**: implement `DRIVE`/`STOP`/`PING` + 200 ms watchdog in
  the OpenRB sketch (parallel to
  [openrb_sketch_reference.ino](../../../LevelingPlatform/openrb_sketch_reference.ino)).
- **Pipeline PR**: replace
  [pipeline.py:171](../../../pipeline.py#L171)
  `RealRobot.send_wheel_omegas()` print-stub with a `WheelMotorClient`
  call. One-line change once `wheel_motor.py` exists.
- **SW_ARCHITECTURE.md update**: extend §4 with the wheel protocol, and
  remove the "TODO: serial 패킷으로 OpenRB 송신" item from §9.
