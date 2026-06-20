# Pi5 ↔ OpenRB-150 Integrated Serial Communication Protocol (v1.1)

> v1.1 — Added `TILT_ASYNC` command (15Hz tilt streaming for visual-servo driving mode).

> A single OpenRB-150 board controls all actuators (wheel DC ×2, leveling DXL ×3, camera tilt DXL ×1, loader DXL ×1, flywheel T-motor ×2) and communicates with the Pi5 over a single strand of USB CDC serial. This document defines the line-based ASCII protocol exchanged over that serial link.
>
> Related docs: [SW_ARCHITECTURE.md](SW_ARCHITECTURE.md), [Driving/wheel_motor.py](../Driving/wheel_motor.py), [LevelingPlatform/leveling_motor.py](../LevelingPlatform/leveling_motor.py), [LevelingPlatform/openrb_sketch_reference.ino](../LevelingPlatform/openrb_sketch_reference.ino)

---

## 1. Physical Channel / Framing

- USB CDC, **115200 baud, 8N1**, 7-bit ASCII
- **Line-based**, `\n` terminated, `\r` ignored, line max 64 bytes
- Pi → OpenRB commands are a **single line**, OpenRB → Pi responses are also a **single line** (for sync commands only)
- Only `DRIVE` is fire-and-forget (no response); all other commands are sync (single-line response)
- Line length exceeded → line discarded + `ERR OVERFLOW`

## 2. Actuator Topology / Dynamixel ID Assignment

| ID | Actuator | Location |
|---|---|---|
| 1, 2, 3 | Leveling platform (3-RRS) | DXL TTL bus |
| 4 | Camera tilt | DXL TTL bus |
| 5 | Loader (feeds 1 round) | DXL TTL bus |
| — | Wheel DC ×2 | OpenRB GPIO + external H-bridge |
| — | T-motor ×2 (flywheel) | OpenRB PWM (or CAN) |

A total of 5 Dynamixels are chained on a single TTL daisy chain. The DC wheel motors and T-motor flywheels are controlled separately via the OpenRB's general output pins.

## 3. Phase Policy / Watchdog

- **Phase-less coexistence**: the firmware operates as a simple dispatcher with no notion of phase. It always accepts every command.
- **DRIVE watchdog**: if the next `DRIVE` does not arrive within **200 ms** after the last `DRIVE` line was received, the firmware forces both wheels to 0. This guarantees a safe stop on SLAM dropout, Pi-side hang, or USB disconnection.
- **No watchdog for other commands**. If rotation was started with `SPIN <rpm> <rpm>`, it is maintained until `SPIN 0 0` or `STOP` arrives.

## 4. Command Table (Pi → OpenRB)

| Command | Args | Response | Sync | Unit · Range |
|---|---|---|---|---|
| `PING` | — | `PONG` | sync | Health check |
| `STATUS` | — | `S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>` | sync | Telemetry |
| `STOP` | — | `OK` | sync | All-Stop (wheels 0, T-motor 0, loader stop, DXL holding) |
| `DRIVE` | `<wL> <wR>` | (none) | f&f | signed int **mrad/s**, ±30000, deadzone 5, **200 ms watchdog** |
| `AIM` | `<s1> <s2> <s3>` | `OK` \| `ERR <reason>` | sync | DXL step ±2047 (waits for motion complete, max 4 s) |
| `HOME` | — | `OK` \| `ERR <reason>` | sync | Leveling ID 1·2·3 → 0,0,0 |
| `TILT` | `<s4>` | `OK` \| `ERR <reason>` | sync | DXL step ±2047, **positive = camera up** (waits for motion complete) |
| `TILT_ASYNC` | `<s4>` | (none) | f&f | DXL step ±2047 (**positive = up**), **200 ms watchdog → hold current position** |
| `SPIN` | `<rpmT> <rpmB>` | `OK` \| `ERR <reason>` | sync(immediate) | unsigned int rpm, 0..max\_rpm (does not wait to reach) |
| `LOAD` | — | `OK` \| `ERR <reason>` | sync | Loader (ID 5) rotates one cycle then OK |
| `STRIKE` | `<rpm> <hold_ms>` | `OK` \| `ERR <reason>` | sync | Convenience: `SPIN rpm rpm` → `delay(hold_ms)` → `LOAD` → `SPIN 0 0` |

### 4.1 Unit Consistency

- **Wheel speed**: signed int **mrad/s** (rad/s × 1000). Positive = forward; sign mapping is corrected via `WheelMotorConfig.direction_signs`.
- **DXL step**: signed int absolute position -2048..+2047. Home offset conversion is the firmware's responsibility.
- **T-motor RPM**: unsigned int. A negative value yields `ERR RANGE`. If bidirectional is needed, extend to a signed domain in v2.
- **Time**: integer in ms.

### 4.2 Error Codes

| Code | Meaning |
|---|---|
| `ERR PARSE` | Line parsing failed (insufficient argument count, non-numeric characters, etc.) |
| `ERR RANGE` | Argument out of allowed range (step, RPM, mrad/s) |
| `ERR HW` | Motor communication failure (no DXL response, T-motor PWM output failure, etc.) |
| `ERR TIMEOUT` | Motion-complete wait exceeded 4 s |
| `ERR OVERFLOW` | Line length exceeded 64 bytes |
| `ERR BUSY` | New command received while the previous sync command's motion had not finished |

## 5. STATUS Response Format

```
S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>
   │   │    └────── DXL step (signed) ─────┘   └─ T-motor rpm ┘    │
   └ wheel mrad/s ┘                                             bitmask
```

**flags bits**:

| bit | Meaning |
|---|---|
| 0 | wheel watchdog tripped (no `DRIVE` within the last 200 ms) |
| 1 | leveling moving (any of ID 1·2·3) |
| 2 | tilt moving (ID 4) |
| 3 | loader moving (ID 5) |
| 4 | flywheel spinning (rpm > 100) |
| 5 | error latched (last `ERR …` not reset) |
| 6 | leveling homed (`HOME` succeeded at least once) |
| 7 | estop active (last command was `STOP`) |

On receiving `PING` or the next valid sync command, bit 5 (error latched) is reset.

## 6. Standard Sequences

### Phase 1 — Visual-servo tilt streaming (15 Hz fire-and-forget)

```
Pi → TILT_ASYNC 800      | (no reply)     # positive = camera up
Pi → TILT_ASYNC 812      | (no reply)
...                      |
(stream stops > 200 ms)  | (firmware: getPresentPosition(4) → setGoalPosition(4, readback))
```

Use `TILT_ASYNC` only for visual-servo Phase 1 streaming. For Phase 2 home/aim,
use sync `TILT` so the caller knows when the camera has settled.

Coexistence: `TILT` and `TILT_ASYNC` both write the same `goal_position` register.
sync `TILT` is polled to motion-complete by the firmware; `TILT_ASYNC` arriving
during a sync `TILT` window returns `ERR BUSY` and is dropped. Sending sync `TILT`
also refreshes the `TILT_ASYNC` watchdog timer so the camera does not snap to a
stale hold target.

### Phase 1 — Driving (15 Hz fire-and-forget)

```
Pi → PING               | OpenRB → PONG
Pi → DRIVE 1234 1180    | (no reply)
Pi → DRIVE 1230 1175    | (no reply)
...
Pi → STOP               | OpenRB → OK     # safe stop just before phase transition
```

### Phase 2 — Aiming + 2 strikes (all sync)

```
Pi → TILT 1024          | OpenRB → OK     # camera 90° up (positive = up)
Pi → AIM 100 -50 200    | OpenRB → OK     # 1st aim
Pi → SPIN 8000 8000     | OpenRB → OK
   (Pi side waits ~1 s for spin-up)
Pi → LOAD               | OpenRB → OK     # fire 1st round
   (Pi re-estimates bell position + re-aims)
Pi → AIM 110 -45 195    | OpenRB → OK     # 2nd aim
Pi → LOAD               | OpenRB → OK     # fire 2nd round
Pi → SPIN 0 0           | OpenRB → OK
Pi → TILT 0             | OpenRB → OK     # return camera to home
```

Or the convenience command:

```
Pi → STRIKE 8000 1000   | OpenRB → OK     # single call: spin-up + load + spin-down
```

## 7. Pi-side Client Structure (proposal)

The existing [Driving/wheel_motor.py](../Driving/wheel_motor.py) and [LevelingPlatform/leveling_motor.py](../LevelingPlatform/leveling_motor.py) each assume an independent serial instance. When integrating into a single OpenRB, **all facades must share one serial handle**.

```
OpenRBClient                       # single serial owner, line-based send/recv
  ├─ WheelMotorClient(self)        # drive(), stop(), ping()    — keeps existing API
  ├─ LevelingMotorClient(self)     # aim(), home(), status()
  ├─ TiltClient(self)              # tilt(step)
  └─ LauncherClient(self)          # spin(t, b), load(), strike(rpm, hold)
```

Each facade does not own the serial directly but only calls `OpenRBClient.send_line(line, expect_reply)`. For backward compatibility, the standalone `WheelMotorClient(cfg)` constructor form is also kept, but internally it auto-creates an `OpenRBClient`.

Thanks to the All-Stop semantics of `STOP`, any facade can bring all actuators to a safe state in a single line. The emergency handler only needs to call `OpenRBClient.stop()` once.

## 8. OpenRB Firmware Responsibilities Summary

- Line parser + dispatcher (reference: [openrb_sketch_reference.ino](../LevelingPlatform/openrb_sketch_reference.ino))
- `DRIVE` 200 ms watchdog: track last receive time, force both wheel PWMs to 0 on expiry
- DC motor PWM conversion (mrad/s → duty); encoder PID is optional
- Manage the 5 DXL IDs, motion-complete polling for `AIM`·`HOME`·`TILT`·`LOAD`
- T-motor output (PWM or CAN abstraction), `SPIN` returns OK immediately (no wait to reach)
- Error latch + `STATUS` flag update
- `STOP` priority: immediately terminate all motion + flywheel 0; DXL stays holding (torque kept ON)

## 9. Versioning · Extension Policy

- This document is **v1**; when adding new commands, the argument/response formats of existing commands are not broken.
- Bidirectional T-motor, CRC, sequence numbers, asynchronous telemetry push, etc. are considered separately in v2 and beyond.
- If firmware version checking is needed, extending the `PING` response to `PONG <ver>` is the lightest path.
