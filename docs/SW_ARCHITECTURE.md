# 2026 ME Capstone Design — SW Pipeline Architecture

> SW pipeline design document for a system that autonomously drives to and strikes an oscillating bell (height ~3 m, vertical oscillation) twice.

> ⚠️ **Implementation status — Phase 1 localization method changed (SLAM → Visual Servo).**
> Phase 1 in §3·§4 of this document was originally designed as **absolute-coordinate
> driving based on ORB-SLAM3 self-pose**, but was **abandoned** due to the **runtime
> instability of the Raspberry Pi 5 + ORB-SLAM combination** (real-time tracking
> dropouts, drift, CPU saturation). The actual implementation drives directly beneath
> the bell using **YOLO bbox + depth-based active-tilt visual servoing** without SLAM
> ([run_phase1_visual_servo.py](../run_phase1_visual_servo.py),
> [Driving/visual_servo_controller.py](../Driving/visual_servo_controller.py)).
> The SLAM/VIO code ([perception/vio/](../perception/vio/), `Pangolin`·`librealsense`
> submodules) is **preserved for reference** but is not used in the current pipeline.

---

## 1. Mission Spec

| Item | Value |
|---|---|
| Goal | **Strike** a vertically oscillating bell **twice** |
| Bell position | Approximately 3 m above the ground |
| Start position | Random point **2 ~ 4 m** away from the bell |
| Autonomy | **Fully automated end to end** (no human intervention) |
| Primary sensor | **Intel RealSense D435i** (RGB + Depth + IMU) |

---

## 2. Hardware → SW Mapping

```
┌───────────────────────────────────────────────────────────┐
│                  Mobile Platform (Tank/Diff)              │
│  ┌─────────┐   ┌──────────────────┐   ┌────────────────┐ │
│  │ 2 wheels│ + │ 3-DOF Leveling   │ + │ Flywheel +     │ │
│  │ (drive) │   │ Platform (3-RRS) │   │ Launcher       │ │
│  └─────────┘   └──────────────────┘   └────────────────┘ │
│       ▲                ▲                      ▲           │
│       │                │                      │           │
│   Phase 1          Phase 2 fine          Phase 2 strike   │
│   coarse driving   aiming (IK)           (firing cmd)     │
└───────────────────────────────────────────────────────────┘
                         ▲
                         │
              ┌──────────┴──────────┐
              │ RealSense D435i     │
              │ (RGB + Depth + IMU) │
              └─────────────────────┘
```

Strategy summary: **Coarsely move near the bell with the mobile base → fine-aim with the leveling platform → strike with the flywheel launcher.**

---

## 3. Perception · High-Level Control 2-Phase Structure

The entire autonomous sequence is split into two phases. The state transition condition between phases is **reaching the Phase 1 goal (within goal tolerance)**.

```
   ┌───────────────────────────────────────────────────────────────┐
   │  PHASE 1: DRIVING                                             │
   │                                                               │
   │  [YOLO26n target detect]  ──▶  multi-frame avg (x, y)         │
   │            │                                                  │
   │            ▼                                                  │
   │  [ORB-SLAM3 self-pose] ──▶ controller ──▶ wheel ω_L, ω_R     │
   │                                                               │
   │  Stop condition: |pose_xy - target_xy| < goal_tolerance       │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  PHASE 2: AIMING                                              │
   │                                                               │
   │  [Camera tilt 90°]  (custom mount)                            │
   │            │                                                  │
   │            ▼                                                  │
   │  [YOLO detect bell] ──▶ depth deproject                       │
   │                          │                                    │
   │                          ▼                                    │
   │  3D vector (plate center → bell)                              │
   │            │                                                  │
   │            ▼                                                  │
   │  [3-RRS IK] ──▶ θ1, θ2, θ3 motor angles ──▶ FIRE             │
   │                                                               │
   │  Repeat: 2 strikes                                           │
   └───────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1 — Driving

### 4.1 Purpose
Autonomously drive the mobile base from the start point to near the bell base (its ground-projected coordinate). Since precise aiming is handled by Phase 2, the stop condition is **reaching within goal tolerance**.

### 4.2 Pipeline

1. **Target localization (one-shot, just before departure)**
   - Detect the bell with YOLO26n from the camera's front-facing pose
   - Remove noise by **averaging detection results over multiple frames** → finalize the world-frame target (x, y) coordinate
   - Afterwards, do not re-detect during driving; use this (x, y) as a fixed target

2. **Self-localization (real-time)**
   - **ORB-SLAM3** RGB-D + Pi-optimized (424×240@15fps, nFeatures=500, viewer OFF) is the default
   - Production API: `OrbSlamLocalizer` in [perception/vio/orbslam_localizer.py](../perception/vio/orbslam_localizer.py) (context manager + `get_pose()` → world-frame `(x, y, θ)`)
   - World coordinate frame: `world_x = camera_z`, `world_y = -camera_x`, `theta = yaw (CCW+)` — the camera's starting pose = origin

3. **Control (real-time)**
   - Simple control logic: linear velocity based on distance to target + PID angular velocity based on heading error
   - Distribute the result into left/right wheel angular velocities (ω_L, ω_R) and transmit every step
   - For the serial protocol, see `SerialCommandSim` in [Driving/simulation.py](../Driving/simulation.py) (Pi5 → OpenRB)

### 4.3 Related Modules

| Role | File |
|---|---|
| YOLO detector skeleton | [perception/detection/detector.py](../perception/detection/detector.py) |
| 2D bbox + depth → 3D coordinate | [perception/detection/position_estimator.py](../perception/detection/position_estimator.py) |
| **Dummy target provider (before YOLO training)** | **[perception/detection/dummy_detector.py](../perception/detection/dummy_detector.py)** |
| **Integrated pipeline orchestrator** | **[pipeline.py](../pipeline.py)** — Phase1↔Phase2 transition + sim/real backend |
| **Localization module (production)** | **[perception/vio/orbslam_localizer.py](../perception/vio/orbslam_localizer.py)** — `OrbSlamLocalizer` → world-frame (x, y, θ) |
| ORB-SLAM3 GUI test runner | [perception/vio/orbslam_runner.py](../perception/vio/orbslam_runner.py) (`--gui`) |
| Camera wrapper | [perception/common/realsense_wrapper.py](../perception/common/realsense_wrapper.py) |
| Integrated entry point | [perception/main.py](../perception/main.py) |
| **High-level control module (production)** | **[Driving/controller.py](../Driving/controller.py)** — pose+target → (v, ω, ω_L, ω_R) |
| **Wheel motor serial client (production)** | **[Driving/wheel_motor.py](../Driving/wheel_motor.py)** — Pi → OpenRB-150 (ASCII protocol, fire-and-forget DRIVE) |
| **Phase-1 only driving runner** | **[Driving/drive_to.py](../Driving/drive_to.py)** — target (x, y) → ORB-SLAM3 → controller → wheel motor |
| **Leveling motor serial client** | **[LevelingPlatform/leveling_motor.py](../LevelingPlatform/leveling_motor.py)** — Pi → OpenRB-150 (ASCII protocol) |
| OpenRB-side reference sketch | [LevelingPlatform/openrb_sketch_reference.ino](../LevelingPlatform/openrb_sketch_reference.ino) |
| Driving simulator (incl. SLAM error model) | [Driving/simulation.py](../Driving/simulation.py) |
| Camera / detection config | [perception/config.py](../perception/config.py) |
| Pi5 deploy guide | [perception/DEPLOY_PI5.md](../perception/DEPLOY_PI5.md) |
| **Phase 1 driver Protocol + SLAM implementation** | **[Driving/phase1_driver.py](../Driving/phase1_driver.py)** |
| **Phase 1 visual-servo controller** | **[Driving/visual_servo_controller.py](../Driving/visual_servo_controller.py)** |
| **Phase 1 visual-servo driver** | **[Driving/visual_servo_driver.py](../Driving/visual_servo_driver.py)** |

### 4.4 Simulator (sanity check)

[Driving/simulation.py](../Driving/simulation.py) is a 2D tank simulator for validating control, communication, and SLAM error behavior before building the actual rover. It includes the following components:

- `TankVehicle` — differential drive model
- `DisturbanceModel` — grass slip + Gaussian disturbance
- `SLAMModel` — SLAM measurement noise + cumulative drift + relocalization failure
- `SLAMFilter` — outlier rejection + confidence-based deceleration
- `SerialCommandSim` — Pi → OpenRB serial (rate limit, deadzone, int16 quantization)
- `NavigationController` — distance-proportional linear velocity + PID angular velocity controller

### 4.5 Phase 1 driver selection

Select one of two drivers via the CLI `--drive-mode {slam,visual_servo}`.

- `slam` (default): existing ORB-SLAM3 pose → DrivingController → wheel ω. Requires world-frame localization.
- `visual_servo`: moves directly beneath the bell using only YOLO bbox + depth + active camera tilt servoing. For SLAM-unstable environments.

The `visual_servo` driver runs a **tilt sweep bootstrap** before entering the main loop (`VisualServoPhase1Driver.acquire_initial_tilt`). It raises 0° → 90° in 5° steps, attempting detection at each step, and adopts the tilt at which the bell is first detected as the initial angle. If detection fails at all tilts because the bell is outside the horizontal FOV, it starts at a fallback of 45° and the FSM `SEARCH` rotates the chassis to reacquire. Thanks to this bootstrap, both close targets (requiring ~70°+ tilt) and far targets (~25° tilt) are handled automatically.

---

## 5. Phase 2 — Aiming & Strike

### 5.1 Purpose
Having arrived near the bell, the **leveling platform aligns the launch direction toward the bell** and the flywheel launcher strikes the bell.

### 5.2 Pipeline

1. **Camera tilt to 90° (vertical)**
   - Since the bell is at ~3 m height, it goes above the field of view of the front-facing camera
   - A custom tilt structure **rotates the camera 90° upward** so the bell comes into the center of the field of view

2. **Bell 3D vector estimation**
   - YOLO detection → bbox → depth deprojection to obtain (X, Y, Z) in the camera coordinate frame
   - Apply the known extrinsic transform between the camera ↔ leveling platform center to obtain the **3D vector from the platform center → bell**
   - **Actual implementation**: [perception/detection/phase2_target.py](../perception/detection/phase2_target.py) `Phase2TargetEstimator` (single frame) + `RealPhase2TargetProvider` (1-second measurement window, per-axis median). Called just before each shot.
   - **Camera→Plate extrinsic**: the lens is located at `(+0.20, 0, -0.10) m` relative to the plate center, 90° pitch-up. The rotation matrix + sign options are exposed as dataclass fields of `CameraToPlateExtrinsic`.

3. **Inverse Kinematics → motor angles**
   - Call `LevelingIK(cfg).aim_at(target_xyz)` in [LevelingPlatform/leveling_ik.py](../LevelingPlatform/leveling_ik.py)
   - Use `angles_steps` (encoder steps) or `angles_rad` from the returned dict as motor commands
   - If `ok=False`, length is unreachable or ball limit is exceeded → realign the base and retry

4. **Fire ×2**
   - Since the oscillating bell must be struck twice, repeat the **strike → re-estimate → re-aim → strike** cycle
   - Because the bell oscillates vertically, the 3D vector must be refreshed just before each strike

### 5.3 3-RRS Leveling Platform — key `LevelingConfig` fields ([LevelingPlatform/leveling_ik.py](../LevelingPlatform/leveling_ik.py))

| Field | Default | Description |
|---|---|---|
| `Rb` | 0.10 m | Base pivot radius |
| `La` | 0.04 m | Crank length |
| `Lc` | 0.12 m | Coupler length |
| `Rp` (derived) | `Rb - La` | Plate joint radius (forced to home pose) |
| `H0` (derived) | `Lc` | Nominal plate center height |
| `motor_phis_deg` | (0, 120, 240) | Motor azimuth |
| `motor_steps` | 4096 | Encoder counts/revolution |
| `ball_max_deg` | 30° | P-side ball joint angle limit |
| `quantize` | True | Whether to round to encoder steps |

API:

```python
from leveling_ik import LevelingIK, LevelingConfig
ik = LevelingIK(LevelingConfig())
out = ik.aim_at(target_xyz)            # or ik.aim_normal(unit_vec)
# out: {angles_deg, angles_rad, angles_steps, ok, ball_deg, c_shift_m, normal}
```

If `ok=False`, the target is unreachable or the ball joint limit is exceeded → the upper layer must slightly realign the mobile base and retry.

---

## 6. Key Coordinate Frames and Alignment

| Frame | Definition | Used in |
|---|---|---|
| **World** | start position/pose = origin | Phase 1 SLAM, target (x, y) |
| **Camera** | RealSense optical frame | immediately after YOLO + depth deprojection |
| **Plate** | leveling plate center (`(0, 0, H0)`) | IK input (target 3D point) |

The Phase 2 IK input target must be a 3D point **relative to the plate center**. Therefore, the Camera→Plate extrinsic transform (including 90° tilt + mount offset) must be calibrated in advance and multiplied with the YOLO+depth result to convert it into the plate coordinate frame.

The Camera→Plate extrinsic transform is encapsulated in `CameraToPlateExtrinsic` in [perception/detection/phase2_target.py](../perception/detection/phase2_target.py) with the following defaults:

- `t_x_m = +0.20`, `t_z_m = -0.10` (lens at (+0.20, 0, -0.10) m relative to plate center)
- `image_right_sign = -1`, `image_down_sign = +1` (natural mount, camera roll 0°)

For the calibration procedure, see §6 of this document (Camera→Plate extrinsic transform).

---

## 7. Execution Entry Points

```bash
# Data collection (for YOLO training)
python perception/main.py capture

# ORB-SLAM3 (default = Pi + no-IMU + headless production module)
python perception/main.py orbslam
# When you want to view it with the legacy GUI test runner
python perception/main.py orbslam --gui

# Detection + 3D position estimation (to be implemented)
python perception/main.py detect

# Driving simulation (single / animate / Monte Carlo)
python Driving/simulation.py --mode single
python Driving/simulation.py --mode animate
python Driving/simulation.py --mode monte_carlo --runs 100

# High-level control module standalone run (single step output)
python Driving/controller.py --x 0 --y 0 --th 0 --tx 3 --ty 2 \
                             --wheel_d 0.10 --wheel_base 0.30

# Leveling platform IK CLI
python LevelingPlatform/leveling_ik.py --target 0.10 0.00 3.0

# Phase-1 only driving runner (standalone, for dev/experimentation)
python Driving/drive_to.py --x 3 --y 2                          # real serial + ORB-SLAM3
python Driving/drive_to.py --x 3 --y 2 --dry-run --verbose      # serial not connected, only prints TX lines

# Integrated pipeline — based on dummy detection (pre-YOLO-training stage)
python3 pipeline.py                              # default sim mode
python3 pipeline.py --phase1-x 4 --phase1-y 3    # change target position
python3 pipeline.py --phase2-jitter 0.10         # bell oscillation amplitude ±10cm
python3 pipeline.py --mode real                  # Pi + camera + ORB-SLAM3

# Phase 1 visual-servo driving mode (SLAM-free)
python3 pipeline.py --drive-mode visual_servo --phase1-x 3 --phase1-y 2
```

---

## 8. Integrated Pipeline — `pipeline.py`

[pipeline.py](../pipeline.py) is an integrated orchestrator that ties the 4 production modules into a single execution flow.

```
┌──────────────────────────────────────────────────────────────────┐
│  pipeline.py                                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  CapstonePipeline                                          │  │
│  │     ├─ phase1_driving()                                    │  │
│  │     │     pose ← Robot.get_pose()                          │  │
│  │     │     cmd  ← DrivingController.compute(pose, target)   │  │
│  │     │     Robot.send_wheel_omegas(ω_L, ω_R)                │  │
│  │     │     until cmd.reached                                │  │
│  │     │                                                      │  │
│  │     └─ phase2_aiming()  ×N strikes                         │  │
│  │           Robot.tilt_camera(90°)                           │  │
│  │           target ← DummyTargetProvider.get_phase2_target() │  │
│  │           ik_out ← LevelingIK.aim_at(target)               │  │
│  │           Robot.send_leveling_angles(ik_out)               │  │
│  │           Robot.fire()                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│             │                            │                       │
│             ▼                            ▼                       │
│        SimulatedRobot              RealRobot                     │
│        (pure Python                (OrbSlamLocalizer +           │
│         diff-drive forward kin)    motor stub)                  │
└──────────────────────────────────────────────────────────────────┘
```

### Backends

| `--mode sim` (default) | `--mode real` |
|---|---|
| Runs anywhere without camera/motor | Requires RealSense + ORB-SLAM3 (Pi5) |
| `SimulatedRobot` — instantly integrates pose via differential drive forward kinematics | `RealRobot` — pose via `OrbSlamLocalizer`, motors are stub output |
| For module integration · Phase transition validation | For real SLAM localization + motor stub validation |

### Phase Transition Conditions

- **Phase 1 → Phase 2**: `out["reached"] == True` from `DrivingController.compute()` (i.e. `distance < goal_tolerance`)
- **Phase 2 end**: configured `num_strikes` strikes completed (default 2)
- **Re-estimate bell position each strike**: `DummyTargetProvider.get_phase2_target()` applies jitter to z to emulate the oscillating bell

### Replacing Dummy → real detection

Once YOLO training is done, the following can be plugged in place of `DummyTargetProvider` in [pipeline.py](../pipeline.py):

- **Phase 1**: average `TargetDetector.detect()` results over N frames → compute world-frame (x, y) with `PositionEstimator.estimate()`
- **Phase 2**: after a 90° camera tilt, compute plate-frame (x, y, z) with the same chain (applying the Camera→Plate extrinsic transform)

As long as you wrap `DummyTargetProvider` with the same signature (`get_phase1_target() → (x, y)`, `get_phase2_target() → (x, y, z)`), the pipeline body needs no modification.

---

## 9. Unimplemented / TODO

- [ ] [perception/detection/detector.py](../perception/detection/detector.py) — YOLO model loading / inference implementation (currently NotImplementedError)
- [ ] Phase 1 multi-frame averaging + Phase 2 camera→plate transform (the Phase 2 part is done: [phase2_target.py](../perception/detection/phase2_target.py))
- [ ] [perception/main.py](../perception/main.py) `detect` mode — Detection + PositionEstimator integration
- [x] Document the Camera ↔ Plate extrinsic calibration procedure (§6 of this document)
- [ ] Phase 2 bench test: finalize extrinsic calibration signs + verify std_z < 5 cm over a 1-second static bell measurement
- [ ] Optimize concurrent use of the Phase 2 ↔ Phase 1 camera stream (currently serial: SLAM stop → camera reopen)
- [x] Wheel serial driver — [Driving/wheel_motor.py](../Driving/wheel_motor.py) (ASCII line protocol)
- [ ] Replace the `RealRobot.send_wheel_omegas` stub in [pipeline.py](../pipeline.py) → with `WheelMotorClient` (separate PR)
- [ ] Add `DRIVE`/`STOP`/`PING` handlers + 200 ms watchdog to the OpenRB firmware (separate PR)
- [x] Camera 90° tilt servo command  ← partially done via TILT_ASYNC v1.1, sync TILT is separate
- [ ] Serial protocol field test — verify round-trip between the ASCII protocol of `wheel_motor.py` and the OpenRB firmware
