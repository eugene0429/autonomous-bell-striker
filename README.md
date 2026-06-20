# Autonomous Bell-Striker Robot

> 2026 KAIST ME Capstone Design — a mobile robot system that **autonomously drives** toward a
> vertically oscillating bell (~3 m above the ground), **fine-aims** with a leveling
> platform, and **strikes it twice** with a flywheel launcher.

🏆 **1st place — 2026 KAIST ME Capstone Design** (Team **team-won**).

The entire process is automated end to end with no human intervention. The primary sensor
is a single **Intel RealSense D435i** (RGB + Depth + IMU). Every actuator except the
flywheel is a **Dynamixel** servo on a single TTL **daisy chain** — 2 drive wheels (XC430,
velocity mode), 3 leveling joints, 1 camera tilt, and 1 loader (7 servos total) — while the
2 flywheel **T-motors** run on PWM ESCs. All of them are controlled by a single
**OpenRB-150** board.

---

## Demo

https://github.com/user-attachments/assets/267859db-94c6-4d44-a497-43313fc8f1da

---

## System Overview

```
Coarsely move near the bell with the mobile base  (Phase 1: Driving)
        → Fine-aim with the leveling platform      (Phase 2: Aiming, 3-RRS IK)
        → Strike with the flywheel launcher ×2      (Phase 2: Strike)
```

![Hardware schematic](https://github.com/user-attachments/assets/3c12ae12-a674-4f7b-8504-1693d8512126)

| Component | Role |
|---|---|
| 2 wheels (Dynamixel diff drive) | Phase 1 coarse driving |
| 3-DOF Leveling Platform (3-RRS) | Phase 2 fine aiming (IK) |
| Flywheel + Launcher | Phase 2 strike |
| Raspberry Pi 5 | Perception · high-level control (YOLO, pipeline) |
| OpenRB-150 | Real-time control of all actuators (firmware) |

For the detailed design see [docs/SW_ARCHITECTURE.md](docs/SW_ARCHITECTURE.md); for the Pi5 ↔ OpenRB serial spec see
[docs/COMMUNICATION_PROTOCOL.md](docs/COMMUNICATION_PROTOCOL.md).

> **Phase 1 localization method changed (SLAM → Visual Servo).** The initial design used
> ORB-SLAM3-based absolute-coordinate driving, but it was abandoned due to the **runtime
> instability of the Pi 5 + ORB-SLAM combination**, switching to SLAM-free
> **YOLO + depth active-tilt visual servoing**. The SLAM/VIO code
> ([perception/vio/](perception/vio/), `Pangolin`·`librealsense` submodules) is preserved
> for reference only (unused in the current pipeline).

---

## How It Works

### Phase 1 — Driving (YOLO + depth visual servo)

The mobile base coarsely drives toward the bell using SLAM-free visual servoing: YOLO
detects the bell, depth + active camera tilt estimate its position, and the diff-drive
wheels close the loop.

![Phase 1 visual servo](https://github.com/user-attachments/assets/58a46286-225e-4add-93af-0a2dc42129a5)
![Phase 1 visual servo](https://github.com/user-attachments/assets/613b813f-aa24-4701-b39e-f028efb1fa82)

### Phase 2 — Aiming & Strike (3-RRS leveling platform + flywheel)

Once near the bell, the 3-RRS leveling platform fine-aims with inverse kinematics
(center-aim — targeting the center of the bell's vertical oscillation), then the
flywheel launcher strikes twice.

![Phase 2 aiming and strike](https://github.com/user-attachments/assets/7dc9e534-1b8b-4b0f-a8ee-525268ca57b2)

---

## Perception — Bell Detection (YOLO26n → Hailo HEF, ~60 FPS)

The bell detector is a **1-class YOLO26n** model fine-tuned from the pretrained `yolo26n.pt`.

**Training data.** ~2,000 real images captured in the target environment, then expanded
with offline [albumentations](perception/training/augment_dataset.py) augmentation —
hue/saturation/**value** shift, brightness/contrast, scale, rotation, flip, and Gaussian
noise. The aggressive value/brightness augmentation makes detection **robust across a wide
range of lighting conditions**.

```bash
# Augment the train split (×5 copies per original), then fine-tune
python -m perception.training.augment_dataset --multiplier 5
python -m perception.training.train --epochs 150 --imgsz 640
```

**On-device inference (~60 FPS).** The trained `.pt` is exported to ONNX
(`opset=11, imgsz=640, nms=False`) and compiled to a Hailo-8L `.hef`, running on the
**Pi 5 + AI Hat+ (Hailo-8L)** at **~60 FPS** — fast enough to close the visual-servo loop
in real time. Full conversion procedure:
[perception/detection/HAILO_HEF_CONVERT.md](perception/detection/HAILO_HEF_CONVERT.md).

---

## Repository Structure

| Path | Description |
|---|---|
| [pipeline.py](pipeline.py) | **Integrated orchestrator** — runs Phase 1 → Phase 2 consecutively in a single process (sharing a single RealSense + single OpenRB) |
| [run_phase1_visual_servo.py](run_phase1_visual_servo.py) | Phase 1 standalone runner (visual-servo driving) |
| [run_phase2_aiming.py](run_phase2_aiming.py) | Phase 2 standalone runner (aiming & strike, center-aim) |
| [run_center_depth_probe.py](run_center_depth_probe.py) | Validation tool — prints bbox center depth (Hailo) |
| [config.yaml](config.yaml) · [config_loader.py](config_loader.py) | Central management of all runtime parameters |
| [Driving/](Driving/) | Wheel motor · visual-servo driving control · simulation |
| [LevelingPlatform/](LevelingPlatform/) | 3-RRS leveling IK · motor control · simulation |
| [perception/](perception/) | RealSense wrapper, YOLO detection (incl. Hailo), data collection/training, VIO (abandoned) |
| [openrb_integrated/](openrb_integrated/) | **Integrated OpenRB-150 firmware** (Arduino) — controls all actuators (wheels, leveling, tilt, loader, flywheel) |
| [docs/](docs/) | SW architecture · communication protocol docs, motion animation (HTML) |

---

## Execution

All parameters are loaded from [config.yaml](config.yaml), and the CLI accepts only the few
most-frequently-toggled arguments (`--mode` / `--dry-run` / `--debug-detect` / `--config`).

```bash
# Simulation — runs anywhere without camera/motor
python3 pipeline.py --mode sim

# Real hardware (Pi5: RealSense + YOLO + OpenRB)
python3 pipeline.py --mode real

# Phase standalone runs
python3 run_phase1_visual_servo.py
python3 run_phase2_aiming.py
```

### Dependencies

```bash
pip install -r perception/requirements.txt
```

> ⚠️ `albumentations` pulls in `opencv-python-headless`, which overwrites the GUI build.
> After installing, follow the recovery procedure in the top comment of
> [perception/requirements.txt](perception/requirements.txt).

- **Pi5 deployment**: [perception/DEPLOY_PI5.md](perception/DEPLOY_PI5.md)
- **Hailo HEF conversion**: [perception/detection/HAILO_HEF_CONVERT.md](perception/detection/HAILO_HEF_CONVERT.md)
- **OpenRB firmware**: upload [openrb_integrated/openrb_integrated.ino](openrb_integrated/openrb_integrated.ino) to the OpenRB-150 via the Arduino IDE

---

## Submodules (optional — for the abandoned SLAM path)

`perception/librealsense` and `perception/Pangolin` are external dependencies of the
abandoned SLAM/VIO path. **They are not needed to run the current pipeline.** Only if you
want to examine the SLAM code:

```bash
git submodule update --init --recursive
```
