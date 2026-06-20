# 2026 ME Capstone Design — SW Pipeline Architecture

> 진동하는 종(높이 ~3 m, 수직 진동)을 자율 주행 후 2회 타격하는 시스템의 SW 파이프라인 설계 문서.

> ⚠️ **구현 현황 — Phase 1 측위 방식 변경 (SLAM → Visual Servo).**
> 본 문서 §3·§4 의 Phase 1 은 원래 **ORB-SLAM3 self-pose 기반 절대 좌표 주행**으로
> 설계되었으나, **Raspberry Pi 5 + ORB-SLAM 조합의 런타임 불안정성**(실시간 트래킹
> 끊김·드리프트·CPU 포화)으로 **폐기**되었다. 실제 구현은 SLAM 없이 **YOLO bbox +
> depth 기반 active-tilt visual servoing** 으로 종 바로 아래까지 주행한다
> ([run_phase1_visual_servo.py](run_phase1_visual_servo.py),
> [Driving/visual_servo_controller.py](Driving/visual_servo_controller.py)).
> SLAM/VIO 코드([perception/vio/](perception/vio/), `Pangolin`·`librealsense`
> 서브모듈)는 **참고용으로 보존**되어 있으나 현재 파이프라인에서는 사용하지 않는다.

---

## 1. Mission Spec

| 항목 | 값 |
|---|---|
| 목표 | 수직 진동하는 종을 **2회 타격** |
| 종 위치 | 지면으로부터 약 3 m 높이 |
| 시작 위치 | 종으로부터 **2 ~ 4 m** 떨어진 랜덤 지점 |
| 자율성 | 시작부터 종료까지 **전 과정 자동화** (사람 개입 없음) |
| 주 센서 | **Intel RealSense D435i** (RGB + Depth + IMU) |

---

## 2. 하드웨어 → SW 매핑

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

전략 요약: **모바일 베이스로 종 근처까지 거칠게 이동 → 레벨링 플랫폼으로 정밀 조준 → 플라이휠 발사부로 타격.**

---

## 3. 인식·상위 제어 2-Phase 구조

전체 자율 시퀀스는 두 페이즈로 분리된다. 페이즈 사이의 상태 전이 조건은 **Phase 1의 목표 도달 (goal tolerance 이내)** 이다.

```
   ┌───────────────────────────────────────────────────────────────┐
   │  PHASE 1: DRIVING                                             │
   │                                                               │
   │  [YOLO26n target detect]  ──▶  multi-frame avg (x, y)         │
   │            │                                                  │
   │            ▼                                                  │
   │  [ORB-SLAM3 self-pose] ──▶ controller ──▶ wheel ω_L, ω_R     │
   │                                                               │
   │  종료 조건: |pose_xy - target_xy| < goal_tolerance            │
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
   │  반복: 2회 타격                                                │
   └───────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1 — Driving

### 4.1 목적
시작 지점에서 종 베이스(지면 투영 좌표) 근처까지 모바일 베이스를 자율 주행시킨다. 정확한 조준은 Phase 2가 담당하므로, **goal tolerance 이내 도달**이 종료 조건이다.

### 4.2 파이프라인

1. **Target localization (one-shot, 출발 직전)**
   - 카메라 정면 자세에서 YOLO26n으로 종 검출
   - **여러 프레임에 걸친 검출 결과의 평균**으로 노이즈 제거 → world-frame 목표 (x, y) 좌표 확정
   - 이후 주행 중에는 재탐지하지 않고 이 (x, y)를 고정 목표로 사용

2. **Self-localization (real-time)**
   - **ORB-SLAM3** RGB-D + Pi-optimized (424×240@15fps, nFeatures=500, viewer OFF) 가 기본
   - Production API: [perception/vio/orbslam_localizer.py](perception/vio/orbslam_localizer.py) 의 `OrbSlamLocalizer` (context manager + `get_pose()` → world-frame `(x, y, θ)`)
   - World 좌표계: `world_x = camera_z`, `world_y = -camera_x`, `theta = yaw (CCW+)` — 카메라 출발 자세 = origin

3. **Control (real-time)**
   - 단순 제어 로직: 목표까지의 거리 기반 선속도 + 헤딩 오차 기반 PID 각속도
   - 결과를 좌·우 바퀴 각속도 (ω_L, ω_R) 로 분배해 매 스텝 송신
   - 시리얼 프로토콜은 [Driving/simulation.py](Driving/simulation.py) 의 `SerialCommandSim` 참고 (Pi5 → OpenRB)

### 4.3 관련 모듈

| 역할 | 파일 |
|---|---|
| YOLO 검출기 골격 | [perception/detection/detector.py](perception/detection/detector.py) |
| 2D bbox + depth → 3D 좌표 | [perception/detection/position_estimator.py](perception/detection/position_estimator.py) |
| **Dummy 타겟 제공기 (YOLO 학습 전)** | **[perception/detection/dummy_detector.py](perception/detection/dummy_detector.py)** |
| **통합 파이프라인 오케스트레이터** | **[pipeline.py](pipeline.py)** — Phase1↔Phase2 전환 + sim/real 백엔드 |
| **측위 모듈 (production)** | **[perception/vio/orbslam_localizer.py](perception/vio/orbslam_localizer.py)** — `OrbSlamLocalizer` → world-frame (x, y, θ) |
| ORB-SLAM3 GUI 테스트 러너 | [perception/vio/orbslam_runner.py](perception/vio/orbslam_runner.py) (`--gui`) |
| 카메라 wrapper | [perception/common/realsense_wrapper.py](perception/common/realsense_wrapper.py) |
| 통합 진입점 | [perception/main.py](perception/main.py) |
| **상위 제어 모듈 (production)** | **[Driving/controller.py](Driving/controller.py)** — pose+target → (v, ω, ω_L, ω_R) |
| **휠 모터 시리얼 클라이언트 (production)** | **[Driving/wheel_motor.py](Driving/wheel_motor.py)** — Pi → OpenRB-150 (ASCII protocol, fire-and-forget DRIVE) |
| **Phase-1 only 주행 러너** | **[Driving/drive_to.py](Driving/drive_to.py)** — 목표 (x, y) → ORB-SLAM3 → controller → 휠 모터 |
| **레벨링 모터 시리얼 클라이언트** | **[LevelingPlatform/leveling_motor.py](LevelingPlatform/leveling_motor.py)** — Pi → OpenRB-150 (ASCII protocol) |
| OpenRB 측 참고 스케치 | [LevelingPlatform/openrb_sketch_reference.ino](LevelingPlatform/openrb_sketch_reference.ino) |
| 주행 시뮬레이터 (SLAM 오차 모델 포함) | [Driving/simulation.py](Driving/simulation.py) |
| 카메라 / 검출 설정 | [perception/config.py](perception/config.py) |
| Pi5 배포 가이드 | [perception/DEPLOY_PI5.md](perception/DEPLOY_PI5.md) |
| **Phase 1 driver Protocol + SLAM 구현체** | **[Driving/phase1_driver.py](Driving/phase1_driver.py)** |
| **Phase 1 visual-servo 컨트롤러** | **[Driving/visual_servo_controller.py](Driving/visual_servo_controller.py)** |
| **Phase 1 visual-servo driver** | **[Driving/visual_servo_driver.py](Driving/visual_servo_driver.py)** |

### 4.4 시뮬레이터 (sanity check)

[Driving/simulation.py](Driving/simulation.py) 는 실제 로버를 만들기 전 제어·통신·SLAM 오차 거동을 검증하기 위한 2D 탱크 시뮬레이터로, 다음 요소를 포함한다:

- `TankVehicle` — 차동 구동 모델
- `DisturbanceModel` — 잔디 슬립 + 가우시안 외란
- `SLAMModel` — SLAM 측정 노이즈 + 누적 드리프트 + relocalization 실패
- `SLAMFilter` — outlier rejection + confidence 기반 감속
- `SerialCommandSim` — Pi → OpenRB 시리얼 (rate limit, 데드존, int16 양자화)
- `NavigationController` — 거리 비례 선속도 + PID 각속도 제어기

### 4.5 Phase 1 driver 선택

CLI `--drive-mode {slam,visual_servo}` 로 두 driver 중 하나를 선택.

- `slam` (기본): 기존 ORB-SLAM3 pose → DrivingController → wheel ω. world-frame 측위 필요.
- `visual_servo`: YOLO bbox + depth + active camera tilt servoing 만으로 종 바로 아래까지 이동. SLAM 불안정 환경용. 자세한 설계는 [docs/superpowers/specs/2026-05-20-visual-servo-driving-design.md](docs/superpowers/specs/2026-05-20-visual-servo-driving-design.md) 참조.

`visual_servo` driver 는 본 루프 진입 전 **틸트 sweep bootstrap** 을 실행한다 (`VisualServoPhase1Driver.acquire_initial_tilt`). 0° → 90° 를 5° step 으로 올리며 매 step 마다 detection 을 시도해, 종이 처음 검출되는 tilt 를 초기 각도로 채택. 종이 horizontal FOV 밖이라 모든 tilt 에서 검출 실패 시 fallback 45° 로 시작하고 FSM `SEARCH` 가 차체를 회전해 재획득. 이 부트스트랩 덕분에 가까운 타겟 (~70°+ tilt 필요) / 먼 타겟 (~25° tilt) 모두 자동 처리.

---

## 5. Phase 2 — Aiming & Strike

### 5.1 목적
종 근방에 도달한 상태에서, **레벨링 플랫폼이 발사 방향을 종으로 정렬**시키고 플라이휠 발사부가 종을 타격한다.

### 5.2 파이프라인

1. **Camera tilt to 90° (vertical)**
   - 종이 ~3 m 높이에 있어 정면 카메라로는 시야에서 위쪽으로 벗어남
   - 커스텀 틸트 구조로 카메라를 **90° 위로 회전**시켜 종이 시야 중앙에 들어오게 함

2. **Bell 3D vector estimation**
   - YOLO 검출 → bbox → depth 디프로젝션으로 카메라 좌표계의 (X, Y, Z)
   - 카메라 ↔ 레벨링 플랫폼 중심 사이의 알려진 외부 변환을 적용해 **플랫폼 중심 → 종까지의 3D 벡터**를 얻음
   - **실제 구현**: [perception/detection/phase2_target.py](perception/detection/phase2_target.py) `Phase2TargetEstimator` (single frame) + `RealPhase2TargetProvider` (1초 측정창, per-axis median). 매 shot 직전 호출.
   - **Camera→Plate extrinsic**: lens 가 plate center 기준 `(+0.20, 0, -0.10) m` 에 위치, 90° pitch-up. 회전행렬 + 부호 옵션은 `CameraToPlateExtrinsic` 의 dataclass 필드로 노출.

3. **Inverse Kinematics → motor angles**
   - [LevelingPlatform/leveling_ik.py](LevelingPlatform/leveling_ik.py) 의 `LevelingIK(cfg).aim_at(target_xyz)` 호출
   - 반환 dict 의 `angles_steps` (인코더 step) 또는 `angles_rad` 를 모터 명령으로 사용
   - `ok=False` 면 길이 불가 또는 볼 한계 초과 → 베이스 재정렬 후 재시도

4. **Fire ×2**
   - 진동 종을 2회 타격해야 하므로 **타격 → 재추정 → 재조준 → 타격** 사이클을 반복
   - 종이 수직 진동하므로 매 타격 직전에 3D 벡터를 다시 갱신해야 한다

### 5.3 3-RRS Leveling Platform — `LevelingConfig` 핵심 필드 ([LevelingPlatform/leveling_ik.py](LevelingPlatform/leveling_ik.py))

| 필드 | 기본값 | 설명 |
|---|---|---|
| `Rb` | 0.10 m | 베이스 피벗 반경 |
| `La` | 0.04 m | 크랭크 길이 |
| `Lc` | 0.12 m | 커플러 길이 |
| `Rp` (파생) | `Rb - La` | 플레이트 조인트 반경 (홈 자세 강제) |
| `H0` (파생) | `Lc` | 명목 플레이트 중심 높이 |
| `motor_phis_deg` | (0, 120, 240) | 모터 azimuth |
| `motor_steps` | 4096 | 인코더 카운트/회전 |
| `ball_max_deg` | 30° | P-side 볼 조인트 각도 한계 |
| `quantize` | True | 인코더 step 으로 round 할지 |

API:

```python
from leveling_ik import LevelingIK, LevelingConfig
ik = LevelingIK(LevelingConfig())
out = ik.aim_at(target_xyz)            # 또는 ik.aim_normal(unit_vec)
# out: {angles_deg, angles_rad, angles_steps, ok, ball_deg, c_shift_m, normal}
```

`ok=False` 면 reach 불가 또는 볼 조인트 한계 초과 → 상위 레이어에서 모바일 베이스를 약간 재정렬해 재시도해야 한다.

---

## 6. 주요 좌표계와 정합

| 좌표계 | 정의 | 사용처 |
|---|---|---|
| **World** | 시작 위치/자세 = 원점 | Phase 1 SLAM, 목표 (x, y) |
| **Camera** | RealSense optical frame | YOLO + depth deprojection 직후 |
| **Plate** | 레벨링 플레이트 중심 (`(0, 0, H0)`) | IK 입력 (target 3D point) |

Phase 2 IK 의 입력 target 은 **플레이트 중심 기준** 3D 점이어야 한다. 따라서 Camera→Plate 외부 변환(틸트 90° + 마운트 오프셋 포함)을 사전에 캘리브레이션해 두고, YOLO+depth 결과에 곱해 plate 좌표계로 변환해야 한다.

Camera→Plate 외부 변환은 [perception/detection/phase2_target.py](perception/detection/phase2_target.py) `CameraToPlateExtrinsic` 에 다음 기본값으로 캡슐화되어 있다:

- `t_x_m = +0.20`, `t_z_m = -0.10` (lens가 plate center 기준 (+0.20, 0, -0.10) m)
- `image_right_sign = -1`, `image_down_sign = +1` (자연 마운트, camera roll 0°)

캘리브레이션 절차는 [Phase 2 design spec](docs/superpowers/specs/2026-05-21-phase2-aiming-pipeline-design.md) §3 참조.

---

## 7. 실행 진입점

```bash
# 데이터 수집 (YOLO 학습용)
python perception/main.py capture

# ORB-SLAM3 (default = Pi + no-IMU + headless production module)
python perception/main.py orbslam
# 레거시 GUI 테스트 러너로 보고 싶을 때
python perception/main.py orbslam --gui

# 검출 + 3D 위치 추정 (구현 예정)
python perception/main.py detect

# 주행 시뮬레이션 (단일 / 애니메이션 / 몬테카를로)
python Driving/simulation.py --mode single
python Driving/simulation.py --mode animate
python Driving/simulation.py --mode monte_carlo --runs 100

# 상위 제어 모듈 단독 실행 (한 스텝 출력)
python Driving/controller.py --x 0 --y 0 --th 0 --tx 3 --ty 2 \
                             --wheel_d 0.10 --wheel_base 0.30

# 레벨링 플랫폼 IK CLI
python LevelingPlatform/leveling_ik.py --target 0.10 0.00 3.0

# Phase-1 only 주행 러너 (단독 실행, 개발/실험용)
python Driving/drive_to.py --x 3 --y 2                          # 실시리얼 + ORB-SLAM3
python Driving/drive_to.py --x 3 --y 2 --dry-run --verbose      # 시리얼 미접속, 송신 라인만 출력

# 통합 파이프라인 — dummy detection 기반 (YOLO 학습 전 단계)
python3 pipeline.py                              # 기본 sim 모드
python3 pipeline.py --phase1-x 4 --phase1-y 3    # 타겟 위치 변경
python3 pipeline.py --phase2-jitter 0.10         # 종 진동 폭 ±10cm
python3 pipeline.py --mode real                  # Pi + 카메라 + ORB-SLAM3

# Phase 1 visual-servo 주행 모드 (SLAM-free)
python3 pipeline.py --drive-mode visual_servo --phase1-x 3 --phase1-y 2
```

---

## 8. 통합 파이프라인 — `pipeline.py`

[pipeline.py](pipeline.py) 는 production 모듈 4개를 하나의 실행 흐름으로 묶은 통합 오케스트레이터다.

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
│         차동구동 정기구학)          모터 stub)                   │
└──────────────────────────────────────────────────────────────────┘
```

### 백엔드

| `--mode sim` (기본) | `--mode real` |
|---|---|
| 카메라/모터 없이 어디서나 실행 | RealSense + ORB-SLAM3 필요 (Pi5) |
| `SimulatedRobot` — 차동 구동 정기구학으로 즉시 자세 적분 | `RealRobot` — `OrbSlamLocalizer` 로 pose, 모터는 stub 출력 |
| 모듈 연동·Phase 전환 검증용 | 실제 SLAM 측위 + 모터 stub 검증용 |

### Phase 전환 조건

- **Phase 1 → Phase 2**: `DrivingController.compute()` 의 `out["reached"] == True` (즉 `distance < goal_tolerance`)
- **Phase 2 종료**: 설정된 `num_strikes` 회 (기본 2회) 타격 완료
- **타격마다 종 위치 재추정**: `DummyTargetProvider.get_phase2_target()` 가 z 에 jitter 를 적용해 진동하는 종을 모사

### Dummy → 실제 detection 으로 교체

YOLO 학습이 끝나면 [pipeline.py](pipeline.py) 의 `DummyTargetProvider` 자리에 다음을 끼워 넣으면 된다:

- **Phase 1**: `TargetDetector.detect()` 결과를 N 프레임 평균 → `PositionEstimator.estimate()` 로 world-frame (x, y) 산출
- **Phase 2**: 카메라 90° 틸트 후 동일 체인으로 plate-frame (x, y, z) 산출 (Camera→Plate 외부 변환 적용)

`DummyTargetProvider` 를 동일 시그니처 (`get_phase1_target() → (x, y)`, `get_phase2_target() → (x, y, z)`) 로 감싸기만 하면 파이프라인 본체는 수정 불필요.

---

## 9. 미구현 / TODO

- [ ] [perception/detection/detector.py](perception/detection/detector.py) — YOLO 모델 로딩 / 추론 구현 (현재 NotImplementedError)
- [ ] Phase 1 다중 프레임 평균 + Phase 2 카메라→플레이트 변환 (Phase 2 부분은 완료: [phase2_target.py](perception/detection/phase2_target.py))
- [ ] [perception/main.py](perception/main.py) `detect` 모드 — Detection + PositionEstimator 통합
- [x] Camera ↔ Plate 외부 변환 캘리브레이션 절차 문서화 ([2026-05-21 design spec](docs/superpowers/specs/2026-05-21-phase2-aiming-pipeline-design.md) §3)
- [ ] Phase 2 bench test: extrinsic 캘리브레이션 부호 확정 + 정적 종 1초 측정 std_z < 5 cm 검증
- [ ] Phase 2 ↔ Phase 1 카메라 stream 동시 사용 최적화 (현재는 직렬: SLAM stop → camera reopen)
- [x] 휠 시리얼 드라이버 — [Driving/wheel_motor.py](Driving/wheel_motor.py) (ASCII line protocol)
- [ ] [pipeline.py](pipeline.py) 의 `RealRobot.send_wheel_omegas` stub → `WheelMotorClient` 로 교체 (별도 PR)
- [ ] OpenRB 펌웨어에 `DRIVE`/`STOP`/`PING` 핸들러 + 200 ms watchdog 추가 (별도 PR)
- [x] 카메라 90° 틸트 서보 명령  ← TILT_ASYNC v1.1 으로 부분 완료, sync TILT 는 별도
- [ ] 시리얼 프로토콜 실측 — `wheel_motor.py` 의 ASCII 프로토콜과 OpenRB 펌웨어의 라운드트립 검증
