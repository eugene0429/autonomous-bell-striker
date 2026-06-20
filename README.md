# Autonomous Bell-Striker Robot

> 2026 ME Capstone Design — 수직 진동하는 종(지면 ~3 m 높이)을 향해 **자율 주행**한 뒤
> 레벨링 플랫폼으로 **정밀 조준**하여 플라이휠 발사부로 **2회 타격**하는 모바일 로봇 시스템.

시작부터 종료까지 사람 개입 없이 전 과정이 자동화된다. 주 센서는 **Intel RealSense
D435i**(RGB + Depth + IMU) 하나이며, 모든 액추에이터(휠 DC ×2, 레벨링 DXL ×3, 카메라
틸트 ×1, 로더 ×1, 플라이휠 T-motor ×2)는 단일 **OpenRB-150** 보드가 제어한다.

---

## 시스템 개요

```
모바일 베이스로 종 근처까지 거칠게 이동  (Phase 1: Driving)
        → 레벨링 플랫폼으로 정밀 조준    (Phase 2: Aiming, 3-RRS IK)
        → 플라이휠 발사부로 타격 ×2      (Phase 2: Strike)
```

| 구성 | 역할 |
|---|---|
| 2 wheels (diff drive) | Phase 1 coarse driving |
| 3-DOF Leveling Platform (3-RRS) | Phase 2 fine aiming (IK) |
| Flywheel + Launcher | Phase 2 strike |
| Raspberry Pi 5 | 인식·상위 제어 (YOLO, 파이프라인) |
| OpenRB-150 | 전 액추에이터 실시간 제어 (펌웨어) |

상세 설계는 [SW_ARCHITECTURE.md](SW_ARCHITECTURE.md), Pi5 ↔ OpenRB 시리얼 규약은
[COMMUNICATION_PROTOCOL.md](COMMUNICATION_PROTOCOL.md) 참고.

> **Phase 1 측위 방식 변경 (SLAM → Visual Servo).** 초기 설계는 ORB-SLAM3 기반
> 절대 좌표 주행이었으나, **Pi 5 + ORB-SLAM 조합의 런타임 불안정성**으로 폐기하고
> SLAM 없는 **YOLO + depth active-tilt visual servoing** 으로 전환했다. SLAM/VIO
> 코드([perception/vio/](perception/vio/), `Pangolin`·`librealsense` 서브모듈)는
> 참고용으로만 보존되어 있다 (현재 파이프라인 미사용).

---

## 저장소 구조

| 경로 | 설명 |
|---|---|
| [pipeline.py](pipeline.py) | **통합 오케스트레이터** — Phase 1 → Phase 2 를 한 프로세스에서 연속 실행 (단일 RealSense + 단일 OpenRB 공유) |
| [run_phase1_visual_servo.py](run_phase1_visual_servo.py) | Phase 1 단독 러너 (visual-servo 주행) |
| [run_phase2_aiming.py](run_phase2_aiming.py) | Phase 2 단독 러너 (조준 & 타격, lead-aim 포함) |
| [run_center_depth_probe.py](run_center_depth_probe.py) | 검증 도구 — bbox 중심 depth 출력 (Hailo) |
| [config.yaml](config.yaml) · [config_loader.py](config_loader.py) | 모든 런타임 파라미터 중앙 관리 |
| [Driving/](Driving/) | 휠 모터 · visual-servo 주행 제어 · 시뮬레이션 |
| [LevelingPlatform/](LevelingPlatform/) | 3-RRS 레벨링 IK · 모터 제어 · 시뮬레이션 |
| [perception/](perception/) | RealSense 래퍼, YOLO 검출(Hailo 포함), 데이터 수집/학습, VIO(폐기) |
| [openrb_integrated_v5/](openrb_integrated_v5/) | **현행 통합 OpenRB-150 펌웨어** (Arduino) |
| [integrated_controller_w_loader/](integrated_controller_w_loader/) | 로더 포함 통합 컨트롤러 펌웨어 변형 |
| [docs/](docs/) | 설계 plans/specs, 동작 애니메이션(HTML) |
| [tests/](tests/) · `*/tests/` | pytest 단위·통합 테스트 |

---

## 실행

모든 파라미터는 [config.yaml](config.yaml) 에서 로드되며, CLI 인자는 가장 자주 토글하는
몇 개(`--mode` / `--dry-run` / `--debug-detect` / `--config`)만 받는다.

```bash
# 시뮬레이션 — 카메라/모터 없이 어디서나 실행
python3 pipeline.py --mode sim

# 실제 하드웨어 (Pi5: RealSense + YOLO + OpenRB)
python3 pipeline.py --mode real

# Phase 단독 실행
python3 run_phase1_visual_servo.py
python3 run_phase2_aiming.py
```

### 의존성

```bash
pip install -r perception/requirements.txt
```

> ⚠️ `albumentations` 가 `opencv-python-headless` 를 끌어와 GUI 빌드를 덮어쓴다.
> 설치 후 [perception/requirements.txt](perception/requirements.txt) 상단 주석의
> 복구 절차를 따를 것.

- **Pi5 배포**: [perception/DEPLOY_PI5.md](perception/DEPLOY_PI5.md)
- **Hailo HEF 변환**: [perception/detection/HAILO_HEF_CONVERT.md](perception/detection/HAILO_HEF_CONVERT.md)
- **OpenRB 펌웨어**: [openrb_integrated_v5/openrb_integrated_v5.ino](openrb_integrated_v5/openrb_integrated_v5.ino) 를 Arduino IDE 로 OpenRB-150 에 업로드

### 테스트

```bash
pytest
```

---

## 서브모듈 (선택 — 폐기된 SLAM 경로용)

`perception/librealsense`, `perception/Pangolin` 은 폐기된 SLAM/VIO 경로의 외부
의존성이다. **현행 파이프라인 실행에는 필요 없다.** SLAM 코드를 살펴보려는 경우에만:

```bash
git submodule update --init --recursive
```
