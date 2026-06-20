# Phase 2 — Aiming & Strike Pipeline Design

> 종 근처 도달 후 카메라 측정 → IK → 발사로 이어지는 Phase 2 파이프라인의 실제 구현 설계.
> 본 spec 은 [SW_ARCHITECTURE.md](../../../SW_ARCHITECTURE.md) §5 의 stub 구현 (`DummyTargetProvider.get_phase2_target` + [pipeline.py](../../../pipeline.py) phase2_aiming) 을 실제 측정 파이프라인으로 교체한다.

---

## 1. Background & Goals

### 시스템 가정 (상위 spec 에서 결정된 사항)

- Phase 1 종료 시 로봇은 종 바로 아래 근방 (lateral residual error 가 IK reach 범위 내) 에 위치한다.
- 카메라는 시스템 메인 플레이트에 마운트되어 있고 레벨링 플랫폼과 **독립**적으로 구동된다.
- 카메라가 90° 위로 틸트된 상태에서, **카메라 렌즈는 플레이트 중심 기준 (0.20, 0, -0.10) m** 에 위치한다 (plate frame: +X 전방 / +Y 좌측 / +Z 위).
- 종은 ~3 m 높이에서 vertical oscillation 함 (endpoint period 0.5–2.5 s, amplitude ~10 cm). Endpoint 에서 순간 zero-velocity, between endpoints 는 constant velocity.
- Detection backend: Hailo HEF (Pi5 + Hailo-8L), ~30 fps.
- 2회 타격 (`num_strikes=2`).

### 목표

`pipeline.CapstonePipeline.phase2_aiming` 가 다음 시퀀스를 실 하드웨어에서 수행하도록 한다:

1. 카메라 90° 틸트
2. **매 shot 직전 1초 측정** → plate-frame 종 위치 (x, y, z) 산출
3. `LevelingIK.aim_at(xyz)` → 모터 각도 → 송신
4. 발사
5. 위 2–4 반복 (`num_strikes` 회)

### Locked design decisions

이 spec 의 모든 설계 결정의 근거는 사전 brainstorming 에서 다음 가정이 합의되었기 때문이다:

| # | 결정 | 근거 |
|---|---|---|
| D1 | **Pure static** (no real-time tracking, no Kalman/endpoint timing) | "거의 머리 위" 가정 하에서 종의 vertical motion 이 plate→bell 벡터의 elevation 각도에 끼치는 영향이 미미 (3m 높이에서 ±10cm 변동 시 aim 각도 변화 < 0.2°). Hybrid 의 추가 가치 없음. |
| D2 | **매 shot 마다 fresh 1초 측정** | Rolling buffer (continuous tracking) 는 첫 shot 대기 시간 동일 + 코드 복잡도 증가. 1초가 종 oscillation 1 cycle 의 대부분을 커버하여 vertical bias 가 median 에서 자연 상쇄. |
| D3 | **Per-axis median aggregation** | Mean 보다 종 진동의 잔존 bias (endpoint 근처에서 시간 더 보냄) 와 spurious detection outlier 에 강함. |
| D4 | **Depth: 기존 `compute_target_depth` (bbox 중앙 ROI, `roi_frac=0.4`, median)** | 이미 구현됨 ([visual_servo_target.py](../../../perception/detection/visual_servo_target.py)). Clapper 픽셀이 minority 라 median 이 종 본체로 수렴. Annular ring / bimodal histogram / segmentation 은 cost/benefit 부족. |
| D5 | **Clapper avoidance 없음** | 종 입구가 충분히 넓어 dead-center 발사로 추를 우회한다는 가정. 추 위치 tracking 불필요. |
| D6 | **IK ball-limit / unreach fallback 없음 (happy path only)** | Phase 1 driver 가 IK reach 범위로 데려다준다는 가정. `ok=False` 시 경고만 출력하고 진행. |
| D7 | **Phase 1 ↔ Phase 2 직렬 카메라 사용** | Phase 1 종료 시 ORB-SLAM3 stop → Phase 2 용 카메라 start. ORB-SLAM3 ↔ RealSense 동시 사용 최적화는 follow-up. |

---

## 2. Architecture Overview

### 데이터 흐름

```
┌───────────────────────────────────────────────────────────────────┐
│  pipeline.py / CapstonePipeline.phase2_aiming()                   │
│                                                                   │
│    robot.tilt_camera(90°)             ── 기존 동작                │
│    sleep(tilt_settle)                                             │
│                                                                   │
│    for shot in 1..N:                                              │
│        target_xyz = phase2_provider.get_phase2_target()  ◀── new  │
│        ik_out     = LevelingIK.aim_at(target_xyz)        ◀── 기존 │
│        robot.send_leveling_angles(ik_out)                         │
│        sleep(plate_settle)                                        │
│        robot.fire()                                               │
│        sleep(strike_interval)                                     │
└───────────────────────────────────────────────────────────────────┘
                          │
                          │ phase2_provider 추상화
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
DummyTargetProvider                 RealPhase2TargetProvider  ◀── new
(sim — 기존)                        (real — 새 모듈)
get_phase2_target():                get_phase2_target():
  → hardcoded xyz                    1. loop 1.0s collecting frames
    + z jitter                       2. per-frame:
                                        detect → bbox →
                                        depth ROI median →
                                        deproject(cx, cy, depth) →
                                        plate-frame 변환
                                     3. per-axis median 집계
                                          │
                                          ▼
                              CameraToPlateExtrinsic
                              (R + t, fixed extrinsic)
```

### 신규 / 수정 파일

| 파일 | 종류 | 역할 |
|---|---|---|
| `perception/detection/phase2_target.py` | **신규** | `CameraToPlateExtrinsic`, `Phase2TargetEstimator`, `RealPhase2TargetProvider`, `Phase2MeasurementError` |
| `pipeline.py` | **수정** | `CapstonePipeline` 가 `phase2_target_provider` 별도 인자 받음. `phase2_aiming` 의 `Phase2MeasurementError` 처리. settle 매직넘버 → config. `build_pipeline` 의 real 모드에서 `RealPhase2TargetProvider` 와이어링. `RealRobot` 가 `camera` / `detector` 보유. |
| `perception/common/realsense_wrapper.py` | **수정** | `pixel_to_3d_with_depth(x, y, depth_m)` helper 추가 — ROI median depth 를 deprojection 입력으로 쓰기 위함. |
| `perception/detection/tests/test_phase2_target.py` | **신규** | 좌표 변환 + 측정 집계 unit tests. |
| `SW_ARCHITECTURE.md` | **수정** | §5 Phase 2 실제 구현 반영, §6 Camera→Plate 변환 수치 명시, §9 TODO 정리. |

### 기존 자산 재사용

- [perception/detection/visual_servo_target.py:16](../../../perception/detection/visual_servo_target.py#L16) `compute_target_depth` — ROI median depth.
- [perception/common/realsense_wrapper.py:259](../../../perception/common/realsense_wrapper.py#L259) `pixel_to_3d` — single-pixel deprojection. 신규 `pixel_to_3d_with_depth` 는 동일 로직에 depth 를 외부 입력으로.
- [LevelingPlatform/leveling_ik.py:102](../../../LevelingPlatform/leveling_ik.py#L102) `LevelingIK.aim_at` — plate-frame xyz → 모터 각도.
- [perception/detection/detector.py](../../../perception/detection/detector.py) `TargetDetector` — Hailo HEF 추론 (별도 PR 에서 구현 중).

---

## 3. Coordinate Transform

### 좌표계 정의

| Frame | Origin | +X | +Y | +Z |
|---|---|---|---|---|
| **Plate** | 플레이트 중심 | 로봇 전방 | 로봇 좌측 | 위 |
| **Camera (optical)** | 카메라 렌즈 | 이미지 오른쪽 | 이미지 아래 | 광축 (장면 쪽) |

### Extrinsic

카메라 렌즈의 plate-frame 위치: **t = (+0.20, 0, -0.10) m**.

카메라 자세: plate +Y 축 (로봇 좌측) 기준 -90° 회전하여 광축이 plate +Z 와 정렬. 자연스러운 마운트 (camera roll 0°) 가정 시 광축 변환 결과:

- optical +Z → plate +Z (위)
- optical +Y → plate +X (로봇 전방, 이미지 아래)
- optical +X → plate -Y (로봇 우측, 이미지 오른쪽)

회전 행렬:

```
R_plate_from_cam = | 0   1   0 |
                   |-1   0   0 |
                   | 0   0   1 |
```

따라서:

```
P_plate = R · P_cam + t

plate_x = + Y_cam + 0.20
plate_y = - X_cam
plate_z = + Z_cam - 0.10
```

### Sanity check

| 입력 P_cam (m) | 출력 P_plate (m) | 의미 |
|---|---|---|
| (0, 0, 0) | (+0.20, 0, -0.10) | 카메라 렌즈 위치 |
| (0, 0, 3.0) | (+0.20, 0, +2.90) | 카메라 lens 바로 위 3m |
| (0, -0.20, 3.10) | (0, 0, +3.00) | 플레이트 중심 진짜 머리 위 3m |
| (+0.10, 0, 1.0) | (+0.20, -0.10, +0.90) | 이미지 오른쪽 → 로봇 우측 |
| (0, +0.10, 1.0) | (+0.30, 0, +0.90) | 이미지 아래 → 로봇 전방 |

### 구현 형태

```python
@dataclass
class CameraToPlateExtrinsic:
    """Fixed extrinsic from camera optical frame → plate frame.

    Default values: natural mounting (camera roll 0°, tilted 90° about plate +Y).
    If calibration shows image-right ≠ plate -Y or image-down ≠ plate +X,
    flip the corresponding sign field.
    """
    t_x_m: float = 0.20          # lens forward of plate center
    t_z_m: float = -0.10         # lens below plate center
    image_right_sign: int = -1   # +1 if image-right → plate +Y; -1 default
    image_down_sign:  int = +1   # +1 if image-down → plate +X (default)

    def transform(self, p_cam: np.ndarray) -> np.ndarray:
        Xc, Yc, Zc = p_cam
        return np.array([
            self.image_down_sign  * Yc + self.t_x_m,
            self.image_right_sign * Xc,
            Zc + self.t_z_m,
        ])
```

### 캘리브레이션 절차

마운트가 자연 자세인지 검증하려면 90° 틸트 + 알려진 위치 마커로:

1. 플레이트 중심 진짜 머리 위 (plate (0, 0, 1.0)) 에 마커 → image cx ≈ 중앙, cy 가 중앙보다 약간 위 (이미지 위쪽) 에 보임을 확인.
2. 로봇 우측 plate (0, -0.20, 1.0) 에 마커 → image 의 오른쪽에 보임. 틀리면 `image_right_sign` 부호 반전.
3. 로봇 전방 plate (+0.20, 0, 1.0) 에 마커 → image 의 아래쪽에 보임. 틀리면 `image_down_sign` 부호 반전.

---

## 4. Measurement Module

### 클래스 시그니처

```python
# perception/detection/phase2_target.py

class Phase2MeasurementError(RuntimeError):
    """Raised when a 1-second measurement window yields zero valid detections."""


class Phase2TargetEstimator:
    """단일 프레임 측정: detection + ROI depth → plate-frame 3D point."""
    def __init__(
        self,
        camera,                                 # RealSenseCamera
        detector,                               # TargetDetector
        extrinsic: CameraToPlateExtrinsic,
        roi_frac: float = 0.4,
        min_conf: float = 0.5,
    ): ...

    def estimate(self, color, depth_image) -> Optional[np.ndarray]:
        """Returns plate-frame (x, y, z) or None.

        depth_image: np.ndarray, depth in raw units (uint16, 1 mm by default).
                     RealSense depth_frame 은 불필요 — ROI median (compute_target_depth)
                     은 np.ndarray, deprojection (pixel_to_3d_with_depth) 은 intrinsics
                     + 외부 입력 depth_m 만 쓰기 때문.
        """


class RealPhase2TargetProvider:
    """1초 측정창 + per-axis median 집계.

    pipeline.CapstonePipeline 가 매 shot 직전 호출.
    DummyTargetProvider.get_phase2_target() 와 동일 시그니처.
    """
    def __init__(
        self,
        camera,                                 # RealSenseCamera
        estimator: Phase2TargetEstimator,
        measurement_duration_s: float = 1.0,
        min_valid_frames: int = 15,
    ): ...

    def get_phase2_target(self) -> Tuple[float, float, float]:
        """Blocking 1.0s measurement. Returns (x, y, z) in plate frame."""
```

### `Phase2TargetEstimator.estimate()` 알고리즘

```
1. detections = detector.detect(color)
2. if not detections: return None
3. pick top-1 detection by confidence; skip if confidence < min_conf
   (가정: FOV 안에 종은 1개)
4. bbox = chosen detection bbox
5. depth_m = compute_target_depth(depth_image, bbox, roi_frac=0.4)
   if depth_m is None: return None
6. cx, cy = bbox center pixel (반올림)
7. P_cam = camera.pixel_to_3d_with_depth(cx, cy, depth_m)
8. P_plate = extrinsic.transform(P_cam)
9. return P_plate
```

핵심 design choice: deprojection 의 depth 입력으로 **ROI median** 을 쓰고, pixel 좌표는 **bbox 중심** 그대로. lateral (x, y) 는 bbox 위치, depth 는 ROI median 으로 분리되어 각 축의 노이즈 특성에 맞게 robust.

### `RealPhase2TargetProvider.get_phase2_target()` 알고리즘

```
1. samples = []
2. t_start = monotonic()
3. while monotonic() - t_start < measurement_duration_s:
       color, depth_image, _depth_frame = camera.get_frames()
       if color is None: continue              # skip invalid frame
       p_plate = estimator.estimate(color, depth_image)
       if p_plate is not None:
           samples.append(p_plate)
4. n_valid = len(samples)
5. if n_valid == 0:
       raise Phase2MeasurementError("no valid detections in 1.0s window")
6. if n_valid < min_valid_frames:
       log.warning(f"only {n_valid} valid frames (< {min_valid_frames})"
                   " — proceeding")
7. samples_arr = np.stack(samples)              # (N, 3)
8. target_xyz = np.median(samples_arr, axis=0)  # per-axis median
9. return tuple(map(float, target_xyz))
```

### `RealSenseCamera.pixel_to_3d_with_depth` 신규 helper

기존 `pixel_to_3d(depth_frame, x, y)` 는 단일 픽셀 `depth_frame.get_distance(x, y)` 를 쓴다. 우리는 ROI median 을 쓰고 싶으므로 다음을 추가:

```python
def pixel_to_3d_with_depth(self, pixel_x, pixel_y, depth_m):
    """Deproject (pixel_x, pixel_y) using externally-provided depth in meters.

    Returns (x, y, z) in camera optical frame, or None if depth_m <= 0.
    """
    if depth_m <= 0:
        return None
    return rs.rs2_deproject_pixel_to_point(
        self.intrinsics, [pixel_x, pixel_y], depth_m
    )
```

### 노이즈 / 실패 케이스 처리

| 케이스 | 대응 |
|---|---|
| 일부 프레임 검출 실패 | samples 에 추가하지 않음. 다른 프레임으로 진행. |
| ROI 내 valid depth 부족 (`compute_target_depth` → None) | skip. |
| Confidence < `min_conf` | skip. |
| 1초 안에 valid < `min_valid_frames` (=15) | `log.warning`, 가진 sample 으로 진행 (mission abort 안 함). |
| 1초 안에 valid 0개 | `Phase2MeasurementError` raise → orchestrator 가 catch 해서 해당 shot skip. |
| 종 vertical motion 으로 인한 깊이 bias | 1초 측정창 + median 으로 자연 상쇄. |

---

## 5. Orchestrator Integration

### `CapstonePipeline.__init__` 변경

```python
class CapstonePipeline:
    def __init__(
        self,
        robot,
        target_provider,                                    # Phase 1 용 (기존)
        ctrl,
        ik,
        dt=0.067,
        phase1_timeout_sec=60.0,
        num_strikes=2,
        strike_interval_sec=1.0,
        drive_mode="slam",
        # ── Phase 2 신규 ──
        phase2_target_provider=None,                        # None → target_provider 재사용
        tilt_settle_sec: float = 0.5,
        plate_settle_sec: float = 0.3,
    ):
        ...
        self.phase2_target_provider = phase2_target_provider or target_provider
        self.tilt_settle_sec = tilt_settle_sec
        self.plate_settle_sec = plate_settle_sec
```

기본값 `None` 으로 두면 기존 sim 모드 (DummyTargetProvider 단일 인스턴스) 와 backward-compat.

### `phase2_aiming` 변경 (delta only)

기존 [pipeline.py:302-343](../../../pipeline.py#L302-L343) 에서:

- L306: `time.sleep(0.3)` → `time.sleep(self.tilt_settle_sec)`
- L314: `target_xyz = self.target_provider.get_phase2_target()` → `self.phase2_target_provider.get_phase2_target()`, 단 `Phase2MeasurementError` catch 추가:
  ```python
  try:
      target_xyz = self.phase2_target_provider.get_phase2_target()
  except Phase2MeasurementError as e:
      print(f"  ✗ measurement failed: {e} — skip shot")
      continue
  ```
- L335: `time.sleep(0.3)` → `time.sleep(self.plate_settle_sec)`

### `build_pipeline` 변경

```python
def build_pipeline(args) -> CapstonePipeline:
    ...
    target_provider = DummyTargetProvider(target_cfg)       # Phase 1 (기존)

    if args.mode == "sim":
        phase2_target_provider = target_provider            # dummy 재사용
    elif args.mode == "real":
        phase2_target_provider = RealPhase2TargetProvider(
            camera=robot.camera,
            estimator=Phase2TargetEstimator(
                camera=robot.camera,
                detector=robot.detector,
                extrinsic=CameraToPlateExtrinsic(),
            ),
            measurement_duration_s=args.phase2_meas_sec,
            min_valid_frames=args.phase2_min_frames,
        )

    return CapstonePipeline(
        robot, target_provider, ctrl, ik,
        dt=args.dt,
        phase1_timeout_sec=args.phase1_timeout,
        num_strikes=args.num_strikes,
        strike_interval_sec=args.strike_interval,
        drive_mode=args.drive_mode,
        phase2_target_provider=phase2_target_provider,
        tilt_settle_sec=args.tilt_settle_sec,
        plate_settle_sec=args.plate_settle_sec,
    )
```

### `RealRobot` 변경

```python
class RealRobot:
    def __init__(self, ...):
        ...
        from common.realsense_wrapper import RealSenseCamera
        from detection.detector import TargetDetector
        from config import CAMERA, DETECTION
        self.camera = RealSenseCamera(CAMERA)               # 신규
        self.detector = TargetDetector(DETECTION)           # 신규 (Hailo HEF)
        ...

    def start(self):
        self.camera.start()                                 # 신규
        self.localizer.start()
        ...

    def stop(self):
        self.localizer.stop()
        self.camera.stop()                                  # 신규
```

ORB-SLAM3 와 RealSense 가 같은 카메라 stream 을 공유하는 부분 (Phase 1 사용 중 Phase 2 카메라 새로 열 때 충돌 등) 은 본 spec 범위 밖. 1차 구현: Phase 1 종료 후 ORB-SLAM3 stop → Phase 2 용 측정 시작의 **직렬** 사용.

### 외부 의존성

본 PR 머지 시점에 `perception/detection/detector.py` 의 `TargetDetector` 가 **Hailo HEF 추론 가능 상태**여야 함 (별도 PR 에서 진행 중). 만약 본 PR 이 먼저 머지되면 real 모드 wiring 부분은 빌드 시 ImportError 또는 런타임 NotImplementedError 가 발생할 수 있다. 이를 방지하기 위해 본 PR 의 `RealRobot.__init__` 에서 `TargetDetector` 임포트는 **lazy** 로 처리하고 (`start()` 시점에 instantiate), real 모드를 실제 실행해야만 노출되도록 한다. Sim 모드 및 unit tests 는 영향 없음.

### 신규 CLI 인자

```
--phase2-meas-sec    1.0     # 측정창 길이
--phase2-min-frames  15      # 최소 valid 프레임 수
--tilt-settle-sec    0.5     # 90° 틸트 후 대기
--plate-settle-sec   0.3     # 모터 명령 후 대기
```

기존 `--phase2-jitter`, `--phase2-x/y/z` 는 sim 모드에서만 의미. real 모드에서는 무시되도록 `--help` 에 명시.

### Sim ↔ Real 호환성

- **Sim**: `phase2_target_provider == target_provider` (DummyTargetProvider). 기존 동작 그대로.
- **Real**: `phase2_target_provider == RealPhase2TargetProvider`. 실제 측정.
- 두 provider 가 같은 시그니처 (`get_phase2_target() -> (x, y, z)`) 만 충족하면 orchestrator 는 동일.

---

## 6. Testing Strategy

### 테스트 계층

```
Unit  ──────────  test_phase2_target.py
  ├─ CameraToPlateExtrinsic.transform       (좌표 변환 sanity 5+)
  ├─ Phase2TargetEstimator.estimate         (mock camera + detect)
  └─ RealPhase2TargetProvider               (mock 30 frames → median)

Integration ─────  기존 pipeline.py --mode sim 회귀
  └─ phase2_target_provider 인자 없이도 동작 (backward compat)
  └─ --phase2-jitter 가 여전히 동작
  └─ num-strikes=2 시 2번 fire

Bench (수동)  ───  Pi5 + RealSense + Hailo + 종 mockup
  ├─ Extrinsic 캘리브레이션 (§3 절차 3 step)
  ├─ Variance @ 정적 종 (1초 측정 100회 → std_z < 5cm 검증)
  └─ Full Phase 2 dry-run (모터 stub, IK ok=True 확인)
```

### Unit tests — `perception/detection/tests/test_phase2_target.py`

**`TestCameraToPlateExtrinsic`** — §3 sanity check 표의 5+ 케이스 + 부호 반전 케이스.

**`TestPhase2TargetEstimator.estimate`**:
- Mock `camera.intrinsics` (fx=fy=615, ppx=320, ppy=240).
- Mock `detector.detect(color)` 고정 bbox 반환.
- Synthetic depth array (bbox 내부 균일).
- 알려진 bbox 중심 + 알려진 depth → 정확한 P_plate 검증.
- `detect()` 빈 리스트 → None.
- `compute_target_depth` None → None.
- Confidence < `min_conf` → None.

**`TestRealPhase2TargetProvider.get_phase2_target`**:
| 시나리오 | 검증 |
|---|---|
| 30 프레임 모두 valid + 동일값 | median = 그 값 |
| 30 프레임 valid + jitter (gaussian) | median ≈ 평균값 (±jitter/√N) |
| 15 valid + 15 None | 15개 sample 로 진행, warning log |
| 모두 None | `Phase2MeasurementError` raise |
| 종 z-진동 sin wave 모사 | per-axis median 이 중심값에 수렴 |

### Integration — sim 모드 회귀

기존 `pipeline.py --mode sim --phase2-jitter 0.05 --num-strikes 2` 가 새 인자 없이도 동일하게 동작해야 함. CI 에 포함.

### Bench test (수동, spec doc 에 절차 기록)

1. **Extrinsic 캘리브레이션**: §3 의 3-step 절차 → R 부호 확정.
2. **Variance @ 정적 종**: 종 진동 정지 후 1초 측정 100회 → measurement std 확인.
   - 기준: `std_z < 5 cm` (3m 거리, RealSense ~3% noise, 30-frame median → 기대 σ ≈ 1.6 cm).
   - `std_x, std_y < 2 cm`.
3. **Full Phase 2 dry-run**: `pipeline.py --mode real --num-strikes 2` — 모터 stub 출력으로 흐름 검증, IK `ok=True` 확인.

### CI

- Unit tests: pytest CI 포함.
- Sim integration: pytest CI 포함.
- Bench tests: 수동 — 결과를 본 spec 의 §8 에 후속 업데이트.

---

## 7. Failure Handling

본 spec 의 happy-path 외 처리는 **최소화**:

| 실패 | 처리 |
|---|---|
| `get_phase2_target()` 가 `Phase2MeasurementError` raise (valid 검출 0개) | 해당 shot skip, 경고 출력, 다음 shot 진행. |
| `IK.aim_at()` 가 `angles_steps=None` (길이 도달 불가) | 해당 shot skip, 경고 출력. |
| `IK` 결과 `ok=False` (ball-limit 초과, 도달 가능) | 경고만 출력하고 진행. 가정 위반이므로 운영 중 발생 시 follow-up 트리거. |
| 카메라 / Hailo 통신 단절 | 본 spec 범위 밖. 상위 supervisor 가 처리. |

---

## 8. Out-of-Scope / Future Work

### 본 설계가 하지 않는 것

| 항목 | 이유 |
|---|---|
| YOLO HEF 변환 / 배포 | 별도 진행. 본 spec 은 `TargetDetector` 가 동작한다는 전제. |
| 종 vertical motion 의 dynamic tracking / Kalman / endpoint timing | "거의 머리 위" 가정으로 불필요. |
| Clapper avoidance / segmentation | 종 입구 충분히 넓다는 가정으로 무시. |
| IK ball-limit fallback (wheel nudge, Phase 1 재호출) | Phase 1 이 도달 범위로 보내준다는 가정. |
| ORB-SLAM3 ↔ RealSense 동시 사용 최적화 | 직렬 사용으로 단순화. |
| Multi-bell scene | Single-bell 가정. Top-1 confidence 채택. |
| Continuous rolling buffer | Per-shot fresh 측정으로 단순화. |
| Fire ↔ projectile-contact closed-loop verification | 본 spec 은 fire trigger 까지만. |
| Phase 1 visual_servo driver 의 Phase 2 직접 진입 (이미 tilt 된 상태) | Phase 2 시작 시 항상 90° 틸트 재명령으로 단순화. |

### 후속 작업의 트리거 (실측 후)

| 관찰 | 활성화되는 후속 |
|---|---|
| Bench §2 에서 `std_z > 10 cm` | Annular ring depth 또는 segmentation 도입 (D4 재고). |
| 명중률 낮음 + miss 방향이 항상 같음 | Extrinsic 부호 재확인. |
| 명중률 낮고 miss 가 종 vertical motion 과 상관 | Hybrid (Kalman + endpoint timing) 도입 (D1 재고). |
| IK `ok=False` 가 자주 발생 | Wheel nudge fallback (D6 재고). |
| Strike 2 가 Strike 1 의 종 흔들림으로 miss | Settle wait 증가 또는 active damping detection 추가. |

---

## 9. Deliverables Checklist

- [ ] **NEW** `perception/detection/phase2_target.py` (~ 150–200 LOC)
  - `Phase2MeasurementError`
  - `CameraToPlateExtrinsic` (dataclass)
  - `Phase2TargetEstimator`
  - `RealPhase2TargetProvider`
- [ ] **MODIFY** `perception/common/realsense_wrapper.py`
  - `RealSenseCamera.pixel_to_3d_with_depth(x, y, depth_m)` helper 추가
- [ ] **MODIFY** `pipeline.py`
  - `CapstonePipeline` 가 `phase2_target_provider`, `tilt_settle_sec`, `plate_settle_sec` 인자 받음
  - `phase2_aiming` 의 `Phase2MeasurementError` 처리 + settle 매직넘버 제거
  - `build_pipeline` 의 real 모드 와이어링
  - `RealRobot` 가 `camera` / `detector` 보유
  - 신규 CLI 인자 4개
- [ ] **NEW** `perception/detection/tests/test_phase2_target.py` (~ 200–250 LOC)
- [ ] **MODIFY** `SW_ARCHITECTURE.md` §5 / §6 / §9 — Phase 2 실 구현 반영, extrinsic 수치 기록, TODO 정리
- [ ] **MODIFY** `pytest.ini` / CI config (필요 시) — sim integration 회귀 포함
