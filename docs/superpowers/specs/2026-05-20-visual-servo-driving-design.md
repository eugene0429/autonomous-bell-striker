# Phase 1 Visual-Servo Driving Mode — Design

> SLAM-free 주행 모드. YOLO bbox + depth + active camera tilt servoing 만으로 종 바로 아래까지 로버를 이동시키고 Phase 2 (조준·타격) 로 핸드오프한다. 기존 SLAM 기반 Phase 1 과 CLI flag 로 병존한다.

작성일: 2026-05-20
관련 문서: [SW_ARCHITECTURE.md](../../../SW_ARCHITECTURE.md), [COMMUNICATION_PROTOCOL.md](../../../COMMUNICATION_PROTOCOL.md)

---

## 1. 배경과 목표

### 문제
현재 [pipeline.py](../../../pipeline.py) 의 Phase 1 은 [OrbSlamLocalizer](../../../perception/vio/orbslam_localizer.py) 에서 world-frame pose `(x, y, θ)` 를 받아 [DrivingController](../../../Driving/controller.py) 가 목표 `(target_x, target_y)` 까지 차동구동 명령을 산출하는 구조다. 실주행 환경에서 ORB-SLAM3 의 tracking 실패 / 재초기화 지연 / 측위 점프가 반복적으로 관측되어 Phase 1 자체가 불안정하다.

### 목표
SLAM 측위에 의존하지 않는 **visual-servoing 기반 주행 모드** 를 새로 만든다. 핵심 아이디어:
- YOLO 가 검출한 종 bbox 가 항상 화면 중앙에 위치하도록 (1) 카메라 틸트를 능동 제어하고 (2) 로버를 회전시킨다
- 카메라가 거의 수직 (틸트 90°±5°) 인 순간 = 로버가 종 바로 아래에 있는 순간 → 정지 후 Phase 2 로 전이
- 전진 속도는 "지면에서 종까지 남은 수평거리" = `depth × cos(tilt)` 에 비례

### 비목표
- Phase 2 (조준·타격) 의 로직은 변경하지 않는다 — Phase 1 driver 인터페이스만 교체.
- 기존 SLAM 모드를 제거하지 않는다 — CLI flag 로 둘 다 선택 가능하게 병존.
- 자동 폴백 (SLAM 실패 → visual_servo 자동 전환) 은 1차 설계 범위 밖.

---

## 2. 결정 요약

| # | 결정 | 대안 | 사유 |
|---|---|---|---|
| Q1 | 두 모드 병존, CLI flag 선택 | (A) 완전 대체 / (C) 자동 폴백 | 모드 격리 → 디버깅 용이, 폴백은 경계 케이스가 복잡해 후속 작업 |
| Q2 | Active 틸트 PI servoing | (B) Passive / (C) Open-loop LUT | 캘리브레이션-tolerant, "타겟이 항상 화면 중앙" 요구와 일치 |
| Q3 | `v ∝ depth × cos(tilt)` (수평거리) | (B) 틸트만 / (C) 가중합 | 물리적 의미가 직관적, D435i depth 신뢰 구간 (0.3~3m) 과 일치, false positive 방지 |
| Q4 | 4-단계 graceful degradation (TRACK→COAST→HOLD→SEARCH) | (A) 단순 timeout / (C) SLAM 폴백 | 깜빡임 노이즈 흡수 + 능동 검색으로 시야 이탈 복구 |
| Q5 | 새 `TILT_ASYNC` 프로토콜 명령 | (B) `DRIVE` 확장 / (C) thread 우회 | High-rate streaming control 에 native 한 fire-and-forget 명령이 정도 |

---

## 3. 아키텍처

### 3.1 모듈 분할

```
Driving/
├── controller.py                  (기존, 변경 없음)
├── phase1_driver.py               (신설) Phase1Driver Protocol + SlamPhase1Driver
├── visual_servo_controller.py     (신설) bbox + depth → (v, ω, tilt_cmd) 순수 함수
└── visual_servo_driver.py         (신설) Phase1Driver 구현체, FSM 보유

LevelingPlatform/
└── tilt_motor.py                  (신설) TiltAsyncClient (f&f) + TiltClient (sync)

perception/detection/
├── detector.py                    (기존)
├── dummy_detector.py              (기존, 확장: get_visual_servo_detection)
└── visual_servo_target.py         (신설) bbox + depth_frame → ROI median depth

pipeline.py                        (수정) --drive-mode flag, Phase1Driver 선택
openrb_integrated_v5/
└── openrb_integrated_v5.ino       (수정) TILT_ASYNC 핸들러 + watchdog

COMMUNICATION_PROTOCOL.md          (수정) v1.1 — TILT_ASYNC 명령 추가
```

### 3.2 통합 흐름

```
pipeline.py
 └─ CapstonePipeline.phase1_driving()
       └─ driver: Phase1Driver = build_driver(args.drive_mode)
             ├─ "slam"          → SlamPhase1Driver(robot, target_provider, ctrl)
             └─ "visual_servo"  → VisualServoPhase1Driver(robot, target_provider, vs_ctrl)
       ok = driver.run()  # blocks until reached or failed
```

`phase1_driving()` 본체는 driver-agnostic 래퍼로 슬림화하고 두 구현체가 동일 `run() -> bool` 시그니처를 제공한다. Phase 2 는 양쪽 driver 모두 `True` 를 반환한 직후 동일하게 실행된다.

### 3.3 인터페이스

```python
# Driving/phase1_driver.py
class Phase1Driver(Protocol):
    def run(self) -> bool: ...   # True = 도달, False = 실패/타임아웃

class SlamPhase1Driver:          # 기존 pipeline.py 의 phase1_driving 본체를 그대로 이관
    def __init__(self, robot, target_provider, ctrl: DrivingController,
                 dt: float = 0.067, timeout_s: float = 60.0): ...
    def run(self) -> bool: ...
```

### 3.4 Robot 인터페이스 확장

기존 `SimulatedRobot` / `RealRobot` ([pipeline.py](../../../pipeline.py)) 에 다음 메서드 추가. 기존 메서드 (`get_pose`, `send_wheel_omegas`, `tilt_camera`, `send_leveling_angles`, `fire`) 는 변경 없음.

| 메서드 | sim | real |
|---|---|---|
| `get_color_depth() -> (np.ndarray, np.ndarray)` | dummy 합성 (또는 우회 경로) | RealSenseCamera 래퍼 |
| `get_tilt_deg() -> float` | 내부 변수 `_tilt_deg` | 마지막 명령 setpoint 또는 STATUS 응답 |
| `send_tilt_async(step: int)` | no-op + 내부 변수 갱신 | `TiltAsyncClient.send(step)` |
| `get_detection_or_none() -> dict \| None` *(sim 전용, 우회 경로)* | `DummyTargetProvider.get_visual_servo_detection()` | (사용 안 함, real 은 `get_color_depth` 경로) |

sim 모드는 합성 영상 → YOLO 추론을 거치는 대신 직접 detection dict 를 반환하는 우회 경로를 둔다. 이는 sim 의 목적이 "controller + FSM 자체 검증" 이지 "YOLO 동작 검증" 이 아니기 때문.

---

## 4. 제어 알고리즘

### 4.1 매 스텝 (15 Hz, dt = 0.067s)

```
1. Sense
   color, depth  ← Robot.get_color_depth()
   detection     ← YoloTargetDetector(color, depth)
                     bbox(x1,y1,x2,y2), conf, depth_m=median(depth[bbox ROI])
   tilt_deg      ← Robot.get_tilt_deg()

2. Visual servoing errors
   cx, cy         = bbox center pixel
   err_x_px       = cx - (W/2)           # 좌(-) / 우(+)
   err_y_px       = cy - (H/2)           # 위(-) / 아래(+)
   horiz_dist_raw = depth_m · cos(tilt_deg)

2.5 Robustness filters (종 vertical oscillation 대비 — §9 참조)
   # LPF on horiz_dist (첫 프레임은 raw 값으로 초기화)
   if horiz_dist_filt is undefined:
       horiz_dist_filt = horiz_dist_raw
   else:
       horiz_dist_filt = α · horiz_dist_raw + (1-α) · horiz_dist_filt_prev
   # tilt dead-band: 작은 진동은 tilt 갱신 skip
   err_y_px_eff   = 0 if |err_y_px| < tilt_err_deadband_px else err_y_px

3. Tilt PI (err_y_px_eff → tilt 증분)
   Δtilt_deg     = -Kp_tilt · err_y_px_eff  -  Ki_tilt · ∫err_y_eff dt
   tilt_cmd_deg  = clip(tilt_deg + Δtilt_deg, 0°, 95°)
   send TILT_ASYNC(step_from_deg(tilt_cmd_deg))

4. Heading PID (err_x_px → ω)
   ω             = -Kp_h · err_x_px - Ki_h · ∫err_x dt - Kd_h · d/dt
   ω             = clip(ω, ±omega_max)

5. Forward velocity (수평거리 + heading 정렬도)
   align         = max(0.2, 1 - |err_x_px| / (W/2))
   v             = clip(Kp_v · horiz_dist_filt · align, 0, v_max)

6. Stop? (debounce: 연속 N 프레임 만족해야 DONE 으로 전이)
   if (85° ≤ tilt_cmd_deg ≤ 95°) AND (horiz_dist_filt < d_stop_m):
       stop_streak += 1
   else:
       stop_streak  = 0
   if stop_streak ≥ stop_debounce_frames:
       send DRIVE 0 0;  return DONE

7. Differential drive 역기구학 (controller.py 의 헬퍼 재사용)
   (v, ω) → (ω_L, ω_R)
   send DRIVE ω_L ω_R
```

### 4.2 좌표·부호 규약

- 화면 좌표: 좌상단 (0, 0), x 우향 +, y 하향 + (OpenCV/RealSense 관례)
- 차동구동 ω 규약: ω > 0 = CCW (위에서 봤을 때 반시계, 차체가 왼쪽으로 yaw), 카메라 광축도 왼쪽으로 향함. `controller.py` 의 `(v, ω) → (ω_L, ω_R)` 변환 (`v_R = v + ω·L/2`, `v_L = v - ω·L/2`) 과 일치.
- `err_x_px > 0` (타겟이 화면 오른쪽) → 카메라 광축을 오른쪽으로 향하게 하려면 차체를 CW 로 yaw → **ω < 0**. 따라서 식 §4.1 step 4 의 부호는 `ω = -Kp_h · err_x_px` (Kp_h > 0 정의).
- `err_y_px > 0` (타겟이 화면 아래쪽) → 틸트를 **줄여야** 함 (`Δtilt < 0`). 따라서 식 §4.1 step 3 의 `-Kp_tilt`. 종이 위로 갈수록 (`err_y_px < 0`) `Δtilt > 0` 으로 카메라 위로 들림.
- `tilt_deg = 0°` 정면 수평, `tilt_deg = 90°` 수직 위. DXL step 변환은 캘리브레이션 상수 `steps_per_deg` 와 home_offset 으로 한다.

### 4.3 게인·임계 초기값

| 파라미터 | 의미 | 초기값 |
|---|---|---|
| `Kp_tilt` | y-pixel → deg/frame | 0.05 |
| `Kp_h` | x-pixel → rad/s | 0.005 |
| `Ki_h` | (rad/s) / (px·s) | 0.001 |
| `Kd_h` | (rad/s) / (px/s) | 0.001 |
| `Kp_v` | (m/s) / m | 0.5 |
| `v_max` | 최대 선속도 | 0.3 m/s |
| `omega_max` | 최대 각속도 | 1.0 rad/s |
| `d_stop_m` | 수평거리 stop 임계 | 0.10 m |
| `tilt_stop_range_deg` | stop 틸트 범위 | (85.0, 95.0) |
| `tilt_max_deg` | 틸트 상한 | 95.0 |
| `depth_roi_frac` | bbox 내부 중앙 ROI 비율 | 0.4 |
| `horiz_dist_lp_alpha` | `horiz_dist` 1차 LPF 계수 (α) | 0.2 (τ ≈ 0.27 s) |
| `tilt_err_deadband_px` | tilt PI 진입 dead-band [px] | 8 (≈ 0.7°) |
| `stop_debounce_frames` | stop 조건 연속 만족 프레임 수 | 8 (≈ 0.53 s) |
| `dt` | 제어 주기 | 0.067 s (15 Hz) |

게인은 모두 튜닝 대상이며, 위 값은 simulation tier 의 시작점이다. 실주행 튜닝 시 우선순위: `Kp_v` (속도) → `Kp_h` (heading) → `Kp_tilt` (틸트 follow). `horiz_dist_lp_alpha` / `tilt_err_deadband_px` / `stop_debounce_frames` 는 종 vertical oscillation (§9 가정: peak-to-peak 0.3~0.5 m, random endpoint period) 흡수를 위한 robustness 파라미터.

### 4.4 Depth ROI median

bbox 중심 1픽셀의 raw depth 는 노이즈가 크고 hole 빈도가 높다. bbox 내부 중앙 40% 영역의 valid (>0) depth median 을 사용한다. valid 픽셀이 N (예: 10) 미만이면 그 프레임은 검출 실패로 간주 (FSM `lost_frames++`).

---

## 5. State Machine

```
                 detection found
       ┌──────────────────────────────────┐
       ▼                                  │
   ┌────────┐  lost ≥ 1              ┌────┴─────┐
   │  TRACK │ ─────────────────────▶ │   COAST  │
   │ (정상) │                        │ (단기 손실)│
   └────────┘ ◀───── found ──────────└────┬─────┘
       │                                   │ lost ≥ 3 (200ms)
       │ stop_cond                         ▼
       │ (§4 step 6)                  ┌─────────┐
       ▼                              │  HOLD   │  ◀── found
   ┌────────┐                         │ (정지 대기)│
   │ DONE   │                         └────┬────┘
   │(phase  │                              │ lost ≥ 15 (1s)
   │ 전환)  │                              ▼
   └────────┘                         ┌─────────┐  found
                                      │  SEARCH │  ──────┐
                                      │(원지 회전)│       │
                                      └────┬────┘  ◀────┘
                                           │ spin_timeout (15s)
                                           ▼
                                      ┌─────────┐
                                      │  FAIL   │ → Phase 1 실패
                                      └─────────┘
```

| State | DRIVE 명령 | TILT_ASYNC 명령 | 진입 조건 | 탈출 조건 |
|---|---|---|---|---|
| `TRACK` | §4 의 정상 제어 | 정상 servoing | 검출 발견 | stop_cond → `DONE` / lost ≥ 1 → `COAST` |
| `COAST` | 직전 (ω_L, ω_R) × 0.7 | 직전 setpoint 유지 | 1 ≤ lost < 3 | found → `TRACK` / lost ≥ 3 → `HOLD` |
| `HOLD` | (0, 0) | 직전 setpoint 유지 | 3 ≤ lost < 15 | found → `TRACK` / lost ≥ 15 → `SEARCH` |
| `SEARCH` | v = 0, ω = ∓ search_omega *(부호: lost 직전 err_x_px 부호 — err_x_px>0 이면 ω<0 으로 우회전)* | 직전 setpoint 유지 | lost ≥ 15 | found → `TRACK` / 누적 회전 15s → `FAIL` |
| `DONE` | (0, 0) | 유지 | stop_cond 만족 | — |
| `FAIL` | (0, 0) | 0° 로 복귀 | spin_timeout | — |

검출 성공 1회로 `lost_frames` 카운터는 즉시 0 으로 리셋. SEARCH 중에도 `TILT_ASYNC` 를 직전 setpoint 로 매 프레임 송신해 200ms watchdog 가 만료되지 않도록 한다.

설계 주석:
- COAST 의 속도 70% 깎기: 검출이 빠진 직전 프레임의 명령이 그대로 유지·증폭되는 패턴을 살짝 완화. 이미 잘못된 명령이라면 빠르게 HOLD 로 강등.
- SEARCH 방향: 마지막으로 본 `err_x_px` 부호 쪽 — 종이 오른쪽으로 빠져나갔으면 오른쪽으로 회전해 다시 잡는다.

---

## 6. Protocol Addition — `TILT_ASYNC`

[COMMUNICATION_PROTOCOL.md](../../../COMMUNICATION_PROTOCOL.md) 를 **v1.1** 로 개정. §4 명령어 표에 다음 행을 추가:

| 명령 | 인자 | 응답 | 동기 | 단위·범위 |
|---|---|---|---|---|
| `TILT_ASYNC` | `<s4>` | (없음) | f&f | DXL step -2047..+2047, **200ms watchdog** |

### 6.1 시맨틱
- 기존 `TILT` 와 동일한 ID 4 DXL 을 제어하지만 **motion-complete polling 없이 goal_position 즉시 갱신** 만 수행.
- 새 setpoint 가 도착하면 펌웨어는 직전 goal 을 덮어쓴다.
- **200ms watchdog**: 마지막 `TILT_ASYNC` 수신 후 200ms 내 추가 라인이 없으면 펌웨어가 **현재 위치 readback 으로 goal 을 덮어 hold**. `DRIVE` watchdog 와 같은 시간 정책이지만 동작은 "0 으로 강제" 가 아니라 "현재 위치 hold". 이유: 틸트는 0° 가 안전 위치가 아니다 — 갑자기 떨어지면 카메라가 휙 내려가 사고로 이어진다.
- **공존**: 두 명령 모두 동일한 `goal_position` 레지스터를 쓰므로 마지막 수신 명령의 setpoint 가 최종값. sync `TILT` 수신 시 펌웨어는 `OK` 응답까지 motion-complete 를 polling 하므로 그동안 들어오는 `TILT_ASYNC` 라인은 `ERR BUSY` 로 거절된다. sync `TILT` 수신 시점에 `last_tilt_async_ms` 도 같이 갱신해 watchdog 가 즉시 만료되지 않도록 한다. Phase 2 진입 시 `TILT 1024` (≈ 90°) 를 sync 로 한 번 보내 위치 확정.
- **STATUS flag bit 2 (tilt moving)**: sync/async 구분 없이 동일 set/clear.

### 6.2 Pi 측 클라이언트
- 신설: [LevelingPlatform/tilt_motor.py](../../../LevelingPlatform/tilt_motor.py) 의 `TiltAsyncClient.send(step)` — fire-and-forget, 응답 대기 없음.
- 동일 파일에 `TiltClient` (sync `TILT`) 도 함께 정의. Phase 2 / 초기화 시 사용.
- 두 facade 는 `OpenRBClient` 공유 시리얼 owner 위에 올라간다.

### 6.3 OpenRB 펌웨어
[openrb_integrated_v5/openrb_integrated_v5.ino](../../../openrb_integrated_v5/openrb_integrated_v5.ino) 변경:
- 명령 dispatcher 에 `TILT_ASYNC` case 추가, `dxl.setGoalPosition(4, step)` 만 호출.
- `last_tilt_async_ms` 타이머 추가. `loop()` 에서 200ms 만료 검사 → `dxl.getPresentPosition(4)` readback → goal 을 readback 으로 덮어 hold.
- DXL profile_velocity / acceleration 은 별도 튜닝 항목 (초기엔 기본값, 15Hz 추종이 부족하면 상향).

---

## 7. Configuration

```python
# Driving/visual_servo_controller.py
@dataclass
class VisualServoConfig:
    # 카메라 frame
    img_w: int = 640
    img_h: int = 480
    # 게인
    kp_tilt: float = 0.05         # deg / px
    ki_tilt: float = 0.0
    kp_h:    float = 0.005        # (rad/s) / px
    ki_h:    float = 0.001
    kd_h:    float = 0.001
    kp_v:    float = 0.5          # (m/s) / m
    # 한계
    v_max:         float = 0.3
    omega_max:     float = 1.0
    tilt_min_deg:  float = 0.0
    tilt_max_deg:  float = 95.0
    # stop
    d_stop_m:            float = 0.10
    tilt_stop_range_deg: Tuple[float, float] = (85.0, 95.0)
    # FSM
    coast_lost_frames: int   = 3
    hold_lost_frames:  int   = 15
    search_omega:      float = 0.6
    search_timeout_s:  float = 15.0
    # depth
    depth_roi_frac:    float = 0.4
    depth_min_valid_pixels: int = 10
    # robustness vs. 종 vertical oscillation (§9 참조)
    horiz_dist_lp_alpha:   float = 0.2   # LPF α on horiz_dist (τ ≈ 0.27s)
    tilt_err_deadband_px:  int   = 8     # |err_y_px| < N → tilt 갱신 skip
    stop_debounce_frames:  int   = 8     # stop 조건 연속 N 프레임 필요
    # 루프
    dt: float = 0.067
```

```python
# Driving/visual_servo_controller.py
class VisualServoController:
    def __init__(self, cfg: VisualServoConfig | None = None): ...

    def reset(self) -> None: ...

    def step(
        self,
        detection: Optional[dict],   # {bbox, conf, depth_m} or None
        tilt_deg_cur: float,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          state             : "TRACK" | "COAST" | "HOLD" | "SEARCH" | "DONE" | "FAIL"
          v, omega          : twist (m/s, rad/s)
          wheel_omega_left  : rad/s
          wheel_omega_right : rad/s
          tilt_cmd_deg      : 0..95
          err_x_px, err_y_px: 시각 오차 (디버깅)
          horiz_dist        : 수평거리 추정 (m)
          reached           : bool   (state == DONE)
          failed            : bool   (state == FAIL)
        """
```

```python
# Driving/visual_servo_driver.py
class VisualServoPhase1Driver:
    def __init__(
        self,
        robot,
        target_provider,             # sim 우회 경로용; real 에서는 무시
        ctrl: VisualServoController,
        dt: float = 0.067,
        timeout_s: float = 60.0,
    ): ...

    def run(self) -> bool:
        """매 dt 마다:
          detection = self._acquire_detection(robot)
          tilt_cur  = robot.get_tilt_deg()
          out       = ctrl.step(detection, tilt_cur)
          robot.send_tilt_async(step_from_deg(out['tilt_cmd_deg']))
          robot.send_wheel_omegas(out['wheel_omega_left'],
                                  out['wheel_omega_right'], dt)
          if out['reached']: return True
          if out['failed']:  return False
        타임아웃: timeout_s 초과 시 False
        """
```

### 7.1 CLI
```bash
python3 pipeline.py --drive-mode visual_servo
python3 pipeline.py --drive-mode slam          # 기본값
# 노이즈 옵션 (sim 전용)
python3 pipeline.py --drive-mode visual_servo \
                    --vs-bbox-noise 5 --vs-depth-noise 0.05 --vs-dropout 0.05
```

---

## 8. Testing & Verification

### Tier 1 — Unit (`VisualServoController.step`)
순수 함수 테스트. 합성 detection dict 입력으로:
- bbox 정중앙 / depth 2m / tilt 30° → v > 0, ω ≈ 0, tilt 증가 방향
- bbox 정중앙 / depth 0.4m / tilt 88° → state == "DONE" (debounce frame 수 만큼 반복 후)
- bbox 오른쪽 100px / depth 2m → ω > 0, |ω| 적절 한계 안
- bbox y 위쪽 50px → tilt_cmd_deg > tilt_deg_cur
- detection = None ×1 → "COAST", ×3 → "HOLD", ×15 → "SEARCH"
- search 시작 후 16초 → "FAIL"
- search 중 found → "TRACK" 복귀
- **horiz_dist LPF**: 첫 프레임은 raw 값 = filt 값, 이후 step 입력 (raw 갑자기 0.5 → 1.5) 에 대해 filt 가 한 프레임에 완전히 따라가지 않음 (1.5 보다 작음)
- **tilt dead-band**: `|err_y_px| < tilt_err_deadband_px` 일 때 `tilt_cmd_deg == tilt_deg_cur` (갱신 skip)
- **stop debounce**: stop 조건이 단 1 프레임 만족 → 아직 "TRACK", 연속 `stop_debounce_frames` 만족 → "DONE"; 중간에 한 프레임 미만족이 끼면 streak 리셋

위치: [Driving/tests/test_visual_servo_controller.py](../../../Driving/tests/test_visual_servo_controller.py) (신설)

### Tier 2 — Integration (`VisualServoPhase1Driver` + `SimulatedRobot`)
- `DummyTargetProvider.get_visual_servo_detection(robot_pose, target_xyz)` 가 핀홀 카메라 모델로 bbox / depth 합성, 카메라 tilt 와 FOV 반영
- **종 vertical oscillation 시뮬레이션** 옵션: `bell_height_amp_m` (peak-to-peak), `bell_endpoint_period_s = (min, max)` — 매 endpoint 도달 시 다음 traverse 시간을 균등 분포로 재샘플링. amp 기본 0 (정지), 진동 활성 시 0.3~0.5 m / period (0.5, 2.5) s 권장.
- 시작 위치 (2, 2), (3, -1), (-2, 3) 에서 모두 `tilt ∈ [85, 95]` AND `horiz_dist < 0.1` 에서 stop 하는지 monte-carlo
- 노이즈 켜고 (bbox_noise=5px, depth_noise=0.05m, dropout=0.05) 100회 → 도달 성공률, 평균 시간
- **종 진동 시나리오 별도 측정**: 위와 동일 노이즈 + `bell_height_amp_m=0.5`, `bell_endpoint_period_s=(0.5, 2.5)` 100회 → 도달 성공률을 정지 시나리오와 비교

위치: [Driving/tests/test_visual_servo_driver_sim.py](../../../Driving/tests/test_visual_servo_driver_sim.py) (신설)

### Tier 3 — Bench (펌웨어 + 실 DXL, 로버 무동작)
- `tmp_tilt_async_test.py` 단독 스크립트:
  - 15Hz 로 sin wave setpoint 1분 송신 → DXL 추종성 확인
  - 송신 200ms+ 일시 중단 → hold 동작 (0° 로 떨어지지 않음) 확인
  - sync `TILT` 명령 중간 삽입 → 깨지지 않음 확인
- 휠 출력은 OpenRB 측에서 disable, `DRIVE` 응답만 확인

### Tier 4 — On-rover
- 실내 평지, 종 모형/마커를 3m 높이에 고정, 2~4m 거리에서 시작
- `--drive-mode visual_servo --dry-run` (DRIVE 라인 로그만, 실모터 off) → 명령 합리성 육안 확인
- 실모터 on → 도달 wall-time, 최종 정렬 오차 측정
- 의도적 가림 → SEARCH 발동 확인
- Phase 2 첫 IK 시도 `ok=True` 비율 측정

### 성공 기준
- Tier 2 (정지 종): 100회 monte-carlo 중 95회 이상 도달, 평균 도달 시간 < 15s
- Tier 2 (진동 종, amp 0.5m): 100회 중 90회 이상 도달 (낮은 bar — 진동이 stop debounce 를 늦출 수 있음)
- Tier 4: 5회 연속 도달 (`tilt 85~95° AND horiz_dist < 0.15m`), Phase 2 IK 첫 시도 `ok=True` 비율 ≥ 60%

---

## 9. 가정과 제약

### 가정
- 카메라 광축이 로버의 정면과 일치 (또는 알려진 yaw offset 으로 보정 가능)
- 종이 단일 객체로 YOLO 에 검출됨 (다중 후보 처리 로직은 1차 범위 밖 — 일단 conf 최댓값 선택)
- 종의 mean altitude ≈ 3 m. **종은 vertical 방향으로만 운동** (horizontal motion 없음), peak-to-peak 진폭 **0.3 ~ 0.5 m**, **endpoint 도달 시마다 다음 traverse 시간을 random 으로 샘플링** (broadband signal). Phase 1 visual_servo 의 목표는 "종이 진동하는 동안 종의 mean 수평 위치 바로 아래에 가능한 한 가깝게 정지" 까지이며, 종의 진동 위상 추정·동기화는 Phase 2 의 별도 책임.
- 시작 위치에서 종이 최소 한 프레임 이상 화면에 보임 (못 보이면 SEARCH 로 시작)

### 제약
- 카메라 FOV 가 좁으면 ( D435i ~69° H × 42° V) tilt 90° 부근에서 종이 화면을 빠르게 가로질러 SEARCH 가 자주 발동될 수 있음 → `coast_lost_frames` / `hold_lost_frames` 튜닝
- D435i depth 신뢰 구간이 0.3~3m → 종 바로 아래 (0.1m 이하 horiz_dist) 에서 depth 가 부정확할 수 있음. Stop 직전 마지막 프레임은 depth 보다 tilt 가 더 신뢰할 수 있는 신호
- DXL profile_velocity 가 15Hz setpoint 변화율을 못 따라가면 tilt 추종 지연 → 제어 진동 (튜닝 항목)

### 종 vertical oscillation 대비 (§4.1 step 2.5 / 6)
기하학적으로 `horiz_dist = depth · cos(tilt)` 는 동시 측정 시 종 고도와 무관하지만, 실제로는 (a) tilt PI 의 종 추적 lag 과 (b) 측정 노이즈 때문에 종이 진동하는 동안 `horiz_dist` 가 흔들리고 `tilt_cmd` 가 종을 따라 떨림. 세 가지 방어 매커니즘:
1. **`horiz_dist` 1차 LPF (α = `horiz_dist_lp_alpha`)**: stop check + 속도 계산 모두 filt 값 사용. broadband 종 신호의 fast 성분 감쇠.
2. **tilt dead-band (`tilt_err_deadband_px`)**: 작은 `err_y_px` (≈ 0.7° 미만) 에서는 tilt 갱신 skip → 미세 진동에 tilt PI 가 chase 하지 않음.
3. **stop debounce (`stop_debounce_frames`)**: stop 조건이 연속 N 프레임 만족해야 DONE. 진동 transient 에 의한 premature stop 차단.

이 세 파라미터는 spec §4.3 표에 명시. 진동 amp/period 가 변경되면 함께 튜닝.

### 알려진 한계
- 위 세 매커니즘으로도 종 진동 amp 가 매우 크거나 (예: > 1m) period 가 LPF τ 보다 훨씬 짧으면 stop 지연이 길어질 수 있음. Phase 2 가 종 진동 위상 추정기를 가지면 visual_servo 도 그 신호를 받아 "종이 가장 낮은 시점" 동기화 stop 으로 확장 가능 (후속).
- Multi-bell / 잘못된 false positive 검출 시 잘못된 방향으로 주행. YOLO 학습 단계에서 클래스 unique 성 확보 필요

---

## 10. 미구현 / 후속

- YOLO 학습 완료 후 `detector.py` 채우기 (기존 SLAM 모드와 공유)
- `RealRobot.get_color_depth()`, `get_tilt_deg()`, `send_tilt_async()` 시리얼 wiring
- `pipeline.py` 의 기존 Phase 1 본체를 `SlamPhase1Driver` 로 추출
- 카메라 광축 yaw offset 캘리브레이션 절차 문서화
- (후속) 자동 폴백 모드 — SLAM tracking 실패 감지 시 visual_servo 로 핸드오프
