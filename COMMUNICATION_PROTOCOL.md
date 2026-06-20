# Pi5 ↔ OpenRB-150 통합 시리얼 통신 프로토콜 (v1.1)

> v1.1 — `TILT_ASYNC` 명령 추가 (visual-servo 주행 모드용 15Hz tilt streaming).

> 단일 OpenRB-150 보드가 모든 액추에이터(휠 DC ×2, 레벨링 DXL ×3, 카메라 틸트 DXL ×1, 로더 DXL ×1, 플라이휠 T-motor ×2)를 제어하며, Pi5 와 한 가닥의 USB CDC 시리얼로 통신한다. 본 문서는 그 시리얼 위에서 오가는 라인 단위 ASCII 프로토콜을 정의한다.
>
> 관련 문서: [SW_ARCHITECTURE.md](SW_ARCHITECTURE.md), [Driving/wheel_motor.py](Driving/wheel_motor.py), [LevelingPlatform/leveling_motor.py](LevelingPlatform/leveling_motor.py), [LevelingPlatform/openrb_sketch_reference.ino](LevelingPlatform/openrb_sketch_reference.ino)

---

## 1. 물리 채널 / 프레이밍

- USB CDC, **115200 baud, 8N1**, 7-bit ASCII
- **라인 단위**, `\n` 종료, `\r` 무시, 라인 최대 64 bytes
- Pi → OpenRB 명령은 **단일 라인**, OpenRB → Pi 응답도 **단일 라인** (sync 명령에 한해)
- `DRIVE` 만 fire-and-forget(응답 없음), 그 외 모든 명령은 sync(단일 라인 응답)
- 라인 길이 초과 → 라인 폐기 + `ERR OVERFLOW`

## 2. 액추에이터 토폴로지 / Dynamixel ID 배정

| ID | 액추에이터 | 위치 |
|---|---|---|
| 1, 2, 3 | 레벨링 플랫폼 (3-RRS) | DXL TTL bus |
| 4 | 카메라 틸트 | DXL TTL bus |
| 5 | 로더 (1발 투입) | DXL TTL bus |
| — | 휠 DC ×2 | OpenRB GPIO + 외부 H-브리지 |
| — | T-motor ×2 (플라이휠) | OpenRB PWM (또는 CAN) |

총 5개의 Dynamixel 이 하나의 TTL 데이지체인에 물린다. DC 휠 모터와 T-motor 플라이휠은 OpenRB 의 일반 출력핀으로 따로 제어한다.

## 3. Phase 정책 / Watchdog

- **무Phase Coexist**: 펌웨어는 phase 개념 없이 단순 dispatcher 로 동작. 모든 명령을 항상 수용.
- **DRIVE watchdog**: 마지막 `DRIVE` 라인 수신 후 **200 ms** 안에 다음 `DRIVE` 가 도착하지 않으면 펌웨어가 두 휠을 0 으로 강제한다. SLAM 끊김·Pi 측 hang·USB 단선 시 안전 정지를 보장한다.
- **그 외 명령은 watchdog 없음**. `SPIN <rpm> <rpm>` 으로 회전을 시작했으면 `SPIN 0 0` 또는 `STOP` 이 들어올 때까지 유지된다.

## 4. 명령어 표 (Pi → OpenRB)

| 명령 | 인자 | 응답 | 동기 | 단위·범위 |
|---|---|---|---|---|
| `PING` | — | `PONG` | sync | 헬스 체크 |
| `STATUS` | — | `S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>` | sync | 텔레메트리 |
| `STOP` | — | `OK` | sync | All-Stop (휠 0, T-motor 0, 로더 정지, DXL holding) |
| `DRIVE` | `<wL> <wR>` | (없음) | f&f | signed int **mrad/s**, ±30000, deadzone 5, **200 ms watchdog** |
| `AIM` | `<s1> <s2> <s3>` | `OK` \| `ERR <reason>` | sync | DXL step ±2047 (motion complete 까지 대기, 최대 4 s) |
| `HOME` | — | `OK` \| `ERR <reason>` | sync | 레벨링 ID 1·2·3 → 0,0,0 |
| `TILT` | `<s4>` | `OK` \| `ERR <reason>` | sync | DXL step ±2047, **양수 = 카메라 위** (motion complete 대기) |
| `TILT_ASYNC` | `<s4>` | (없음) | f&f | DXL step ±2047 (**양수 = 위**), **200 ms watchdog → 현재 위치 hold** |
| `SPIN` | `<rpmT> <rpmB>` | `OK` \| `ERR <reason>` | sync(즉시) | unsigned int rpm, 0..max\_rpm (도달 대기 X) |
| `LOAD` | — | `OK` \| `ERR <reason>` | sync | 로더(ID 5) 1사이클 회전 후 OK |
| `STRIKE` | `<rpm> <hold_ms>` | `OK` \| `ERR <reason>` | sync | 편의: `SPIN rpm rpm` → `delay(hold_ms)` → `LOAD` → `SPIN 0 0` |

### 4.1 단위 일관성

- **휠 속도**: signed int **mrad/s** (rad/s × 1000). 양수 = 전진, 부호 매핑은 `WheelMotorConfig.direction_signs` 로 보정.
- **DXL step**: signed int 절대 위치 -2048..+2047. home offset 변환은 펌웨어 측 책임.
- **T-motor RPM**: unsigned int. 음수가 들어오면 `ERR RANGE`. 양방향이 필요하면 v2 에서 부호 도메인으로 확장한다.
- **시간**: ms 단위 정수.

### 4.2 에러 코드

| 코드 | 의미 |
|---|---|
| `ERR PARSE` | 라인 파싱 실패 (인자 수 부족, 비숫자 문자 등) |
| `ERR RANGE` | 인자가 허용 범위 밖 (step, RPM, mrad/s) |
| `ERR HW` | 모터 통신 실패 (DXL 응답 없음, T-motor PWM 출력 실패 등) |
| `ERR TIMEOUT` | motion complete 대기가 4 s 를 초과 |
| `ERR OVERFLOW` | 라인 길이가 64 bytes 를 초과 |
| `ERR BUSY` | 직전 sync 명령의 모션이 끝나지 않은 상태에서 새 명령 수신 |

## 5. STATUS 응답 포맷

```
S <wL> <wR> <s1> <s2> <s3> <s4> <s5> <rpmT> <rpmB> <flags>
   │   │    └────── DXL step (signed) ─────┘   └─ T-motor rpm ┘    │
   └ wheel mrad/s ┘                                             bitmask
```

**flags 비트**:

| bit | 의미 |
|---|---|
| 0 | wheel watchdog tripped (직전 200 ms 내 `DRIVE` 없음) |
| 1 | leveling moving (ID 1·2·3 중 하나라도) |
| 2 | tilt moving (ID 4) |
| 3 | loader moving (ID 5) |
| 4 | flywheel spinning (rpm > 100) |
| 5 | error latched (마지막 `ERR …` 가 reset 안 됨) |
| 6 | leveling homed (한 번이라도 `HOME` 성공) |
| 7 | estop active (마지막 명령이 `STOP`) |

`PING` 또는 다음 정상 sync 명령 수신 시 bit 5 (error latched) 는 reset 한다.

## 6. 표준 시퀀스

### Phase 1 — Visual-servo tilt streaming (15 Hz fire-and-forget)

```
Pi → TILT_ASYNC 800      | (no reply)     # 양수 = 카메라 위
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
Pi → STOP               | OpenRB → OK     # phase 전환 직전 안전 정지
```

### Phase 2 — Aiming + 2회 타격 (모두 sync)

```
Pi → TILT 1024          | OpenRB → OK     # 카메라 90° up (양수 = 위)
Pi → AIM 100 -50 200    | OpenRB → OK     # 1차 조준
Pi → SPIN 8000 8000     | OpenRB → OK
   (Pi 측에서 ~1 s spin-up 대기)
Pi → LOAD               | OpenRB → OK     # 1발 발사
   (Pi 가 종 위치 재추정 + 재조준)
Pi → AIM 110 -45 195    | OpenRB → OK     # 2차 조준
Pi → LOAD               | OpenRB → OK     # 2발 발사
Pi → SPIN 0 0           | OpenRB → OK
Pi → TILT 0             | OpenRB → OK     # 카메라 원위치
```

또는 편의 명령:

```
Pi → STRIKE 8000 1000   | OpenRB → OK     # spin-up + load + spin-down 단일 호출
```

## 7. Pi 측 클라이언트 구조 (제안)

기존 [Driving/wheel_motor.py](Driving/wheel_motor.py) 와 [LevelingPlatform/leveling_motor.py](LevelingPlatform/leveling_motor.py) 는 각자 독립된 시리얼 인스턴스를 가정하고 있다. 단일 OpenRB 통합 시 **하나의 시리얼 핸들을 모든 facade 가 공유**해야 한다.

```
OpenRBClient                       # 단일 시리얼 owner, 라인 단위 send/recv
  ├─ WheelMotorClient(self)        # drive(), stop(), ping()    — 기존 API 유지
  ├─ LevelingMotorClient(self)     # aim(), home(), status()
  ├─ TiltClient(self)              # tilt(step)
  └─ LauncherClient(self)          # spin(t, b), load(), strike(rpm, hold)
```

각 facade 는 시리얼을 직접 소유하지 않고 `OpenRBClient.send_line(line, expect_reply)` 만 호출한다. 기존 코드 호환성을 위해 `WheelMotorClient(cfg)` 단독 생성자 형태도 남기되, 내부에서 `OpenRBClient` 를 자동 생성하도록 한다.

`STOP` 의 All-Stop 시맨틱 덕분에 어느 facade 에서 호출하든 한 줄로 모든 액추에이터를 안전 상태로 만들 수 있다. 비상 핸들러는 `OpenRBClient.stop()` 한 번만 호출하면 된다.

## 8. OpenRB 펌웨어 책임 요약

- 라인 파서 + dispatcher (참고: [openrb_sketch_reference.ino](LevelingPlatform/openrb_sketch_reference.ino))
- `DRIVE` 200 ms watchdog: 마지막 수신 시각 추적, 만료 시 두 휠 PWM 0
- DC 모터 PWM 변환 (mrad/s → duty); 인코더 PID 는 옵션
- DXL 5개 ID 관리, `AIM`·`HOME`·`TILT`·`LOAD` 의 motion-complete polling
- T-motor 출력 (PWM 또는 CAN 추상화), `SPIN` 즉시 OK 응답 (도달 대기 없음)
- 에러 latch + `STATUS` flag 갱신
- `STOP` 우선순위: 모든 모션 즉시 종료 + flywheel 0; DXL 은 holding(토크 ON 유지)

## 9. 버전·확장 정책

- 본 문서는 **v1** 이며, 새 명령 추가 시 기존 명령의 인자 형식·응답 형식은 깨뜨리지 않는다.
- 양방향 T-motor, CRC, 시퀀스 번호, 비동기 텔레메트리 push 등은 v2 이상에서 별도 검토.
- 펌웨어 버전 확인이 필요하면 `PING` 응답을 `PONG <ver>` 로 확장하는 것이 가장 가벼운 경로.
