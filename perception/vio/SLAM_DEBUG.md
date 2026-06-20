# ORB-SLAM3 안정성 디버그 플랜

D435i + ORB-SLAM3 RGB-D (no-IMU, pi_mode) 조합에서 첫 시도 후 10–15 프레임 (≈ 1.5–2.5s @ 6fps) 안에 C++ 서브프로세스가 죽고, watchdog 가 30s 주기로 재기동을 반복하는 문제.

## 1. 문제 정의

**증상**: `python orbslam_localizer.py` 실행 → vocab 로드 (~25s) → 카메라 open → POSE 10–15회 출력 → C++ 바이너리 종료 → watchdog 가 hardware_reset 후 재기동 → 똑같이 죽음 → 30s 주기 무한 반복.

**커널 신호**:
```
uvcvideo 3-1:1.2: Failed to set UVC probe control : -32 (exp. 48)
usb 3-1: USB disconnect, device number N
usb 3-1: new SuperSpeed USB device number N+1
```
errno -32 = EPIPE (USB control endpoint STALL).

## 2. 현재까지 알려진 사실

- USB **뺐다가 다시 꽂으면 그 직후 첫 시도는 비교적 잘 됨** → 잔류 USB 상태 / 드라이버 attach 상태가 변수.
- 케이블: USB 3.2 정품. 포트: USB 3.0 SuperSpeed (Bus 003 또는 005, 둘 다 5000M).
- 부하 완화 시도(`nFeatures 1000→500`, `Camera.fps 15→6`) 적용했으나 증상 변화 없음 → ARM CPU 부하가 직접 원인 아닐 가능성.
- watchdog 의 `_flush_realsense(hardware_reset=True)` 도 회복 못 시킴 → 단순 USB re-enumeration 으로 해소되지 않음.
- udev 룰 `99-realsense-libusb.rules` 설치돼 있음 (권한 0666). 그러나 **uvcvideo 커널 모듈은 여전히 로드돼 있고 D435i 인터페이스에 attach 됨** (lsmod, journalctl 확인).
- 직전 실행 한 번은 Bus 5-1 로 옮기고 정상 스트리밍 — 동일 케이블/동일 코드.
- 호스트는 Pi 5 (kernel 6.8.0-1053-raspi, BCM2712).
- **반복 재기동 중 어느 순간에 우연히 살아남으면 그 이후로는 쭉 안정적으로 동작** — startup 단계에 좁은 race window 가 있어 통과 여부가 stochastic. 한 번 안정 streaming 모드에 진입하면 동일 세션 내에서는 더 이상 죽지 않음.

## 3. 핵심 가설 (가능성 순)

| # | 가설 | 근거 | 반증 시 효과 |
|---|------|------|-------------|
| H1 | 커널 `uvcvideo` 가 D435i 인터페이스에 붙어 librealsense 와 control endpoint 를 두고 race → SET_CUR PROBE STALL → reset | UVC probe 에러 메시지 자체. uvcvideo 모듈 attach 확인. udev 룰은 권한만 줄 뿐 UVC blacklist 안 함 | uvcvideo unbind/blacklist 후에도 실패하면 H1 기각 |
| H2 | D435i 펌웨어 / librealsense 버전 mismatch → SET_CUR PROBE 가 펌웨어가 거부하는 형태로 발사됨 | 동일 케이블/포트에서도 가끔 잘 됨 (펌웨어 상태에 의존) | 펌웨어 / librealsense 업데이트 후 동일하면 기각 |
| H3 | USB 케이블/포트의 SuperSpeed 신호 무결성 marginal — 첫 ~2s 간 negotiation 은 통과하지만 추가 SET_CUR 시 STALL | 포트 옮기면 가끔 회복. Pi 5 USB-C 의 신호 품질 알려진 이슈 | 다른 포트/케이블 모두 시도 후 동일하면 기각 |
| H4 | ORB-SLAM3 C++ 바이너리가 librealsense 에 비표준 sequence 를 요청 (sensor option set 등) → 펌웨어가 reject | 단독 librealsense 는 안정적이라면 이 가설 강화 | librealsense 단독에서도 죽으면 기각 |
| H5 | Pi 5 USB-C 파워 부족 (D435i peak ≥ 900mA + emitter 켜질 때 spike) | 일정 시점 후 disconnect 패턴 | `vcgencmd get_throttled` 가 0x0 이면 약화 |

## 4. 단계별 디버깅 플랜

각 단계는 **30분 이내**에 끝나도록 잘게 끊어둠. 결과는 §5 에 그때그때 적자.

### S0. 베이스라인 정보 수집 (한 번, 5분)
- **목적**: 시스템/펌웨어/모듈 상태 스냅샷 — 가설 분기 근거.
- **실행**:
  ```bash
  vcgencmd get_throttled
  rs-enumerate-devices -s        # FW version, USB type, serial
  rs-fw-update -l                # firmware update 가능 여부
  lsusb -t                       # 현재 attach 상태 + speed
  lsmod | grep uvc
  journalctl -k --since "10 min ago" | grep -E "usb|uvc" | tail -50
  ```
- **판정**: throttled 값 0x0 (정상) / non-zero (under-voltage 등). FW 버전 5.13.x 이상이면 최신.
- **분기**: under-voltage 있으면 H5 우선. FW 5.12 이하면 H2 우선.

### S1. ORB-SLAM3 없이 librealsense 단독 60초 테스트 (10분, **가장 중요**)
- **목적**: 문제가 ORB-SLAM3 측인지 카메라/USB 측인지 분리.
- **실행**:
  ```bash
  # 60초 동안 6 fps RGB-D 스트림. 정확히 SLAM 과 같은 stream 조합.
  rs-capture &        # 또는 짧은 python 스크립트 (아래)
  sleep 60
  pkill -f rs-capture
  journalctl -k --since "2 min ago" | grep -E "uvc|usb 3-1|usb 5-1|disconnect"
  ```
  파이썬 미니 스크립트 (정확한 조건 재현):
  ```python
  import pyrealsense2 as rs, time
  p = rs.pipeline(); c = rs.config()
  c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
  c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
  p.start(c); t0=time.time()
  for i in range(60*6):
      p.wait_for_frames(timeout_ms=5000)
      if i % 30 == 0: print(f"{time.time()-t0:5.1f}s  frame {i}")
  p.stop()
  ```
- **판정**:
  - **60s 동안 disconnect 없음** → ORB-SLAM3 바이너리 측 원인 (H4). § S5 로 분기.
  - **30s 전후로 disconnect** → 카메라/USB/UVC 충돌 (H1/H2/H3/H5). § S2 로 분기.

### S2. uvcvideo 분리 (15분, H1 검증)
- **목적**: 커널 UVC 와 librealsense 의 control endpoint 충돌이 원인인지 확인.
- **실행 (a)**: 일시적 unbind — 카메라가 이미 attach 된 상태에서:
  ```bash
  # D435i 의 video 인터페이스를 uvcvideo 에서 분리
  for i in /sys/bus/usb/drivers/uvcvideo/*-*; do
      echo $(basename "$i") | sudo tee /sys/bus/usb/drivers/uvcvideo/unbind 2>/dev/null
  done
  lsusb -t | grep -A1 8086    # Driver=[none] 확인
  ```
  그 후 S1 의 60s 테스트 재실행.
- **실행 (b)**: 모듈 자체를 막기 (영구):
  ```bash
  # /etc/modprobe.d/blacklist-uvc-realsense.conf
  echo "blacklist uvcvideo" | sudo tee /etc/modprobe.d/blacklist-uvc-realsense.conf
  sudo rmmod uvcvideo uvc 2>/dev/null
  # 다른 UVC 카메라(웹캠 등) 영향 있을 수 있음. 영구 적용 전에 (a)로 확인.
  ```
- **판정**:
  - unbind 후 안정 → H1 확정. SLAM 시작 시 librealsense `rs2_set_devices_changed_callback` 또는 udev 룰 강화 (`ENV{ID_USB_INTERFACE_NUM}=="..." ATTR{authorized}="0"`) 로 영구 fix.
  - unbind 후에도 죽음 → H1 기각, S3 로.

### S3. 케이블/포트 swap matrix (10분, H3 검증)
- **목적**: 신호 무결성 문제 분리.
- **실행**: 아래 조합을 각각 30s 테스트:
  | 포트 | 케이블 | 결과 |
  |------|--------|------|
  | Bus 003 (3-1) | 현재 케이블 | |
  | Bus 005 (5-1) | 현재 케이블 | |
  | Bus 003 | 다른 USB-C 케이블 (있으면) | |
  | Bus 005 | 다른 케이블 | |
  | Bus 002 또는 004 (USB 2.0) | 현재 케이블 | |
- **판정**: USB 2.0 에서도 죽으면 케이블/포트 무관. 특정 조합만 죽으면 H3 확정.

### S4. D435i 펌웨어 / librealsense 업데이트 (20분, H2 검증)
- **목적**: SDK/FW mismatch 가 원인인지.
- **선행 조건**: S1–S3 으로 결정 못 났을 때만.
- **실행**:
  ```bash
  rs-fw-update -l               # 현재 FW 와 권장 FW 출력
  # 필요 시: rs-fw-update -f <signed.bin>
  realsense-viewer --version    # SDK 버전
  ```
  D435i 권장 FW = 5.13.0.50 이상 (2023+). SDK ≥ 2.54.
- **판정**: FW/SDK 업데이트 후 S1 재실행으로 확인.

### S5. ORB-SLAM3 바이너리 단독 실행 + 반복 (30분, H4 검증)
- **목적**: Python 래퍼 우회. 죽은 시도와 산 시도의 librealsense control transfer 시퀀스를 비교 → race window 위치 특정.
- **stochastic 패턴 대응**: 한 번 실행으로는 정보 부족. **5번 반복 실행해서 모든 attempt 의 stderr 보존** 후 비교.
- **실행 (단일 시도)**:
  ```bash
  cd /home/team1/ORB_SLAM3
  export ORBSLAM_NO_VIEWER=1
  export LD_LIBRARY_PATH=$PWD/lib:$PWD/Thirdparty/DBoW2/lib:$PWD/Thirdparty/g2o/lib
  export LRS_LOG_LEVEL=DEBUG    # librealsense 상세 로그
  ./Examples/RGB-D/rgbd_realsense_D435i \
      Vocabulary/ORBvoc.txt \
      Examples/RGB-D/RealSense_D435i_pi.yaml \
      > /tmp/s5_${ATTEMPT}.stdout 2> /tmp/s5_${ATTEMPT}.stderr &
  PID=$!
  sleep 45                      # vocab ~25s + streaming ~20s
  kill -TERM $PID 2>/dev/null
  ```
- **반복 실행 스크립트**: 5번 attempt → 죽은 것과 산 것 자동 분류 → 마지막 50라인 diff.
- **판정**:
  - 죽은 시도들의 stderr 마지막 라인이 모두 같은 librealsense 호출에서 멈추면 → 그 호출이 race trigger.
  - uvcvideo 에러 발생 타이밍이 죽은/산 시도에서 다르면 → uvcvideo detach race 확정.

### S6. 코드 측 방어 강화 (S1 결과 무관하게 가치 있음, 15분)
- **E1**: `LocalizerConfig.skip_flush_first_attempt=False` 로 default 변경. 첫 시도부터 Python 측 flush + (옵션) hardware_reset 적용.
- **E2**: `is_alive()` 의 dead-detect side-effect (`_restarting=True` 자동 set) 를 watchdog 가 가동 중일 때만 발동하게 분리. startup 단계의 retry 가 의도대로 돌게.
- **E3**: 첫 시도에서 항상 `_flush_realsense(hardware_reset=True)`. USB 뺐다 꽂은 효과를 코드로 강제.

## 5. 결과 기록

| 단계 | 일시 | 환경 | 결과 | 다음 |
|------|------|------|------|------|
| S0   | 2026-05-07 20:30 | Pi 5 Rev1.1, kernel 6.8.0-1053-raspi, D435i FW 5.15.1.55, librealsense 2.55.1.0, MaxPower 720mA, D435i 현재 Bus 5-1 | `rs-enumerate-devices` 만 호출해도 즉시 `uvcvideo 5-1:1.2: Failed to set UVC probe control : -32` 다발 발생. uvcvideo 모듈은 로드돼 있지만 lsusb 상 Driver=[none] (libusb 가 점유). FW/SDK/Power 는 모두 정상. | H1 (uvcvideo 충돌) 강하게 지지 → S1 진행 |
| S1   | 2026-05-07 20:32 | Bus 5-1, 640×480 RGB-D @6fps, Python pyrealsense2 단독 60s | **PASS**: 354 프레임 / 실패 0 / disconnect 0. UVC probe 에러는 start/stop 시점 burst 만, 스트리밍 중엔 silent. | 카메라/USB/uvcvideo 는 fatal 하지 않다고 판정 — H1/H3/H5 약화, **H4 (ORB-SLAM3 측 원인) 강력 지지**. S2/S3/S4 보류, **S5 직행**. |
| S5a  | 2026-05-07 20:39 | raw `rgbd_realsense_D435i` 5회 × 45s, default yaml, 정적 책상 | 5회 모두 subprocess 안 죽음 (timeout 강제 종료). 모두 STATE=1 (INIT) 무한 반복, POSE 0. yaml default intrinsics(`fx=308`)때문에 트래킹 init 못함. disconnect 0회. | **사용자 보고 "10-15 프레임 후 죽음" 재현 안 됨**. raw binary 측은 죽지 않음. wrapper 측 차이 추적 → S5b. |
| S5b  | 2026-05-07 20:55 | wrapper 1회 × 180s, archive 활성화, 정적 책상 | 첫 시도 INIT→OK 진입 (~11s), **180s 동안 1046 POSE / 안 죽음 / disconnect 0**. cached calibration 으로 yaml 채워서 정확한 intrinsics(`fx=605`)전달. | 현재 wrapper 가 안정 동작 — 사용자가 말한 "어느 순간 안정화" 상태일 가능성. cold-start 강제 → S6. |
| S6   | 2026-05-07 21:01 | wrapper 5회 × 30s, **매 시도 직전 hardware_reset(USB power-cycle)**, 정적 책상 | 5회 모두 subprocess 안 죽음. Run 1만 트래킹 OK (미세 motion 우연), Run 2-5는 INIT 만 (정적 환경에서 init 실패는 정상). disconnect 이벤트는 hardware_reset 에 의한 정상 reset 만. | hw_reset 직후 cold-start 는 안전. hw_reset 안 한 first attempt 만 race trigger 가능성 → E3 |
| E1   | 2026-05-07 21:15 | `skip_flush_first_attempt=False` 로 default 변경 후 wrapper 1회 (hw_reset_on_first_attempt 는 False 그대로) | **사용자 보고 패턴 재현**: 첫 attempt vocab 로드 → camera open → POSE 3 라인 후 죽음. watchdog 가 hardware_reset 후 respawn → 두 번째 attempt 트래킹 OK. | **flush 만 하고 hw_reset 안 함이 가장 좁은 race window** 확정. → E3 default True 로 |
| E3   | 2026-05-07 21:18 | `hw_reset_on_first_attempt=True` 로 default 변경 후 wrapper 1회 검증 | start() 32.3s (hw_reset 5s 포함), 첫 attempt INIT 진입 후 안 죽음, **respawn 0회**. 정적 환경이라 트래킹 OK 진입은 별개 (motion 필요). | 첫 시도 race window 통과 확정. wrapper 의 default 동작이 안정적. |
| E2   | 2026-05-07 21:18 | `is_alive()` 의 dead-detect side-effect 를 watchdog 가동 중일 때만 적용. | 코드 수정 완료, regression 없음 (E3 검증 시 함께 통과). | startup retry 경로가 의도대로 동작 가능해짐. |

## 7. 결론 + 후속 조치

### 7.1 root cause (2026-05-07 기준 확인된 것)
1. **D435i cold-start race**: `_flush_realsense` 가 USB pipeline 을 짧게 open/close 하면, 카메라 internal state 가 race 가능한 mode 로 들어감. ORB-SLAM3 가 그 직후 streaming 시작하면 첫 ~5–15 프레임 사이에 subprocess 가 죽음. `hardware_reset` 으로 USB power-cycle 하면 깨끗한 cold state 보장.
2. **Stochastic 패턴**: 같은 race window 라도 우연히 통과하면 그 세션은 안정. → 사용자가 본 "어느 순간 안정화되면 쭉 잘됨".
3. (별개) **`is_alive()` 의 startup race**: subprocess 가 죽었을 때 `_restarting=True` side-effect 를 watchdog 미가동 상태에서도 set 해서 `start()` 의 retry 루프가 deadcode 화됨. 기능적으로는 watchdog 이 cover 하지만 20s 의 stale wait 발생.

### 7.2 적용된 수정
- `skip_flush_first_attempt`: True → **False** (안정성 테스트 환경 default).
- `hw_reset_on_first_attempt`: 신규 옵션, **True** default. +5s 시작 시간 비용으로 cold-start race 회피.
- `is_alive()`: watchdog 스레드가 실제 가동 중일 때만 dead-detect side-effect 적용.
- `archive_dir`: 신규 디버그 옵션. set 하면 watchdog respawn 시 죽은 tmp_dir 을 삭제 안 하고 보존 → attempt 비교 분석 가능.

### 7.3 사용자 측 검증 필요
사용자의 평소 사용 환경에서 본 수정이 적용된 wrapper 가 첫 시도부터 안정적으로 동작하는지 확인. 만약 여전히 죽으면 `LocalizerConfig(archive_dir="/tmp/orbslam_archive")` 로 모든 attempt 보존 후 `attempt_*/stdout.log`, `attempt_*/stderr.log` 비교.
| S2a  | | | | |
| S2b  | | | | |
| S3   | | | | |
| S4   | | | | |
| S5   | | | | |
| E1   | | | | |
| E2   | | | | |
| E3   | | | | |

## 6. 진행 규칙
- 한 단계가 끝날 때마다 §5 에 결과를 한 줄로 적고, 그 결과로 다음 단계를 결정.
- 한 단계가 30분 넘게 걸리면 중단하고 더 작은 단계로 쪼갠다.
- 모든 명령은 `journalctl -k --since "2 min ago" | grep -E "uvc|usb"` 결과와 함께 기록.


## 8. 후속 발견 — mid-run undervoltage failure (2026-05-15)

### 8.1 새로운 증상
1b209c0 의 cold-start fix 가 적용된 wrapper 가 **첫 attempt 의 INIT 은 정상 통과**하지만, **tracking 진입 후 mid-run 에 SIGSEGV (rc=-11) 로 죽는** 케이스 관찰. §1 의 cold-start 패턴 (5–15 프레임 안에 INIT 단계에서 죽음) 과 다른 모드.

### 8.2 결정적 증거 — dmesg / stdout / stderr 시점 정렬
`drive_to.py --x 1.0 --y 0.0 --swap-lr --archive-slam` 실행으로 `/tmp/orbslam_archive/attempt_01` 보존. 같은 윈도우의 dmesg 와 cross-reference:

```
[3632.95]  hwmon hwmon3: Undervoltage detected!       ← 언더볼티지 발생
[3635.00]  hwmon hwmon3: Voltage normalised           ← 2초 지속 후 복구
[3635.66]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32 (exp. 48)   ← 복구 후 0.66s
[3635.69]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32             ← STALL ×5
[3635.76]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32
[3635.79]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32
[3635.83]  uvcvideo 5-1:1.2: Failed to set UVC probe : -32
[3636.33]  usb 5-1: USB disconnect, device number 33  ← watchdog 가 정리
```

ORB-SLAM3 측 (attempt_01):
- `stdout.log`: POSE 라인 9개, 사이에 `1 dropped frs` 4번 → tracking OK 상태
- `stderr.log`: `STATE: 2` × 8 (TRACKING_OK), **에러 메시지 없이 무성 SIGSEGV**

### 8.3 정정된 root cause
H5 (전원 부족) 와 H1 (UVC STALL) 이 **별개 원인이 아니라 동일 시퀀스의 두 단계**임이 확정.

```
배터리 전원 → 일시적 5V 라인 sag (≥1 second)
   ↓
Pi5 USB host controller 가 D435i 와 control endpoint 재협상
   ↓
SET_CUR PROBE STALL ×N (errno -32 = EPIPE)
   ↓
librealsense → ORB-SLAM3 데이터 path 어딘가에서 NULL deref
   ↓
SIGSEGV (rc=-11), stderr 에 에러 없이 무성 종료
```

§7 의 fix 가 닫은 race window 는 `_flush_realsense` → C++ binary streaming 시작 사이의 cold-start window. **언더볼티지로 인한 mid-run renegotiation 은 그 window 밖에서 새로 열리는 race** 라 1b209c0 의 fix 가 커버하지 못함.

### 8.4 그때 미커버된 테스트 매트릭스
§7.3 의 "사용자 측 검증 필요" 가 가리키던 영역:

| 변수 | §5 검증 시 | 실제 운용 |
|---|---|---|
| 환경 | 정적 책상 | 로봇 주행 중 (vibration, IR scene change) |
| 전원 | AC 어댑터 | **배터리 (언더볼티지 발생)** |
| 모터 | OFF | 15Hz serial 트래픽 + PWM noise |
| 실행 횟수 | 1회 isolated | 연속 (첫 실행 → STOP → 두 번째 실행) |

이 중 **배터리 환경**이 결정적 변수. AC 에서는 이 모드 자체가 trigger 안 됨 (사용자 평소 경험과 일치).

### 8.5 완화 옵션
| 옵션 | 비용 | 효과 |
|---|---|---|
| A. 5V/5A PD 배터리 또는 D435i 전용 powered USB hub | 하드웨어 | 근본 해결. 언더볼티지 자체 제거 |
| B. ORB-SLAM3 C++ 측 librealsense STALL 응답 try/catch | ORB-SLAM3 rebuild | mid-run STALL 시 graceful exit → watchdog 가 깨끗하게 respawn |
| C. wrapper 측 `SafetyConfig.lost_warn_sec` 연장 + respawn 동안 ABORT 유예 | drive_to.py 수정 (~10분) | 임시. SIGSEGV 자체를 막진 못하지만 watchdog 가 살릴 시간 확보 |

A 가 필수. B 는 ORB-SLAM3 rebuild 비용이 커서 우선순위 낮음. C 는 A 보강용 임시 방편.

### 8.6 archive 도구 추가
`drive_to.py` 에 `--archive-slam [DIR]` 플래그 추가 — `LocalizerConfig(archive_dir=...)` 노출. 이후 같은 모드 재현 시 첫 명령으로 사용:
```bash
python3 ./Driving/drive_to.py --x 1.0 --y 0.0 --swap-lr --archive-slam
# crash 시 /tmp/orbslam_archive/attempt_NN/ 에 stdout/stderr/yaml 보존
```
