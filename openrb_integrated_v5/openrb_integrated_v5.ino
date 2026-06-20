/**
 * ============================================================
 *  openrb_integrated_v5.ino
 *  Pi5 ↔ OpenRB-150 통합 펌웨어 (프로토콜 v1)
 *
 *  액추에이터
 *    휠 XC430    ×2 : ID 6 (LEFT), ID 7 (RIGHT)  — Velocity Mode
 *    레벨링 DXL  ×3 : ID 1, 2, 3                 — Position Mode
 *    카메라 틸트    : ID 4                        — Position Mode
 *    피더(로더)     : ID 5                        — Position Mode
 *    T-motor ESC ×2 : PWM 1000~2000 µs (Servo 라이브러리)
 *
 *  통신
 *    USB CDC 115200 8N1, 라인단위 ASCII ≤64 bytes
 *    DRIVE → fire-and-forget (응답 없음), 200 ms watchdog
 *    그 외 → sync (단일 라인 응답)
 *
 *  속도 단위 변환
 *    프로토콜 : mrad/s  (signed int, ±30000)
 *    XC430    : velocity unit = 0.229 rpm/unit
 *               mrad/s → unit = mrad/s × 0.041576
 *                 (= 1/1000 / (2π/60) / 0.229)
 *    부호     : ID6 정방향 = 양수 unit  (WHEEL_DIR_SIGN_L = +1)
 *               ID7 정방향 = 음수 unit  (WHEEL_DIR_SIGN_R = -1)
 *
 *  핀 배정
 *    T-motor TOP    : pin 9
 *    T-motor BOTTOM : pin 10
 *    (휠은 DXL TTL 버스 — 별도 GPIO 없음)
 *
 *  의존 라이브러리
 *    Dynamixel2Arduino  (ROBOTIS)
 *    Servo              (Arduino 내장)
 *
 *  변경 이력
 *    v2.7 : 로더(ID 5) 진동 저감 — velocity P↓ + I↑ 재균형.
 *           - LOADER_VEL_P_GAIN 2000 → 400, LOADER_VEL_I_GAIN(신규) = 3200.
 *           - 증상: 공 올리는 힘은 충분하나 회전 중 진동(떨림) 과다. 원인은
 *             stall 돌파용으로 극단적으로 높였던 P-gain(=2000) 이 회전 중에도
 *             속도 리플마다 PWM 을 크게 출렁이게 한 것.
 *           - 해결: P 를 낮춰 오실레이션 제거, 정지마찰 돌파는 I(적분 wind-up)
 *             가 담당하도록 부스트. loaderRotateBy / initLoaderVelMotor 양쪽에서
 *             VELOCITY_I_GAIN 도 매번 기록 (reboot 시 EEPROM default 복귀 대비).
 *    v2.6 : FF ↓ + P ↑ (도착 흔들림 추가 억제 + 정적 sag 보강)
 *           - DXL_FEEDFORWARD_ACC_GAIN 280 → 220, DXL_POSITION_P_GAIN 1500 →
 *             1800. D 는 600 그대로 유지.
 *           - 증상: v2.5 의 (FF=280, D=600) 으로 흔들림 줄었지만 완전히 안
 *             잡힘 + 정지 후 무게로 미세 sag 관측.
 *           - 해결:
 *               1) FF 한 단계 더 낮춰 (220) trajectory acc step-change 가 만든
 *                  PWM 임펄스 강도 추가 감소.
 *               2) P 를 1500 → 1800 으로 ↑ 하여 정지 시점의 작은 position
 *                  error 에도 더 강한 holding torque. sag 직접 보상.
 *           - 균형: FF↓ 와 P↑ 는 충돌 않음. FF 는 trajectory 추종 단계의 PWM
 *             보충, P 는 steady-state holding 의 stiffness — 작용 phase 가
 *             다름.
 *
 *    v2.5 : FF/D gain 균형 조정 (도착 부근 흔들림 잡기)
 *           - DXL_FEEDFORWARD_ACC_GAIN 400 → 280, DXL_POSITION_D_GAIN 400 → 600.
 *           - 증상: v2.4 의 FF 200 으로 부드러움 부족 → 400 까지 올렸더니
 *             overshoot + 도착 후 미세 흔들림 관측. FF 는 trajectory acceleration
 *             의 step-change 를 그대로 PWM 임펄스로 만들기 때문에 강할수록
 *             도착 부근 ringing 트리거.
 *           - 해결: FF 한 단계 낮춰 임펄스 강도 ↓, 동시에 D 를 한 단계 올려
 *             잔여 임펄스 흡수. P/D/FF 의 새 균형점 (P=1500, D=600, FF=280).
 *           - 변동성: 무게나 기구 특성에 따라 sweet spot 이 ±20% 범위에서
 *             움직임. 또 흔들리면 FF 를 220 까지 추가로 낮추고, sluggish 해지면
 *             FF 를 320 으로 살짝 다시 올린다.
 *
 *    v2.4 : Feedforward ACC gain 도입 (micro-trajectory transition 부드럽게)
 *           - FEEDFORWARD_2ND_GAIN default 0 → 200 으로 일괄 적용.
 *             initPosMotor 에 ff_acc_gain 파라미터 추가.
 *           - 증상: v2.3 의 D gain 으로 ringing 은 잡혔지만 "스텝 느낌"이 특히
 *             부하 받으며 내려갈 때 (helping gravity) 심함. 매 16 ms AIMF tick
 *             의 micro-trajectory 가 중력 도움으로 예상보다 빠르게 도달 → P/D
 *             가 반응하는 사이 다음 trajectory 가 새로 시작 → 매 cycle 같은
 *             초입-종단 반복이 step 으로 체감.
 *           - 해결: Trajectory generator 의 desired acceleration 을 PWM 에
 *             직접 더해 P/D 가 따라잡기 전에 미리 보상. 매 micro-trajectory
 *             간 transition 이 매끄러워짐.
 *           - 200 은 보수적 시작값. 부족하면 400~800 까지 상향 가능. 더 강한
 *             대응이 필요하면 FEEDFORWARD_1ST_GAIN (velocity FF) 도 함께 도입.
 *
 *    v2.3 : Position D gain 도입 (오버슈트·ringing 억제)
 *           - POSITION_D_GAIN default 0 → 400 으로 일괄 적용.
 *             initPosMotor 에 position_d_gain 파라미터 추가.
 *           - 증상: v2.2 의 PROFILE_VEL/ACC 상향 + v2.1 의 P gain 1500 조합에서
 *             반응성은 빨라졌지만 P=1500/D=0 underdamped 라 무게 부하 시
 *             ringing 발생. 매 AIMF tick (16 ms) 마다 새 trajectory 가 시작될
 *             때 직전 ringing 이 다 안 죽고 임펄스로 작용 → "스텝화 + 진동"
 *             체감.
 *           - 해결: D term 으로 -velocity feedback 추가 → 오버슈트 즉시 감쇠.
 *             400 은 D/P ≈ 0.27 로 살짝 over-damped 쪽. 진동 명확히 줄어듬.
 *           - 부족하면 600~800 까지 올릴 수 있음. 단 너무 높으면 encoder noise
 *             증폭으로 고주파 chatter 발생 — 청각으로 확인.
 *
 *    v2.2 : Position Mode 프로파일 가속도 ↑ (AIMF 스트리밍 stutter 완화)
 *           - DXL_PROFILE_VEL 400 → 700, DXL_PROFILE_ACC 80 → 250.
 *           - 증상: GUI 드래그로 60 Hz AIMF 스트리밍할 때 무게 부하 상황에서
 *             stutter / jerky 추종. 원인은 ACC=80 이면 max velocity 까지 가속
 *             시간이 ~320 ms 라 매 16 ms tick (=새 goal 도착) 안에 가속 phase
 *             를 못 채우고 다음 trajectory 로 덮어써지면서 매번 같은 초입
 *             동작만 반복.
 *           - ACC 250 이면 max vel 도달 ~180 ms. 한 tick 내 가속 phase 가 더
 *             많은 비율을 차지해 정상 속도 영역으로 진입 → 부드러운 추종.
 *             VEL 도 함께 700 으로 올려 큰 goal-gap 도 한 cycle 내에 부분 커버.
 *           - 부족하면 ACC 를 300~500 까지 더 올리고, 험/진동 나면 별도
 *             POSITION_D_GAIN (default 0) 을 400~600 으로 추가하면 됨.
 *
 *    v2.1 : Position Mode P gain 강화 + STATUS 응답 포맷 명문화
 *           - Dynamixel POSITION_P_GAIN 기본 800 → 1500 으로 일괄 상향
 *             (Kp_eff = P/128 기준 6.25 → 11.7). LVL_1/2/3 + TILT 에 모두 적용.
 *             initPosMotor 에 position_p_gain 파라미터 추가 (default 새 상수
 *             DXL_POSITION_P_GAIN).
 *           - 증상: HOME 시 잔여 오차가 DXL_ARRIVED_TOL=10 (~0.88°) 안에 못
 *             들어와 waitMotion → ERR TIMEOUT 반복. STATUS 로 BEFORE
 *             (-201, -102, 305) → AFTER (-6, -2, 5) 의 잔여 오차 경계 관측됨.
 *           - 해결: P gain ↑ 로 잔여 오차를 tolerance 안으로 수렴. 부족 시
 *             DXL_POSITION_P_GAIN 을 2000~3000 까지 더 올릴 수 있음.
 *           - STATUS 응답 포맷이 v1.2/v1.4 의 휠·로더 확장으로 11 필드
 *             (S wL wR s1 s2 s3 s4 s5 rpmT rpmB flags) 로 늘어났으나
 *             leveling_motor.py 파서가 옛 5 필드 포맷 그대로였음. Python 쪽
 *             status() 파서 동기화 (same-commit).
 *
 *    v2.0 : 로더(ID 5) "평상시 free, LOAD 시 잠깐 회전" 워크플로
 *           - 사용자가 손으로 로더를 자유 회전시켜 공을 적재한 뒤 LOAD 를
 *             누르면, 모터가 잠깐 깨어나 현재 위치 + 90° 까지 회전하고 다시
 *             torque off 로 돌아가도록 변경.
 *           - initLoaderVelMotor: 부팅 시 torque OFF 로 끝남 (한 번도 켜지
 * 않음).
 *           - loaderRotateBy: 진입 시 torqueOn → P-gain MOVE 보장 → 현재 위치
 *             를 baseline 으로 읽음 (누적 추적기 g_loader_goal_raw 사용 안 함,
 *             STATUS 보고용으로만 유지) → 폴링 → vel=0 → torqueOff.
 *           - handleStop: 로더는 torqueOff 로 마무리 (사용자가 손으로 풀 수
 * 있게).
 *           - 2단 P-gain (MOVE/HOLD) 제거 — torque off 상태에서 HOLD gain 은
 *             의미 없음. 단일 LOADER_VEL_P_GAIN(=2000) 만 유지.
 *           - ensureLoaderReady 단순화: HW err 체크만 (torque 상태는 LOAD 가
 *             직접 켜므로 사전 확인 불필요).
 *
 *    v1.9 : 로더(ID 5) Position → Velocity Mode 전환 (마찰 극복)
 *           - Position Mode (Step Mode 포함) 은 PWM = position_error × Kp /
 *             128 으로, 모터가 goal 에 다가갈수록 error 가 줄어 PWM 이 포화에서
 *             풀린다. 이때 잔여 토크가 정지마찰보다 작으면 목표 직전에 정지.
 *             Dynamixel Wizard 로 작은 각도 변화를 줘도 안 돌아가는 동일 현상.
 *           - Velocity Mode 는 속도 제어기 integral wind-up 이 있어 모터가
 *             막히면 PWM 이 끝까지 max 로 올라가 마찰을 뚫는다.
 *           - 로더 init 을 initVelMotor 로 변경. setup/handleRecover 분기 정리.
 *           - LOAD/STRIKE 의 로더 회전을 loaderRotateBy(delta) 헬퍼로 통합:
 *             setGoalVelocity(LOADER_VEL) → 위치 폴링 → setGoalVelocity(0).
 *             누적 목표 추적기는 그대로 → 회당 정확히 +90° advance, 회차 간
 *             누적 드리프트 없음 (코스팅 오버슈트는 상수 offset 으로 흡수).
 *           - handleStop: 로더는 setGoalVelocity(0) 으로 능동 정지.
 *           - LOADER_PROFILE_VEL/ACC 및 initPosMotor 의 op_mode 분기 사용처
 *             제거 (param 자체는 기본값으로 호환 유지).
 *           - recoverLoader: Velocity Mode init 으로 변경.
 *           - TIMEOUT 후 tracker 가 +1024 step 앞서 가는 drift 버그 수정.
 *             loaderShutdownCheckAndRecover 가 HW err 유무와 무관하게 항상
 *             g_loader_goal_raw 를 present position 으로 재동기화. 안 그러면
 *             부하 부족으로 매 LOAD 가 영원히 더 큰 목표를 갖게 되어 "재부팅
 *             도 안되고 명령도 안먹는" 상태에 빠짐.
 *           - recoverLoader 강건화: reboot 후 대기 250→500 ms, initVelMotor
 *             3 회 재시도 (200 ms 간격). 콜드 reboot race 흡수.
 *           - initLoaderVelMotor 분리: 로더 VELOCITY_P_GAIN 부스트.
 *             default 100 은 stall 시 P-term 9% PWM 만 나와 I-gain wind-up
 *             기다리다 timeout. 부스트로 stall 즉시 PWM 포화.
 *           - 2단 P-gain (MOVE=2000 / HOLD=200) — Velocity 루프에 D-gain 이
 *             없어 high-Kp idle 상태에서 작은 외란에 oscillation.
 * loaderRotateBy 가 모션 시작 시 MOVE 로 올렸다가 끝나면 HOLD 로 되돌림.
 * handleStop 도 HOLD 로 복귀시킴 (모션 도중 STOP 케이스 대비). 손가락으로 살짝
 * 건드려도 진동하던 문제 해결.
 *           - LOAD/STRIKE 진입 시 ensureLoaderReady() 사전 점검: HW err 면
 *             reboot, torque off 면 re-enable. 이전 LOAD 의 overload 잔재가
 *             다음 LOAD 를 씹어먹던 케이스 차단.
 *
 *    v1.8 : 로더(ID 5) 회전 분해능 개선 + 역회전 버그 제거 + 토크 부족 해결
 *           - LOADER_CYCLE_STEP 2047 (≈180°) → 1024 (90°). 1 LOAD = 90° CCW.
 *           - 로더만 OP_EXTENDED_POSITION 모드로 init. Position Mode (0..4095)
 *             에서는 goal 이 4095 → 0 경계를 넘을 때 모터가 짧은 각도 (CCW
 *             90°) 가 아닌 반대 방향 (CW 270°) 으로 역회전하던 문제 해결.
 *             Extended 에서는 누적 raw 가 단조 증가하므로 항상 90° CCW 단방향.
 *           - 로더만 PROFILE_VELOCITY/ACCELERATION = 0 (Step Mode) 으로 init.
 *             기존 400/80 은 출발 50 ms 시점에 PWM 17% 수준이라 무거운 공의
 *             정지마찰을 못 이김. 0/0 은 goal 즉시 적용 → PWM Limit 까지 즉발
 *             포화. Dynamixel Wizard 기본값(0/0) 과 동일한 거동.
 *           - initPosMotor 에 profile_vel/profile_acc 파라미터 추가. 모터별로
 *             프로파일을 다르게 설정할 수 있도록 (default = DXL_PROFILE_*).
 *           - LOAD/STRIKE 가 WAIT_TIMEOUT 받으면 HARDWARE_ERROR_STATUS 를
 *             읽어, 0 이 아니면 (overload 등) 자동으로 reboot → 재init → goal
 *             추적기 재캡쳐 까지 수행. 응답: "ERR OVERLOAD 0x<bits>". 사용자는
 *             LOAD 를 다시 보내면 정상 동작. (이전: torque 자가차단 후 명령이
 *             씹히는 듯 보였음 — 매번 OpenRB 리셋해야 풀렸음.)
 *           - 누적 목표 추적기 g_loader_goal_raw 추가. (cur + step) 누적 시
 *             DXL_ARRIVED_TOL(=10) 만큼의 도달 오차가 회당 누적되어 다회 회전
 *             후 위상이 어긋나는 문제 제거.
 *           - setup() 에서 로더 기준점 캡쳐. 부팅 시점의 PRESENT_POSITION 을
 *             기준점(0°) 으로 잡고 매 LOAD 가 거기서 +90° CCW 누적되도록.
 *             모터는 부팅 시 어디로도 이동하지 않는다.
 *           - handleStop() 종료 시 추적기를 현재 위치로 재동기화. STOP 으로
 *             중단된 지점부터 다음 LOAD 가 정확히 +90° 누적되도록.
 *           - handleStrike() 의 2단계 LOAD 도 같은 추적기를 사용.
 *
 *    v1.0 : 초기 통합 (DC 모터 + FF/PI)
 *    v1.1 : leveling_motor 개선사항 반영
 *           (DXL 오프셋, motion-complete 이중판정,
 *            PROFILE_VEL/ACC, setup ping 체크)
 *    v1.2 : 휠 DC+엔코더 → XC430 (ID 6,7) Velocity Mode 교체
 *           제거: MDD10A 핀, AB 엔코더 ISR, FF+PI 제어루프,
 *                 WheelCtrl 구조체
 *           추가: wheelSetVelocity(), mradToUnit() 변환,
 *                 XC430 watchdog (velocity 0 write)
 *    v1.3 : Sync Write 적용
 *           wheelSetVelocity() → ID 6·7 Goal Velocity 동시 write
 *           handleAim() / handleHome() → ID 1·2·3 Goal Position 동시 write
 *           wheelStop() → ID 6·7 velocity 0 동시 write
 *    v1.4 : 레벨링 성능 + abort 시맨틱
 *           - DXL_BAUDRATE 57600 → 1 Mbps + 부팅 시 자동 업그레이드
 *           - waitMotion() : SyncRead (LVL_1·2·3 동시), MOVING 레지스터 제거
 *           - PROFILE_VEL/ACC 200/50 → 400/80
 *           - waitMotion() 반환을 WAIT_ARRIVED/TIMEOUT/ABORTED 3-state 로 변경
 *           - 모션 핸들러가 ABORTED 시 ERR ABORTED 응답 (STOP 식별 가능)
 *           - handleStrike() delay() → drainable wait + 단계간 estop 체크
 *    v1.7 : 모터 init 견고화 + RECOVER 커맨드
 *           - ensureDxlBaud() : 첫 ping 성공 시 즉시 return 하던 버그 수정.
 *             모든 ID 를 확인하고, target baud 에 없는 모터는 factory baud
 *             에서 찾아 EEPROM 업그레이드 후 재검증. mixed-baud 상태를
 *             자동 복구.
 *           - 모터별 ping retry (기본 3회, 20 ms 간격). 콜드부팅 시 모터가
 *             아직 안정화 안 된 race 를 완화.
 *           - 부트 실패 시 모터별로 명시적 로그 송신:
 *               "ERR INIT_ID<n>"  (PING/모드/torque 설정 실패)
 *           - g_motor_init_failed[8] 로 실패한 모터를 트래킹.
 *           - 신규 RECOVER 커맨드: 실패한 모터만 다시 init 시도.
 *               응답:  "OK"                — 전부 복구 (또는 실패 모터 없음)
 *                      "ERR INIT a,b,c"   — 여전히 실패한 ID 목록
 *           - initPosMotor / initVelMotor 헬퍼로 setup/RECOVER 코드 공유.
 *
 *    v1.6 : SyncWrite packet 캐시 버그 수정
 *           - Dynamixel2Arduino 가 InfoSyncWriteInst_t 의 packet 을 cache 하여
 *             두 번째 호출부터 데이터 변경이 모터에 반영 안 되는 문제.
 *           - syncMoveLevel / wheelSetVelocity / wheelStop 매 호출 시
 *             is_info_changed = true 로 강제 → packet 재인코딩.
 *           - 증상: AIMF rapid stream 에서 첫 waypoint 만 적용되고 후속
 *             명령은 OK 응답에도 불구하고 motor 에 반영 안 됨.
 *
 *    v1.5 : 스트리밍 AIMF + estop 시맨틱 정합 + Drive Mode 강제
 *           - 비블로킹 AIMF 명령 추가 (waitMotion 없음, 즉시 OK)
 *           - GUI 드래그·연속 추종에서 stutter 해소
 *           - 도달 보장 시퀀스는 기존 AIM 유지
 *           - 모션 개시 핸들러 (AIM/AIMF/HOME/TILT/LOAD/STRIKE) 가
 *             진입 시점에 g_estop=false 로 clear.
 *             원인: SAMD21 USB-CDC reopen 시 reset 안 됨 → 이전 세션의
 *             cleanup STOP 이 g_estop 을 래치한 채 다음 세션 시작 →
 *             첫 HOME/AIM 이 waitMotion 첫 iteration 에서 ABORTED 반환.
 *           - STOP 의 모션-중단 의미는 그대로 유지 (waitMotion 이
 *             drainSerial 후 g_estop 체크).
 *           - setup() 에서 Position Mode 모터 (ID 1~5) 의 Drive Mode 를
 *             0 (Velocity-based profile) 으로 강제. Time-based profile 이면
 *             AIMF 스트리밍 시 매번 타이머가 리셋되어 모터가 실질 정지.
 * ============================================================
 */

#include <Dynamixel2Arduino.h>
#include <Servo.h>

using namespace ControlTableItem;

// ═══════════════════════════════════════════════════════════════
//  ★ 사용자 설정 영역 ★
// ═══════════════════════════════════════════════════════════════

// ── DXL 버스 ─────────────────────────────────────────────────
//   운영 baud = 1 Mbps. 모터가 출고 baud(57600) 상태면 setup() 의
//   ensureDxlBaud() 가 EEPROM 에 1 Mbps 를 한 번 기입한 뒤 재연결한다.
#define DXL_SERIAL Serial1
#define DXL_DIR_PIN -1
#define DXL_BAUDRATE 1000000UL       // 운영 baud (1 Mbps)
#define DXL_BAUDRATE_FACTORY 57600UL // auto-upgrade 경로
#define DXL_PROTOCOL 2.0f

// ── DXL ID ───────────────────────────────────────────────────
#define ID_LVL_1 1
#define ID_LVL_2 2
#define ID_LVL_3 3
#define ID_TILT 4
#define ID_LOAD 5
//   ID_WHEEL_L / ID_WHEEL_R 는 *물리적* 좌/우 휠을 지칭한다 (모터 케이스에
//   부착된 ID 스티커가 아니라 봇 진행방향 기준의 좌/우). 배선/장착이 swap
//   되어 있어 ID 7 이 물리적 왼쪽, ID 6 이 물리적 오른쪽 — 만약 봇 분해 후
//   재배선해서 swap 이 정정되면 6 ↔ 7 만 다시 바꾸면 된다.
#define ID_WHEEL_L 7
#define ID_WHEEL_R 6

// ── 휠 방향 부호 ─────────────────────────────────────────────
//   전진(양수 mrad/s) 시 각 모터가 올바른 방향으로 돌도록 조정.
//   두 모터가 봇 중심선을 기준으로 마주보게 장착되므로 좌우 부호는 보통 반대.
#define WHEEL_DIR_SIGN_L 1 // +1 or -1
#define WHEEL_DIR_SIGN_R -1

// ── 카메라 틸트(ID 4) 방향 부호 ──────────────────────────────
//   규약: 양수 step = 카메라 위 (컨트롤러 _step_from_deg + 0°→90° 스윕과 일치).
//   이 보드의 물리적 장착에서 양수=위가 되도록 +1 로 검증됨. 모터가 반대로
//   돌면 +1 ↔ -1 만 바꾸면 된다.
//   handleTilt / handleTiltAsync 양쪽에 동일하게 적용되어 sync·async 경로가
//   일치한다. (watchdog hold / STOP 은 present position 을 그대로 되쓰므로
//   부호와 무관.)
#define TILT_DIR_SIGN +1 // +1 or -1

// ── 레벨링(ID 1,2,3) 방향 부호 ───────────────────────────────
//   규약: 양수 step = leveling_sim 시각화의 양수 회전 방향.
//   세 모터 모두 봇에 장착했을 때 sim 과 정확히 반대로 회전하므로 -1.
//   stepToRaw / rawToStep 에 대칭 적용 → AIM 입력과 STATUS readback 의
//   step 축이 sim 축과 일치. 어느 한 모터만 어긋나면 해당 SIGN 만 바꾸면 됨.
#define LVL_DIR_SIGN_1 -1
#define LVL_DIR_SIGN_2 -1
#define LVL_DIR_SIGN_3 -1

// ── XC430 velocity unit 변환 ─────────────────────────────────
//   1 unit = 0.229 rpm = 0.229 × 2π/60 rad/s ≈ 0.023980 rad/s
//   mrad/s → unit : mrad/s / 1000 / 0.023980 ≈ mrad/s × 0.041701
//   (데이터시트 정확값: 0.229 rpm/unit)
#define MRAD_TO_UNIT 0.041701f

// ── XC430 velocity 상한 (unit) ────────────────────────────────
//   XC430-W150-R 무부하 최대 ≈ 60 rpm ≈ 262 unit
//   안전 마진 포함 230 unit 으로 제한
#define WHEEL_UNIT_MAX 230

// ── Position Mode DXL 오프셋 (ID 1~5, step 단위) ─────────────
//   4096 step = 1회전, 90° = 1024 step
static const int DXL_OFFSET[8] = {
    0, // [0] 미사용
    0, // [1] LVL_1  (+90°)
    0, // [2] LVL_2  (+90°)
    0, // [3] LVL_3
    0, // [4] TILT
    0, // [5] LOAD
    0, // [6] 미사용 (물리적 오른쪽 휠 — Velocity Mode)
    0  // [7] 미사용 (물리적 왼쪽 휠 — Velocity Mode)
};

// ── Position Mode 모션 판정 ───────────────────────────────────
//   MOVING 레지스터 검사는 제거 — 1 Mbps 에서도 read transaction 1 회당
//   ~300 µs 가 누적되므로 position tolerance 만으로 판정한다.
#define DXL_ARRIVED_TOL 10
#define DXL_POLL_MS 5
#define DXL_PROFILE_VEL 700 // (v1.3: 200, v1.4: 400, v2.2: 700) — AIMF 추종 ↑
#define DXL_PROFILE_ACC 250 // (v1.3:  50, v1.4:  80, v2.2: 250) — stutter 완화
// 위 두 값의 의미 (XL/XC X-series 데이터시트):
//   PROFILE_VEL : 0.229 rpm/unit. 700 unit ≈ 160 rpm ≈ 2.67 rev/s.
//   PROFILE_ACC : 214.577 rev/min² per unit. 250 unit ≈ 894 rev/min² ≈
//                 14.9 rev/s². → max vel 까지 가속 시간 ≈ 180 ms.
// AIMF 스트리밍은 60 Hz (16 ms) 로 새 goal 이 들어오므로, ACC 가 너무 낮으면
// 매 mini-trajectory 가 가속 단계 도중에 덮어써져 매번 같은 초입만 반복 →
// stutter 체감. 250 으로 올리면 한 tick 안에 정상 속도에 더 가까워져 추종이
// 부드러워짐. 무게 부하에 따라 300~500 까지 더 올릴 수 있고, 험/진동 나면
// POSITION_D_GAIN 을 0 → 400~600 으로 추가해 잡는다.
// Feedforward 2nd-order (Acceleration) gain (RAM, addr 88). default 0 → 200.
// (v2.4)
//   Trajectory generator 가 매 step 계산하는 desired acceleration 을 PWM 에
//   직접 더한다. P/D 는 오차 기반 reactive 인 반면 FF 는 trajectory 모양을
//   따라가도록 proactive 하게 PWM 을 인가 → micro-trajectory 간 transition
//   부드러워짐.
//   "내려갈 때" 특히 효과적: 중력이 helping force 라 trajectory desired acc
//   가 음수인 구간에서 모터가 trajectory 보다 빠르게 도달하던 오버슈트를
//   FF 의 음수 PWM 이 미리 보상.
//   200→400 까지 올리니 overshoot/도착 후 흔들림 관측. 280 으로 낮춰 + D ↑ 로
//   잔여 임펄스 흡수했지만 (v2.5) 여전히 흔들림 남음. 220 으로 추가 하향
//   (v2.6) — FF 임펄스 강도를 한 단계 더 줄임. step 느낌 다시 강해지면 250 쯤
//   으로 살짝 복귀.
#define DXL_FEEDFORWARD_ACC_GAIN 220
// Position D gain (RAM, addr 80). default 0 → 400. (v2.3)
//   D term 은 position_error 의 미분 → 사실상 -velocity feedback. P gain 만 큰
//   상태에서는 P 가 만든 가속이 정지마찰·관성을 지나 오버슈트 → ringing 으로
//   이어지기 쉽다. D 가 0 이면 ringing 이 깨끗하게 안 죽고 다음 AIMF tick 의
//   새 trajectory 시작에 임펄스로 작용 → "스텝화 + 진동" 체감.
//   400 은 P=1500 기준 D/P ≈ 0.27 로 살짝 over-damped 쪽. FF 임펄스가 만든
//   잔여 흔들림이 보여 600 (D/P ≈ 0.40) 으로 한 단계 강화. (v2.5)
//   필요 시 800 까지 올릴 수 있음. 너무 높으면 고주파 chatter (encoder noise
//   가 D term 으로 증폭) — 모터에서 "치치치" 또는 "윙~윙" 톤 들리면 즉시
//   한 단계 낮춤.
#define DXL_POSITION_D_GAIN 600
// Position P gain (RAM, addr 84). default 800 → 1500. (v2.1)
//   Kp_eff = P_GAIN / 128. PWM ∝ position_error × Kp_eff. goal 근처에서 잔여
//   오차가 ±DXL_ARRIVED_TOL(=10 step ≈ 0.88°) 안에 못 들어오면 waitMotion
//   TIMEOUT. 3-RRS 기구 마찰 + PROFILE_VEL/ACC 공격적 setting 조합에서 default
//   800 은 잔여 오차 ±15~30 까지 남는 케이스가 다수 관측됨 (HOME 시 STATUS
//   확인으로 진단). 1500 으로 올려 수렴 강도 ↑ (v2.1).
//   v2.6: 정지 후 무게로 인한 미세 sag 관측 → 1500 → 1800 으로 stiffness ↑.
//   필요 시 2000~3000 까지 더 올릴 수 있음. 너무 높으면 험/오버슈트.
#define DXL_POSITION_P_GAIN 1800
// ── 로더(ID 5) Velocity Mode 회전 ────────────────────────────
//   Position Mode 는 (Step Mode 라도) PWM = error × Kp / 128 이라 모터가 goal
//   에 다가갈수록 PWM 이 포화에서 풀려 토크가 급감 → 정지마찰을 못 이기고
//   목표 직전에 멈추는 현상. Velocity Mode 는 속도 PI 의 integral wind-up 이
//   있어 모터가 막히면 PWM 이 끝까지 max 로 올라가 마찰을 뚫음.
//
//   LOADER_VEL          : Velocity 명령 raw unit (0.229 rpm/unit, signed).
//                         200 unit ≈ 45.8 rpm → 90° (1024 step) ≈ 0.33 s.
//                         (XM430 기본 Velocity Limit 230 이내. 더 빠르게 하려면
//                          EEPROM Velocity Limit 도 같이 올려야 함.)
//   LOADER_VEL_P_GAIN   : 모션 중 velocity P-gain.
//   LOADER_VEL_I_GAIN   : 모션 중 velocity I-gain.
//
//   v2.7 진동 저감: 예전 P-gain=2000 (default 100 의 20배) 은 stall 시 P-term
//   만으로 PWM 을 즉발 포화시켜 정지마찰을 돌파했지만, 같은 게인이 *회전 중*
//   에는 작은 속도 리플마다 PWM 을 크게 출렁이게 만들어 모터가 떨림(진동).
//   힘은 충분하나 진동 과다 → P 를 낮추고(오실레이션 제거) 정지마찰 돌파는
//   I(적분 wind-up)에게 넘긴다: stall 시 ∫error 가 누적·포화되어 PWM 을 끝까지
//   밀어 정지마찰을 뚫는다 (P 보다 약간 느리지만 부드럽다). default I=1920 보다
//   부스트해 돌파 속도를 보존. 진동이 더 줄길 원하면 P 를 더 낮추고(예: 200),
//   힘/응답이 부족하면 I 를 올린다(예: 4000).
//
//   v2.0: 로더는 "평상시 torque off, LOAD 시 잠깐 torque on" 모드. 사용자가
//   손으로 자유 회전시켜 공을 적재한 뒤 LOAD 를 누르면, 모터가 잠깐 깨어나
//   "현재 위치 + 90°" 까지 회전하고 다시 torque off 로 돌아감. 진동 / holding
//   stall 문제 자체가 사라짐.
#define LOADER_VEL 400
#define LOADER_VEL_P_GAIN 500
#define LOADER_VEL_I_GAIN 4000
// 오버슈트 방지 — 마지막 구간 감속 + 정지 후 active brake.
//   LOADER_VEL_SLOW   : 감속 구간 속도 (raw). 폴링 지연으로 인한 오버슈트를
//                       이 속도 기반으로 결정. 80 raw ≈ 18 rpm = 1.25 step/ms.
//   LOADER_BRAKE_ZONE : next 까지 남은 step 이 이 값 이하면 SLOW 로 전환.
//                       180 step ≈ 16° — 1024 step (90°) 의 ~18%.
//   LOADER_BRAKE_MS   : setGoalVelocity(0) 후 torqueOff 까지 대기. 이 동안
//                       velocity PI loop 가 active brake 로 관성 흡수.
#define LOADER_VEL_SLOW 80
#define LOADER_BRAKE_ZONE 180
#define LOADER_BRAKE_MS 80UL

// ── T-motor ESC ──────────────────────────────────────────────
#define PIN_ESC_TOP 9
#define PIN_ESC_BOT 10
#define ESC_MIN_US 1000
#define ESC_MAX_US 2000
#define TMOTOR_MAX_RPM 10000

// ── 프로토콜 상수 ────────────────────────────────────────────
#define LINE_MAX 64
#define WATCHDOG_MS 200UL
#define MOTION_TIMEOUT 4000UL
#define WHEEL_MRAD_MAX 30000
#define WHEEL_DEADZONE 5
#define DXL_STEP_MIN -2047
#define DXL_STEP_MAX 2047
// 로더(ID 5) 1회 LOAD 당 회전량 (raw step). 4096 step = 360° 이므로
// 1024 step = 90°. DRIVE_MODE bit0=0(Normal) 에서 +값 = CCW.
#define LOADER_CYCLE_STEP 1024

// ═══════════════════════════════════════════════════════════════
//  전역 객체 / 변수
// ═══════════════════════════════════════════════════════════════

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
Servo escTop, escBot;

// 휠
uint32_t g_last_drive_ms = 0;
bool g_watchdog_tripped = false;
bool g_drive_active = false;

// TILT_ASYNC fire-and-forget watchdog (200 ms hold-in-place)
uint32_t g_last_tilt_async_ms = 0;

// T-motor
uint16_t g_rpmTop = 0, g_rpmBot = 0;

// Position Mode DXL
bool g_leveling_moving = false;
bool g_tilt_moving = false;
bool g_loader_moving = false;
bool g_err_latched = false;
bool g_homed = false;
bool g_estop = false;
int32_t g_dxl_target[8] = {0};
bool g_motion_busy = false; // waitMotion() 실행 중 guard

// 로더 누적 목표 위치 (raw, multi-turn). 매 LOAD 마다 +LOADER_CYCLE_STEP 가산.
// 로더는 Velocity Mode (v1.9) — present position 은 multi-turn 으로 보고되므로
// wrap 없이 단조 증가. loaderRotateBy 가 폴링 종료 조건 (present ≥ next - tol)
// 으로 이 값을 사용.
//   초기값은 setup() 에서 부팅 시점의 PRESENT_POSITION 으로 덮어쓴다 —
//   "전원 켤 때 모터가 놓여 있던 각도" 가 기준점이고, 매 LOAD 가 거기서
//   +90° CCW 누적된다. 모터는 부팅 시 어디로도 이동하지 않는다.
int32_t g_loader_goal_raw = 0;

// setup() 에서 init (ping + mode + torqueOn) 에 실패한 모터를 트래킹.
// RECOVER 커맨드가 이 플래그가 set 된 ID 만 다시 시도한다.
// 인덱스 = DXL ID (1..7).
bool g_motor_init_failed[8] = {false};

// 시리얼 버퍼
char g_buf[LINE_MAX + 2];
uint8_t g_buf_idx = 0;

// ── Sync Write 버퍼 ───────────────────────────────────────────
// Goal Velocity (addr 104, 4 bytes signed) — 휠 ID 6·7
DYNAMIXEL::InfoSyncWriteInst_t g_sw_vel;
DYNAMIXEL::XELInfoSyncWrite_t g_sw_vel_xel[2];
int32_t g_sw_vel_data[2]; // [0]=L, [1]=R

// Goal Position (addr 116, 4 bytes) — 레벨링 ID 1·2·3
DYNAMIXEL::InfoSyncWriteInst_t g_sw_pos;
DYNAMIXEL::XELInfoSyncWrite_t g_sw_pos_xel[3];
int32_t g_sw_pos_data[3]; // [0]=LVL1 [1]=LVL2 [2]=LVL3

// ── Sync Read 버퍼 ────────────────────────────────────────────
// Present Position (addr 132, 4 bytes) — 레벨링 ID 1·2·3
// waitMotion() 의 3축 동시 위치 폴링에 사용 → transaction 3→1.
DYNAMIXEL::InfoSyncReadInst_t g_sr_pos;
DYNAMIXEL::XELInfoSyncRead_t g_sr_pos_xel[3];
int32_t g_sr_pos_data[3]; // [0]=LVL1 [1]=LVL2 [2]=LVL3

// ── waitMotion 반환 코드 ──────────────────────────────────────
// ARRIVED  : 모든 모터가 tolerance 내 도달
// TIMEOUT  : 시간 초과 (HW 문제 가능성)
// ABORTED  : 대기 중 STOP 수신 → g_estop 가 true 로 전환
enum WaitResult : uint8_t {
  WAIT_ARRIVED = 0,
  WAIT_TIMEOUT = 1,
  WAIT_ABORTED = 2,
};

// ═══════════════════════════════════════════════════════════════
//  휠 (XC430 Velocity Mode)
// ═══════════════════════════════════════════════════════════════

/**
 * mrad/s → XC430 velocity unit 변환
 *   deadzone 미만 → 0, 결과 ±WHEEL_UNIT_MAX 클램핑
 */
int32_t mradToUnit(int32_t mrad, int dir_sign) {
  if (abs(mrad) < WHEEL_DEADZONE)
    return 0;
  float unit = (float)mrad * MRAD_TO_UNIT * (float)dir_sign;
  return (int32_t)constrain((long)unit, -WHEEL_UNIT_MAX, WHEEL_UNIT_MAX);
}

/** 두 휠에 속도 지령 — Sync Write (ID 6·7 동시) */
void wheelSetVelocity(int32_t mrad_L, int32_t mrad_R) {
  g_sw_vel_data[0] = mradToUnit(mrad_L, WHEEL_DIR_SIGN_L);
  g_sw_vel_data[1] = mradToUnit(mrad_R, WHEEL_DIR_SIGN_R);
  g_sw_vel.is_info_changed = true; // 매 호출 packet 재인코딩 (캐시 재사용 방지)
  dxl.syncWrite(&g_sw_vel);
}

/** 두 휠 즉시 정지 — Sync Write (velocity = 0 동시) */
void wheelStop() {
  g_sw_vel_data[0] = 0;
  g_sw_vel_data[1] = 0;
  g_sw_vel.is_info_changed = true;
  dxl.syncWrite(&g_sw_vel);
}

/**
 * 현재 실제 속도를 mrad/s 로 읽기 (STATUS 응답용)
 *   getPresentVelocity() = signed velocity unit
 */
int32_t wheelReadMrad(uint8_t id, int dir_sign) {
  int32_t unit = (int32_t)dxl.getPresentVelocity(id);
  return (int32_t)((float)unit / MRAD_TO_UNIT / (float)dir_sign);
}

// ═══════════════════════════════════════════════════════════════
//  T-motor ESC
// ═══════════════════════════════════════════════════════════════

uint16_t rpmToUs(uint16_t rpm) {
  if (rpm == 0)
    return ESC_MIN_US;
  uint16_t r = min(rpm, (uint16_t)TMOTOR_MAX_RPM);
  return (uint16_t)(ESC_MIN_US +
                    ((uint32_t)r * (ESC_MAX_US - ESC_MIN_US)) / TMOTOR_MAX_RPM);
}

void setTmotor(uint16_t rpmT, uint16_t rpmB) {
  escTop.writeMicroseconds(rpmToUs(rpmT));
  escBot.writeMicroseconds(rpmToUs(rpmB));
  g_rpmTop = rpmT;
  g_rpmBot = rpmB;
}

// ═══════════════════════════════════════════════════════════════
//  Position Mode DXL 유틸리티 (ID 1~5)
// ═══════════════════════════════════════════════════════════════

// 레벨링 ID 1..3 에 대해 방향 부호 lookup. 그 외 ID 는 +1 (no-op).
static inline int32_t lvlDirSign(uint8_t id) {
  switch (id) {
  case ID_LVL_1: return LVL_DIR_SIGN_1;
  case ID_LVL_2: return LVL_DIR_SIGN_2;
  case ID_LVL_3: return LVL_DIR_SIGN_3;
  default:       return 1;
  }
}

int32_t stepToRaw(uint8_t id, int32_t step) {
  int32_t signed_step = lvlDirSign(id) * step;
  int32_t adj = constrain(signed_step + DXL_OFFSET[id], DXL_STEP_MIN, DXL_STEP_MAX);
  return adj + 2048;
}

int32_t rawToStep(uint8_t id, int32_t raw) {
  return lvlDirSign(id) * ((raw - 2048) - DXL_OFFSET[id]);
}

int32_t dxlReadStep(uint8_t id) {
  return rawToStep(id, (int32_t)dxl.getPresentPosition(id));
}

bool dxlMove(uint8_t id, int32_t step) {
  int32_t raw = stepToRaw(id, step);
  if (!dxl.setGoalPosition(id, (uint32_t)raw))
    return false;
  g_dxl_target[id] = raw;
  return true;
}

/**
 * 레벨링 3축 Sync Write — Goal Position 동시 전송
 *   s1/s2/s3: 각 축 목표 step
 *   성공 true, 실패 false
 */
bool syncMoveLevel(int32_t s1, int32_t s2, int32_t s3) {
  g_sw_pos_data[0] = stepToRaw(ID_LVL_1, s1);
  g_sw_pos_data[1] = stepToRaw(ID_LVL_2, s2);
  g_sw_pos_data[2] = stepToRaw(ID_LVL_3, s3);
  g_dxl_target[ID_LVL_1] = g_sw_pos_data[0];
  g_dxl_target[ID_LVL_2] = g_sw_pos_data[1];
  g_dxl_target[ID_LVL_3] = g_sw_pos_data[2];
  // 매 호출마다 packet 재인코딩 강제. 미설정 시 라이브러리가 이전 packet 을
  // 캐시 재사용하여 첫 syncWrite 이후의 데이터 갱신이 모터에 반영 안 되는
  // 증상 발생 (AIMF rapid stream 에서 첫 waypoint 만 적용되는 원인).
  g_sw_pos.is_info_changed = true;
  return dxl.syncWrite(&g_sw_pos);
}

// dispatch() 전방 선언 — waitMotion() 내부에서 시리얼 드레인 시 호출
void dispatch(char *line);

// 모터 init 헬퍼 전방 선언 — handleRecover() 가 정의보다 위에서 호출
static bool pingRetry(uint8_t id, uint8_t retries = 3);
static bool initPosMotor(uint8_t id, uint8_t op_mode = OP_POSITION,
                         uint32_t profile_vel = DXL_PROFILE_VEL,
                         uint32_t profile_acc = DXL_PROFILE_ACC,
                         uint16_t position_p_gain = DXL_POSITION_P_GAIN,
                         uint16_t position_d_gain = DXL_POSITION_D_GAIN,
                         uint16_t ff_acc_gain = DXL_FEEDFORWARD_ACC_GAIN);
static bool initVelMotor(uint8_t id);
static bool initLoaderVelMotor();
static bool ensureLoaderReady();
static bool recoverLoader();
static bool drainableWait(uint32_t ms);

/**
 * waitMotion() 내부 시리얼 드레인
 *   블로킹 대기 중에도 시리얼 버퍼(64 bytes)가 넘치지 않도록
 *   수신된 바이트를 즉시 소비하고 완성된 라인은 dispatch()로 처리.
 *
 *   재귀 안전:
 *     waitMotion() 중 Pi 는 현재 sync 명령의 OK/ERR 을 기다리므로
 *     AIM·HOME·TILT·LOAD·STRIKE(= waitMotion 호출 명령)를 보내지 않음.
 *     따라서 dispatch() → waitMotion() 재귀는 실제로 발생하지 않음.
 */
static void drainSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r')
      continue;
    if (c == '\n') {
      g_buf[g_buf_idx] = '\0';
      if (g_buf_idx > 0)
        dispatch(g_buf);
      g_buf_idx = 0;
    } else if (g_buf_idx < LINE_MAX) {
      g_buf[g_buf_idx++] = c;
    } else {
      g_buf_idx = 0;
      while (Serial.available() && Serial.read() != '\n')
        ;
      sendErr("OVERFLOW");
    }
  }
}

/**
 * motion-complete 판정 (position tolerance only)
 *   - LVL_1·2·3 batch : SyncRead 1 회로 3축 present position 동시 갱신.
 *   - 그 외 (TILT/LOAD/단일축) : per-id getPresentPosition.
 *   - 폴링 간격마다 drainSerial() 호출 → 버퍼 오버플로우 방지.
 *
 * 반환
 *   WAIT_ARRIVED  : 모두 tolerance 내 도달
 *   WAIT_TIMEOUT  : timeout 만료
 *   WAIT_ABORTED  : 대기 중 STOP 수신 (g_estop=true) → 호출자에게 명시적으로
 * 알림 (STOP 이 target 을 current 로 만들어 ARRIVED 처럼 보이는 가짜 성공을
 * 방지)
 */
WaitResult waitMotion(const uint8_t *ids, uint8_t n,
                      uint32_t timeout_ms = MOTION_TIMEOUT) {
  bool done[8] = {false};
  g_motion_busy = true;
  uint32_t start = millis();

  // Fast path: 3-축 레벨링 일괄 폴링 (transaction 3→1)
  const bool lvl_batch = (n == 3 && ids[0] == ID_LVL_1 && ids[1] == ID_LVL_2 &&
                          ids[2] == ID_LVL_3);

  while (millis() - start < timeout_ms) {
    drainSerial();

    // drainSerial() 가 handleStop() 을 호출했다면 g_estop=true.
    // STOP 은 g_dxl_target 을 current 로 바꿔 다음 폴링에서 arrived 가
    // true 로 보이므로, 여기서 명시적으로 가로채 가짜 성공을 막는다.
    if (g_estop) {
      g_motion_busy = false;
      return WAIT_ABORTED;
    }

    bool any_pending = false;

    if (lvl_batch) {
      g_sr_pos.is_info_changed = true;
      uint8_t recv = dxl.syncRead(&g_sr_pos);
      if (recv == 3) {
        for (uint8_t i = 0; i < 3; i++) {
          if (done[i])
            continue;
          int32_t present = g_sr_pos_data[i];
          bool arrived =
              (abs(present - g_dxl_target[ids[i]]) <= DXL_ARRIVED_TOL);
          if (arrived)
            done[i] = true;
          else
            any_pending = true;
        }
      } else {
        any_pending = true; // 통신 실패 — 재시도
      }
    } else {
      for (uint8_t i = 0; i < n; i++) {
        if (done[i])
          continue;
        uint8_t id = ids[i];
        int32_t present = (int32_t)dxl.getPresentPosition(id);
        bool arrived = (abs(present - g_dxl_target[id]) <= DXL_ARRIVED_TOL);
        if (arrived)
          done[i] = true;
        else
          any_pending = true;
      }
    }

    if (!any_pending) {
      g_motion_busy = false;
      return WAIT_ARRIVED;
    }
    delay(DXL_POLL_MS);
  }
  g_motion_busy = false;
  return WAIT_TIMEOUT;
}

/**
 * waitMotion 결과를 ERR 응답으로 변환 (ABORTED 와 TIMEOUT 구분).
 *   handleAim/Home/Tilt/Load/Strike 공통.
 */
void replyByResult(WaitResult r) {
  switch (r) {
  case WAIT_ARRIVED:
    sendOk();
    break;
  case WAIT_TIMEOUT:
    sendErr("TIMEOUT");
    break;
  case WAIT_ABORTED:
    sendErr("ABORTED");
    break;
  }
}

// ═══════════════════════════════════════════════════════════════
//  응답 헬퍼
// ═══════════════════════════════════════════════════════════════

void sendOk() { Serial.println("OK"); }
void sendErr(const char *reason) {
  Serial.print("ERR ");
  Serial.println(reason);
  g_err_latched = true;
}

// ═══════════════════════════════════════════════════════════════
//  명령 핸들러
// ═══════════════════════════════════════════════════════════════

void handlePing() {
  g_err_latched = false;
  Serial.println("PONG");
}

void handleStatus() {
  int32_t wL = wheelReadMrad(ID_WHEEL_L, WHEEL_DIR_SIGN_L);
  int32_t wR = wheelReadMrad(ID_WHEEL_R, WHEEL_DIR_SIGN_R);

  int32_t s[6];
  for (uint8_t id = 1; id <= 5; id++)
    s[id] = dxlReadStep(id);

  uint8_t flags = 0;
  if (g_watchdog_tripped)
    flags |= (1 << 0);
  if (g_leveling_moving)
    flags |= (1 << 1);
  if (g_tilt_moving)
    flags |= (1 << 2);
  if (g_loader_moving)
    flags |= (1 << 3);
  if (g_rpmTop > 100 || g_rpmBot > 100)
    flags |= (1 << 4);
  if (g_err_latched)
    flags |= (1 << 5);
  if (g_homed)
    flags |= (1 << 6);
  if (g_estop)
    flags |= (1 << 7);

  // 출력 포맷: S wL wR p1 p2 p3 p4 p5 rpmT rpmB flags   (11 fields, v1.2+)
  Serial.print("S ");
  Serial.print(wL);
  Serial.print(' ');
  Serial.print(wR);
  Serial.print(' ');
  for (uint8_t id = 1; id <= 5; id++) {
    Serial.print(s[id]);
    Serial.print(' ');
  }
  Serial.print(g_rpmTop);
  Serial.print(' ');
  Serial.print(g_rpmBot);
  Serial.print(' ');
  Serial.println(flags);
}

void handleStop() {
  wheelStop();
  g_drive_active = false;
  g_watchdog_tripped = false;

  setTmotor(0, 0);

  // Position Mode 모터 (ID 1~4): 현재 위치로 goal 고정 → 그 자리에 잠금.
  for (uint8_t id = 1; id <= 4; id++) {
    int32_t cur = (int32_t)dxl.getPresentPosition(id);
    dxl.setGoalPosition(id, (uint32_t)cur);
    g_dxl_target[id] = cur;
  }
  // 로더(ID 5) 는 Velocity Mode — vel 0 으로 명령 비움 + torqueOff 로 사용자가
  // 손으로 다시 자유 회전 가능한 상태로 복귀 (v2.0 평상시 free state 와 동일).
  // 다음 LOAD 는 어차피 현재 present 를 baseline 으로 새로 잡으므로 tracker
  // 재동기화도 같이.
  dxl.setGoalVelocity(ID_LOAD, 0);
  dxl.torqueOff(ID_LOAD);
  g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
  g_dxl_target[ID_LOAD] = g_loader_goal_raw;

  g_leveling_moving = false;
  g_tilt_moving = false;
  g_loader_moving = false;
  g_estop = true;
  g_err_latched = false;
  sendOk();
}

void handleDrive(char *args) {
  // DRIVE 는 fire-and-forget (프로토콜 §4 "응답 없음"). 에러 경로에서도 절대
  // 라인을 송신하지 않는다 — stale ERR 한 줄이 버퍼에 남으면 다음 sync 명령
  // (PING/STOP/TILT)의 readline() 이 그걸 OK/PONG 대신 읽어 응답이 desync 됨.
  // 그래서 parse 실패는 조용히 drop, 범위 초과는 ERR 대신 클램프 (TILT_ASYNC 와
  // 동일 패턴).
  long vL, vR;
  if (sscanf(args, "%ld %ld", &vL, &vR) != 2) {
    return;
  }
  vL = constrain(vL, (long)-WHEEL_MRAD_MAX, (long)WHEEL_MRAD_MAX);
  vR = constrain(vR, (long)-WHEEL_MRAD_MAX, (long)WHEEL_MRAD_MAX);

  wheelSetVelocity((int32_t)vL, (int32_t)vR);
  g_last_drive_ms = millis();
  g_watchdog_tripped = false;
  g_drive_active = true;
  g_estop = false;
  // 응답 없음 (fire-and-forget)
}

void handleAim(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long s1, s2, s3;
  if (sscanf(args, "%ld %ld %ld", &s1, &s2, &s3) != 3) {
    sendErr("PARSE");
    return;
  }
  if (s1 < DXL_STEP_MIN || s1 > DXL_STEP_MAX || s2 < DXL_STEP_MIN ||
      s2 > DXL_STEP_MAX || s3 < DXL_STEP_MIN || s3 > DXL_STEP_MAX) {
    sendErr("RANGE");
    return;
  }

  // 명시적 모션 개시 — 이전 세션의 STOP 래치를 해제.
  // (SAMD21 USB-CDC reopen 은 reset 안 되므로 globals 가 세션 간 유지됨)
  g_estop = false;

  if (!syncMoveLevel((int32_t)s1, (int32_t)s2, (int32_t)s3)) {
    sendErr("HW");
    return;
  }

  g_leveling_moving = true;
  const uint8_t ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3};
  WaitResult r = waitMotion(ids, 3);
  g_leveling_moving = false;
  replyByResult(r);
}

/**
 * handleAimF — 비블로킹 (streaming) AIM.
 *   syncWrite 로 GOAL_POSITION 만 갱신하고 즉시 OK 반환.
 *   Dynamixel 서보는 모션 중 GOAL_POSITION 덮어쓰기를 자체 지원 (재orientation)
 *   하므로, GUI 드래그처럼 60 Hz 로 연속 갱신해도 stutter 없이 추종한다.
 *
 *   AIM 과의 차이: waitMotion() 호출 없음 → host 가 RTT (~3 ms) 만에 다음
 *   명령 송신 가능. "도달 보장" 이 필요한 시퀀스에는 AIM 을 쓸 것.
 *
 *   g_leveling_moving 은 true 로 두고 (사용자가 명령 중인 상태), STOP/HOME 이
 *   클리어한다.
 */
void handleAimF(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long s1, s2, s3;
  if (sscanf(args, "%ld %ld %ld", &s1, &s2, &s3) != 3) {
    sendErr("PARSE");
    return;
  }
  if (s1 < DXL_STEP_MIN || s1 > DXL_STEP_MAX || s2 < DXL_STEP_MIN ||
      s2 > DXL_STEP_MAX || s3 < DXL_STEP_MIN || s3 > DXL_STEP_MAX) {
    sendErr("RANGE");
    return;
  }
  g_estop = false; // 명시적 모션 개시
  if (!syncMoveLevel((int32_t)s1, (int32_t)s2, (int32_t)s3)) {
    sendErr("HW");
    return;
  }
  g_leveling_moving = true;
  sendOk(); // ← waitMotion 안 함, 즉시 응답
}

void handleHome() {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  g_estop = false; // 명시적 모션 개시
  if (!syncMoveLevel(0, 0, 0)) {
    sendErr("HW");
    return;
  }

  g_leveling_moving = true;
  const uint8_t ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3};
  WaitResult r = waitMotion(ids, 3);
  g_leveling_moving = false;
  if (r != WAIT_ARRIVED) {
    replyByResult(r);
    return;
  }
  g_homed = true;
  sendOk();
}

void handleTilt(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long s4;
  if (sscanf(args, "%ld", &s4) != 1) {
    sendErr("PARSE");
    return;
  }
  if (s4 < DXL_STEP_MIN || s4 > DXL_STEP_MAX) {
    sendErr("RANGE");
    return;
  }
  g_estop = false; // 명시적 모션 개시
  if (!dxlMove(ID_TILT, (int32_t)(TILT_DIR_SIGN * s4))) {
    sendErr("HW");
    return;
  }

  g_tilt_moving = true;
  const uint8_t ids[] = {ID_TILT};
  WaitResult r = waitMotion(ids, 1);
  g_tilt_moving = false;
  // Refresh TILT_ASYNC watchdog so the hold-rewrite doesn't fire immediately
  // after a sync TILT (semantically harmless, but wastes a setGoalPosition).
  g_last_tilt_async_ms = millis();
  replyByResult(r);
}

void handleTiltAsync(char *args) {
  // Fire-and-forget tilt setpoint. No motion-complete poll, no reply.
  // 200 ms watchdog (g_last_tilt_async_ms) holds-in-place if stream stalls.
  long s4;
  if (sscanf(args, "%ld", &s4) != 1) {
    return;   // silently drop parse errors — f&f path
  }
  if (s4 < DXL_STEP_MIN) s4 = DXL_STEP_MIN;
  if (s4 > DXL_STEP_MAX) s4 = DXL_STEP_MAX;
  if (g_motion_busy) {
    return;   // sync TILT motion-complete in progress — drop
  }
  g_estop = false;
  dxlMove(ID_TILT, (int32_t)(TILT_DIR_SIGN * s4));
  g_last_tilt_async_ms = millis();
}

void handleSpin(char *args) {
  long rT, rB;
  if (sscanf(args, "%ld %ld", &rT, &rB) != 2) {
    sendErr("PARSE");
    return;
  }
  if (rT < 0 || rB < 0 || rT > TMOTOR_MAX_RPM || rB > TMOTOR_MAX_RPM) {
    sendErr("RANGE");
    return;
  }
  setTmotor((uint16_t)rT, (uint16_t)rB);
  sendOk();
}

/**
 * loaderShutdownCheckAndRecover — TIMEOUT 직후 후속 처리.
 *
 *   1) 추적기 재동기화 (항상): loaderRotateBy 가 polling 전에 이미 g_loader_
 *      goal_raw 를 +delta 만큼 advance 시킨 상태로 timeout 됐다는 뜻 → tracker
 *      가 실제 위치보다 앞서 있음. 그대로 두면 다음 LOAD 목표가 실제+2048 이
 *      되어 더 무거운 일이 되고, 결국 영원히 도달 못함. present 로 되돌려
 *      다음 LOAD 가 "현재 자리 + 90°" 로 정상화되게 함.
 *
 *   2) HARDWARE_ERROR_STATUS 읽기 — 0 이 아니면 (overload/overheat 등) reboot
 *      + 재init 으로 클리어. Velocity Mode 에서는 4 초 timeout 안에 overload
 *      bit 가 트립 안 할 때가 많아 (XL/XM 시리즈 default ~5 초+) 0 반환이
 *      흔함. 그래도 추적기 resync 만으로 다음 LOAD 가 정상 동작한다.
 *
 *   반환:
 *      0 : HW 에러 없음, 추적기만 재동기화. 호출자는 일반 TIMEOUT 응답.
 *      1 : HW 에러 감지 + reboot 복구 성공. 호출자는 OVERLOAD 응답.
 *     -1 : HW 에러 감지 + reboot 복구 실패. 호출자는 INIT_FAIL 응답.
 */
static int8_t loaderShutdownCheckAndRecover(int32_t *hw_err_out) {
  // (1) 항상: tracker 를 모터가 실제로 도달한 지점으로 되돌림.
  g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
  g_dxl_target[ID_LOAD] = g_loader_goal_raw;

  // (2) HW err 확인 + 필요 시 reboot.
  int32_t hw_err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, ID_LOAD);
  if (hw_err_out)
    *hw_err_out = hw_err;
  if (hw_err == 0)
    return 0;
  return recoverLoader() ? 1 : -1;
}

/**
 * loaderRotateBy — 로더를 "현재 물리 위치 + delta_step" 까지 회전.
 *
 *   v2.0 워크플로:
 *     1. torqueOn — 평상시엔 torque off 라 모터가 자유 회전 가능. 여기서 깨움.
 *     2. P-gain MOVE 보장 (reboot 후엔 EEPROM default 로 되어 있을 수 있음).
 *     3. 현재 present_position 을 새로 읽어 baseline 으로 사용. 사용자가 손으로
 *        돌려놓은 위치가 반영됨. 누적 추적기 (g_loader_goal_raw) 는 사용 안 함.
 *     4. setGoalVelocity(LOADER_VEL) → drainSerial 폴링 → present 가 baseline +
 *        delta_step 도달하면 완료.
 *     5. setGoalVelocity(0) → torqueOff. 사용자 손에 다시 자유 회전 가능
 * 상태로.
 *
 *   반환: WaitResult — 호출자가 응답 송신에 사용.
 *     WAIT_ARRIVED : 정상 도달, motor torque off 상태.
 *     WAIT_TIMEOUT : 시간 내 도달 실패 (motor 도 torque off 로 되돌림).
 *     WAIT_ABORTED : STOP 으로 중단됨 (g_estop=true), motor torque off.
 */
static WaitResult loaderRotateBy(int32_t delta_step) {
  // 1. torque ON — 평상시 free state 에서 모터 깨우기.
  if (!dxl.torqueOn(ID_LOAD))
    return WAIT_TIMEOUT;
  // 2. PI-gain (reboot 등으로 EEPROM default 로 되돌아갔을 수 있으니 매번 설정).
  //    낮은 P = 회전 중 진동 억제, 높은 I = stall 시 wind-up 으로 정지마찰 돌파.
  dxl.writeControlTableItem(VELOCITY_P_GAIN, ID_LOAD, LOADER_VEL_P_GAIN);
  dxl.writeControlTableItem(VELOCITY_I_GAIN, ID_LOAD, LOADER_VEL_I_GAIN);

  // 3. 현재 위치 = baseline. 사용자가 손으로 돌려놓은 임의의 각도가 출발점.
  int32_t start = (int32_t)dxl.getPresentPosition(ID_LOAD);
  int32_t next = start + delta_step;
  g_loader_goal_raw = next;     // STATUS 보고용
  g_dxl_target[ID_LOAD] = next; // STATUS 보고용

  if (!dxl.setGoalVelocity(ID_LOAD, LOADER_VEL)) {
    dxl.torqueOff(ID_LOAD);
    return WAIT_TIMEOUT;
  }

  g_motion_busy = true;
  g_loader_moving = true;
  uint32_t t0 = millis();
  WaitResult result = WAIT_TIMEOUT;
  bool braking = false;

  while (millis() - t0 < MOTION_TIMEOUT) {
    drainSerial();
    if (g_estop) {
      result = WAIT_ABORTED;
      break;
    }
    int32_t present = (int32_t)dxl.getPresentPosition(ID_LOAD);
    // 감속 구간 진입: 마지막 LOADER_BRAKE_ZONE step 은 SLOW 속도로.
    // 폴링 지연(5ms × 6 step/ms = 30 step ≈ 2.6°) 영향 최소화.
    if (!braking && present >= next - LOADER_BRAKE_ZONE) {
      dxl.setGoalVelocity(ID_LOAD, LOADER_VEL_SLOW);
      braking = true;
    }
    if (present >= next - DXL_ARRIVED_TOL) {
      result = WAIT_ARRIVED;
      break;
    }
    delay(DXL_POLL_MS);
  }

  // 5. 정지: vel=0 명령 후 PI loop 가 active brake 하도록 잠깐 유지 →
  //    그 다음에 torqueOff. 즉시 torqueOff 하면 관성 코스팅으로 오버슈트.
  dxl.setGoalVelocity(ID_LOAD, 0);
  drainableWait(LOADER_BRAKE_MS);
  dxl.torqueOff(ID_LOAD);
  g_loader_moving = false;
  g_motion_busy = false;
  return result;
}

void handleLoad() {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  g_estop = false; // 명시적 모션 개시

  // 사전 점검: 이전 LOAD 의 overload 잔재 (HW err / torque off) 가 있으면
  // 여기서 reboot/re-enable 까지 자동 처리. 안 그러면 LOAD 명령이 "씹히는"
  // 듯 보임 (torque 꺼진 채로 velocity 만 써봐야 모터 안 돔).
  if (!ensureLoaderReady()) {
    sendErr("LOADER_NOT_READY");
    return;
  }

  WaitResult r = loaderRotateBy(LOADER_CYCLE_STEP);

  if (r == WAIT_TIMEOUT) {
    int32_t hw_err = 0;
    int8_t rc = loaderShutdownCheckAndRecover(&hw_err);
    if (rc == 1) {
      Serial.print("ERR OVERLOAD 0x");
      Serial.println((unsigned)hw_err, HEX);
      g_err_latched = true;
      return;
    }
    if (rc == -1) {
      sendErr("OVERLOAD_REINIT_FAIL");
      return;
    }
    // rc == 0: HW 에러 없음 → 일반 TIMEOUT 응답
  }
  replyByResult(r);
}

/**
 * drainable wait — delay() 대체. ms 동안 시리얼을 계속 처리하면서 대기.
 *   대기 중 g_estop 가 true 가 되면 즉시 false 반환 (STOP 으로 중단).
 *   true 반환 = 정상 만료.
 */
static bool drainableWait(uint32_t ms) {
  uint32_t t0 = millis();
  while (millis() - t0 < ms) {
    drainSerial();
    if (g_estop)
      return false;
    delay(1);
  }
  return true;
}

void handleStrike(char *args) {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }
  long rpm, hold_ms;
  if (sscanf(args, "%ld %ld", &rpm, &hold_ms) != 2) {
    sendErr("PARSE");
    return;
  }
  if (rpm < 0 || rpm > TMOTOR_MAX_RPM || hold_ms < 0) {
    sendErr("RANGE");
    return;
  }
  g_estop = false; // 명시적 모션 개시

  // ── 1) Spin-up ──────────────────────────────────────────
  setTmotor((uint16_t)rpm, (uint16_t)rpm);
  // delay() 가 아니라 drainable wait — 그 동안 들어온 STOP 즉시 반영.
  if (!drainableWait((uint32_t)hold_ms)) {
    // STOP 이 이미 setTmotor(0,0) 수행 + sendOk 송신.
    // 여기서 STRIKE 자체의 응답으로 ERR ABORTED 반환.
    sendErr("ABORTED");
    return;
  }

  // ── 2) LOAD ─────────────────────────────────────────────
  // 사전 점검 (HW err / torque off 잔재 흡수)
  if (!ensureLoaderReady()) {
    setTmotor(0, 0);
    sendErr("LOADER_NOT_READY");
    return;
  }
  // Velocity Mode + 위치 폴링 — Position Mode 의 PWM 포화 해제 문제 회피.
  WaitResult r = loaderRotateBy(LOADER_CYCLE_STEP);

  if (r != WAIT_ARRIVED) {
    setTmotor(0, 0); // 안전: spin-down 보장
    if (r == WAIT_TIMEOUT) {
      int32_t hw_err = 0;
      int8_t rc = loaderShutdownCheckAndRecover(&hw_err);
      if (rc == 1) {
        Serial.print("ERR OVERLOAD 0x");
        Serial.println((unsigned)hw_err, HEX);
        g_err_latched = true;
        return;
      }
      if (rc == -1) {
        sendErr("OVERLOAD_REINIT_FAIL");
        return;
      }
    }
    replyByResult(r); // 일반 TIMEOUT / ABORTED 응답
    return;
  }

  // ── 3) Spin-down ────────────────────────────────────────
  setTmotor(0, 0);
  sendOk();
}

/**
 * handleRecover — 부팅 시 init 실패한 모터를 다시 init 시도.
 *   g_motor_init_failed[id] = true 인 ID 만 대상 (성공한 모터는 손대지 않음).
 *   ID 가 Position/Velocity 어디에 속하는지 자동 판별.
 *
 *   응답
 *     OK                  — 실패 모터가 없거나 전부 복구됨
 *     ERR INIT a,b,c      — 시도했으나 여전히 실패한 ID 목록 (쉼표 구분)
 *
 *   주의: 모션 진행 중 (g_motion_busy) 에는 BUSY 응답. 모션 중 torqueOff/On 이
 *   하면 다른 모터의 SyncWrite/Read 타이밍에 영향 가능.
 */
void handleRecover() {
  if (g_motion_busy) {
    sendErr("BUSY");
    return;
  }

  // Position Mode: ID 1~4. 로더(ID 5) 는 아래 vel_ids 에 포함.
  const uint8_t pos_ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3, ID_TILT};
  const uint8_t vel_ids[] = {ID_LOAD, ID_WHEEL_L, ID_WHEEL_R};
  uint8_t still_failed[8];
  uint8_t fail_count = 0;

  for (uint8_t i = 0; i < 4; i++) {
    uint8_t id = pos_ids[i];
    if (!g_motor_init_failed[id])
      continue;
    if (initPosMotor(id)) {
      g_motor_init_failed[id] = false;
    } else {
      still_failed[fail_count++] = id;
    }
  }
  for (uint8_t i = 0; i < 3; i++) {
    uint8_t id = vel_ids[i];
    if (!g_motor_init_failed[id])
      continue;
    bool ok = (id == ID_LOAD) ? initLoaderVelMotor() : initVelMotor(id);
    if (ok) {
      g_motor_init_failed[id] = false;
    } else {
      still_failed[fail_count++] = id;
    }
  }

  if (fail_count == 0) {
    g_err_latched = false;
    sendOk();
    return;
  }

  Serial.print("ERR INIT ");
  for (uint8_t i = 0; i < fail_count; i++) {
    Serial.print(still_failed[i]);
    if (i + 1 < fail_count)
      Serial.print(',');
  }
  Serial.println();
  g_err_latched = true;
}

// ═══════════════════════════════════════════════════════════════
//  라인 디스패처
// ═══════════════════════════════════════════════════════════════
void dispatch(char *line) {
  while (*line == ' ')
    line++;
  if (*line == '\0')
    return;

  char cmd[16] = {0};
  char *p = line;
  uint8_t ci = 0;
  while (*p && *p != ' ' && ci < 15)
    cmd[ci++] = *p++;
  cmd[ci] = '\0';
  for (uint8_t i = 0; cmd[i]; i++)
    if (cmd[i] >= 'a' && cmd[i] <= 'z')
      cmd[i] -= 32;
  while (*p == ' ')
    p++;

  if (!strcmp(cmd, "PING"))
    handlePing();
  else if (!strcmp(cmd, "STATUS"))
    handleStatus();
  else if (!strcmp(cmd, "STOP"))
    handleStop();
  else if (!strcmp(cmd, "DRIVE"))
    handleDrive(p);
  else if (!strcmp(cmd, "AIM"))
    handleAim(p);
  else if (!strcmp(cmd, "AIMF"))
    handleAimF(p);
  else if (!strcmp(cmd, "HOME"))
    handleHome();
  else if (!strcmp(cmd, "TILT"))
    handleTilt(p);
  else if (!strcmp(cmd, "TILT_ASYNC"))
    handleTiltAsync(p);
  else if (!strcmp(cmd, "SPIN"))
    handleSpin(p);
  else if (!strcmp(cmd, "LOAD"))
    handleLoad();
  else if (!strcmp(cmd, "STRIKE"))
    handleStrike(p);
  else if (!strcmp(cmd, "RECOVER"))
    handleRecover();
  else
    sendErr("PARSE");
}

// ── 모터 init 헬퍼 ─────────────────────────────────────────────
// setup() 과 handleRecover() 가 공유. 재시도 + per-motor 진단 로그.

/**
 * pingRetry — 짧은 간격으로 N 회 ping 재시도.
 *   콜드부팅 직후 모터 펌웨어가 아직 안정화 안 됐을 때의 race 를 완화.
 *   1Mbps 에서 모터 1대 ping 은 ~300 µs, retries × 20 ms 만큼만 추가 latency.
 */
static bool pingRetry(uint8_t id, uint8_t retries) {
  for (uint8_t k = 0; k < retries; k++) {
    if (dxl.ping(id))
      return true;
    delay(20);
  }
  return false;
}

/**
 * initPosMotor — Position 계열 모터 1대 초기화.
 *   ping → torqueOff → DriveMode(0)/OpMode/Profile → torqueOn 의 시퀀스.
 *   torqueOn 의 응답까지 확인해야 hardware-error 로 shutdown 된 모터를 잡아낼
 *   수 있다 (overload/overheat/electrical shock 시 모터가 자동 torque off
 *   하고 재-enable 을 거부함).
 *
 *   op_mode = OP_POSITION         : 단일 회전 (0..4095 wrap) — 레벨링/틸트
 *           = OP_EXTENDED_POSITION : 다회전 누적 — 로더 전용
 *
 *   profile_vel / profile_acc : 각 0 이면 Step Mode (프로파일 비활성, goal 즉시
 *     적용 → 출발 직후부터 max PWM). 일반 모터는 DXL_PROFILE_VEL/ACC 사용해
 *     사다리꼴 트래젝토리로 부드럽게 추종. 로더처럼 정지마찰 큰 부하는 0/0.
 *
 *   position_p_gain : Position PID 의 P term (RAM, addr 84). default 800 보다
 *     크게 잡아 goal 근처에서의 잔여 오차를 ±DXL_ARRIVED_TOL 안으로 수렴시킴.
 *     너무 크면 험/오버슈트. 기본 DXL_POSITION_P_GAIN (1500) 사용.
 *
 *   position_d_gain : Position PID 의 D term (RAM, addr 80). default 0 →
 *     기본 DXL_POSITION_D_GAIN (400). P 가 만든 가속의 오버슈트·ringing 을
 *     억제 → 무게 부하/AIMF 스트리밍에서의 "스텝 + 진동" 체감 감소.
 *
 *   ff_acc_gain : Feedforward 2nd-order (acceleration) gain (RAM, addr 88).
 *     default 0 → 기본 DXL_FEEDFORWARD_ACC_GAIN (200). Trajectory generator
 *     의 desired acceleration 을 PWM 에 미리 더해 micro-trajectory 간 추종
 *     lag/오버슈트를 보상. AIMF 스트리밍 + 부하 (특히 helping force 인 중력
 *     하강) 에서 효과적.
 */
static bool initPosMotor(uint8_t id, uint8_t op_mode, uint32_t profile_vel,
                         uint32_t profile_acc, uint16_t position_p_gain,
                         uint16_t position_d_gain, uint16_t ff_acc_gain) {
  if (!pingRetry(id))
    return false;
  dxl.torqueOff(id);
  dxl.writeControlTableItem(DRIVE_MODE, id, 0);
  dxl.setOperatingMode(id, op_mode);
  dxl.writeControlTableItem(PROFILE_VELOCITY, id, profile_vel);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, id, profile_acc);
  dxl.writeControlTableItem(POSITION_D_GAIN, id, position_d_gain);
  dxl.writeControlTableItem(POSITION_P_GAIN, id, position_p_gain);
  dxl.writeControlTableItem(FEEDFORWARD_2ND_GAIN, id, ff_acc_gain);
  if (!dxl.torqueOn(id))
    return false;
  g_dxl_target[id] = (int32_t)dxl.getPresentPosition(id);
  return true;
}

/**
 * initVelMotor — Velocity Mode 모터(휠) 1대 초기화.
 *   ping → torqueOff → OpMode(VEL) → torqueOn → goal 0.
 */
static bool initVelMotor(uint8_t id) {
  if (!pingRetry(id))
    return false;
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_VELOCITY);
  if (!dxl.torqueOn(id))
    return false;
  dxl.setGoalVelocity(id, 0);
  return true;
}

/**
 * initLoaderVelMotor — 로더(ID 5) 전용 Velocity Mode init.
 *   v2.0: torque OFF 상태로 끝남. LOAD 명령 처리 시 loaderRotateBy 가 잠깐
 *   torqueOn → 회전 → torqueOff 로 사이클을 돌린다. 평상시 사용자가 손으로
 *   로더를 자유 회전시켜 공을 적재할 수 있음.
 *
 *   시퀀스: ping → torqueOff → OpMode(VEL) → P-gain RAM 기록 → goal_vel=0
 *   → torqueOff 유지. ping 외엔 torque 가 켜진 적이 없어 motor 가 한 번도
 *   움직이지 않음 (콜드 부팅 안전).
 */
static bool initLoaderVelMotor() {
  if (!pingRetry(ID_LOAD))
    return false;
  dxl.torqueOff(ID_LOAD);
  dxl.setOperatingMode(ID_LOAD, OP_VELOCITY);
  dxl.writeControlTableItem(VELOCITY_P_GAIN, ID_LOAD, LOADER_VEL_P_GAIN);
  dxl.writeControlTableItem(VELOCITY_I_GAIN, ID_LOAD, LOADER_VEL_I_GAIN);
  // goal_vel 을 미리 0 으로 — 다음 torqueOn 시점에 모터가 갑자기 튀지 않도록.
  dxl.setGoalVelocity(ID_LOAD, 0);
  // torque ON 안 함. 다음 LOAD 가 명시적으로 켤 때까지 free wheel 상태.
  return true;
}

/**
 * ensureLoaderReady — LOAD 진입 직전의 사전 점검.
 *   HARDWARE_ERROR_STATUS != 0 → recoverLoader (reboot + 재init). 이전 LOAD 가
 *   overload 로 motor 자가 차단된 상태 그대로 두면 다음 torqueOn 자체가 실패.
 *
 *   v2.0: torque 상태는 체크 안 함 — 평상시 torque off 가 정상 상태이며,
 *   loaderRotateBy 가 진입 시 명시적으로 torqueOn 한다.
 *
 *   true 반환 = 정상 동작 가능, false = 복구 실패 (호출자가 ERR 응답).
 */
static bool ensureLoaderReady() {
  int32_t hw_err = dxl.readControlTableItem(HARDWARE_ERROR_STATUS, ID_LOAD);
  if (hw_err != 0) {
    return recoverLoader();
  }
  return true;
}

/**
 * recoverLoader — Hardware shutdown 상태의 로더(ID 5) 자동 복구.
 *   overheat/overload 등으로 모터가 Hardware Error Status 를 set 하고 torque
 *   를 자가 차단했을 때 호출. reboot 으로 모든 휘발성 상태 + error flag
 *   클리어 → Velocity Mode 로 init 재실행 → g_loader_goal_raw 를 현재 위치로
 *   재캡쳐 (모터가 목표 못 채우고 멈춘 지점부터 다음 LOAD 가 누적되도록).
 *
 *   delay(500) + initVelMotor 3 회 재시도 : XL/XM datasheet 는 200 ms+ 권장
 *   이지만 콜드 reboot 시 500 ms 이후에야 안정적으로 응답하는 케이스 관측됨.
 *   첫 init 실패해도 200 ms 간격으로 최대 3 회 재시도해서 race 흡수.
 */
static bool recoverLoader() {
  dxl.reboot(ID_LOAD);
  delay(500);
  for (uint8_t attempt = 0; attempt < 3; attempt++) {
    if (initLoaderVelMotor()) {
      g_motor_init_failed[ID_LOAD] = false;
      g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
      g_dxl_target[ID_LOAD] = g_loader_goal_raw;
      return true;
    }
    delay(200);
  }
  g_motor_init_failed[ID_LOAD] = true;
  return false;
}

/**
 * ensureDxlBaud — 모든 ID 를 target baud 에 맞춤.
 *   v1.7: 첫 ping 성공 시 즉시 return 하던 v1.6 이전 버그 수정.
 *   1) target baud 로 모든 ID ping → 응답한 모터 마킹.
 *   2) target 에 응답 못 한 모터만 factory baud 로 재검색.
 *   3) factory 에서 발견된 모터의 EEPROM 에 target baud 기입.
 *   4) target baud 로 재연결 → 최종 검증 로그.
 *
 *   반환: 하나라도 target baud 에서 응답하면 true.
 *         (false 면 버스 자체가 죽었다는 뜻 → ERR INIT_BAUD)
 */
static bool ensureDxlBaud() {
  const uint8_t all_ids[] = {ID_LVL_1, ID_LVL_2,   ID_LVL_3,  ID_TILT,
                             ID_LOAD,  ID_WHEEL_L, ID_WHEEL_R};
  const uint8_t n = sizeof(all_ids) / sizeof(all_ids[0]);
  bool ok_at_target[7] = {false};

  // (1) target baud 전수 검사
  dxl.begin(DXL_BAUDRATE);
  delay(20);
  uint8_t ok_count = 0;
  for (uint8_t i = 0; i < n; i++) {
    if (pingRetry(all_ids[i], 2)) {
      ok_at_target[i] = true;
      ok_count++;
    }
  }
  if (ok_count == n)
    return true; // 정상 경로 — 전 모터가 이미 target baud

  // (2) 누락 모터만 factory baud 로 검색 + 업그레이드
  dxl.begin(DXL_BAUDRATE_FACTORY);
  delay(20);
  for (uint8_t i = 0; i < n; i++) {
    if (ok_at_target[i])
      continue;
    if (!pingRetry(all_ids[i], 2))
      continue;
    dxl.torqueOff(all_ids[i]);
    dxl.setBaudrate(all_ids[i], DXL_BAUDRATE); // EEPROM write
    delay(30);
    Serial.print("INFO BAUD_UPGRADED ");
    Serial.println(all_ids[i]);
  }

  // (3) target baud 재연결 + 최종 검증
  delay(100);
  dxl.begin(DXL_BAUDRATE);
  delay(20);
  bool any_alive = false;
  for (uint8_t i = 0; i < n; i++) {
    if (pingRetry(all_ids[i], 2)) {
      any_alive = true;
    } else {
      Serial.print("ERR INIT_PING_ID");
      Serial.println(all_ids[i]);
    }
  }
  return any_alive;
}

// ═══════════════════════════════════════════════════════════════
//  setup
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  while (!Serial)
    ; // 배터리 단독 운용 시 제거

  // T-motor ESC arming
  escTop.attach(PIN_ESC_TOP);
  escBot.attach(PIN_ESC_BOT);
  escTop.writeMicroseconds(ESC_MIN_US);
  escBot.writeMicroseconds(ESC_MIN_US);
  delay(2000);

  // DXL 버스 — 운영 baud 보장 (모터가 출고 baud 면 자동 업그레이드)
  dxl.setPortProtocolVersion(DXL_PROTOCOL);
  if (!ensureDxlBaud()) {
    g_err_latched = true;
    Serial.println("ERR INIT_BAUD");
  }

  // Position Mode: ID 1~4 (레벨링 + 틸트)
  // Drive Mode = 0 (Velocity-based profile) 을 initPosMotor 가 강제함.
  // Time-based profile 이면 AIMF 50 ms 스트리밍 시 타이머가 매번 리셋되어
  // 모터가 실질 정지하기 때문 (v1.5 기록 참조).
  // 로더(ID 5) 는 Velocity Mode — Position Mode 의 PWM∝error 한계로 정지마찰
  // 못 이김 (v1.9 기록).
  const uint8_t pos_ids[] = {ID_LVL_1, ID_LVL_2, ID_LVL_3, ID_TILT};
  for (uint8_t i = 0; i < 4; i++) {
    uint8_t id = pos_ids[i];
    if (initPosMotor(id)) {
      g_motor_init_failed[id] = false;
    } else {
      g_motor_init_failed[id] = true;
      g_err_latched = true;
      Serial.print("ERR INIT_ID");
      Serial.println(id);
    }
  }

  // Velocity Mode: ID 5 (로더) — 2단 VELOCITY_P_GAIN (MOVE/HOLD) 변종
  if (initLoaderVelMotor()) {
    g_motor_init_failed[ID_LOAD] = false;
  } else {
    g_motor_init_failed[ID_LOAD] = true;
    g_err_latched = true;
    Serial.print("ERR INIT_ID");
    Serial.println(ID_LOAD);
  }

  // 로더(ID 5) 기준점 캡쳐 — 부팅 시점의 현재 위치를 0° 로 정의한다.
  // setGoalPosition 호출 없음: 모터는 어디로도 이동하지 않고 그 자리에 머문다.
  // 이후 매 LOAD 가 이 기준점에서 +90° CCW 누적된다.
  if (!g_motor_init_failed[ID_LOAD]) {
    g_loader_goal_raw = (int32_t)dxl.getPresentPosition(ID_LOAD);
    g_dxl_target[ID_LOAD] = g_loader_goal_raw;
  }

  // Velocity Mode: ID 6~7 (XC430 휠)
  const uint8_t whl_ids[] = {ID_WHEEL_L, ID_WHEEL_R};
  for (uint8_t i = 0; i < 2; i++) {
    uint8_t id = whl_ids[i];
    if (initVelMotor(id)) {
      g_motor_init_failed[id] = false;
    } else {
      g_motor_init_failed[id] = true;
      g_err_latched = true;
      Serial.print("ERR INIT_ID");
      Serial.println(id);
    }
  }

  // ── Sync Write 구조체 초기화 (1회) ───────────────────────
  // 이후 wheelSetVelocity/wheelStop/syncMoveLevel 에서
  // data 값만 갱신하고 dxl.syncWrite() 호출
  g_sw_vel_xel[0].id = ID_WHEEL_L;
  g_sw_vel_xel[0].p_data = (uint8_t *)&g_sw_vel_data[0];
  g_sw_vel_xel[1].id = ID_WHEEL_R;
  g_sw_vel_xel[1].p_data = (uint8_t *)&g_sw_vel_data[1];
  g_sw_vel.addr = 104; // Goal Velocity
  g_sw_vel.addr_length = 4;
  g_sw_vel.p_xels = g_sw_vel_xel;
  g_sw_vel.xel_count = 2;

  g_sw_pos_xel[0].id = ID_LVL_1;
  g_sw_pos_xel[0].p_data = (uint8_t *)&g_sw_pos_data[0];
  g_sw_pos_xel[1].id = ID_LVL_2;
  g_sw_pos_xel[1].p_data = (uint8_t *)&g_sw_pos_data[1];
  g_sw_pos_xel[2].id = ID_LVL_3;
  g_sw_pos_xel[2].p_data = (uint8_t *)&g_sw_pos_data[2];
  g_sw_pos.addr = 116; // Goal Position
  g_sw_pos.addr_length = 4;
  g_sw_pos.p_xels = g_sw_pos_xel;
  g_sw_pos.xel_count = 3;

  // Sync Read: 레벨링 3축 PRESENT_POSITION (waitMotion fast path)
  g_sr_pos_xel[0].id = ID_LVL_1;
  g_sr_pos_xel[0].p_recv_buf = (uint8_t *)&g_sr_pos_data[0];
  g_sr_pos_xel[1].id = ID_LVL_2;
  g_sr_pos_xel[1].p_recv_buf = (uint8_t *)&g_sr_pos_data[1];
  g_sr_pos_xel[2].id = ID_LVL_3;
  g_sr_pos_xel[2].p_recv_buf = (uint8_t *)&g_sr_pos_data[2];
  g_sr_pos.addr = 132; // Present Position
  g_sr_pos.addr_length = 4;
  g_sr_pos.p_xels = g_sr_pos_xel;
  g_sr_pos.xel_count = 3;

  g_last_drive_ms = millis();
}

// ═══════════════════════════════════════════════════════════════
//  loop
// ═══════════════════════════════════════════════════════════════
void loop() {
  // 1. 시리얼 수신 & 라인 조립 (waitMotion 내부와 공유).
  //    NOTE: drainSerial() 안에서 handleDrive / handleTiltAsync 가
  //    g_last_*_ms = millis() 로 갱신할 수 있으므로, `now` 는 반드시
  //    drainSerial() *뒤* 에 캡쳐해야 한다. 그렇지 않으면 uint32_t
  //    언더플로우 (now < g_last_*_ms → 4B 근처) 로 매 iteration WATCHDOG_MS
  //    임계치를 넘어버려, 도착한 DRIVE 가 같은 iteration 에서 wheelStop()
  //    으로 덮어써진다.
  drainSerial();

  uint32_t now = millis();

  // 2. DRIVE 200 ms watchdog
  //    XC430 내부 루프는 계속 돌므로 명시적으로 velocity=0 write 필요.
  if (!g_estop && g_drive_active) {
    if ((now - g_last_drive_ms) > WATCHDOG_MS && !g_watchdog_tripped) {
      wheelStop();
      g_watchdog_tripped = true;
    }
  }

  // 3. TILT_ASYNC 200 ms watchdog → hold current position.
  //    Unlike DRIVE (which forces velocity=0), tilt must not snap to a
  //    default angle — 0° would let the camera fall. Snapshot present
  //    position and rewrite it as the goal.
  if (g_last_tilt_async_ms != 0 &&
      (now - g_last_tilt_async_ms) > WATCHDOG_MS) {
    int32_t present = (int32_t)dxl.getPresentPosition(ID_TILT);
    dxl.setGoalPosition(ID_TILT, (uint32_t)present);
    g_last_tilt_async_ms = 0;   // re-armed on next TILT_ASYNC
  }
}
