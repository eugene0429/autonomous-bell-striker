/*
 * Leveling Platform Controller — OpenRB-150 reference sketch.
 *
 * Pi5 (LevelingMotorClient) 와의 시리얼 프로토콜은
 * [leveling_motor.py](leveling_motor.py) 헤더 docstring 참조.
 *
 * 본 스케치는 *프로토콜 파싱·디스패치 골격* 만 제공한다.
 * 실제 모터 제어 (Dynamixel goalPosition 쓰기, 모션 완료 검출 등) 는
 * MARK [TODO-FW] 가 있는 곳에 채워 넣을 것.
 *
 * Build env
 *   - Arduino IDE 2.x  +  OpenRB-150 board package
 *   - Library: DYNAMIXEL2Arduino (모터 제어용, 팀원이 채울 부분)
 *   - Serial : USB CDC, 115200 8N1
 *
 * Wiring assumption
 *   - OpenRB-150 의 DYNAMIXEL 포트에 모터 3개 (ID 1, 2, 3) 데이지체인.
 *   - 모터 1개당 4096 step / rev (XL330 / XM430 등).
 *
 * Protocol summary  (모든 라인 '\n' 종료, 7-bit ASCII)
 *
 *   Pi → OpenRB                    OpenRB → Pi
 *   ───────────                    ──────────
 *   PING                           PONG
 *   AIM <s1> <s2> <s3>             OK   |  ERR <reason>
 *   HOME                           OK   |  ERR <reason>
 *   STATUS                         S <s1> <s2> <s3> <flags>
 *   STOP                           OK
 *
 *   flags bits: 0=motor1 moving, 1=2, 2=3, 3=error, 4=homed
 */

#include <Arduino.h>

// MARK [TODO-FW] — actual motor library
// #include <Dynamixel2Arduino.h>
// Dynamixel2Arduino dxl(Serial1, /*DIR_PIN=*/-1);

// ───────────────────────────────────────────────
//   Config
// ───────────────────────────────────────────────
static const long   SERIAL_BAUD       = 115200;
static const int    NUM_MOTORS        = 3;
static const int    MOTOR_IDS[NUM_MOTORS] = {1, 2, 3};
static const int    STEP_MIN          = -2048;
static const int    STEP_MAX          = +2047;
static const unsigned long MOTION_TIMEOUT_MS = 4000;
static const int    READBUF_LEN       = 64;     // 한 줄 최대 길이

// ───────────────────────────────────────────────
//   State (IK 가 마지막으로 명령한 절대 step + 플래그)
// ───────────────────────────────────────────────
struct MotorState {
  int  target_step  = 0;
  bool moving       = false;
};
static MotorState  g_motor[NUM_MOTORS];
static bool        g_error_latched = false;
static bool        g_homed         = false;
static char        g_buf[READBUF_LEN];
static uint8_t     g_buf_len = 0;

// ───────────────────────────────────────────────
//   Reply helpers
// ───────────────────────────────────────────────
static inline void replyOK()                         { Serial.println("OK"); }
static inline void replyPONG()                       { Serial.println("PONG"); }
static inline void replyERR(const char *reason)      {
  Serial.print("ERR ");
  Serial.println(reason);
}
static inline uint8_t flagsByte() {
  uint8_t f = 0;
  for (int i = 0; i < NUM_MOTORS; ++i) if (g_motor[i].moving) f |= (1u << i);
  if (g_error_latched) f |= 0x08;
  if (g_homed)         f |= 0x10;
  return f;
}
static void replySTATUS() {
  Serial.print("S ");
  for (int i = 0; i < NUM_MOTORS; ++i) {
    Serial.print(currentStep(i));   // MARK [TODO-FW]: 실제 위치 읽기
    Serial.print(' ');
  }
  Serial.println(flagsByte());
}

// ───────────────────────────────────────────────
//   MARK [TODO-FW] — 모터 제어 (팀원이 채워 넣을 부분)
// ───────────────────────────────────────────────
static int currentStep(int motor_idx) {
  // dxl.getPresentPosition(MOTOR_IDS[motor_idx]) 등으로 읽고
  // signed step domain (-2048..+2047) 으로 변환해서 반환.
  // 임시 stub: 마지막 명령 위치를 그대로 반환.
  return g_motor[motor_idx].target_step;
}

static bool driveMotor(int motor_idx, int target_step) {
  // dxl.setGoalPosition(MOTOR_IDS[motor_idx], target_step + offset, UNIT_RAW)
  // 명령만 내고 즉시 return true (모션 완료 대기는 waitMotionComplete 가 담당).
  g_motor[motor_idx].target_step = target_step;
  g_motor[motor_idx].moving      = true;
  return true;
}

static bool stopAllMotors() {
  // dxl.torqueOff(MOTOR_IDS[i]) 등으로 즉시 정지.
  for (int i = 0; i < NUM_MOTORS; ++i) g_motor[i].moving = false;
  return true;
}

static bool waitMotionComplete(unsigned long timeout_ms) {
  // 모든 모터가 goal 에 도달할 때까지 polling (또는 dxl.getMoving() 사용).
  unsigned long t0 = millis();
  while (millis() - t0 < timeout_ms) {
    bool any = false;
    for (int i = 0; i < NUM_MOTORS; ++i) {
      // bool moving_i = dxl.readControlTableItem(MOVING, MOTOR_IDS[i]);
      bool moving_i = false;            // stub: 즉시 완료 처리
      g_motor[i].moving = moving_i;
      any |= moving_i;
    }
    if (!any) return true;
    delay(2);
  }
  // timeout
  stopAllMotors();
  g_error_latched = true;
  return false;
}

// ───────────────────────────────────────────────
//   Command handlers
// ───────────────────────────────────────────────
static void handleAIM(const char *args) {
  int s[NUM_MOTORS];
  if (sscanf(args, "%d %d %d", &s[0], &s[1], &s[2]) != NUM_MOTORS) {
    replyERR("PARSE");
    return;
  }
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (s[i] < STEP_MIN || s[i] > STEP_MAX) { replyERR("OUT_OF_RANGE"); return; }
  }
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!driveMotor(i, s[i])) { replyERR("HW"); return; }
  }
  if (!waitMotionComplete(MOTION_TIMEOUT_MS)) {
    replyERR("TIMEOUT");
    return;
  }
  replyOK();
}

static void handleHOME() {
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!driveMotor(i, 0)) { replyERR("HW"); return; }
  }
  if (!waitMotionComplete(MOTION_TIMEOUT_MS)) {
    replyERR("TIMEOUT");
    return;
  }
  g_homed = true;
  replyOK();
}

static void handleSTOP() {
  stopAllMotors();
  replyOK();
}

static void dispatch(char *line) {
  // line 은 '\n' 제거된 상태, 0-terminated.
  if      (strncmp(line, "PING",  4) == 0)  replyPONG();
  else if (strncmp(line, "AIM ",  4) == 0)  handleAIM(line + 4);
  else if (strncmp(line, "HOME",  4) == 0)  handleHOME();
  else if (strncmp(line, "STATUS", 6) == 0) replySTATUS();
  else if (strncmp(line, "STOP",  4) == 0)  handleSTOP();
  else replyERR("PARSE");
}

// ───────────────────────────────────────────────
//   Setup / loop
// ───────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {/* USB-CDC ready 대기 */ }

  // MARK [TODO-FW] — 모터 초기화 (Dynamixel TTL 통신, torque on, etc.)
  // dxl.begin(1000000);
  // dxl.setPortProtocolVersion(2.0);
  // for (int i = 0; i < NUM_MOTORS; ++i) {
  //   dxl.torqueOff(MOTOR_IDS[i]);
  //   dxl.setOperatingMode(MOTOR_IDS[i], OP_POSITION);
  //   dxl.torqueOn(MOTOR_IDS[i]);
  // }
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;                      // CR 무시
    if (c == '\n') {
      g_buf[g_buf_len] = '\0';
      if (g_buf_len > 0) dispatch(g_buf);
      g_buf_len = 0;
    } else if (g_buf_len < READBUF_LEN - 1) {
      g_buf[g_buf_len++] = c;
    } else {
      // overflow → 라인 폐기 + ERR
      g_buf_len = 0;
      replyERR("OVERFLOW");
    }
  }
}
