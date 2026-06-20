/*
 * Leveling Platform Controller — OpenRB-150
 *
 * Pi5 (LevelingMotorClient) 와의 시리얼 프로토콜:
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
 *
 * Build env
 *   - Arduino IDE 2.x  +  OpenRB-150 board package
 *   - Library: DYNAMIXEL2Arduino (>= 0.6.0, InfoSyncWriteInst_t 지원)
 *   - Serial : USB CDC, 115200 8N1
 *
 * Wiring
 *   - OpenRB-150 의 DYNAMIXEL 포트에 모터 3개 (ID 1, 2, 3) 데이지체인.
 *   - 모터 1개당 4096 step / rev (XL330 / XM430 등).
 *   - step 0 은 각 모터의 물리적 홈 위치 (센터, raw = CENTER_RAW).
 *
 * Latency notes
 *   - DXL 버스를 1 Mbps 로 운영. 모터가 출고 baud(57600) 면 첫 부팅 시
 *     자동으로 EEPROM 에 1 Mbps 를 기입하고 재연결한다 (idempotent).
 *   - 3축 명령은 SyncWrite, 3축 상태 읽기는 SyncRead 로 묶어 transaction 1회.
 */

#include <Arduino.h>
#include <Dynamixel2Arduino.h>

// ───────────────────────────────────────────────
//   Config
// ───────────────────────────────────────────────
static const long     SERIAL_BAUD          = 115200;
static const long     DXL_BAUD_TARGET      = 1000000;  // 운영 baud (1 Mbps)
static const long     DXL_BAUD_FACTORY     = 57600;    // 모터 출고 baud — auto-upgrade 경로
static const float    DXL_PROTOCOL         = 2.0;

static const int      NUM_MOTORS           = 3;
static const uint8_t  MOTOR_IDS[NUM_MOTORS] = {1, 2, 3};

// signed step 범위 (-2048 ~ +2047, 총 4096 step = 1회전)
static const int      STEP_MIN             = -2048;
static const int      STEP_MAX             = +2047;

// raw 위치 기준점: step 0 이 매핑되는 raw 값
static const int32_t  CENTER_RAW           = 2048;

// 모션 완료 감지 (position tolerance only — MOVING 레지스터 폴링 제거)
static const unsigned long MOTION_TIMEOUT_MS  = 4000;
static const int32_t       ARRIVED_TOLERANCE  = 10;
static const uint16_t      MOTION_POLL_MS     = 5;

// 프로파일 (속도/가속도 제한 — 0 이면 프로파일 비활성)
static const uint32_t PROFILE_VEL          = 400;
static const uint32_t PROFILE_ACC          = 80;

static const int      READBUF_LEN          = 64;

// XL330 / XM430 control table (Protocol 2.0)
static const uint16_t ADDR_GOAL_POSITION    = 116;
static const uint16_t LEN_GOAL_POSITION     = 4;
static const uint16_t ADDR_PRESENT_POSITION = 132;
static const uint16_t LEN_PRESENT_POSITION  = 4;

// ───────────────────────────────────────────────
//   Dynamixel
// ───────────────────────────────────────────────
Dynamixel2Arduino dxl(Serial1, /*DIR_PIN=*/-1);
using namespace ControlTableItem;

// ───────────────────────────────────────────────
//   SyncWrite / SyncRead infrastructure
// ───────────────────────────────────────────────
typedef struct __attribute__((packed)) { int32_t goal;    } sw_pos_t;
typedef struct __attribute__((packed)) { int32_t present; } sr_pos_t;

static sw_pos_t g_sw_data[NUM_MOTORS];
static sr_pos_t g_sr_data[NUM_MOTORS];

static DYNAMIXEL::InfoSyncWriteInst_t g_sw_infos;
static DYNAMIXEL::XELInfoSyncWrite_t  g_sw_xels[NUM_MOTORS];

static DYNAMIXEL::InfoSyncReadInst_t  g_sr_infos;
static DYNAMIXEL::XELInfoSyncRead_t   g_sr_xels[NUM_MOTORS];

// ───────────────────────────────────────────────
//   State
// ───────────────────────────────────────────────
struct MotorState {
  int32_t target_raw  = CENTER_RAW;
  bool    moving      = false;
};
static MotorState  g_motor[NUM_MOTORS];
static bool        g_error_latched  = false;
static bool        g_homed          = false;
static char        g_buf[READBUF_LEN];
static uint8_t     g_buf_len        = 0;

// ───────────────────────────────────────────────
//   Unit conversion helpers
// ───────────────────────────────────────────────
static inline int32_t stepToRaw(int step) {
  return (int32_t)constrain(step, STEP_MIN, STEP_MAX) + CENTER_RAW;
}

static inline int stepFromRaw(int32_t raw) {
  return (int)(raw - CENTER_RAW);
}

// ───────────────────────────────────────────────
//   Reply helpers
// ───────────────────────────────────────────────
static inline void replyOK()                    { Serial.println("OK"); }
static inline void replyPONG()                  { Serial.println("PONG"); }
static inline void replyERR(const char *reason) {
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

// ───────────────────────────────────────────────
//   Bus operations (SyncWrite / SyncRead)
// ───────────────────────────────────────────────

/* 3축 PRESENT_POSITION 을 한 transaction 으로 읽어 g_sr_data 에 채움. */
static bool readAllPresent() {
  g_sr_infos.is_info_changed = true;
  uint8_t recv = dxl.syncRead(&g_sr_infos);
  return recv == NUM_MOTORS;
}

/* 3축 GOAL_POSITION 을 한 transaction 으로 송신. 비블로킹. */
static bool driveAll(const int target_steps[NUM_MOTORS]) {
  for (int i = 0; i < NUM_MOTORS; ++i) {
    g_sw_data[i].goal     = stepToRaw(target_steps[i]);
    g_motor[i].target_raw = g_sw_data[i].goal;
    g_motor[i].moving     = true;
  }
  g_sw_infos.is_info_changed = true;
  return dxl.syncWrite(&g_sw_infos);
}

/*
 * stopAllMotors — 현재 위치를 새 목표로 잡아 즉시 정지.
 *   토크는 유지 (홀딩) — torqueOff 토글 방식 대비 자중 낙하 없음.
 */
static bool stopAllMotors() {
  if (!readAllPresent()) return false;
  int hold[NUM_MOTORS];
  for (int i = 0; i < NUM_MOTORS; ++i) {
    hold[i] = stepFromRaw(g_sr_data[i].present);
  }
  bool ok = driveAll(hold);
  for (int i = 0; i < NUM_MOTORS; ++i) g_motor[i].moving = false;
  return ok;
}

// ───────────────────────────────────────────────
//   Motion completion polling
// ───────────────────────────────────────────────

/*
 * waitMotionComplete — position tolerance 만으로 도달 판정 (MOVING 레지스터 미사용).
 *   매 폴링 사이클에 syncRead 1 회로 3축 위치 동시 갱신.
 */
static bool waitMotionComplete(unsigned long timeout_ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < timeout_ms) {
    if (!readAllPresent()) {
      delay(MOTION_POLL_MS);
      continue;
    }
    bool any_moving = false;
    for (int i = 0; i < NUM_MOTORS; ++i) {
      if (!g_motor[i].moving) continue;
      bool arrived = (abs(g_sr_data[i].present - g_motor[i].target_raw)
                      <= ARRIVED_TOLERANCE);
      if (arrived) g_motor[i].moving = false;
      else         any_moving = true;
    }
    if (!any_moving) return true;
    delay(MOTION_POLL_MS);
  }
  stopAllMotors();
  g_error_latched = true;
  return false;
}

// ───────────────────────────────────────────────
//   STATUS reply
// ───────────────────────────────────────────────
static void replySTATUS() {
  readAllPresent();   // 실패해도 직전 캐시 값 사용
  Serial.print("S ");
  for (int i = 0; i < NUM_MOTORS; ++i) {
    Serial.print(stepFromRaw(g_sr_data[i].present));
    Serial.print(' ');
  }
  Serial.println(flagsByte());
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
    if (s[i] < STEP_MIN || s[i] > STEP_MAX) {
      replyERR("OUT_OF_RANGE");
      return;
    }
  }
  if (!driveAll(s)) {
    replyERR("HW");
    return;
  }
  if (!waitMotionComplete(MOTION_TIMEOUT_MS)) {
    replyERR("TIMEOUT");
    return;
  }
  replyOK();
}

static void handleHOME() {
  const int zeros[NUM_MOTORS] = {0, 0, 0};
  if (!driveAll(zeros)) {
    replyERR("HW");
    return;
  }
  if (!waitMotionComplete(MOTION_TIMEOUT_MS)) {
    replyERR("TIMEOUT");
    return;
  }
  g_homed         = true;
  g_error_latched = false;
  replyOK();
}

static void handleSTOP() {
  stopAllMotors();
  replyOK();
}

// ───────────────────────────────────────────────
//   Command dispatcher
// ───────────────────────────────────────────────
static void dispatch(char *line) {
  if      (strncmp(line, "PING",   4) == 0) replyPONG();
  else if (strncmp(line, "AIM ",   4) == 0) handleAIM(line + 4);
  else if (strncmp(line, "HOME",   4) == 0) handleHOME();
  else if (strncmp(line, "STATUS", 6) == 0) replySTATUS();
  else if (strncmp(line, "STOP",   4) == 0) handleSTOP();
  else replyERR("PARSE");
}

// ───────────────────────────────────────────────
//   DXL baud bring-up
// ───────────────────────────────────────────────

/*
 * ensureDxlBaud — 모터 baud 를 DXL_BAUD_TARGET (1 Mbps) 에 맞춤.
 *   1) target baud 로 ping → 성공이면 즉시 반환 (재부팅 후 정상 경로).
 *   2) factory baud 로 fallback → 모든 모터 EEPROM 에 target baud 기입.
 *   3) target baud 로 재연결, ping 검증.
 *
 *   EEPROM 기입은 1회성. 이후 부팅부터는 (1) 에서 바로 통과.
 */
static bool ensureDxlBaud() {
  dxl.begin(DXL_BAUD_TARGET);
  if (dxl.ping(MOTOR_IDS[0])) return true;

  dxl.begin(DXL_BAUD_FACTORY);
  bool any_found = false;
  for (int i = 0; i < NUM_MOTORS; ++i) {
    if (!dxl.ping(MOTOR_IDS[i])) continue;
    any_found = true;
    dxl.torqueOff(MOTOR_IDS[i]);
  }
  if (!any_found) return false;

  for (int i = 0; i < NUM_MOTORS; ++i) {
    // setBaudrate 는 baud 값을 BAUD_RATE 레지스터 인덱스로 변환해 EEPROM 기입.
    // 응답 ACK 까지는 기존 baud, ACK 직후 모터가 target baud 로 전환.
    dxl.setBaudrate(MOTOR_IDS[i], DXL_BAUD_TARGET);
    delay(20);
  }
  delay(100);
  dxl.begin(DXL_BAUD_TARGET);
  return dxl.ping(MOTOR_IDS[0]);
}

// ───────────────────────────────────────────────
//   Setup helpers
// ───────────────────────────────────────────────
static void setupSyncStructs() {
  // SyncWrite: GOAL_POSITION × NUM_MOTORS
  g_sw_infos.packet.p_buf        = nullptr;
  g_sw_infos.packet.is_completed = false;
  g_sw_infos.addr                = ADDR_GOAL_POSITION;
  g_sw_infos.addr_length         = LEN_GOAL_POSITION;
  g_sw_infos.p_xels              = g_sw_xels;
  g_sw_infos.xel_count           = NUM_MOTORS;
  for (int i = 0; i < NUM_MOTORS; ++i) {
    g_sw_xels[i].id     = MOTOR_IDS[i];
    g_sw_xels[i].p_data = (uint8_t*)&g_sw_data[i].goal;
  }
  g_sw_infos.is_info_changed = true;

  // SyncRead: PRESENT_POSITION × NUM_MOTORS
  g_sr_infos.packet.p_buf        = nullptr;
  g_sr_infos.packet.is_completed = false;
  g_sr_infos.addr                = ADDR_PRESENT_POSITION;
  g_sr_infos.addr_length         = LEN_PRESENT_POSITION;
  g_sr_infos.p_xels              = g_sr_xels;
  g_sr_infos.xel_count           = NUM_MOTORS;
  for (int i = 0; i < NUM_MOTORS; ++i) {
    g_sr_xels[i].id         = MOTOR_IDS[i];
    g_sr_xels[i].p_recv_buf = (uint8_t*)&g_sr_data[i].present;
  }
  g_sr_infos.is_info_changed = true;
}

// ───────────────────────────────────────────────
//   Setup / Loop
// ───────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) { /* USB-CDC ready 대기 */ }

  dxl.setPortProtocolVersion(DXL_PROTOCOL);

  if (!ensureDxlBaud()) {
    g_error_latched = true;
    Serial.println("ERR INIT_BAUD");
  }

  for (int i = 0; i < NUM_MOTORS; ++i) {
    uint8_t id = MOTOR_IDS[i];

    if (!dxl.ping(id)) {
      g_error_latched = true;
      Serial.print("ERR INIT_PING_");
      Serial.println(id);
      continue;
    }

    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_POSITION);
    dxl.writeControlTableItem(PROFILE_VELOCITY,     id, PROFILE_VEL);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, PROFILE_ACC);
    dxl.torqueOn(id);

    // 초기 target_raw 를 현재 위치로 (갑작스런 점프 방지)
    g_motor[i].target_raw = dxl.getPresentPosition(id);
    g_motor[i].moving     = false;
  }

  setupSyncStructs();
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      g_buf[g_buf_len] = '\0';
      if (g_buf_len > 0) dispatch(g_buf);
      g_buf_len = 0;
    } else if (g_buf_len < READBUF_LEN - 1) {
      g_buf[g_buf_len++] = c;
    } else {
      g_buf_len = 0;
      replyERR("OVERFLOW");
    }
  }
}
