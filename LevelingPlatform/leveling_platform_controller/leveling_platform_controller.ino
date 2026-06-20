/*
 * Leveling Platform Controller — OpenRB-150
 *
 * Serial protocol with the Pi5 (LevelingMotorClient):
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
 *   - Library: DYNAMIXEL2Arduino (>= 0.6.0, InfoSyncWriteInst_t support)
 *   - Serial : USB CDC, 115200 8N1
 *
 * Wiring
 *   - 3 motors (ID 1, 2, 3) daisy-chained on the OpenRB-150 DYNAMIXEL port.
 *   - 4096 step / rev per motor (XL330 / XM430, etc.).
 *   - step 0 is each motor's physical home position (center, raw = CENTER_RAW).
 *
 * Latency notes
 *   - DXL bus runs at 1 Mbps. If a motor is at factory baud (57600), the first boot
 *     automatically writes 1 Mbps into its EEPROM and reconnects (idempotent).
 *   - 3-axis commands use SyncWrite, 3-axis state reads use SyncRead — one transaction each.
 */

#include <Arduino.h>
#include <Dynamixel2Arduino.h>

// ───────────────────────────────────────────────
//   Config
// ───────────────────────────────────────────────
static const long     SERIAL_BAUD          = 115200;
static const long     DXL_BAUD_TARGET      = 1000000;  // operating baud (1 Mbps)
static const long     DXL_BAUD_FACTORY     = 57600;    // motor factory baud — auto-upgrade path
static const float    DXL_PROTOCOL         = 2.0;

static const int      NUM_MOTORS           = 3;
static const uint8_t  MOTOR_IDS[NUM_MOTORS] = {1, 2, 3};

// signed step range (-2048 ~ +2047, total 4096 steps = 1 revolution)
static const int      STEP_MIN             = -2048;
static const int      STEP_MAX             = +2047;

// raw position reference: the raw value that step 0 maps to
static const int32_t  CENTER_RAW           = 2048;

// motion completion detection (position tolerance only — MOVING register polling removed)
static const unsigned long MOTION_TIMEOUT_MS  = 4000;
static const int32_t       ARRIVED_TOLERANCE  = 10;
static const uint16_t      MOTION_POLL_MS     = 5;

// profile (velocity/acceleration limits — 0 disables the profile)
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

/* Read 3-axis PRESENT_POSITION in one transaction and fill g_sr_data. */
static bool readAllPresent() {
  g_sr_infos.is_info_changed = true;
  uint8_t recv = dxl.syncRead(&g_sr_infos);
  return recv == NUM_MOTORS;
}

/* Send 3-axis GOAL_POSITION in one transaction. Non-blocking. */
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
 * stopAllMotors — stop immediately by setting the current position as the new goal.
 *   Torque stays engaged (holding) — no gravity drop compared to the torqueOff toggle approach.
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
 * waitMotionComplete — judge arrival by position tolerance alone (MOVING register not used).
 *   One syncRead per polling cycle refreshes all 3 axis positions at once.
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
  readAllPresent();   // on failure, use the previously cached values
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
 * ensureDxlBaud — set the motor baud to DXL_BAUD_TARGET (1 Mbps).
 *   1) ping at target baud → return immediately if it succeeds (normal path after reboot).
 *   2) fall back to factory baud → write target baud into every motor's EEPROM.
 *   3) reconnect at target baud, verify with ping.
 *
 *   The EEPROM write is one-time. Later boots pass directly at step (1).
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
    // setBaudrate converts the baud value to a BAUD_RATE register index and writes it to EEPROM.
    // The old baud is used up to the response ACK; right after the ACK the motor switches to target baud.
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
  while (!Serial && millis() < 3000) { /* wait for USB-CDC ready */ }

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

    // initialize target_raw to the current position (prevents a sudden jump)
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
